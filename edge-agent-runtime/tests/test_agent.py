"""
Edge Agent Runtime — 测试套件。
"""

import time
import pytest
from edge_agent import (
    Agent,
    Perception,
    Action,
    SlidingWindowMemory,
    KeyValueMemory,
    PriorityMemory,
    RuleReasoner,
    ThresholdReasoner,
    CompositeReasoner,
)
from edge_agent.core import Sensor, Actuator, Memory


# ── Helpers ──────────────────────────────────────────────────────

class MockSensor(Sensor):
    name = "mock"

    def __init__(self, name: str = "mock", values=None):
        self.name = name
        self._values = list(values or [])
        self._index = 0

    def read(self) -> Perception:
        val = self._values[self._index % len(self._values)]
        self._index += 1
        return Perception(sensor_name=self.name, value=val)


class ConstSensor(Sensor):
    """Always returns a fixed value."""

    def __init__(self, name: str, value):
        self.name = name
        self._value = value

    def read(self) -> Perception:
        return Perception(sensor_name=self.name, value=self._value)


class RecordingActuator(Actuator):
    """Records all actions for assertion."""

    def __init__(self, name: str = "act"):
        self.name = name
        self.actions: list[Action] = []

    def execute(self, action: Action) -> bool:
        self.actions.append(action)
        return True


# ── Perception / Action ─────────────────────────────────────────

class TestPerception:
    def test_default_timestamp(self):
        before = time.time()
        p = Perception(sensor_name="x", value=42)
        after = time.time()
        assert before <= p.timestamp <= after

    def test_metadata(self):
        p = Perception(sensor_name="x", value=1, metadata={"unit": "C"})
        assert p.metadata["unit"] == "C"


class TestAction:
    def test_defaults(self):
        a = Action(actuator_name="fan", command="on")
        assert a.params == {}
        assert a.priority == 0
        assert a.reason == ""


# ── Memory ───────────────────────────────────────────────────────

class TestSlidingWindowMemory:
    def test_store_and_recall(self):
        mem = SlidingWindowMemory(5)
        for i in range(3):
            mem.store(Perception(sensor_name="t", value=i))
        items = mem.recall(limit=10)
        assert len(items) == 3
        assert items[-1].value == 2

    def test_eviction(self):
        mem = SlidingWindowMemory(3)
        for i in range(5):
            mem.store(Perception(sensor_name="t", value=i))
        assert mem.size == 3
        items = mem.recall(limit=10)
        assert items[0].value == 2  # evicted 0, 1
        assert items[-1].value == 4

    def test_latest_by_sensor(self):
        mem = SlidingWindowMemory(100)
        mem.store(Perception(sensor_name="temp", value=20))
        mem.store(Perception(sensor_name="temp", value=25))
        mem.store(Perception(sensor_name="hum", value=60))
        latest = mem.latest_by_sensor()
        assert latest["temp"].value == 25
        assert latest["hum"].value == 60

    def test_clear(self):
        mem = SlidingWindowMemory(10)
        mem.store(Perception(sensor_name="x", value=1))
        mem.clear()
        assert mem.size == 0


class TestKeyValueMemory:
    def test_store_and_get(self):
        mem = KeyValueMemory()
        mem.store(Perception(sensor_name="temp", value=25))
        assert mem.get("temp").value == 25

    def test_overwrite(self):
        mem = KeyValueMemory()
        mem.store(Perception(sensor_name="temp", value=20))
        mem.store(Perception(sensor_name="temp", value=25))
        assert mem.get("temp").value == 25
        assert len(mem.get_history("temp")) == 2

    def test_history_limit(self):
        mem = KeyValueMemory()
        for i in range(60):
            mem.store(Perception(sensor_name="x", value=i))
        assert len(mem.get_history("x")) == 50  # max_history


class TestPriorityMemory:
    def test_priority_ordering(self):
        mem = PriorityMemory(capacity=10)
        mem.store(Perception(sensor_name="a", value=1, metadata={"priority": 1}))
        mem.store(Perception(sensor_name="b", value=2, metadata={"priority": 5}))
        mem.store(Perception(sensor_name="c", value=3, metadata={"priority": 3}))
        items = mem.recall(limit=10)
        assert items[0].value == 2  # highest priority
        assert items[1].value == 3

    def test_eviction_by_priority(self):
        mem = PriorityMemory(capacity=2)
        mem.store(Perception(sensor_name="a", value=1, metadata={"priority": 0}))
        mem.store(Perception(sensor_name="b", value=2, metadata={"priority": 5}))
        mem.store(Perception(sensor_name="c", value=3, metadata={"priority": 3}))
        # capacity=2, lowest priority (a, p=0) should be evicted
        items = mem.recall(limit=10)
        values = {i.value for i in items}
        assert 1 not in values


# ── Reasoner ────────────────────────────────────────────────────

class TestConditionParser:
    """Test _parse_condition indirectly through RuleReasoner."""

    def _decide(self, rules, sensor_values):
        """Helper: run RuleReasoner with given sensor values."""
        reasoner = RuleReasoner(rules)
        mem = SlidingWindowMemory(100)
        perceptions = [Perception(sensor_name=k, value=v) for k, v in sensor_values.items()]
        for p in perceptions:
            mem.store(p)
        return reasoner.decide(perceptions, mem)

    def test_simple_gt(self):
        actions = self._decide(
            [{"when": "temp > 30", "then": "fan_on", "actuator": "fan"}],
            {"temp": 35},
        )
        assert len(actions) == 1
        assert actions[0].command == "fan_on"

    def test_simple_lt(self):
        actions = self._decide(
            [{"when": "temp < 18", "then": "heater_on", "actuator": "heater"}],
            {"temp": 15},
        )
        assert len(actions) == 1

    def test_no_match(self):
        actions = self._decide(
            [{"when": "temp > 30", "then": "fan_on", "actuator": "fan"}],
            {"temp": 25},
        )
        assert len(actions) == 0

    def test_range_condition(self):
        rules = [{"when": "18 <= temp <= 30", "then": "all_off", "actuator": "fan"}]
        actions = self._decide(rules, {"temp": 25})
        assert len(actions) == 1

        actions = self._decide(rules, {"temp": 35})
        assert len(actions) == 0

    def test_multiple_rules(self):
        rules = [
            {"when": "temp > 30", "then": "fan_on", "actuator": "fan"},
            {"when": "temp < 18", "then": "heater_on", "actuator": "heater"},
            {"when": "18 <= temp <= 30", "then": "standby", "actuator": "fan"},
        ]
        actions = self._decide(rules, {"temp": 25})
        assert len(actions) == 1
        assert actions[0].command == "standby"


class TestThresholdReasoner:
    def test_high_threshold(self):
        reasoner = ThresholdReasoner({
            "temp": {"high": 30, "actuator": "fan", "high_action": "on"},
        })
        p = [Perception(sensor_name="temp", value=35)]
        actions = reasoner.decide(p, SlidingWindowMemory())
        assert len(actions) == 1
        assert actions[0].command == "on"

    def test_low_threshold(self):
        reasoner = ThresholdReasoner({
            "temp": {"low": 18, "actuator": "heater", "low_action": "on"},
        })
        p = [Perception(sensor_name="temp", value=15)]
        actions = reasoner.decide(p, SlidingWindowMemory())
        assert len(actions) == 1
        assert actions[0].command == "on"

    def test_normal_no_action(self):
        reasoner = ThresholdReasoner({
            "temp": {"high": 30, "low": 18, "actuator": "fan"},
        })
        p = [Perception(sensor_name="temp", value=22)]
        actions = reasoner.decide(p, SlidingWindowMemory())
        assert len(actions) == 0


class TestCompositeReasoner:
    def test_combines_actions(self):
        r1 = RuleReasoner([{"when": "temp > 25", "then": "alert", "actuator": "fan"}])
        r2 = ThresholdReasoner({"hum": {"high": 80, "actuator": "dehum", "high_action": "on"}})

        composite = CompositeReasoner([r1, r2])
        perceptions = [
            Perception(sensor_name="temp", value=30),
            Perception(sensor_name="hum", value=85),
        ]
        mem = SlidingWindowMemory(100)
        for p in perceptions:
            mem.store(p)

        actions = composite.decide(perceptions, mem)
        assert len(actions) == 2
        commands = {a.command for a in actions}
        assert "alert" in commands
        assert "on" in commands


# ── Agent Integration ───────────────────────────────────────────

class TestAgent:
    def test_tick_basic(self):
        sensor = ConstSensor("temp", 35)
        actuator = RecordingActuator("fan")

        rules = [{"when": "temp > 30", "then": "fan_on", "actuator": "fan"}]
        agent = Agent(
            sensors=[sensor],
            reasoner=RuleReasoner(rules),
            actuators=[actuator],
        )

        actions = agent.tick()
        assert len(actions) == 1
        assert actuator.actions[0].command == "fan_on"

    def test_tick_no_action(self):
        sensor = ConstSensor("temp", 20)
        actuator = RecordingActuator("fan")

        rules = [{"when": "temp > 30", "then": "fan_on", "actuator": "fan"}]
        agent = Agent(
            sensors=[sensor],
            reasoner=RuleReasoner(rules),
            actuators=[actuator],
        )

        actions = agent.tick()
        assert len(actions) == 0

    def test_run_max_loops(self):
        sensor = MockSensor("temp", [25, 30, 35])
        actuator = RecordingActuator("fan")

        agent = Agent(
            sensors=[sensor],
            reasoner=RuleReasoner([{"when": "temp > 30", "then": "on", "actuator": "fan"}]),
            actuators=[actuator],
        )
        agent.run(interval=0.01, max_loops=3)
        assert agent._loop_count == 3

    def test_status(self):
        agent = Agent(
            sensors=[ConstSensor("t", 0)],
            reasoner=RuleReasoner([]),
            actuators=[RecordingActuator("a")],
            name="test-agent",
        )
        status = agent.status
        assert status["name"] == "test-agent"
        assert status["loops"] == 0

    def test_multi_sensor(self):
        temp_sensor = ConstSensor("temp", 35)
        hum_sensor = ConstSensor("hum", 85)
        fan = RecordingActuator("fan")
        dehum = RecordingActuator("dehum")

        reasoner = ThresholdReasoner({
            "temp": {"high": 30, "actuator": "fan", "high_action": "on"},
            "hum": {"high": 80, "actuator": "dehum", "high_action": "on"},
        })

        agent = Agent(
            sensors=[temp_sensor, hum_sensor],
            reasoner=reasoner,
            actuators=[fan, dehum],
        )

        actions = agent.tick()
        assert len(actions) == 2
        assert fan.actions[0].command == "on"
        assert dehum.actions[0].command == "on"

    def test_sensor_error_handled(self):
        """Sensor errors should not crash the agent."""
        class BrokenSensor(Sensor):
            name = "broken"
            def read(self):
                raise IOError("sensor disconnected")

        agent = Agent(
            sensors=[BrokenSensor()],
            reasoner=RuleReasoner([]),
            actuators=[RecordingActuator("a")],
        )
        actions = agent.tick()
        assert actions == []

    def test_unknown_actuator_ignored(self):
        """Actions targeting unknown actuators should be silently ignored."""
        sensor = ConstSensor("temp", 35)
        rules = [{"when": "temp > 30", "then": "on", "actuator": "nonexistent"}]
        agent = Agent(
            sensors=[sensor],
            reasoner=RuleReasoner(rules),
            actuators=[RecordingActuator("fan")],  # different name
        )
        actions = agent.tick()
        assert len(actions) == 0  # no matching actuator


# ── End-to-end: Plant Care Scenario ─────────────────────────────

class TestPlantCareScenario:
    """完整场景测试：植物照料 Agent。"""

    def test_dry_soil_triggers_water(self):
        soil = ConstSensor("soil_moisture", 20)
        pump = RecordingActuator("pump")

        rules = [
            {"when": "soil_moisture < 30", "then": "water", "actuator": "pump"},
            {"when": "soil_moisture >= 30", "then": "stop", "actuator": "pump"},
        ]
        agent = Agent(sensors=[soil], reasoner=RuleReasoner(rules), actuators=[pump])
        actions = agent.tick()

        assert any(a.command == "water" for a in actions)

    def test_wet_soil_stops_water(self):
        soil = ConstSensor("soil_moisture", 60)
        pump = RecordingActuator("pump")

        rules = [
            {"when": "soil_moisture < 30", "then": "water", "actuator": "pump"},
            {"when": "soil_moisture >= 30", "then": "stop", "actuator": "pump"},
        ]
        agent = Agent(sensors=[soil], reasoner=RuleReasoner(rules), actuators=[pump])
        actions = agent.tick()

        assert any(a.command == "stop" for a in actions)

    def test_multi_condition_greenhouse(self):
        """温室场景：温度高 + 湿度高 → 开风扇和除湿。"""
        temp = ConstSensor("temp", 35)
        hum = ConstSensor("humidity", 90)
        soil = ConstSensor("soil_moisture", 25)
        fan = RecordingActuator("fan")
        dehum = RecordingActuator("dehumidifier")
        pump = RecordingActuator("pump")

        reasoner = CompositeReasoner([
            ThresholdReasoner({
                "temp": {"high": 30, "actuator": "fan", "high_action": "on"},
                "humidity": {"high": 80, "actuator": "dehumidifier", "high_action": "on"},
            }),
            RuleReasoner([
                {"when": "soil_moisture < 30", "then": "water", "actuator": "pump"},
            ]),
        ])

        agent = Agent(
            sensors=[temp, hum, soil],
            reasoner=reasoner,
            actuators=[fan, dehum, pump],
        )

        actions = agent.tick()
        assert len(actions) == 3
        assert fan.actions[0].command == "on"
        assert dehum.actions[0].command == "on"
        assert pump.actions[0].command == "water"

# ─── history() ──────────────────────────────
class TestHistory:
    def _make_agent(self):
        sensor = ConstSensor("temp", 35)
        actuator = RecordingActuator("fan")
        rules = [{"when": "temp > 30", "then": "fan_on", "actuator": "fan"}]
        return Agent(sensors=[sensor], reasoner=RuleReasoner(rules), actuators=[actuator])

    def test_empty_history(self):
        agent = self._make_agent()
        assert agent.history() == []

    def test_records_ticks(self):
        agent = self._make_agent()
        agent.tick()
        agent.tick()
        h = agent.history()
        assert len(h) == 2
        assert h[0]["loop"] == 1
        assert h[1]["loop"] == 2

    def test_last_param(self):
        agent = self._make_agent()
        for _ in range(5):
            agent.tick()
        h = agent.history(last=3)
        assert len(h) == 3
        assert h[0]["loop"] == 3

    def test_tick_record_fields(self):
        agent = self._make_agent()
        agent.tick()
        h = agent.history()
        assert "loop" in h[0]
        assert "perceptions" in h[0]
        assert "actions" in h[0]
