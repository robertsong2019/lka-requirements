"""
Edge Agent Runtime — 核心抽象与 Agent 循环。

零第三方依赖。
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("edge_agent")


# ── Data containers ──────────────────────────────────────────────

@dataclass
class Perception:
    """一次感知的数据。"""

    sensor_name: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Agent 决策要执行的动作。"""

    actuator_name: str
    command: str
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 0=normal, higher=more urgent
    reason: str = ""


# ── Abstract interfaces ──────────────────────────────────────────

class Sensor(ABC):
    """传感器抽象基类。"""

    name: str = "sensor"

    @abstractmethod
    def read(self) -> Perception:
        """读取传感器数据，返回一个 Perception。"""
        ...


class Actuator(ABC):
    """执行器抽象基类。"""

    name: str = "actuator"

    @abstractmethod
    def execute(self, action: Action) -> bool:
        """执行动作，返回是否成功。"""
        ...


class Reasoner(ABC):
    """推理器抽象基类。"""

    @abstractmethod
    def decide(self, perceptions: List[Perception], memory: "Memory") -> List[Action]:
        """根据感知和记忆，返回要执行的动作列表。"""
        ...


class Memory(ABC):
    """记忆抽象基类。"""

    @abstractmethod
    def store(self, perception: Perception) -> None:
        """存储一次感知。"""
        ...

    @abstractmethod
    def recall(self, limit: int = 10) -> List[Perception]:
        """回忆最近 limit 条感知。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空记忆。"""
        ...


# ── Agent loop ──────────────────────────────────────────────────

class Agent:
    """
    边缘 Agent 主循环。

    Usage::

        agent = Agent(
            sensors=[MySensor()],
            reasoner=RuleReasoner(rules),
            actuators=[MyActuator()],
            memory=SlidingWindowMemory(100),
        )
        agent.run(interval=1.0)  # blocking
    """

    def __init__(
        self,
        sensors: List[Sensor],
        reasoner: Reasoner,
        actuators: List[Actuator],
        memory: Optional[Memory] = None,
        name: str = "edge-agent",
    ) -> None:
        self.sensors = sensors
        self.reasoner = reasoner
        self.actuators = {a.name: a for a in actuators}
        # Import here to avoid circular imports at module level
        if memory is not None:
            self.memory = memory
        else:
            from .memory import SlidingWindowMemory
            self.memory = SlidingWindowMemory(100)
        self.name = name
        self._running = False
        self._loop_count = 0
        self._history: List[Dict] = []

    # ── Single tick (useful for testing) ─────────────────────

    def tick(self) -> List[Action]:
        """执行一次完整的 Agent 循环：感知→推理→行动。"""
        # 1. Perceive
        perceptions = []
        for sensor in self.sensors:
            try:
                p = sensor.read()
                perceptions.append(p)
                self.memory.store(p)
            except Exception as exc:
                logger.warning("[%s] sensor %s read failed: %s", self.name, sensor.name, exc)

        # 2. Reason
        actions = self.reasoner.decide(perceptions, self.memory)

        # 3. Act
        executed: List[Action] = []
        for action in sorted(actions, key=lambda a: -a.priority):
            actuator = self.actuators.get(action.actuator_name)
            if actuator is None:
                logger.warning("[%s] unknown actuator: %s", self.name, action.actuator_name)
                continue
            try:
                ok = actuator.execute(action)
                if ok:
                    executed.append(action)
                    logger.info(
                        "[%s] action: %s → %s (%s)",
                        self.name, action.actuator_name, action.command, action.reason,
                    )
                else:
                    logger.warning("[%s] action failed: %s %s", self.name, action.actuator_name, action.command)
            except Exception as exc:
                logger.error("[%s] actuator %s error: %s", self.name, action.actuator_name, exc)

        self._loop_count += 1
        record = {"loop": self._loop_count, "perceptions": len(perceptions), "actions": len(executed)}
        self._history.append(record)
        return executed

    # ── Continuous loop ──────────────────────────────────────

    def run(self, interval: float = 1.0, max_loops: Optional[int] = None) -> None:
        """
        持续运行 Agent 循环。

        Args:
            interval: 每次循环间隔（秒）
            max_loops: 最大循环次数，None 表示无限
        """
        self._running = True
        logger.info("[%s] starting agent loop (interval=%.1fs)", self.name, interval)

        try:
            while self._running:
                if max_loops is not None and self._loop_count >= max_loops:
                    logger.info("[%s] reached max_loops=%d", self.name, max_loops)
                    break
                self.tick()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("[%s] stopped by user", self.name)
        finally:
            self._running = False
            logger.info("[%s] agent stopped after %d loops", self.name, self._loop_count)

    def stop(self) -> None:
        """停止 Agent 循环。"""
        self._running = False

    @property
    def status(self) -> Dict[str, Any]:
        """返回 Agent 当前状态。"""
        return {
            "name": self.name,
            "running": self._running,
            "loops": self._loop_count,
            "sensors": [s.name for s in self.sensors],
            "actuators": list(self.actuators.keys()),
            "memory_type": type(self.memory).__name__,
        }

    def history(self, last: int = 10) -> List[Dict]:
        """返回最近 N 次 tick 的执行记录。"""
        return self._history[-last:]
