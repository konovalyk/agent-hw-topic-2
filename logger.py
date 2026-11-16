"""Лог траєкторії ReAct-циклу у JSON."""

from __future__ import annotations

import contextvars
import json
import time
from datetime import datetime, timezone
from typing import Any

_CURRENT: contextvars.ContextVar[TrajectoryLogger | None] = contextvars.ContextVar(
    "trajectory_logger",
    default=None,
)


def current_logger() -> TrajectoryLogger | None:
    return _CURRENT.get()


def bind_logger(logger: TrajectoryLogger | None):
    return _CURRENT.set(logger)


def reset_logger(token: contextvars.Token) -> None:
    _CURRENT.reset(token)


def clip(value: object, limit: int = 500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit]


class TrajectoryLogger:
    """Збирає кроки agent/tools і зберігає їх у trajectory.json."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.start_time = time.monotonic()
        self.run_id: str | None = None

    def set_run(self, run_id: str) -> None:
        self.run_id = run_id

    def log_step(
        self,
        step_num: int,
        node: str,
        input_data: object,
        output_data: object,
        tool_calls: list | None = None,
        extra: dict | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "step": step_num,
            "node": node,
            "input": clip(input_data),
            "output": clip(output_data),
            "tool_calls": tool_calls or [],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "elapsed_ms": int((time.monotonic() - self.start_time) * 1000),
        }
        if self.run_id:
            row["run_id"] = self.run_id
        if extra:
            row.update(extra)
        self.steps.append(row)

    def save(self, filepath: str) -> None:
        payload = {
            "total_steps": len(self.steps),
            "total_time_ms": int((time.monotonic() - self.start_time) * 1000),
            "trajectory": self.steps,
        }
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
