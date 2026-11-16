"""Обмеження кроку, часу і повторюваних викликів інструментів."""

from __future__ import annotations

import hashlib
import json
import time

MAX_STEPS = 12
TIMEOUT_SECONDS = 120
MAX_REPEATS = 3


def call_hash(tool_name: str, args: dict) -> str:
    """Відбиток назви і аргументів. Однаковий виклик дає той самий рядок."""
    payload = json.dumps(
        {"name": tool_name, "args": args},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_args(args: object) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            loaded = json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}
        if isinstance(loaded, dict):
            return loaded
    return {}


class LoopDetector:
    """Зупиняє агента, якщо той самий виклик повторюється max_repeats разів поспіль."""

    def __init__(self, max_repeats: int = MAX_REPEATS):
        self.recent_calls: list[str] = []
        self.max_repeats = max_repeats

    def check(self, tool_name: str, args: dict) -> bool:
        """Повертає True, коли останні max_repeats викликів збігаються."""
        self.recent_calls.append(call_hash(tool_name, args))
        if len(self.recent_calls) < self.max_repeats:
            return False
        last = self.recent_calls[-self.max_repeats :]
        return len(set(last)) == 1


def preview_loop(
    existing: list[str],
    calls: list[tuple[str, dict]],
    max_repeats: int = MAX_REPEATS,
) -> bool:
    """Перевіряє зациклення на копії історії, не змінюючи стан графа."""
    detector = LoopDetector(max_repeats)
    detector.recent_calls = list(existing)
    for name, args in calls:
        if detector.check(name, args):
            return True
    return False


def timed_out(started_at: float, timeout_seconds: float, now: float | None = None) -> bool:
    current = time.monotonic() if now is None else now
    return (current - started_at) >= timeout_seconds
