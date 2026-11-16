"""Прогін доменних кейсів і демонстрація захисту. Пише JSON-артефакти."""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage

from agent import build_app, initial_state, message_text, run_agent
from config import get_settings
from logger import TrajectoryLogger, bind_logger, reset_logger
from safety import MAX_REPEATS, MAX_STEPS, TIMEOUT_SECONDS
import tools
from tools import BudgetInput, FetchInput

_HTTP_POST = tools.http_post
_GET_SETTINGS = tools.get_settings

ROOT = Path(__file__).resolve().parent

DOMAIN_CASES = [
    {
        "id": "TC-001",
        "complexity": "simple",
        "query": (
            "Знайди варіанти поїзда або автобуса Київ–Львів на найближчі вихідні. "
            "Достатньо результатів пошуку, сторінки не відкривай."
        ),
        "expected": "Один web_search і коротка відповідь з URL варіантів дороги, без розкладу по днях.",
        "required_tools": ["web_search"],
    },
    {
        "id": "TC-002",
        "complexity": "simple",
        "query": "Що подивитися у Львові на вихідних? Дай кілька місць із пошуку.",
        "expected": "web_search і перелік місць. web_fetch лише якщо summary порожній на деталі.",
        "required_tools": ["web_search"],
    },
    {
        "id": "TC-003",
        "complexity": "medium",
        "query": "Порівняй поїзд і автобус Київ–Львів на найближчі вихідні: час у дорозі і ціна.",
        "expected": "Два пошуки або пошук і fetch сторінки розкладу, порівняння часу і ціни.",
        "required_tools": ["web_search"],
    },
    {
        "id": "TC-004",
        "complexity": "medium",
        "query": (
            "Підбери квиток і житло у Львові на 2 ночі найближчих вихідних "
            "у межах 8000 грн. Перевір бюджет."
        ),
        "expected": "web_search, web_fetch і check_budget. Вердикт fits, tight або deficit.",
        "required_tools": ["web_search", "web_fetch", "check_budget"],
    },
    {
        "id": "TC-005",
        "complexity": "complex",
        "query": (
            "Склади план вихідних у Львові: виїзд із Києва, 2 ночі, музеї, стеля 12000 грн. "
            "Врахуй час у дорозі і час на візити. Якщо на сторінці немає тривалості, оціни її. "
            "Запропонуй місця, які не влізли, як альтернативи."
        ),
        "expected": (
            "Усі чотири інструменти, слоти по днях, alternatives і вердикт бюджету. "
            "Точна ціна квитка в expected не фіксується, бо пошук живий."
        ),
        "required_tools": ["web_search", "web_fetch", "estimate_duration", "check_budget"],
    },
]


class _PlanBuilder:
    """Відповідь formatter без звернення до мережі."""

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        return self.schema(
            answer="Інструмент повернув помилку валідації, повторювати виклик не буду.",
            confidence=0.4,
            sources=[],
            budget_verdict="unknown",
            plan=[],
            alternatives=[],
        )


class _Scripted:
    def __init__(self, replies: list[AIMessage]):
        self.replies = replies
        self.calls = 0

    def bind_tools(self, bound_tools):
        return self

    def invoke(self, messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply

    def with_structured_output(self, schema, **kwargs):
        return _PlanBuilder(schema)


def _tool_message(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _mock_search(url, payload, api_key, timeout=45):
    return 200, {
        "results": [
            {
                "title": "Поїзд Київ — Львів",
                "url": "https://example.com/train",
                "summary": "Близько 5 годин, квиток від 400 грн.",
                "date": None,
            }
        ]
    }


def _enable_mock() -> None:
    tools.http_post = _mock_search
    tools.get_settings = lambda: {
        "api_key": "test-key",
        "base_url": "https://api.example",
        "model": "deepseek-flash",
        "search_base_url": "https://api.example/v1/tools",
    }


def _restore_tools() -> None:
    """Повертає мережеві функції після безпекових прогонів на моках."""
    tools.http_post = _HTTP_POST
    tools.get_settings = _GET_SETTINGS


def collect_tool_names(result: dict) -> list[str]:
    names: list[str] = []
    for message in result.get("messages", []):
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name")
            if name:
                names.append(name)
    return names


def _row(case: dict, *, actual: str, status: str, steps: int, tool_calls: list[str], elapsed_ms: int, extra: dict | None = None) -> dict:
    row = {
        "test_id": case["id"],
        "query": case["query"],
        "expected": case["expected"],
        "actual": actual[:4000],
        "status": status,
        "steps": steps,
        "tool_calls": tool_calls,
        "elapsed_ms": elapsed_ms,
        "complexity": case["complexity"],
        "hit_max_steps": "ліміт кроків" in actual,
        "hit_timeout": "timeout" in actual,
        "hit_loop": "зациклення" in actual,
    }
    if extra:
        row.update(extra)
    return row


def run_safety_cases(logger: TrajectoryLogger | None = None) -> list[dict]:
    """Демонстрація валідації, зациклення, max_steps і timeout на моках."""
    owned = logger or TrajectoryLogger()
    _enable_mock()
    results = []
    try:
        results.append(_validation_case(owned))
        results.append(_loop_case(owned))
        results.append(_max_steps_case(owned))
        results.append(_timeout_case(owned))
    finally:
        _restore_tools()
    return results


def _invoke(llm, logger: TrajectoryLogger, run_id: str, query: str, **limits) -> tuple[dict, int]:
    logger.set_run(run_id)
    app = build_app(llm, **limits)
    token = bind_logger(logger)
    started = time.monotonic()
    try:
        result = app.invoke(initial_state(query), config={"recursion_limit": 40})
    finally:
        reset_logger(token)
    return result, int((time.monotonic() - started) * 1000)


def _validation_case(logger: TrajectoryLogger) -> dict:
    case = {
        "id": "TC-S01",
        "complexity": "safety",
        "query": "Перевірка невалідного URL і від'ємного бюджету",
        "expected": "Валідатор відхиляє адресу без схеми і від'ємну суму. Граф повертає текст помилки моделі.",
    }
    notes = []
    try:
        FetchInput(url="notaurl")
        notes.append("URL не відхилено")
    except Exception as exc:  # noqa: BLE001 — сам текст помилки є очікуваним результатом
        notes.append(f"URL: {exc}")
    try:
        BudgetInput(limit_uah=1000, transport_uah=-20, stay_uah=0, activities_uah=0, meals_uah=0)
        notes.append("бюджет не відхилено")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Бюджет: {exc}")
    llm = _Scripted([
        _tool_message("web_fetch", {"url": "notaurl"}, "bad-url"),
        AIMessage(content="Адреса некоректна, той самий виклик не повторюю."),
    ])
    result, elapsed = _invoke(llm, logger, case["id"], case["query"], max_steps=MAX_STEPS)
    transcript = "\n".join(message_text(message) for message in result["messages"])
    actual = transcript + "\n" + "\n".join(notes)
    status = "success" if "Помилка валідації" in transcript and "від'ємною" in actual else "error"
    return _row(
        case,
        actual=actual,
        status=status,
        steps=result.get("step_count", -1),
        tool_calls=collect_tool_names(result),
        elapsed_ms=elapsed,
    )


def _loop_case(logger: TrajectoryLogger) -> dict:
    case = {
        "id": "TC-S02",
        "complexity": "safety",
        "query": "Повтори один і той самий пошук квитків Київ–Львів",
        "expected": f"Той самий web_search зупиняється на повторенні {MAX_REPEATS}.",
    }
    llm = _Scripted([
        _tool_message("web_search", {"query": "квитки Київ Львів", "count": 1}, "loop"),
    ])
    result, elapsed = _invoke(llm, logger, case["id"], case["query"])
    actual = message_text(result["messages"][-1])
    status = "success" if "зациклення" in actual else "error"
    return _row(case, actual=actual, status=status, steps=result.get("step_count", -1), tool_calls=collect_tool_names(result), elapsed_ms=elapsed)


def _max_steps_case(logger: TrajectoryLogger) -> dict:
    case = {
        "id": "TC-S03",
        "complexity": "safety",
        "query": "Шукай житло, доки не впрешся в ліміт кроків",
        "expected": "При max_steps=2 граф зупиняється і повертає часткову відповідь.",
    }
    llm = _Scripted([
        _tool_message("web_search", {"query": "житло Львів", "count": 1}, "steps"),
    ])
    result, elapsed = _invoke(llm, logger, case["id"], case["query"], max_steps=2)
    actual = message_text(result["messages"][-1])
    status = "success" if "ліміт кроків" in actual else "error"
    return _row(case, actual=actual, status=status, steps=result.get("step_count", -1), tool_calls=collect_tool_names(result), elapsed_ms=elapsed)


def _timeout_case(logger: TrajectoryLogger) -> dict:
    case = {
        "id": "TC-S04",
        "complexity": "safety",
        "query": "Запит, у якого час уже вичерпано на старті",
        "expected": "timeout спрацьовує до виклику моделі.",
    }
    llm = _Scripted([AIMessage(content="модель не повинна відповідати")])
    logger.set_run(case["id"])
    app = build_app(llm, timeout_seconds=TIMEOUT_SECONDS)
    token = bind_logger(logger)
    started = time.monotonic()
    try:
        result = app.invoke(
            initial_state(case["query"], started_at=time.monotonic() - TIMEOUT_SECONDS - 5),
            config={"recursion_limit": 40},
        )
    finally:
        reset_logger(token)
    elapsed = int((time.monotonic() - started) * 1000)
    actual = message_text(result["messages"][-1])
    status = "success" if "timeout" in actual and llm.calls == 0 else "error"
    return _row(case, actual=actual, status=status, steps=result.get("step_count", -1), tool_calls=collect_tool_names(result), elapsed_ms=elapsed)


def run_domain_cases(logger: TrajectoryLogger | None = None) -> list[dict]:
    """Живі кейси. Без ключа лишаються в результатах зі статусом skipped."""
    owned = logger or TrajectoryLogger()
    if not get_settings()["api_key"]:
        return [
            _row(
                case,
                actual="Пропущено: не задано DEEPSEEK_API_KEY.",
                status="skipped",
                steps=0,
                tool_calls=[],
                elapsed_ms=0,
            )
            for case in DOMAIN_CASES
        ]
    results = []
    for case in DOMAIN_CASES:
        owned.set_run(case["id"])
        token = bind_logger(owned)
        started = time.monotonic()
        try:
            result = run_agent(case["query"], max_steps=MAX_STEPS, timeout_seconds=TIMEOUT_SECONDS)
            actual = message_text(result["messages"][-1])
            names = collect_tool_names(result)
            missing = [name for name in case["required_tools"] if name not in names]
            if any(flag in actual for flag in ("ліміт кроків", "timeout", "зациклення")):
                status = "partial"
            elif missing:
                status = "partial"
            else:
                status = "success"
            results.append(
                _row(
                    case,
                    actual=actual,
                    status=status,
                    steps=result.get("step_count", -1),
                    tool_calls=names,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    extra={"missing_tools": missing},
                )
            )
        except Exception as exc:  # noqa: BLE001 — збій кейсу не зупиняє решту прогону
            results.append(
                _row(
                    case,
                    actual=f"Error: {exc}",
                    status="error",
                    steps=-1,
                    tool_calls=[],
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
            )
        finally:
            reset_logger(token)
    return results


def main() -> None:
    logger = TrajectoryLogger()
    results = [*run_safety_cases(logger), *run_domain_cases(logger)]
    logger.save(str(ROOT / "trajectory.json"))
    (ROOT / "test_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved: {len(results)} test cases")
    for row in results:
        print(f"  {row['test_id']}: {row['status']} ({row['steps']} steps, {row['elapsed_ms']} ms)")


if __name__ == "__main__":
    main()
