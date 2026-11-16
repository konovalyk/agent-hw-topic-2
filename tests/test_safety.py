"""Захист графа: зациклення, ліміт кроків, timeout, валідація і лог."""

import json

from langchain_core.messages import AIMessage

from agent import TravelPlan, build_app, initial_state, message_text
from logger import TrajectoryLogger, bind_logger, reset_logger
from safety import LoopDetector
import tools
from tools import web_fetch


class ScriptedLLM:
    """Відповіді агента за списком. Структурований крок не чіпає цей лічильник."""

    def __init__(self, replies: list[AIMessage]):
        self.replies = replies
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply

    def with_structured_output(self, schema, **kwargs):
        return _Structured(schema)


class _Structured:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        return TravelPlan(
            answer="Тестовий план вихідних.",
            confidence=1.4,
            sources=[" https://example.com/train "],
            budget_verdict="невідомо",
            plan=[],
            alternatives=[" Ратуша "],
        )


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def search_ok(url, payload, api_key, timeout=45):
    if str(url).endswith("/search"):
        return 200, {
            "results": [
                {
                    "title": "Поїзд",
                    "url": "https://example.com/train",
                    "summary": "5 годин, 400 грн",
                    "date": None,
                }
            ]
        }
    return 200, {"title": "Поїзд", "content": "Квиток 400 грн. У дорозі 5 годин."}


def run_graph(llm, *, max_steps=12, timeout_seconds=120, started_at=None, query="План"):
    app = build_app(llm, max_steps=max_steps, timeout_seconds=timeout_seconds)
    return app.invoke(initial_state(query, started_at=started_at), config={"recursion_limit": 40})


def test_loop_detector_requires_three_identical_calls():
    detector = LoopDetector(max_repeats=3)
    assert detector.check("web_search", {"query": "Львів", "count": 1}) is False
    assert detector.check("web_search", {"query": "Львів", "count": 1}) is False
    assert detector.check("web_search", {"query": "Львів", "count": 1}) is True
    other = LoopDetector(max_repeats=3)
    other.check("web_search", {"query": "Львів", "count": 1})
    other.check("web_search", {"query": "Одеса", "count": 1})
    assert other.check("web_search", {"query": "Львів", "count": 1}) is False


def test_graph_stops_on_repeated_search(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "http_post", search_ok)
    monkeypatch.setattr(tools, "get_settings", lambda: {
        "api_key": "test-key",
        "base_url": "https://api.example",
        "model": "deepseek-flash",
        "search_base_url": "https://api.example/v1/tools",
    })
    llm = ScriptedLLM([
        tool_call("web_search", {"query": "квитки Київ Львів", "count": 1}, "c1"),
    ])
    logger = TrajectoryLogger()
    token = bind_logger(logger)
    try:
        result = run_graph(llm, query="Квитки")
    finally:
        reset_logger(token)
    text = message_text(result["messages"][-1])
    assert "зациклення" in text
    assert llm.calls == 3
    logger.save(str(tmp_path / "trajectory.json"))
    saved = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))
    assert saved["total_steps"] == len(saved["trajectory"])
    assert saved["trajectory"][0]["node"] == "agent"


def test_graph_stops_at_max_steps(monkeypatch):
    monkeypatch.setattr(tools, "http_post", search_ok)
    monkeypatch.setattr(tools, "get_settings", lambda: {
        "api_key": "test-key",
        "base_url": "https://api.example",
        "model": "deepseek-flash",
        "search_base_url": "https://api.example/v1/tools",
    })
    llm = ScriptedLLM([
        tool_call("web_search", {"query": "житло Львів", "count": 1}, "c1"),
    ])
    result = run_graph(llm, max_steps=2, query="Житло")
    assert "ліміт кроків" in message_text(result["messages"][-1])
    assert result["step_count"] == 2


def test_graph_stops_on_timeout():
    llm = ScriptedLLM([AIMessage(content="цього не має бути")])
    result = run_graph(llm, started_at=-10_000, timeout_seconds=120, query="Погода")
    assert "timeout" in message_text(result["messages"][-1])
    assert llm.calls == 0


def test_invalid_url_returns_to_the_model(monkeypatch):
    monkeypatch.setattr(tools, "http_post", search_ok)
    llm = ScriptedLLM([
        tool_call("web_fetch", {"url": "notaurl"}, "bad"),
        AIMessage(content="Адреса некоректна, пошук не повторюю."),
    ])
    result = run_graph(llm, query="Відкрий сторінку")
    transcript = "\n".join(message_text(message) for message in result["messages"])
    assert "Помилка валідації" in transcript
    plan = TravelPlan.model_validate_json(message_text(result["messages"][-1]))
    assert plan.confidence == 1
    assert plan.budget_verdict == "unknown"
    assert plan.sources == ["https://example.com/train"]
    assert plan.alternatives == ["Ратуша"]


def test_fetch_step_logs_parser_meta(monkeypatch, tmp_path):
    class Parser:
        model_name = "logged-parser"

        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            return {
                "source_url": "https://example.com/museum",
                "title": "Музей",
                "kind": "place",
                "price_amount": None,
                "duration_minutes": None,
                "facts": ["без годин"],
            }

    def fake_post(url, payload, api_key, timeout=45):
        return 200, {"title": "Музей", "content": "Годин роботи немає."}

    monkeypatch.setattr(tools, "http_post", fake_post)
    monkeypatch.setattr(tools, "get_settings", lambda: {
        "api_key": "test-key",
        "base_url": "https://api.example",
        "model": "deepseek-flash",
        "search_base_url": "https://api.example/v1/tools",
    })
    monkeypatch.setattr("config.build_llm", lambda: Parser())
    text = web_fetch.invoke({"url": "https://example.com/museum"})
    assert "null" in text
    logger = TrajectoryLogger()
    logger.log_step(
        1,
        "tools",
        "web_fetch",
        text,
        ["web_fetch"],
        extra={
            "parser_model": tools.last_parser_meta["parser_model"],
            "parser_ok": tools.last_parser_meta["parser_ok"],
        },
    )
    logger.save(str(tmp_path / "trajectory.json"))
    row = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))["trajectory"][0]
    assert row["parser_ok"] is True
    assert row["parser_model"] == "logged-parser"
    assert len(row["output"]) <= 500


def test_formatter_reports_structured_error_instead_of_raw_answer():
    class BrokenLLM:
        def bind_tools(self, bound_tools):
            return self

        def invoke(self, messages):
            return AIMessage(content="ГОТОВА ВІДПОВІДЬ БЕЗ JSON")

        def with_structured_output(self, schema, **kwargs):
            raise RuntimeError("schema rejected")

    logger = TrajectoryLogger()
    token = bind_logger(logger)
    try:
        result = run_graph(BrokenLLM(), query="План")
    finally:
        reset_logger(token)
    text = message_text(result["messages"][-1])
    assert text.startswith("Не вдалося зібрати TravelPlan")
    assert "schema rejected" in text
    assert "ГОТОВА ВІДПОВІДЬ" not in text
    formatter_rows = [row for row in logger.steps if row["node"] == "formatter"]
    assert formatter_rows
    assert "schema rejected" in formatter_rows[-1]["output"]
