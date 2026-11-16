"""Перевірки інструментів без мережі і без ключа."""

import json

import pytest
from pydantic import ValidationError

import tools
from config import get_settings
from tools import (
    BudgetInput,
    DurationInput,
    FetchInput,
    PageCard,
    SearchInput,
    check_budget,
    estimate_duration,
    extract_card,
    web_fetch,
    web_search,
)


def settings_with_key() -> dict[str, str]:
    return {
        "api_key": "test-key",
        "base_url": "https://api.example",
        "model": "deepseek-flash",
        "search_base_url": "https://api.example/v1/tools",
    }


class ParserDouble:
    """Підміна моделі: історія агента сюди не приходить."""

    model_name = "parser-double"

    def __init__(self, fail_first: bool = False):
        self.fail_first = fail_first
        self.builds = 0
        self.seen: list = []

    def with_structured_output(self, schema, **kwargs):
        self.builds += 1
        if self.fail_first and self.builds == 1:
            raise RuntimeError("json schema недоступна")
        return self

    def invoke(self, messages):
        self.seen = messages
        return PageCard(
            source_url="https://ignored.example",
            title="Музей",
            kind="Place",
            price_amount=None,
            price_currency="",
            duration_minutes=None,
            area="центр",
            facts=["  є в тексті  ", "", "другий", "третій", "зайвий"],
        )


def test_search_validation_rejects_blank_query():
    with pytest.raises(ValidationError, match="1 до 128"):
        SearchInput(query="   ", count=1)


def test_search_validation_rejects_wide_count():
    with pytest.raises(ValidationError, match="1 до 5"):
        SearchInput(query="Львів", count=10)


def test_fetch_rejects_credentials_and_bad_scheme():
    with pytest.raises(ValidationError, match="логін"):
        FetchInput(url="https://user:pass@example.com/page")
    with pytest.raises(ValidationError, match="http"):
        FetchInput(url="ftp://example.com/file")


def test_budget_rejects_negative_and_zero_limit():
    with pytest.raises(ValidationError, match="від'ємною"):
        BudgetInput(
            limit_uah=1000,
            transport_uah=-1,
            stay_uah=0,
            activities_uah=0,
            meals_uah=0,
        )
    with pytest.raises(ValidationError, match="більшим за нуль"):
        BudgetInput(
            limit_uah=0,
            transport_uah=0,
            stay_uah=0,
            activities_uah=0,
            meals_uah=0,
        )


def test_duration_rejects_short_category():
    with pytest.raises(ValidationError, match="2 символи"):
        DurationInput(category=" я ")


def test_search_returns_summary_without_page_body(monkeypatch):
    captured = {}

    def fake_post(url, payload, api_key, timeout=45):
        captured["url"] = url
        captured["payload"] = payload
        captured["api_key"] = api_key
        return 200, {
            "results": [
                {
                    "title": "Поїзд до Львова",
                    "url": "https://example.com/train",
                    "summary": "У дорозі 5 годин, від 400 грн.",
                    "date": "2026-11-16T00:00:00+00:00",
                    "content": "SECRET_PAGE",
                }
            ]
        }

    monkeypatch.setattr(tools, "http_post", fake_post)
    monkeypatch.setattr(tools, "get_settings", settings_with_key)
    text = web_search.invoke({"query": " поїзд Київ Львів ", "count": 1})
    assert captured["url"].endswith("/search")
    assert captured["payload"]["need_content"] is False
    assert captured["payload"]["query"] == "поїзд Київ Львів"
    assert "Поїзд до Львова" in text
    assert "https://example.com/train" in text
    assert "SECRET_PAGE" not in text


def test_search_empty_results_are_not_an_error(monkeypatch):
    monkeypatch.setattr(tools, "http_post", lambda *args, **kwargs: (200, {"results": []}))
    monkeypatch.setattr(tools, "get_settings", settings_with_key)
    assert web_search.invoke({"query": "нічого такого", "count": 2}) == "Нічого не знайдено"


def test_search_without_key_does_not_call_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("мережа не потрібна")

    monkeypatch.setattr(tools, "http_post", explode)
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: {**settings_with_key(), "api_key": ""},
    )
    assert web_search.invoke({"query": "Львів", "count": 1}) == "Не задано DEEPSEEK_API_KEY"


def test_search_retries_429_once(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, payload, api_key, timeout=45):
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, {"error": {"message": "забагато"}}
        return 200, {"results": []}

    monkeypatch.setattr(tools, "http_post", fake_post)
    monkeypatch.setattr(tools.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(tools, "get_settings", settings_with_key)
    assert web_search.invoke({"query": "Львів", "count": 1}) == "Нічого не знайдено"
    assert calls["n"] == 2


def test_search_clips_long_summary(monkeypatch):
    def fake_post(url, payload, api_key, timeout=30):
        assert timeout == tools.SEARCH_TIMEOUT
        return 200, {
            "results": [
                {
                    "title": "Довгий опис",
                    "url": "https://example.com/long",
                    "summary": "а" * 2000,
                    "date": None,
                }
            ]
        }

    monkeypatch.setattr(tools, "http_post", fake_post)
    monkeypatch.setattr(tools, "get_settings", settings_with_key)
    text = web_search.invoke({"query": "Львів", "count": 1})
    summary_line = text.splitlines()[-1]
    assert summary_line == ("а" * tools.SUMMARY_LIMIT) + "…"


def test_fetch_timeout_is_not_retried(monkeypatch):
    calls = {"n": 0, "timeout": None}

    def fake_post(url, payload, api_key, timeout=30):
        calls["n"] += 1
        calls["timeout"] = timeout
        return 504, {"error": {"message": "timed out"}}

    monkeypatch.setattr(tools, "http_post", fake_post)
    monkeypatch.setattr(tools.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(tools, "get_settings", settings_with_key)
    text = web_fetch.invoke({"url": "https://example.com/slow"})
    assert "тимчасово недоступне" in text
    assert calls["n"] == 1
    assert calls["timeout"] == tools.FETCH_TIMEOUT


def test_fetch_403_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, payload, api_key, timeout=45):
        calls["n"] += 1
        return 403, {"error": {"message": "blocked"}}

    monkeypatch.setattr(tools, "http_post", fake_post)
    monkeypatch.setattr(tools.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(tools, "get_settings", settings_with_key)
    text = web_fetch.invoke({"url": "https://example.com/secret"})
    assert "політикою" in text
    assert calls["n"] == 1


def test_fetch_404_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, payload, api_key, timeout=45):
        calls["n"] += 1
        return 404, {"error": {"message": "gone"}}

    monkeypatch.setattr(tools, "http_post", fake_post)
    monkeypatch.setattr(tools, "get_settings", settings_with_key)
    text = web_fetch.invoke({"url": "https://example.com/missing"})
    assert "Немає вмісту" in text
    assert calls["n"] == 1


def test_fetch_parses_page_without_agent_state(monkeypatch):
    parser = ParserDouble(fail_first=True)

    def fake_post(url, payload, api_key, timeout=45):
        assert url.endswith("/fetch")
        return 200, {
            "title": "Аптека-музей",
            "content": "Вхід 40 грн. Скільки гуляти, не написано. " + ("текст " * 2000),
        }

    monkeypatch.setattr(tools, "http_post", fake_post)
    monkeypatch.setattr(tools, "get_settings", settings_with_key)
    monkeypatch.setattr("config.build_llm", lambda: parser)
    text = web_fetch.invoke({"url": "https://example.com/museum"})
    card = json.loads(text.split("PageCard:\n", 1)[1])
    assert card["price_amount"] is None
    assert card["duration_minutes"] is None
    assert card["kind"] == "place"
    assert card["price_currency"] is None
    assert card["facts"] == ["є в тексті", "другий", "третій"]
    assert card["source_url"] == "https://example.com/museum"
    assert len(parser.seen) == 2
    assert len(parser.seen[1].content) < 6200
    assert tools.last_parser_meta["parser_ok"] is True
    assert tools.last_parser_meta["parser_model"] == "parser-double"
    assert parser.builds == 2


def test_fetch_returns_snippet_when_parser_fails(monkeypatch):
    def fake_post(url, payload, api_key, timeout=45):
        return 200, {"title": "Сторінка", "content": "ЦІНА_ТАЄМНА " + ("а" * 800)}

    def broken(markdown, source_url, llm=None):
        raise RuntimeError("схема розвалилась")

    monkeypatch.setattr(tools, "http_post", fake_post)
    monkeypatch.setattr(tools, "get_settings", settings_with_key)
    monkeypatch.setattr("config.build_llm", lambda: ParserDouble())
    monkeypatch.setattr(tools, "extract_card", broken)
    text = web_fetch.invoke({"url": "https://example.com/raw"})
    assert text.startswith("Картку не розібрано.")
    assert "ЦІНА_ТАЄМНА" in text
    assert len(text) < 600
    assert tools.last_parser_meta["parser_ok"] is False
    assert "схема розвалилась" in str(tools.last_parser_meta["parser_error"])


def test_extract_card_keeps_nulls_from_page():
    parser = ParserDouble()
    card = extract_card("На сторінці немає чисел.", "https://example.com/place", llm=parser)
    assert card.duration_minutes is None
    assert card.price_amount is None
    assert "На сторінці немає чисел." in parser.seen[-1].content


def test_estimate_duration_known_and_default():
    museum = estimate_duration.invoke({"category": "Музей"})
    unknown = estimate_duration.invoke({"category": "ринок"})
    road = estimate_duration.invoke({"category": "дорога"})
    assert "150 хв" in museum
    assert "duration_source: default" in museum
    assert "90 хв" in unknown
    assert "20 хв" in road


def test_budget_verdicts():
    assert "fits" in check_budget.invoke(
        {"limit_uah": 100, "transport_uah": 85, "stay_uah": 0, "activities_uah": 0, "meals_uah": 0}
    )
    assert "tight" in check_budget.invoke(
        {"limit_uah": 100, "transport_uah": 86, "stay_uah": 0, "activities_uah": 0, "meals_uah": 0}
    )
    assert "tight" in check_budget.invoke(
        {"limit_uah": 100, "transport_uah": 100, "stay_uah": 0, "activities_uah": 0, "meals_uah": 0}
    )
    assert "deficit" in check_budget.invoke(
        {"limit_uah": 100, "transport_uah": 101, "stay_uah": 0, "activities_uah": 0, "meals_uah": 0}
    )


def test_settings_build_search_url(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.delenv("SEARCH_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    assert get_settings()["search_base_url"] == "https://api.deepseek.com/v1/tools"
    assert get_settings()["base_url"] == "https://api.deepseek.com/v1"
