"""Інструменти планувальника вихідних: пошук, сторінка, тривалість і бюджет."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Literal
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import get_settings, invoke_structured, llm_model_name

RETRY_STATUSES = {429}
UNAVAILABLE_STATUSES = {408, 429, 500, 502, 503, 504}
SEARCH_TIMEOUT = 30
FETCH_TIMEOUT = 20
PARSER_LIMIT = 6000
SNIPPET_LIMIT = 500
SUMMARY_LIMIT = 800

# Останній розбір сторінки. Вузол tools дописує ці поля в крок траєкторії.
last_parser_meta: dict[str, object] = {
    "parser_model": None,
    "parser_ok": None,
    "parser_error": None,
}

PARSER_PROMPT = """Витягни з тексту сторінки лише ті факти, які в ньому написані.
Не вигадуй ціну, валюту і тривалість. Якщо числа немає, залиш null.
Не конвертуй валюту і не підставляй типову тривалість візиту.
kind: transport для квитків і дороги, stay для житла, place для місця, інакше other.
facts: не більше трьох коротких пунктів українською.
"""


class SearchInput(BaseModel):
    """Параметри веб-пошуку DeepSeek."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(description='Пошуковий запит, 1–128 символів. Наприклад, "поїзд Київ Львів квитки"')
    count: int = Field(default=5, description="Скільки результатів повернути, від 1 до 5")

    @field_validator("query")
    @classmethod
    def query_length(cls, value: str) -> str:
        if not value or not 1 <= len(value) <= 128:
            raise ValueError("Запит має містити від 1 до 128 символів")
        return value

    @field_validator("count")
    @classmethod
    def count_range(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("Кількість результатів має бути від 1 до 5")
        return value


class FetchInput(BaseModel):
    """Адреса сторінки для DeepSeek Fetch."""

    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(description="Повний http або https URL сторінки з результатів пошуку")

    @field_validator("url")
    @classmethod
    def http_url(cls, value: str) -> str:
        if len(value) > 8192:
            raise ValueError("URL довший за 8192 символи")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("URL має бути http або https і містити хост")
        if parsed.username or parsed.password:
            raise ValueError("URL не може містити логін або пароль")
        return value


class DurationInput(BaseModel):
    """Категорія місця, для якого немає тривалості на сторінці."""

    model_config = ConfigDict(str_strip_whitespace=True)

    category: str = Field(description='Категорія: музей, галерея, парк, оглядовий майданчик, обід, вечеря, кафе або дорога')

    @field_validator("category")
    @classmethod
    def category_present(cls, value: str) -> str:
        if len(value) < 2:
            raise ValueError("Категорія має містити щонайменше 2 символи")
        return value


class BudgetInput(BaseModel):
    """Суми в гривнях для перевірки ліміту поїздки."""

    model_config = ConfigDict(str_strip_whitespace=True)

    limit_uah: float = Field(description="Верхня межа бюджету в грн, більша за нуль")
    transport_uah: float = Field(description="Вартість дороги туди й назад у грн")
    stay_uah: float = Field(description="Вартість житла за всі ночі у грн")
    activities_uah: float = Field(description="Квитки на місця у грн")
    meals_uah: float = Field(description="Орієнтовна вартість їжі у грн")

    @field_validator("limit_uah")
    @classmethod
    def limit_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Ліміт має бути більшим за нуль")
        return value

    @field_validator("transport_uah", "stay_uah", "activities_uah", "meals_uah")
    @classmethod
    def amount_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Сума не може бути від'ємною")
        return value


class PageCard(BaseModel):
    """Коротка картка сторінки. Порожні числа означають, що в тексті їх не було."""

    model_config = ConfigDict(extra="ignore")

    source_url: str
    title: str = ""
    kind: Literal["transport", "stay", "place", "other"] = "other"
    price_amount: float | None = None
    price_currency: str | None = None
    duration_minutes: int | None = None
    area: str | None = None
    facts: list[str] = Field(default_factory=list)

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"transport", "stay", "place", "other"}:
                return lowered
        return value

    @field_validator("price_currency", mode="before")
    @classmethod
    def blank_currency(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("price_amount")
    @classmethod
    def price_missing_if_negative(cls, value: float | None) -> float | None:
        if value is None or value < 0:
            return None
        return value

    @field_validator("duration_minutes")
    @classmethod
    def duration_missing_if_not_positive(cls, value: int | None) -> int | None:
        if value is None or value <= 0:
            return None
        return value

    @field_validator("facts")
    @classmethod
    def three_facts(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        return cleaned[:3]


# Відомі категорії. Усе, чого немає в таблиці, отримує 90 хвилин.
DURATION_MINUTES = {
    "museum": 150,
    "gallery": 150,
    "park": 60,
    "viewpoint": 60,
    "lunch": 60,
    "dinner": 90,
    "cafe": 45,
    "transfer": 20,
}
DURATION_ALIASES = {
    "museum": "museum",
    "музей": "museum",
    "gallery": "gallery",
    "галерея": "gallery",
    "park": "park",
    "парк": "park",
    "viewpoint": "viewpoint",
    "оглядовий майданчик": "viewpoint",
    "lunch": "lunch",
    "обід": "lunch",
    "dinner": "dinner",
    "вечеря": "dinner",
    "cafe": "cafe",
    "кафе": "cafe",
    "transfer": "transfer",
    "дорога": "transfer",
    "дорога до центру": "transfer",
}


def http_post(url: str, payload: dict, api_key: str, timeout: float = SEARCH_TIMEOUT) -> tuple[int, object]:
    """POST JSON. Повертає код і тіло. Таймаут і обрив мережі кодує як 504 без повтору."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, _parse_body(raw)
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        return error.code, _parse_body(raw)
    except Exception as error:  # noqa: BLE001 — таймаут і обрив мережі йдуть у ту саму гілку повтору
        return 504, {"error": {"message": str(error)}}


def _parse_body(raw: str) -> object:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def call_api(path: str, payload: dict, timeout: float) -> tuple[int, object]:
    """Виклик Search або Fetch. Повторює лише 429, один раз."""
    settings = get_settings()
    if not settings["api_key"]:
        return 0, {"error": {"message": "Не задано DEEPSEEK_API_KEY"}}
    url = f"{settings['search_base_url']}{path}"
    status, body = http_post(url, payload, settings["api_key"], timeout=timeout)
    if status in RETRY_STATUSES:
        time.sleep(1)
        status, body = http_post(url, payload, settings["api_key"], timeout=timeout)
    return status, body


def failure_text(status: int, body: object) -> str:
    if status == 0:
        return "Не задано DEEPSEEK_API_KEY"
    if status in UNAVAILABLE_STATUSES:
        return "Джерело тимчасово недоступне"
    if status == 401:
        return "Ключ API відхилено"
    if status == 402:
        return "Недостатньо балансу API"
    if status == 403:
        return "Сторінку відхилено політикою джерела"
    if status == 404:
        return "Немає вмісту за цією адресою"
    detail = body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        detail = body["error"].get("message") or body["error"]
    return f"Помилка джерела ({status}): {detail}"[:300]


def format_search_results(results: object) -> str:
    if not isinstance(results, list) or not results:
        return "Нічого не знайдено"
    blocks: list[str] = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        date = item.get("date") or "невідома"
        summary = _clip_text(item.get("summary") or item.get("snippet") or "", SUMMARY_LIMIT)
        blocks.append(
            f"{index}. {item.get('title') or 'Без назви'}\n"
            f"URL: {item.get('url') or ''}\n"
            f"Дата: {date}\n"
            f"{summary}"
        )
    return "\n\n".join(blocks) if blocks else "Нічого не знайдено"


def _clip_text(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def extract_card(markdown: str, source_url: str, llm: object | None = None) -> PageCard:
    """Окремий виклик моделі. Історія агента сюди не потрапляє."""
    from config import build_llm

    model = llm or build_llm()
    text = markdown[:PARSER_LIMIT]
    raw = invoke_structured(
        model,
        PageCard,
        [
            SystemMessage(content=PARSER_PROMPT),
            HumanMessage(content=f"URL: {source_url}\n\n{text}"),
        ],
    )
    if isinstance(raw, PageCard):
        card = raw
    else:
        card = PageCard.model_validate(raw)
    card.source_url = source_url
    return card


def _set_parser_meta(model: object | None, ok: bool, error: str | None = None) -> None:
    last_parser_meta["parser_model"] = llm_model_name(model) if model is not None else None
    last_parser_meta["parser_ok"] = ok
    last_parser_meta["parser_error"] = None if ok or not error else error[:500]


@tool(args_schema=SearchInput)
def web_search(query: str, count: int = 5) -> str:
    """Знайти сторінки в інтернеті через DeepSeek Search.

    Викликай, коли потрібні квитки, житло або місця, і ще немає конкретного URL.
    Не відкривай сторінки цим інструментом: він повертає назву, URL, дату і короткий summary.
    Якщо результатів немає, отримаєш «Нічого не знайдено» і не треба повторювати той самий запит.

    Args:
        query: Пошуковий запит від 1 до 128 символів.
        count: Кількість результатів від 1 до 5.

    Returns:
        Нумеровані картки результатів або повідомлення, що нічого не знайдено.
    """
    status, body = call_api(
        "/search",
        {"query": query, "count": count, "need_content": False},
        timeout=SEARCH_TIMEOUT,
    )
    if status != 200 or not isinstance(body, dict):
        return failure_text(status, body)
    return format_search_results(body.get("results"))


@tool(args_schema=FetchInput)
def web_fetch(url: str) -> str:
    """Відкрити одну сторінку і повернути коротку картку фактів.

    Викликай лише для одного-двох URL, коли summary з web_search не містить ціни
    або тривалості. Повертає PageCard: kind, price_amount, price_currency,
    duration_minutes, area, facts. null означає, що на сторінці цього не було.
    Не повторюй той самий URL після відмови політики або відсутності вмісту.

    Args:
        url: Адреса http(s) без логіна і пароля.

    Returns:
        JSON картки PageCard або коротке пояснення, чому сторінку не прочитано.
    """
    global last_parser_meta
    _set_parser_meta(None, False)
    status, body = call_api("/fetch", {"url": url}, timeout=FETCH_TIMEOUT)
    if status != 200 or not isinstance(body, dict):
        return failure_text(status, body)
    content = str(body.get("content") or "")
    title = str(body.get("title") or "")
    if not content.strip() and not title.strip():
        return "Немає вмісту за цією адресою"
    model = None
    try:
        from config import build_llm

        model = build_llm()
        card = extract_card(f"{title}\n\n{content}", url, llm=model)
        if not card.title:
            card.title = title
        _set_parser_meta(model, True)
        return "PageCard:\n" + card.model_dump_json(ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 — зламаний розбір не валить граф
        _set_parser_meta(model, False, str(exc))
        snippet = content[:SNIPPET_LIMIT]
        return f"Картку не розібрано.\n{snippet}"


@tool(args_schema=DurationInput)
def estimate_duration(category: str) -> str:
    """Оцінити тривалість візиту, якщо сторінка її не назвала.

    Викликай, коли в PageCard поле duration_minutes дорівнює null.
    Музей і галерея — 150 хв, парк і оглядовий майданчик — 60, обід — 60,
    вечеря — 90, кафе — 45, дорога до центру — 20. Невідома категорія — 90 хв.
    Джерело оцінки завжди default, навіть для відомої категорії.

    Args:
        category: Тип місця або прийому їжі українською чи англійською.

    Returns:
        Рядок із хвилинами і позначкою duration_source: default.
    """
    key = DURATION_ALIASES.get(category.casefold())
    minutes = DURATION_MINUTES.get(key or "", 90)
    label = key or category.casefold()
    return f"{label}: {minutes} хв (duration_source: default)"


@tool(args_schema=BudgetInput)
def check_budget(
    limit_uah: float,
    transport_uah: float,
    stay_uah: float,
    activities_uah: float,
    meals_uah: float,
) -> str:
    """Перевірити, чи вміщається поїздка в ліміт.

    Викликай після того, як ціни вже зібрані в гривнях. Інструмент не шукає ціни
    і не конвертує валюту. До 85% ліміту включно — fits, вище і до 100% — tight,
    понад ліміт — deficit.

    Args:
        limit_uah: Стеля бюджету в грн.
        transport_uah: Дорога в грн.
        stay_uah: Житло в грн.
        activities_uah: Вхідні квитки в грн.
        meals_uah: Їжа в грн.

    Returns:
        Сума, частка ліміту і вердикт fits, tight або deficit.
    """
    total = transport_uah + stay_uah + activities_uah + meals_uah
    if total <= limit_uah * 0.85:
        verdict = "fits"
    elif total <= limit_uah:
        verdict = "tight"
    else:
        verdict = "deficit"
    share = total / limit_uah
    return (
        f"Разом {total:.2f} грн із ліміту {limit_uah:.2f} грн ({share:.0%}). "
        f"Вердикт: {verdict}."
    )


ALL_TOOLS = [web_search, web_fetch, estimate_duration, check_budget]
