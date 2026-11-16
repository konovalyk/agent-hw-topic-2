"""ReAct-граф планування вихідних: agent, tools, halt і formatter."""

from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

from logger import current_logger
from safety import (
    MAX_REPEATS,
    MAX_STEPS,
    TIMEOUT_SECONDS,
    LoopDetector,
    normalize_args,
    preview_loop,
    timed_out,
)
from tools import ALL_TOOLS, last_parser_meta

SYSTEM_PROMPT = """Ти агент планування вихідних. Відповідай українською.
Факти про квитки, житло і місця бери лише з інструментів.
Шукай українською і додавай у запит «грн», якщо потрібна ціна.
Спочатку web_search. За всю відповідь відкрий не більше двох сторінок через web_fetch,
і лише коли summary не містить ціни в гривнях або тривалості.
Якщо web_fetch повернув «Картку не розібрано», бери факти з уривка
і переходь до estimate_duration та check_budget. Нові URL після цього не відкривай.
Ціну і хвилини бери з PageCard. null не замінюй здогадкою.
Якщо duration_minutes порожній або в уривку немає тривалості, виклич estimate_duration.
У повний день заклади обід 60 хв і вечерю 90 хв.
Перед check_budget переведи всі ціни в грн. Не підставляй суму, якої не було в джерелі.
Після «Нічого не знайдено», відмови політики, «тимчасово недоступне»
або «Картку не розібрано» той самий виклик не повторюй.
Місця, що не влізли в час або бюджет, перелічи як альтернативи.
У джерелах залишай URL сторінок.
"""


class DayPlan(BaseModel):
    """Один день поїздки."""

    day: str = Field(description="День, наприклад п'ятниця або 2026-11-21")
    slots: list[str] = Field(
        default_factory=list,
        description="Слоти у вигляді '10:00 Музей — 150 хв'",
    )


class TravelPlan(BaseModel):
    """Структурована відповідь агента."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(description="Короткий план або відповідь українською")
    confidence: float = Field(description="Впевненість від 0 до 1")
    sources: list[str] = Field(default_factory=list, description="URL джерел")
    budget_verdict: str = Field(description="fits, tight, deficit або unknown")
    plan: list[DayPlan] = Field(default_factory=list, description="Дні і слоти")
    alternatives: list[str] = Field(default_factory=list, description="Місця, які не влізли")

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @field_validator("budget_verdict")
    @classmethod
    def known_verdict(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"fits", "tight", "deficit", "unknown"}:
            return "unknown"
        return normalized

    @field_validator("sources", "alternatives")
    @classmethod
    def drop_blank(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int
    started_at: float
    recent_call_hashes: Annotated[list, operator.add]
    halted: bool


def message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def tool_calls_of(message: object) -> list:
    calls = getattr(message, "tool_calls", None) or []
    return list(calls)


def call_parts(tool_call: dict) -> tuple[str, dict]:
    return str(tool_call.get("name", "")), normalize_args(tool_call.get("args"))


def _log(step: int, node: str, input_data: object, output_data: object, tool_calls: list | None = None, extra: dict | None = None) -> None:
    logger = current_logger()
    if logger is None:
        return
    logger.log_step(step, node, input_data, output_data, tool_calls, extra)


def limit_message(reason: str, state: AgentState) -> str:
    bits: list[str] = []
    for message in state["messages"][-4:]:
        text = message_text(message).strip()
        if text:
            bits.append(text[:200])
    tail = " | ".join(bits)
    body = f"{reason} Часткова відповідь: {tail}" if tail else reason
    return body[:1500]


def initial_state(query: str, started_at: float | None = None) -> AgentState:
    return {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ],
        "step_count": 0,
        "started_at": time.monotonic() if started_at is None else started_at,
        "recent_call_hashes": [],
        "halted": False,
    }


def build_app(
    llm,
    *,
    max_steps: int = MAX_STEPS,
    timeout_seconds: float = TIMEOUT_SECONDS,
    max_repeats: int = MAX_REPEATS,
):
    """Збирає граф. llm уже має бути клієнтом чату, інструменти чіпляються тут."""
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_map = {item.name: item for item in ALL_TOOLS}

    def agent_node(state: AgentState) -> dict:
        step = state["step_count"]
        if timed_out(state["started_at"], timeout_seconds):
            text = limit_message("Час виконання вичерпано (timeout).", state)
            _log(step, "agent", "timeout", text)
            return {"messages": [AIMessage(content=text)], "halted": True, "step_count": step}
        if step >= max_steps:
            text = limit_message(f"Досягнуто ліміт кроків ({max_steps}).", state)
            _log(step, "agent", "max_steps", text)
            return {"messages": [AIMessage(content=text)], "halted": True, "step_count": step}

        last_input = message_text(state["messages"][-1])
        response = llm_with_tools.invoke(state["messages"])
        names = [call.get("name", "") for call in tool_calls_of(response)]
        _log(step + 1, "agent", last_input, message_text(response) or ", ".join(names), names)
        return {"messages": [response], "step_count": step + 1}

    def tools_node(state: AgentState) -> dict:
        message = state["messages"][-1]
        outputs: list[ToolMessage] = []
        hashes: list[str] = []
        detector = LoopDetector(max_repeats)
        detector.recent_calls = list(state["recent_call_hashes"])
        before = len(detector.recent_calls)
        for tool_call in tool_calls_of(message):
            name, args = call_parts(tool_call)
            detector.check(name, args)
            tool = tool_map.get(name)
            if tool is None:
                result = f"Невідомий інструмент: {name}"
            else:
                try:
                    result = str(tool.invoke(args))
                except Exception as exc:  # noqa: BLE001 — валідація і збої повертаються моделі
                    result = f"Помилка валідації: {exc}"
            outputs.append(
                ToolMessage(content=result, tool_call_id=tool_call.get("id", name))
            )
            extra = None
            if name == "web_fetch":
                extra = {
                    "parser_model": last_parser_meta.get("parser_model"),
                    "parser_ok": last_parser_meta.get("parser_ok"),
                }
                if last_parser_meta.get("parser_error"):
                    extra["parser_error"] = last_parser_meta["parser_error"]
            _log(state["step_count"], "tools", {name: args}, result, [name], extra)
        hashes = detector.recent_calls[before:]
        return {"messages": outputs, "recent_call_hashes": hashes}

    def formatter_node(state: AgentState) -> dict:
        if timed_out(state["started_at"], timeout_seconds):
            text = limit_message("Час виконання вичерпано під час форматування.", state)
            _log(state["step_count"], "formatter", "timeout", text)
            return {"messages": [AIMessage(content=text)], "halted": True}
        from config import invoke_structured

        prompt = HumanMessage(
            content=(
                "Сформуй TravelPlan українською лише з фактів у діалозі. "
                "Не вигадуй ціни. Якщо бюджет не перевіряли, budget_verdict = unknown. "
                "Порожній plan допустимий, коли користувач не просив розклад."
            )
        )
        try:
            plan = invoke_structured(llm, TravelPlan, [*state["messages"], prompt])
            if not isinstance(plan, TravelPlan):
                plan = TravelPlan.model_validate(plan)
            text = plan.model_dump_json(ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001 — помилку видно в траєкторії, а не підміняємо готовою відповіддю
            text = f"Не вдалося зібрати TravelPlan: {exc}"
        _log(state["step_count"], "formatter", "TravelPlan", text)
        return {"messages": [AIMessage(content=text)]}

    def halt_node(state: AgentState) -> dict:
        if timed_out(state["started_at"], timeout_seconds):
            reason = "Час виконання вичерпано (timeout)."
        elif state["step_count"] >= max_steps:
            reason = f"Досягнуто ліміт кроків ({max_steps})."
        else:
            reason = "Виявлено зациклення: той самий інструмент з тими самими аргументами."
        text = limit_message(reason, state)
        _log(state["step_count"], "halt", reason, text)
        return {"messages": [AIMessage(content=text)], "halted": True}

    def route(state: AgentState) -> Literal["tools", "formatter", "halt", "__end__"]:
        if state.get("halted"):
            return "__end__"
        message = state["messages"][-1]
        calls = tool_calls_of(message)
        if calls:
            pairs = [call_parts(call) for call in calls]
            blocked = (
                timed_out(state["started_at"], timeout_seconds)
                or state["step_count"] >= max_steps
                or preview_loop(state["recent_call_hashes"], pairs, max_repeats)
            )
            return "halt" if blocked else "tools"
        return "formatter"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("formatter", formatter_node)
    graph.add_node("halt", halt_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    graph.add_edge("formatter", END)
    graph.add_edge("halt", END)
    return graph.compile()


def run_agent(
    query: str,
    *,
    llm=None,
    max_steps: int = MAX_STEPS,
    timeout_seconds: float = TIMEOUT_SECONDS,
    max_repeats: int = MAX_REPEATS,
    started_at: float | None = None,
):
    """Запускає граф на одному запиті. Клієнт будується тут, якщо його не передали."""
    from config import build_llm

    app = build_app(
        llm or build_llm(),
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        max_repeats=max_repeats,
    )
    return app.invoke(
        initial_state(query, started_at=started_at),
        config={"recursion_limit": 40},
    )
