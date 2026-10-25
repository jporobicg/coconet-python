"""Tool-calling loop. The language model never computes numbers itself."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from coconet_assistant.ollama_client import LLMResponse
from coconet_assistant.options import INTERVENTION_TYPES, config_field_names, explicit_ssps
from coconet_assistant.prompts import SYSTEM_PROMPT
from coconet_assistant.tools import TOOL_SCHEMAS, AssistantStore, ToolDispatcher

MAX_TOOL_ROUNDS = 8


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


@dataclass
class AgentTurn:
    assistant_text: str
    messages: list[dict[str, Any]]
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    pending_scenario_id: str | None = None


def _tool_message(call_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_name": name,
        "tool_call_id": call_id,
        "content": json.dumps(payload, default=str),
    }


def _check_requested_climate(
    result: dict[str, Any],
    requested: set[float],
    store: AssistantStore,
) -> dict[str, Any]:
    """Reject a built scenario whose SSP contradicts the one the manager named.

    Small local models sometimes read "SSP1-2.6" as 1.9. The manager's own words win.
    """
    if not result.get("ok") or not requested:
        return result
    built = result["result"]["config"].get("SSP")
    if built in requested:
        return result
    store.last_validated_id = None
    wanted = ", ".join(str(v) for v in sorted(requested))
    return {
        "ok": False,
        "tool": "build_scenario",
        "errors": [
            f"The manager asked for SSP {wanted} but the scenario uses SSP {built}. "
            f"Call build_scenario again with ssp={wanted}."
        ],
    }


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}  # fmt: skip
# Intervention params that are dates or names, not magnitudes the manager must state.
_NON_MAGNITUDE_PARAMS = frozenset(
    {"type", "region", "start_year", "upper_start_year", "lower_start_year"}
)


def _stated_numbers(texts: list[str]) -> set[float]:
    numbers: set[float] = set()
    for text in texts:
        for match in re.finditer(r"\d+(?:\.\d+)?", text.replace(",", "")):
            numbers.add(float(match.group()))
        for word in re.findall(r"[a-z]+", text.lower()):
            if word in _NUMBER_WORDS:
                numbers.add(float(_NUMBER_WORDS[word]))
    return numbers


def _is_stated(value: Any, stated: set[float]) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    v = float(value)
    # "30%" may arrive as 0.3, and "0.3" as 30.
    return any(math.isclose(v, s) or math.isclose(v * 100, s) or math.isclose(v, s * 100)
               for s in stated)  # fmt: skip


@dataclass
class _Dropped:
    """A value the language model supplied that the manager never stated."""

    label: str
    value: Any
    field: str | None  # CoconetConfig field it would have set, when known


def _param_field(itype: str, key: str) -> str | None:
    hint = str(INTERVENTION_TYPES.get(itype, {}).get("params", {}).get(key, ""))
    first = hint.split(" ")[0] if hint else ""
    return first if first in config_field_names() else None


def _drop_unstated_magnitudes(
    args: dict[str, Any],
    stated: set[float],
) -> tuple[dict[str, Any], list[_Dropped]]:
    """Remove numbers the manager never gave, so the documented defaults apply.

    Small local models fill gaps with invented values (10 vessels, 100 ha, a cover
    threshold of 20 instead of 0.2, a run to 2100). Intervention start years are
    exempt: the app applies sensible defaults to those.
    """
    dropped: list[_Dropped] = []
    cleaned = dict(args)
    items = []
    for item in args.get("interventions") or []:
        if not isinstance(item, dict):
            items.append(item)
            continue
        kept = dict(item)
        itype = str(item.get("type", "intervention"))
        for key, value in item.items():
            if key not in _NON_MAGNITUDE_PARAMS and not _is_stated(value, stated):
                kept.pop(key)
                dropped.append(_Dropped(f"{itype} {key}", value, _param_field(itype, key)))
        items.append(kept)
    if "interventions" in args:
        cleaned["interventions"] = items
    overrides = dict(args.get("config_overrides") or {})
    for key, value in list(overrides.items()):
        if not _is_stated(value, stated):
            overrides.pop(key)
            dropped.append(_Dropped(key, value, key))
    if "config_overrides" in args:
        cleaned["config_overrides"] = overrides
    for key in ("ensemble_runs", "end_year"):
        value = args.get(key)
        if value is not None and not _is_stated(value, stated):
            cleaned.pop(key)
            dropped.append(_Dropped(key, value, key))
    return cleaned, dropped


def _dropped_notes(dropped: list[_Dropped], config: dict[str, Any]) -> list[str]:
    """Card notes for dropped values, skipping those that equal the default anyway."""
    notes = []
    for item in dropped:
        if item.field is not None and config.get(item.field) == item.value:
            continue
        if item.label == "end_year":
            notes.append(
                f"end_year={item.value} was not in your request; the run ends in "
                f"{config.get('end_year')}. Say an end year if you want another."
            )
            continue
        notes.append(
            f"{item.label}={item.value} was not in your request, so the default is used. "
            "Say a value if you want another."
        )
    return notes


def run_agent_turn(
    client: LLMClient,
    store: AssistantStore,
    user_text: str,
    *,
    history: list[dict[str, Any]] | None = None,
    dispatcher: ToolDispatcher | None = None,
    on_tool: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> AgentTurn:
    """Send one manager message through the tool loop and return the reply."""

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    ## ~ LLM turn with grounded tool calls ~ ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    tools = dispatcher or ToolDispatcher(store)
    messages: list[dict[str, Any]] = list(history or [])
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    messages.append({"role": "user", "content": user_text})

    tool_results: list[dict[str, Any]] = []
    assistant_text = ""
    requested_ssps = explicit_ssps(user_text)
    # Numbers the manager has given in this conversation (earlier turns count too,
    # so "same again under SSP5-8.5" keeps the values from the first question).
    manager_texts = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
    stated = _stated_numbers(manager_texts)

    for _ in range(max_rounds):
        response = client.chat(messages, tools=TOOL_SCHEMAS)
        if response.tool_calls:
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    for call in response.tool_calls
                ],
            }
            messages.append(assistant_message)
            for call in response.tool_calls:
                arguments = call["arguments"]
                dropped: list[_Dropped] = []
                if call["name"] == "build_scenario":
                    arguments, dropped = _drop_unstated_magnitudes(arguments, stated)
                result = tools.call(call["name"], arguments)
                if call["name"] == "build_scenario":
                    result = _check_requested_climate(result, requested_ssps, store)
                    if dropped and result.get("ok"):
                        spec = store.scenarios[result["result"]["scenario_id"]]
                        spec.notes.extend(_dropped_notes(dropped, spec.config))
                        result["result"]["notes"] = list(spec.notes)
                tool_results.append({"name": call["name"], "result": result})
                if on_tool is not None:
                    on_tool(call["name"], call["arguments"], result)
                messages.append(_tool_message(call["id"], call["name"], result))
            continue

        assistant_text = response.content or ""
        messages.append({"role": "assistant", "content": assistant_text})
        break
    else:
        assistant_text = (
            "The assistant reached the tool-call limit without a final answer. "
            "The last tool results are still valid; no extra numbers were invented."
        )
        messages.append({"role": "assistant", "content": assistant_text})

    return AgentTurn(
        assistant_text=assistant_text,
        messages=messages,
        tool_results=tool_results,
        pending_scenario_id=store.last_validated_id,
    )
