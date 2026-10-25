"""Tool-calling loop. The language model never computes numbers itself."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from coconet_assistant.ollama_client import LLMResponse
from coconet_assistant.options import explicit_ssps
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
                result = tools.call(call["name"], call["arguments"])
                if call["name"] == "build_scenario":
                    result = _check_requested_climate(result, requested_ssps, store)
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
