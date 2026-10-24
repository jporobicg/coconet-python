from __future__ import annotations

from pathlib import Path
from typing import Any

from coconet_assistant.agent import run_agent_turn
from coconet_assistant.ollama_client import LLMResponse
from coconet_assistant.scenario import ScenarioValidationError
from coconet_assistant.tools import TOOL_SCHEMAS, AssistantStore, ToolDispatcher
from tests.unit.test_assistant_summarise import write_output


class ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        assert tools is None or tools is TOOL_SCHEMAS or tools
        if not self.responses:
            return LLMResponse(content="done", tool_calls=[], raw={})
        return self.responses.pop(0)


def _fake_runner(output_src: Path):
    def runner(config: Any, configure_logs: bool = False) -> Any:
        dest = Path(config.output_file)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(output_src.read_text(encoding="utf-8"), encoding="utf-8")
        return type("Result", (), {"output_file": str(dest)})()

    return runner


def test_tool_schemas_cover_requested_names() -> None:
    names = {item["function"]["name"] for item in TOOL_SCHEMAS}
    assert names == {
        "list_options",
        "build_scenario",
        "validate_scenario",
        "run_ensemble",
        "summarise_results",
        "compare_scenarios",
        "plot",
    }


def test_run_ensemble_requires_confirmation(tmp_path: Path) -> None:
    store = AssistantStore(cache_dir=tmp_path / "cache")
    tools = ToolDispatcher(store)
    built = tools.call(
        "build_scenario",
        {
            "climate": "SSP2-4.5",
            "region": "N",
            "interventions": [
                {"type": "cots_control", "start_year": 2030, "vessels": 3},
            ],
        },
    )
    assert built["ok"]
    scenario_id = built["result"]["scenario_id"]
    refused = tools.call("run_ensemble", {"scenario_id": scenario_id, "confirmed": False})
    assert refused["ok"] is False
    assert any(
        "confirmed" in err.lower() or "not confirmed" in err.lower() for err in refused["errors"]
    )


def test_tool_layer_run_summarise_compare(tmp_path: Path, fixture_dir: Path) -> None:
    output_src = write_output(tmp_path / "fixture.csv", intervention=True)
    store = AssistantStore(
        cache_dir=tmp_path / "cache",
        reefs_file=str(fixture_dir / "reefs_two.csv"),
        coastline_file=str(fixture_dir / "coastline_clip.csv"),
    )
    tools = ToolDispatcher(store, runner=_fake_runner(output_src))
    built = tools.call(
        "build_scenario",
        {
            "name": "demo",
            "climate": "4.5",
            "end_year": 2050,
            "ensemble_runs": 4,
            "interventions": [
                {
                    "type": "coral_seeding",
                    "start_year": 2030,
                    "reefs": 10,
                    "hectares": 8,
                    "threshold": 0.3,
                }
            ],
        },
    )
    scenario_id = built["result"]["scenario_id"]
    validated = tools.call("validate_scenario", {"scenario_id": scenario_id})
    assert validated["ok"] is True
    store.mark_confirmed(scenario_id)
    ran = tools.call(
        "run_ensemble",
        {"scenario_id": scenario_id, "confirmed": True, "include_baseline": True},
    )
    assert ran["ok"] is True
    run_id = ran["result"]["run"]["run_id"]
    base_id = ran["result"]["baseline_run"]["run_id"]
    summary = tools.call("summarise_results", {"run_id": run_id})
    assert summary["ok"] is True
    assert "facts" in summary["result"]
    compared = tools.call(
        "compare_scenarios",
        {"run_ids": [run_id, base_id], "baseline_run_id": base_id},
    )
    assert compared["ok"] is True
    plotted = tools.call("plot", {"run_id": run_id, "baseline_run_id": base_id})
    assert plotted["ok"] is True
    assert plotted["result"]["panels"]

    # Repeat should hit the disk cache (same fake output, same fingerprint).
    again = tools.call(
        "run_ensemble",
        {"scenario_id": scenario_id, "confirmed": True, "include_baseline": False},
    )
    assert again["ok"] is True
    assert again["result"]["run"]["cached"] is True


def test_agent_loop_with_mocked_llm(tmp_path: Path, fixture_dir: Path) -> None:
    output_src = write_output(tmp_path / "fixture.csv", intervention=True)
    store = AssistantStore(
        cache_dir=tmp_path / "cache",
        reefs_file=str(fixture_dir / "reefs_two.csv"),
        coastline_file=str(fixture_dir / "coastline_clip.csv"),
    )
    tools = ToolDispatcher(store, runner=_fake_runner(output_src))
    client = ScriptedLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "1",
                        "name": "build_scenario",
                        "arguments": {
                            "climate": "SSP2-4.5",
                            "region": "northern",
                            "ensemble_runs": 8,
                            "interventions": [
                                {"type": "cots_control", "start_year": 2030, "vessels": 4},
                                {"type": "coral_seeding", "start_year": 2030},
                            ],
                        },
                    }
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "2",
                        "name": "validate_scenario",
                        "arguments": {},  # filled below after first call
                    }
                ],
                raw={},
            ),
            LLMResponse(
                content="Proposed settings are ready. Please confirm before I run the ensemble.",
                tool_calls=[],
                raw={},
            ),
        ]
    )

    # First turn: build. The scripted validate call has empty arguments; patch after build.
    def on_tool(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        if name == "build_scenario" and result.get("ok"):
            client.responses[0].tool_calls[0]["arguments"] = {
                "scenario_id": result["result"]["scenario_id"]
            }

    turn = run_agent_turn(
        client,
        store,
        "what if we control CoTS in the northern sector and seed thermally tolerant coral from 2030 under SSP2-4.5?",
        dispatcher=tools,
        on_tool=on_tool,
    )
    assert store.last_validated_id is not None
    assert "confirm" in turn.assistant_text.lower()
    spec = store.scenarios[store.last_validated_id]
    assert spec.config["SSP"] == 4.5
    assert spec.config["CoTS_vessels_N"] == 4
    assert spec.config["start_coral_seeding"] == 2030

    # Running without confirmation is still blocked at the tool layer.
    blocked = tools.call(
        "run_ensemble",
        {"scenario_id": store.last_validated_id, "confirmed": False},
    )
    assert blocked["ok"] is False


def test_unknown_tool_is_reported(tmp_path: Path) -> None:
    tools = ToolDispatcher(AssistantStore(cache_dir=tmp_path / "cache"))
    result = tools.call("invent_numbers", {})
    assert result["ok"] is False
    assert "Unknown tool" in result["errors"][0]


def test_validation_error_object() -> None:
    err = ScenarioValidationError(["a", "b"])
    assert err.errors == ["a", "b"]
    assert "a" in str(err)
