"""Manager-facing Streamlit chat for CoCoNet scenario exploration."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from coconet_assistant.agent import run_agent_turn
from coconet_assistant.ollama_client import (
    OllamaClient,
    OllamaNotAvailable,
    default_ollama_host,
    default_ollama_model,
)
from coconet_assistant.prompts import SYSTEM_PROMPT
from coconet_assistant.scenario import display_settings
from coconet_assistant.tools import AssistantStore, ToolDispatcher

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REEFS = REPO_ROOT / "legacy" / "reefs2024.csv"
DEFAULT_COAST = REPO_ROOT / "legacy" / "coastline.csv"


def _init_state() -> None:
    if "store" not in st.session_state:
        st.session_state.store = AssistantStore(
            cache_dir=REPO_ROOT / ".coconet_assistant_cache",
            reefs_file=str(DEFAULT_REEFS) if DEFAULT_REEFS.is_file() else None,
            coastline_file=str(DEFAULT_COAST) if DEFAULT_COAST.is_file() else None,
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []
    if "progress" not in st.session_state:
        st.session_state.progress = None
    if "last_plots" not in st.session_state:
        st.session_state.last_plots = []
    if "ollama_error" not in st.session_state:
        st.session_state.ollama_error = None


def _client() -> OllamaClient:
    return OllamaClient(
        host=st.session_state.get("ollama_host") or default_ollama_host(),
        model=st.session_state.get("ollama_model") or default_ollama_model(),
    )


def _render_settings(scenario_id: str) -> None:
    store: AssistantStore = st.session_state.store
    spec = store.scenarios.get(scenario_id)
    if spec is None:
        return
    st.subheader("Proposed settings")
    st.caption("These values will be sent to CoCoNet if you confirm. Nothing has been run yet.")
    rows = display_settings(spec.config)
    st.dataframe(
        [
            {
                "Setting": row["label"],
                "Field": row["field"],
                "Value": row["value"],
                "Model default": row["default"],
            }
            for row in rows
        ],
        hide_index=True,
        use_container_width=True,
    )
    for note in spec.notes:
        st.info(note)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Confirm and run ensemble", type="primary"):
            _run_confirmed(scenario_id)
    with col_b:
        if st.button("Cancel proposal"):
            store.last_validated_id = None
            st.rerun()


def _run_confirmed(scenario_id: str) -> None:
    store: AssistantStore = st.session_state.store
    store.mark_confirmed(scenario_id)
    tools = ToolDispatcher(store)
    progress_box = st.empty()

    def _progress(payload: dict) -> None:
        st.session_state.progress = payload
        fraction = float(payload.get("fraction") or 0)
        progress_box.progress(min(max(fraction, 0.0), 1.0), text=payload.get("message", ""))

    store.progress_hook = _progress
    try:
        run_payload = tools.call(
            "run_ensemble",
            {"scenario_id": scenario_id, "confirmed": True, "include_baseline": True},
        )
    except Exception as exc:
        st.error(str(exc))
        store.progress_hook = None
        return
    store.progress_hook = None
    if not run_payload.get("ok"):
        st.error("; ".join(run_payload.get("errors") or ["Run failed."]))
        return

    result = run_payload["result"]
    run_id = result["run"]["run_id"]
    summary = tools.call("summarise_results", {"run_id": run_id})
    plot_args = {"run_id": run_id}
    compare_args = {"run_ids": [run_id]}
    if result.get("baseline_run"):
        base_id = result["baseline_run"]["run_id"]
        tools.call("summarise_results", {"run_id": base_id})
        plot_args["baseline_run_id"] = base_id
        compare_args["run_ids"].append(base_id)
        compare_args["baseline_run_id"] = base_id
    plot_payload = tools.call("plot", plot_args)
    compare_payload = None
    if len(compare_args["run_ids"]) > 1:
        compare_payload = tools.call("compare_scenarios", compare_args)

    figures = []
    if plot_payload.get("ok"):
        figures = plot_payload["result"].get("figure_paths") or []
    st.session_state.last_plots = figures

    explanation = _explain_results(
        {
            "run": run_payload,
            "summary": summary,
            "plot": plot_payload,
            "comparison": compare_payload,
        }
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": explanation, "plots": figures}
    )
    store.last_validated_id = None
    progress_box.progress(1.0, text="Finished")
    st.rerun()


def _explain_results(payload: dict) -> str:
    facts = []
    summary = (payload.get("summary") or {}).get("result") or {}
    facts.extend(summary.get("facts") or [])
    comparison = (payload.get("comparison") or {}).get("result") or {}
    facts.extend(comparison.get("facts") or [])
    suggestions = comparison.get("suggestions") or []
    caveats = summary.get("caveats") or comparison.get("caveats") or []
    try:
        response = _client().chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Write a short manager briefing using ONLY these tool facts. "
                        "Do not add numbers that are not listed.\n\n"
                        + json.dumps(
                            {
                                "facts": facts,
                                "suggestions": suggestions,
                                "caveats": caveats,
                                "rubble_note": summary.get("rubble_note"),
                            },
                            default=str,
                        )
                    ),
                },
            ],
            tools=None,
        )
        return response.content or "\n".join(facts)
    except OllamaNotAvailable:
        blocks = ["**Results (from the model tools)**", *facts]
        if suggestions:
            blocks.extend(["", "**Next scenarios worth trying**", *suggestions])
        if caveats:
            blocks.extend(["", "**Caveats**", *caveats])
        return "\n\n".join(blocks)


def _handle_chat(user_text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})
    store: AssistantStore = st.session_state.store
    lowered = user_text.strip().lower()
    if lowered in {"yes", "y", "confirm", "run", "run it", "go"} and store.last_validated_id:
        _run_confirmed(store.last_validated_id)
        return
    try:
        turn = run_agent_turn(
            _client(),
            store,
            user_text,
            history=st.session_state.agent_history,
        )
    except OllamaNotAvailable as exc:
        st.session_state.ollama_error = str(exc)
        st.session_state.messages.append({"role": "assistant", "content": str(exc)})
        return
    st.session_state.agent_history = turn.messages
    st.session_state.messages.append({"role": "assistant", "content": turn.assistant_text})


def _sidebar() -> None:
    store: AssistantStore = st.session_state.store
    st.sidebar.header("Local model")
    st.session_state.ollama_host = st.sidebar.text_input(
        "Ollama host",
        value=st.session_state.get("ollama_host") or default_ollama_host(),
    )
    st.session_state.ollama_model = st.sidebar.text_input(
        "Model tag",
        value=st.session_state.get("ollama_model") or default_ollama_model(),
        help="Any local Ollama model that supports tool calling, e.g. qwen2.5 or llama3.1.",
    )
    if st.sidebar.button("Check Ollama"):
        try:
            info = _client().check()
            st.session_state.ollama_error = None
            st.sidebar.success(f"Connected. Models: {', '.join(info['models']) or '(none listed)'}")
        except (OllamaNotAvailable, ImportError) as exc:
            st.session_state.ollama_error = str(exc)
            st.sidebar.error(str(exc))

    st.sidebar.header("Model inputs")
    reefs = st.sidebar.text_input("Reefs CSV", value=store.reefs_file or "")
    coast = st.sidebar.text_input("Coastline CSV", value=store.coastline_file or "")
    store.reefs_file = reefs or None
    store.coastline_file = coast or None
    cache = st.sidebar.text_input("Cache directory", value=str(store.cache_dir))
    store.cache_dir = Path(cache)
    st.sidebar.caption(
        "Completed ensembles are stored on disk. Asking the same question again "
        "reuses the cached output."
    )
    if store.runs:
        st.sidebar.header("Completed runs")
        for run in store.runs.values():
            st.sidebar.write(f"- `{run['run_id']}` — {run.get('name')}")


def main() -> None:
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    ## ~ Manager chat workspace ~ ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    st.set_page_config(page_title="CoCoNet manager assistant", layout="wide")
    _init_state()
    st.title("CoCoNet manager assistant")
    st.write(
        "Ask a management question in plain language. The assistant will translate "
        "it into a CoCoNet ensemble, show the settings for your confirmation, then "
        "explain coral, CoTS, and fish results against a baseline. "
        "All numbers come from the model tools. The language model runs locally "
        "through Ollama so data stays on this machine."
    )

    _sidebar()
    if st.session_state.ollama_error:
        st.warning(st.session_state.ollama_error)

    store: AssistantStore = st.session_state.store
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for path in message.get("plots") or []:
                if Path(path).is_file():
                    st.image(path)

    if store.last_validated_id:
        _render_settings(store.last_validated_id)

    if st.session_state.last_plots:
        st.subheader("Latest plots")
        for path in st.session_state.last_plots:
            if Path(path).is_file():
                st.image(path)

    prompt = st.chat_input(
        'Example: "What if we control CoTS in the northern sector and seed '
        'thermally tolerant coral from 2030 under SSP2-4.5?"'
    )
    if prompt:
        _handle_chat(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
