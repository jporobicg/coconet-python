"""Manager-facing Streamlit chat for CoCoNet scenario exploration."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

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
from coconet_assistant.summarise import FACT_METRICS, format_metric
from coconet_assistant.tools import AssistantStore, ToolDispatcher

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REEFS = REPO_ROOT / "legacy" / "reefs2024.csv"
DEFAULT_COAST = REPO_ROOT / "legacy" / "coastline.csv"
POLL_SECONDS = 3
VERDICTS = {"better": "Clear improvement", "worse": "Clear deterioration", "unclear": "Not clear"}


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
    if "job" not in st.session_state:
        st.session_state.job = None
    if "ollama_error" not in st.session_state:
        st.session_state.ollama_error = None


def _client() -> OllamaClient:
    return OllamaClient(
        host=st.session_state.get("ollama_host") or default_ollama_host(),
        model=st.session_state.get("ollama_model") or default_ollama_model(),
    )


def _duration(seconds: float) -> str:
    if seconds < 90:
        return "about a minute"
    if seconds < 3600:
        return f"about {round(seconds / 60)} minutes"
    return f"about {seconds / 3600:.1f} hours"


def _range(seconds: float) -> str:
    # The estimate uses the slowest measured speed; a quieter machine ran ~30% faster.
    if seconds < 3600:
        return _duration(seconds)
    return f"about {0.7 * seconds / 3600:.0f} to {seconds / 3600:.0f} hours"


def _render_settings(scenario_id: str) -> None:
    store: AssistantStore = st.session_state.store
    spec = store.scenarios.get(scenario_id)
    if spec is None:
        return
    st.subheader("Proposed settings")
    st.caption("These values will be sent to CoCoNet if you confirm. Nothing has been run yet.")
    rows = [
        {
            "Setting": row["label"],
            "Field": row["field"],
            "Value": row["value"],
            "Model default": row["default"],
        }
        for row in display_settings(spec.config)
        if row["field"] not in {"reefs_file", "coastline_file"}
    ]
    # Tall enough to show every row: the start years must never hide behind a scroll bar.
    st.dataframe(rows, hide_index=True, use_container_width=True, height=36 * (len(rows) + 1) + 4)
    replaced = [note for note in spec.notes if "was not in your request" in note]
    other = [note for note in spec.notes if note not in replaced]
    if other:
        st.info("**How these settings were chosen**\n\n" + "\n".join(f"- {n}" for n in other))
    if replaced:
        st.warning(
            "**Values the assistant proposed that you did not give** (defaults used instead)"
            "\n\n" + "\n".join(f"- {n}" for n in replaced)
        )

    plan = ToolDispatcher(store).run_plan(scenario_id)
    if plan["cached"] and plan["baseline_cached"]:
        st.success("These runs are already computed. Results will load immediately.")
    else:
        shared = " The baseline reuses the scenario's spin-up." if plan["spinup_shared"] else ""
        st.warning(
            f"Estimated run time on this machine: {_range(plan['estimate_seconds'])} "
            f"for the scenario and its no-intervention baseline, depending on other load "
            f"on the machine.{shared} The run continues "
            "if you close this tab; ask the same question again later to load the results."
        )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Confirm and run ensemble", type="primary"):
            _start_job(scenario_id)
    with col_b:
        if st.button("Cancel proposal"):
            store.last_validated_id = None
            st.rerun()


def _start_job(scenario_id: str) -> None:
    """Run the confirmed scenario and its baseline in a background thread.

    The thread keeps running if the browser tab closes; finished runs land in the
    disk cache, so asking again later loads them.
    """
    store: AssistantStore = st.session_state.store
    store.mark_confirmed(scenario_id)
    job: dict[str, Any] = {"scenario_id": scenario_id, "progress": None, "done": False}

    def work() -> None:
        # Streamlit calls are not allowed from this thread: only update ``job``.
        store.progress_hook = lambda payload: job.update(progress=payload)
        try:
            job["payload"] = ToolDispatcher(store).call(
                "run_ensemble", {"scenario_id": scenario_id, "include_baseline": True}
            )
        except Exception as exc:  # reported to the manager, never swallowed
            job["payload"] = {"ok": False, "errors": [str(exc)]}
        finally:
            store.progress_hook = None
            job["done"] = True

    threading.Thread(target=work, name=f"coconet-{scenario_id}", daemon=True).start()
    st.session_state.job = job
    store.last_validated_id = None
    st.rerun()


def _render_job(job: dict[str, Any]) -> None:
    progress = job.get("progress") or {}
    fraction = min(max(float(progress.get("fraction") or 0.0), 0.0), 1.0)
    st.progress(fraction, text=progress.get("message") or "Starting the ensemble...")
    st.caption(
        "CoCoNet is running the scenario and its no-intervention baseline. "
        "You can close this tab; the results are cached when the run finishes."
    )
    time.sleep(POLL_SECONDS)
    st.rerun()


def _short(text: str) -> str:
    # Table cells must stay narrow so the verdict column is visible without scrolling.
    return text.replace(" percentage points", " pts")


def _results_table(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for the manager: levels, paired change, and its verdict. Numbers from tools."""
    baseline = comparison["baseline"]
    year = comparison.get("end_year")
    table = []
    for row in comparison["rows"]:
        metric = row["metric"]
        if metric not in FACT_METRICS:
            continue
        for name, value in row["values"].items():
            paired = value.get("paired")
            if name == baseline or paired is None:
                continue
            base = row["values"].get(baseline, {})
            table.append(
                {
                    "Indicator": row["label"],
                    f"With ({year})": format_metric(metric, value["mean"]),
                    f"Without ({year})": format_metric(metric, base.get("mean", 0.0)),
                    "Paired change": _short(format_metric(metric, paired["mean"], signed=True)),
                    "10-90% range": _short(
                        f"{format_metric(metric, paired['q10'], signed=True)} to "
                        f"{format_metric(metric, paired['q90'], signed=True)}"
                    ),
                    "Better in": f"{paired['n_better']} of {paired['n_pairs']}",
                    "Verdict": VERDICTS[paired["direction"]],
                }
            )
    return table


def _finish_job(job: dict[str, Any]) -> None:
    st.session_state.job = None
    payload = job.get("payload") or {}
    if not payload.get("ok"):
        message = "; ".join(payload.get("errors") or ["Run failed."])
        st.session_state.messages.append(
            {"role": "assistant", "content": f"The run did not complete: {message}"}
        )
        return
    result = payload["result"]
    run_id = result["run"]["run_id"]
    base_id = result["baseline_run"]["run_id"]
    with st.spinner("Summarising the ensemble..."):
        sections = ToolDispatcher(st.session_state.store).report(run_id, base_id)
    explanation = _explain_results(sections)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": explanation,
            "sections": [
                {
                    "title": section["region_label"],
                    "table": _results_table(section["comparison"]),
                    "plots": section["figure_paths"],
                }
                for section in sections
            ],
        }
    )
    st.rerun()


def _explain_results(sections: list[dict[str, Any]]) -> str:
    facts: list[str] = []
    suggestions: list[str] = []
    caveats: list[str] = []
    for section in sections:
        comparison = section["comparison"]
        facts.extend(comparison.get("facts") or [])
        suggestions.extend(s for s in comparison.get("suggestions") or [] if s not in suggestions)
        caveats = comparison.get("caveats") or caveats
    try:
        response = _client().chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Write a short manager briefing using ONLY these tool facts. "
                        "Lead with the targeted region, then the whole GBR. Say which "
                        "changes are clear and which are within replicate noise. The run "
                        "is complete: do not ask for confirmation. Do not add numbers that "
                        "are not listed.\n\n"
                        + json.dumps(
                            {"facts": facts, "suggestions": suggestions, "caveats": caveats},
                            default=str,
                        )
                    ),
                },
            ],
            tools=None,
        )
        return response.content or "\n\n".join(facts)
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
        _start_job(store.last_validated_id)
        return
    try:
        with st.spinner("The assistant is preparing a scenario..."):
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


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for section in message.get("sections") or []:
            st.markdown(f"#### Results: {section['title']}")
            if section["table"]:
                st.dataframe(section["table"], hide_index=True, use_container_width=True)
            plots = [p for p in section["plots"] if Path(p).is_file()]
            for left, right in zip(plots[::2], [*plots[1::2], None], strict=False):
                cols = st.columns(2)
                cols[0].image(left)
                if right:
                    cols[1].image(right)


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
        _render_message(message)

    job = st.session_state.job
    if job is not None:
        if job["done"]:
            _finish_job(job)
        else:
            _render_job(job)
        return

    if store.last_validated_id:
        _render_settings(store.last_validated_id)

    prompt = st.chat_input(
        'Example: "What if we control CoTS in the northern sector and seed '
        'thermally tolerant coral from 2030 under SSP2-4.5?"'
    )
    if prompt:
        _handle_chat(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
