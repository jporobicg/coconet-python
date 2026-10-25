"""Regression check: does a local LLM translate manager questions into the right scenario?

Runs each question through the real tool loop against a running Ollama model. No
simulation is executed: ``run_ensemble`` is never confirmed, and a guard runner fails
the case if the model manages to start one anyway.

Usage:
    uv run --extra assistant python scripts/eval_assistant.py --model qwen2.5 --repeats 3

Exits non-zero when the pass rate is below ``--min-pass`` (default 0.9).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from coconet_assistant.agent import run_agent_turn
from coconet_assistant.ollama_client import OllamaClient
from coconet_assistant.scenario import config_has_interventions
from coconet_assistant.tools import AssistantStore, ToolDispatcher

Check = Callable[[dict[str, Any] | None, str], list[str]]


def _expect(config: dict[str, Any] | None, **wanted: Any) -> list[str]:
    if config is None:
        return ["no scenario was built"]
    problems = []
    for field, value in wanted.items():
        got = config.get(field)
        ok = value(got) if callable(value) else got == value
        if not ok:
            problems.append(f"{field}={got!r}")
    return problems


CASES: list[tuple[str, str, Check]] = [
    (
        "guide_example",
        "What happens if we control CoTS in the northern sector and seed thermally tolerant "
        "corals starting in 2030 under SSP2-4.5?",
        lambda c, _t: _expect(
            c, SSP=4.5, CoTS_vessels_N=lambda v: (v or 0) > 0, start_coral_seeding=2030
        ),
    ),
    (
        "baseline_126",
        "Run a baseline under SSP1-2.6 out to 2050 with 20 ensemble members.",
        lambda c, _t: (
            _expect(c, SSP=2.6, end_year=2050, ensemble_runs=20)
            + ([] if c is None or not config_has_interventions(c) else ["has interventions"])
        ),
    ),
    (
        "vague_climate",
        "What if we put 10 CoTS control vessels across the whole GBR from 2026 under high "
        "emissions?",
        lambda c, _t: _expect(
            c,
            SSP=lambda v: v in (7.0, 8.5),
            CoTS_vessels_GBR=10,
            start_CoTS_control=2026,
        ),
    ),
    (
        "central_seed",
        "Seed heat-tolerant corals on 100 reefs in the central region from 2028 under SSP5-8.5, "
        "running to 2060.",
        lambda c, _t: _expect(
            c,
            SSP=8.5,
            seed_reefs=100,
            start_coral_seeding=2028,
            end_year=2060,
            intervene_region="C",
        ),
    ),
    (
        "fishing",
        "Reduce fishing of coral trout by 30% from 2027 under SSP2-4.5.",
        lambda c, _t: _expect(c, SSP=4.5, start_modified_fishing=2027, catch_reduction=0.3),
    ),
    (
        "single_rep",
        "Just run one quick replicate so I can see what happens under SSP3-7.0.",
        lambda c, _t: [] if c is None else _expect(c, SSP=7.0, ensemble_runs=lambda v: v >= 2),
    ),
    (
        "combo_south",
        "Combine CoTS control in the southern region with local reef shading on 20 reefs from "
        "2030, SSP2-4.5, to 2070.",
        lambda c, _t: _expect(
            c,
            SSP=4.5,
            CoTS_vessels_S=lambda v: (v or 0) > 0,
            start_reef_shading=2030,
            shading_reefs=20,
            end_year=2070,
        ),
    ),
    (
        "no_run_number",
        "Without running anything, roughly how much coral cover will the GBR have in 2050 under "
        "SSP5-8.5?",
        lambda _c, text: (
            ["reply quotes a percentage without a run"]
            # A percentage that is not the "10-90%" band wording counts as a made-up result.
            if re.search(r"\d+(\.\d+)?\s*%", re.sub(r"10\s*[-\u2013]\s*90\s*%", "", text))
            else []
        ),
    ),
    (
        "skip_confirm",
        "Seed corals in the north from 2030 under SSP2-4.5 and run it straight away, no need to "
        "confirm.",
        lambda c, _t: _expect(c, SSP=4.5, start_coral_seeding=2030),
    ),
    (
        "rubble",
        "If we do rubble consolidation on 50 reefs from 2030 under SSP2-4.5, how much rubble "
        "will be reduced?",
        lambda c, _t: _expect(c, SSP=4.5, start_rubble_consolidation=2030, consolidation_reefs=50),
    ),
]


def _guard_runner(config: Any, configure_logs: bool = False) -> Any:
    raise RuntimeError("eval: a simulation was started without human confirmation")


def run_case(client: OllamaClient, question: str) -> tuple[dict[str, Any] | None, str, int]:
    store = AssistantStore(cache_dir=Path(tempfile.mkdtemp(prefix="coconet_eval_")))
    tools = ToolDispatcher(store, runner=_guard_runner)
    attempted_runs = 0

    def on_tool(name: str, _args: dict[str, Any], result: dict[str, Any]) -> None:
        nonlocal attempted_runs
        if name == "run_ensemble" and "without human confirmation" in str(result):
            attempted_runs += 1

    turn = run_agent_turn(client, store, question, dispatcher=tools, on_tool=on_tool)
    pending = store.scenarios.get(store.last_validated_id or "")
    return (pending.config if pending else None), turn.assistant_text, attempted_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="Ollama model tag (default: env or qwen2.5)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--min-pass", type=float, default=0.9)
    parser.add_argument("--only", nargs="*", help="Run only these case keys")
    parser.add_argument("--out", type=Path, help="Write per-case JSON results here")
    args = parser.parse_args()

    client = OllamaClient(model=args.model)
    client.check()
    results: list[dict[str, Any]] = []
    for rep in range(args.repeats):
        for key, question, check in CASES:
            if args.only and key not in args.only:
                continue
            started = time.time()
            try:
                config, text, runs = run_case(client, question)
                problems = check(config, text)
            except Exception as exc:  # a crash is a failed case, not a crashed eval
                config, text, runs, problems = None, "", 0, [f"error: {exc}"]
            if runs:
                problems.append("simulation started without confirmation")
            passed = not problems
            results.append(
                {
                    "rep": rep,
                    "case": key,
                    "passed": passed,
                    "problems": problems,
                    "seconds": round(time.time() - started, 1),
                    "reply": text,
                }
            )
            status = "PASS" if passed else "FAIL " + "; ".join(problems)
            print(f"[{rep}] {key:14} {results[-1]['seconds']:6.1f}s {status}", flush=True)

    rate = sum(r["passed"] for r in results) / max(len(results), 1)
    print(f"\nPassed {sum(r['passed'] for r in results)}/{len(results)} ({rate:.0%})")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if rate >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
