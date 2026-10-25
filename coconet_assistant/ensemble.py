"""Run validated scenarios as ensembles through ``coconet.api``."""

from __future__ import annotations

import logging
import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from coconet.api import run_coconet
from coconet.config import effective_ensemble_workers

from coconet_assistant.cache import load_cached_run, run_cache_key, save_cached_run
from coconet_assistant.scenario import (
    MIN_ENSEMBLE_RUNS,
    ScenarioSpec,
    ScenarioValidationError,
    config_as_coconet,
    validate_scenario,
)

ProgressCallback = Callable[[dict[str, Any]], None]
LOGGER = logging.getLogger(__name__)

# Measured on the full GBR network (reefs2024.csv, 3,805 reefs) on a 40-core workstation:
# the serial ensemble-0 spinup and one simulation ensemble-year (ensembles in parallel).
SECONDS_PER_SPINUP_YEAR = 85.0
SECONDS_PER_ENSEMBLE_YEAR = 160.0


def estimate_run_seconds(
    config: dict[str, Any],
    *,
    include_baseline: bool,
    spinup_cached: bool,
    cpu_count: int | None = None,
) -> float:
    """Rough wall-clock estimate for a scenario (and its baseline) on the full GBR.

    The baseline shares the scenario's spinup, so only the first run pays for it.
    """
    runs = int(config["ensemble_runs"])
    workers = effective_ensemble_workers(int(config.get("ensemble_threads", 0)), runs)
    if cpu_count is not None:
        workers = max(1, min(workers, cpu_count))
    waves = math.ceil(runs / workers)
    years = int(config["end_year"]) - int(config["start_year"]) + 1
    per_run = waves * years * SECONDS_PER_ENSEMBLE_YEAR
    spinup = (
        0.0
        if spinup_cached
        else (int(config["spinup_backtrack_years"]) + 1) * (SECONDS_PER_SPINUP_YEAR)
    )
    return spinup + per_run * (2 if include_baseline else 1)


_PROGRESS_RE = re.compile(r"Progress:\s*ensemble=(?P<ensemble>\S+)\s+year=(?P<year>\S+)")


class EnsembleProgressHandler(logging.Handler):
    """Watch ``coconet.model`` INFO lines of the form ``Progress: ensemble=… year=…``."""

    def __init__(
        self,
        callback: ProgressCallback | None,
        *,
        ensemble_runs: int,
        start_year: int,
        end_year: int,
        spinup_years: int = 0,
    ) -> None:
        super().__init__(level=logging.INFO)
        self.callback = callback
        self.ensemble_runs = max(ensemble_runs, 1)
        self.start_year = start_year
        self.end_year = end_year
        year_span = max(end_year - start_year + 1, 1)
        # Ensemble-0 spinup also logs one progress line per year.
        self.expected_steps = self.ensemble_runs * year_span + max(spinup_years, 0)
        self.seen = 0

    def emit(self, record: logging.LogRecord) -> None:
        if self.callback is None:
            return
        message = record.getMessage()
        match = _PROGRESS_RE.search(message)
        if match is None:
            return
        self.seen += 1
        try:
            ensemble = int(float(match.group("ensemble")))
        except ValueError:
            ensemble = 0
        try:
            year = int(float(match.group("year")))
        except ValueError:
            year = self.start_year
        fraction = min(self.seen / self.expected_steps, 0.99)
        self.callback(
            {
                "ensemble": ensemble,
                "year": year,
                "seen_steps": self.seen,
                "expected_steps": self.expected_steps,
                "fraction": fraction,
                "message": f"ensemble {ensemble}, year {year}",
            }
        )


def run_ensemble(
    config: dict[str, Any] | ScenarioSpec,
    *,
    cache_dir: str | Path,
    confirmed: bool = False,
    run_name: str | None = None,
    progress_callback: ProgressCallback | None = None,
    runner: Callable[..., Any] | None = None,
    spinup_checkpoint: Any | None = None,
    spinup_sink: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a validated scenario as an ensemble (never a single stochastic replicate).

    ``confirmed`` must be true: the manager has to accept the proposed settings
    before the model starts. Completed runs are cached under ``cache_dir``.

    ``spinup_checkpoint`` skips the ensemble-0 spinup (the caller guarantees the
    runs are identical up to ``start_year``). When ``spinup_sink`` is given, the
    run's own checkpoint is stored in it under ``"checkpoint"`` (``None`` for a
    cached run). It is kept out of the returned payload, which the LLM sees.
    """

    if not confirmed:
        raise ScenarioValidationError(
            [
                "run_ensemble refused: the manager has not confirmed the proposed "
                "settings. Show the settings and wait for confirmation."
            ]
        )

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    ## ~ Validate, cache lookup, then run ~ ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    if isinstance(config, ScenarioSpec):
        name = run_name or config.name
        fields = validate_scenario(config.config)
    else:
        fields = validate_scenario(config)
        name = run_name or "scenario"

    if int(fields["ensemble_runs"]) < MIN_ENSEMBLE_RUNS:
        raise ScenarioValidationError(
            [f"ensemble_runs must be >= {MIN_ENSEMBLE_RUNS} (never a single stochastic run)."]
        )

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    key = run_cache_key(fields)
    cached = load_cached_run(cache_root, key)
    if cached is not None:
        cached["name"] = name
        cached["run_id"] = key
        if progress_callback is not None:
            progress_callback(
                {
                    "ensemble": fields["ensemble_runs"],
                    "year": fields["end_year"],
                    "seen_steps": 1,
                    "expected_steps": 1,
                    "fraction": 1.0,
                    "message": "loaded from cache",
                    "cached": True,
                }
            )
        return cached

    run_folder = cache_root / key
    run_folder.mkdir(parents=True, exist_ok=True)
    output_path = run_folder / "output.csv"
    fields_for_run = dict(fields)
    fields_for_run["output_file"] = str(output_path)
    fields_for_run["log_level"] = fields_for_run.get("log_level") or "INFO"

    cfg = config_as_coconet(fields_for_run)
    handler = EnsembleProgressHandler(
        progress_callback,
        ensemble_runs=int(fields_for_run["ensemble_runs"]),
        start_year=int(fields_for_run["start_year"]),
        end_year=int(fields_for_run["end_year"]),
        spinup_years=(
            0
            if spinup_checkpoint is not None
            else int(fields_for_run["spinup_backtrack_years"]) + 1
        ),
    )
    model_logger = logging.getLogger("coconet.model")
    model_logger.addHandler(handler)
    execute = runner if runner is not None else run_coconet
    extra = {"spinup_checkpoint": spinup_checkpoint} if spinup_checkpoint is not None else {}
    try:
        result = execute(cfg, configure_logs=False, **extra)
        if spinup_sink is not None:
            spinup_sink["checkpoint"] = getattr(result, "spinup_checkpoint", None)
        output_file = getattr(result, "output_file", str(output_path))
        if Path(output_file) != output_path and Path(output_file).is_file():
            shutil.copyfile(output_file, output_path)
        if not output_path.is_file():
            raise FileNotFoundError(f"Model finished but output file is missing: {output_path}")
    except Exception:
        if run_folder.exists():
            shutil.rmtree(run_folder, ignore_errors=True)
        raise
    finally:
        model_logger.removeHandler(handler)

    if progress_callback is not None:
        progress_callback(
            {
                "ensemble": fields_for_run["ensemble_runs"],
                "year": fields_for_run["end_year"],
                "seen_steps": handler.expected_steps,
                "expected_steps": handler.expected_steps,
                "fraction": 1.0,
                "message": "finished",
                "cached": False,
            }
        )

    payload = {
        "run_id": key,
        "name": name,
        "output_file": str(output_path),
        "config": fields_for_run,
        "cached": False,
        "ensemble_runs": int(fields_for_run["ensemble_runs"]),
        "spinup_reused": spinup_checkpoint is not None,
    }
    save_cached_run(cache_root, key, {k: v for k, v in payload.items() if k != "output_file"})
    return payload
