"""Stable library entry points for running CoCoNet outside the bundled CLI.

Other clients (custom CLIs, orchestration, containers) should prefer this module:
load configuration via :func:`load_coconet_config`, then execute with
:func:`run_coconet`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from coconet.config import CoconetConfig
from coconet.logging_utils import configure_logging
from coconet.model import CoconetModel, SpinupCheckpoint

__all__ = [
    "CoconetRunResult",
    "load_coconet_config",
    "run_coconet",
]

logger = logging.getLogger(__name__)


def _config_field_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(CoconetConfig))


def _apply_mapping_overrides(config: CoconetConfig, values: Mapping[str, Any]) -> None:
    valid = _config_field_names()
    for key, value in values.items():
        if key not in valid:
            raise TypeError(f"Unknown CoconetConfig attribute: {key!r}")
        setattr(config, key, value)


@dataclass(slots=True)
class CoconetRunResult:
    """Outcome of :func:`run_coconet` (extend with metrics later if needed)."""

    output_file: str
    # State after the ensemble-0 spinup; pass it to another run whose settings are
    # identical up to start_year (e.g. a no-intervention baseline) to skip its spinup.
    spinup_checkpoint: SpinupCheckpoint | None = None


def load_coconet_config(
    *,
    config_file: str | Path | None = None,
    parameter_file: str | Path | None = None,
    env_prefix: str = "COCONET_",
    scenario: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> CoconetConfig:
    """Build a :class:`~coconet.config.CoconetConfig` from files, env, and overrides.

    Resolution order:

    1. Defaults from :class:`~coconet.config.CoconetConfig`.
    2. Optional YAML ``config_file`` (same keys as the dataclass fields).
    3. Optional legacy NetLogo CSV ``parameter_file``.
    4. Environment variables ``{env_prefix}FIELD`` (see :meth:`CoconetConfig.from_file`).
    5. ``scenario`` mapping (flat dict of dataclass fields).
    6. Keyword ``overrides`` (same names as dataclass fields); ``None`` values are skipped.

    Unknown keys in ``scenario`` or ``overrides`` raise ``TypeError``.
    """
    cfg = CoconetConfig.from_file(
        config_file=config_file,
        parameter_file=parameter_file,
        env_prefix=env_prefix,
    )
    if scenario is not None:
        _apply_mapping_overrides(cfg, scenario)
    for key, value in overrides.items():
        if value is None:
            continue
        _apply_mapping_overrides(cfg, {key: value})
    return cfg


def run_coconet(
    config: CoconetConfig,
    *,
    configure_logs: bool = False,
    log_level: str | None = None,
    spinup_checkpoint: SpinupCheckpoint | None = None,
) -> CoconetRunResult:
    """Run the simulation for ``config`` and return basic result metadata.

    ``configure_logs``: when true, call :func:`~coconet.logging_utils.configure_logging`
    using ``log_level`` if set, otherwise ``config.log_level``.

    ``spinup_checkpoint``: the ``spinup_checkpoint`` of an earlier result. The caller
    must ensure both runs are identical up to ``start_year`` (same reefs, climate,
    biology, and no intervention starting before ``start_year``); the spinup is then
    skipped and the output is the same as a full run.

    Parallel ensembles use the ``spawn`` start method, so scripts that call this
    function must guard their entry point with ``if __name__ == "__main__":``.

    Idiomatic logging for library embedders: set ``configure_logs=False`` and configure
    the standard logging module (or your framework's logging) yourself before calling.
    """
    if configure_logs:
        level = log_level if log_level is not None else config.log_level
        configure_logging(level)

    logger.debug(
        "CoCoNet run starting: reefs_file=%s coastline_file=%s output_file=%s",
        config.reefs_file,
        config.coastline_file,
        config.output_file,
    )
    model = CoconetModel(config, spinup_checkpoint=spinup_checkpoint)
    model.run()
    logger.debug("CoCoNet run finished: output_file=%s", config.output_file)
    return CoconetRunResult(
        output_file=config.output_file,
        spinup_checkpoint=model.spinup_checkpoint_out,
    )
