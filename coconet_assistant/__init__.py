"""Manager-facing assistant layer on top of the CoCoNet Python port.

This package translates plain-language management questions into validated
``CoconetConfig`` scenarios, runs ensembles through ``coconet.api``, and
summarises results. Numbers reported to managers must come from these tools,
not from the language model.
"""

from __future__ import annotations

from coconet_assistant.scenario import (
    DEFAULT_ENSEMBLE_RUNS,
    MIN_ENSEMBLE_RUNS,
    ScenarioSpec,
    ScenarioValidationError,
    build_scenario,
    validate_scenario,
)

__all__ = [
    "DEFAULT_ENSEMBLE_RUNS",
    "MIN_ENSEMBLE_RUNS",
    "ScenarioSpec",
    "ScenarioValidationError",
    "build_scenario",
    "validate_scenario",
]
