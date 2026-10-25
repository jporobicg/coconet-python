from __future__ import annotations

import json
from dataclasses import fields

import pytest
from coconet.config import CoconetConfig
from coconet_assistant.options import SUPPORTED_SSP, config_field_names, list_options, resolve_ssp
from coconet_assistant.scenario import (
    DEFAULT_ENSEMBLE_RUNS,
    MIN_ENSEMBLE_RUNS,
    ScenarioValidationError,
    baseline_config,
    build_scenario,
    validate_scenario,
)


def test_list_options_summary_fits_small_context() -> None:
    options = list_options()
    assert "parameters" not in options
    assert "fields" not in options["intervention_types"]["coral_seeding"]
    assert options["intervention_types"]["coral_seeding"]["params"]["reefs"] == "seed_reefs"
    # Keep the default tool payload well under a few thousand tokens (~4 chars per token).
    assert len(json.dumps(options)) < 8000


def test_list_options_uses_real_config_fields() -> None:
    options = list_options(detail="fields")
    catalog_fields = {item["field"] for item in options["parameters"]}
    assert catalog_fields == {f.name for f in fields(CoconetConfig)}
    assert options["climate_scenarios"]["allowed"] == list(SUPPORTED_SSP)
    assert "cots_control" in options["intervention_types"]
    assert "coral_seeding" in options["intervention_types"]
    assert options["intervention_types"]["cots_control"]["fields"][0] == "start_CoTS_control"


def test_fractions_must_be_between_zero_and_one() -> None:
    with pytest.raises(ScenarioValidationError, match="seed_threshold is a fraction"):
        build_scenario(
            climate="4.5",
            interventions=[{"type": "coral_seeding", "start_year": 2030, "threshold": 20}],
        )


def test_whole_reef_region_aliases() -> None:
    spec = build_scenario(
        climate="SSP2-4.5",
        region="all",
        interventions=[{"type": "cots_control", "start_year": 2030}],
    )
    assert spec.config["CoTS_vessels_GBR"] > 0
    assert build_scenario(climate="2.6", region="whole GBR").config["SSP"] == 2.6


def test_resolve_ssp_aliases() -> None:
    assert resolve_ssp("SSP2-4.5") == 4.5
    assert resolve_ssp("4.5") == 4.5
    assert resolve_ssp(2.6) == 2.6
    with pytest.raises(ValueError, match="Unsupported climate"):
        resolve_ssp(3.0)


def test_build_northern_cots_and_seeding() -> None:
    spec = build_scenario(
        name="northern CoTS + TT seed",
        climate="SSP2-4.5",
        region="northern sector",
        ensemble_runs=20,
        interventions=[
            {"type": "cots_control", "start_year": 2030, "vessels": 6},
            {"type": "coral_seeding", "start_year": 2030},
        ],
    )
    assert spec.config["SSP"] == 4.5
    assert spec.config["ensemble_runs"] == 20
    assert spec.config["start_CoTS_control"] == 2030
    assert spec.config["CoTS_vessels_N"] == 6
    assert spec.config["start_coral_seeding"] == 2030
    assert spec.config["seed_reefs"] > 0
    assert spec.config["seed_hectares"] > 0
    assert spec.config["seed_threshold"] > 0
    assert spec.config["end_year"] >= 2050
    assert spec.config["intervene_region"] == "N"
    validate_scenario(spec)


def test_build_rejects_unknown_override() -> None:
    with pytest.raises(ScenarioValidationError, match="Unknown CoconetConfig"):
        build_scenario(config_overrides={"not_a_field": 1})


def test_validate_rejects_single_ensemble() -> None:
    with pytest.raises(ScenarioValidationError, match="never a single stochastic"):
        validate_scenario({"ensemble_runs": 1, "SSP": 2.6})


def test_validate_rejects_cots_without_vessels() -> None:
    with pytest.raises(ScenarioValidationError, match="CoTS_vessels"):
        validate_scenario(
            {
                "SSP": 2.6,
                "ensemble_runs": 20,
                "start_CoTS_control": 2030,
                "end_year": 2050,
            }
        )


def test_validate_rejects_zero_seed_threshold() -> None:
    with pytest.raises(ScenarioValidationError, match="seed_threshold"):
        validate_scenario(
            {
                "SSP": 4.5,
                "ensemble_runs": 20,
                "end_year": 2050,
                "start_coral_seeding": 2030,
                "seed_reefs": 10,
                "seed_hectares": 8,
                "seed_threshold": 0.0,
            }
        )


def test_validate_unknown_key_lists_the_field() -> None:
    with pytest.raises(ScenarioValidationError, match="magic_knob"):
        validate_scenario({"SSP": 2.6, "ensemble_runs": 20, "magic_knob": 1})


def test_baseline_strips_interventions() -> None:
    spec = build_scenario(
        climate=2.6,
        interventions=[{"type": "cots_control", "start_year": 2030, "region": "N", "vessels": 4}],
    )
    base = baseline_config(spec.config)
    assert base["SSP"] == 2.6
    assert base["ensemble_runs"] == DEFAULT_ENSEMBLE_RUNS
    assert base["start_CoTS_control"] == 9999
    assert base["CoTS_vessels_N"] == 0
    validate_scenario(base)


def test_min_ensemble_constant() -> None:
    assert MIN_ENSEMBLE_RUNS >= 2
    assert "ensemble_runs" in config_field_names()
