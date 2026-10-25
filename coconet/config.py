from __future__ import annotations

import csv
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


def effective_ensemble_workers(ensemble_threads: int, ensemble_runs: int) -> int:
    """Cap for parallel simulation worker processes (ensembles 1..ensemble_runs).

    ensemble_threads: 0 = auto (min(CPU count, simulation ensembles)), 1 = serial,
    N > 1 = use at most N workers (still capped by ensemble_runs).
    """
    sim = max(0, ensemble_runs)
    if sim == 0:
        return 1
    if ensemble_threads <= 0:
        cap = os.cpu_count() or 1
        return max(1, min(cap, sim))
    return max(1, min(int(ensemble_threads), sim))


def use_parallel_ensemble_run(ensemble_threads: int, ensemble_runs: int) -> bool:
    return ensemble_runs > 0 and effective_ensemble_workers(ensemble_threads, ensemble_runs) > 1


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            number = float(value)
            if number.is_integer():
                return int(number)
            return number
        return int(value)
    except ValueError:
        return value


@dataclass(slots=True)
class CoconetConfig:
    # Input/output
    reefs_file: str = "legacy/reefs2024.csv"
    coastline_file: str = "legacy/coastline.csv"
    output_file: str = "output.csv"
    parameter_file: str | None = None
    log_level: str = "INFO"
    # 0 = auto (min(CPU cores, simulation ensemble count)); 1 = serial; N > 1 = process pool cap.
    ensemble_threads: int = 0

    # Scenario controls (matching NetLogo globals)
    SSP: float = 2.6
    ensemble_runs: int = 20
    start_year: int = 1956
    # Years before start_year where ensemble-0 spinup begins (legacy NetLogo: 50).
    spinup_backtrack_years: int = 50
    save_year: int = 1986
    projection_year: int = 2025
    search_year: int = 9999
    end_year: int = 2030

    start_catchment_restore: int = 9999
    restore_timeframe: float = 0.0

    start_CoTS_control: int = 9999
    eco_threshold: float = 8.0
    CoTS_threshold: float = 999999.0
    coral_threshold: float = 0.0
    CoTS_vessels_GBR: int = 0
    CoTS_vessels_FN: int = 0
    CoTS_vessels_N: int = 0
    CoTS_vessels_C: int = 0
    CoTS_vessels_S: int = 0
    CoTS_vessels_sector: int = 0

    intervene_lon_min: float = 140.0
    intervene_lon_max: float = 155.0
    intervene_lat_min: float = -25.0
    intervene_lat_max: float = -9.0
    # Region code (FN, N, C, S) that box-based interventions are limited to, on top of
    # the lat/lon box. "GBR" (default) applies no region filter, as in NetLogo.
    intervene_region: str = "GBR"

    start_modified_zoning: int = 9999
    rezoned_reefs: int = 0
    start_modified_fishing: int = 9999
    catch_reduction: float = 0.0
    start_lower_sizelimit: int = 9999
    start_upper_sizelimit: int = 9999
    start_CoTSlimit: int = 9999

    start_emperor_release: int = 9999
    release_reefs: int = 500
    release_threshold: float = 9999999.0
    release_number: float = 0.0

    start_regional_shading: int = 9999
    regional_shading_reduction: float = 0.0
    start_rubble_consolidation: int = 9999
    consolidation_reefs: int = 0
    consolidation_threshold: float = 0.0
    consolidation_hectares: float = 0.0

    start_coral_seeding: int = 9999
    seed_reefs: int = 0
    seed_threshold: float = 0.0
    seed_hectares: float = 0.0
    hybrid_fraction: float = 0.0
    dominance: float = 0.0

    start_coral_slick: int = 9999
    slick_reefs: int = 0
    slick_threshold: float = 0.0
    slick_hectares: float = 0.0

    start_reef_shading: int = 9999
    shading_reefs: int = 0
    reef_shading_reduction: float = 0.0
    start_pH_protection: int = 9999
    pH_reefs: int = 0
    pH_protection: float = 0.0

    # Legacy interface-only globals
    search_mode: int = 0
    perfect_intervention: str = ""
    unregulated_fishing: bool = False

    @classmethod
    def from_file(
        cls,
        config_file: str | Path | None = None,
        parameter_file: str | Path | None = None,
        env_prefix: str = "COCONET_",
    ) -> CoconetConfig:
        config = cls()

        if config_file is not None:
            loaded = yaml.safe_load(Path(config_file).read_text()) or {}
            for k, v in loaded.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        if parameter_file is not None:
            config.parameter_file = str(parameter_file)
            config._apply_legacy_parameter_file(Path(parameter_file))

        config._apply_env_overrides(env_prefix)
        return config

    def _apply_env_overrides(self, env_prefix: str) -> None:
        field_lookup = {f.name.lower(): f.name for f in fields(self)}
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue
            key = env_key[len(env_prefix) :].lower()
            if key not in field_lookup:
                continue
            attr = field_lookup[key]
            setattr(self, attr, _parse_scalar(env_value))

    def _apply_legacy_parameter_file(self, path: Path) -> None:
        label_to_attr = {
            "climate scenario": "SSP",
            "ensemble runs": "ensemble_runs",
            "start year": "start_year",
            "spinup backtrack (years)": "spinup_backtrack_years",
            "save year": "save_year",
            "projection year": "projection_year",
            "search year": "search_year",
            "end year": "end_year",
            "cots control start year": "start_CoTS_control",
            "cots control ecological threshold (cots per ha)": "eco_threshold",
            "cots control cots threshold (cots per ha)": "CoTS_threshold",
            "cots control coral threshold (cots per ha)": "coral_threshold",
            "cots vessels across gbr": "CoTS_vessels_GBR",
            "cots vessels in far-northern region": "CoTS_vessels_FN",
            "cots vessels in northern region": "CoTS_vessels_N",
            "cots vessels in central region": "CoTS_vessels_C",
            "cots vessels in southern region": "CoTS_vessels_S",
            "catchment restoration start year": "start_catchment_restore",
            "catchment restoration timescale (years)": "restore_timeframe",
            "future zoning start year": "start_modified_zoning",
            "number of reefs included in future rezoning": "rezoned_reefs",
            "reduction in fisheries catch start year": "start_modified_fishing",
            "fractional reduction in fisheries catches": "catch_reduction",
            "upper fish size limit start year": "start_upper_sizelimit",
            "lower fish size limit start year": "start_lower_sizelimit",
            "exclude fishing from active outbreak reefs start year": "start_CoTSlimit",
            "emperor release start year": "start_emperor_release",
            "number of release reefs": "release_reefs",
            "maximum adult emperors (per ha) for release": "release_threshold",
            "number of juvenile emperors released per reef": "release_number",
            "regional shading start year": "start_regional_shading",
            "absolute dhw reduction due to regional shading (dhw)": "regional_shading_reduction",
            "minimum longitude of interventions": "intervene_lon_min",
            "maximum longitude of interventions": "intervene_lon_max",
            "minimum latitude of interventions": "intervene_lat_min",
            "maximum latitude of interventions": "intervene_lat_max",
            "rubble consolidation start year": "start_rubble_consolidation",
            "annual number of consolidated reefs": "consolidation_reefs",
            "minimum rubble cover threshold for consolidation [0 1]": "consolidation_threshold",
            "total annual consolidated area (ha)": "consolidation_hectares",
            "thermally tolerant coral seeding start year": "start_coral_seeding",
            "annual number of reefs seeded with coral": "seed_reefs",
            "maximum coral cover threshold for coral seeding [0 1]": "seed_threshold",
            "total annual area of seeded corals (ha)": "seed_hectares",
            "fraction of staghorn acropora corals able to hybridise with thermally tolerant corals [0 1]": "hybrid_fraction",
            "dominance of thermally tolerant corals in setting thermal tolerance of hybrids [0 1]": "dominance",
            "coral slicks start year": "start_coral_slick",
            "annual number of reefs with coral slicks released": "slick_reefs",
            "maximum coral cover threshold for coral slicks [0 1]": "slick_threshold",
            "total annual area of slick corals (ha)": "slick_hectares",
            "reef shading start year": "start_reef_shading",
            "annual number of reefs locally shaded": "shading_reefs",
            "fractional dhw reduction due to local shading [0 1]": "reef_shading_reduction",
            "ocean acidification treatment start year": "start_pH_protection",
            "annual number of reefs treated for ocean acidification": "pH_reefs",
            "fractional protection from ocean acidification [0 1]": "pH_protection",
            "cots vessels in active sector": "CoTS_vessels_sector",
        }

        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                label = row[0].strip().lower()
                # Header row "ensemble" marks end of config lines (not e.g. "ensemble runs").
                if label.startswith("ensemble") and label == "ensemble":
                    break
                attr = label_to_attr.get(label)
                if attr is None:
                    continue
                setattr(self, attr, _parse_scalar(row[1]))
