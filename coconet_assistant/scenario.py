"""Build and validate manager scenarios against the real ``CoconetConfig`` schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any

from coconet.config import CoconetConfig

from coconet_assistant.options import (
    FIELD_LABELS,
    INTERVENTION_START_FIELDS,
    INTERVENTION_TYPES,
    REGION_FILTER_CODES,
    SUPPORTED_SSP,
    VESSEL_FIELDS,
    config_field_names,
    config_field_types,
    default_config_values,
    resolve_cots_vessel_field,
    resolve_region_code,
    resolve_ssp,
)

DEFAULT_ENSEMBLE_RUNS = 20
MIN_ENSEMBLE_RUNS = 2
MAX_ENSEMBLE_RUNS = 200
INACTIVE_START_YEAR = 9999

# Defaults used only when a manager enables an intervention without a magnitude.
DEFAULT_COTS_VESSELS = 5
DEFAULT_SEED_REEFS = 50
DEFAULT_SEED_HECTARES = 40.0
DEFAULT_SEED_THRESHOLD = 0.30
DEFAULT_SLICK_REEFS = 50
DEFAULT_SLICK_HECTARES = 40.0
DEFAULT_SLICK_THRESHOLD = 0.30
DEFAULT_CONSOLIDATION_REEFS = 50
DEFAULT_CONSOLIDATION_HECTARES = 40.0
DEFAULT_CONSOLIDATION_THRESHOLD = 0.10
DEFAULT_CATCH_REDUCTION = 0.20
DEFAULT_RESTORE_TIMEFRAME = 10.0
DEFAULT_SHADING_REEFS = 50
DEFAULT_REEF_SHADING_REDUCTION = 0.30
DEFAULT_REGIONAL_SHADING_REDUCTION = 0.30
DEFAULT_PH_REEFS = 50
DEFAULT_PH_PROTECTION = 0.30
DEFAULT_RELEASE_NUMBER = 100.0
DEFAULT_REZONED_REEFS = 50

# Config fields that are fractions or cover proportions in [0, 1].
FRACTION_FIELDS: tuple[str, ...] = (
    "seed_threshold",
    "slick_threshold",
    "consolidation_threshold",
    "hybrid_fraction",
    "dominance",
    "catch_reduction",
    "reef_shading_reduction",
    "regional_shading_reduction",
    "pH_protection",
    "coral_threshold",
)

# Paths and logging are not part of the scientific scenario identity.
NON_SCENARIO_FIELDS = frozenset(
    {
        "output_file",
        "log_level",
        "parameter_file",
        "ensemble_threads",
    }
)


class ScenarioValidationError(ValueError):
    """Raised when a proposed config is not a valid ``CoconetConfig`` scenario."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class ScenarioSpec:
    """A named, validated set of ``CoconetConfig`` field values."""

    scenario_id: str
    name: str
    description: str
    config: dict[str, Any]
    interventions: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "interventions": self.interventions,
            "notes": self.notes,
            "display": display_settings(self.config),
        }


def _new_scenario_id(config: dict[str, Any], name: str) -> str:
    payload = json.dumps(
        {"name": name, "config": _identity_fields(config)},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"scen_{digest}"


def _identity_fields(config: dict[str, Any]) -> dict[str, Any]:
    return {k: config[k] for k in sorted(config) if k not in NON_SCENARIO_FIELDS}


def _coerce_config_value(field_name: str, value: Any) -> Any:
    types = config_field_types()
    expected = types[field_name]
    type_name = str(expected)

    if value is None:
        return value

    if "bool" in type_name and "int" not in type_name:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ScenarioValidationError([f"{field_name} must be a boolean, got {value!r}"])

    if "int" in type_name and "float" not in type_name:
        if isinstance(value, bool):
            raise ScenarioValidationError([f"{field_name} must be an integer, got {value!r}"])
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        raise ScenarioValidationError([f"{field_name} must be an integer, got {value!r}"])

    if "float" in type_name:
        if isinstance(value, bool):
            raise ScenarioValidationError([f"{field_name} must be a number, got {value!r}"])
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise ScenarioValidationError(
                    [f"{field_name} must be a number, got {value!r}"]
                ) from exc
        raise ScenarioValidationError([f"{field_name} must be a number, got {value!r}"])

    if expected is str or type_name == "str":
        return str(value)

    return value


def _apply_cots_control(config: dict[str, Any], item: dict[str, Any], notes: list[str]) -> None:
    start = int(item.get("start_year", config.get("projection_year", 2025)))
    config["start_CoTS_control"] = start
    if "eco_threshold" in item:
        config["eco_threshold"] = float(item["eco_threshold"])
    if "cots_threshold" in item:
        config["CoTS_threshold"] = float(item["cots_threshold"])
    if "coral_threshold" in item:
        config["coral_threshold"] = float(item["coral_threshold"])

    region = item.get("region")
    vessels = int(item.get("vessels", DEFAULT_COTS_VESSELS))
    if region:
        field = resolve_cots_vessel_field(str(region))
        config[field] = vessels
        notes.append(f"CoTS control uses {field}={vessels} from year {start}.")
        return

    # No region: if the caller already set a vessel field, keep it; else GBR-wide.
    if not any(int(config.get(name, 0)) > 0 for name in VESSEL_FIELDS):
        config["CoTS_vessels_GBR"] = vessels
        notes.append(f"CoTS control uses CoTS_vessels_GBR={vessels} from year {start}.")


def _apply_coral_seeding(config: dict[str, Any], item: dict[str, Any]) -> None:
    config["start_coral_seeding"] = int(item.get("start_year", 2030))
    config["seed_reefs"] = int(item.get("reefs", DEFAULT_SEED_REEFS))
    config["seed_hectares"] = float(item.get("hectares", DEFAULT_SEED_HECTARES))
    config["seed_threshold"] = float(item.get("threshold", DEFAULT_SEED_THRESHOLD))
    if "hybrid_fraction" in item:
        config["hybrid_fraction"] = float(item["hybrid_fraction"])
    if "dominance" in item:
        config["dominance"] = float(item["dominance"])


def _apply_coral_slick(config: dict[str, Any], item: dict[str, Any]) -> None:
    config["start_coral_slick"] = int(item.get("start_year", 2030))
    config["slick_reefs"] = int(item.get("reefs", DEFAULT_SLICK_REEFS))
    config["slick_hectares"] = float(item.get("hectares", DEFAULT_SLICK_HECTARES))
    config["slick_threshold"] = float(item.get("threshold", DEFAULT_SLICK_THRESHOLD))


def _apply_rubble(config: dict[str, Any], item: dict[str, Any], notes: list[str]) -> None:
    config["start_rubble_consolidation"] = int(item.get("start_year", 2030))
    config["consolidation_reefs"] = int(item.get("reefs", DEFAULT_CONSOLIDATION_REEFS))
    config["consolidation_hectares"] = float(item.get("hectares", DEFAULT_CONSOLIDATION_HECTARES))
    config["consolidation_threshold"] = float(
        item.get("threshold", DEFAULT_CONSOLIDATION_THRESHOLD)
    )
    notes.append(
        "Rubble consolidation will run inside the model, but rubble cover is not "
        "written to output.csv so trends cannot be plotted from the standard file."
    )


def _apply_intervention(config: dict[str, Any], item: dict[str, Any], notes: list[str]) -> None:
    itype = str(item.get("type", "")).strip().lower()
    if itype not in INTERVENTION_TYPES:
        known = ", ".join(sorted(INTERVENTION_TYPES))
        raise ScenarioValidationError(
            [f"Unknown intervention type {itype!r}. Valid types: {known}"]
        )

    if itype == "cots_control":
        _apply_cots_control(config, item, notes)
        return
    if itype == "coral_seeding":
        _apply_coral_seeding(config, item)
        return
    if itype == "coral_slick":
        _apply_coral_slick(config, item)
        return
    if itype == "rubble_consolidation":
        _apply_rubble(config, item, notes)
        return
    if itype == "catchment_restore":
        config["start_catchment_restore"] = int(item.get("start_year", 2030))
        config["restore_timeframe"] = float(item.get("timescale_years", DEFAULT_RESTORE_TIMEFRAME))
        return
    if itype == "fishing_reduction":
        config["start_modified_fishing"] = int(item.get("start_year", 2030))
        config["catch_reduction"] = float(item.get("fraction", DEFAULT_CATCH_REDUCTION))
        return
    if itype == "zoning":
        config["start_modified_zoning"] = int(item.get("start_year", 2030))
        config["rezoned_reefs"] = int(item.get("reefs", DEFAULT_REZONED_REEFS))
        return
    if itype == "size_limits":
        if "upper_start_year" in item:
            config["start_upper_sizelimit"] = int(item["upper_start_year"])
        if "lower_start_year" in item:
            config["start_lower_sizelimit"] = int(item["lower_start_year"])
        if "start_year" in item and "upper_start_year" not in item:
            config["start_upper_sizelimit"] = int(item["start_year"])
        return
    if itype == "cots_fishing_limit":
        config["start_CoTSlimit"] = int(item.get("start_year", 2030))
        return
    if itype == "emperor_release":
        config["start_emperor_release"] = int(item.get("start_year", 2030))
        config["release_reefs"] = int(item.get("reefs", config.get("release_reefs", 500)))
        config["release_number"] = float(item.get("number", DEFAULT_RELEASE_NUMBER))
        if "threshold" in item:
            config["release_threshold"] = float(item["threshold"])
        return
    if itype == "regional_shading":
        config["start_regional_shading"] = int(item.get("start_year", 2030))
        config["regional_shading_reduction"] = float(
            item.get("reduction", DEFAULT_REGIONAL_SHADING_REDUCTION)
        )
        return
    if itype == "reef_shading":
        config["start_reef_shading"] = int(item.get("start_year", 2030))
        config["shading_reefs"] = int(item.get("reefs", DEFAULT_SHADING_REEFS))
        config["reef_shading_reduction"] = float(
            item.get("reduction", DEFAULT_REEF_SHADING_REDUCTION)
        )
        return
    if itype == "ph_protection":
        config["start_pH_protection"] = int(item.get("start_year", 2030))
        config["pH_reefs"] = int(item.get("reefs", DEFAULT_PH_REEFS))
        config["pH_protection"] = float(item.get("protection", DEFAULT_PH_PROTECTION))
        return
    if itype == "intervention_box":
        mapping = {
            "lon_min": "intervene_lon_min",
            "lon_max": "intervene_lon_max",
            "lat_min": "intervene_lat_min",
            "lat_max": "intervene_lat_max",
        }
        for src, dest in mapping.items():
            if src in item:
                config[dest] = float(item[src])
            elif dest in item:
                config[dest] = float(item[dest])


def _maybe_extend_end_year(config: dict[str, Any], notes: list[str], end_year_set: bool) -> None:
    if end_year_set:
        return
    active_starts = [
        int(config[name])
        for name in INTERVENTION_START_FIELDS
        if int(config.get(name, INACTIVE_START_YEAR)) < INACTIVE_START_YEAR
    ]
    if not active_starts:
        return
    latest = max(active_starts)
    if int(config["end_year"]) <= latest:
        config["end_year"] = latest + 20
        notes.append(
            f"end_year extended to {config['end_year']} so interventions that start "
            f"in {latest} have a 20-year projection window."
        )


def build_scenario(
    name: str | None = None,
    description: str = "",
    climate: Any | None = None,
    ssp: Any | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    projection_year: int | None = None,
    ensemble_runs: int | None = None,
    region: str | None = None,
    interventions: list[dict[str, Any]] | None = None,
    config_overrides: dict[str, Any] | None = None,
    apply_region_bounds: bool = True,
) -> ScenarioSpec:
    """Translate a structured manager request into a ``CoconetConfig`` field dict.

    ``config_overrides`` must use real ``CoconetConfig`` field names. Unknown keys
    are rejected. Intervention types are the catalog in ``options.INTERVENTION_TYPES``.
    """

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    ## ~ Start from model defaults ~ ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    config = default_config_values()
    notes: list[str] = []
    intervention_list = list(interventions or [])
    end_year_set = end_year is not None

    if climate is not None and ssp is not None and resolve_ssp(climate) != resolve_ssp(ssp):
        raise ScenarioValidationError(
            [f"climate {climate!r} and ssp {ssp!r} do not resolve to the same SSP"]
        )
    if climate is not None or ssp is not None:
        config["SSP"] = resolve_ssp(climate if climate is not None else ssp)

    if start_year is not None:
        config["start_year"] = int(start_year)
    if end_year is not None:
        config["end_year"] = int(end_year)
    if projection_year is not None:
        config["projection_year"] = int(projection_year)
    if ensemble_runs is not None:
        config["ensemble_runs"] = int(ensemble_runs)
    else:
        config["ensemble_runs"] = DEFAULT_ENSEMBLE_RUNS

    region_code: str | None = None
    if region:
        region_code = resolve_region_code(region)
        if region_code is None and region.strip().lower() not in {"sector", "active sector"}:
            raise ScenarioValidationError(
                [f"Unknown region {region!r}. Use GBR, FN, N, C, S, or sector."]
            )
        if apply_region_bounds and region_code in REGION_FILTER_CODES:
            # Use the reefs' own Region codes (as CoTS control does) rather than a
            # lat/lon box, so every lever in a package targets the same reefs.
            config["intervene_region"] = region_code
            notes.append(
                f"Seeding, shading, and other reef interventions are limited to reefs in "
                f"the {region_code} region (the model's own Region codes, as for CoTS control)."
            )

    for item in intervention_list:
        if region and "region" not in item and str(item.get("type", "")).lower() == "cots_control":
            item = {**item, "region": region}
        _apply_intervention(config, item, notes)

    if config_overrides:
        known = config_field_names()
        unknown = sorted(k for k in config_overrides if k not in known)
        if unknown:
            raise ScenarioValidationError([f"Unknown CoconetConfig field(s): {', '.join(unknown)}"])
        for key, value in config_overrides.items():
            config[key] = _coerce_config_value(key, value)

    _maybe_extend_end_year(config, notes, end_year_set)

    if climate is not None or ssp is not None:
        notes.append(f"Climate driver SSP={config['SSP']}.")

    display_name = name or _default_name(config)
    spec = ScenarioSpec(
        scenario_id=_new_scenario_id(config, display_name),
        name=display_name,
        description=description,
        config=config,
        interventions=intervention_list,
        notes=notes,
    )
    # Always validate what we just built so callers can show errors immediately.
    validate_scenario(spec.config)
    return spec


def _default_name(config: dict[str, Any]) -> str:
    parts = [f"SSP{config['SSP']}"]
    if int(config.get("start_CoTS_control", INACTIVE_START_YEAR)) < INACTIVE_START_YEAR:
        parts.append("CoTS")
    if int(config.get("start_coral_seeding", INACTIVE_START_YEAR)) < INACTIVE_START_YEAR:
        parts.append("TT-seed")
    if int(config.get("start_rubble_consolidation", INACTIVE_START_YEAR)) < INACTIVE_START_YEAR:
        parts.append("rubble")
    parts.append(f"{config['start_year']}-{config['end_year']}")
    parts.append(f"ens{config['ensemble_runs']}")
    return " ".join(parts)


def _is_active_start(year: Any) -> bool:
    try:
        return int(year) < INACTIVE_START_YEAR
    except (TypeError, ValueError):
        return False


def validate_scenario(config: dict[str, Any] | ScenarioSpec) -> dict[str, Any]:
    """Check a proposed config against the real schema and intervention consistency.

    Returns the coerced config on success. Raises ``ScenarioValidationError``
    with every problem found (does not stop at the first error).
    """

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    ## ~ Schema and consistency checks ~ ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    raw = dict(config.config) if isinstance(config, ScenarioSpec) else dict(config)

    errors: list[str] = []
    known = config_field_names()
    unknown = sorted(k for k in raw if k not in known)
    if unknown:
        errors.append(f"Unknown CoconetConfig field(s): {', '.join(unknown)}")

    coerced: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            continue
        try:
            coerced[key] = _coerce_config_value(key, value)
        except ScenarioValidationError as exc:
            errors.extend(exc.errors)

    # Fill missing keys from defaults so we can reason about the full record.
    defaults = default_config_values()
    for key in known:
        if key not in coerced:
            coerced[key] = defaults[key]

    ssp = coerced.get("SSP")
    if ssp not in SUPPORTED_SSP:
        errors.append(
            f"SSP must be one of {list(SUPPORTED_SSP)} (model uses exact equality). Got {ssp!r}."
        )

    runs = coerced.get("ensemble_runs")
    if not isinstance(runs, int) or runs < MIN_ENSEMBLE_RUNS:
        errors.append(
            f"ensemble_runs must be an integer >= {MIN_ENSEMBLE_RUNS} "
            f"(never a single stochastic replicate). Got {runs!r}."
        )
    elif runs > MAX_ENSEMBLE_RUNS:
        errors.append(f"ensemble_runs={runs} exceeds the safety cap of {MAX_ENSEMBLE_RUNS}.")

    start = coerced.get("start_year")
    end = coerced.get("end_year")
    save = coerced.get("save_year")
    proj = coerced.get("projection_year")
    if isinstance(start, int) and isinstance(end, int) and start > end:
        errors.append(f"start_year ({start}) is after end_year ({end}).")
    if isinstance(save, int) and isinstance(start, int) and save < start:
        errors.append(f"save_year ({save}) is before start_year ({start}).")
    if isinstance(proj, int) and isinstance(end, int) and proj > end:
        errors.append(
            f"projection_year ({proj}) is after end_year ({end}); "
            "scenario drivers would never apply."
        )

    lon_min = coerced.get("intervene_lon_min")
    lon_max = coerced.get("intervene_lon_max")
    lat_min = coerced.get("intervene_lat_min")
    lat_max = coerced.get("intervene_lat_max")
    if (
        isinstance(lon_min, (int, float))
        and isinstance(lon_max, (int, float))
        and lon_min >= lon_max
    ):
        errors.append("intervene_lon_min must be < intervene_lon_max.")
    if (
        isinstance(lat_min, (int, float))
        and isinstance(lat_max, (int, float))
        and lat_min >= lat_max
    ):
        errors.append("intervene_lat_min must be < intervene_lat_max.")

    for name in FRACTION_FIELDS:
        value = coerced.get(name)
        if isinstance(value, (int, float)) and not 0.0 <= float(value) <= 1.0:
            errors.append(
                f"{name} is a fraction between 0 and 1 (for example 0.3 for 30%). Got {value!r}."
            )

    region_filter = coerced.get("intervene_region")
    if region_filter not in ("GBR", *REGION_FILTER_CODES):
        errors.append(f"intervene_region must be GBR, FN, N, C, or S. Got {region_filter!r}.")

    if _is_active_start(coerced.get("start_CoTS_control")):
        vessels = [int(coerced.get(name, 0) or 0) for name in VESSEL_FIELDS]
        if not any(v > 0 for v in vessels):
            errors.append(
                "start_CoTS_control is set but every CoTS_vessels_* count is 0. "
                "Set vessels for GBR, FN, N, C, S, or sector."
            )
        if isinstance(end, int) and int(coerced["start_CoTS_control"]) > end:
            errors.append("start_CoTS_control is after end_year; control would never run.")

    if _is_active_start(coerced.get("start_coral_seeding")):
        if int(coerced.get("seed_reefs", 0) or 0) <= 0:
            errors.append("start_coral_seeding is set but seed_reefs <= 0.")
        if float(coerced.get("seed_hectares", 0) or 0) <= 0:
            errors.append("start_coral_seeding is set but seed_hectares <= 0.")
        if float(coerced.get("seed_threshold", 0) or 0) <= 0:
            errors.append(
                "seed_threshold is 0: every reef with cover >= 0 is skipped, so seeding "
                "would treat no reefs. Use a value such as 0.3."
            )
        if isinstance(end, int) and int(coerced["start_coral_seeding"]) > end:
            errors.append("start_coral_seeding is after end_year; seeding would never run.")

    if _is_active_start(coerced.get("start_coral_slick")):
        if int(coerced.get("slick_reefs", 0) or 0) <= 0:
            errors.append("start_coral_slick is set but slick_reefs <= 0.")
        if float(coerced.get("slick_hectares", 0) or 0) <= 0:
            errors.append("start_coral_slick is set but slick_hectares <= 0.")

    if _is_active_start(coerced.get("start_rubble_consolidation")):
        if int(coerced.get("consolidation_reefs", 0) or 0) <= 0:
            errors.append("start_rubble_consolidation is set but consolidation_reefs <= 0.")
        if float(coerced.get("consolidation_hectares", 0) or 0) <= 0:
            errors.append("start_rubble_consolidation is set but consolidation_hectares <= 0.")

    if _is_active_start(coerced.get("start_catchment_restore")) and (
        float(coerced.get("restore_timeframe", 0) or 0) <= 0
    ):
        errors.append("start_catchment_restore is set but restore_timeframe must be > 0.")

    if _is_active_start(coerced.get("start_modified_fishing")):
        frac = float(coerced.get("catch_reduction", 0) or 0)
        if frac <= 0 or frac > 1:
            errors.append("catch_reduction must be in (0, 1] when fishing reduction is active.")

    if _is_active_start(coerced.get("start_reef_shading")):
        if int(coerced.get("shading_reefs", 0) or 0) <= 0:
            errors.append("start_reef_shading is set but shading_reefs <= 0.")
        red = float(coerced.get("reef_shading_reduction", 0) or 0)
        if red <= 0 or red > 1:
            errors.append("reef_shading_reduction must be in (0, 1] when local shading is active.")

    if _is_active_start(coerced.get("start_pH_protection")):
        if int(coerced.get("pH_reefs", 0) or 0) <= 0:
            errors.append("start_pH_protection is set but pH_reefs <= 0.")
        prot = float(coerced.get("pH_protection", 0) or 0)
        if prot <= 0 or prot > 1:
            errors.append("pH_protection must be in (0, 1] when pH treatment is active.")

    if errors:
        raise ScenarioValidationError(errors)

    return coerced


def baseline_config(config: dict[str, Any]) -> dict[str, Any]:
    """Same climate, years, and ensemble size, with every intervention turned off."""
    defaults = default_config_values()
    out = dict(config)
    for name in INTERVENTION_START_FIELDS:
        out[name] = defaults[name]
    for name in VESSEL_FIELDS:
        out[name] = defaults[name]
    for name in (
        "restore_timeframe",
        "rezoned_reefs",
        "catch_reduction",
        "release_number",
        "regional_shading_reduction",
        "consolidation_reefs",
        "consolidation_hectares",
        "seed_reefs",
        "seed_hectares",
        "hybrid_fraction",
        "dominance",
        "slick_reefs",
        "slick_hectares",
        "shading_reefs",
        "reef_shading_reduction",
        "pH_reefs",
        "pH_protection",
        "intervene_region",
    ):
        out[name] = defaults[name]
    return out


def targeted_regions(config: dict[str, Any]) -> list[str]:
    """Region codes a scenario's interventions are confined to (empty when GBR-wide)."""
    codes: list[str] = []
    for code in REGION_FILTER_CODES:
        if int(config.get(f"CoTS_vessels_{code}", 0) or 0) > 0:
            codes.append(code)
    region = config.get("intervene_region")
    if region in REGION_FILTER_CODES and region not in codes and config_has_interventions(config):
        codes.append(str(region))
    return codes


def active_start_years(config: dict[str, Any]) -> list[int]:
    """Start years of the interventions a config switches on."""
    return [
        int(config[name])
        for name in INTERVENTION_START_FIELDS
        if _is_active_start(config.get(name, INACTIVE_START_YEAR))
    ]


def spinup_shared_with_baseline(config: dict[str, Any]) -> bool:
    """True when no intervention starts before ``start_year``.

    The ensemble-0 spinup runs up to ``start_year``, so such a scenario and its
    no-intervention baseline have identical spinups and can share one.
    """
    start = int(config["start_year"])
    return all(
        int(config.get(name, INACTIVE_START_YEAR)) > start for name in INTERVENTION_START_FIELDS
    )


def config_has_interventions(config: dict[str, Any]) -> bool:
    if any(_is_active_start(config.get(name)) for name in INTERVENTION_START_FIELDS):
        return True
    return any(int(config.get(name, 0) or 0) > 0 for name in VESSEL_FIELDS)


def display_settings(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Human-readable rows for the confirmation card (only values that differ from defaults)."""
    defaults = default_config_values()
    rows: list[dict[str, Any]] = []
    always = ("SSP", "ensemble_runs", "start_year", "end_year", "projection_year")
    for name in always:
        if name in config:
            rows.append(
                {
                    "field": name,
                    "label": FIELD_LABELS.get(name, name),
                    "value": config[name],
                    "default": defaults.get(name),
                }
            )
    for name in sorted(config):
        if name in always or name in NON_SCENARIO_FIELDS:
            continue
        if name not in defaults:
            continue
        if config[name] != defaults[name]:
            rows.append(
                {
                    "field": name,
                    "label": FIELD_LABELS.get(name, name),
                    "value": config[name],
                    "default": defaults[name],
                }
            )
    return rows


def spec_from_config(
    config: dict[str, Any],
    *,
    name: str | None = None,
    description: str = "",
    notes: list[str] | None = None,
) -> ScenarioSpec:
    coerced = validate_scenario(config)
    display_name = name or _default_name(coerced)
    return ScenarioSpec(
        scenario_id=_new_scenario_id(coerced, display_name),
        name=display_name,
        description=description,
        config=coerced,
        interventions=[],
        notes=list(notes or []),
    )


def config_as_coconet(config: dict[str, Any]) -> CoconetConfig:
    """Build a ``CoconetConfig`` from a validated field dict."""
    known = config_field_names()
    values = {f.name: config[f.name] for f in fields(CoconetConfig) if f.name in config}
    extra = sorted(k for k in config if k not in known)
    if extra:
        raise ScenarioValidationError([f"Unknown CoconetConfig field(s): {', '.join(extra)}"])
    return CoconetConfig(**values)


def scenario_fingerprint(config: dict[str, Any]) -> str:
    identity = _identity_fields(validate_scenario(config))
    payload = json.dumps(identity, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def spec_to_jsonable(spec: ScenarioSpec) -> dict[str, Any]:
    return spec.to_dict()


def coconet_config_to_dict(cfg: CoconetConfig) -> dict[str, Any]:
    return asdict(cfg)
