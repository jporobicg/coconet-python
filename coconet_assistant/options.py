"""Catalog of valid CoCoNet options, taken from ``CoconetConfig`` and the model.

Do not invent intervention or climate names here. Every field listed is a real
``CoconetConfig`` attribute. Climate values are the SSP numbers the bleaching
and pH branches actually test in ``coconet.model``.
"""

from __future__ import annotations

import re
from dataclasses import fields
from typing import Any

from coconet.config import CoconetConfig
from coconet.model import CORAL_GROUPS

# SSP numbers the projection bleaching / pH branches handle (see CoconetModel.bleaching).
SUPPORTED_SSP = (1.9, 2.6, 4.5, 7.0, 8.5)

# Manager phrases -> the numeric SSP the model uses (SSP2-4.5 is 4.5, not "2.4.5").
SSP_ALIASES: dict[str, float] = {
    "1.9": 1.9,
    "ssp1-1.9": 1.9,
    "ssp1.9": 1.9,
    "2.6": 2.6,
    "ssp1-2.6": 2.6,
    "ssp2.6": 2.6,
    "4.5": 4.5,
    "ssp2-4.5": 4.5,
    "ssp4.5": 4.5,
    "7.0": 7.0,
    "7": 7.0,
    "ssp3-7.0": 7.0,
    "ssp7.0": 7.0,
    "8.5": 8.5,
    "ssp5-8.5": 8.5,
    "ssp8.5": 8.5,
}

# Region codes that limit box-based interventions (CoconetConfig.intervene_region).
REGION_FILTER_CODES: tuple[str, ...] = ("FN", "N", "C", "S")

# Simulation timing fields the language model may not change (see ToolDispatcher).
SIMULATION_TIMING_FIELDS: tuple[str, ...] = (
    "start_year",
    "spinup_backtrack_years",
    "save_year",
    "projection_year",
)

# Reef Region column codes used by CoTS control (control_region in CoconetModel).
COTS_REGION_FIELDS: dict[str, str] = {
    "gbr": "CoTS_vessels_GBR",
    "fn": "CoTS_vessels_FN",
    "far-northern": "CoTS_vessels_FN",
    "far northern": "CoTS_vessels_FN",
    "far-northern region": "CoTS_vessels_FN",
    "n": "CoTS_vessels_N",
    "northern": "CoTS_vessels_N",
    "northern region": "CoTS_vessels_N",
    "northern sector": "CoTS_vessels_N",
    "c": "CoTS_vessels_C",
    "central": "CoTS_vessels_C",
    "central region": "CoTS_vessels_C",
    "s": "CoTS_vessels_S",
    "southern": "CoTS_vessels_S",
    "southern region": "CoTS_vessels_S",
    "sector": "CoTS_vessels_sector",
    "active sector": "CoTS_vessels_sector",
}

# Approximate lat/lon boxes so spatial interventions (seeding, shading, rubble)
# can follow a named region. CoTS regional control uses region codes, not this box.
# Bounds are a coarse GBRMPA-style split, not official management polygons.
REGION_BOUNDS: dict[str, dict[str, float]] = {
    "FN": {
        "intervene_lon_min": 142.0,
        "intervene_lon_max": 146.5,
        "intervene_lat_min": -13.5,
        "intervene_lat_max": -9.0,
    },
    "N": {
        "intervene_lon_min": 143.0,
        "intervene_lon_max": 147.5,
        "intervene_lat_min": -16.5,
        "intervene_lat_max": -13.0,
    },
    "C": {
        "intervene_lon_min": 145.0,
        "intervene_lon_max": 153.0,
        "intervene_lat_min": -20.0,
        "intervene_lat_max": -16.0,
    },
    "S": {
        "intervene_lon_min": 150.0,
        "intervene_lon_max": 154.5,
        "intervene_lat_min": -25.0,
        "intervene_lat_max": -20.0,
    },
    "GBR": {
        "intervene_lon_min": 140.0,
        "intervene_lon_max": 155.0,
        "intervene_lat_min": -25.0,
        "intervene_lat_max": -9.0,
    },
}

REGION_CODE_ALIASES: dict[str, str] = {
    "gbr": "GBR",
    "all": "GBR",
    "all reefs": "GBR",
    "all regions": "GBR",
    "everywhere": "GBR",
    "whole gbr": "GBR",
    "entire gbr": "GBR",
    "gbr-wide": "GBR",
    "whole reef": "GBR",
    "great barrier reef": "GBR",
    "the great barrier reef": "GBR",
    "fn": "FN",
    "far-northern": "FN",
    "far northern": "FN",
    "far-northern region": "FN",
    "n": "N",
    "northern": "N",
    "northern region": "N",
    "northern sector": "N",
    "c": "C",
    "central": "C",
    "central region": "C",
    "s": "S",
    "southern": "S",
    "southern region": "S",
}

# Coral group codes written to output.csv (C_sa ... C_tt).
CORAL_GROUP_LABELS: dict[str, str] = {
    "sa": "staghorn Acropora",
    "ta": "tabular Acropora",
    "mo": "Montipora",
    "po": "Poritidae (massive)",
    "fa": "faviid",
    "tt": "thermally tolerant",
}

# Output columns the assistant can summarise. Rubble (R_reef) is simulated
# but is not written to the standard output CSV.
OUTPUT_METRICS: dict[str, str] = {
    "coral_total": "Total coral cover (all six groups)",
    "C_sa": "Staghorn Acropora reef-mean cover",
    "C_ta": "Tabular Acropora reef-mean cover",
    "C_mo": "Montipora reef-mean cover",
    "C_po": "Poritidae reef-mean cover",
    "C_fa": "Faviid reef-mean cover",
    "C_tt": "Thermally tolerant coral cover",
    "S_manta": "CoTS density (manta-tow equivalent)",
    "emperor": "Emperor fish (all age classes)",
    "trout": "Coral trout (all age classes)",
    "Triggerfish": "Triggerfish density",
    "Benthic_invert": "Benthic invertebrate density",
    "DHW": "Degree heating weeks (bleaching driver)",
}

# Human labels matching the legacy parameter CSV / output preamble.
FIELD_LABELS: dict[str, str] = {
    "SSP": "Climate scenario (SSP number)",
    "ensemble_runs": "Ensemble runs (stochastic replicates 1..N; 0 is spin-up)",
    "start_year": "Start year",
    "spinup_backtrack_years": "Spinup backtrack (years)",
    "save_year": "Save year",
    "projection_year": "Projection year",
    "search_year": "Search year",
    "end_year": "End year",
    "start_CoTS_control": "CoTS control start year",
    "eco_threshold": "CoTS control ecological threshold (CoTS per ha)",
    "CoTS_threshold": "CoTS control CoTS threshold (CoTS per ha)",
    "coral_threshold": "CoTS control coral threshold (cover, despite legacy name)",
    "CoTS_vessels_GBR": "CoTS vessels across GBR",
    "CoTS_vessels_FN": "CoTS vessels in Far-northern Region",
    "CoTS_vessels_N": "CoTS vessels in Northern Region",
    "CoTS_vessels_C": "CoTS vessels in Central Region",
    "CoTS_vessels_S": "CoTS vessels in Southern Region",
    "CoTS_vessels_sector": "CoTS vessels in active sector",
    "start_catchment_restore": "Catchment restoration start year",
    "restore_timeframe": "Catchment restoration timescale (years)",
    "start_modified_zoning": "Future zoning start year",
    "rezoned_reefs": "Number of reefs included in future rezoning",
    "start_modified_fishing": "Reduction in fisheries catch start year",
    "catch_reduction": "Fractional reduction in fisheries catches",
    "start_upper_sizelimit": "Upper fish size limit start year",
    "start_lower_sizelimit": "Lower fish size limit start year",
    "start_CoTSlimit": "Exclude fishing from active outbreak reefs start year",
    "start_emperor_release": "Emperor release start year",
    "release_reefs": "Number of release reefs",
    "release_threshold": "Maximum adult emperors (per ha) for release",
    "release_number": "Number of juvenile emperors released per reef",
    "start_regional_shading": "Regional shading start year",
    "regional_shading_reduction": "DHW reduction due to regional shading (applied as a multiplier)",
    "intervene_lon_min": "Minimum longitude of interventions",
    "intervene_lon_max": "Maximum longitude of interventions",
    "intervene_lat_min": "Minimum latitude of interventions",
    "intervene_lat_max": "Maximum latitude of interventions",
    "intervene_region": "Region of seeding, shading, and other reef interventions",
    "start_rubble_consolidation": "Rubble consolidation start year",
    "consolidation_reefs": "Annual number of consolidated reefs",
    "consolidation_threshold": "Minimum rubble cover threshold for consolidation [0 1]",
    "consolidation_hectares": "Total annual consolidated area (ha)",
    "start_coral_seeding": "Thermally tolerant coral seeding start year",
    "seed_reefs": "Annual number of reefs seeded with coral",
    "seed_threshold": "Maximum coral cover threshold for coral seeding [0 1]",
    "seed_hectares": "Total annual area of seeded corals (ha)",
    "hybrid_fraction": "Fraction of staghorn Acropora able to hybridise with thermally tolerant corals [0 1]",
    "dominance": "Dominance of thermally tolerant corals in hybrid thermal tolerance [0 1]",
    "start_coral_slick": "Coral slicks start year",
    "slick_reefs": "Annual number of reefs with coral slicks released",
    "slick_threshold": "Maximum coral cover threshold for coral slicks [0 1]",
    "slick_hectares": "Total annual area of slick corals (ha)",
    "start_reef_shading": "Reef shading start year",
    "shading_reefs": "Annual number of reefs locally shaded",
    "reef_shading_reduction": "Fractional DHW reduction due to local shading [0 1]",
    "start_pH_protection": "Ocean acidification treatment start year",
    "pH_reefs": "Annual number of reefs treated for ocean acidification",
    "pH_protection": "Fractional protection from ocean acidification [0 1]",
    "reefs_file": "Reef attribute / spatial input CSV",
    "coastline_file": "Coastline input CSV",
    "output_file": "Run output CSV",
    "ensemble_threads": "Parallel ensemble worker cap (0 = auto, 1 = serial)",
    "log_level": "Logging level",
    "search_mode": "Search mode (0 = normal output)",
    "perfect_intervention": "Idealised intervention mode string (from 2026)",
    "unregulated_fishing": "Bypass fishing regulation drivers when true",
}

# Manager-facing intervention types. Each maps onto real CoconetConfig fields.
INTERVENTION_TYPES: dict[str, dict[str, Any]] = {
    "cots_control": {
        "description": "Crown-of-thorns culling. Uses region vessel counts, not the lat/lon box.",
        "fields": [
            "start_CoTS_control",
            "eco_threshold",
            "CoTS_threshold",
            "coral_threshold",
            "CoTS_vessels_GBR",
            "CoTS_vessels_FN",
            "CoTS_vessels_N",
            "CoTS_vessels_C",
            "CoTS_vessels_S",
            "CoTS_vessels_sector",
        ],
        "params": {
            "start_year": "start_CoTS_control",
            "vessels": "one of the CoTS_vessels_* fields chosen by region",
            "region": "GBR | FN | N | C | S | sector",
            "eco_threshold": "eco_threshold",
        },
    },
    "coral_seeding": {
        "description": "Thermally tolerant coral seeding (seed_tt_coral).",
        "fields": [
            "start_coral_seeding",
            "seed_reefs",
            "seed_threshold",
            "seed_hectares",
            "hybrid_fraction",
            "dominance",
        ],
        "params": {
            "start_year": "start_coral_seeding",
            "reefs": "seed_reefs",
            "hectares": "seed_hectares",
            "threshold": "seed_threshold (reefs at or above this cover are skipped)",
        },
    },
    "coral_slick": {
        "description": "Coral slicks drawn from a source site community (seed_coral_slick).",
        "fields": [
            "start_coral_slick",
            "slick_reefs",
            "slick_threshold",
            "slick_hectares",
        ],
        "params": {
            "start_year": "start_coral_slick",
            "reefs": "slick_reefs",
            "hectares": "slick_hectares",
            "threshold": "slick_threshold",
        },
    },
    "rubble_consolidation": {
        "description": "Rubble consolidation (consolidate_rubble). Rubble is simulated but not in output.csv.",
        "fields": [
            "start_rubble_consolidation",
            "consolidation_reefs",
            "consolidation_threshold",
            "consolidation_hectares",
        ],
        "params": {
            "start_year": "start_rubble_consolidation",
            "reefs": "consolidation_reefs",
            "hectares": "consolidation_hectares",
            "threshold": "consolidation_threshold",
        },
    },
    "catchment_restore": {
        "description": "Catchment restoration; requires restore_timeframe > 0.",
        "fields": ["start_catchment_restore", "restore_timeframe"],
        "params": {
            "start_year": "start_catchment_restore",
            "timescale_years": "restore_timeframe",
        },
    },
    "fishing_reduction": {
        "description": "Fractional cut to emperor and coral-trout catch caps.",
        "fields": ["start_modified_fishing", "catch_reduction"],
        "params": {
            "start_year": "start_modified_fishing",
            "fraction": "catch_reduction [0 1]",
        },
    },
    "zoning": {
        "description": "Future rezoning of the first N priority reefs still open.",
        "fields": ["start_modified_zoning", "rezoned_reefs"],
        "params": {"start_year": "start_modified_zoning", "reefs": "rezoned_reefs"},
    },
    "size_limits": {
        "description": "Upper and/or lower fish size limits (skip largest or smallest targeted ages).",
        "fields": ["start_upper_sizelimit", "start_lower_sizelimit"],
        "params": {
            "upper_start_year": "start_upper_sizelimit",
            "lower_start_year": "start_lower_sizelimit",
        },
    },
    "cots_fishing_limit": {
        "description": "Exclude fishing from active CoTS-outbreak reefs.",
        "fields": ["start_CoTSlimit"],
        "params": {"start_year": "start_CoTSlimit"},
    },
    "emperor_release": {
        "description": "Release juvenile emperors onto depleted reefs.",
        "fields": [
            "start_emperor_release",
            "release_reefs",
            "release_threshold",
            "release_number",
        ],
        "params": {
            "start_year": "start_emperor_release",
            "reefs": "release_reefs",
            "number": "release_number",
            "threshold": "release_threshold",
        },
    },
    "regional_shading": {
        "description": "Regional shading inside the intervention bounding box.",
        "fields": ["start_regional_shading", "regional_shading_reduction"],
        "params": {
            "start_year": "start_regional_shading",
            "reduction": "regional_shading_reduction",
        },
    },
    "reef_shading": {
        "description": "Local reef shading (fractional DHW reduction on chosen reefs).",
        "fields": ["start_reef_shading", "shading_reefs", "reef_shading_reduction"],
        "params": {
            "start_year": "start_reef_shading",
            "reefs": "shading_reefs",
            "reduction": "reef_shading_reduction",
        },
    },
    "ph_protection": {
        "description": "Ocean acidification treatment on chosen reefs.",
        "fields": ["start_pH_protection", "pH_reefs", "pH_protection"],
        "params": {
            "start_year": "start_pH_protection",
            "reefs": "pH_reefs",
            "protection": "pH_protection [0 1]",
        },
    },
    "intervention_box": {
        "description": "Lat/lon box applied to most spatial interventions (not CoTS regional control).",
        "fields": [
            "intervene_lon_min",
            "intervene_lon_max",
            "intervene_lat_min",
            "intervene_lat_max",
        ],
        "params": {
            "lon_min": "intervene_lon_min",
            "lon_max": "intervene_lon_max",
            "lat_min": "intervene_lat_min",
            "lat_max": "intervene_lat_max",
        },
    },
}

INTERVENTION_START_FIELDS: tuple[str, ...] = (
    "start_catchment_restore",
    "start_CoTS_control",
    "start_modified_zoning",
    "start_modified_fishing",
    "start_lower_sizelimit",
    "start_upper_sizelimit",
    "start_CoTSlimit",
    "start_emperor_release",
    "start_regional_shading",
    "start_rubble_consolidation",
    "start_coral_seeding",
    "start_coral_slick",
    "start_reef_shading",
    "start_pH_protection",
)

VESSEL_FIELDS: tuple[str, ...] = (
    "CoTS_vessels_GBR",
    "CoTS_vessels_FN",
    "CoTS_vessels_N",
    "CoTS_vessels_C",
    "CoTS_vessels_S",
    "CoTS_vessels_sector",
)


def config_field_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(CoconetConfig))


def config_field_types() -> dict[str, Any]:
    return {f.name: f.type for f in fields(CoconetConfig)}


def default_config_values() -> dict[str, Any]:
    cfg = CoconetConfig()
    return {name: getattr(cfg, name) for name in config_field_names()}


def resolve_ssp(value: Any) -> float:
    """Map a manager phrase or number onto a supported SSP."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        ssp = float(value)
        if ssp in SUPPORTED_SSP:
            return ssp
        raise ValueError(
            f"Unsupported climate scenario {value!r}. "
            f"The model implements SSP numbers {list(SUPPORTED_SSP)}."
        )
    compact = str(value).strip().lower().replace(" ", "").replace("_", "-")
    if compact in SSP_ALIASES:
        return SSP_ALIASES[compact]
    # Also try just the trailing number.
    for alias, ssp in SSP_ALIASES.items():
        if compact.endswith(alias) or compact == alias:
            return ssp
    raise ValueError(
        f"Unsupported climate scenario {value!r}. "
        f"Use one of {list(SUPPORTED_SSP)} or names such as SSP2-4.5."
    )


_EXPLICIT_SSP_PATTERNS = (
    re.compile(r"\bssp\s*\d\s*[-\u2013\u2014]\s*(\d\.\d)", re.IGNORECASE),
    re.compile(r"\bssp\s*(\d\.\d)\b", re.IGNORECASE),
)


def explicit_ssps(text: str) -> set[float]:
    """SSP values the manager named explicitly (``SSP1-2.6``, ``SSP 4.5``).

    Bare numbers are ignored so percentages and years never count as climate.
    """
    found: set[float] = set()
    for pattern in _EXPLICIT_SSP_PATTERNS:
        for match in pattern.finditer(text):
            value = float(match.group(1))
            if value in SUPPORTED_SSP:
                found.add(value)
    return found


def resolve_region_code(value: str) -> str | None:
    key = value.strip().lower()
    if key in REGION_CODE_ALIASES:
        return REGION_CODE_ALIASES[key]
    upper = value.strip().upper()
    if upper in REGION_BOUNDS:
        return upper
    return None


def normalise_summary_region(value: str | None) -> str | None:
    """Region code for filtering output rows; ``None`` means all reefs (GBR-wide)."""
    if value is None or not str(value).strip():
        return None
    code = resolve_region_code(str(value))
    if code is None:
        raise ValueError(f"Unknown region {value!r}. Use FN, N, C, S, or GBR for all reefs.")
    return None if code == "GBR" else code


def region_label(code: str | None) -> str:
    names = {"FN": "Far-Northern", "N": "Northern", "C": "Central", "S": "Southern"}
    return f"{names[code]} region ({code}) reefs only" if code in names else "all reefs (GBR-wide)"


def resolve_cots_vessel_field(region: str) -> str:
    key = region.strip().lower()
    if key in COTS_REGION_FIELDS:
        return COTS_REGION_FIELDS[key]
    code = resolve_region_code(region)
    if code is not None and code.lower() in COTS_REGION_FIELDS:
        return COTS_REGION_FIELDS[code.lower()]
    raise ValueError(
        f"Unknown CoTS region {region!r}. "
        f"Valid names: GBR, FN (far-northern), N (northern), C (central), "
        f"S (southern), or sector (active sector 1-11)."
    )


def _intervention_summary() -> dict[str, dict[str, Any]]:
    """Intervention catalogue without raw field lists (the LLM should use ``params``)."""
    return {
        name: {k: v for k, v in spec.items() if k != "fields"}
        for name, spec in INTERVENTION_TYPES.items()
    }


def list_options(detail: str = "summary") -> dict[str, Any]:
    """Tool payload: valid interventions, climate scenarios, parameters, ranges.

    ``detail="summary"`` (default) is the compact catalogue the language model needs
    to build a scenario. ``detail="fields"`` adds every ``CoconetConfig`` field with
    its default; it is several thousand tokens, so it is only sent on request.
    """
    if detail not in {"summary", "fields"}:
        raise ValueError("detail must be 'summary' or 'fields'.")
    summary: dict[str, Any] = {
        "climate_scenarios": {
            "field": "SSP",
            "allowed": list(SUPPORTED_SSP),
            "aliases": {
                "SSP1-1.9": 1.9,
                "SSP1-2.6": 2.6,
                "SSP2-4.5": 4.5,
                "SSP3-7.0": 7.0,
                "SSP5-8.5": 8.5,
            },
        },
        "regions": {
            "codes": ["GBR", "FN", "N", "C", "S"],
            "names": "FN far-northern, N northern, C central, S southern, GBR whole reef",
        },
        "intervention_types": _intervention_summary(),
        "simulation_years": (
            "Simulation start, save, and projection years are fixed by the app. "
            "Only set end_year and each intervention's own start_year."
        ),
        "ensemble_runs": {"min": 2, "default": 20, "typical": [20, 50]},
        "honest_limits": [
            "Rubble is simulated (R_site / R_reef) but is not written to output.csv.",
            "The model is a GBR-scale abstraction, not a reef-by-reef operational forecast.",
            "Ensemble bands are stochastic spread only; they omit structural and data error.",
            "Discrete SSP values only; there is no continuous climate pathway.",
        ],
    }
    if detail == "summary":
        summary["more"] = "Call list_options with detail='fields' for every raw config field."
        return summary

    defaults = default_config_values()
    parameters: list[dict[str, Any]] = []
    for name, typ in config_field_types().items():
        entry: dict[str, Any] = {
            "field": name,
            "label": FIELD_LABELS.get(name, name),
            "type": str(typ),
            "default": defaults[name],
        }
        if name == "SSP":
            entry["allowed"] = list(SUPPORTED_SSP)
            entry["notes"] = (
                "Model compares SSP with == 1.9, 2.6, 4.5, 7.0, 8.5. SSP2-4.5 means 4.5."
            )
        if name == "ensemble_runs":
            entry["range"] = {"min": 2, "typical": [20, 50], "default": 20}
            entry["notes"] = (
                "Stochastic replicates are ensembles 1..N. Ensemble 0 is spin-up. "
                "The assistant never runs a single stochastic replicate."
            )
        parameters.append(entry)

    return {
        "climate_scenarios": {
            "field": "SSP",
            "allowed": list(SUPPORTED_SSP),
            "aliases": {
                "SSP1-1.9": 1.9,
                "SSP1-2.6": 2.6,
                "SSP2-4.5": 4.5,
                "SSP3-7.0": 7.0,
                "SSP5-8.5": 8.5,
            },
        },
        "coral_groups": {
            code: CORAL_GROUP_LABELS[code] for code in CORAL_GROUPS if code in CORAL_GROUP_LABELS
        },
        "cots_regions": {
            "codes": ["GBR", "FN", "N", "C", "S", "sector"],
            "vessel_fields": list(VESSEL_FIELDS),
        },
        "intervention_types": INTERVENTION_TYPES,
        "intervention_bounding_box": {
            "fields": [
                "intervene_lon_min",
                "intervene_lon_max",
                "intervene_lat_min",
                "intervene_lat_max",
            ],
            "default": {k: defaults[k] for k in REGION_BOUNDS["GBR"]},
            "region_boxes": REGION_BOUNDS,
            "notes": (
                "Applied to seeding, slicks, shading, rubble, emperor release, pH. "
                "CoTS regional control uses reef Region codes instead."
            ),
        },
        "parameters": parameters,
        "output_metrics": OUTPUT_METRICS,
        "honest_limits": [
            "Rubble is simulated (R_site / R_reef) but is not written to output.csv.",
            "The model is a GBR-scale abstraction, not a reef-by-reef operational forecast.",
            "Ensemble bands are stochastic spread only; they omit structural and data error.",
            "Discrete SSP values only; there is no continuous climate pathway.",
        ],
    }
