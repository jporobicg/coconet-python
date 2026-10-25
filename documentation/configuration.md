---
title: Configuration
description: Reference for CoconetConfig, YAML keys, legacy CSV labels, and environment variables.
---

Configuration is represented by the **`CoconetConfig`** dataclass in `coconet/config.py`.

### Merge order (shared loader)

These steps apply to **`CoconetConfig.from_file`** and to the high-level library helper **`load_coconet_config`** in `coconet/api.py` (**library entry point**). The **`load_coconet_config`** function adds optional **`scenario`** (a flat mapping of field names) and keyword **overrides** *after* environment variables; see [Python API]({% link python-api.md %}).

1. **Defaults** on the dataclass.
2. **YAML** file when provided (CLI **`--config`**, or `from_file` / `load_coconet_config` **`config_file`**): any key matching a field name updates that field.
3. **Legacy parameter CSV** when provided (CLI **`--parameter-file`**, or `parameter_file=`): rows are `Label, value`; labels map to fields (case-insensitive). The first row whose first cell is exactly `Ensemble` ends the parameter block (starts the output table in legacy templates).
4. **Environment variables** with prefix **`COCONET_`** (default): suffix is matched case-insensitively to field names; values are parsed as booleans, integers, floats, or strings.
5. **`load_coconet_config` only:** optional **`scenario`** dict, then keyword **`overrides`** (`None` skipped). Unknown keys raise **`TypeError`**.

### CLI vs library after the merge

- **CLI** ([Command-line interface]({% link cli.md %})): after the shared merge (steps 1–4 via **`load_coconet_config`**), the CLI applies **flag overrides** for output path, reef/coastline paths, **`log_level`**, and **`ensemble_threads`** when those flags are present.
- **Library**: pass paths and knobs as **`load_coconet_config`** arguments or mutate **`CoconetConfig`** before **`run_coconet`** / **`CoconetModel.run()`**.

## Fields (full reference)

### Paths, logging, execution

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `reefs_file` | str | `legacy/reefs2024.csv` | Reef attribute / spatial CSV. Not loaded from legacy parameter CSV; use YAML, env, or CLI `--reefs-file`. |
| `coastline_file` | str | `legacy/coastline.csv` | Two-column LON, LAT. Same as above for overrides. |
| `output_file` | str | `output.csv` | Main results CSV. |
| `parameter_file` | str \| None | None | Set internally when loading a legacy CSV. |
| `log_level` | str | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`. |
| `ensemble_threads` | int | `0` | After ensemble-0 spin-up: `0` = auto worker count, `1` = serial ensembles, `N` caps process pool. See [Model overview]({% link model-overview.md %}). |

### Simulation schedule

| Field | Type | Default |
| --- | --- | --- |
| `SSP` | float | `2.6` |
| `ensemble_runs` | int | `20` |
| `start_year` | int | `1956` |
| `spinup_backtrack_years` | int | `50` |
| `save_year` | int | `1986` |
| `projection_year` | int | `2025` |
| `search_year` | int | `9999` |
| `end_year` | int | `2030` |

### Catchment and zoning / fishing

| Field | Type | Default |
| --- | --- | --- |
| `start_catchment_restore` | int | `9999` |
| `restore_timeframe` | float | `0.0` |
| `start_modified_zoning` | int | `9999` |
| `rezoned_reefs` | int | `0` |
| `start_modified_fishing` | int | `9999` |
| `catch_reduction` | float | `0.0` |
| `start_lower_sizelimit` | int | `9999` |
| `start_upper_sizelimit` | int | `9999` |
| `start_CoTSlimit` | int | `9999` |

### CoTS control

| Field | Type | Default |
| --- | --- | --- |
| `start_CoTS_control` | int | `9999` |
| `eco_threshold` | float | `8.0` |
| `CoTS_threshold` | float | `999999.0` |
| `coral_threshold` | float | `0.0` |
| `CoTS_vessels_GBR` | int | `0` |
| `CoTS_vessels_FN` | int | `0` |
| `CoTS_vessels_N` | int | `0` |
| `CoTS_vessels_C` | int | `0` |
| `CoTS_vessels_S` | int | `0` |
| `CoTS_vessels_sector` | int | `0` |

### Intervention bounding box

Applied to most spatial interventions (see README / model for exceptions such as regional CoTS control).

| Field | Type | Default |
| --- | --- | --- |
| `intervene_lon_min` | float | `140.0` |
| `intervene_lon_max` | float | `155.0` |
| `intervene_lat_min` | float | `-25.0` |
| `intervene_lat_max` | float | `-9.0` |
| `intervene_region` | str | `"GBR"` |

`intervene_region` (Python-only) limits the same interventions to reefs whose own
Region code is `FN`, `N`, `C`, or `S`, as CoTS regional control does. The default
`"GBR"` applies no region filter, so output matches NetLogo.

### Stocking, shading, rubble, coral interventions

| Field | Type | Default |
| --- | --- | --- |
| `start_emperor_release` | int | `9999` |
| `release_reefs` | int | `500` |
| `release_threshold` | float | `9999999.0` |
| `release_number` | float | `0.0` |
| `start_regional_shading` | int | `9999` |
| `regional_shading_reduction` | float | `0.0` |
| `start_rubble_consolidation` | int | `9999` |
| `consolidation_reefs` | int | `0` |
| `consolidation_threshold` | float | `0.0` |
| `consolidation_hectares` | float | `0.0` |
| `start_coral_seeding` | int | `9999` |
| `seed_reefs` | int | `0` |
| `seed_threshold` | float | `0.0` |
| `seed_hectares` | float | `0.0` |
| `hybrid_fraction` | float | `0.0` |
| `dominance` | float | `0.0` |
| `start_coral_slick` | int | `9999` |
| `slick_reefs` | int | `0` |
| `slick_threshold` | float | `0.0` |
| `slick_hectares` | float | `0.0` |
| `start_reef_shading` | int | `9999` |
| `shading_reefs` | int | `0` |
| `reef_shading_reduction` | float | `0.0` |
| `start_pH_protection` | int | `9999` |
| `pH_reefs` | int | `0` |
| `pH_protection` | float | `0.0` |

### Legacy interface / special modes

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `search_mode` | int | `0` | `1` enables search-year behaviour and benefit outputs. YAML/env only (not legacy CSV). |
| `perfect_intervention` | str | `""` | Idealised intervention mode string from 2026 onward. YAML/env only. |
| `unregulated_fishing` | bool | `false` | Relaxes fishing regulation drivers when true. YAML/env only. |

## Legacy CSV label map

The authoritative mapping from human-readable labels to fields is `label_to_attr` in `coconet/config.py` (method `_apply_legacy_parameter_file`). The repository **README.md** contains detailed tables explaining the **role** of each scenario knob (CoTS, fishing, interventions). This documentation site defers those semantics to the README for a single detailed source, or to method docstrings in `coconet/model.py`.

If a label is **missing** from a template file, the Python **default** from `CoconetConfig` remains in effect.

For **what each scenario field does** in the model (CoTS, fishing, rubble, seeding, etc.), see [Scenario parameter semantics]({% link scenario-parameters.md %}).

## Helpers (library use)

- **`load_coconet_config`** / **`run_coconet`** (`coconet.api`) — preferred **library** entry points for building config and running the model (see [Python API]({% link python-api.md %})).
- **`effective_ensemble_workers(ensemble_threads, ensemble_runs)`** — cap on parallel worker processes.
- **`use_parallel_ensemble_run(ensemble_threads, ensemble_runs)`** — whether the model will use a process pool after spin-up.
