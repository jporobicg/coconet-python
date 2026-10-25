---
title: Scenario parameter semantics
description: Legacy CSV labels, CoconetConfig fields, and the role of each scenario knob (from the repository README).
---

This page documents how **legacy parameter CSV** rows map to **`CoconetConfig`** and what each control does in the model. It mirrors the authoritative discussion in the repository README so users can read scenarios without leaving the docs site. You can supply these files from the **CLI** (`--parameter-file`) or when building config in Python via **`load_coconet_config`** / **`CoconetConfig.from_file`** (**library**); see [Getting started]({% link getting-started.md %}).

For load order (YAML, CSV, environment), see [Configuration]({% link configuration.md %}). For the Python `label_to_attr` map, see `coconet/config.py`.

## Legacy parameter CSV rules

Rows are `Label, value` pairs. Blank lines are skipped. Parsing is **case-insensitive** on the label. The first row whose first cell is exactly `Ensemble` (the output table header) ends the config block; everything after that is treated as data rows, not parameters.

If a label is **missing** from a template file, the value stays at the Python default in `CoconetConfig`.

**YAML / environment-only** (not read from the legacy CSV): `reefs_file`, `coastline_file`, `output_file`, `ensemble_threads`, `log_level`, `search_mode`, `perfect_intervention`, `unregulated_fishing`. Override these via `--config`, `COCONET_*` env vars, or CLI flags (`--reefs-file`, `--coastline-file`, `--output-file`, …) where supported.

## Simulation schedule

| Legacy CSV label | `CoconetConfig` field | Role |
| --- | --- | --- |
| Climate scenario | `SSP` | Shared socioeconomic pathway used for post-`projection_year` bleaching severity, coral growth `pH` stress (`sqrt(SSP)`), and related scenario branches. Supported values in code: `1.9`, `2.6`, `4.5`, `7.0`, `8.5`. |
| Ensemble runs | `ensemble_runs` | Number of stochastic ensemble replicates (`1..ensemble_runs`); ensemble `0` is spin-up. |
| Start year | `start_year` | Calendar year when saved initial conditions are loaded for ensembles `>= 1`. |
| Spinup backtrack (years) | `spinup_backtrack_years` | For ensemble `0` only, simulation begins at `start_year - spinup_backtrack_years` to warm up the system. |
| Save year | `save_year` | Initial state for ensembles `>= 1` is taken from the end of the spin-up trajectory in the year before `save_year` (see model loop). |
| Projection year | `projection_year` | Before this year, historical cyclone/bleaching drivers apply where coded; from this year onward, scenario-based projection drivers apply. |
| End year | `end_year` | Last calendar year in the main annual loop. |
| Search year | `search_year` | When `search_mode == 1`, used to switch search behaviour and benefit accumulation (optimization-style mode; default `search_mode` is `0`). |

## Crown-of-thorns (CoTS) control

All CoTS control logic runs only when `year >= start_CoTS_control`. Vessel counts set how many control “vessels” are simulated for that annual step; **each non-zero regional/GBR/sector option runs its own control pass** in sequence (separate dive budgets). Intervention reefs are still filtered by region for the regional modes.

| Legacy CSV label | `CoconetConfig` field | Role |
| --- | --- | --- |
| CoTS control start year | `start_CoTS_control` | First year any CoTS control runs. |
| CoTS control ecological threshold (CoTS per ha) | `eco_threshold` | Site-level CoTS density above which culling dives target that site; also scales post-cull age-class distribution. |
| CoTS control CoTS threshold (CoTS per ha) | `CoTS_threshold` | Reef-level mean CoTS (`S_manta_r`) must be **below** this (or coral above `coral_threshold`) for the reef to be eligible for control. |
| CoTS control coral threshold (CoTS per ha) | `coral_threshold` | If reef mean coral cover exceeds this, the reef is eligible even when CoTS density is high (legacy naming says “CoTS per ha” but the field is compared to coral cover). |
| CoTS vessels across GBR | `CoTS_vessels_GBR` | Sets vessel count and `control_region = "GBR"` for `control_cots()`. |
| CoTS vessels in Far-northern Region | `CoTS_vessels_FN` | Same for region `FN`. |
| CoTS vessels in Northern Region | `CoTS_vessels_N` | Same for region `N`. |
| CoTS vessels in Central Region | `CoTS_vessels_C` | Same for region `C`. |
| CoTS vessels in Southern Region | `CoTS_vessels_S` | Same for region `S`. |
| CoTS vessels in active sector | `CoTS_vessels_sector` | Vessel count for `control_cots_by_sector()`: targets the sector (1–11) with highest mean CoTS; uses a slightly different dive-cost formula than regional control. |

## Catchment, zoning, and fishing

| Legacy CSV label | `CoconetConfig` field | Role |
| --- | --- | --- |
| Catchment restoration start year | `start_catchment_restore` | First year catchment recovery can proceed. |
| Catchment restoration timescale (years) | `restore_timeframe` | Each year, `catchment_condition` moves toward `1` by `1/restore_timeframe` (exponential approach). Must be `> 0` for the update to run. Used in cyclone/flood loading (`flood_load`). |
| Future zoning start year | `start_modified_zoning` | Year assigned as `future_rezone_year` for the first `rezoned_reefs` reefs in priority order (among reefs that still have `rezone_year == 9999`). |
| Number of reefs included in future rezoning | `rezoned_reefs` | Count of reefs that receive a planned future closure year. |
| Reduction in fisheries catch start year | `start_modified_fishing` | From this year on, global annual emperor and coral-trout catch caps are multiplied by `1 - catch_reduction`. |
| Fractional reduction in fisheries catches | `catch_reduction` | Fractional cut applied to `e_annual` and `g_annual` (see `apply_fishing`). |
| Upper fish size limit start year | `start_upper_sizelimit` | From this year, fishing removes no `E_5` / `G_5` (largest age classes). |
| Lower fish size limit start year | `start_lower_sizelimit` | From this year, fishing removes no `E_3` / `G_3` (smallest targeted ages). |
| Exclude fishing from active outbreak reefs start year | `start_CoTSlimit` | From this year, if a weighted CoTS outbreak index on the reef exceeds `68`, all targeted emperor/trout catch on that visit is set to zero. |

## Other species and shading

| Legacy CSV label | `CoconetConfig` field | Role |
| --- | --- | --- |
| Emperor release start year | `start_emperor_release` | First year `release_fish()` runs. |
| Number of release reefs | `release_reefs` | Maximum reefs to stock per year (priority order, within intervention lat/lon box). |
| Maximum adult emperors (per ha) for release | `release_threshold` | Reefs at or above this total adult emperor biomass are **skipped** (release targets depleted reefs). |
| Number of juvenile emperors released per reef | `release_number` | Juveniles added to `E_1`, spread per reef area (ha). |
| Regional shading start year | `start_regional_shading` | First year `shade_regional_coral()` runs. |
| Absolute DHW reduction due to regional shading (DHW) | `regional_shading_reduction` | For reefs inside the intervention bounding box, bleaching DHW is multiplied by `(1 - regional_shading[reef])` after this value is assigned (same multiplicative pattern as local shading; legacy label says “absolute” but the implementation scales DHW). |

## Intervention bounding box

Applied to most spatial interventions (not CoTS regional control, which uses reef region codes). Reef centre must satisfy `intervene_lon_min < x < intervene_lon_max` and `intervene_lat_min < y < intervene_lat_max`.

| Legacy CSV label | `CoconetConfig` field |
| --- | --- |
| Minimum longitude of interventions | `intervene_lon_min` |
| Maximum longitude of interventions | `intervene_lon_max` |
| Minimum latitude of interventions | `intervene_lat_min` |
| Maximum latitude of interventions | `intervene_lat_max` |
| Region of interventions (Python only; written to the preamble only when set) | `intervene_region` |

`intervene_region` adds a Region-code test (`FN`, `N`, `C`, `S`) on top of the box. Lat/lon
boxes overlap the GBRMPA regions (for example, a box around the Northern region also
holds many Far-Northern reefs), so use the Region code when an intervention should
target one region exactly. The default `"GBR"` disables the test.

## Rubble, coral seeding, slicks, local shading, ocean acidification

| Legacy CSV label | `CoconetConfig` field | Role |
| --- | --- | --- |
| Rubble consolidation start year | `start_rubble_consolidation` | Enables `consolidate_rubble()`. |
| Annual number of consolidated reefs | `consolidation_reefs` | Cap on reefs treated per year. |
| Minimum rubble cover threshold for consolidation [0 1] | `consolidation_threshold` | Reef mean rubble must be **strictly greater** than this to qualify. |
| Total annual consolidated area (ha) | `consolidation_hectares` | Divided by `ha_per_site` to give per-site rubble removal at a chosen site. |
| Thermally tolerant coral seeding start year | `start_coral_seeding` | Enables `seed_tt_coral()`. |
| Annual number of reefs seeded with coral | `seed_reefs` | Max reefs per year. |
| Maximum coral cover threshold for coral seeding [0 1] | `seed_threshold` | Reefs at or above this mean coral cover are **skipped**. |
| Total annual area of seeded corals (ha) | `seed_hectares` | Determines fractional cover added per seeding event. |
| Fraction of staghorn … hybridise … [0 1] | `hybrid_fraction` | In `spawn_corals()`, scales hybridisation of branching (`sa`) with thermally tolerant (`tt`) corals for recruitment composition and thermal traits. |
| Dominance of thermally tolerant corals … [0 1] | `dominance` | Weights how hybrid thermal tolerance blends toward `tt` vs `sa` in the same hybrid calculation. |
| Coral slicks start year | `start_coral_slick` | Enables `seed_coral_slick()`. |
| Annual number of reefs with coral slicks released | `slick_reefs` | Max reefs per year. |
| Maximum coral cover threshold for coral slicks [0 1] | `slick_threshold` | Reefs with thermally tolerant reef-mean cover `C_r["tt"]` **at or above** this are **skipped**. |
| Total annual area of slick corals (ha) | `slick_hectares` | Slick fraction per site; material is drawn from a random source site’s community composition. |
| Reef shading start year | `start_reef_shading` | Enables `shade_local_reef()`. |
| Annual number of reefs locally shaded | `shading_reefs` | Each chosen reef gets `reef_shading[reef] = reef_shading_reduction`. |
| Fractional DHW reduction due to local shading [0 1] | `reef_shading_reduction` | Bleaching DHW multiplied by `(1 - reef_shading[reef])`. |
| Ocean acidification treatment start year | `start_pH_protection` | Enables `increase_pH()`. |
| Annual number of reefs treated for ocean acidification | `pH_reefs` | Reefs per year receive `pH_protect[reef] = pH_protection`. |
| Fractional protection from ocean acidification [0 1] | `pH_protection` | From `projection_year` onward, coral growth uses `pH_effect_t = (1 - pH_protect[reef]) * sqrt(SSP)`. |

## Search mode and “perfect” interventions

- **`search_mode`** (`0` default): normal output; `1` enables search-year logic and benefit tracking (see `model.py`).
- **`perfect_intervention`**: string mode applied from year **2026** onward (`_apply_perfect_intervention`), e.g. idealised starfish control or shading (not set via legacy CSV).
- **`unregulated_fishing`**: if true, sets zoning years so fishing regulation drivers are bypassed.

For exact equations, see the corresponding methods on `CoconetModel` in `coconet/model.py` and the label map in `coconet/config.py`.

## Docker image (optional runs)

The README also documents running published **`ghcr.io`** images with bind mounts and optional `COCONET_*` environment variables. See the **Docker** section in the [repository README]({{ site.repo_web_url }}/blob/main/README.md).
