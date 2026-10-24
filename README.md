# CoCoNet Python Port

This repository contains a headless Python port of the legacy NetLogo CoCoNet model.

| Resource | Link |
| --- | --- |
| **Documentation** (installation, configuration, CLI, API, I/O) | [gbrrestoration.github.io/coconet-python](https://gbrrestoration.github.io/coconet-python/) |
| **PyPI package** (`pip install coconet-python`) | [pypi.org/project/coconet-python](https://pypi.org/project/coconet-python/) |
| **Source** | [github.com/gbrrestoration/coconet-python](https://github.com/gbrrestoration/coconet-python) |

The docs site is built with **Jekyll** from the [`documentation/`](https://github.com/gbrrestoration/coconet-python/tree/main/documentation) tree. Maintainers enable **Settings → Pages → Build and deployment: GitHub Actions**; pushes to **`main`** run **Deploy documentation to GitHub Pages** and update the URL above.

> Links to the docs below use the **published site** so they work from this README on **GitHub** and on **PyPI** (which does not resolve repository-relative paths).

## Two ways to run CoCoNet

The same simulation engine is exposed for **shell-oriented workflows** and for **programmatic control**. Pick the entry point that matches how you are integrating the model.

### 1. Command-line interface (CLI)

Use this when you want **files, flags, and environment variables**—for example local terminals, Docker (the published image uses the CLI as its entrypoint), shell scripts, or CI jobs that only need to pass paths.

- **Console script:** `coconet` (installed with the package).
- **Module form:** `python -m coconet` (same behaviour as `coconet`).
- **Typical inputs:** `--config` (YAML), `--parameter-file` (legacy CSV), `--reefs-file`, `--coastline-file`, `--output-file`, plus `COCONET_*` environment overrides. See **`coconet --help`** and the [CLI documentation](https://gbrrestoration.github.io/coconet-python/cli/).

The CLI handles **logging bootstrap**, optional **CPU profiling** (`--profile`, extra dependency), and applies a fixed set of **command-line overrides** on top of the shared configuration loader.

### 2. Python library (PyPI package **coconet-python**)

Use this when you need **full control from Python**—custom CLIs, web services, notebooks, schedulers, or multi-step pipelines. Install from **PyPI** (distribution name **`coconet-python`**; import package remains **`coconet`**):

```bash
pip install coconet-python
```

The stable library surface is **`load_coconet_config`** and **`run_coconet`** in `coconet.api` (re-exported from `coconet`). Build a `CoconetConfig` from YAML paths, legacy CSV, environment, an optional **`scenario`** dict, and keyword overrides, then run:

```python
from coconet import load_coconet_config, run_coconet

cfg = load_coconet_config(
    config_file="config/example.yaml",
    reefs_file="legacy/reefs2024.csv",
    coastline_file="legacy/coastline.csv",
    output_file="out/run.csv",
)
run_coconet(cfg, configure_logs=True)
```

For **advanced** embedding you can still construct **`CoconetModel`** directly from **`CoconetConfig`**. See the [Python API](https://gbrrestoration.github.io/coconet-python/python-api/) page for precedence, logging, and `__all__`.

## Manager assistant (optional)

A Streamlit chat (`app.py` + `coconet_assistant/`) lets reef managers ask
plain-language questions, confirm a scenario, and run an **ensemble** through
the same Python API. A local Ollama model (Llama or Qwen) stays on-site; every
number in the briefing comes from the model tools. This is an optional extra
and is not required to run CoCoNet.

```bash
uv sync --extra assistant
ollama pull qwen2.5
uv run --extra assistant streamlit run app.py
```

See [docs/manager-assistant.md](docs/manager-assistant.md) for install, example
questions, architecture, and limits.

## Why this exists

The `legacy/` directory contains the original monolithic NetLogo model (`CoCoNet V3_rubble.nlogo`) and its data files. This port migrates the model to a modern Python stack for:

- headless execution in cloud/CI environments,
- typed, maintainable code,
- easier configuration through files and environment variables,
- reproducible scenario runs without a GUI.

## Project layout

- `legacy/` - original NetLogo model and data files.
- `coconet/` - Python implementation:
  - `config.py` - typed configuration and legacy parameter CSV parsing,
  - `model.py` - simulation engine (ported procedures),
  - `api.py` - **library** entry points (`load_coconet_config`, `run_coconet`) for PyPI consumers and embedders,
  - `cli.py` - **CLI** entry point (`coconet` / `python -m coconet`),
  - `__main__.py` - delegates to the CLI for `python -m coconet`.
- `coconet_assistant/` - optional manager chat tools (scenario build/validate, ensembles, summaries).
- `app.py` - Streamlit UI for the assistant (`pip install 'coconet-python[assistant]'`).

## Running with uv (from a git checkout)

```bash
uv sync
uv run coconet --parameter-file legacy/I_2p6.csv --output-file output.csv
```

You can also use YAML config:

```bash
uv run coconet --config config/example.yaml --output-file output.csv
```

## Docker (GitHub Container Registry)

A container image is built with [GitHub Actions](https://github.com/gbrrestoration/coconet-python/blob/main/.github/workflows/docker-publish.yml) on pushes to the default branch (`main`) and on SemVer tags `v*`. It is published to **GitHub Container Registry** as [`ghcr.io/gbrrestoration/coconet-python`](https://github.com/gbrrestoration/coconet-python/pkgs/container/coconet-python) (pull: `docker pull ghcr.io/gbrrestoration/coconet-python:latest`).

The image is based on `python:3.12-slim-bookworm`, installs dependencies with **uv** (`uv sync --frozen`), bundles `legacy/` and `config/` under `/app`, and runs as UID **1000**. The container entrypoint is the **`coconet` CLI** (see [Two ways to run CoCoNet](#two-ways-to-run-coconet)). If you need the **library** interface instead, add `pip install coconet-python` (or copy the package) in your own image and call `load_coconet_config` / `run_coconet` from your code.

### Input and output paths

Point the CLI at inputs and outputs using flags (or YAML / environment variables). For data on the host, mount a host directory into the container and pass **container paths** to the CLI.

| Item | CLI flags | Environment (optional) |
| --- | --- | --- |
| Reef table CSV | `--reefs-file` | `COCONET_REEFS_FILE` |
| Coastline CSV | `--coastline-file` | `COCONET_COASTLINE_FILE` |
| Scenario YAML | `--config` | (YAML keys `reefs_file`, `coastline_file`, …) |
| Legacy parameter CSV | `--parameter-file` | (set via YAML `parameter_file` if needed) |
| Run output CSV | `--output-file` | `COCONET_OUTPUT_FILE` |

**Example: bundled inputs, outputs on the host**

```bash
mkdir -p ./out
docker run --rm \
  -v "$(pwd)/out:/out" \
  ghcr.io/gbrrestoration/coconet-python:latest \
  --config /app/config/example.yaml \
  --output-file /out/run.csv
```

**Example: custom inputs and outputs on the host**

```bash
mkdir -p ./out
docker run --rm \
  -v "/path/to/mydata:/data:ro" \
  -v "$(pwd)/out:/out" \
  ghcr.io/gbrrestoration/coconet-python:latest \
  --reefs-file /data/reefs2024.csv \
  --coastline-file /data/coastline.csv \
  --parameter-file /data/myparams.csv \
  --output-file /out/results.csv
```

Use `-e COCONET_ENSEMBLE_RUNS=2` (and other `COCONET_*` variables) if you prefer environment overrides; see below. Run `docker run --rm ghcr.io/gbrrestoration/coconet-python:latest --help` for all CLI options.

**Permissions:** the process in the image runs as UID **1000** (`coconet`). Writable bind mounts (for `--output-file`, profiling output, etc.) must allow that user to create files—for example `sudo chown 1000:1000 ./out` on the host directory, or a Docker volume instead of a root-owned host path. Overriding with `--user` is not recommended because `/app` in the image is not world-readable.

Environment overrides are supported with `COCONET_` prefix, for example:

```bash
export COCONET_ENSEMBLE_RUNS=2
export COCONET_END_YEAR=1990
uv run coconet --parameter-file legacy/I_2p6.csv
```

## Logging

The CLI includes lifecycle logging for setup, progress, and run completion.

You can control logging level with either CLI or environment variables:

```bash
uv run coconet --config config/example.yaml --log-level DEBUG
```

```bash
export COCONET_LOG_LEVEL=WARNING
uv run coconet --config config/example.yaml
```

Supported levels are: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`.

If both are provided, CLI takes precedence over the environment variable.

## Notes on parity

The implementation ports the NetLogo procedures and keeps important NetLogo semantics where feasible (seed resets, annual loop structure, reef/site-level state updates, interventions, and output schema). This is designed to support model-output comparison workflows between the legacy and Python implementations.

## Legacy parameter CSV (`--parameter-file`)

The documentation site ([**Scenario parameter semantics**](https://gbrrestoration.github.io/coconet-python/scenario-parameters/)) carries the same tables as the following sections for easier browsing in HTML.

Rows are `Label, value` pairs. Blank lines are skipped. Parsing is **case-insensitive** on the label. The first row whose first cell is exactly `Ensemble` (the output table header) ends the config block; everything after that is treated as data rows, not parameters.

Labels are mapped in `coconet/config.py` (`label_to_attr`). If a label from a template file is **missing**, the value stays at the Python default in `CoconetConfig` (for example `legacy/I_2p6.csv` has no `Spinup backtrack (years)` row, so the default `50` is used, and no `CoTS vessels in active sector` row, so default `0` is used).

**YAML / environment-only** (not read from the legacy CSV): `reefs_file`, `coastline_file`, `output_file`, `ensemble_threads`, `log_level`, `search_mode`, `perfect_intervention`, `unregulated_fishing`. Override these via `--config`, `COCONET_*` env vars, or CLI flags (`--reefs-file`, `--coastline-file`, `--output-file`, …) where supported.

### Simulation schedule

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

### Crown-of-thorns (CoTS) control

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

### Catchment, zoning, and fishing

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

### Other species and shading

| Legacy CSV label | `CoconetConfig` field | Role |
| --- | --- | --- |
| Emperor release start year | `start_emperor_release` | First year `release_fish()` runs. |
| Number of release reefs | `release_reefs` | Maximum reefs to stock per year (priority order, within intervention lat/lon box). |
| Maximum adult emperors (per ha) for release | `release_threshold` | Reefs at or above this total adult emperor biomass are **skipped** (release targets depleted reefs). |
| Number of juvenile emperors released per reef | `release_number` | Juveniles added to `E_1`, spread per reef area (ha). |
| Regional shading start year | `start_regional_shading` | First year `shade_regional_coral()` runs. |
| Absolute DHW reduction due to regional shading (DHW) | `regional_shading_reduction` | For reefs inside the intervention bounding box, bleaching DHW is multiplied by `(1 - regional_shading[reef])` after this value is assigned (same multiplicative pattern as local shading; legacy label says “absolute” but the implementation scales DHW). |

### Intervention bounding box

Applied to most spatial interventions (not CoTS regional control, which uses reef region codes). Reef centre must satisfy `intervene_lon_min < x < intervene_lon_max` and `intervene_lat_min < y < intervene_lat_max`.

| Legacy CSV label | `CoconetConfig` field |
| --- | --- |
| Minimum longitude of interventions | `intervene_lon_min` |
| Maximum longitude of interventions | `intervene_lon_max` |
| Minimum latitude of interventions | `intervene_lat_min` |
| Maximum latitude of interventions | `intervene_lat_max` |

### Rubble, coral seeding, slicks, local shading, ocean acidification

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

### Search mode and “perfect” interventions

- **`search_mode`** (`0` default): normal output; `1` enables search-year logic and benefit tracking (see `model.py`).
- **`perfect_intervention`**: string mode applied from year **2026** onward (`_apply_perfect_intervention`), e.g. idealised starfish control or shading (not set via legacy CSV).
- **`unregulated_fishing`**: if true, sets zoning years so fishing regulation drivers are bypassed.

For the exact equations, see the corresponding methods on `CoconetModel` in `coconet/model.py` and the label map in `coconet/config.py`.
