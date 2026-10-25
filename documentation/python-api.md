---
title: Python API
description: The library entry point published as coconet-python on PyPI—load_coconet_config, run_coconet, and advanced use of CoconetModel.
---

This page is the **library entry point**: use it when you install **`coconet-python`** from **PyPI** and **`import coconet`** from your own Python code—services, notebooks, alternative CLIs, batch orchestration, or tests.

For **file- and flag-driven** runs from a shell or the **official Docker** image, use the **[CLI]({% link cli.md %})** (`coconet` / `python -m coconet`) instead.

## PyPI distribution vs import name

- **PyPI / pip:** distribution name **`coconet-python`** (`pip install coconet-python`; extras e.g. `pip install "coconet-python[profile]"`). The name **`coconet`** was already taken on PyPI.
- **Python import:** package **`coconet`** (lowercase), unchanged.

## Stable surface (`coconet.__all__`)

Re-exported from `coconet/__init__.py`:

| Symbol | Role |
| --- | --- |
| **`load_coconet_config`** | Merge defaults, optional YAML, optional legacy parameter CSV, `COCONET_*` env, optional **`scenario`** dict, and keyword **overrides** into a **`CoconetConfig`**. |
| **`run_coconet`** | Run **`CoconetModel(config)`** to completion; returns **`CoconetRunResult`** (e.g. **`output_file`**). |
| **`CoconetRunResult`** | Small result object from **`run_coconet`**; safe to extend in future releases. |
| **`CoconetConfig`** | Typed configuration dataclass; can still be built with **`from_file`** if you do not need **`scenario`** / **`load_coconet_config`** precedence. |
| **`CoconetModel`** | Simulation engine; **advanced** direct use (see below). |

Other modules (`cli`, `netlogo`, internal details on `model`) are available but **not** part of the stable **`__all__`**.

## Recommended: `load_coconet_config` and `run_coconet`

Precedence inside **`load_coconet_config`**:

1. **`CoconetConfig`** defaults  
2. Optional **`config_file`** (YAML keys matching dataclass fields)  
3. Optional **`parameter_file`** (legacy CSV)  
4. **`COCONET_*`** environment variables (`env_prefix` default `COCONET_`)  
5. Optional **`scenario`** mapping (flat dict of field names → values)  
6. Keyword **`overrides`**; **`None`** values are skipped  

Unknown keys in **`scenario`** or **`overrides`** raise **`TypeError`**.

```python
from pathlib import Path
from coconet import load_coconet_config, run_coconet

cfg = load_coconet_config(
    config_file=Path("config/example.yaml"),
    parameter_file=Path("legacy/I_2p6.csv"),  # or None
    scenario={"ensemble_runs": 3},
    output_file="runs/my_run.csv",
)
if __name__ == "__main__":  # required: parallel ensembles use the "spawn" start method
    result = run_coconet(cfg, configure_logs=True)
    print(result.output_file)
```

Parallel simulation ensembles start fresh worker processes (`spawn`), which re-import
your script. Without the `if __name__ == "__main__":` guard the script fails when the
workers start, which is only after the full ensemble-0 spinup.

### Reusing a spinup

`run_coconet` returns the state after the ensemble-0 spinup as
`result.spinup_checkpoint`. A second run whose settings are identical up to
`start_year` (same reefs, climate, and biology, with no intervention starting before
`start_year`) can start from it and skip its own spinup:

```python
scenario = run_coconet(scenario_cfg)
baseline = run_coconet(baseline_cfg, spinup_checkpoint=scenario.spinup_checkpoint)
```

The output is identical to a full run. The caller is responsible for checking that the
two configurations really match up to `start_year`.

### Logging

- **`run_coconet(..., configure_logs=True)`** calls **`configure_logging`** using **`log_level`** if passed, else **`config.log_level`**.
- **`configure_logs=False`** leaves **`logging`** alone—use this when your host app (e.g. Django, Celery) already configured loggers, or when the CLI has already configured logging (the CLI uses **`False`** after its own bootstrap).

## Advanced: `CoconetConfig.from_file` and `CoconetModel`

You can bypass **`load_coconet_config`** and mutate **`CoconetConfig`** by hand, then run the engine directly. This matches older examples and fine-grained tests.

```python
from coconet import CoconetConfig, CoconetModel
from coconet.logging_utils import configure_logging

cfg = CoconetConfig.from_file(
    config_file="config/example.yaml",
    parameter_file=None,
    env_prefix="COCONET_",
)
cfg.output_file = "runs/my_run.csv"
cfg.ensemble_runs = 3

configure_logging(cfg.log_level)
model = CoconetModel(cfg)
model.run()
```

`CoconetModel.run()` performs full setup, optional parallel ensemble execution, and writes the configured **`output_file`** (and search-mode side outputs when applicable).

## Advanced imports (not in `__all__`)

Examples of symbols other code may touch when extending or testing:

| Module | Examples |
| --- | --- |
| `coconet.config` | `effective_ensemble_workers`, `use_parallel_ensemble_run` |
| `coconet.netlogo` | `NetLogoRng`, `heading_from_dx_dy`, `nl_ceiling`, `nl_median`, `nl_round` |
| `coconet.model` | `CORAL_GROUPS`, `ReefKernel`, `SpinupCheckpoint` (in-memory spin-up snapshot for parallel workers) |

Treat anything outside **`__all__`** as **internal** unless you rely on a specific parity or test contract.

## Dependencies

Declared in `pyproject.toml`: NumPy, pandas, PyYAML, threadpoolctl (BLAS thread limiting inside worker processes).

See: [Architecture]({% link architecture.md %}), [Configuration]({% link configuration.md %}), [CLI]({% link cli.md %}).