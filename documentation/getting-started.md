---
title: Getting started
description: Install CoCoNet Python, choose CLI vs library entry points, run your first simulation, and understand repository layout.
---

## Two entry points

CoCoNet exposes the same simulation through:

1. **Command-line interface (CLI)** — for **files, flags, and environment variables** (terminals, Docker, shell scripts, simple CI). Use the **`coconet`** command or **`python -m coconet`**.
2. **Python library** — for **programmatic control** from other Python code. Install the **PyPI** distribution **`coconet-python`**; **`import coconet`** unchanged.

See [CLI]({% link cli.md %}) and [Python API]({% link python-api.md %}) for full detail.

## Requirements

- **Python 3.11+**
- Dependencies (see `pyproject.toml`): NumPy 2.2+, pandas 2.2+, PyYAML 6+, threadpoolctl 3.5+

## Install

### From PyPI (library + CLI)

The published package is **`coconet-python`**. After a normal install you get **both** the import package **`coconet`** and the **`coconet`** console script:

```bash
pip install coconet-python
coconet --help
```

```bash
python -c "from coconet import load_coconet_config, run_coconet; print('ok')"
```

### From a git checkout (development)

From the repository root:

```bash
uv sync
uv run coconet --parameter-file legacy/I_2p6.csv --output-file output.csv
```

With a YAML config:

```bash
uv run coconet --config config/example.yaml --output-file output.csv
```

Editable / local install without uv:

```bash
pip install .
coconet --config config/example.yaml --output-file output.csv
```

Optional extras:

```bash
uv sync --extra profile          # CPU profiles via pyinstrument
uv sync --extra assistant        # manager Streamlit chat (see docs/manager-assistant.md)
# or: pip install "coconet-python[profile]" / pip install "coconet-python[assistant]"
```

## First run (CLI)

```bash
# After pip install coconet-python, from a checkout or copy of inputs:
coconet --parameter-file legacy/I_2p6.csv --output-file output.csv
```

or with YAML:

```bash
coconet --config config/example.yaml --output-file output.csv
```

## First run (library)

```python
from coconet import load_coconet_config, run_coconet

cfg = load_coconet_config(
    config_file="config/example.yaml",
    output_file="output.csv",
)
run_coconet(cfg, configure_logs=True)
```

Use **`configure_logs=True`** when your application does not already configure the standard **`logging`** module; set it **`False`** and configure logging yourself for tighter integration with frameworks.

## Environment overrides

Any `CoconetConfig` field can be overridden with the `COCONET_` prefix (case-insensitive suffix matching the field name). This applies to **both** the CLI (before/during config load) and the library (`load_coconet_config` / `CoconetConfig.from_file`).

```bash
export COCONET_ENSEMBLE_RUNS=2
export COCONET_END_YEAR=1990
coconet --parameter-file legacy/I_2p6.csv
```

See [Configuration]({% link configuration.md %}) for precedence.

## Repository layout

| Path | Purpose |
| --- | --- |
| `coconet/` | Package: `config.py`, `model.py`, **`api.py`** (library entry), **`cli.py`** (CLI), `netlogo.py`, `logging_utils.py`, `__main__.py` |
| `config/` | Example YAML (e.g. `example.yaml`) |
| `legacy/` | Original NetLogo model, reef and coastline CSVs, scenario parameter files |
| `documentation/` | This Jekyll documentation site |
| `coconet_assistant/` | Optional manager chat tools (not required by the core model) |
| `app.py` | Streamlit UI for the assistant (`uv sync --extra assistant`) |
| `viz/` | Optional React viewer for exploring `output.csv` |
| `gen-charts/` | Optional Node tooling to render charts from output |

## Typical workflow

1. Choose **parameter file** (legacy CSV) and/or **YAML** for scenario settings.
2. Point **reefs** and **coastline** inputs at CSVs (defaults often use `legacy/` in a checkout).
3. Run the **CLI** (`coconet` / `python -m coconet`) **or** build config in Python with **`load_coconet_config`** and call **`run_coconet`** (**library**).
4. Consume **`output.csv`** (and optionally `priority_reef_benefit.csv` in search mode).

Details: [Inputs and outputs]({% link inputs-and-outputs.md %}), [CLI]({% link cli.md %}), [Python API]({% link python-api.md %}).
