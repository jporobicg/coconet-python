# Manager assistant

A Streamlit chat on top of the CoCoNet Python port. Reef managers can ask
plain-language questions, confirm a scenario, run an **ensemble**, and read
coral / CoTS / fish results against a baseline. A local open model (Llama or
Qwen via Ollama) does the talking. Scientific numbers come only from the
Python tools, never from the language model.

## Install

1. Install CoCoNet with the optional assistant extra (from a checkout):

   ```bash
   uv sync --extra assistant
   # or: pip install -e ".[assistant]"
   ```

   The extra adds Streamlit, httpx (Ollama HTTP), and matplotlib. The core
   `coconet` model does not require any of these.

2. Install [Ollama](https://ollama.com) and start it on this machine:

   ```bash
   ollama serve
   ```

3. Pull a tool-capable local model. Either of these is a reasonable default:

   ```bash
   ollama pull qwen2.5
   ollama pull llama3.1
   ```

   The default tag is `qwen2.5`. Override with `COCONET_ASSISTANT_MODEL` or the
   sidebar in the app. The client is model-agnostic: any Ollama chat model that
   supports tool/function calling will work.

4. Point the model at the reef and coastline CSVs (defaults are
   `legacy/reefs2024.csv` and `legacy/coastline.csv` in a checkout).

If Ollama is not running, the app shows a clear message (`Ollama is not running
at http://127.0.0.1:11434. Start it with \`ollama serve\` …`) and does not
invent results.

## How to run

From the repository root:

```bash
uv run --extra assistant streamlit run app.py
```

or, after `pip install -e ".[assistant]"`:

```bash
streamlit run app.py
```

Optional environment:

| Variable | Role | Default |
| --- | --- | --- |
| `COCONET_ASSISTANT_MODEL` | Ollama model tag | `qwen2.5` |
| `COCONET_OLLAMA_HOST` | Ollama base URL | `http://127.0.0.1:11434` |
| `COCONET_OLLAMA_NUM_CTX` | Context window sent with every request | `16384` |

Every request also sends `temperature: 0` and `seed: 0`, so the same question gives
the same proposed scenario. The context window is set explicitly because Ollama's
own default (2–4k tokens) silently truncates the system prompt once tool results
arrive.

`.streamlit/config.toml` turns off Streamlit usage statistics and binds the app to
`localhost`, so only this machine can open it. Change `server.address` deliberately
if you want to share the app on a trusted network.

Completed ensembles are written under `.coconet_assistant_cache/` (configurable
in the sidebar). Asking the same scenario again reuses the cached `output.csv`.

## Example questions

- What if we control CoTS in the northern sector and seed thermally tolerant coral from 2030 under SSP2-4.5?
- Run a baseline under SSP1-2.6 out to 2050 with 20 ensemble members.
- Add rubble consolidation from 2030 on 50 reefs. What can you actually report about rubble?
- Compare the last two runs and suggest what to try next.
- Same interventions, but CoTS vessels across the GBR instead of the northern region.

Typical flow:

1. You ask a question.
2. The assistant calls `list_options` and `build_scenario`. Simulation timing
   (`start_year`, `save_year`, `projection_year`, spin-up) is fixed by the app; the
   assistant only sets `end_year` and each intervention's start year.
3. You see the proposed `CoconetConfig` settings (field labels and values) and
   click **Confirm and run ensemble** (or type “yes”). Only this human
   confirmation can start a run: a `confirmed` flag from the language model is
   ignored.
4. CoCoNet runs ensembles 1..N (default 20, never a single stochastic run). A
   no-intervention baseline with the same climate, years, and ensemble size is
   run unless you turn that off.
5. Plots show mean and 10–90% ensemble bands. The assistant writes a short
   briefing using only the tool facts, plus caveats.

## Architecture

```
Manager  →  Streamlit app.py
                 │
                 ├─ Ollama (local chat + tool calls)
                 │
                 └─ coconet_assistant/
                        tools.py        list_options, build_scenario,
                                        validate_scenario, run_ensemble,
                                        summarise_results, compare_scenarios, plot
                        scenario.py     maps manager language onto CoconetConfig
                        ensemble.py     coconet.api.run_coconet + disk cache
                        summarise.py    ensemble means and 10–90% bands
                        plot.py         series + optional matplotlib PNGs
```

Rules the language model is given:

- Only report numbers that appear in tool outputs.
- Validate every proposed config against the real `CoconetConfig` schema.
- Wait for confirmation before `run_ensemble`.
- Never request `ensemble_runs = 1`.

Climate names such as “SSP2-4.5” map to the numeric `SSP` the model actually
tests (`4.5`). CoTS “northern sector / northern region” maps to
`CoTS_vessels_N`. Thermally tolerant seeding maps to `start_coral_seeding`,
`seed_reefs`, `seed_hectares`, and `seed_threshold`.

## What the plots show

For each year, reefs are averaged inside each stochastic ensemble, then the
mean and 10th–90th percentiles are taken across ensembles 1..N. Ensemble 0
(spin-up) is excluded.

Every summary, comparison, and plot states which reefs it covers: all reefs
(GBR-wide) or one region (`FN`, `N`, `C`, `S`). Summaries are stored per run and
region, and a comparison never mixes regions.

Reported series: total coral cover (sum of the six groups), each group,
`S_manta` (CoTS), emperors (`E_1`…`E_5`), coral trout (`G_1`…`G_5`).

## Limitations

- CoCoNet is a GBR-scale abstraction, not a forecast for a named reef and not
  an optimiser of real vessel-days.
- Ensemble bands are stochastic spread only. They omit parameter, structural,
  and observational error.
- Climate is one of five SSP numbers (`1.9`, `2.6`, `4.5`, `7.0`, `8.5`).
- Rubble is simulated (`R_site` / `R_reef`) and rubble consolidation can run,
  but rubble is **not** written to `output.csv`, so the assistant will not
  invent a rubble trend.
- Most spatial interventions treat reefs in priority order inside a lat/lon
  box. CoTS regional control uses Region codes (`FN`, `N`, `C`, `S`, `GBR`)
  instead. Named-region boxes used for seeding are a coarse split, not official
  GBRMPA polygons.
- A full GBR ensemble is a heavy run. Start with 20 replicates and a shorter
  end year if you are testing the app. Cached repeats are instant.

## Tests

Unit tests cover scenario building/validation, result summaries, and the tool
layer with a **mocked** language model (no Ollama required):

```bash
uv run pytest tests/unit/test_assistant_scenario.py \
              tests/unit/test_assistant_summarise.py \
              tests/unit/test_assistant_tools.py
```

To check how well a real local model translates manager questions, run the
evaluation set against a running Ollama. It checks 10 typical questions against
their expected settings and never runs a simulation:

```bash
uv run --extra assistant python scripts/eval_assistant.py --model qwen2.5 --repeats 3
```

It exits non-zero when fewer than 90% of cases pass. Run it before changing the
default model, the system prompt, or the tool schemas.
