"""Summarise and compare ensemble output. All figures come from the CSV."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from coconet_assistant.options import CORAL_GROUP_LABELS, OUTPUT_METRICS
from coconet_assistant.output import add_derived_columns, filter_replicates, read_output_table
from coconet_assistant.scenario import display_settings

DEFAULT_METRICS: tuple[str, ...] = (
    "coral_total",
    "C_sa",
    "C_ta",
    "C_mo",
    "C_po",
    "C_fa",
    "C_tt",
    "S_manta",
    "emperor",
    "trout",
    "Triggerfish",
    "Benthic_invert",
    "DHW",
)

CAVEATS: tuple[str, ...] = (
    "Numbers are spatial means of reef-level output, then the mean and 10-90% "
    "range across stochastic ensembles (ensembles 1..N). Ensemble 0 is spin-up "
    "and is excluded.",
    "Ensemble bands describe stochastic spread in this model, not all real-world "
    "uncertainty (parameter, structural, or observational).",
    "Rubble is simulated inside the model (R_site / R_reef) but is not written "
    "to the standard output.csv, so rubble trends cannot be reported from that file.",
    "CoCoNet is a GBR-scale abstraction. It is not a forecast for a named reef "
    "or an operational management optimiser.",
    "Climate is represented by a small set of SSP numbers (1.9, 2.6, 4.5, 7.0, 8.5), "
    "not a continuous emissions pathway.",
    "Most spatial interventions treat reefs in priority order inside a lat/lon box; "
    "CoTS regional control uses Region codes (FN, N, C, S, GBR) instead.",
)


def _round_float(value: float) -> float:
    if not np.isfinite(value):
        return value
    return float(round(value, 6))


def _series_from_grouped(grouped: pd.DataFrame, metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, block in grouped.groupby("Year"):
        values = pd.to_numeric(block[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "year": int(pd.to_numeric(year)),
                "mean": _round_float(float(values.mean())),
                "q10": _round_float(float(values.quantile(0.10))),
                "q90": _round_float(float(values.quantile(0.90))),
                "n_ensembles": int(values.shape[0]),
            }
        )
    rows.sort(key=lambda item: item["year"])
    return rows


def ensemble_year_means(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean across reefs for each ensemble and year (one value per replicate-year)."""
    present = [m for m in metrics if m in frame.columns]
    if not present:
        return pd.DataFrame(columns=["Ensemble", "Year", *metrics])
    numeric = frame[["Ensemble", "Year", *present]].copy()
    for name in present:
        numeric[name] = pd.to_numeric(numeric[name], errors="coerce")
    return numeric.groupby(["Ensemble", "Year"], as_index=False)[present].mean()


def summarise_output(
    output_file: str,
    *,
    region: str | None = None,
    metrics: list[str] | None = None,
    scenario_name: str | None = None,
    scenario_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe summary from a completed CoCoNet output file."""

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    ## ~ Read output and aggregate ensembles ~ ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    chosen = list(metrics or DEFAULT_METRICS)
    table = add_derived_columns(read_output_table(output_file))
    table = filter_replicates(table, region=region)
    if table.empty:
        raise ValueError(
            "No stochastic ensemble rows (Ensemble > 0) found in the output file"
            + (f" for Region={region}" if region else "")
            + "."
        )

    grouped = ensemble_year_means(table, chosen)
    n_ensembles = int(table["Ensemble"].nunique())
    years = sorted(int(y) for y in table["Year"].unique())
    series: dict[str, list[dict[str, Any]]] = {}
    end_year_snapshot: dict[str, Any] = {}
    last_year = years[-1] if years else None

    for metric in chosen:
        if metric not in grouped.columns:
            continue
        series[metric] = _series_from_grouped(grouped, metric)
        if last_year is not None and series[metric]:
            last = next((row for row in series[metric] if row["year"] == last_year), None)
            if last is not None:
                end_year_snapshot[metric] = last

    facts = _facts_from_snapshot(
        end_year_snapshot,
        last_year=last_year,
        n_ensembles=n_ensembles,
        region=region,
        scenario_name=scenario_name,
    )

    return {
        "scenario_name": scenario_name,
        "region": region,
        "n_ensembles": n_ensembles,
        "n_reef_rows": len(table),
        "years": years,
        "end_year": last_year,
        "metrics": {m: OUTPUT_METRICS.get(m, m) for m in series},
        "series": series,
        "end_year_snapshot": end_year_snapshot,
        "coral_groups": CORAL_GROUP_LABELS,
        "facts": facts,
        "caveats": list(CAVEATS),
        "settings": display_settings(scenario_config) if scenario_config else [],
        "rubble_note": (
            "Rubble cover is not present in output.csv, so no rubble trend is reported."
        ),
    }


def _facts_from_snapshot(
    snapshot: dict[str, Any],
    *,
    last_year: int | None,
    n_ensembles: int,
    region: str | None,
    scenario_name: str | None,
) -> list[str]:
    """Deterministic sentences the language model may paraphrase (not invent)."""
    facts: list[str] = []
    who = scenario_name or "This scenario"
    where = f" in region {region}" if region else ""
    if last_year is None:
        facts.append(f"{who} produced no yearly snapshot.")
        return facts
    facts.append(
        f"{who}{where}: summary is for year {last_year} using {n_ensembles} "
        "stochastic ensemble replicates (mean and 10-90% range of reef-mean values)."
    )
    for metric in (
        "coral_total",
        "C_tt",
        "C_sa",
        "S_manta",
        "emperor",
        "trout",
    ):
        row = snapshot.get(metric)
        if not row:
            continue
        label = OUTPUT_METRICS.get(metric, metric)
        facts.append(
            f"{label}: mean {row['mean']} (10-90% {row['q10']} to {row['q90']}) "
            f"from {row['n_ensembles']} ensembles in {last_year}."
        )
    return facts


def compare_summaries(
    summaries: dict[str, dict[str, Any]],
    *,
    baseline_name: str | None = None,
) -> dict[str, Any]:
    """Side-by-side comparison. ``summaries`` maps scenario name -> summarise_output()."""
    if len(summaries) < 2:
        raise ValueError("compare_scenarios needs at least two named summaries.")

    names = list(summaries)
    if baseline_name is None:
        baseline_name = next((n for n in names if "baseline" in n.lower()), names[0])
    if baseline_name not in summaries:
        raise ValueError(f"Baseline name {baseline_name!r} is not in the summary set.")

    baseline = summaries[baseline_name]
    shared_years = set(baseline.get("years") or [])
    for summary in summaries.values():
        shared_years &= set(summary.get("years") or [])
    last_year = max(shared_years) if shared_years else None

    comparison_rows: list[dict[str, Any]] = []
    metric_keys: set[str] = set()
    for summary in summaries.values():
        metric_keys.update((summary.get("end_year_snapshot") or {}).keys())

    for metric in DEFAULT_METRICS:
        if metric not in metric_keys:
            continue
        row: dict[str, Any] = {
            "metric": metric,
            "label": OUTPUT_METRICS.get(metric, metric),
            "year": last_year,
            "values": {},
        }
        base_mean = None
        base_snap = (baseline.get("end_year_snapshot") or {}).get(metric)
        if base_snap:
            base_mean = base_snap["mean"]
        for name, summary in summaries.items():
            snap = (summary.get("end_year_snapshot") or {}).get(metric)
            if not snap:
                continue
            entry = {
                "mean": snap["mean"],
                "q10": snap["q10"],
                "q90": snap["q90"],
            }
            if base_mean is not None and name != baseline_name:
                entry["difference_vs_baseline"] = _round_float(snap["mean"] - base_mean)
                if base_mean != 0:
                    entry["percent_vs_baseline"] = _round_float(
                        100.0 * (snap["mean"] - base_mean) / base_mean
                    )
            row["values"][name] = entry
        comparison_rows.append(row)

    facts = _comparison_facts(comparison_rows, baseline_name, last_year)
    suggestions = suggest_next_scenarios(summaries, comparison_rows, baseline_name)

    return {
        "baseline": baseline_name,
        "scenarios": names,
        "end_year": last_year,
        "shared_years": sorted(shared_years),
        "rows": comparison_rows,
        "facts": facts,
        "suggestions": suggestions,
        "caveats": list(CAVEATS),
    }


def _comparison_facts(
    rows: list[dict[str, Any]],
    baseline_name: str,
    last_year: int | None,
) -> list[str]:
    facts: list[str] = []
    if last_year is None:
        return ["No shared years between the compared scenarios."]
    facts.append(
        f"Comparison year {last_year}. Differences are intervention mean minus "
        f"'{baseline_name}' mean, using the same aggregation as summarise_results."
    )
    for row in rows:
        if row["metric"] not in {"coral_total", "C_tt", "S_manta", "emperor", "trout"}:
            continue
        for name, value in row["values"].items():
            if name == baseline_name:
                facts.append(
                    f"{row['label']} in {baseline_name}: mean {value['mean']} "
                    f"(10-90% {value['q10']} to {value['q90']})."
                )
                continue
            diff = value.get("difference_vs_baseline")
            pct = value.get("percent_vs_baseline")
            extra = ""
            if diff is not None:
                extra = f" difference vs {baseline_name}: {diff}"
                if pct is not None:
                    extra += f" ({pct}%)."
            facts.append(
                f"{row['label']} in {name}: mean {value['mean']} "
                f"(10-90% {value['q10']} to {value['q90']}).{extra}"
            )
    return facts


def suggest_next_scenarios(
    summaries: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    baseline_name: str,
) -> list[str]:
    """Rule-based next tests. Suggestions do not invent numeric outcomes."""
    suggestions: list[str] = []
    coral_row = next((r for r in rows if r["metric"] == "coral_total"), None)
    cots_row = next((r for r in rows if r["metric"] == "S_manta"), None)
    tt_row = next((r for r in rows if r["metric"] == "C_tt"), None)

    improved = False
    declined = False
    if coral_row:
        for name, value in coral_row["values"].items():
            if name == baseline_name:
                continue
            diff = value.get("difference_vs_baseline")
            if diff is None:
                continue
            if diff > 0:
                improved = True
            if diff < 0:
                declined = True

    if declined:
        suggestions.append(
            "Total coral cover is lower than the baseline in at least one intervention. "
            "A useful next test is the same actions under a lower SSP (for example 2.6), "
            "or adding reef shading / earlier CoTS control."
        )
    if improved:
        suggestions.append(
            "Total coral cover is higher than the baseline in at least one intervention. "
            "A useful next test is a smaller intervention budget, or the same package "
            "started later, to see how much of the gain remains."
        )
    if cots_row:
        for name, value in cots_row["values"].items():
            if name == baseline_name:
                continue
            diff = value.get("difference_vs_baseline")
            if diff is not None and diff >= 0:
                suggestions.append(
                    "CoTS (S_manta) is not lower than the baseline. Try more vessels "
                    "in the same region, an earlier start_CoTS_control, or GBR-wide "
                    "vessels rather than a single region."
                )
                break
    if tt_row:
        for name, value in tt_row["values"].items():
            if name == baseline_name:
                continue
            diff = value.get("difference_vs_baseline")
            if diff is not None and diff <= 0:
                suggestions.append(
                    "Thermally tolerant cover (C_tt) did not increase versus the baseline. "
                    "Check seed_threshold (reefs at or above it are skipped), seed_hectares, "
                    "and that the intervention box actually covers the intended reefs."
                )
                break

    configs = [s.get("settings") for s in summaries.values()]
    ssp_values = set()
    for settings in configs:
        if not settings:
            continue
        for row in settings:
            if row.get("field") == "SSP":
                ssp_values.add(row.get("value"))
    if ssp_values == {4.5} or ssp_values == {7.0} or ssp_values == {8.5}:
        suggestions.append(
            "Every compared run uses the same relatively high SSP. "
            "Running the same interventions under SSP 2.6 shows how much of the "
            "outcome is climate versus management."
        )

    if not suggestions:
        suggestions.append(
            "Try changing one lever at a time: climate (SSP), CoTS vessel region, "
            "seeding start year, or the intervention bounding box."
        )
    return suggestions


def summarise_results(
    output_file: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Public alias used by the tool layer."""
    return summarise_output(output_file, **kwargs)
