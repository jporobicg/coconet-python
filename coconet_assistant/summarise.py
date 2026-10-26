"""Summarise and compare ensemble output. All figures come from the CSV."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from coconet_assistant.options import CORAL_GROUP_LABELS, OUTPUT_METRICS, region_label
from coconet_assistant.output import (
    CORAL_COLUMNS,
    EMPEROR_COLUMNS,
    TROUT_COLUMNS,
    add_derived_columns,
    filter_replicates,
    read_output_table,
)
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
    "Interventions treat reefs in priority order. A named region uses the reefs' own "
    "Region codes (FN, N, C, S) for every lever, CoTS control included.",
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


SOURCE_COLUMNS: tuple[str, ...] = (
    *CORAL_COLUMNS,
    *EMPEROR_COLUMNS,
    *TROUT_COLUMNS,
    "S_manta",
    "Triggerfish",
    "Benthic_invert",
    "DHW",
)


def region_year_sums(output_file: str, metrics: list[str] | None = None) -> pd.DataFrame:
    """Per ensemble, year, and Region: the reef count and the sum of each metric.

    Reading a full GBR output once and keeping these sums lets every region (or the
    whole GBR) be summarised without re-reading millions of rows.
    """
    chosen = list(metrics or DEFAULT_METRICS)
    table = add_derived_columns(read_output_table(output_file, columns=list(SOURCE_COLUMNS)))
    table = filter_replicates(table)
    if "Region" not in table.columns:
        table["Region"] = "GBR"
    present = [m for m in chosen if m in table.columns]
    for name in present:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    sums = table.groupby(["Ensemble", "Year", "Region"], as_index=False)[present].sum()
    counts = table.groupby(["Ensemble", "Year", "Region"], as_index=False).size()
    return sums.merge(counts.rename(columns={"size": "n_reefs"}), on=["Ensemble", "Year", "Region"])


def replicate_means(sums: pd.DataFrame, region: str | None = None) -> pd.DataFrame:
    """Reef-mean of each metric per ensemble and year, for one region or all reefs."""
    subset = sums if region is None else sums[sums["Region"] == region]
    metrics = [c for c in subset.columns if c not in {"Ensemble", "Year", "Region", "n_reefs"}]
    totals = subset.groupby(["Ensemble", "Year"], as_index=False)[[*metrics, "n_reefs"]].sum()
    for name in metrics:
        totals[name] = totals[name] / totals["n_reefs"]
    return totals


def summarise_output(
    output_file: str,
    *,
    region: str | None = None,
    metrics: list[str] | None = None,
    scenario_name: str | None = None,
    scenario_config: dict[str, Any] | None = None,
    sums: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe summary from a completed CoCoNet output file.

    ``sums`` (from :func:`region_year_sums`) avoids re-reading the output file.
    """

    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    ## ~ Read output and aggregate ensembles ~ ##
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    chosen = list(metrics or DEFAULT_METRICS)
    if sums is None:
        sums = region_year_sums(output_file, chosen)
    if region is not None:
        sums = sums[sums["Region"] == region]
    if sums.empty:
        raise ValueError(
            "No stochastic ensemble rows (Ensemble > 0) found in the output file"
            + (f" for Region={region}" if region else "")
            + "."
        )

    grouped = replicate_means(sums)
    n_ensembles = int(grouped["Ensemble"].nunique())
    years = sorted(int(y) for y in grouped["Year"].unique())
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
        "region_label": region_label(region),
        "n_ensembles": n_ensembles,
        "n_reef_rows": int(sums["n_reefs"].sum()),
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
    where = f" ({region_label(region)})"
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


# Metrics where a lower value is the better outcome.
LOWER_IS_BETTER = frozenset({"S_manta", "DHW"})
# Cover metrics are fractions in output.csv; managers read them as % cover.
COVER_METRICS = frozenset({"coral_total", *CORAL_COLUMNS})
FACT_METRICS: tuple[str, ...] = ("coral_total", "C_tt", "S_manta", "emperor", "trout")


def format_metric(metric: str, value: float, *, signed: bool = False) -> str:
    if metric in COVER_METRICS:
        text = f"{100 * value:+.1f}" if signed else f"{100 * value:.1f}"
        return f"{text} percentage points" if signed else f"{text}%"
    return f"{value:+.3g}" if signed else f"{value:.3g}"


def paired_difference(
    scenario: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str,
    *,
    year: int,
    from_year: int | None = None,
) -> dict[str, Any] | None:
    """Scenario minus baseline, replicate by replicate (ensemble i against ensemble i).

    Both runs share seeds and are identical before the first intervention, so the
    paired difference removes most stochastic noise that unpaired means carry.
    """
    if metric not in scenario.columns or metric not in baseline.columns:
        return None
    s = scenario.set_index(["Ensemble", "Year"])[metric]
    b = baseline.set_index(["Ensemble", "Year"])[metric]
    diff = (s - b).dropna()
    if diff.empty or year not in set(diff.index.get_level_values("Year")):
        return None
    end = diff.xs(year, level="Year")
    better = end < 0 if metric in LOWER_IS_BETTER else end > 0
    q10, q90 = float(end.quantile(0.10)), float(end.quantile(0.90))
    clear = q10 > 0 or q90 < 0
    result: dict[str, Any] = {
        "year": year,
        "mean": _round_float(float(end.mean())),
        "q10": _round_float(q10),
        "q90": _round_float(q90),
        "n_pairs": int(end.shape[0]),
        "n_better": int(better.sum()),
        # The 10-90% range of paired differences excludes zero.
        "clear": bool(clear),
        "direction": (
            "better" if clear and bool(better.mean() > 0.5) else "worse" if clear else "unclear"
        ),
    }
    if from_year is not None:
        window = diff[diff.index.get_level_values("Year") >= from_year]
        integrated = window.groupby(level="Ensemble").sum()
        if not integrated.empty:
            result["integrated"] = {
                "from_year": from_year,
                "to_year": year,
                "mean": _round_float(float(integrated.mean())),
                "q10": _round_float(float(integrated.quantile(0.10))),
                "q90": _round_float(float(integrated.quantile(0.90))),
            }
    return result


def paired_series(
    scenario: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str,
) -> list[dict[str, Any]]:
    """Per year: mean and 10-90% range of the replicate-by-replicate difference."""
    if metric not in scenario.columns or metric not in baseline.columns:
        return []
    s = scenario.set_index(["Ensemble", "Year"])[metric]
    b = baseline.set_index(["Ensemble", "Year"])[metric]
    diff = (s - b).dropna()
    rows = []
    for year, values in diff.groupby(level="Year"):
        rows.append(
            {
                "year": int(pd.to_numeric(year)),
                "mean": _round_float(float(values.mean())),
                "q10": _round_float(float(values.quantile(0.10))),
                "q90": _round_float(float(values.quantile(0.90))),
                "n_ensembles": int(values.shape[0]),
            }
        )
    return rows


def compare_summaries(
    summaries: dict[str, dict[str, Any]],
    *,
    baseline_name: str | None = None,
    replicates: dict[str, pd.DataFrame] | None = None,
    effect_from_year: int | None = None,
) -> dict[str, Any]:
    """Side-by-side comparison. ``summaries`` maps scenario name -> summarise_output().

    ``replicates`` maps the same names to :func:`replicate_means` frames; when given,
    each scenario also gets a paired, replicate-by-replicate difference from the
    baseline. ``effect_from_year`` (first intervention year) sets the start of the
    time-integrated difference.
    """
    if len(summaries) < 2:
        raise ValueError("compare_scenarios needs at least two named summaries.")

    names = list(summaries)
    if baseline_name is None:
        baseline_name = next((n for n in names if "baseline" in n.lower()), names[0])
    if baseline_name not in summaries:
        raise ValueError(f"Baseline name {baseline_name!r} is not in the summary set.")

    regions = {summary.get("region") for summary in summaries.values()}
    if len(regions) > 1:
        raise ValueError(
            "compare_scenarios received summaries for different regions "
            f"({sorted(str(r or 'GBR') for r in regions)}); compare one region at a time."
        )
    region = regions.pop()

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
            "lower_is_better": metric in LOWER_IS_BETTER,
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
            entry: dict[str, Any] = {
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
                if replicates and name in replicates and baseline_name in replicates:
                    paired = paired_difference(
                        replicates[name],
                        replicates[baseline_name],
                        metric,
                        year=int(last_year) if last_year is not None else -1,
                        from_year=effect_from_year,
                    )
                    if paired is not None:
                        entry["paired"] = paired
            row["values"][name] = entry
        comparison_rows.append(row)

    facts = _comparison_facts(comparison_rows, baseline_name, last_year, region)
    suggestions = suggest_next_scenarios(summaries, comparison_rows, baseline_name)

    return {
        "region": region,
        "region_label": region_label(region),
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
    region: str | None = None,
) -> list[str]:
    facts: list[str] = []
    if last_year is None:
        return ["No shared years between the compared scenarios."]
    facts.append(
        f"Comparison year {last_year}, {region_label(region)}. A paired change compares each "
        f"replicate with its own '{baseline_name}' twin; it is called clear only when the "
        "10-90% range of those paired changes excludes zero."
    )
    for row in rows:
        metric = row["metric"]
        if metric not in FACT_METRICS:
            continue
        for name, value in row["values"].items():
            level = (
                f"{row['label']} in {name}: mean {format_metric(metric, value['mean'])} "
                f"(10-90% {format_metric(metric, value['q10'])} to {format_metric(metric, value['q90'])})."
            )
            paired = value.get("paired")
            if name == baseline_name or paired is None:
                diff = value.get("difference_vs_baseline")
                if diff is not None:
                    level += f" Difference of means vs {baseline_name}: {format_metric(metric, diff, signed=True)}."
                facts.append(level)
                continue
            verdict = {
                "better": "a clear improvement",
                "worse": "a clear deterioration",
                "unclear": "within replicate noise (not a clear effect)",
            }[paired["direction"]]
            facts.append(
                f"{level} Paired change vs {baseline_name}: "
                f"{format_metric(metric, paired['mean'], signed=True)} "
                f"(10-90% {format_metric(metric, paired['q10'], signed=True)} to "
                f"{format_metric(metric, paired['q90'], signed=True)}); better in "
                f"{paired['n_better']} of {paired['n_pairs']} replicates; {verdict}."
            )
    return facts


def _directions(row: dict[str, Any] | None, baseline_name: str) -> set[str]:
    """Paired verdicts ("better", "worse", "unclear") for the non-baseline runs."""
    if row is None:
        return set()
    out = set()
    for name, value in row["values"].items():
        if name != baseline_name and value.get("paired"):
            out.add(value["paired"]["direction"])
    return out


def suggest_next_scenarios(
    summaries: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    baseline_name: str,
) -> list[str]:
    """Rule-based next tests. Suggestions do not invent numeric outcomes.

    Verdicts use the paired replicate comparison, so a difference inside the
    replicate noise is never reported as an improvement or a decline.
    """
    suggestions: list[str] = []
    by_metric = {r["metric"]: r for r in rows}
    coral = _directions(by_metric.get("coral_total"), baseline_name)
    cots = _directions(by_metric.get("S_manta"), baseline_name)
    tt = _directions(by_metric.get("C_tt"), baseline_name)

    if "worse" in coral:
        suggestions.append(
            "Total coral cover is clearly lower than the baseline in at least one run. "
            "A useful next test is the same actions under a lower SSP (for example 2.6), "
            "or adding reef shading or earlier CoTS control."
        )
    if "better" in coral:
        suggestions.append(
            "Total coral cover is clearly higher than the baseline in at least one run. "
            "A useful next test is a smaller intervention, or the same package started "
            "later, to see how much of the gain remains."
        )
    if coral == {"unclear"}:
        suggestions.append(
            "The change in total coral cover is within replicate noise. Look at the "
            "targeted region rather than the whole GBR, or try a larger intervention."
        )
    if cots and "better" not in cots:
        suggestions.append(
            "CoTS density is not clearly lower than the baseline. Try more vessels in "
            "the same region, an earlier start_CoTS_control, or GBR-wide vessels."
        )
    if tt and "better" not in tt:
        suggestions.append(
            "Thermally tolerant cover (C_tt) did not clearly increase. Check "
            "seed_threshold (reefs at or above it are skipped), seed_hectares, and the "
            "intervention region."
        )

    ssp_values = set()
    for summary in summaries.values():
        for row in summary.get("settings") or []:
            if row.get("field") == "SSP":
                ssp_values.add(row.get("value"))
    if ssp_values in ({4.5}, {7.0}, {8.5}):
        suggestions.append(
            "Every compared run uses the same relatively high SSP. "
            "Running the same interventions under SSP 2.6 shows how much of the "
            "outcome is climate versus management."
        )

    if not suggestions:
        suggestions.append(
            "Try changing one lever at a time: climate (SSP), CoTS vessel region, "
            "seeding start year, or the intervention region."
        )
    return suggestions


def summarise_results(
    output_file: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Public alias used by the tool layer."""
    return summarise_output(output_file, **kwargs)
