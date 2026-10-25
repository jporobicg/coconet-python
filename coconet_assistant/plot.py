"""Build plot data (always) and optional matplotlib figures (if installed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from coconet_assistant.options import CORAL_GROUP_LABELS, OUTPUT_METRICS

PLOT_GROUPS: dict[str, list[str]] = {
    "coral_total": ["coral_total"],
    "coral_groups": ["C_sa", "C_ta", "C_mo", "C_po", "C_fa", "C_tt"],
    "cots": ["S_manta"],
    "fish": ["emperor", "trout"],
    "other": ["Triggerfish", "Benthic_invert", "DHW"],
}


def _metric_label(metric: str) -> str:
    if metric in CORAL_GROUP_LABELS:
        return f"{CORAL_GROUP_LABELS[metric]} ({metric})"
    return OUTPUT_METRICS.get(metric, metric)


def plot_series_payload(
    summary: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    groups: list[str] | None = None,
) -> dict[str, Any]:
    """Numeric series the app and the language model can both use."""
    chosen = list(groups or ["coral_total", "coral_groups", "cots", "fish"])
    panels: list[dict[str, Any]] = []
    for group in chosen:
        metrics = PLOT_GROUPS.get(group)
        if not metrics:
            continue
        traces: list[dict[str, Any]] = []
        for metric in metrics:
            series = (summary.get("series") or {}).get(metric)
            if not series:
                continue
            traces.append(
                {
                    "metric": metric,
                    "label": _metric_label(metric),
                    "scenario": summary.get("scenario_name") or "scenario",
                    "years": [row["year"] for row in series],
                    "mean": [row["mean"] for row in series],
                    "q10": [row["q10"] for row in series],
                    "q90": [row["q90"] for row in series],
                }
            )
            if baseline is not None:
                base_series = (baseline.get("series") or {}).get(metric)
                if base_series:
                    traces.append(
                        {
                            "metric": metric,
                            "label": f"{_metric_label(metric)} (baseline)",
                            "scenario": baseline.get("scenario_name") or "baseline",
                            "years": [row["year"] for row in base_series],
                            "mean": [row["mean"] for row in base_series],
                            "q10": [row["q10"] for row in base_series],
                            "q90": [row["q90"] for row in base_series],
                            "is_baseline": True,
                        }
                    )
        if traces:
            panels.append({"group": group, "traces": traces})
    return {
        "n_ensembles": summary.get("n_ensembles"),
        "region": summary.get("region"),
        "region_label": summary.get("region_label") or "all reefs (GBR-wide)",
        "panels": panels,
        "notes": [
            "Shaded bands are the 10-90% range across stochastic ensembles.",
            "Lines are the ensemble-mean of reef-mean values.",
        ],
    }


PANEL_TITLES: dict[str, str] = {
    "coral_total": "Total coral cover",
    "coral_groups": "Coral cover by group",
    "cots": "Crown-of-thorns starfish",
    "fish": "Emperors and coral trout",
    "other": "Other indicators",
    "change_coral_total": "Change in total coral cover vs no intervention",
    "change_S_manta": "Change in CoTS density vs no intervention",
}
PANEL_UNITS: dict[str, str] = {
    "coral_total": "Coral cover (%)",
    "coral_groups": "Coral cover (%)",
    "cots": "CoTS per manta tow (model units)",
    "fish": "Fish abundance (model units)",
    "other": "Model units",
    "change_coral_total": "Change in coral cover (percentage points)",
    "change_S_manta": "Change in CoTS per manta tow",
}
COVER_PANELS = frozenset({"coral_total", "coral_groups", "change_coral_total"})
SCENARIO_COLOUR = "#1f6fb4"
BASELINE_COLOUR = "#7f7f7f"
GROUP_COLOURS = ("#1f6fb4", "#e07b1a", "#2a9d5c", "#c23b3b", "#8a5fbf", "#0f9fb0")
YEARS_BEFORE_INTERVENTION = 10


def _short_label(trace: dict[str, Any], single_metric: bool) -> str | None:
    if trace.get("is_difference"):
        return "Paired change (each replicate minus its baseline twin)"
    if single_metric:
        return "No intervention (baseline)" if trace.get("is_baseline") else "With interventions"
    if trace.get("is_baseline"):
        return None  # explained once by a proxy legend entry
    code = str(trace["metric"]).removeprefix("C_")
    label = CORAL_GROUP_LABELS.get(code, str(trace["label"]))
    return label[:1].upper() + label[1:]


def render_plots(
    payload: dict[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "scenario",
) -> list[str]:
    """Write one PNG per panel. Returns paths. Requires matplotlib.

    Cover is shown in %, the window starts a decade before the first intervention,
    and the intervention start and the region are marked on every figure.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plot rendering needs matplotlib. Install with: pip install 'coconet-python[assistant]'"
        ) from exc

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    effect_year = payload.get("effect_from_year")
    first_year = effect_year - YEARS_BEFORE_INTERVENTION if effect_year else None
    paths: list[str] = []
    for panel in payload.get("panels") or []:
        group = panel["group"]
        scale = 100.0 if group in COVER_PANELS else 1.0
        metrics = list(dict.fromkeys(t["metric"] for t in panel["traces"]))
        single = len(metrics) == 1
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for trace in panel["traces"]:
            keep = [
                i for i, y in enumerate(trace["years"]) if first_year is None or y >= first_year
            ]
            years = [trace["years"][i] for i in keep]
            mean = [scale * trace["mean"][i] for i in keep]
            q10 = [scale * trace["q10"][i] for i in keep]
            q90 = [scale * trace["q90"][i] for i in keep]
            if single:
                colour = BASELINE_COLOUR if trace.get("is_baseline") else SCENARIO_COLOUR
            else:
                colour = GROUP_COLOURS[metrics.index(trace["metric"]) % len(GROUP_COLOURS)]
            if trace.get("is_difference"):
                colour = SCENARIO_COLOUR
            style = "--" if trace.get("is_baseline") else "-"
            ax.plot(years, mean, style, color=colour, lw=2, label=_short_label(trace, single))
            if single:
                ax.fill_between(years, q10, q90, color=colour, alpha=0.15, lw=0)
        if not single:
            ax.plot(
                [], [], "--", color="#555555", lw=1.5, label="Dashed: no intervention (baseline)"
            )
        if panel.get("kind") == "difference":
            ax.axhline(0, color="#333333", lw=1)
        if effect_year:
            ax.axvline(effect_year, color="#333333", lw=1, ls=":")
            ax.text(
                effect_year,
                0.5,
                f" Interventions start ({effect_year}) ",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="right",
                va="center",
                fontsize=8,
                color="#333333",
            )
        ax.set_xlabel("Year")
        ax.set_ylabel(PANEL_UNITS.get(group, group))
        ax.set_title(
            f"{PANEL_TITLES.get(group, group)} \u2014 {payload.get('region_label', 'all reefs')}",
            fontsize=11,
        )
        ax.legend(fontsize=8, loc="best", frameon=False)
        ax.grid(True, alpha=0.25)
        if panel.get("kind") == "difference":
            fig.text(
                0.01,
                0.01,
                "Each replicate is compared with its baseline twin (same random seeds). "
                "Shading: 10-90% range of those paired changes; above 0 = higher than baseline.",
                fontsize=7,
                color="#555555",
            )
        elif single:
            fig.text(
                0.01,
                0.01,
                f"Lines: mean of {payload.get('n_ensembles', 'all')} replicates. "
                "Shading: 10-90% range across replicates.",
                fontsize=7,
                color="#555555",
            )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = folder / f"{stem}_{group}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_results(
    summary: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    groups: list[str] | None = None,
    stem: str = "scenario",
    effect_from_year: int | None = None,
    differences: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    payload = plot_series_payload(summary, baseline=baseline, groups=groups)
    payload["effect_from_year"] = effect_from_year
    for metric, rows in (differences or {}).items():
        if not rows:
            continue
        payload["panels"].append(
            {
                "group": f"change_{metric}",
                "kind": "difference",
                "traces": [
                    {
                        "metric": metric,
                        "label": f"{_metric_label(metric)} change vs baseline",
                        "years": [row["year"] for row in rows],
                        "mean": [row["mean"] for row in rows],
                        "q10": [row["q10"] for row in rows],
                        "q90": [row["q90"] for row in rows],
                        "is_difference": True,
                    }
                ],
            }
        )
    paths: list[str] = []
    if output_dir is not None:
        try:
            paths = render_plots(payload, output_dir, stem=stem)
        except ImportError:
            payload["render_error"] = (
                "matplotlib is not installed; numeric series are still returned."
            )
    payload["figure_paths"] = paths
    return payload
