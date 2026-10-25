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
        "region": summary.get("region"),
        "region_label": summary.get("region_label") or "all reefs (GBR-wide)",
        "panels": panels,
        "notes": [
            "Shaded bands are the 10-90% range across stochastic ensembles.",
            "Lines are the ensemble-mean of reef-mean values.",
        ],
    }


def render_plots(
    payload: dict[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "scenario",
) -> list[str]:
    """Write one PNG per panel. Returns paths. Requires matplotlib."""
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
    paths: list[str] = []
    for panel in payload.get("panels") or []:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for trace in panel["traces"]:
            years = trace["years"]
            mean = trace["mean"]
            q10 = trace["q10"]
            q90 = trace["q90"]
            (line,) = ax.plot(years, mean, label=trace["label"])
            ax.fill_between(years, q10, q90, color=line.get_color(), alpha=0.18)
        ax.set_xlabel("Year")
        ax.set_ylabel(panel["group"].replace("_", " "))
        title = panel["group"].replace("_", " ").title()
        ax.set_title(f"{title} — {payload.get('region_label', 'all reefs (GBR-wide)')}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = folder / f"{stem}_{panel['group']}.png"
        fig.savefig(path, dpi=120)
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
) -> dict[str, Any]:
    payload = plot_series_payload(summary, baseline=baseline, groups=groups)
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
