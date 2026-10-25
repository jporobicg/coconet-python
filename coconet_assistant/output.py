"""Parse the CoCoNet output.csv preamble-plus-table layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CORAL_COLUMNS = ("C_sa", "C_ta", "C_mo", "C_po", "C_fa", "C_tt")
EMPEROR_COLUMNS = ("E_1", "E_2", "E_3", "E_4", "E_5")
TROUT_COLUMNS = ("G_1", "G_2", "G_3", "G_4", "G_5")


def find_table_header_row(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.lstrip("\ufeff").strip()
            if stripped.startswith("Ensemble,") and "Reef_ID" in stripped:
                return index
    raise ValueError(
        f"No output table header (line starting with 'Ensemble,' and containing "
        f"'Reef_ID') in {path}"
    )


def read_output_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load reef-year rows, skipping the legacy metadata preamble.

    ``columns`` limits the read to those columns (plus Ensemble, Year, Region); a
    full GBR output has millions of rows, so this saves time and memory.
    """
    output_path = Path(path)
    if not output_path.is_file():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    header_row = find_table_header_row(output_path)
    usecols = None
    if columns is not None:
        wanted = {"Ensemble", "Year", "Region", *columns}
        usecols = lambda name: str(name).strip() in wanted  # noqa: E731
    frame = pd.read_csv(output_path, skiprows=header_row, usecols=usecols)
    frame.columns = [str(col).strip() for col in frame.columns]
    if "Ensemble" not in frame.columns or "Year" not in frame.columns:
        raise ValueError(f"Output table is missing Ensemble/Year columns: {output_path}")

    frame["Ensemble"] = pd.to_numeric(frame["Ensemble"], errors="coerce")
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    frame = frame.dropna(subset=["Ensemble", "Year"])
    frame["Ensemble"] = frame["Ensemble"].astype(int)
    frame["Year"] = frame["Year"].astype(int)
    if "Region" in frame.columns:
        frame["Region"] = frame["Region"].astype(str).str.strip()
    return frame


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add coral_total, emperor, and trout columns when the source columns exist."""
    out = frame.copy()
    present_coral = [c for c in CORAL_COLUMNS if c in out.columns]
    if present_coral:
        out["coral_total"] = out[present_coral].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    present_e = [c for c in EMPEROR_COLUMNS if c in out.columns]
    if present_e:
        out["emperor"] = out[present_e].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    present_g = [c for c in TROUT_COLUMNS if c in out.columns]
    if present_g:
        out["trout"] = out[present_g].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return out


def filter_replicates(
    frame: pd.DataFrame,
    *,
    region: str | None = None,
) -> pd.DataFrame:
    """Keep stochastic ensembles (1..N) and optionally one Region code."""
    out = frame[frame["Ensemble"] > 0].copy()
    if region:
        code = region.strip()
        if "Region" not in out.columns:
            raise ValueError("Output table has no Region column to filter on.")
        out = out[out["Region"] == code]
    return out
