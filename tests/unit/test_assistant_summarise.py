from __future__ import annotations

from pathlib import Path

from coconet_assistant.output import read_output_table
from coconet_assistant.plot import plot_series_payload
from coconet_assistant.summarise import compare_summaries, summarise_output

HEADER = (
    "Ensemble, Year, Reef_ID, Region, Shelf_position, Rezone_year, Priority, "
    "Longitude, Latitude, km_offshore, Reef_sites,"
    "C_sa, C_ta, C_mo, C_po, C_fa, C_tt, C_out_degree, DHW, "
    "bleach_sa, bleach_ta, bleach_mo, bleach_po, bleach_fa, bleach_tt, "
    "Maximum_cyclone_category, cyclone_sa, cyclone_ta, cyclone_mo, cyclone_po, cyclone_fa, cyclone_tt, "
    "predate_sa, predate_ta, predate_mo, predate_po, predate_fa, predate_tt, "
    "S_1, S_2, S_3, S_4, S_5, S_6, S_manta, Control_dives, Benthic_invert, Triggerfish, "
    "E_1, E_2, E_3, E_4, E_5, E_catch_kg,G_1, G_2, G_3, G_4, G_5, G_catch_kg"
)
COLUMNS = [part.strip() for part in HEADER.split(",")]


def _row(
    ensemble: int,
    year: int,
    reef_id: str,
    region: str,
    c_sa: float,
    c_tt: float,
    s_manta: float,
    emperor: float,
    trout: float,
) -> str:
    values = {name: 0 for name in COLUMNS}
    values.update(
        {
            "Ensemble": ensemble,
            "Year": year,
            "Reef_ID": reef_id,
            "Region": region,
            "Shelf_position": "I",
            "Rezone_year": 9999,
            "Priority": "N",
            "Longitude": 146.0,
            "Latitude": -15.0,
            "km_offshore": 20,
            "Reef_sites": 4,
            "C_sa": c_sa,
            "C_ta": 0.02,
            "C_mo": 0.01,
            "C_po": 0.01,
            "C_fa": 0.01,
            "C_tt": c_tt,
            "DHW": 4.0,
            "S_manta": s_manta,
            "Benthic_invert": 1.0,
            "Triggerfish": 0.5,
            "E_1": emperor,
            "G_1": trout,
        }
    )
    return ",".join(str(values[name]) for name in COLUMNS)


def write_output(path: Path, *, intervention: bool) -> Path:
    scale = 1.2 if intervention else 1.0
    lines = [
        "Climate scenario, 4.5",
        "",
        "Ensemble runs, 2",
        HEADER,
        _row(1, 2040, "A", "N", 0.10 * scale, 0.04 * scale, 2.0 / scale, 8.0, 3.0),
        _row(1, 2040, "B", "C", 0.08 * scale, 0.03 * scale, 2.4 / scale, 7.0, 2.5),
        _row(2, 2040, "A", "N", 0.12 * scale, 0.05 * scale, 1.6 / scale, 9.0, 3.5),
        _row(2, 2040, "B", "C", 0.09 * scale, 0.03 * scale, 2.0 / scale, 8.0, 3.0),
        _row(1, 2050, "A", "N", 0.11 * scale, 0.06 * scale, 1.8 / scale, 8.5, 3.2),
        _row(1, 2050, "B", "C", 0.07 * scale, 0.04 * scale, 2.2 / scale, 7.5, 2.8),
        _row(2, 2050, "A", "N", 0.13 * scale, 0.07 * scale, 1.4 / scale, 9.5, 3.6),
        _row(2, 2050, "B", "C", 0.10 * scale, 0.05 * scale, 1.8 / scale, 8.2, 3.1),
        # Spin-up row must be ignored.
        _row(0, 1980, "A", "N", 0.99, 0.99, 99.0, 99.0, 99.0),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_output_skips_preamble_and_spinup(tmp_path: Path) -> None:
    path = write_output(tmp_path / "out.csv", intervention=True)
    table = read_output_table(path)
    assert next(iter(table.columns)) == "Ensemble"
    assert 0 in set(table["Ensemble"])
    summary = summarise_output(str(path), scenario_name="demo")
    assert summary["n_ensembles"] == 2
    assert summary["end_year"] == 2050
    assert 1980 not in summary["years"]
    coral = summary["end_year_snapshot"]["coral_total"]
    assert coral["n_ensembles"] == 2
    assert coral["q10"] <= coral["mean"] <= coral["q90"]
    assert "Rubble" in summary["rubble_note"]
    assert any("coral" in fact.lower() for fact in summary["facts"])


def test_region_filter(tmp_path: Path) -> None:
    path = write_output(tmp_path / "out.csv", intervention=True)
    summary = summarise_output(str(path), region="N", scenario_name="north")
    assert summary["region"] == "N"
    assert summary["n_reef_rows"] == 4


def test_compare_and_plot_payload(tmp_path: Path) -> None:
    intervention = summarise_output(
        str(write_output(tmp_path / "int.csv", intervention=True)),
        scenario_name="seed+CoTS",
        scenario_config={"SSP": 4.5, "ensemble_runs": 2, "start_year": 1956, "end_year": 2050},
    )
    baseline = summarise_output(
        str(write_output(tmp_path / "base.csv", intervention=False)),
        scenario_name="baseline",
        scenario_config={"SSP": 4.5, "ensemble_runs": 2, "start_year": 1956, "end_year": 2050},
    )
    compared = compare_summaries(
        {"seed+CoTS": intervention, "baseline": baseline},
        baseline_name="baseline",
    )
    coral = next(row for row in compared["rows"] if row["metric"] == "coral_total")
    assert coral["values"]["seed+CoTS"]["difference_vs_baseline"] > 0
    assert compared["facts"]
    assert compared["suggestions"]
    payload = plot_series_payload(intervention, baseline=baseline)
    groups = {panel["group"] for panel in payload["panels"]}
    assert "coral_total" in groups
    assert "cots" in groups
    assert payload["panels"][0]["traces"][0]["years"]
