from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from coconet.api import run_coconet
from coconet.model import CoconetModel


def test_intervene_region_limits_box_interventions(make_tiny_config) -> None:
    # Both fixture reefs are in the Far-Northern (FN) region and inside the default box.
    model = CoconetModel(make_tiny_config())
    model.setup(init_output=False)
    assert model._reef_in_intervention_bounds(0)

    model.cfg.intervene_region = "FN"
    assert model._reef_in_intervention_bounds(0)

    model.cfg.intervene_region = "N"
    assert not model._reef_in_intervention_bounds(0)

    model.cfg.regional_shading_reduction = 0.5
    model.shade_regional_coral()
    assert np.all(model.regional_shading == 0.0)


def test_intervene_region_default_keeps_netlogo_preamble(make_tiny_config) -> None:
    cfg = make_tiny_config()
    run_coconet(cfg)
    assert "Region of interventions" not in Path(cfg.output_file).read_text()


def _spinup_config(make_tiny_config, out: Path, threads: int):
    return make_tiny_config(
        output_file=str(out),
        ensemble_runs=2,
        ensemble_threads=threads,
        start_year=1956,
        end_year=1957,
        save_year=1956,
        spinup_backtrack_years=1,
    )


@pytest.mark.integration
@pytest.mark.parametrize("threads", [1, 2])
def test_restored_spinup_matches_full_run(make_tiny_config, tmp_path: Path, threads: int) -> None:
    full = run_coconet(_spinup_config(make_tiny_config, tmp_path / "full.csv", threads))
    assert full.spinup_checkpoint is not None

    restored_cfg = _spinup_config(make_tiny_config, tmp_path / "restored.csv", threads)
    restored = run_coconet(restored_cfg, spinup_checkpoint=full.spinup_checkpoint)

    full_text = Path(full.output_file).read_text()
    assert any(line.startswith("2,") for line in full_text.splitlines()), "no replicate rows"
    assert Path(restored.output_file).read_text() == full_text
    assert restored.spinup_checkpoint is full.spinup_checkpoint
