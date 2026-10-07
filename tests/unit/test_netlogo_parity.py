"""Unit tests for NetLogo parity fixes.

Tests verify Python implementation matches legacy NetLogo semantics by exercising
actual model code. All tests except nl_median must FAIL on base commit 41e8a74.
"""

import math
import numpy as np
import pytest
from unittest.mock import patch

from coconet.api import load_coconet_config
from coconet.model import CoconetModel
from coconet.netlogo import NetLogoRng, nl_median, nl_round


def test_random_int_with_non_integer_includes_ceiling():
    """NetLogo random 2.5 draws from {0,1,2}, not {0,1}."""
    rng = NetLogoRng(42)
    results = [rng.random_int(2.5) for _ in range(500)]
    result_set = set(results)
    assert result_set == {0, 1, 2}, f"Fix gives {{0,1,2}}; without fix: {result_set}"


def test_coral_site_sum_order(fixture_dir):
    """Coral summation order: fa+po in grow/spawn (lines 1468, 1852), po+fa in initialise-run (lines 1174, 1216).
    
    Tests all three methods with order-sensitive values.
    """
    cfg = load_coconet_config(
        scenario={
            "reefs_file": str(fixture_dir / "reefs_two.csv"),
            "coastline_file": str(fixture_dir / "coastline_clip.csv"),
            "output_file": "/tmp/test.csv",
            "ensemble_runs": 1,
            "start_year": 1956,
            "end_year": 1957,
            "save_year": 9999,
            "spinup_backtrack_years": 0,
        }
    )
    
    # Test initialise_run (both ensemble==0 and ensemble>0 use po+fa)
    model = CoconetModel(cfg)
    model.setup()
    
    model.C_i["sa"][:] = 1.0
    model.C_i["ta"][:] = 1e-16
    model.C_i["mo"][:] = 1e-16
    model.C_i["po"][:] = 1e-16
    model.C_i["fa"][:] = -1e-16
    model.C_i["tt"][:] = 0.0
    model.C_site_i[:] = 999.0
    
    model.ensemble = 1
    model.initialise_run()
    
    po_fa = (model.C["sa"] + model.C["ta"] + model.C["mo"]
             + model.C["po"] + model.C["fa"] + model.C["tt"])
    fa_po = (model.C["sa"] + model.C["ta"] + model.C["mo"]
             + model.C["fa"] + model.C["po"] + model.C["tt"])
    
    diff = np.abs(po_fa - fa_po).max()
    assert diff > 1e-17, f"Test needs values where order matters; diff={diff:.2e}"
    assert np.array_equal(model.C_site, po_fa), "initialise_run must use po+fa order"
    assert not np.array_equal(model.C_site, fa_po), "Must not match fa+po"
    
    # Test grow_corals uses fa+po order
    # Use sa=1.0 for order sensitivity, rates=0, R_site negative to keep C+R < 0.7
    model2 = CoconetModel(cfg)
    model2.setup()
    model2.rng.seed(42)
    
    sl = model2._sites_for_reef(0)
    
    model2.C["sa"][sl] = 1.0
    model2.C["ta"][sl] = 1e-16
    model2.C["mo"][sl] = 1e-16
    model2.C["fa"][sl] = -1e-16
    model2.C["po"][sl] = 1e-16
    model2.C["tt"][sl] = 0.0
    for g in ("sa", "ta", "mo", "fa", "po", "tt"):
        model2.rate[g][sl] = 0.0
    model2.C_site[sl] = 1.0
    model2.R_site[sl] = -0.35  # Negative to keep C+R=0.65 < 0.7
    
    model2.grow_corals(0)
    
    # After grow_corals, C values may have changed. Check if C_site matches one of the orders
    # using the actual post-grow_corals C values
    fa_po_grow = (model2.C["sa"][sl] + model2.C["ta"][sl] + model2.C["mo"][sl]
                  + model2.C["fa"][sl] + model2.C["po"][sl] + model2.C["tt"][sl])
    po_fa_grow = (model2.C["sa"][sl] + model2.C["ta"][sl] + model2.C["mo"][sl]
                  + model2.C["po"][sl] + model2.C["fa"][sl] + model2.C["tt"][sl])
    
    # C_site should equal one of the two orders
    matches_fa_po = np.array_equal(model2.C_site[sl], fa_po_grow)
    matches_po_fa = np.array_equal(model2.C_site[sl], po_fa_grow)
    assert matches_fa_po or matches_po_fa, \
        f"grow_corals C_site must match one order; C_site={model2.C_site[sl][0]:.20e}, fa_po={fa_po_grow[0]:.20e}, po_fa={po_fa_grow[0]:.20e}"
    # With the fix, it should match fa_po
    assert matches_fa_po, "grow_corals must use fa+po order"
    
    # Test spawn_corals uses fa+po order
    model3 = CoconetModel(cfg)
    model3.setup()
    model3.rng.seed(42)
    
    sl = model3._sites_for_reef(0)
    
    model3.C["sa"][sl] = 1.0
    model3.C["ta"][sl] = 1e-16
    model3.C["mo"][sl] = 1e-16
    model3.C["fa"][sl] = -1e-16
    model3.C["po"][sl] = 1e-16
    model3.C["tt"][sl] = 0.0
    model3.C_site[sl] = 1.0
    model3.R_site[sl] = -0.35
    
    for g in ("sa", "ta", "mo", "fa", "po", "tt"):
        model3.thermal[g][sl] = 50.0
    
    model3.spawn_corals(0)
    
    fa_po_spawn = (model3.C["sa"][sl] + model3.C["ta"][sl] + model3.C["mo"][sl]
                   + model3.C["fa"][sl] + model3.C["po"][sl] + model3.C["tt"][sl])
    po_fa_spawn = (model3.C["sa"][sl] + model3.C["ta"][sl] + model3.C["mo"][sl]
                   + model3.C["po"][sl] + model3.C["fa"][sl] + model3.C["tt"][sl])
    
    matches_fa_po_spawn = np.array_equal(model3.C_site[sl], fa_po_spawn)
    matches_po_fa_spawn = np.array_equal(model3.C_site[sl], po_fa_spawn)
    assert matches_fa_po_spawn or matches_po_fa_spawn, \
        f"spawn_corals C_site must match one order"
    assert matches_fa_po_spawn, "spawn_corals must use fa+po order"


def test_nl_median_when_hi_less_than_lo():
    """nl_median(10, 5, 1) returns 5 (middle), not 1 like np.clip."""
    assert nl_median(10.0, 5.0, 1.0) == 5.0
    assert np.clip(5.0, 10.0, 1.0) == 1.0


def test_c_out_degree_uses_instance_accumulator(fixture_dir):
    """spawn_corals uses self._recruits_total, not local var + return."""
    cfg = load_coconet_config(
        scenario={
            "reefs_file": str(fixture_dir / "reefs_two.csv"),
            "coastline_file": str(fixture_dir / "coastline_clip.csv"),
            "output_file": "/tmp/test.csv",
            "ensemble_runs": 0,
            "start_year": 1956,
            "end_year": 1957,
            "save_year": 9999,
            "spinup_backtrack_years": 0,
        }
    )
    model = CoconetModel(cfg)
    model.setup()
    model.rng.seed(42)
    
    for g in ("sa", "ta", "mo", "fa", "po", "tt"):
        model.C[g][:] = 0.15
        model.rate[g][:] = 0.06
    
    # Set thermal thresholds to enable recruitment
    for g in ("sa", "ta", "mo", "fa", "po", "tt"):
        model.thermal[g][:] = 50.0
    
    model.spawn_corals(0)
    
    has_attr = hasattr(model, '_recruits_total')
    assert has_attr, "Without fix: no _recruits_total attribute"
    assert model.C_out_degree[0] == model._recruits_total, \
        "C_out_degree must equal _recruits_total accumulator"
    # Note: With this test fixture's connectivity (con1=0.012, dis1=622km), spawning produces
    # zero recruits. The key test is that _recruits_total exists and equals C_out_degree.


def test_rubble_retention_uses_reef_who(fixture_dir):
    """grow_corals caps R_site at (reef_who % 11)/20 (legacy line 1449)."""
    cfg = load_coconet_config(
        scenario={
            "reefs_file": str(fixture_dir / "reefs_two.csv"),
            "coastline_file": str(fixture_dir / "coastline_clip.csv"),
            "output_file": "/tmp/test.csv",
            "ensemble_runs": 0,
            "start_year": 1956,
            "end_year": 1957,
            "save_year": 9999,
            "spinup_backtrack_years": 0,
        }
    )
    model = CoconetModel(cfg)
    model.setup()
    
    reef_idx = 1
    sl = model._sites_for_reef(reef_idx)
    
    reef_cap = (model.reef_who[reef_idx] % 11) / 20.0
    site_cap_first = (model.site_who[sl][0] % 11) / 20.0
    
    assert reef_cap != site_cap_first, f"Test requires different caps: reef={reef_cap} site={site_cap_first}"
    
    model.R_site[sl] = 0.99
    model.C_site[sl] = 0.05
    model.rng.seed(42)
    model.grow_corals(reef_idx)
    
    actual_max = model.R_site[sl].max()
    assert np.allclose(actual_max, reef_cap, atol=1e-6), \
        f"With fix: R_site={actual_max:.3f} matches reef_cap={reef_cap:.3f}; without fix would use site_cap={site_cap_first:.3f}"


def test_cots_spawn_gate_uses_degrees(fixture_dir):
    """CoTS _cots_spawn_gate uses math.radians (legacy line 1067, sin takes degrees)."""
    cfg = load_coconet_config(
        scenario={
            "reefs_file": str(fixture_dir / "reefs_two.csv"),
            "coastline_file": str(fixture_dir / "coastline_clip.csv"),
            "output_file": "/tmp/test.csv",
            "ensemble_runs": 0,
            "start_year": 2012,
            "end_year": 2013,
            "save_year": 9999,
            "spinup_backtrack_years": 0,
        }
    )
    model = CoconetModel(cfg)
    model.setup()
    model.rng.seed(42)
    model.year = 2012
    model.S_phase = 0.0
    
    # Patch random_float to return 0.0 for deterministic test
    with patch.object(type(model.rng), 'random_float', return_value=0.0):
        gate = model._cots_spawn_gate()
    
    # Expected with degrees: sin(radians(-2*pi/8)) ≈ sin(-45°)
    expected_degrees = 0.5 * (1 + math.sin(math.radians(2 * 3.1416 * (-2) / 16)))
    expected_radians = 0.5 * (1 + math.sin(2 * 3.1416 * (-2) / 16))
    
    assert abs(gate - expected_degrees) < 1e-10, \
        f"With fix: gate={gate:.6f} == degrees={expected_degrees:.6f}"
    assert abs(expected_degrees - expected_radians) > 0.3, \
        f"degrees={expected_degrees:.3f} vs radians={expected_radians:.3f}"


def test_cots_spawn_gate_controls_spawning(fixture_dir):
    """CoTS spawning loop uses _cots_spawn_gate to gate spawn_cots calls."""
    cfg = load_coconet_config(
        scenario={
            "reefs_file": str(fixture_dir / "reefs_two.csv"),
            "coastline_file": str(fixture_dir / "coastline_clip.csv"),
            "output_file": "/tmp/test.csv",
            "ensemble_runs": 0,
            "start_year": 2012,
            "end_year": 2013,
            "save_year": 9999,
            "spinup_backtrack_years": 0,
        }
    )
    
    # Test with gate returning 1.0 (always spawn)
    model = CoconetModel(cfg)
    model.setup()
    model.ensemble = 1
    model.initialise_run()
    
    spawn_cots_called = []
    orig_spawn_cots = model.spawn_cots
    def track_spawn_cots(reef_idx):
        spawn_cots_called.append(True)
        orig_spawn_cots(reef_idx)
    model.spawn_cots = track_spawn_cots
    
    with patch.object(model, '_cots_spawn_gate', return_value=1.0):
        model._run_ensemble_year_steps()
    
    assert len(spawn_cots_called) > 0, "With gate=1.0, spawn_cots must be called"
    
    # Test with gate returning 0.0 (never spawn)
    model2 = CoconetModel(cfg)
    model2.setup()
    model2.ensemble = 1
    model2.initialise_run()
    
    spawn_cots_called2 = []
    orig_spawn_cots2 = model2.spawn_cots
    def track_spawn_cots2(reef_idx):
        spawn_cots_called2.append(True)
        orig_spawn_cots2(reef_idx)
    model2.spawn_cots = track_spawn_cots2
    
    with patch.object(model2, '_cots_spawn_gate', return_value=0.0):
        model2._run_ensemble_year_steps()
    
    assert len(spawn_cots_called2) == 0, "With gate=0.0, spawn_cots must not be called"


def test_nl_round_half_up():
    """nl_round rounds half up: -2.5→-2, not -3."""
    assert nl_round(-2.5) == -2, "Without fix: returns -3"
    assert nl_round(-0.5) == 0, "Without fix: returns -1"
    assert nl_round(2.5) == 3


def test_reef_iteration_order_varies_with_seed(fixture_dir):
    """Reef loops use rng.permutation, not range (legacy ask reefs at lines 945+)."""
    cfg = load_coconet_config(
        scenario={
            "reefs_file": str(fixture_dir / "reefs_two.csv"),
            "coastline_file": str(fixture_dir / "coastline_clip.csv"),
            "output_file": "/tmp/test.csv",
            "ensemble_runs": 0,
            "start_year": 1956,
            "end_year": 1957,
            "save_year": 9999,
            "spinup_backtrack_years": 0,
        }
    )
    
    grow_orders_seen = set()
    spawn_fish_orders_seen = set()
    
    for seed in range(20):
        model = CoconetModel(cfg)
        model.setup()
        model.rng.seed(seed)
        model.ensemble = 1
        model.initialise_run()
        
        grow_order = []
        orig_grow = model.grow_corals
        def track_grow(reef_idx):
            grow_order.append(reef_idx)
            orig_grow(reef_idx)
        model.grow_corals = track_grow
        
        spawn_fish_order = []
        orig_spawn_fish = model.spawn_fish
        def track_spawn_fish(reef_idx):
            spawn_fish_order.append(reef_idx)
            orig_spawn_fish(reef_idx)
        model.spawn_fish = track_spawn_fish
        
        model._run_ensemble_year_steps()
        
        if len(grow_order) >= 2:
            grow_orders_seen.add(tuple(grow_order[:2]))
        if len(spawn_fish_order) >= 2:
            spawn_fish_orders_seen.add(tuple(spawn_fish_order[:2]))
    
    # Must see [1, 0] order at least once (base would always show [0, 1])
    assert (1, 0) in grow_orders_seen, \
        f"grow_corals must use permutation; without fix always [0,1]; saw {grow_orders_seen}"
    assert (1, 0) in spawn_fish_orders_seen, \
        f"spawn_fish must use permutation; without fix always [0,1]; saw {spawn_fish_orders_seen}"
