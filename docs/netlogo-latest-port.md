# NetLogo Latest Port Documentation

This document describes the behavioral changes ported from the latest NetLogo model (`CoCoNet_latest.nlogo`) to the Python implementation, building upon the nine existing NetLogo-translation fixes.

## Background

The Python port was originally based on the legacy NetLogo model (`CoCoNet V3_rubble.nlogo`). This update ports behavioral changes from the latest NetLogo model while preserving the nine translation fixes that ensure faithful NetLogo semantics:

1. `random_int` non-integer bounds (ceiling inclusion)
2. Coral site summation order (fa+po in grow/spawn)
3. `nl_median` three-way ordering (not np.clip)
4. C_out_degree single accumulator
5. Rubble retention by reef who (not site who)
6. Sin in degrees (with radians conversion)
7. `nl_round` half-up rounding
8. Reef permutation (RNG-based ask order)
9. initialise_run C_site order (po+fa)

## Behavioral Changes Ported

### 1. Coral Allee Effect (spawn_corals)

**NetLogo procedure:** `spawn-corals` (~L1756-1769)

Every coral source term is now scaled by an Allee fertilization factor:

```
exp(-1 * (C_allee / (small + C_group))^0.5)
```

This applies to all six coral groups (sa, ta, mo, po, fa, tt), including the hybrid and thermally-tolerant formulas. Low local cover sharply reduces larval output.

**Python location:** `coconet/model.py`, `spawn_corals()` method (~L1238-1260)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_coral_allee_effect`

### 2. CoTS Allee Effect (spawn_cots)

**NetLogo procedure:** `spawn-cots` (~L1962)

Replaced hard spawning threshold gate with continuous Allee effect:

**Old:** `if (S_2+...+S_6) > S_spawning_threshold then add source`  
**New:** `(age-weighted source) * exp(-1 * (S_allee / (small + ΣS))^0.5)`

Sparse sites contribute reduced amounts instead of zero; dense sites are no longer capped by the old step function.

**Python location:** `coconet/model.py`, `spawn_cots()` method (~L1401-1415)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_cots_allee_continuous`

### 3. Fish Juvenile Coral Dependence (grow_fish)

**NetLogo procedure:** `grow-fish` (~L1358-1365)

Strengthened coral dependence for fish juveniles:

**Old:** `E_1 = E_0 * C_reef^0.2`, `G_1 = G_0 * C_reef^0.2`  
**New:** `E_1 = E_0 * C_reef^0.6`, `G_1 = G_0 * C_reef^0.6`

**Python location:** `coconet/model.py`, `grow_fish()` method (~L886, ~L901)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_fish_coral_exponent_0_6`

### 4. Fish Latitude Larval Factor Floor (spawn_fish)

**NetLogo procedure:** `spawn-fish` (~L1687, ~L1714)

Wrapped latitude-dependent larval survival in max(0, ...):

**Formula:** `max(0, 1 - 0.004*(y+25)^2)`

Prevents negative values at far southern/northern latitudes.

**Python location:** `coconet/model.py`, `spawn_fish()` method (~L1187, ~L1199)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_fish_latitude_floor_at_zero`

### 5. CoTS Predation Floors (consume_cots)

**NetLogo procedure:** `consume-cots` (~L1624-1632)

Changed post-predation floors for adult/subadult CoTS:

**Old:** `max(1, ...)` for S_2...S_6  
**New:** `max(0, ...)` for S_2...S_6 (S_1 floor stays at 10)

Adult CoTS can now be fully removed by predation.

**Python location:** `coconet/model.py`, `consume_cots()` method (~L1172)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_cots_predation_floors_updated`

### 6. Default Parameters (setup)

**NetLogo procedure:** `setup` (~L519-567)

Updated hard-coded interface fallbacks (used when no parameter file supplied):

| Parameter | Old | New |
|-----------|-----|-----|
| dhw_scale | 30 | 50 |
| E_recruit | 0.0028 | 0.005 |
| E_mort | 0.026 | 0.025 |
| G_recruit | 4 | 7 |
| C_recruit | 0.05 | 0.08 |
| S_spawning_failure | 0.7 | 0.5 |
| S_recruit | 300000 | 200000 |
| S_prefer | 0.55 | 0.4 |

**New parameters:**
- C_allee: 0.02 (coral Allee threshold)
- S_allee: 3 (CoTS Allee threshold)

**Note:** S_spawning_threshold removed from active use (kept for backward compatibility)

**Python location:** `coconut/model.py`, `__init__()` method (~L101-140)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_default_parameters_match_new_netlogo`

### 7. Historical Bleaching Events (bleaching)

**NetLogo procedure:** `bleaching` (~L2287-2350)

Updated historical DHW events and formulas:

**New events:**
- 2003: DHW 6.0, latitude (-22, -21)
- 2022: DHW 8.0, latitude (-16, -15)  
- 2024: DHW 10.5, latitude (-22, -21)

**Updated events:**
- 1998: 8 → 4.5 DHW, latitude (-21, -20) → (-22, -21)
- 2002: 10 → 12.0 DHW
- 2016: 9 → 11.5 DHW
- 2017: 8 → 11.5 DHW, latitude (-17, -16) → (-16, -15)
- 2020: 6 → 10.5 DHW

**Python location:** `coconet/model.py`, `bleaching()` method (~L1700-1718)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_historical_bleaching_events_updated`

### 8. SSP 7.0 Bleaching Formula (bleaching)

**NetLogo procedure:** `bleaching` (~L2350)

Updated SSP 7.0 projection formula:

**Old:** `(8 - random(16)) + 0.0039 * (year - 2010)^2 + 2`  
**New:** `(8 - random(16)) + 0.0037 * (year - 2010)^2 + 4`

**Python location:** `coconet/model.py`, `bleaching()` method (~L1722)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_ssp70_bleaching_formula_updated`

### 9. Bleaching Radius Restructuring (bleaching)

**NetLogo procedure:** `bleaching` (~L2341-2350)

Combined radius calculation:

**Old:** `bleaching_radius = dhw_scale * dhw_max * (...)`, then divide by `per_km` in radial decline  
**New:** `bleaching_radius = (dhw_scale * per_km) * dhw_max * (...)`

Algebraically equivalent but cleaner; combined with dhw_scale 30→50 enlarges events.

**Python location:** `coconet/model.py`, `bleaching()` method (~L1730-1738)

### 10. Cyclone Mortality Guards (cyclone_mortality)

**NetLogo procedure:** `cyclone-mortality` (~L2650-2660)

Added division-by-zero guards and same radius restructuring as bleaching:

**Old:** `rate_i / rate`  
**New:** `rate_i / (rate + small)`

Also updated radial decline to match bleaching pattern (removed extra `/per_km`).

**Python location:** `coconet/model.py`, `cyclone_mortality()` method (~L1847-1873)

### 11. Perfect Intervention Combinations (go)

**NetLogo procedure:** `go` (~L1066-1102)

**Removed:** "ShadingPlusControl"

**Added four new combinations:**
- "Control-plus-replenishment"
- "Control-plus-shading"
- "Replenishment-plus-shading"
- "Control-plus-replenishment-plus-shading"

**Python location:** `coconet/model.py`, `_apply_perfect_intervention()` method (~L2019-2069)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_perfect_intervention_combos_updated`

## What Was NOT Changed

The following remain identical to maintain the nine existing translation fixes:

- `grow-corals`: Competition logic, rubble retention by reef who, R_site normalization
- `consume-corals`: Median-based rubble addition
- Coral site summation order (fa+po in grow/spawn, po+fa in initialise_run)
- C_out_degree single accumulator
- nl_median three-way ordering
- sin() degree conversion
- random_int ceiling inclusion
- nl_round half-up rounding
- Reef RNG permutation order

## Testing

All behavioral changes include focused unit tests in:
- `tests/unit/test_netlogo_latest_port.py` (12 tests total)

All existing NetLogo parity tests continue to pass:
- `tests/unit/test_netlogo_parity.py` (9 tests)

Full unit test suite: 48 tests passing

## RNG Stream Fidelity Fixes

Three additional fixes ensure bit-identical RNG stream alignment with NetLogo (required for deterministic parity when climate events are disabled):

### 1. Bleaching Discarded Rand Draw

**NetLogo location:** Line 2351 in `bleaching`

NetLogo draws `rand = (0.5 + random-float 0.5) * radial_decline` once per affected reef in the `ask reefs in-radius bleaching_radius` loop, but this value is immediately overwritten by `rand = random-float(1.0)` inside the `ask sites-here` loop (line 2357). Python now includes this discarded draw to maintain RNG stream alignment.

**Python location:** `coconet/model.py`, `bleaching()` method (~L1769)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_bleaching_rng_discarded_draw_per_reef`

### 2. Climate Procedures RNG-Permuted Iteration

**NetLogo location:** Lines 2349, 2651 (`ask reefs in-radius ...`)

NetLogo's `ask reefs in-radius bleaching_radius` and `ask reefs in-radius cyclone_radius` iterate in random order. Python was iterating in sorted order. Now uses `self.rng._rs.permutation(affected)` to match NetLogo's random ask order.

**Python location:** 
- `coconet/model.py`, `bleaching()` method (~L1765)
- `coconet/model.py`, `cyclone_mortality()` method (~L1891)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_climate_procedures_use_permuted_reef_iteration`

### 3. Empty Latitude Band Warnings

**NetLogo behavior:** `ask one-of reefs with [y > lat1 and y < lat2]` errors when no reefs match the latitude band.

Python silently skips historical events with empty latitude bands but now emits `warnings.warn` messages identifying the year and band.

**Python location:**
- `coconet/model.py`, `bleaching()` method (~L1717)
- `coconet/model.py`, `cyclone()` method (~L1858)

**Test:** `tests/unit/test_netlogo_latest_port.py::test_empty_latitude_band_warning`

## Testing

## References

- NEW NetLogo model: `CoCoNet_latest.nlogo`
- Legacy model: `legacy/CoCoNet V3_rubble.nlogo`
- Summary: `LATEST_VS_LEGACY.md`
- Code diff: `latest_vs_legacy_code.diff`
