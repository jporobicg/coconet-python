# NetLogo Parity Fixes

This document describes the fixes applied to make the Python port faithfully reproduce the legacy NetLogo model behavior.

## Summary

Nine port bugs were fixed to ensure the Python implementation matches the semantics of the legacy NetLogo model (`legacy/CoCoNet V3_rubble*.nlogo`):

1. **NetLogo `random` with non-integer bounds**
2. **Coral summation order**
3. **Median vs clip semantics**
4. **C_out_degree accumulation**
5. **Rubble retention uses reef who**
6. **Sin uses degrees**
7. **nl_round semantics**
8. **Random iteration order for ask**
9. **initialise-run C_site recomputation**

---

## Fix 1: NetLogo `random` with non-integer bounds

**NetLogo behavior:**  
`random N` with non-integer N draws from `0..int(N)` inclusive of `int(N)`.  
Example: `random 2.5` gives `{0, 1, 2}` with mean ~1.0.

**Python bug:**  
`NetLogoRng.random_int` used `floor(N)` as an exclusive bound, giving `{0, 1}` instead.

**Fix:**  
```python
def random_int(self, upper: float) -> int:
    n = int(upper)
    if n != upper:
        n = n + 1 if upper > 0 else n - 1
    if n == 0:
        return 0
    if n > 0:
        return int(self._rs.randint(0, n))
    return -int(self._rs.randint(0, -n))
```

**Files changed:** `coconet/netlogo.py`

**Impact:** Affects fish recruitment draws and CoTS spawning when bounds are non-integer.

---

## Fix 2: Coral summation order

**NetLogo order:**  
- grow-corals (lines 1468, 1483, 1492): `C_site = C_sa + C_ta + C_mo + C_fa + C_po + C_tt`
- spawn-corals (lines 1852, 1863): `C_site = C_sa + C_ta + C_mo + C_fa + C_po + C_tt`

**Python bug:**  
Used `C_sa + C_ta + C_mo + C_po + C_fa + C_tt`

**Fix:**  
Changed summation sites in `grow_corals` and `spawn_corals` to match NetLogo order.

**Files changed:** `coconet/model.py` (grow_corals lines ~1048, ~1069; spawn_corals lines ~1288, ~1301)

**Impact:** Floating-point associativity causes 1-ulp differences that compound through thresholds.

---

## Fix 3: Median vs clip semantics

**NetLogo behavior:**  
`median (list lo x hi)` returns the middle value of the three.  
When `hi < lo`, returns the actual median, not `hi`.

**Python bug:**  
Used `np.clip(x, lo, hi)` which returns `hi` when `hi < lo`.

**Fix:**  
```python
b = np.array([nl_median(100.0, bb, hh) for bb, hh in zip(b, self.B_max * (c_site + r_site))])
t = np.array([nl_median(1.0, tt, hh) for tt, hh in zip(t, self.T_max * c_site)])
```

**Files changed:** `coconet/model.py` (lines ~846, ~857)

**Impact:** Latent in tested configuration (condition never met), but would matter when `C+R < 0.01` or `C_site < 0.005`.

---

## Fix 4: C_out_degree accumulation

**NetLogo behavior:**  
Maintains one running sum across both coral kernels.

**Python bug:**  
Summed per kernel and then added: `recruits_total += kernel1_result + kernel2_result`

**Fix:**  
Use a single instance variable `self._recruits_total` accumulated across both kernel calls:
```python
self._recruits_total = 0.0
self._spawn_coral_kernel(...)  # accumulates into self._recruits_total
self._spawn_coral_kernel(...)  # continues accumulating
self.C_out_degree[reef_idx] = self._recruits_total
```

**Files changed:** `coconet/model.py` (lines ~1247, ~1262, ~1309, ~1389)

**Impact:** Floating-point associativity difference, latent when only one kernel contributes.

---

## Fix 5: Rubble retention uses reef who

**NetLogo behavior (grow-corals, line 1449):**  
In `grow-corals`, the procedure runs in reef context, so `who` refers to the reef's who:  
```netlogo
let rubble_retention ( remainder who 11 ) / 20
```

**Python bug:**  
Used site who: `rubble_retention = (site_who % 11) / 20.0`

**Fix:**  
```python
rubble_retention = (self.reef_who[reef_idx] % 11) / 20.0
```

**Files changed:** `coconet/model.py` (line ~1015)

**Impact:** Determines maximum rubble cover per reef.

---

## Fix 6: Sin uses degrees

**NetLogo behavior (line 1067):**  
`sin` takes degrees as input:
```netlogo
sin (2 * 3.1416 * ( 2010 + S_phase + random-float 4 - year ) / 16 )
```

**Note:** The legacy NetLogo passes `2*3.1416*(...)/16` to `sin` (which expects degrees), though the expression appears to be intended as radians. This is an upstream modelling question. The Python port now faithfully reproduces NetLogo as written.

**Python bug:**  
Used `math.sin(...)` which takes radians.

**Fix:**  
```python
math.sin(math.radians(2 * 3.1416 * (2010 + self.S_phase + self.rng.random_float(4) - self.year) / 16))
```

**Files changed:** `coconet/model.py` (line ~466)

**Impact:** CoTS spawning success gate has correct period and shape matching NetLogo. Major impact in full model when CoTS spawning is active.

---

## Fix 7: nl_round semantics

**NetLogo behavior:**  
`round` uses Java Math.round which rounds half up (toward +∞):  
- `round 2.5` = 3  
- `round -2.5` = -2 (not -3)

**Python bug:**  
Used `math.ceil(value - 0.5)` for negative values, giving `round(-2.5) = -3`.

**Fix:**  
```python
def nl_round(value: float) -> int:
    return math.floor(value + 0.5)
```

**Files changed:** `coconet/netlogo.py`

**Impact:** Minor, affects negative half-integers in population calculations.

---

## Fix 8: Random iteration order for ask

**NetLogo behavior:**  
`ask reefs` and `ask sites` visit agents in a new random order each time.

**Python bug:**  
Iterated in fixed index order: `for reef_idx in range(self.number_of_reefs)`

**Fix:**  
```python
for reef_idx in self.rng._rs.permutation(self.number_of_reefs):
    reef_idx = int(reef_idx)
    # ... process reef
```

**Files changed:** `coconet/model.py` (lines ~450, ~461, ~470, ~473)

**Impact:** Order matters in `spawn_fish` where natal recruits overwrite, then non-natal recruits add. Fixed order gives systematic bias; random order gives each reef ~50% chance to receive other reefs' recruits. Latent in tested configuration (fish extinct by 1908), but systematic in larger/fished-less setups.

---

## Fix 9: initialise-run C_site recomputation

**NetLogo behavior (initialise-run, lines 1174, 1216):**  
`initialise-run` recomputes `C_site` from the saved coral covers:
```netlogo
set C_site C_sa + C_ta + C_mo + C_po + C_fa + C_tt
```

Note: initialise-run uses po+fa order, while grow-corals and spawn-corals use fa+po order.

**Python bug:**  
Copied `C_site_i` directly: `self.C_site[:] = self.C_site_i`

**Fix:**  
```python
self.C_site[:] = (
    self.C["sa"]
    + self.C["ta"]
    + self.C["mo"]
    + self.C["po"]
    + self.C["fa"]
    + self.C["tt"]
)
```

**Files changed:** `coconet/model.py` (line ~793)

**Impact:** 1-ulp difference at the start of each simulation ensemble.

---

## Testing

Each fix has focused unit tests in `tests/unit/test_netlogo_parity.py` verifying NetLogo semantics:
- `test_random_int_with_non_integer_includes_ceiling`: RNG bounds
- `test_coral_site_sum_order`: Coral summation order (initialise-run po+fa; grow/spawn fa+po verified by inspection)
- `test_nl_median_when_hi_less_than_lo`: Median vs clip when hi<lo
- `test_c_out_degree_uses_instance_accumulator`: Single accumulator across kernels
- `test_rubble_retention_uses_reef_who`: Rubble cap uses reef who, not site who
- `test_cots_spawn_gate_uses_degrees`: Sin takes degrees via math.radians
- `test_cots_spawn_gate_controls_spawning`: Gate controls spawn_cots calls
- `test_nl_round_half_up`: Round half-up semantics
- `test_reef_iteration_order_varies_with_seed`: Permutation order varies per seed

All existing unit and integration tests pass with these fixes.

---

## References

- **NetLogo source:** `legacy/CoCoNet V3_rubble.nlogo` — reference implementation
