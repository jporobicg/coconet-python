from __future__ import annotations

import numpy as np
import pytest
from coconet.netlogo import NetLogoRng, heading_from_dx_dy, nl_median, nl_round


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.4, 1),
        (1.5, 2),
        (-1.4, -1),
        (-1.5, -1),
        (-2.5, -2),
    ],
)
def test_nl_round(value: float, expected: int) -> None:
    assert nl_round(value) == expected


def test_nl_median_ordering_network_matches_sorted_middle() -> None:
    assert nl_median(3.0, 1.0, 2.0) == 2.0 == float(np.median(np.array([3.0, 1.0, 2.0])))


def test_heading_from_dx_dy_north_and_east() -> None:
    dx = np.array([0.0, 1.0])
    dy = np.array([1.0, 0.0])
    h = heading_from_dx_dy(dx, dy)
    assert np.isclose(h[0], 0.0)
    assert np.isclose(h[1], 90.0)


def test_netlogo_rng_deterministic() -> None:
    a = NetLogoRng(42)
    b = NetLogoRng(42)
    assert [a.random_float(1.0) for _ in range(5)] == [b.random_float(1.0) for _ in range(5)]


def test_netlogo_rng_one_of_empty() -> None:
    assert NetLogoRng(1).one_of(np.array([], dtype=np.int32)) is None
