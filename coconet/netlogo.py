from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np


def nl_round(value: float) -> int:
    return math.floor(value + 0.5)


def nl_ceiling(value: float) -> int:
    return math.ceil(value)


def nl_median(a: float, b: float, c: float) -> float:
    # NetLogo's median of three numbers. Profiling showed np.median on a fresh
    # length-3 array dominated spawn/consume/cyclone hot paths; a tiny sort network
    # matches the same middle-element result for three scalars without allocation.
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
        if a > b:
            a, b = b, a
    return float(b)


def heading_from_dx_dy(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """NetLogo heading in degrees clockwise from north."""
    heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    return cast(np.ndarray, heading)


@dataclass(slots=True)
class NetLogoRng:
    seed_value: int = 1
    _rs: np.random.RandomState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rs = np.random.RandomState(self.seed_value)

    def seed(self, value: int) -> None:
        self.seed_value = int(value)
        self._rs.seed(self.seed_value)

    def random_float(self, upper: float = 1.0) -> float:
        return float(self._rs.random_sample() * upper)

    def random_int(self, upper: float) -> int:
        n = int(upper)
        if n != upper:
            n = n + 1 if upper > 0 else n - 1
        if n == 0:
            return 0
        if n > 0:
            return int(self._rs.randint(0, n))
        return -int(self._rs.randint(0, -n))

    def one_of(self, indices: np.ndarray) -> int | None:
        if indices.size == 0:
            return None
        idx = self._rs.randint(0, indices.size)
        return int(indices[idx])
