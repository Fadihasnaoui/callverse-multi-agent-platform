"""Small sampling helpers."""

from __future__ import annotations

import math
from enum import Enum
from typing import Mapping, TypeVar

import numpy as np

E = TypeVar("E", bound=Enum)


def lognormal_from_mean_cv(rng: np.random.Generator, mean: float, cv: float) -> float:
    """Sample a lognormal with the given arithmetic mean and coefficient of variation."""
    if cv <= 0:
        return float(mean)
    sigma2 = math.log(1.0 + cv * cv)
    mu = math.log(mean) - sigma2 / 2.0
    return float(rng.lognormal(mu, math.sqrt(sigma2)))


def categorical(rng: np.random.Generator, probs: Mapping[E, float]) -> E:
    """Sample a key of ``probs`` (iteration order of the mapping is the category order)."""
    keys = list(probs)
    weights = np.array([probs[k] for k in keys], dtype=float)
    weights = weights / weights.sum()
    return keys[int(rng.choice(len(keys), p=weights))]
