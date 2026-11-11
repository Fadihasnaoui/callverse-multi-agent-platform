"""Customer patience distributions (selectable; all parameters in ``PatienceConfig``).

For a configured mean ``m`` (request type x channel [x recontact multiplier]):

* exponential (B1):  X ~ Exp(mean = m)                               -- mode at 0
* weibull (B1.5):    X = f + s * W,  W ~ Weibull(k),  s = (m - f) / Gamma(1 + 1/k)
* lognormal:         X = f + Y,  Y ~ LogNormal(mean = m - f, cv)

All three have E[X] = m, so moving away from the exponential does not make
customers more patient on average; it removes the probability spike near zero.
Each sample uses exactly one draw from the caller's (per-contact) RNG stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from callverse_workforce.config.models import PatienceConfig


@dataclass(frozen=True)
class PatienceDistribution:
    model: str
    mean: float
    floor: float
    shape: float | None = None  # weibull k
    scale: float | None = None  # weibull s
    mu: float | None = None  # lognormal log-mean of the positive component
    sigma: float | None = None  # lognormal log-sd of the positive component

    @classmethod
    def from_config(cls, mean: float, cfg: PatienceConfig) -> PatienceDistribution:
        if cfg.model == "exponential":
            return cls("exponential", mean, 0.0)
        if cfg.model == "weibull":
            f, k = cfg.weibull.floor_minutes, cfg.weibull.shape
            return cls("weibull", mean, f, shape=k, scale=(mean - f) / math.gamma(1.0 + 1.0 / k))
        f, cv = cfg.lognormal.floor_minutes, cfg.lognormal.cv
        sigma = math.sqrt(math.log(1.0 + cv * cv))
        return cls("lognormal", mean, f, mu=math.log(mean - f) - sigma * sigma / 2.0, sigma=sigma)

    def sample(self, rng: np.random.Generator) -> float:
        if self.model == "exponential":
            return float(rng.exponential(self.mean))
        if self.model == "weibull":
            return self.floor + self.scale * float(rng.weibull(self.shape))  # type: ignore[operator]
        return self.floor + float(rng.lognormal(self.mu, self.sigma))  # type: ignore[arg-type]

    def cdf(self, x: float) -> float:
        if x <= self.floor:
            return 0.0
        y = x - self.floor
        if self.model == "exponential":
            return 1.0 - math.exp(-y / self.mean)
        if self.model == "weibull":
            return 1.0 - math.exp(-((y / self.scale) ** self.shape))  # type: ignore[operator]
        return 0.5 * (1.0 + math.erf((math.log(y) - self.mu) / (self.sigma * math.sqrt(2.0))))  # type: ignore[operator]

    def quantile(self, p: float) -> float:
        if self.model == "exponential":
            return -self.mean * math.log(1.0 - p)
        if self.model == "weibull":
            return self.floor + self.scale * (-math.log(1.0 - p)) ** (1.0 / self.shape)  # type: ignore[operator]
        z = math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)
        return self.floor + math.exp(self.mu + self.sigma * z)  # type: ignore[operator]


def _erfinv(y: float) -> float:
    """Inverse error function via Newton refinement (only used for reporting quantiles)."""
    x = 0.0
    for _ in range(60):
        err = math.erf(x) - y
        x -= err / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))
    return x


def sample_patience(rng: np.random.Generator, mean: float, cfg: PatienceConfig) -> float:
    return PatienceDistribution.from_config(mean, cfg).sample(rng)
