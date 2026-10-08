"""Hurst exponent estimators for a log-volatility series.

The main estimator is the one of Gatheral, Jaisson and Rosenbaum (2018), "Volatility is
rough", Quant. Finance 18(6). For a series x_t = log sigma_t observed on a
regular grid (one step per trading day for daily realised variance), define

    m(q, Delta) = mean over t of |x_{t + Delta} - x_t|^q.

If x has stationary increments that scale like Delta^H, then m(q, Delta) is proportional
to Delta^(q H). Regressing log m(q, Delta) on log Delta over a range of lags gives a slope
zeta_q, and H = zeta_q / q. Doing this for several q and regressing zeta_q on q through
the origin gives a single H that uses every moment at once. Linearity of zeta_q in q is
itself a check: a multifractal series would give a curved zeta_q.

Standard errors are the ordinary least squares standard errors of the slope. They treat
the points on the log-log plot as independent, which they are not (the same data enter
every lag), so they understate the sampling error of H. Measured on 2026-10-08 with 200
fBm paths of 5,552 steps, q = 2 and lags 1 to 50, the spread of H across paths was 6 times
the mean regression SE at H = 0.1, 12 times at H = 0.5 and 20 times at H = 0.7.
`monte_carlo_se` gives the honest error bar: the spread of the estimate across fBm paths
of the same length and H.

`hurst_variogram2` is a cross-check built on second-order increments
x_{t + 2 Delta} - 2 x_{t + Delta} + x_t, whose variance also scales like Delta^(2H). It is
blind to linear trends and is the standard form used in the change-of-frequency
estimators that Session 01 will look at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HurstFit:
    """Result of one log-log regression.

    q: moment order. lags: the lags used, in grid steps. m: the moment at each lag.
    slope: d log m / d log Delta (zeta_q for the GJR estimator). intercept: of that line.
    H: slope / q. se_slope, se_H: OLS standard errors. r2: coefficient of determination.
    """

    q: float
    lags: np.ndarray
    m: np.ndarray
    slope: float
    intercept: float
    H: float
    se_slope: float
    se_H: float
    r2: float


@dataclass(frozen=True)
class ZetaFit:
    """Multi-q GJR fit: zeta_q against q, regressed through the origin.

    qs, zetas, se_zetas: one entry per q. H: the slope of zeta_q on q. se_H: its OLS
    standard error. fits: the per-q HurstFit objects.
    """

    qs: np.ndarray
    zetas: np.ndarray
    se_zetas: np.ndarray
    H: float
    se_H: float
    fits: tuple[HurstFit, ...]


def moments(x: np.ndarray, q: float, lags: Sequence[int] | np.ndarray) -> np.ndarray:
    """m(q, Delta) = mean_t |x_{t+Delta} - x_t|^q for each Delta in lags (grid steps)."""
    x = np.asarray(x, dtype=np.float64)
    lags = _check_lags(lags, len(x))
    return np.array([np.mean(np.abs(x[lag:] - x[:-lag]) ** q) for lag in lags])


def hurst_gjr(
    x: np.ndarray, q: float = 2.0, lags: Sequence[int] | np.ndarray = range(1, 51)
) -> HurstFit:
    """GJR estimator from a single moment q: regress log m(q, Delta) on log Delta.

    x is log volatility (log sigma, or 0.5 * log RV) on a regular grid. The slope does
    not depend on the time unit, so lags are given in grid steps.
    """
    if q <= 0:
        raise ValueError(f"q must be > 0, got {q}")
    lags = _check_lags(lags, len(x))
    m = moments(x, q, lags)
    slope, intercept, se, r2 = _ols(np.log(lags), np.log(m))
    return HurstFit(q, lags, m, slope, intercept, slope / q, se, se / q, r2)


def hurst_gjr_multi(
    x: np.ndarray,
    qs: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    lags: Sequence[int] | np.ndarray = range(1, 51),
) -> ZetaFit:
    """GJR estimator over several q: fit zeta_q for each q, then zeta_q = H q."""
    fits = tuple(hurst_gjr(x, q, lags) for q in qs)
    q = np.array([f.q for f in fits])
    z = np.array([f.slope for f in fits])
    se_z = np.array([f.se_slope for f in fits])
    H = float(q @ z / (q @ q))
    resid = z - H * q
    dof = max(len(q) - 1, 1)
    se_H = float(np.sqrt(resid @ resid / dof / (q @ q)))
    return ZetaFit(q, z, se_z, H, se_H, fits)


def hurst_variogram2(x: np.ndarray, lags: Sequence[int] | np.ndarray = range(1, 26)) -> HurstFit:
    """Second-order variogram: E(x_{t+2D} - 2 x_{t+D} + x_t)^2 scales like D^(2H).

    Returns a HurstFit with q = 2, so H = slope / 2.
    """
    x = np.asarray(x, dtype=np.float64)
    lags = _check_lags(lags, len(x) // 2)
    v = np.array([np.mean((x[2 * d :] - 2 * x[d:-d] + x[: -2 * d]) ** 2) for d in lags])
    slope, intercept, se, r2 = _ols(np.log(lags), np.log(v))
    return HurstFit(2.0, lags, v, slope, intercept, slope / 2, se, se / 2, r2)


def monte_carlo_se(
    H: float,
    n: int,
    q: float = 2.0,
    lags: Sequence[int] | np.ndarray = range(1, 51),
    reps: int = 40,
    *,
    seed: int,
) -> float:
    """Standard deviation of hurst_gjr(q, lags) across `reps` fBm paths of n + 1 points.

    This is the sampling error of H for a series that really is fBm with this H and
    length. Cost is reps times one estimate plus one FFT per pair of paths.
    """
    from roughvol.fbm import fbm

    paths = fbm(float(np.clip(H, 0.01, 0.99)), n, 1.0, reps, seed=seed)
    return float(np.std([hurst_gjr(p, q, lags).H for p in paths], ddof=1))


def _ols(xv: np.ndarray, yv: np.ndarray) -> tuple[float, float, float, float]:
    """Slope, intercept, slope standard error and R^2 of y = a + b x by least squares."""
    n = len(xv)
    xm, ym = xv.mean(), yv.mean()
    sxx = np.sum((xv - xm) ** 2)
    slope = float(np.sum((xv - xm) * (yv - ym)) / sxx)
    intercept = float(ym - slope * xm)
    resid = yv - intercept - slope * xv
    sst = np.sum((yv - ym) ** 2)
    r2 = float(1 - resid @ resid / sst) if sst > 0 else 1.0
    se = float(np.sqrt(resid @ resid / (n - 2) / sxx)) if n > 2 else float("nan")
    return slope, intercept, se, r2


def _check_lags(lags: Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    lags = np.unique(np.asarray(lags, dtype=np.int64))
    if len(lags) < 2:
        raise ValueError("need at least two distinct lags")
    if lags[0] < 1 or lags[-1] >= n:
        raise ValueError(f"lags must lie in [1, {n - 1}], got {lags[0]}..{lags[-1]}")
    return lags
