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
of the same length and H. `HurstEstimate` carries that Monte Carlo standard error.
The regression standard error stays on `HurstFit` and is not the headline error.

The skew route fits `|psi| = A T^{-alpha}` and returns `H = 1/2 - alpha`. That fit is
deterministic, so its Monte Carlo standard error is 0. The uncertainty to report with
a market skew is the change in H across maturity windows.

Per-regime vol of vol uses the intercept of `m(2, Delta)`. If `log sigma = (nu / 2) B`
with `B` standard fractional Brownian motion in units of one grid step
(`Var(B_t) = t^{2H}`), then `m(2, Delta) = (nu / 2)^2 Delta^{2H}`. The regression
`log m = a + 2 H log Delta` has `nu = 2 exp(a / 2)`. This is the scaling for
`log sigma = log v / 2` when `log v` has vol of vol `nu`.

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


@dataclass(frozen=True)
class HurstEstimate:
    """One Hurst estimate and the Monte Carlo standard error of that estimator.

    ``se_mc`` is the spread of the estimator across fractional Brownian paths of
    the same length, not the ordinary least squares standard error. ``lags`` are
    grid steps for the realised-variance route and maturities for the skew
    route. ``q`` is the moment order. The skew route has no moment and stores
    NaN. ``method`` is ``"gjr"`` or ``"skew"``. ``n_obs`` is the number of
    points that entered the fit.
    """

    H: float
    se_mc: float
    lags: np.ndarray
    q: float
    method: str
    n_obs: int


@dataclass(frozen=True)
class RegimeVolEstimate:
    """Hurst exponent and vol of vol on one regime, with Monte Carlo standard errors.

    ``nu`` comes from the intercept of ``m(2, Delta)`` on a unit-step grid.
    See the module note for the factor of 2. ``se_mc_H`` and ``se_mc_nu`` are
    the spreads of those two estimators on fractional Brownian paths of the
    same length, at the estimated ``(H, nu)``.
    """

    H: float
    nu: float
    se_mc_H: float
    se_mc_nu: float
    lags: np.ndarray
    n_obs: int


def hurst_estimate(
    x: np.ndarray,
    q: float = 2.0,
    lags: Sequence[int] | np.ndarray = range(1, 51),
    *,
    seed: int,
    reps: int = 40,
    method: str = "gjr",
) -> HurstEstimate:
    """GJR estimate with the Monte Carlo standard error in place of the regression SE."""
    if method != "gjr":
        raise ValueError(f"hurst_estimate method must be 'gjr', got {method!r}")
    x = np.asarray(x, dtype=np.float64)
    fit = hurst_gjr(x, q, lags)
    se_mc = monte_carlo_se(fit.H, len(x) - 1, q, fit.lags, reps, seed=seed)
    return HurstEstimate(fit.H, se_mc, fit.lags, float(q), method, int(len(x)))


def lag_window_sensitivity(
    x: np.ndarray,
    windows: Sequence[tuple[int, int]] = ((1, 10), (1, 20), (1, 50), (1, 100), (1, 200)),
    q: float = 2.0,
    *,
    seed: int,
    reps: int = 40,
) -> tuple[HurstEstimate, ...]:
    """One GJR estimate per inclusive lag window ``(low, high)``, in grid steps."""
    x = np.asarray(x, dtype=np.float64)
    out = []
    for low, high in windows:
        if low < 1 or high <= low:
            raise ValueError(f"lag window must satisfy 1 <= low < high, got {(low, high)}")
        if high >= len(x):
            continue
        out.append(hurst_estimate(x, q, range(int(low), int(high) + 1), seed=seed, reps=reps))
    if not out:
        raise ValueError(f"no lag window fits a series of length {len(x)}")
    return tuple(out)


def hurst_from_skew(
    T: np.ndarray,
    psi: np.ndarray,
    T_window: tuple[float, float] | None = None,
) -> HurstEstimate:
    """``H = 1/2 - alpha`` from ``|psi| = A T^{-alpha}``.

    ``T_window`` is ``(low, high)`` in years, inclusive. ``None`` uses every
    finite positive maturity with a nonzero skew. The fit is deterministic, so
    ``se_mc`` is 0. A market number should be read next to
    :func:`skew_window_sensitivity`.
    """
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    psi = np.asarray(psi, dtype=np.float64).reshape(-1)
    if T.shape != psi.shape:
        raise ValueError(f"T and psi must have the same length, got {T.shape} and {psi.shape}")
    use = np.isfinite(T) & np.isfinite(psi) & (T > 0.0) & (psi != 0.0)
    if T_window is not None:
        low, high = (float(T_window[0]), float(T_window[1]))
        if low >= high:
            raise ValueError(f"T_window must have low < high, got {T_window}")
        use = use & (T >= low) & (T <= high)
    if int(use.sum()) < 2:
        raise ValueError("need at least two maturities with a nonzero skew in the window")
    maturity = T[use]
    slope, _intercept, _se, _r2 = _ols(np.log(maturity), np.log(np.abs(psi[use])))
    # slope = d log|psi| / d log T = -alpha, and alpha = 1/2 - H.
    return HurstEstimate(
        H=float(0.5 + slope),
        se_mc=0.0,
        lags=maturity,
        q=float("nan"),
        method="skew",
        n_obs=int(maturity.size),
    )


def skew_window_sensitivity(
    T: np.ndarray,
    psi: np.ndarray,
    windows: Sequence[tuple[float, float] | None],
) -> tuple[HurstEstimate, ...]:
    """``hurst_from_skew`` on each maturity window. ``None`` means every expiry."""
    return tuple(hurst_from_skew(T, psi, window) for window in windows)


def vol_of_vol_from_intercept(intercept: float) -> float:
    """``nu = 2 exp(intercept / 2)`` for a unit-step log-volatility series.

    The regression is ``log m(2, Delta) = intercept + 2 H log Delta``, with
    ``Delta`` in grid steps. See the module note.
    """
    return float(2.0 * np.exp(0.5 * float(intercept)))


def regime_estimate(
    x: np.ndarray,
    lags: Sequence[int] | np.ndarray = range(1, 51),
    *,
    seed: int,
    reps: int = 40,
) -> RegimeVolEstimate:
    """``(H, nu)`` on one regime, with Monte Carlo standard errors.

    The series must stay inside one regime for every lag in the window.
    Two long blocks, estimated separately, are the two-regime case.
    """
    from roughvol.fbm import fbm

    if reps < 2:
        raise ValueError(f"reps must be at least 2, got {reps}")
    x = np.asarray(x, dtype=np.float64)
    fit = hurst_gjr(x, 2.0, lags)
    nu = vol_of_vol_from_intercept(fit.intercept)
    n = len(x) - 1
    H_draw = float(np.clip(fit.H, 0.01, 0.99))
    paths = fbm(H_draw, n, float(n), reps, seed=seed)
    heights = []
    vols = []
    for path in paths:
        trial = hurst_gjr((nu / 2.0) * path, 2.0, fit.lags)
        heights.append(trial.H)
        vols.append(vol_of_vol_from_intercept(trial.intercept))
    return RegimeVolEstimate(
        H=fit.H,
        nu=nu,
        se_mc_H=float(np.std(heights, ddof=1)),
        se_mc_nu=float(np.std(vols, ddof=1)),
        lags=fit.lags,
        n_obs=int(len(x)),
    )


def regime_estimates(
    x: np.ndarray,
    labels: np.ndarray,
    lags: Sequence[int] | np.ndarray = range(1, 51),
    *,
    seed: int,
    reps: int = 40,
) -> dict:
    """``(H, nu)`` on the longest constant run of each label.

    A lag that would cross into another label is not used: the structure
    function is computed inside the run only.
    """
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if labels.shape != x.shape:
        raise ValueError(f"labels shape {labels.shape} does not match the series {x.shape}")
    if labels.size < 3:
        raise ValueError("labels must cover the series")
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], change, [labels.size]))
    best: dict = {}
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        key = labels[start].item() if hasattr(labels[start], "item") else labels[start]
        if key not in best or (end - start) > (best[key][1] - best[key][0]):
            best[key] = (int(start), int(end))
    return {
        key: regime_estimate(x[start:end], lags, seed=seed + i, reps=reps)
        for i, (key, (start, end)) in enumerate(sorted(best.items(), key=lambda item: str(item[0])))
    }


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
