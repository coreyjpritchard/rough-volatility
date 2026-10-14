"""Black-Scholes in forward form, for the conditional Monte Carlo estimator.

The forward ``F`` is today's forward to expiry. ``df`` discounts the
premium. ``kind`` is ``"call"`` or ``"put"``. A zero total volatility
returns the intrinsic value.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _call_flag(kind) -> np.ndarray:
    kind = np.asarray(kind)
    bad = ~np.isin(kind, ("call", "put"))
    if np.any(bad):
        raise ValueError(f"kind must be 'call' or 'put', got {np.unique(kind[bad])}")
    return kind == "call"


def price(F, K, T, sigma, df=1.0, kind="call") -> np.ndarray:
    """Discounted Black price. Arguments broadcast."""
    F, K, T, sigma, df = (np.asarray(a, dtype=np.float64) for a in (F, K, T, sigma, df))
    is_call = _call_flag(kind)
    vol = sigma * np.sqrt(np.maximum(T, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / vol + 0.5 * vol
    d2 = d1 - vol
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    zero = ~np.isfinite(vol) | (vol == 0.0)
    call = np.where(zero, np.maximum(F - K, 0.0), call)
    put = np.where(zero, np.maximum(K - F, 0.0), put)
    return df * np.where(is_call, call, put)


def _vega(F, K, T, sigma) -> np.ndarray:
    vol = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / vol + 0.5 * vol
    return F * np.exp(-0.5 * d1 * d1) / _SQRT_2PI * np.sqrt(T)


def implied_vol(price_, F, K, T, df=1.0, kind="call", *, tol: float = 1e-10) -> np.ndarray:
    """Black implied volatility. NaN where the price has no volatility.

    Safeguarded Newton on ``[1e-6, 5]``. A finite return reprices to a
    residual below ``tol`` times the price, plus ``1e-12``.
    """
    p, F, K, T, df = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (price_, F, K, T, df))
    )
    is_call = np.broadcast_to(_call_flag(kind), p.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        undiscounted = p / df
        use_call = K >= F
        target = np.where(
            is_call == use_call,
            undiscounted,
            undiscounted - np.where(is_call, F - K, K - F),
        )
        upper = np.where(use_call, F, K)
        valid = np.isfinite(target) & (T > 0.0) & (target > 0.0) & (target < upper)
    kind_otm = np.where(use_call, "call", "put")
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    target_i, f, k, tt, kd = (a.ravel()[idx] for a in (target, F, K, T, kind_otm))
    lo = np.full(idx.size, 1e-6)
    hi = np.full(idx.size, 5.0)
    x = np.clip(np.maximum(np.sqrt(2.0 * np.abs(np.log(f / k)) / tt), 0.2), lo, hi)
    for _ in range(80):
        diff = price(f, k, tt, x, 1.0, kd) - target_i
        if np.all(np.abs(diff) <= tol * np.maximum(target_i, 1e-300) + 1e-12):
            break
        hi = np.where(diff > 0.0, x, hi)
        lo = np.where(diff <= 0.0, x, lo)
        vega = _vega(f, k, tt, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = x - diff / vega
        inside = np.isfinite(step) & (step > lo) & (step < hi)
        x = np.where(inside, step, 0.5 * (lo + hi))
    diff = price(f, k, tt, x, 1.0, kd) - target_i
    ok = np.abs(diff) <= tol * np.maximum(target_i, 1e-300) + 1e-12
    out.flat[idx[ok]] = x[ok]
    return out
