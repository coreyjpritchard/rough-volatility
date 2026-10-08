"""Heston variance paths, as a control whose volatility has H = 1/2.

Under Heston the variance follows the square-root (CIR) diffusion

    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t.

It is driven by Brownian motion, so on short time scales log v moves like Brownian motion
and the GJR estimator should report H close to 1/2. At lags comparable to 1 / kappa mean
reversion flattens m(q, Delta) and pulls the estimate down, so the control is meaningful
only at lags well below 1 / kappa.

Discretisation: full truncation Euler (Lord, Koekkoek and van Dijk 2010). The drift and
diffusion use max(v, 0), so the scheme never takes the square root of a negative number.
With 2 kappa theta > xi^2 (the Feller condition) the true process stays positive.
"""

from __future__ import annotations

import numpy as np


def heston_variance(
    n: int,
    T: float,
    m: int = 1,
    *,
    seed: int,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    v0: float = 0.04,
    substeps: int = 10,
) -> np.ndarray:
    """Simulate m variance paths on t_i = i * T / n, i = 0..n, shape (m, n + 1).

    Each reported step is split into `substeps` Euler steps. Returned values are
    max(v, 0) on the reported grid. Defaults satisfy the Feller condition
    (2 * 2 * 0.04 = 0.16 > 0.09) and give a mean-reversion time 1 / kappa of half a year.
    """
    if n < 1 or m < 1 or substeps < 1:
        raise ValueError("n, m and substeps must be >= 1")
    if T <= 0 or kappa <= 0 or theta <= 0 or xi <= 0 or v0 < 0:
        raise ValueError("T, kappa, theta, xi must be > 0 and v0 >= 0")
    rng = np.random.default_rng(seed)
    dt = T / (n * substeps)
    sq = np.sqrt(dt)
    out = np.empty((m, n + 1))
    v = np.full(m, float(v0))
    out[:, 0] = v
    for i in range(n):
        z = rng.standard_normal((substeps, m))
        for j in range(substeps):
            vp = np.maximum(v, 0.0)
            v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sq * z[j]
        out[:, i + 1] = np.maximum(v, 0.0)
    return out
