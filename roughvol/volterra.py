"""Hybrid scheme for the Riemann-Liouville process.

Bennedsen, Lunde and Pakkanen (2017), "Hybrid scheme for Brownian
semistationary processes". The process simulated here is the truncated
Brownian semistationary process with the power kernel, their equation
(2.14) with sigma = 1 and g(x) = x^{H - 1/2}:

    Y_t = int_0^t (t - s)^{H - 1/2} dW_s,    H in (0, 1).

The variance is t^{2H} / (2H). At H = 1/2 the kernel is 1 and Y is
Brownian motion, so the variance is t.

The scheme (their equation (2.15), implemented as (3.7)) keeps the first
``kappa`` lags as an exact Gaussian and replaces the tail by a Riemann sum
at the optimal points b*_k of their Proposition 2.8. The tail is a discrete
convolution and is applied by FFT. ``kappa`` is 0, 1 or 2. The default is
1. ``kappa`` = 0 is the plain Riemann sum. Theorem 2.5: raising ``kappa``
changes the constant in the mean-square error, not the rate. The rate is
n^{-2H}.

``power_kernel_weights`` and ``wiener_integral`` are the tail weights and
the convolution. Those two pieces are what the archive repository's
``roughvol/paths.py`` held. That repository was not readable here, so the
formulas are the paper's, not a line copy.

Draw order of :func:`hybrid_scheme`. One child of the seed is spawned.
Then, at each time step, for each component of the vector
(dW, I_1, ..., I_kappa), one standard normal per path. Path index runs
fastest: each draw is a vector of length m. The components are mapped to
the covariance by the lower Cholesky factor (L such that L L^T = Sigma,
Sigma as in their equation (3.5)). At H = 1/2 every kernel integral equals
the Wiener increment, the covariance is singular, and one normal per path
is drawn.

The covariance Sigma is indexed (dW, I_1, ..., I_kappa). I_k is the
integral of (k * dt - tau)^{H - 1/2} against dW on an interval of length
dt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import hyp2f1

from roughvol.rng import generator, spawn

_KAPPA = (0, 1, 2)


@dataclass(frozen=True)
class HybridPaths:
    """One hybrid-scheme draw.

    ``times`` has shape ``(n + 1,)`` and starts at 0. ``Y`` has shape
    ``(m, n + 1)`` and column 0 is 0. ``dW`` has shape ``(m, n)``: column
    j is the Wiener increment on ``[times[j], times[j + 1]]``.
    ``components`` has shape ``(m, n, kappa + 1)``. The last axis is
    ``(dW, I_1, ..., I_kappa)`` on that step, the Gaussian vector the
    scheme adds into ``Y``. At ``H = 1/2`` every ``I_k`` equals ``dW``.
    """

    times: np.ndarray
    Y: np.ndarray
    dW: np.ndarray
    components: np.ndarray


def volterra_variance(H: float, t: np.ndarray | float) -> np.ndarray:
    """Var(Y_t) = t^{2H} / (2H). Zero at t = 0."""
    _check_hurst(H)
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros(t.shape, dtype=np.float64)
    pos = t > 0.0
    out[pos] = np.asarray(t, dtype=np.float64)[pos] ** (2.0 * H) / (2.0 * H)
    return out


def optimal_points(alpha: float, n: int) -> np.ndarray:
    """b*_k for k = 1..n, Proposition 2.8.

    For alpha = 0 the kernel is flat and every point in the cell is exact.
    The value returned is the alpha -> 0 limit, the geometric mean of the
    cell, which keeps the weights continuous in H.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not -0.5 < alpha < 0.5:
        raise ValueError(f"alpha must be in (-1/2, 1/2), got {alpha}")
    k = np.arange(1, n + 1, dtype=np.float64)
    if abs(alpha) < 1e-12:
        out = np.empty(n, dtype=np.float64)
        out[0] = np.exp(-1.0)
        if n > 1:
            kk = k[1:]
            out[1:] = np.exp(kk * np.log(kk) - (kk - 1.0) * np.log(kk - 1.0) - 1.0)
        return out
    # int_{k-1}^{k} u^alpha du, then the L2-optimal constant to the power 1/alpha.
    integral = (k ** (alpha + 1.0) - (k - 1.0) ** (alpha + 1.0)) / (alpha + 1.0)
    return np.exp(np.log(integral) / alpha)


def power_kernel_weights(H: float, n: int, T: float, *, kappa: int = 0) -> np.ndarray:
    """Riemann weights (b*_k * dt)^{H - 1/2} for lags k = 1..n.

    Shape ``(n,)``. The first ``kappa`` entries are 0: those lags are the
    exact Gaussian block, not the Riemann sum. ``kappa`` = 0 returns every
    weight. ``dt = T / n``.
    """
    _check_hurst(H)
    _check_grid(n, T)
    _check_kappa(kappa)
    dt = T / n
    alpha = H - 0.5
    if abs(alpha) < 1e-12:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = (optimal_points(alpha, n) * dt) ** alpha
    if kappa:
        weights[:kappa] = 0.0
    return weights


def wiener_integral(weights: np.ndarray, dW: np.ndarray) -> np.ndarray:
    """Convolution Y_i = sum_{k=1}^{i} weights[k - 1] * dW[:, i - k].

    ``weights`` has shape ``(n,)`` and ``dW`` has shape ``(m, n)``. The
    result has shape ``(m, n + 1)`` and column 0 is 0. The sum is a linear
    convolution, evaluated by FFT.
    """
    weights = np.asarray(weights, dtype=np.float64)
    dW = np.asarray(dW, dtype=np.float64)
    if dW.ndim != 2:
        raise ValueError(f"dW must have shape (m, n), got {dW.shape}")
    n = dW.shape[1]
    if weights.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {weights.shape}")
    if n == 0:
        return np.zeros((dW.shape[0], 1), dtype=np.float64)
    nfft = next_fast_len(2 * n)
    freq = rfft(weights, n=nfft) * rfft(dW, n=nfft, axis=1)
    conv = irfft(freq, n=nfft, axis=1)[:, :n]
    out = np.zeros((dW.shape[0], n + 1), dtype=np.float64)
    out[:, 1:] = conv
    return out


def block_covariance(H: float, kappa: int, dt: float) -> np.ndarray:
    """Covariance of (dW, I_1, ..., I_kappa) on one step of length ``dt``.

    Shape ``(kappa + 1, kappa + 1)``. This is Sigma in Bennedsen, Lunde and
    Pakkanen (2017), equation (3.5), with their n equal to 1/dt. At
    H = 1/2 every I_k equals dW, so every entry is ``dt`` and the matrix
    has rank 1.
    """
    _check_hurst(H)
    _check_kappa(kappa)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if kappa == 0:
        return np.array([[dt]], dtype=np.float64)
    alpha = H - 0.5
    size = kappa + 1
    sigma = np.empty((size, size), dtype=np.float64)
    if abs(alpha) < 1e-12:
        sigma.fill(dt)
        return sigma
    n = 1.0 / dt
    sigma[0, 0] = dt
    for j in range(1, kappa + 1):
        # Paper index j_paper = j + 1, because their column 1 is dW and their
        # column j_paper is the integral at lag j_paper - 1.
        upper = float(j)
        lower = float(j - 1)
        sigma[0, j] = sigma[j, 0] = (upper ** (alpha + 1.0) - lower ** (alpha + 1.0)) / (
            (alpha + 1.0) * n ** (alpha + 1.0)
        )
        sigma[j, j] = (upper ** (2.0 * alpha + 1.0) - lower ** (2.0 * alpha + 1.0)) / (
            (2.0 * alpha + 1.0) * n ** (2.0 * alpha + 1.0)
        )
    for j in range(1, kappa + 1):
        for k in range(j + 1, kappa + 1):
            sigma[j, k] = sigma[k, j] = _cross_covariance(alpha, j, k, n)
    return sigma


def riemann_liouville_covariance(H: float, times: np.ndarray) -> np.ndarray:
    """Exact Cov(Y_s, Y_t) on ``times``. Shape ``(len(times), len(times))``.

    For 0 < s <= t,

        int_0^s (s - u)^{alpha} (t - u)^{alpha} du
            = s^{alpha + 1} (t - s)^{alpha} / (alpha + 1)
              * 2F1(-alpha, alpha + 1; alpha + 2; -s / (t - s)),

    with alpha = H - 1/2, and the diagonal is t^{2H} / (2H). At H = 1/2
    the covariance is min(s, t).
    """
    _check_hurst(H)
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size < 1 or t[0] != 0.0:
        raise ValueError("times must be 1-d and start at 0")
    n = t.size
    cov = np.zeros((n, n), dtype=np.float64)
    if abs(H - 0.5) < 1e-12:
        return np.minimum(t[:, None], t[None, :])
    diag = volterra_variance(H, t)
    np.fill_diagonal(cov, diag)
    alpha = H - 0.5
    ii, jj = np.triu_indices(n, k=1)
    s = np.minimum(t[ii], t[jj])
    u = np.maximum(t[ii], t[jj])
    mask = s > 0.0
    if not np.any(mask):
        return cov
    a = s[mask]
    c = u[mask] - a
    bpar = alpha + 1.0
    val = a**bpar * c**alpha / bpar * hyp2f1(-alpha, bpar, bpar + 1.0, -a / c)
    cov[ii[mask], jj[mask]] = val
    cov[jj[mask], ii[mask]] = val
    return cov


def hybrid_covariance(H: float, n: int, T: float = 1.0, *, kappa: int = 1) -> np.ndarray:
    """Covariance of the hybrid-scheme grid, including t = 0.

    Deterministic: each step is an independent Gaussian and the lag-lag
    covariance is the block in :func:`block_covariance`, or the product of
    Riemann weights on the tail. Shape ``(n + 1, n + 1)``.
    """
    _check_hurst(H)
    _check_grid(n, T)
    _check_kappa(kappa)
    gamma = _lag_covariance(H, n, T, kappa)
    cov = np.zeros((n + 1, n + 1), dtype=np.float64)
    for gap in range(n):
        diag = np.array([gamma[k, k + gap] for k in range(n - gap)], dtype=np.float64)
        prefix = np.cumsum(diag)
        idx = np.arange(1, n - gap + 1)
        cov[idx, idx + gap] = prefix
        if gap:
            cov[idx + gap, idx] = prefix
    return cov


def hybrid_scheme(
    H: float,
    n: int,
    T: float = 1.0,
    m: int = 1,
    *,
    seed: int | np.random.SeedSequence,
    kappa: int = 1,
) -> HybridPaths:
    """Draw m paths. One child of ``seed`` feeds the whole run.

    See the module note for the draw order. ``Y`` has shape ``(m, n + 1)``.
    """
    _check_shape(n, T, m, kappa, H)
    rng = generator(spawn(seed, 1)[0])
    return hybrid_scheme_rng(H, n, T, m, rng, kappa=kappa)


def hybrid_scheme_rng(
    H: float,
    n: int,
    T: float,
    m: int,
    rng: np.random.Generator,
    *,
    kappa: int = 1,
) -> HybridPaths:
    """The scheme, drawing from an existing generator. Does not spawn.

    ``rBergomi`` uses this on its volatility child so the spot driver can
    take the next child. The draw order inside the generator is the one in
    the module note.
    """
    _check_shape(n, T, m, kappa, H)
    dt = T / n
    times = np.linspace(0.0, T, n + 1)
    if abs(H - 0.5) < 1e-12:
        # The kernel is 1, so every I_k equals dW and the covariance is singular.
        # One normal per path per step. Y is that Brownian motion on the nose.
        shocks = _draw_normals(rng, n, 1, m)
        dW = np.sqrt(dt) * shocks[:, 0, :]
        Y = np.zeros((m, n + 1), dtype=np.float64)
        Y[:, 1:] = np.cumsum(dW, axis=0).T
        # The kernel is 1, so each exact integral equals the increment.
        # No extra normal is drawn.
        components = np.repeat(dW.T[:, :, None], kappa + 1, axis=2)
        return HybridPaths(times=times, Y=Y, dW=dW.T, components=components)

    sigma = block_covariance(H, kappa, dt)
    chol = np.linalg.cholesky(sigma)
    shocks = _draw_normals(rng, n, kappa + 1, m)
    # shocks: (n, components, m). Cholesky on the component axis.
    correlated = np.einsum("ij,sjm->sim", chol, shocks)
    dW = correlated[:, 0, :]
    weights = power_kernel_weights(H, n, T, kappa=kappa)
    Y = wiener_integral(weights, dW.T)
    for lag in range(1, kappa + 1):
        # I_lag of step j lands at time index j + lag.
        block = correlated[: n - lag + 1, lag, :]
        Y[:, lag : lag + block.shape[0]] += block.T
    # (steps, components, paths) -> (paths, steps, components).
    components = np.transpose(correlated, (2, 0, 1)).copy()
    return HybridPaths(times=times, Y=Y, dW=dW.T, components=components)


def volterra(
    H: float,
    n: int,
    T: float = 1.0,
    m: int = 1,
    *,
    seed: int | np.random.SeedSequence,
    kappa: int = 1,
) -> np.ndarray:
    """Y paths, shape ``(m, n + 1)``, column 0 equal to 0.

    Thin wrapper around :func:`hybrid_scheme` for the same calling shape as
    :func:`roughvol.fbm.fbm`.
    """
    return hybrid_scheme(H, n, T, m, seed=seed, kappa=kappa).Y


def cholesky_volterra(
    H: float,
    n: int,
    T: float = 1.0,
    m: int = 1,
    *,
    seed: int | np.random.SeedSequence,
) -> np.ndarray:
    """Exact draw on the grid, by Cholesky of the covariance. Shape ``(m, n + 1)``.

    One child of ``seed``, then one matrix of m by n standard normals, path
    index fastest (the array is row-major with shape ``(m, n)``). This is a
    different draw order from :func:`hybrid_scheme`, so the paths do not
    match. The finite-dimensional law does, up to rounding of the factor.
    Cost is O(n^3) once, then O(m n^2).
    """
    _check_hurst(H)
    _check_grid(n, T)
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    times = np.linspace(0.0, T, n + 1)
    cov = riemann_liouville_covariance(H, times)[1:, 1:]
    cov = 0.5 * (cov + cov.T)
    chol = np.linalg.cholesky(cov)
    rng = generator(spawn(seed, 1)[0])
    z = rng.standard_normal((m, n))
    out = np.zeros((m, n + 1), dtype=np.float64)
    out[:, 1:] = z @ chol.T
    return out


def _lag_covariance(H: float, n: int, T: float, kappa: int) -> np.ndarray:
    """Covariance of the lag-k and lag-l contributions of one step. Shape (n, n)."""
    dt = T / n
    weights = power_kernel_weights(H, n, T, kappa=0)
    if kappa == 0:
        return np.outer(weights, weights) * dt
    sigma = block_covariance(H, kappa, dt)
    lags = np.arange(1, n + 1)
    k = lags[:, None]
    ell = lags[None, :]
    gamma = np.zeros((n, n), dtype=np.float64)
    both = (k <= kappa) & (ell <= kappa)
    k_only = (k <= kappa) & (ell > kappa)
    ell_only = (k > kappa) & (ell <= kappa)
    neither = (k > kappa) & (ell > kappa)
    kk, ll = np.nonzero(both)
    gamma[kk, ll] = sigma[lags[kk], lags[ll]]
    kk, ll = np.nonzero(k_only)
    gamma[kk, ll] = weights[ll] * sigma[lags[kk], 0]
    kk, ll = np.nonzero(ell_only)
    gamma[kk, ll] = weights[kk] * sigma[0, lags[ll]]
    kk, ll = np.nonzero(neither)
    gamma[kk, ll] = weights[kk] * weights[ll] * sigma[0, 0]
    return gamma


def _cross_covariance(alpha: float, j: int, k: int, n: float) -> float:
    """Sigma_{j, k} for 1 <= j < k, equation (3.5). ``j`` and ``k`` are lags."""
    # The paper's columns are 1-based with column 1 = dW, so lag j is their column j+1.
    # Their formula uses (j_paper - 1) = lag. The term that would read 0^alpha is the
    # integral from a left endpoint of 0, and that term is 0 because the power alpha+1 > 0.
    term1 = j ** (alpha + 1.0) * k**alpha * float(hyp2f1(-alpha, 1.0, alpha + 2.0, j / k))
    if j == 1:
        term2 = 0.0
    else:
        term2 = (
            (j - 1) ** (alpha + 1.0)
            * (k - 1) ** alpha
            * float(hyp2f1(-alpha, 1.0, alpha + 2.0, (j - 1) / (k - 1)))
        )
    return (term1 - term2) / ((alpha + 1.0) * n ** (2.0 * alpha + 1.0))


def _draw_normals(rng: np.random.Generator, n_steps: int, n_comp: int, m: int) -> np.ndarray:
    """Standard normals, shape (n_steps, n_comp, m).

    At each step, component 0, then 1, and so on. Each draw has length m,
    so the path index runs fastest.
    """
    out = np.empty((n_steps, n_comp, m), dtype=np.float64)
    for step in range(n_steps):
        for comp in range(n_comp):
            out[step, comp] = rng.standard_normal(m)
    return out


def _check_hurst(H: float) -> None:
    if not 0.0 < H < 1.0:
        raise ValueError(f"H must be in (0, 1), got {H}")


def _check_kappa(kappa: int) -> None:
    if kappa not in _KAPPA:
        raise ValueError(f"kappa must be one of {list(_KAPPA)}, got {kappa}")


def _check_grid(n: int, T: float) -> None:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")


def _check_shape(n: int, T: float, m: int, kappa: int, H: float) -> None:
    _check_hurst(H)
    _check_grid(n, T)
    _check_kappa(kappa)
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
