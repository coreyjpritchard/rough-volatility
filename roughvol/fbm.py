"""Exact simulation of fractional Brownian motion.

Standard fractional Brownian motion B^H with Hurst exponent H in (0, 1) is the centred
Gaussian process with B^H_0 = 0 and

    Cov(B^H_s, B^H_t) = (s^(2H) + t^(2H) - |t - s|^(2H)) / 2.

Its increments over a step of length dt are stationary with variance dt^(2H), so
increments scale like dt^H. On the grid t_i = i * dt the increments form fractional
Gaussian noise (fGn), whose autocovariance at lag k is dt^(2H) * gamma(k) with

    gamma(k) = (|k + 1|^(2H) - 2 |k|^(2H) + |k - 1|^(2H)) / 2.

At H = 1/2, gamma(k) = 0 for k >= 1 and B^H is Brownian motion.

Two exact samplers are provided. Both reproduce the finite-dimensional law on the grid
exactly; the only error is floating-point rounding.

- Davies and Harte (1987), circulant embedding: embed the n x n Toeplitz covariance of
  fGn in a 2n x 2n circulant matrix, diagonalise it by FFT, and colour complex white
  noise. Cost O(n log n). Exact only if the circulant eigenvalues are nonnegative. This is
  proved for H <= 1/2 (Craigmile 2003). On 2026-10-08 it also held numerically for every
  H in {0.9, 0.95, 0.99} and n in {2, 4, ..., 4096}, so the fallback below is a guard
  that the session apps do not reach.
- Cholesky factorisation of the Toeplitz covariance. Cost O(n^3) time and 8 n^2 bytes.
  Used when the embedding check fails, or on request.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Method = Literal["auto", "davies-harte", "cholesky"]


def fbm_covariance(H: float, s: np.ndarray | float, t: np.ndarray | float) -> np.ndarray:
    """Cov(B^H_s, B^H_t) = (s^(2H) + t^(2H) - |t - s|^(2H)) / 2 for standard fBm."""
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return 0.5 * (s ** (2 * H) + t ** (2 * H) - np.abs(t - s) ** (2 * H))


def fgn_autocovariance(H: float, n: int) -> np.ndarray:
    """Autocovariance gamma(k), k = 0..n-1, of unit-step fractional Gaussian noise."""
    _check_hurst(H)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    k = np.arange(n, dtype=np.float64)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def circulant_eigenvalues(H: float, n: int) -> np.ndarray:
    """Eigenvalues of the 2n x 2n circulant embedding of the fGn covariance.

    The first row is gamma(0), ..., gamma(n), gamma(n - 1), ..., gamma(1). Davies-Harte
    is exact if and only if every eigenvalue is nonnegative.
    """
    gamma = fgn_autocovariance(H, n + 1)
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    return np.fft.fft(row).real


def fbm_increments(
    H: float, n: int, T: float = 1.0, m: int = 1, *, seed: int, method: Method = "auto"
) -> np.ndarray:
    """Draw m independent fGn sequences of length n on the grid dt = T / n.

    Returns an array of shape (m, n): row j holds B^H_{t_{i+1}} - B^H_{t_i}, i = 0..n-1.

    method="auto" uses Davies-Harte and falls back to Cholesky when the smallest
    circulant eigenvalue is below -1e-10 * the largest. With the same seed, the same
    white noise is coloured for every H, so changing H changes only the roughness of the
    path, not its overall shape. That is what makes the path "morph" in the session app.
    """
    _check_hurst(H)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if method not in ("auto", "davies-harte", "cholesky"):
        raise ValueError(f"unknown method {method!r}")

    rng = np.random.default_rng(seed)
    scale = (T / n) ** H

    if method != "cholesky":
        lam = circulant_eigenvalues(H, n)
        ok = lam.min() >= -1e-10 * lam.max()
        if ok:
            return scale * _davies_harte(lam, n, m, rng)
        if method == "davies-harte":
            raise ValueError(
                f"circulant embedding is not nonnegative definite for H={H}, n={n}; "
                "use method='cholesky' or 'auto'"
            )

    gamma = fgn_autocovariance(H, n)
    lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    chol = np.linalg.cholesky(gamma[lag])
    return scale * (rng.standard_normal((m, n)) @ chol.T)


def fbm(
    H: float, n: int, T: float = 1.0, m: int = 1, *, seed: int, method: Method = "auto"
) -> np.ndarray:
    """Simulate m paths of standard fBm on t_i = i * T / n, i = 0..n.

    Returns float64 of shape (m, n + 1); column 0 is B^H_0 = 0. See `fbm_increments` for
    the sampler and the meaning of `method`.
    """
    inc = fbm_increments(H, n, T, m, seed=seed, method=method)
    out = np.zeros((m, n + 1))
    np.cumsum(inc, axis=1, out=out[:, 1:])
    return out


def _davies_harte(lam: np.ndarray, n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Colour complex white noise with the circulant eigenvalues lam (length 2n).

    With Z = Z1 + i Z2 standard complex normal of length 2n, Y = FFT(sqrt(lam / 2n) Z)
    has E[Y Y*] = 2C and E[Y Y^T] = 0, where C is the circulant matrix. So Re(Y) and
    Im(Y) are independent with covariance C, and their first n entries are exact fGn.
    One FFT therefore gives two paths.
    """
    size = 2 * n
    k = (m + 1) // 2
    z = rng.standard_normal((k, size)) + 1j * rng.standard_normal((k, size))
    y = np.fft.fft(np.sqrt(np.clip(lam, 0.0, None) / size) * z, axis=1)[:, :n]
    return np.concatenate([y.real, y.imag], axis=0)[:m]


def _check_hurst(H: float) -> None:
    if not 0.0 < H < 1.0:
        raise ValueError(f"H must be in (0, 1), got {H}")
