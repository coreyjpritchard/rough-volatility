"""Law of the fBm sampler against closed forms.

fBm is Gaussian, which fixes the standard errors used below. For m independent draws of
a centred pair (X, Y) with variances a, b and covariance c, the estimator mean(X Y) of c
has standard error sqrt((a b + c^2) / m). For a sample correlation of m independent pairs
with true value rho, the standard error is about (1 - rho^2) / sqrt(m).

Tolerance is 4 SE throughout, with m = 4000 paths. With 4 SE a correct sampler fails one
comparison in about 16,000, so the 30 or so comparisons here are safe at any seed.
"""

import numpy as np
import pytest

M = 4000
N = 64
SEED = 20261008
PAIRS = [(0.25, 0.25), (0.25, 0.75), (0.5, 1.0), (1.0, 1.0)]


def theory_cov(H, s, t):
    return 0.5 * (s ** (2 * H) + t ** (2 * H) - abs(t - s) ** (2 * H))


def test_shape_start_and_seed(impl):
    B = impl.fbm.fbm(0.3, N, 1.0, 5, seed=0)
    assert B.shape == (5, N + 1)
    assert np.all(B[:, 0] == 0.0)
    np.testing.assert_array_equal(B, impl.fbm.fbm(0.3, N, 1.0, 5, seed=0))
    assert not np.array_equal(B, impl.fbm.fbm(0.3, N, 1.0, 5, seed=1))
    inc = impl.fbm.fbm_increments(0.3, N, 1.0, 5, seed=0)
    np.testing.assert_allclose(np.diff(B, axis=1), inc, atol=1e-12)


@pytest.mark.parametrize("H", [0.0, 1.0, -0.2, 1.3])
def test_rejects_bad_hurst(impl, H):
    with pytest.raises(ValueError):
        impl.fbm.fbm(H, N, 1.0, 2, seed=0)


@pytest.mark.parametrize("method", ["davies-harte", "cholesky"])
@pytest.mark.parametrize("H", [0.1, 0.3, 0.7, 0.9])
def test_covariance_matches_theory(impl, H, method):
    B = impl.fbm.fbm(H, N, 1.0, M, seed=SEED, method=method)
    for s, t in PAIRS:
        X, Y = B[:, round(s * N)], B[:, round(t * N)]
        a, b, c = theory_cov(H, s, s), theory_cov(H, t, t), theory_cov(H, s, t)
        se = np.sqrt((a * b + c**2) / M)
        assert abs(np.mean(X * Y) - c) < 4 * se, (s, t)


def test_brownian_increments_are_uncorrelated(impl):
    inc = impl.fbm.fbm_increments(0.5, N, 1.0, M, seed=SEED)
    x, y = inc[:, :-1].ravel(), inc[:, 1:].ravel()
    rho = np.corrcoef(x, y)[0, 1]
    # At H = 1/2 all increments are independent, so all M * (N - 1) pairs count.
    assert abs(rho) < 4 / np.sqrt(x.size)


@pytest.mark.parametrize("H", [0.1, 0.3, 0.7])
def test_lag_one_correlation(impl, H):
    inc = impl.fbm.fbm_increments(H, N, 1.0, M, seed=SEED)
    # One pair per path, so the M pairs are independent.
    rho = np.corrcoef(inc[:, 10], inc[:, 11])[0, 1]
    target = 2 ** (2 * H - 1) - 1
    assert abs(rho - target) < 4 * (1 - target**2) / np.sqrt(M)


def test_increment_scaling(impl):
    # Var(B_dt) = dt^(2H). One increment per path, so the M draws are independent and the
    # sample variance has SE = var * sqrt(2 / M).
    H = 0.2
    for n in (32, 64):
        x = impl.fbm.fbm_increments(H, n, 1.0, M, seed=SEED + n)[:, 0]
        target = (1 / n) ** (2 * H)
        assert abs(np.mean(x**2) - target) < 4 * target * np.sqrt(2 / M)


def test_falls_back_to_cholesky(impl, monkeypatch):
    if not hasattr(impl.fbm, "circulant_eigenvalues"):
        pytest.skip("implementation has no circulant_eigenvalues to break")

    def broken(H, n):
        lam = np.ones(2 * n)
        lam[0] = -1.0
        return lam

    monkeypatch.setattr(impl.fbm, "circulant_eigenvalues", broken)
    with pytest.raises(ValueError):
        impl.fbm.fbm(0.9, 8, 1.0, 2, seed=0, method="davies-harte")
    auto = impl.fbm.fbm(0.9, 8, 1.0, 2, seed=0, method="auto")
    chol = impl.fbm.fbm(0.9, 8, 1.0, 2, seed=0, method="cholesky")
    np.testing.assert_array_equal(auto, chol)
