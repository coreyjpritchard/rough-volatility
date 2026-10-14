"""Hybrid scheme against the closed form and against an exact Gaussian draw.

The block covariance is deterministic: elementary entries must match the
integral to 1e-12, and the one hypergeometric cross entry at kappa = 2 must
match a quadrature to 1e-12.

The variance check is the z-rule on a shared-path grid. n = 64 steps on
[0, 1], m = 2,000 paths, R = 4 replications, kappa = 1. Nodes are the 64
times after 0. The standard error of the second moment of a centred normal
is sigma^2 * sqrt(2 / m). The max bound applies to each replication. The
mean of z is checked across the R replications. H = 1/2 is included: there
the scheme is Brownian motion and the variance is exact, so the z scores
are Monte Carlo noise only.

The law check uses the same grid against an independent Cholesky draw of
the exact covariance. Means and second moments are separate z grids. The
standard error of a difference of two independent sample means is
sigma * sqrt(2 / m), and of two second moments is sigma^2 * sqrt(4 / m).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from tests.zrule import check_replication_means, check_z

M = 2_000
N = 64
R = 4
T = 1.0
SEED = 20261014


def _variance_z(impl, H, seed):
    Y = impl.volterra.volterra(H, N, T, M, seed=seed)
    t = np.linspace(0.0, T, N + 1)[1:]
    true = t ** (2 * H) / (2 * H)
    moment = np.mean(Y[:, 1:] ** 2, axis=0)
    se = true * np.sqrt(2.0 / M)
    return (moment - true) / se


def _law_z(impl, H, seed):
    Yh = impl.volterra.volterra(H, N, T, M, seed=seed)
    Yc = impl.volterra.cholesky_volterra(H, N, T, M, seed=seed + 10_000)
    t = np.linspace(0.0, T, N + 1)[1:]
    true = t ** (2 * H) / (2 * H)
    se_mean = np.sqrt(true * 2.0 / M)
    z_mean = (Yh[:, 1:].mean(axis=0) - Yc[:, 1:].mean(axis=0)) / se_mean
    se_second = true * np.sqrt(4.0 / M)
    z_second = (np.mean(Yh[:, 1:] ** 2, axis=0) - np.mean(Yc[:, 1:] ** 2, axis=0)) / se_second
    return z_mean, z_second


def _assert_shared_grid(rows):
    for z in rows:
        failure = check_z(z, mean=False, fraction=False)
        assert failure is None, failure
    failure = check_replication_means(np.array([np.mean(z) for z in rows]))
    assert failure is None, failure


def test_package_does_not_import_volsurface():
    from pathlib import Path

    import roughvol

    root = roughvol.__path__[0]
    for path in Path(root).rglob("*.py"):
        text = path.read_text()
        assert "import volsurface" not in text
        assert "from volsurface" not in text


def test_shape_start_and_seed(impl):
    draw = impl.volterra.hybrid_scheme(0.1, N, T, 5, seed=0)
    assert draw.Y.shape == (5, N + 1)
    assert draw.dW.shape == (5, N)
    assert draw.times.shape == (N + 1,)
    assert draw.times[0] == 0.0
    assert np.all(draw.Y[:, 0] == 0.0)
    np.testing.assert_array_equal(draw.Y, impl.volterra.volterra(0.1, N, T, 5, seed=0))
    assert not np.array_equal(draw.Y, impl.volterra.volterra(0.1, N, T, 5, seed=1))


@pytest.mark.parametrize("H", [0.0, 1.0, -0.2, 1.3])
def test_rejects_bad_hurst(impl, H):
    with pytest.raises(ValueError):
        impl.volterra.volterra(H, 8, 1.0, 2, seed=0)


def test_rejects_bad_kappa(impl):
    with pytest.raises(ValueError):
        impl.volterra.hybrid_scheme(0.1, 8, 1.0, 2, seed=0, kappa=3)


def test_optimal_points_lie_in_the_cell_and_beat_the_endpoints(impl):
    alpha = -0.4
    b = impl.volterra.optimal_points(alpha, 6)
    assert b.shape == (6,)
    assert 0.0 < b[0] <= 1.0
    for k, bk in enumerate(b[1:], start=2):
        assert k - 1.0 < bk < k

        def cell(point, k=k):
            upper = k ** (2 * alpha + 1) - (k - 1) ** (2 * alpha + 1)
            mid = k ** (alpha + 1) - (k - 1) ** (alpha + 1)
            return (
                upper / (2 * alpha + 1)
                - 2 * point**alpha * mid / (alpha + 1)
                + point ** (2 * alpha)
            )

        assert cell(bk) < cell(float(k))
        assert cell(bk) < cell(float(k - 1) + 1e-6)


def test_wiener_integral_matches_the_direct_sum(impl):
    rng = np.random.default_rng(SEED)
    weights = rng.normal(size=17)
    dW = rng.normal(size=(4, 17))
    got = impl.volterra.wiener_integral(weights, dW)
    direct = np.zeros((4, 18))
    for i in range(1, 18):
        direct[:, i] = sum(weights[k] * dW[:, i - 1 - k] for k in range(i))
    np.testing.assert_allclose(got, direct, atol=1e-12)


def test_block_covariance_matches_the_closed_form(impl):
    # Elementary entries, written as the integrals, not as a call into the library.
    H = 0.1
    alpha = H - 0.5
    dt = 1.0 / 64.0
    sigma = impl.volterra.block_covariance(H, 2, dt)
    assert sigma.shape == (3, 3)
    np.testing.assert_allclose(sigma[0, 0], dt, atol=1e-12)
    for lag in (1, 2):
        cov_dw = dt ** (alpha + 1) * (lag ** (alpha + 1) - (lag - 1) ** (alpha + 1)) / (alpha + 1)
        var_i = (
            dt ** (2 * alpha + 1)
            * (lag ** (2 * alpha + 1) - (lag - 1) ** (2 * alpha + 1))
            / (2 * alpha + 1)
        )
        np.testing.assert_allclose(sigma[0, lag], cov_dw, atol=1e-12)
        np.testing.assert_allclose(sigma[lag, lag], var_i, atol=1e-12)

    def cross(tau):
        return (dt - tau) ** alpha * (2 * dt - tau) ** alpha

    quad_cross, _ = quad(cross, 0.0, dt, epsabs=1e-14, limit=200)
    np.testing.assert_allclose(sigma[1, 2], quad_cross, atol=1e-12)
    np.testing.assert_allclose(sigma, sigma.T, atol=0.0)


def test_brownian_limit_is_exact(impl):
    # H = 1/2: Y increments are the Wiener increments, for every kappa.
    for kappa in (0, 1, 2):
        draw = impl.volterra.hybrid_scheme(0.5, N, T, 3, seed=SEED, kappa=kappa)
        np.testing.assert_allclose(np.diff(draw.Y, axis=1), draw.dW, atol=1e-12)
    # kappa does not change the draw at H = 1/2: one normal per step.
    a = impl.volterra.hybrid_scheme(0.5, N, T, 3, seed=SEED, kappa=1)
    b = impl.volterra.hybrid_scheme(0.5, N, T, 3, seed=SEED, kappa=2)
    np.testing.assert_array_equal(a.Y, b.Y)


def test_draw_order(impl):
    # Replay the documented order: one child, then at each step the components
    # (dW, I_1) as vectors of length m, mapped by the lower Cholesky factor.
    H, n, m, kappa = 0.2, 8, 3, 1
    seed = SEED
    got = impl.volterra.hybrid_scheme(H, n, 1.0, m, seed=seed, kappa=kappa)
    rng = impl.rng.generator(impl.rng.spawn(seed, 1)[0])
    dt = 1.0 / n
    chol = np.linalg.cholesky(impl.volterra.block_covariance(H, kappa, dt))
    calls = []
    correlated = np.empty((n, kappa + 1, m))
    for step in range(n):
        for comp in range(kappa + 1):
            draw = rng.standard_normal(m)
            calls.append(draw.shape)
            correlated[step, comp] = draw
    correlated = np.einsum("ij,sjm->sim", chol, correlated)
    dW = correlated[:, 0, :]
    weights = impl.volterra.power_kernel_weights(H, n, 1.0, kappa=kappa)
    Y = impl.volterra.wiener_integral(weights, dW.T)
    Y[:, 1:] += correlated[:, 1, :].T
    assert calls == [(m,)] * (n * (kappa + 1))
    np.testing.assert_allclose(got.Y, Y, atol=1e-12)
    np.testing.assert_allclose(got.dW, dW.T, atol=1e-12)


def test_hybrid_covariance_matches_a_step_sum(impl):
    # Brute force on a short grid: sum the lag-lag covariance over steps.
    H, n, kappa = 0.15, 5, 1
    cov = impl.volterra.hybrid_covariance(H, n, 1.0, kappa=kappa)
    sigma = impl.volterra.block_covariance(H, kappa, 1.0 / n)
    weights = impl.volterra.power_kernel_weights(H, n, 1.0, kappa=0)
    brute = np.zeros((n + 1, n + 1))
    for step in range(n):
        for i in range(step + 1, n + 1):
            for j in range(step + 1, n + 1):
                brute[i, j] += _contrib_cov(i - step, j - step, kappa, sigma, weights)
    np.testing.assert_allclose(cov, brute, atol=1e-12)


def _contrib_cov(lag_i, lag_j, kappa, sigma, weights):
    dt = sigma[0, 0]

    def coeff(lag):
        # Component 0 is dW. Component `lag` is I_lag when lag <= kappa.
        c = np.zeros(sigma.shape[0])
        if lag <= kappa:
            c[lag] = 1.0
        else:
            c[0] = weights[lag - 1]
        return c

    a = coeff(lag_i)
    b = coeff(lag_j)
    return (
        float(a @ sigma @ b)
        if lag_i <= kappa or lag_j <= kappa
        else (weights[lag_i - 1] * weights[lag_j - 1] * dt)
    )


@pytest.mark.parametrize("H", [0.05, 0.1, 0.3, 0.5])
def test_variance_matches_normalisation(impl, H):
    # Distinct seeds per H so the four parametrized cases are not one stream.
    base = SEED + int(round(1000 * H))
    rows = [_variance_z(impl, H, base + r) for r in range(R)]
    _assert_shared_grid(rows)


@pytest.mark.parametrize("H", [0.05, 0.1, 0.3])
def test_law_matches_cholesky(impl, H):
    means, seconds = [], []
    for r in range(R):
        z_mean, z_second = _law_z(impl, H, SEED + 50 + r)
        means.append(z_mean)
        seconds.append(z_second)
    _assert_shared_grid(means)
    _assert_shared_grid(seconds)


def test_scheme_variance_error_shrinks_with_kappa(impl):
    # Deterministic. At H = 0.1 and n = 64 the Riemann sum (kappa = 0) is a
    # long way off, and kappa = 1 and 2 sit close to the closed form. This is
    # the ordering the covariance figure is about.
    H, n = 0.1, 64
    true = impl.volterra.volterra_variance(H, np.linspace(0.0, 1.0, n + 1))
    errors = []
    for kappa in (0, 1, 2):
        diag = np.diag(impl.volterra.hybrid_covariance(H, n, 1.0, kappa=kappa))
        rel = np.max(np.abs(diag[1:] - true[1:]) / true[1:])
        errors.append(rel)
    assert errors[0] > 0.1
    assert errors[1] < 2e-3
    assert errors[2] < errors[1] < errors[0]
