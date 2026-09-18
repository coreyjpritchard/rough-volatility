"""Tests for paths.wiener_integral, paths.power_kernel_*, variation.theta_sum.

Behind every tolerance here: a Wiener integral of a *deterministic* integrand
is exactly Gaussian, so the sample variance of m draws has an exact
SE(s^2)/sigma^2 = sqrt(2 / (m - 1)). At m = 20000 that is 0.01000, so 4*SE is
4.00%. theta_sum is a different animal (Pearson kurtosis about 15, not 3 - see
test_theta_sum_mean), so its mean tests use the sample CLT SE instead: the
Gaussian SE(s^2) formula would understate it by sqrt((k - 1) / 2) = sqrt(7).
The 15 is a pooled estimate over 2e6 draws (14.99); the sample kurtosis of any
one m = 20000 batch scatters about it with sd ~ 1.5, so do not quote a batch.
"""

import numpy as np
import pytest


def test_shape_and_zero_start(paths_mod):
    n, T, m, seed = 64, 1.0, 5, 0
    integral = paths_mod.wiener_integral(lambda t: np.ones_like(t), n, T, m, seed)
    assert integral.shape == (m, n + 1)
    assert integral.dtype == np.float64
    assert np.all(integral[:, 0] == 0.0)


def test_flat_integrand_is_brownian(paths_mod):
    n, T, m, seed = 64, 1.0, 5, 0
    integral = paths_mod.wiener_integral(lambda t: np.ones_like(t), n, T, m, seed)
    W = paths_mod.brownian(n, T, m, seed)
    np.testing.assert_allclose(integral, W, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "n, T, m",
    [
        (0, 1.0, 5),
        (64, 1.0, 0),
        (64, 0.0, 5),
        (64, -1.0, 5),
    ],
)
def test_wiener_integral_invalid(paths_mod, n, T, m):
    with pytest.raises(ValueError):
        paths_mod.wiener_integral(lambda t: np.ones_like(t), n, T, m, seed=0)


def test_wiener_integral_bad_integrand_shape(paths_mod):
    # f must return one weight per step; a wrong length (or a scalar, which would
    # otherwise broadcast into a constant-weight integral) is an error, not a default.
    with pytest.raises(ValueError):
        paths_mod.wiener_integral(lambda t: np.ones(3), 64, 1.0, 5, seed=0)


@pytest.mark.parametrize("a", [-0.4, 0.0, 0.3])
def test_naive_paths_agree(paths_mod, a):
    n, T, m, seed = 128, 1.0, 7, 3
    naive = paths_mod.power_kernel_integral(a, n, T, m, seed, exact=False)
    direct = paths_mod.wiener_integral(lambda s: (T - s) ** a, n, T, m, seed)[:, -1]
    np.testing.assert_allclose(naive, direct, rtol=1e-12)


def test_naive_weights_are_left_points(paths_mod):
    a, n, T = -0.4, 16, 2.0
    w = paths_mod.power_kernel_weights(a, n, T, exact=False)
    grid = np.linspace(0.0, T, n + 1)
    expected = (T - grid[:-1]) ** a
    np.testing.assert_allclose(w, expected, rtol=1e-15)


@pytest.mark.parametrize("a", [-0.49, -0.4, -0.2, 0.0, 0.3, 1.0])
@pytest.mark.parametrize("n", [8, 512, 4096])
@pytest.mark.parametrize("T", [1.0, 2.0])
def test_exact_weights_telescope(paths_mod, a, n, T):
    w = paths_mod.power_kernel_weights(a, n, T, exact=True)
    dt = T / n
    lhs = dt * np.sum(w**2)
    rhs = T ** (2 * a + 1) / (2 * a + 1)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


@pytest.mark.parametrize(
    "a, n, m, T",
    [
        (-0.5, 8, 5, 1.0),
        (-0.7, 8, 5, 1.0),
        (0.0, 0, 5, 1.0),
        (0.0, 8, 0, 1.0),
        (0.0, 8, 5, 0.0),
        (0.0, 8, 5, -1.0),
    ],
)
def test_power_kernel_integral_invalid(paths_mod, a, n, m, T):
    with pytest.raises(ValueError):
        paths_mod.power_kernel_integral(a, n, T, m, seed=0)


# power_kernel_weights takes no m, so the m = 0 row above is not its business: it
# validates only a, n and T, and those are the five cases it can actually reject.
@pytest.mark.parametrize(
    "a, n, T",
    [
        (-0.5, 8, 1.0),
        (-0.7, 8, 1.0),
        (0.0, 0, 1.0),
        (0.0, 8, 0.0),
        (0.0, 8, -1.0),
    ],
)
def test_power_kernel_weights_invalid(paths_mod, a, n, T):
    with pytest.raises(ValueError):
        paths_mod.power_kernel_weights(a, n, T)


def test_isometry_variance(paths_mod):
    n, T, m = 512, 1.0, 20000
    cases = [
        (0.0, False, 20260201),
        (0.3, False, 20260202),
        (-0.4, True, 20260203),
    ]
    for a, exact, seed in cases:
        draws = paths_mod.power_kernel_integral(a, n, T, m, seed, exact=exact)
        s2 = draws.var(ddof=1)
        target = T ** (2 * a + 1) / (2 * a + 1)
        assert abs(s2 / target - 1.0) < 0.045


def test_naive_loses_variance_on_singular_kernel(paths_mod):
    a, T, n = -0.4, 1.0, 1024
    w = paths_mod.power_kernel_weights(a, n, T, exact=False)
    dt = T / n
    naive_var = dt * np.sum(w**2)
    target = T ** (2 * a + 1) / (2 * a + 1)
    deficit = 1.0 - naive_var / target
    assert abs(deficit - 0.22283) < 0.002


@pytest.mark.parametrize("a, expected_slope", [(-0.4, -0.2), (0.3, -1.0)])
def test_naive_bias_decay_rate(paths_mod, a, expected_slope):
    T = 1.0
    ns = np.array([2**k for k in range(6, 15)])
    deficits = []
    target = T ** (2 * a + 1) / (2 * a + 1)
    for n in ns:
        w = paths_mod.power_kernel_weights(a, n, T, exact=False)
        dt = T / n
        naive_var = dt * np.sum(w**2)
        deficits.append(abs(1.0 - naive_var / target))
    slope, _ = np.polyfit(np.log(ns), np.log(deficits), 1)
    tol = 0.01 if expected_slope == -0.2 else 0.02
    assert abs(slope - expected_slope) < tol


def test_theta_sum_identity(paths_mod, variation_mod):
    n, T, m, seed = 512, 1.0, 200, 20260204
    W = paths_mod.brownian(n, T, m, seed)
    qv = variation_mod.quadratic_variation(W)
    for shift in (0.0, 3.0):
        X = W + shift
        for theta in (0.0, 0.5, 1.0):
            S = variation_mod.theta_sum(X, theta)
            expected = (X[:, -1] ** 2 - X[:, 0] ** 2) / 2.0 + (theta - 0.5) * qv
            np.testing.assert_allclose(S, expected, atol=1e-10, rtol=1e-10)


def test_theta_sum_mean(paths_mod, variation_mod):
    n, T, m, seed = 512, 1.0, 20000, 20260204
    W = paths_mod.brownian(n, T, m, seed)
    for theta in (0.0, 0.5, 1.0):
        S = variation_mod.theta_sum(W, theta)
        mean = S.mean()
        se = S.std(ddof=1) / np.sqrt(m)
        assert abs(mean - theta * T) < 4.0 * se


@pytest.mark.parametrize("theta", [-0.1, 1.1])
def test_theta_sum_invalid_theta(variation_mod, theta):
    path = np.array([0.0, 1.0, 0.5])
    with pytest.raises(ValueError):
        variation_mod.theta_sum(path, theta)


def test_theta_sum_invalid_short_path(variation_mod):
    with pytest.raises(ValueError):
        variation_mod.theta_sum(np.array([0.0]), 0.0)


@pytest.mark.slow
def test_isometry_band_is_four_sigma(paths_mod):
    a, n, m, T = -0.4, 256, 4000, 1.0
    target = T ** (2 * a + 1) / (2 * a + 1)
    theory_se_rel = np.sqrt(2.0 / (m - 1))
    rels = []
    for seed in range(2000, 2200):
        draws = paths_mod.power_kernel_integral(a, n, T, m, seed, exact=True)
        s2 = draws.var(ddof=1)
        rels.append(s2 / target - 1.0)
    rels = np.array(rels)
    sd = rels.std(ddof=1)
    assert sd > 0.0
    ratio = sd / theory_se_rel
    assert 0.7 < ratio < 1.3
    assert np.max(np.abs(rels)) < 0.1
