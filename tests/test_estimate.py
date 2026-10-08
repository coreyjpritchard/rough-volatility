"""The GJR estimator recovers a known H.

Each recovery test estimates H on REPS independent paths and compares the mean with the
true value. The tolerance is 2 Monte Carlo standard errors (sd / sqrt(REPS)) plus a bias
allowance of BIAS_TOL = 0.01. On 2026-10-08, with 200 paths of 5,552 steps, q = 2 and lags
1 to 50, the measured bias was below 0.001 for H in {0.1, 0.15, 0.3, 0.5, 0.7}; for the
multi-q estimator it was below 0.003. The allowance therefore covers the bias with room,
and the Monte Carlo term covers the noise of 40 paths (about 0.003 at H = 0.5).

The Heston control has a looser tolerance, stated in its test.
"""

import numpy as np
import pytest

REPS = 40
N = 4096
LAGS = range(1, 51)
BIAS_TOL = 0.01
SEED = 20261008


def _recovers(estimates, H):
    est = np.asarray(estimates)
    se = est.std(ddof=1) / np.sqrt(len(est))
    assert abs(est.mean() - H) < 2 * se + BIAS_TOL, (est.mean(), se)


def test_moments_of_a_line(impl):
    # x_t = t has |x_{t+D} - x_t|^q = D^q exactly, so zeta_q = q and H = 1 for every q.
    x = np.arange(500.0)
    fit = impl.estimate.hurst_gjr(x, 1.5, range(1, 30))
    np.testing.assert_allclose(fit.m, np.arange(1, 30) ** 1.5)
    assert fit.H == pytest.approx(1.0)
    assert impl.estimate.hurst_gjr_multi(x, lags=range(1, 30)).H == pytest.approx(1.0)


@pytest.mark.parametrize("H", [0.1, 0.3, 0.5, 0.7])
def test_recovers_hurst_on_fbm(impl, H):
    paths = impl.fbm.fbm(H, N, 1.0, REPS, seed=SEED)
    _recovers([impl.estimate.hurst_gjr(p, 2.0, LAGS).H for p in paths], H)


@pytest.mark.parametrize("H", [0.1, 0.5])
def test_multi_q_recovers_hurst_on_fbm(impl, H):
    paths = impl.fbm.fbm(H, N, 1.0, REPS, seed=SEED)
    fits = [impl.estimate.hurst_gjr_multi(p, lags=LAGS) for p in paths]
    _recovers([f.H for f in fits], H)
    # fBm is monofractal: zeta_q / q is the same for every q.
    z = np.mean([f.zetas / f.qs for f in fits], axis=0)
    assert np.ptp(z) < 0.02


def test_variogram2_recovers_hurst(impl):
    paths = impl.fbm.fbm(0.3, N, 1.0, REPS, seed=SEED)
    _recovers([impl.estimate.hurst_variogram2(p, range(1, 26)).H for p in paths], 0.3)


def test_brownian_motion_gives_one_half(impl):
    # Built directly from iid normals, so this does not rely on the fBm sampler.
    rng = np.random.default_rng(SEED)
    paths = np.cumsum(rng.standard_normal((REPS, N)), axis=1)
    _recovers([impl.estimate.hurst_gjr(p, 2.0, LAGS).H for p in paths], 0.5)


def test_regression_se_understates_sampling_error(impl):
    # The OLS SE treats overlapping lags as independent. Record that it is optimistic, so
    # nobody reads it as the error bar on H.
    paths = impl.fbm.fbm(0.15, N, 1.0, REPS, seed=SEED)
    fits = [impl.estimate.hurst_gjr(p, 2.0, LAGS) for p in paths]
    spread = np.std([f.H for f in fits], ddof=1)
    assert all(f.se_H > 0 for f in fits)
    assert spread > 3 * np.mean([f.se_H for f in fits])


def test_heston_variance_gives_one_half(impl):
    # Heston variance is driven by Brownian motion, so at short lags H should be 1/2.
    # Tolerance 0.06 on the mean of 10 paths of 10 years of daily data, lags 1 to 10 days.
    # Mean reversion (1 / kappa = 126 trading days) and the level-dependent volatility of
    # log v bias the estimate down: measured mean 0.47 with sd 0.016 across paths at the
    # default parameters, so the Monte Carlo error of the mean is about 0.005.
    v = impl.heston_vol.heston_variance(2520, 10.0, 10, seed=SEED)
    h = [impl.estimate.hurst_gjr(0.5 * np.log(p), 2.0, range(1, 11)).H for p in v]
    assert abs(np.mean(h) - 0.5) < 0.06
