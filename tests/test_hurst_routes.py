"""Hurst routes: realised variance, the skew, and per-regime vol of vol.

The fBm check reuses the session 00 control design: 4,096 steps, 40 paths,
lags 1 to 50. The band is one Monte Carlo standard deviation of the
estimator, the spread across those paths, not the regression standard error.

The skew tolerance on an exact power law is 1e-10. The SPX number is the
refit of the committed term structure, which this pull request computed from
the 14 October 2026 chain. It is not a copied logarithm.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

REPS = 40
N = 4096
LAGS = range(1, 51)
SEED = 20261014
TERM = Path(__file__).resolve().parents[1] / "docs" / "figures" / "rvc_spx_term.csv"


def _spread_band(values, truth):
    sample = np.asarray(values, dtype=np.float64)
    se_mc = float(sample.std(ddof=1))
    mean = float(sample.mean())
    assert abs(mean - truth) < se_mc, (mean, se_mc, truth)
    return mean, se_mc


@pytest.mark.parametrize("H", [0.05, 0.15, 0.5])
def test_fbm_mean_lies_inside_monte_carlo_se(impl, H):
    """m = 40 paths of 4,096 steps. The band is the path-to-path standard deviation."""
    paths = impl.fbm.fbm(H, N, 1.0, REPS, seed=SEED)
    estimates = [impl.estimate.hurst_gjr(path, 2.0, LAGS).H for path in paths]
    mean, se_mc = _spread_band(estimates, H)
    assert se_mc > 0.0
    assert abs(mean - H) < se_mc


def test_reported_error_is_the_monte_carlo_se(impl):
    path = impl.fbm.fbm(0.15, N, 1.0, 1, seed=SEED + 3)[0]
    estimate = impl.estimate.hurst_estimate(path, lags=LAGS, seed=SEED + 3)
    fit = impl.estimate.hurst_gjr(path, 2.0, LAGS)
    direct = impl.estimate.monte_carlo_se(fit.H, len(path) - 1, 2.0, LAGS, seed=SEED + 3)
    assert estimate.method == "gjr"
    assert estimate.n_obs == len(path)
    assert estimate.se_mc == pytest.approx(direct)
    assert estimate.se_mc > 3.0 * fit.se_H


def test_skew_route_recovers_exact_power_law(impl):
    maturities = np.geomspace(0.01, 1.0, 15)
    for H in (0.05, 0.167, 0.4):
        alpha = 0.5 - H
        psi = -1.7 * maturities ** (-alpha)
        full = impl.estimate.hurst_from_skew(maturities, psi)
        window = impl.estimate.hurst_from_skew(maturities, psi, (0.05, 0.4))
        assert full.H == pytest.approx(H, abs=1e-10)
        assert window.H == pytest.approx(H, abs=1e-10)
        assert math.isnan(full.se_mc)
        assert math.isnan(window.se_mc)
        assert full.method == "skew"
        assert window.n_obs < full.n_obs
        assert full.n_obs == 15


def test_spx_skew_route_on_the_committed_term_structure(impl):
    """All 40 expiries of the recomputed 14 October 2026 SPX term structure.

    H = 0.1671745060 is the all-expiry fit, checked here to 1e-10 so a silent
    edit of the table is visible. The same points on the prechosen windows
    span 0.115277 to 0.207468. Each fit is deterministic, so se_mc is NaN.
    """
    rows = np.genfromtxt(TERM, delimiter=",", names=True)
    windows = (
        None,
        (1.0 / 252.0, 1.0 / 12.0),
        (1.0 / 52.0, 0.25),
        (1.0 / 12.0, 1.0),
        (1.0 / 52.0, 1.0),
    )
    fits = impl.estimate.skew_window_sensitivity(rows["T"], rows["skew"], windows)
    heights = [fit.H for fit in fits]
    assert fits[0].n_obs == 40
    assert heights[0] == pytest.approx(0.16717450602652617, abs=1e-10)
    assert min(heights) == pytest.approx(0.11527725839990882, abs=1e-10)
    assert max(heights) == pytest.approx(0.20746848517500888, abs=1e-10)
    assert all(math.isnan(fit.se_mc) for fit in fits)


def test_two_regimes_recover_nu_within_monte_carlo_se(impl):
    """Two blocks, H = 0.15, nu_per_day = 0.6 and 1.8. Twenty paths, 4,096 steps each.

    The Monte Carlo SE is the standard deviation of the twenty estimates.
    Increments that would cross the join are not used: each label is estimated
    on its own run. ``regime_estimate`` draws 8 fractional Brownian paths for
    its reported standard error.
    """
    H = 0.15
    nus = (0.6, 1.8)
    reps = 20
    found = {0: [], 1: []}
    heights = {0: [], 1: []}
    reported = {0: [], 1: []}
    for r in range(reps):
        blocks = []
        for i, nu in enumerate(nus):
            block = impl.fbm.fbm(H, N, float(N), 1, seed=SEED + 1000 * i + r)[0]
            blocks.append((nu / 2.0) * block)
        series = np.concatenate(blocks)
        labels = np.concatenate([np.full(N + 1, i, dtype=int) for i in range(2)])
        fits = impl.estimate.regime_estimates(series, labels, LAGS, seed=SEED + 50 + r, reps=8)
        for i in range(2):
            found[i].append(fits[i].nu_per_day)
            heights[i].append(fits[i].H)
            reported[i].append(fits[i].se_mc_nu_per_day)
    for i, nu in enumerate(nus):
        mean, se_mc = _spread_band(found[i], nu)
        _spread_band(heights[i], H)
        assert mean == pytest.approx(nu, abs=se_mc)
        assert abs(mean - nu) < float(np.mean(reported[i]))


def test_nu_per_day_converts_to_annual_rbergomi_nu(impl):
    """The map uses the increment factor sqrt(2 H c_H), at an H other than 1/2.

    On rv5 at lags 1 to 50, nu_per_day = 0.6997318719333629 and
    H = 0.1475349541453704. Dividing by sqrt(2 H c_H) gives
    1.3282759090799114. The level scaling nu_per_day * 252^H is 1.582057
    and is a different number at this H. At H = 1/2 the factor is 1, so
    that case cannot tell the two formulas apart.
    """
    nu_per_day = 0.6997318719333629
    H = 0.1475349541453704
    annual = impl.estimate.rbergomi_nu(nu_per_day, H)
    factor = math.sqrt(2.0 * H * impl.estimate.c_hurst(H))
    level = nu_per_day * 252.0**H
    assert H != pytest.approx(0.5)
    assert factor > 1.0
    assert annual == pytest.approx(level / factor)
    assert annual == pytest.approx(1.3282759090799114, abs=1e-12)
    assert abs(annual - level) > 0.2
    assert impl.estimate.c_hurst(0.5) == pytest.approx(1.0)
    assert impl.estimate.rbergomi_nu(1.0, 0.5, trading_days=256) == pytest.approx(16.0)
    estimate = impl.estimate.RegimeVolEstimate(
        H=H,
        nu_per_day=nu_per_day,
        se_mc_H=0.01,
        se_mc_nu_per_day=0.01,
        lags=np.array([1.0, 2.0]),
        n_obs=10,
    )
    assert estimate.to_rbergomi_nu() == pytest.approx(annual)
    with pytest.raises(ValueError, match="trading_days"):
        impl.estimate.rbergomi_nu(1.0, 0.1, trading_days=0.0)


def test_monte_carlo_se_of_h_propagates_into_rbergomi_nu(impl):
    """The H error is carried by a central difference, with nu_per_day fixed.

    nu(H) = nu_per_day * 252^H / sqrt(2 H c_H). The derivative is
    (nu(H + ε) - nu(H - ε)) / (2ε) at ε = 1e-6. The standard error is the
    lags 1 to 50 cell of docs/figures/rvc_routes.csv. The product is the
    uncertainty reported with 1.328276.
    """
    nu_per_day = 0.6997318719333629
    H = 0.14753495414537035
    se_H = 0.013373510585712234
    eps = 1e-6

    def nu(h: float) -> float:
        return impl.estimate.rbergomi_nu(nu_per_day, h)

    slope = (nu(H + eps) - nu(H - eps)) / (2.0 * eps)
    carried = abs(slope) * se_H
    assert nu(H) == pytest.approx(1.3282759090799114, abs=1e-12)
    assert slope == pytest.approx(8.61604592, abs=1e-6)
    assert carried == pytest.approx(0.11522678, abs=1e-8)
    assert nu(H + se_H) == pytest.approx(1.44830641, abs=1e-8)
    assert nu(H - se_H) == pytest.approx(1.21761606, abs=1e-8)
    assert nu(0.12249034699599022) == pytest.approx(1.128162, abs=5e-7)
    assert nu(0.1582991406377058) == pytest.approx(1.424117, abs=5e-7)


def test_simulated_rbergomi_nu_is_recovered_within_the_replication_bound(impl):
    """H = 0.15, nu = 1.5, 16 paths, 6 years of daily steps, lags 1 to 20.

    The series is log sigma = log v / 2. The converter is applied at the
    simulated H. The band is the replication bound, 3.5 standard errors of
    the mean across the 16 paths. The level scaling, without sqrt(2 H c_H),
    falls outside that band.
    """
    from tests.zrule import check_replication_means

    H = 0.15
    nu = 1.5
    paths = 16
    years = 6
    steps = 252 * years
    model = impl.rbergomi.RoughBergomi(
        H=H, nu=nu, rho=-0.7, xi0=0.04, n_steps=steps, n_paths=paths, kappa=1
    )
    times = np.linspace(0.0, float(years), steps + 1)
    variance = model.simulate(paths, times, seed=SEED).state["v"]
    lags = range(1, 21)
    recovered = []
    level = []
    for row in 0.5 * np.log(variance):
        fit = impl.estimate.hurst_gjr(row, 2.0, lags)
        nu_per_day = impl.estimate.vol_of_vol_from_intercept(fit.intercept)
        recovered.append(impl.estimate.rbergomi_nu(nu_per_day, H))
        level.append(nu_per_day * 252.0**H)
    recovered = np.asarray(recovered, dtype=np.float64)
    level = np.asarray(level, dtype=np.float64)
    failure = check_replication_means(recovered - nu)
    assert failure is None, failure
    assert check_replication_means(level - nu) is not None
