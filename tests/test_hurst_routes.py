"""Hurst routes: realised variance, the skew, and per-regime vol of vol.

The fBm check reuses the session 00 control design: 4,096 steps, 40 paths,
lags 1 to 50. The band is one Monte Carlo standard deviation of the
estimator, the spread across those paths, not the regression standard error.

The skew tolerance on an exact power law is 1e-10. The SPX number is the
refit of the committed term structure, which this pull request computed from
the 14 October 2026 chain. It is not a copied logarithm.
"""

from __future__ import annotations

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
        assert full.se_mc == 0.0
        assert full.method == "skew"
        assert window.n_obs < full.n_obs
        assert full.n_obs == 15


def test_spx_skew_route_on_the_committed_term_structure(impl):
    """All 40 expiries of the recomputed 14 October 2026 SPX term structure.

    H = 0.1671745060 is the output of this pull request's fit, checked here
    to 1e-10 so a silent edit of the table is visible.
    """
    rows = np.genfromtxt(TERM, delimiter=",", names=True)
    estimate = impl.estimate.hurst_from_skew(rows["T"], rows["skew"])
    assert estimate.n_obs == 40
    assert estimate.H == pytest.approx(0.16717450602652617, abs=1e-10)


def test_two_regimes_recover_nu_within_monte_carlo_se(impl):
    """Two blocks, H = 0.15, nu = 0.6 and 1.8. Twenty paths, 4,096 steps each.

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
            found[i].append(fits[i].nu)
            heights[i].append(fits[i].H)
            reported[i].append(fits[i].se_mc_nu)
    for i, nu in enumerate(nus):
        mean, se_mc = _spread_band(found[i], nu)
        _spread_band(heights[i], H)
        assert mean == pytest.approx(nu, abs=se_mc)
        assert abs(mean - nu) < float(np.mean(reported[i]))
