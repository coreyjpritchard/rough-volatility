"""Tests for variation.quadratic_variation and variation.p_variation.

Reuses paths_mod (same impl as variation_mod, via the impl fixture) to build
Brownian paths to measure.
"""

import numpy as np
import pytest


def _subsample(path: np.ndarray, n_max: int, n: int) -> np.ndarray:
    """Coarsen a path of n_max steps to n steps by taking every stride-th point."""
    stride = n_max // n
    return path[..., :: stride]


def test_qv_equals_p_variation_at_two(paths_mod, variation_mod):
    n, m, seed = 1024, 2, 20260105
    W = paths_mod.brownian(n, 1.0, m, seed)
    qv = variation_mod.quadratic_variation(W)
    v2 = variation_mod.p_variation(W, 2.0)
    np.testing.assert_allclose(qv, v2, rtol=1e-12)


def test_running_qv(paths_mod, variation_mod):
    n, m, seed = 512, 3, 20260106
    W = paths_mod.brownian(n, 1.0, m, seed)
    running = variation_mod.quadratic_variation(W, running=True)
    total = variation_mod.quadratic_variation(W, running=False)
    assert running.shape == (m, n)
    assert np.all(np.diff(running, axis=1) >= 0.0)
    np.testing.assert_allclose(running[:, -1], total, rtol=1e-12)


def test_qv_mean_is_T(paths_mod, variation_mod):
    n, T, m, seed = 4096, 1.0, 200, 20260102
    W = paths_mod.brownian(n, T, m, seed)
    qv = variation_mod.quadratic_variation(W)
    mean_qv = qv.mean()
    se = qv.std(ddof=1) / np.sqrt(m)
    assert abs(mean_qv - T) < 4.0 * se


def test_qv_spread_matches_theory(paths_mod, variation_mod):
    n, T, m, seed = 4096, 1.0, 200, 20260102
    W = paths_mod.brownian(n, T, m, seed)
    qv = variation_mod.quadratic_variation(W)
    s = qv.std(ddof=1)
    theory = T * np.sqrt(2.0 / n)
    se = s / np.sqrt(2.0 * (m - 1))
    assert abs(s / theory - 1.0) < 4.0 * se / theory


def test_p_variation_slopes(paths_mod, variation_mod):
    n_max, T, seed = 2**14, 1.0, 20260103
    W = paths_mod.brownian(n_max, T, 1, seed)
    strides = [2**k for k in range(6, -1, -1)]
    ns = np.array([n_max // s for s in strides])
    # Bands sized to about 5 sd, not 4: the guard test below measures 4*sd on
    # seeds 2000-2199 as 0.0457 / 0.0859 / 0.1313 for p = 1, 2, 3, and a band
    # that close to 4*sd fails ~1 seed in 6 for reasons unrelated to any bug
    # (see test_slope_band_is_four_sigma, which still checks 4*sd <= band).
    bands = {1: 0.06, 2: 0.11, 3: 0.17}
    for p, band in bands.items():
        vs = np.array(
            [variation_mod.p_variation(_subsample(W, n_max, n), float(p)).item() for n in ns]
        )
        slope, _ = np.polyfit(np.log(ns), np.log(vs), 1)
        expected = 1.0 - p / 2.0
        assert abs(slope - expected) < band


def test_p_variation_direction(paths_mod, variation_mod):
    n_max, T, seed = 2**14, 1.0, 20260103
    W = paths_mod.brownian(n_max, T, 1, seed)
    coarse = _subsample(W, n_max, 2**8)
    fine = _subsample(W, n_max, 2**14)
    v1_coarse = variation_mod.p_variation(coarse, 1.0).item()
    v1_fine = variation_mod.p_variation(fine, 1.0).item()
    v3_coarse = variation_mod.p_variation(coarse, 3.0).item()
    v3_fine = variation_mod.p_variation(fine, 3.0).item()
    # Band sized from the across-seed spread of the ratio itself, not from a
    # per-doubling figure compounded six times: the six doublings are nested reads
    # of one path, so they are not independent. Measured over 300 seeds on this
    # construction: mean 8.03, sd 0.385, so 4 sd is 8.03 +/- 1.54.
    assert 6.3 < v1_fine / v1_coarse < 9.7
    # p = 3 side, same 300 seeds: mean 0.127, sd 0.020, so 0.2 is about 3.7 sd away.
    assert v3_fine / v3_coarse < 0.2


@pytest.mark.parametrize("p", [0.0, -1.0])
def test_p_variation_invalid(variation_mod, p):
    path = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        variation_mod.p_variation(path, p)


def test_p_variation_invalid_short_path(variation_mod):
    with pytest.raises(ValueError):
        variation_mod.p_variation(np.array([0.0]), 2.0)


@pytest.mark.parametrize("running", [False, True])
def test_qv_invalid_short_path(variation_mod, running):
    # A 1-point path has no increments: raise, rather than silently returning the
    # plausible-looking realised variance of 0.0 that an empty sum would give.
    with pytest.raises(ValueError):
        variation_mod.quadratic_variation(np.array([0.0]), running=running)


def test_slope_band_is_four_sigma(paths_mod, variation_mod):
    n_max, T = 2**14, 1.0
    strides = [2**k for k in range(6, -1, -1)]
    ns = np.array([n_max // s for s in strides])
    # Measured 4*sd on seeds 2000-2199: 0.0457 / 0.0859 / 0.1313 for p = 1, 2, 3,
    # comfortably inside the widened bands above (guard criterion stays 4*sd <= band).
    bands = {1: 0.06, 2: 0.11, 3: 0.17}
    slopes = {p: [] for p in bands}
    for seed in range(2000, 2200):
        W = paths_mod.brownian(n_max, T, 1, seed)
        for p in bands:
            vs = np.array(
                [
                    variation_mod.p_variation(_subsample(W, n_max, n), float(p)).item()
                    for n in ns
                ]
            )
            slope, _ = np.polyfit(np.log(ns), np.log(vs), 1)
            slopes[p].append(slope)
    for p, band in bands.items():
        sd = np.std(slopes[p], ddof=1)
        assert sd > 0.0
        assert 4.0 * sd <= band
