"""Tests for paths.brownian (roughvol and, optionally, exercises)."""

import numpy as np
import pytest


def test_shape_and_zero_start(paths_mod):
    n, T, m, seed = 64, 1.0, 5, 0
    W = paths_mod.brownian(n, T, m, seed)
    assert W.shape == (m, n + 1)
    assert W.dtype == np.float64
    assert np.all(W[:, 0] == 0.0)


def test_seed_determinism(paths_mod):
    n, m = 64, 3
    W0a = paths_mod.brownian(n, 1.0, m, seed=0)
    W0b = paths_mod.brownian(n, 1.0, m, seed=0)
    W1 = paths_mod.brownian(n, 1.0, m, seed=1)
    np.testing.assert_array_equal(W0a, W0b)
    assert not np.array_equal(W0a, W1)


@pytest.mark.parametrize(
    "n, T, m",
    [
        (0, 1.0, 5),
        (64, 1.0, 0),
        (64, 0.0, 5),
        (64, -1.0, 5),
    ],
)
def test_invalid_arguments(paths_mod, n, T, m):
    with pytest.raises(ValueError):
        paths_mod.brownian(n, T, m, seed=0)


@pytest.mark.parametrize("n", [256, 1])
def test_terminal_variance(paths_mod, n):
    # n = 1 is the off-by-one probe: a grid of dt = T/(n+1) rather than T/n is a
    # 0.4% error at n = 256, invisible against Monte Carlo noise, but a 50% error
    # here, where the band is about 1%.
    T, m, seed = 2.0, 20000, 20260101
    W = paths_mod.brownian(n, T, m, seed)
    WT = W[:, -1]
    var_hat = WT.var(ddof=1)
    se = var_hat * np.sqrt(2.0 / (m - 1))
    assert abs(var_hat - T) < 4.0 * se


def test_increment_variance_is_flat(paths_mod):
    n, T, m, seed = 512, 1.0, 4000, 20260104
    W = paths_mod.brownian(n, T, m, seed)
    dt = T / n
    increments = np.diff(W, axis=1)
    col_var = increments.var(axis=0, ddof=1)
    se = dt * np.sqrt(2.0 / (m - 1))
    # Bonferroni over n columns: z = 5, not the usual 4.
    assert np.all(np.abs(col_var - dt) < 5.0 * se)
