"""Tests for smile.atm_skew.

Every test here is a numerical-precision test on a fixed callable smile, not
a statistical one, so no Monte Carlo is needed.
"""

import numpy as np
import pytest


def _flat_smile(k):
    return np.full_like(k, 0.2)


def _exponential_smile(k):
    """0.2 * exp(0.3 k): slope 0.06 at k = 0, with a non-zero third derivative.

    The third derivative matters: for a quadratic the central difference is
    exact at every h, so a hard-coded or doubled step would go undetected.
    """
    return 0.2 * np.exp(0.3 * k)


def _falling_smile(k):
    """Slope -0.06 at k = 0, to pin that atm_skew returns an absolute value."""
    return 0.2 * np.exp(-0.3 * k)


def _vectorised_smile(k, seen=None):
    """A vectorised smile of the documented kind: array of k in, array out.

    An interpolator wrapped in np.atleast_1d, a pandas-backed fit or a
    vectorised lambda all behave this way, and none of them can be handed a
    0-dimensional array and have float() called on the result.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if seen is not None:
        seen.append(k.shape)
    return 0.2 + 0.05 * k


@pytest.mark.parametrize("h", [1e-3, 1e-2])
def test_flat_smile_zero_skew(smile_mod, h):
    psi = smile_mod.atm_skew(_flat_smile, h=h)
    assert abs(psi) < 1e-6


def test_atm_skew_matches_analytic_slope(smile_mod):
    """Against the true derivative, with the O(h**2) truncation error visible.

    d/dk [0.2 exp(0.3 k)] at k = 0 is 0.06, and the central difference is high
    by 0.2 * 0.3**3 * h**2 / 6, so a tenfold h must cost a hundredfold error.
    """
    err_small = abs(smile_mod.atm_skew(_exponential_smile, h=1e-3) - 0.06)
    err_large = abs(smile_mod.atm_skew(_exponential_smile, h=1e-2) - 0.06)
    np.testing.assert_allclose(err_large / err_small, 100.0, rtol=1e-3)
    np.testing.assert_allclose(
        err_large, 0.2 * 0.3**3 * 1e-2**2 / 6.0, rtol=1e-6
    )


def test_atm_skew_is_absolute(smile_mod):
    """A falling smile gives the same psi as the mirrored rising one."""
    np.testing.assert_allclose(
        smile_mod.atm_skew(_falling_smile, h=1e-3),
        smile_mod.atm_skew(_exponential_smile, h=1e-3),
        rtol=1e-12,
    )


def test_atm_skew_calls_a_vectorised_smile_once_on_the_pair(smile_mod):
    """The documented contract: one call, on the array [-h, +h].

    The shapes are recorded because a scalar-style implementation (evaluating
    at a 0-d array and calling float on the answer) raises TypeError on any
    genuinely vectorised smile, and that failure must not come back.
    """
    seen: list[tuple[int, ...]] = []
    psi = smile_mod.atm_skew(lambda k: _vectorised_smile(k, seen), h=1e-3)
    assert seen == [(2,)]
    np.testing.assert_allclose(psi, 0.05, atol=1e-10)


def test_atm_skew_rejects_small_h(smile_mod):
    with pytest.raises(ValueError):
        smile_mod.atm_skew(_flat_smile, h=1e-4)
