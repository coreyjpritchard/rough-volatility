"""Tests for bs.bs_price, bs.vega and bs.implied_vol.

Every test here is a numerical-precision test on a deterministic grid, not a
statistical one, so no Monte Carlo is needed. Seed 20260919 is kept for
consistency with the ticket even though the grids below are fixed arrays.
"""

import numpy as np
import pytest

SEED = 20260919

# Reference prices computed independently of `roughvol` and of scipy, from the
# closed form with math.erf, and confirmed to 20 significant figures with
# mpmath at 30-digit precision. The first is the standard textbook cell
# S = K = 100, tau = 1, sigma = 0.2, r = 0.05 (call 10.450584).
REF_ATM_CALL = 10.450583572185565
REF_ATM_PUT = 5.573526022256971
REF_OFF_ATM_CALL = 5.071235559904638


def _round_trip_grid():
    """A well-conditioned moneyness/maturity grid.

    The wings at tau = 0.05 are deliberately excluded because the round-trip
    tolerance below cannot hold there. At K/S = 1.30 the price is 1.83e-11 and
    a 1e-10 price residual admits a 7.0e-03 error in sigma, seven orders of
    magnitude outside the asserted 1e-8; at K/S = 0.70 the price sits on the
    forward intrinsic and vega is 1.15e-15, so no price tolerance pins sigma at
    all. The test's own `1e-10 / worst_vega < 1e-8` pre-check fails on the
    un-narrowed grid. Both cells are also flagged nan by `implied_vol`'s
    two-sided identifiability check; see
    `test_inversion_flags_unidentifiable_wings`.
    """
    moneyness = np.array([0.85, 1.0, 1.15])
    taus = np.array([0.25, 1.0, 3.0])
    S = 1.0
    K = S * moneyness[:, None]
    tau = taus[None, :]
    K, tau = np.broadcast_arrays(K, tau)
    return S, K, tau


def test_reference_prices(bs_mod):
    """Pin bs_price to independently computed values, so a slip in d1 shows up.

    Put-call parity and the round trip are both blind to d1: parity holds for
    any d1 via Phi(x) + Phi(-x) = 1, and the round trip inverts the same
    formula it prices with, so an error cancels. Only a reference value catches
    a missing 0.5 * sigma**2 or an r sign slip.
    """
    np.testing.assert_allclose(
        bs_mod.bs_price(100.0, 100.0, 1.0, 0.2, r=0.05, call=True),
        REF_ATM_CALL, rtol=1e-12,
    )
    np.testing.assert_allclose(
        bs_mod.bs_price(100.0, 100.0, 1.0, 0.2, r=0.05, call=False),
        REF_ATM_PUT, rtol=1e-12,
    )
    np.testing.assert_allclose(
        bs_mod.bs_price(100.0, 110.0, 0.5, 0.3, r=0.02, call=True),
        REF_OFF_ATM_CALL, rtol=1e-12,
    )


def test_vega_matches_central_difference(bs_mod):
    """vega is d(price)/d(sigma), so pin it against a difference of bs_price."""
    S, K, tau, sigma, r, dv = 100.0, 110.0, 0.5, 0.3, 0.02, 1e-5
    up = bs_mod.bs_price(S, K, tau, sigma + dv, r=r, call=True)
    down = bs_mod.bs_price(S, K, tau, sigma - dv, r=r, call=True)
    np.testing.assert_allclose(
        bs_mod.vega(S, K, tau, sigma, r=r), (up - down) / (2.0 * dv), rtol=1e-8
    )


@pytest.mark.parametrize("call", [True, False])
def test_round_trip(bs_mod, call):
    """price -> implied_vol -> sigma, on cells where sigma is identifiable.

    Tolerance arithmetic: `tol` bounds the price residual, so the vol error is
    bounded by tol / vega. The measured worst-case vega on this grid is
    4.9035e-02, giving 1e-10 / 4.9035e-02 = 2.04e-09, so the 1e-8 assertion
    below is justified by the grid rather than by luck.
    """
    S, K, tau = _round_trip_grid()
    sigma = 0.2
    price = bs_mod.bs_price(S, K, tau, sigma, r=0.0, call=call)
    worst_vega = np.min(bs_mod.vega(S, K, tau, sigma, r=0.0))
    print(f"worst-case vega on the round-trip grid: {worst_vega:.6g}")
    assert 1e-10 / worst_vega < 1e-8, (
        f"measured worst-case vega {worst_vega:.3g} makes the vol error bound "
        f"{1e-10 / worst_vega:.3g}, so a 1e-8 round-trip tolerance is unsafe"
    )
    sigma_hat = bs_mod.implied_vol(price, S, K, tau, r=0.0, call=call, tol=1e-10)
    assert np.all(np.isfinite(sigma_hat))
    assert np.max(np.abs(sigma_hat - sigma)) < 1e-8


def test_vega_positive(bs_mod):
    S, K, tau = _round_trip_grid()
    sigma = 0.2
    v = bs_mod.vega(S, K, tau, sigma, r=0.0)
    assert np.all(v > 0.0)


def test_vega_underflows_in_the_far_wing(bs_mod):
    """vega is positive as a formula but exactly 0.0 in float64 far enough out.

    This is the case the bisection safeguard exists for: dividing by it would
    give +-inf. `implied_vol` must not raise there, and must not return a
    number either, since none is recoverable.
    """
    assert bs_mod.vega(1.0, 1.3, 0.05, 1e-6) == 0.0
    assert bs_mod.vega(1.0, 3.0, 0.01, 0.1) == 0.0
    price = bs_mod.bs_price(1.0, 3.0, 0.01, 0.1, r=0.0, call=True)
    assert np.isnan(bs_mod.implied_vol(price, 1.0, 3.0, 0.01, r=0.0, call=True))


@pytest.mark.parametrize("r", [0.0, 0.03])
def test_put_call_parity(bs_mod, r):
    S = 1.0
    K = np.array([0.8, 1.0, 1.2])
    tau = np.array([0.1, 1.0])
    K, tau = np.meshgrid(K, tau, indexing="ij")
    sigma = 0.25
    call_price = bs_mod.bs_price(S, K, tau, sigma, r=r, call=True)
    put_price = bs_mod.bs_price(S, K, tau, sigma, r=r, call=False)
    np.testing.assert_allclose(
        call_price - put_price, S - K * np.exp(-r * tau), atol=1e-12
    )


def test_price_bounds_enforced(bs_mod):
    S, K, tau = 1.0, 1.0, 1.0
    # A call price above S (the upper no-arbitrage bound) is unreachable at any sigma.
    with pytest.raises(ValueError):
        bs_mod.implied_vol(S + 0.1, S, K, tau, r=0.0, call=True)
    # A put price below its intrinsic floor max(K*exp(-r*tau) - S, 0) = 0 here.
    with pytest.raises(ValueError):
        bs_mod.implied_vol(-0.1, S, K, tau, r=0.0, call=False)


def test_inversion_converges_deep_otm(bs_mod):
    """Deep OTM and ITM strikes still invert, at a maturity where vega survives.

    tau = 1 and sigma = 0.3 keep vega above 1e-4 out to K/S = 2.5, so the
    safeguarded step has to cope with a small vega without the price becoming
    uninformative.
    """
    moneyness = np.array([0.5, 0.6, 2.0, 2.5])
    S, tau, sigma = 1.0, 1.0, 0.30
    K = S * moneyness
    worst_vega = np.min(bs_mod.vega(S, K, tau, sigma, r=0.0))
    assert worst_vega > 1e-4
    price = bs_mod.bs_price(S, K, tau, sigma, r=0.0, call=True)
    sigma_hat = bs_mod.implied_vol(price, S, K, tau, r=0.0, call=True, tol=1e-10)
    assert np.all(np.isfinite(sigma_hat))
    assert np.all(sigma_hat > 0.0)
    assert np.max(np.abs(sigma_hat - sigma)) < 1e-10 / worst_vega


def test_inversion_flags_unidentifiable_wings(bs_mod):
    """Where the price cannot pin sigma, implied_vol returns nan, not a guess.

    At tau = 0.05, sigma = 0.15 the K/S = 2.5 call is worth 2.47e-167 and its
    vega is 1.24e-163; at S = 1, K = 5, tau = 0.01 the price underflows to
    0.0 outright. A plain Newton-bisection solver drives the price residual
    under tol at some arbitrary sigma (2.63 in the second case) and returns it.

    The last two cells are the ones a vega-at-the-iterate check lets through.
    At K/S = 1.475, tau = 0.05, sigma = 0.10 the bisection lands near 0.3066,
    where vega is 1.14e-08 and so looks healthy, while the true sigma is 0.10:
    only a two-sided test on the price catches it. At K/S = 1.30, tau = 0.05
    the returned 0.2070 is out by 7.0e-03 for the same reason.
    """
    moneyness = np.array([0.5, 0.6, 2.0, 2.5])
    S, tau, sigma = 1.0, 0.05, 0.15
    K = S * moneyness
    price = bs_mod.bs_price(S, K, tau, sigma, r=0.0, call=True)
    assert np.all(np.isnan(bs_mod.implied_vol(price, S, K, tau, r=0.0, call=True)))

    lone = bs_mod.bs_price(1.0, 5.0, 0.01, 0.05, r=0.0, call=True)
    assert np.isnan(bs_mod.implied_vol(lone, 1.0, 5.0, 0.01, r=0.0, call=True))

    for K_x, sigma_x in [(1.475, 0.10), (1.30, 0.20)]:
        p_x = bs_mod.bs_price(1.0, K_x, 0.05, sigma_x, r=0.0, call=True)
        assert np.isnan(bs_mod.implied_vol(p_x, 1.0, K_x, 0.05, r=0.0, call=True))

    # A deep ITM, short-dated put: bs_price's own output lands a ulp below the
    # intrinsic floor here (S=1, K=1.2, tau=0.05, sigma=0.1, r=0.03), so the
    # no-arbitrage guard must clip it in rather than raise, and the
    # identifiability check then flags it nan rather than returning a guess.
    p_edge = bs_mod.bs_price(1.0, 1.2, 0.05, 0.1, r=0.03, call=False)
    assert np.isnan(bs_mod.implied_vol(p_edge, 1.0, 1.2, 0.05, r=0.03, call=False))


def test_implied_vol_vectorised_no_raise_on_boundary_prices(bs_mod):
    """A vectorised call must not abort because one element sits on its own bound.

    The SPX-scale grid below (S=4500, r=0.045, K=3000..6000 step 25,
    tau=0.0027, sigma=0.2) is the shape of input that first showed the bug: a
    whole-array `np.any` guard on an exact comparison raised for the entire
    call when a single element came back a few ulp under its own no-arbitrage
    bound, purely from float round-off in bs_price's closing subtraction. The
    grid is NOT a regression test on its own, because whether any of its prices
    actually falls below its floor depends on the platform's libm and on the
    scipy build, and on this platform none of them currently does; it is kept
    only to show that a whole surface inverts in one call.

    The deterministic half of the test is built with np.nextafter instead. A
    price exactly one ulp (one float64 step) below the intrinsic floor must be
    clipped in rather than rejected, for a call and for a put. Both of those
    cases are deep in the money at the floor, where the price has stopped
    depending on sigma, so the identifiability check must then return nan for
    them. A price a further 1e-9 below the floor is genuinely outside the bound
    and must still raise.
    """
    S, r, tau, sigma = 4500.0, 0.045, 0.0027, 0.2
    K = np.arange(3000.0, 6000.0 + 25.0, 25.0)
    price = bs_mod.bs_price(S, K, tau, sigma, r=r, call=True)
    sigma_hat = bs_mod.implied_vol(price, S, K, tau, r=r, call=True)
    assert sigma_hat.shape == K.shape

    tau_b, r_b = 0.05, 0.03
    cases = [
        # (S, K, call, intrinsic floor)
        (1.0, 0.8, True, 1.0 - 0.8 * np.exp(-r_b * tau_b)),
        (1.0, 1.2, False, 1.2 * np.exp(-r_b * tau_b) - 1.0),
    ]
    for S_b, K_b, is_call, floor in cases:
        one_ulp_below = np.nextafter(floor, -np.inf)
        assert one_ulp_below < floor
        out = bs_mod.implied_vol(
            one_ulp_below, S_b, K_b, tau_b, r=r_b, call=is_call
        )
        # Asserting nan rather than merely "did not raise": the clip puts the
        # price on the intrinsic floor, where no sigma is recoverable, so a
        # regression that silently invented one would be caught here too.
        assert np.isnan(out)

        clearly_below = np.nextafter(floor, -np.inf) - 1e-9
        with pytest.raises(ValueError):
            bs_mod.implied_vol(
                clearly_below, S_b, K_b, tau_b, r=r_b, call=is_call
            )
