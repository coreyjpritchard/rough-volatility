import numpy as np
from scipy.stats import norm


def bs_price(
    S: np.ndarray | float, K: np.ndarray | float, tau: np.ndarray | float,
    sigma: np.ndarray | float, r: float = 0.0, call: bool = True,
) -> np.ndarray:
    """Black-Scholes price of a European call or put, vectorised over inputs.

    With d1 = (log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    and d2 = d1 - sigma * sqrt(tau),

        call: C = S * Phi(d1) - K * exp(-r * tau) * Phi(d2),
        put:  P = K * exp(-r * tau) * Phi(-d2) - S * Phi(-d1),

    with Phi the standard normal CDF. Bookwork - not cited. S, K, tau, sigma
    broadcast against each other (numpy broadcasting rules); every input must
    be positive, tau and sigma strictly so.

    Parameters
    ----------
    S : array_like
        Spot price(s), > 0.
    K : array_like
        Strike(s), > 0.
    tau : array_like
        Time to maturity, > 0.
    sigma : array_like
        Volatility, > 0.
    r : float, default 0.0
        Continuously compounded risk-free rate.
    call : bool, default True
        True for a call, False for a put.

    Returns
    -------
    numpy.ndarray
        Broadcast price(s), same shape as the broadcast of the inputs.

    Raises
    ------
    ValueError
        If any of S, K, tau, sigma is <= 0.

    Notes
    -----
    Cost: O(size of the broadcast output), one call to scipy.stats.norm.cdf.
    """
    S, K, tau, sigma = np.asarray(S, dtype=float), np.asarray(K, dtype=float), \
        np.asarray(tau, dtype=float), np.asarray(sigma, dtype=float)
    if np.any(S <= 0) or np.any(K <= 0) or np.any(tau <= 0) or np.any(sigma <= 0):
        raise ValueError("S, K, tau and sigma must all be strictly positive")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


def vega(
    S: np.ndarray | float, K: np.ndarray | float, tau: np.ndarray | float,
    sigma: np.ndarray | float, r: float = 0.0,
) -> np.ndarray:
    """Black-Scholes vega, d(price)/d(sigma), same for a call and a put.

        vega = S * phi(d1) * sqrt(tau),  d1 as in `bs_price`, phi the
        standard normal density.

    As a formula vega is strictly positive for every tau, sigma > 0, which is
    what makes `implied_vol`'s inversion well posed (see `implied_vol`). In
    float64 it nevertheless underflows to exactly 0.0 once |d1| exceeds about
    38, because phi(d1) is then below the smallest representable double: for
    example vega(1, 1.3, 0.05, 1e-6) and vega(1, 3.0, 0.01, 0.1) both return
    0.0. That is why `implied_vol` safeguards its Newton step with a bisection
    branch rather than dividing by vega unconditionally. Bookwork - not cited.

    Parameters
    ----------
    S, K, tau, sigma : array_like
        As in `bs_price`.
    r : float, default 0.0
        As in `bs_price`.

    Returns
    -------
    numpy.ndarray
        Broadcast vega(s), positive as a formula but 0.0 where phi(d1)
        underflows (see above).

    Raises
    ------
    ValueError
        If any of S, K, tau, sigma is <= 0.

    Notes
    -----
    Cost: O(size of the broadcast output).
    """
    S, K, tau, sigma = np.asarray(S, dtype=float), np.asarray(K, dtype=float), \
        np.asarray(tau, dtype=float), np.asarray(sigma, dtype=float)
    if np.any(S <= 0) or np.any(K <= 0) or np.any(tau <= 0) or np.any(sigma <= 0):
        raise ValueError("S, K, tau and sigma must all be strictly positive")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return S * norm.pdf(d1) * np.sqrt(tau)


def implied_vol(
    price: np.ndarray | float, S: np.ndarray | float, K: np.ndarray | float,
    tau: np.ndarray | float, r: float = 0.0, call: bool = True, tol: float = 1e-10,
    vol_tol: float = 1e-2,
) -> np.ndarray:
    """Invert `bs_price` for sigma by safeguarded Newton on vega.

    `tol` bounds the PRICE residual |bs_price(..., sigma_hat) - price|, not
    the volatility error directly: near the root,
    sigma_hat - sigma_true ~= (price residual) / vega(sigma_true), so the vol
    error is roughly tol / vega. At S = K = 1, tau = 1, sigma = 0.2, r = 0,
    vega = 0.39695, so tol = 1e-10 gives a vol error of about 2.53e-10; deep
    out of the money, where vega is small, the same tol buys a looser vol
    error.

    Each Newton step sigma_{n+1} = sigma_n - (bs_price(sigma_n) - price) /
    vega(sigma_n) is clipped to a bracket [sigma_lo, sigma_hi] known to
    contain the root (bs_price is strictly increasing in sigma, per `vega`);
    a step that leaves the bracket is replaced by a bisection step on that
    bracket. The safeguard guarantees that the iterate stays inside a bracket
    containing the root and that the PRICE residual is driven below `tol`. It
    does NOT guarantee that sigma itself is accurate where vega is negligible,
    because many volatilities then reproduce the same price to within `tol`.

    Two post-conditions are therefore checked elementwise after iterating, and
    sigma is returned as nan wherever either fails:

      * the price residual is below `tol`;
      * both probes sigma_hat - `vol_tol` and sigma_hat + `vol_tol` move the
        price by more than `tol`. Since bs_price is increasing in sigma, a
        sigma outside (sigma_hat - vol_tol, sigma_hat + vol_tol) then prices
        more than `tol` away from the observed price and cannot be the answer,
        so the true sigma lies within `vol_tol` of the one returned. The lower
        probe is clipped at the search floor 1e-6.

    So a returned (non-nan) sigma is accurate to `vol_tol` or better, and
    callers such as a smile fit can filter the wings by testing for nan. This
    is a two-sided test on the price, not a conditioning estimate at
    sigma_hat: where the price is flat in sigma near the root, a healthy vega
    at whatever iterate the bisection lands on does not rescue it.

    A price outside the no-arbitrage bound (max(S - K*exp(-r*tau), 0) <=
    price <= S for a call; max(K*exp(-r*tau) - S, 0) <= price <=
    K*exp(-r*tau) for a put) has no implied volatility at any sigma > 0 and
    is rejected before iterating. A price within a few ulp of a bound (as
    `bs_price` itself can produce, e.g. one ulp below the intrinsic floor
    for a deep in-the-money, short-dated option) is clipped into the bound
    instead of raising, and the two-sided identifiability post-condition
    below then returns nan for it if it is not otherwise pinned down;
    a price further outside the bound than that still raises ValueError.

    The search runs over the fixed bracket [1e-6, 20.0], starting at its
    midpoint. An upper end of 20.0 (2000% annualised) is far above any
    volatility an equity index surface produces, so the root is inside.

    Parameters
    ----------
    price : array_like
        Observed option price(s), broadcastable against S, K, tau.
    S, K, tau : array_like
        As in `bs_price`.
    r : float, default 0.0
        As in `bs_price`.
    call : bool, default True
        As in `bs_price`.
    tol : float, default 1e-10
        Bound on the price residual (see above), not on the returned sigma.
    vol_tol : float, default 1e-2
        Conditioning floor in volatility units: an element whose price pins
        sigma no better than this is returned as nan rather than guessed.

    Returns
    -------
    numpy.ndarray
        Implied volatility, same broadcast shape as the inputs, > 0 where
        identifiable and nan elsewhere (see above).

    Raises
    ------
    ValueError
        If any price lies outside its no-arbitrage bound, or S, K, tau <= 0.

    Notes
    -----
    Cost: O(size of the broadcast output x number of Newton/bisection steps);
    a handful of iterations per element at this tol. Bookwork - the
    Newton-bisection hybrid is standard numerical practice, not a quoted
    result; no equation number is claimed.
    """
    price, S, K, tau = np.broadcast_arrays(
        np.asarray(price, dtype=float), np.asarray(S, dtype=float),
        np.asarray(K, dtype=float), np.asarray(tau, dtype=float),
    )
    if np.any(S <= 0) or np.any(K <= 0) or np.any(tau <= 0):
        raise ValueError("S, K and tau must all be strictly positive")

    if call:
        lo = np.maximum(S - K * np.exp(-r * tau), 0.0)
        hi = S
    else:
        lo = np.maximum(K * np.exp(-r * tau) - S, 0.0)
        hi = K * np.exp(-r * tau)
    # bs_price can land a few ulp outside its own no-arbitrage bound (float
    # round-off in the closing subtraction, worst for deep ITM/short-dated
    # options), so the slack scales with S, K rather than being one fixed
    # absolute tolerance: eight float64 steps at whichever of the upper bound
    # and 1.0 is larger, applied at the lower bound as well.
    eps = 8.0 * np.spacing(np.maximum(hi, 1.0))
    if np.any(price < lo - eps) or np.any(price > hi + eps):
        raise ValueError("price lies outside its no-arbitrage bound")
    price = np.clip(price, lo, hi)

    sigma_lo = np.full(price.shape, 1e-6)
    sigma_hi = np.full(price.shape, 20.0)
    sigma = 0.5 * (sigma_lo + sigma_hi)

    for _ in range(100):
        p = bs_price(S, K, tau, sigma, r=r, call=call)
        diff = p - price
        if np.all(np.abs(diff) < tol):
            break
        sigma_hi = np.where(diff > 0, sigma, sigma_hi)
        sigma_lo = np.where(diff < 0, sigma, sigma_lo)
        v = vega(S, K, tau, sigma, r=r)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Far enough out of the money vega underflows to exactly 0.0 and this
            # step is +-inf or nan. That is the case the bracket exists for: the
            # next line sends those elements to a bisection step instead.
            newton_sigma = sigma - diff / v
        out_of_bracket = (newton_sigma <= sigma_lo) | (newton_sigma >= sigma_hi)
        sigma = np.where(out_of_bracket, 0.5 * (sigma_lo + sigma_hi), newton_sigma)

    residual = np.abs(bs_price(S, K, tau, sigma, r=r, call=call) - price)
    # Two-sided identifiability: step sigma by +-vol_tol and see whether the price
    # notices. bs_price is increasing in sigma, so if both probes move it by more
    # than tol, no sigma outside the band can reproduce the observed price.
    probe_lo = np.maximum(sigma - vol_tol, 1e-6)
    probe_hi = sigma + vol_tol
    with np.errstate(invalid="ignore"):
        moved_lo = np.abs(bs_price(S, K, tau, probe_lo, r=r, call=call) - price) > tol
        moved_hi = np.abs(bs_price(S, K, tau, probe_hi, r=r, call=call) - price) > tol
    identified = (residual < tol) & moved_lo & moved_hi
    return np.where(identified, sigma, np.nan)
