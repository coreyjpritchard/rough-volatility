from collections.abc import Callable

import numpy as np


def atm_skew(smile: Callable[[np.ndarray], np.ndarray], h: float = 1e-3) -> float:
    """The ATM skew psi = |d(sigma_BS)/dk| at k = 0, by a central difference.

        psi = |smile(h) - smile(-h)| / (2 * h),

    with k = log(K / F) the log-moneyness and `smile` the map k -> sigma at a
    FIXED maturity tau (call `atm_skew` once per maturity to build a term
    structure). `smile` is a callable taking a numpy array of log-moneyness
    values and returning the implied volatility at each; it is called once, on
    the pair [-h, +h], and never trained or fitted here.

    h >= 1e-3 is required: the central-difference truncation error is O(h**2)
    and shrinks with h, but any numerical noise already present in `smile`
    (round-off, or the ~1e-10-level price-residual noise in `implied_vol`'s
    own inversion) is divided by h, so shrinking h below about 1e-3 makes
    that noise the dominant error rather than reducing it. This is why a flat
    surface's skew is tested to 1e-6, not to `implied_vol`'s own 1e-10.

    Parameters
    ----------
    smile : callable
        k (numpy.ndarray) -> sigma (numpy.ndarray) at one fixed tau.
    h : float, default 1e-3
        Central-difference half-step in log-moneyness, h >= 1e-3.

    Returns
    -------
    float
        psi(tau) >= 0.

    Raises
    ------
    ValueError
        If h < 1e-3.

    Notes
    -----
    Cost: one call to `smile`, on an array of two points. Bookwork - the
    central-difference formula is not cited; the object it estimates,
    psi(tau), is defined in notebook 00, section 3, after Bayer, Friz and
    Gatheral (2016).
    """
    if h < 1e-3:
        raise ValueError(
            "h must be >= 1e-3: a smaller step lets numerical noise in "
            "`smile` dominate over the shrinking truncation error"
        )
    k = np.asarray([-h, h], dtype=float)
    sigma = np.asarray(smile(k), dtype=float).ravel()
    return float(abs((sigma[1] - sigma[0]) / (2.0 * h)))
