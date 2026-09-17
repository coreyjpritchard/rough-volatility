import numpy as np


def quadratic_variation(path: np.ndarray, running: bool = False) -> np.ndarray:
    """Realised quadratic variation of a sampled path, on its own grid.

    For a path sampled at t_0 < ... < t_n,

        QV_n = sum_{i=1}^{n} (X_{t_i} - X_{t_{i-1}})**2,

    the p = 2 case of `p_variation`. For Brownian motion on a uniform grid of n
    steps over [0, T] it is exact in the mean and converges in L^2:

        E[QV_n] = T   for every n,        Var(QV_n) = 2 * T**2 / n,

    because each squared increment is dt * Z**2 with Z ~ N(0, 1), E[Z**2] = 1
    and Var(Z**2) = 2. One path's QV therefore sits within about T*sqrt(2/n) of
    T: the rate is n**(-1/2).

    Parameters
    ----------
    path : numpy.ndarray
        Sampled path(s); last axis is time, at least 2 points. Shape (m, n + 1)
        as returned by `roughvol.paths.brownian`, or (n + 1,).
    running : bool, default False
        If True, return the cumulative sum rather than the total: realised
        [X]_t at each grid point after the first.

    Returns
    -------
    numpy.ndarray
        Totals of shape path.shape[:-1] (a scalar for one path), or, when
        running is True, shape path.shape[:-1] + (n,).

    Raises
    ------
    ValueError
        If the last axis has fewer than 2 points.

    Notes
    -----
    Cost: O(path.size), one pass.

    This is the sum over the grid the path was sampled on, not a supremum over
    all partitions, and the two are different objects at p = 2 as well. The
    strong 2-variation of Brownian motion — the supremum of sum (dX)**2 over
    all partitions of an interval — is almost surely infinite; the sup is finite
    only for p > 2. Index 2 is the *critical* exponent, the one at which the
    supremum still fails. What converges is the grid sum as the mesh goes to
    zero: for a continuous semimartingale it converges in probability (and
    almost surely along dyadic refinements) to [X]_t, whatever the refining
    sequence. That is why quadratic variation is the exponent one can measure on
    sampled data; see `p_variation` for p != 2.
    """
    if path.shape[-1] < 2:
        raise ValueError(f"path must have at least 2 points, got {path.shape[-1]}")

    squared = np.diff(path, axis=-1) ** 2
    if running:
        return np.cumsum(squared, axis=-1)
    return np.sum(squared, axis=-1)


def p_variation(path: np.ndarray, p: float) -> np.ndarray:
    """Sum of |increment|**p along the sampling grid.

        V_p(n) = sum_{i=1}^{n} |X_{t_i} - X_{t_{i-1}}|**p.

    For Brownian motion on a uniform grid of n steps over [0, T],

        E[V_p(n)] = mu_p * T**(p/2) * n**(1 - p/2),
        mu_p = E|Z|**p = 2**(p/2) * Gamma((p + 1) / 2) / sqrt(pi),  Z ~ N(0, 1),

    so log E[V_p] is linear in log n with slope 1 - p/2: the sum diverges for
    p < 2, is constant for p = 2, and vanishes for p > 2. That crossover, p = 2,
    is the p-variation index of Brownian motion. A path of Hurst index H has
    index 1/H, so rougher paths blow up at larger p; the general statement is
    taken up in notebook 01, not derived here.

    Parameters
    ----------
    path : numpy.ndarray
        Sampled path(s); the last axis is time, at least 2 points.
    p : float
        Exponent, p > 0.

    Returns
    -------
    numpy.ndarray
        Shape path.shape[:-1] (a scalar for a single path).

    Raises
    ------
    ValueError
        If p <= 0, or the last axis has fewer than 2 points.

    Notes
    -----
    Cost: O(path.size), one pass; |.|**p dominates for non-integer p.

    Bookwork: no equation number is cited. As in `quadratic_variation` this is
    the grid sum, not a supremum over partitions — which is the point: the grid
    sum is what refining a partition actually shows you.
    """
    if p <= 0:
        raise ValueError(f"p must be > 0, got {p}")
    if path.shape[-1] < 2:
        raise ValueError(f"path must have at least 2 points, got {path.shape[-1]}")

    return np.sum(np.abs(np.diff(path, axis=-1)) ** p, axis=-1)
