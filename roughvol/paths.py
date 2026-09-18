import numpy as np


def brownian(n: int, T: float, m: int, seed: int) -> np.ndarray:
    """Simulate standard Brownian motion on a uniform grid.

    Builds m independent paths of W on the grid t_i = i * T / n, i = 0..n, from
    iid Gaussian increments

        W_{t_i} - W_{t_{i-1}} = sqrt(dt) * Z_i,    Z_i ~ N(0, 1),  dt = T / n,

    so E[W_t] = 0, Var(W_t) = t, and increments over disjoint intervals are
    independent. Every identity in notebook 00 rests on that independence.

    Shape convention, fixed here and followed by every simulator in this
    package: the returned array has shape (m, n + 1) and column 0 is the t = 0
    value, identically 0.0. So W[:, -1] is W_T, W.shape[1] - 1 is the number of
    steps, and the matching time grid is np.linspace(0.0, T, n + 1).

    Parameters
    ----------
    n : int
        Number of steps, n >= 1; the grid has n + 1 points.
    T : float
        Horizon, T > 0.
    m : int
        Number of independent paths, m >= 1.
    seed : int
        Seed for numpy.random.default_rng; one seed, one array, bit for bit.

    Returns
    -------
    numpy.ndarray
        Float64, shape (m, n + 1), each row starting at 0.0.

    Raises
    ------
    ValueError
        If n < 1, m < 1 or T <= 0.

    Notes
    -----
    Cost: O(m * n) time and memory; m * n normals are drawn. Bookwork — no
    paper or equation number is cited for this construction.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")

    rng = np.random.default_rng(seed)
    dt = T / n
    increments = np.sqrt(dt) * rng.standard_normal((m, n))
    W = np.empty((m, n + 1), dtype=np.float64)
    W[:, 0] = 0.0
    np.cumsum(increments, axis=1, out=W[:, 1:])
    return W
