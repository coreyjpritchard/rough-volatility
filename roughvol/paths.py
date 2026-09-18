from collections.abc import Callable

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


def wiener_integral(
    f: Callable[[np.ndarray], np.ndarray], n: int, T: float, m: int, seed: int
) -> np.ndarray:
    """Left-point approximation of the Wiener integral of a deterministic f.

    On the grid t_i = i * T / n, i = 0..n, with dt = T / n,

        I_{t_k} = sum_{i=0}^{k-1} f(t_i) * (W_{t_{i+1}} - W_{t_i}),

    the integrand evaluated at the LEFT end of each step and never at the
    right. For deterministic f the Ito isometry gives

        E[I_t] = 0,    Var(I_t) = integral_0^t f(s)**2 ds,

    and I_t is exactly Gaussian, being a fixed linear combination of
    independent normals. The discrete sum has variance dt * sum_i f(t_i)**2,
    a Riemann sum for that integral which equals it only in the limit; see
    `power_kernel_weights` for a kernel where the gap closes very slowly.

    Increments are drawn exactly as in `brownian`: one call to
    rng.standard_normal((m, n)) scaled by sqrt(dt), rng = default_rng(seed),
    so one seed means the same underlying path in `brownian`, here and in
    `power_kernel_integral`, and the three compare path by path.

    Parameters
    ----------
    f : callable
        Deterministic integrand, vectorised over an array of times; evaluated
        once, on t_0..t_{n-1}, and expected to return shape (n,).
    n : int
        Number of steps, n >= 1; the grid has n + 1 points.
    T : float
        Horizon, T > 0.
    m : int
        Number of independent paths, m >= 1.
    seed : int
        Seed for numpy.random.default_rng.

    Returns
    -------
    numpy.ndarray
        Float64, shape (m, n + 1), column 0 identically 0.0: the running
        integral, so [:, -1] is its value at T.

    Raises
    ------
    ValueError
        If n < 1, m < 1, T <= 0, or f does not return shape (n,).

    Notes
    -----
    Cost: O(m * n) time and memory. Bookwork - no paper or equation number is
    cited; the isometry is derived in notebook 00, section 2.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")

    grid = np.linspace(0.0, T, n + 1)
    weights = np.asarray(f(grid[:-1]), dtype=np.float64)
    if weights.shape != (n,):
        raise ValueError(f"f must return shape ({n},), got {weights.shape}")

    rng = np.random.default_rng(seed)
    dt = T / n
    increments = np.sqrt(dt) * rng.standard_normal((m, n))
    integral = np.empty((m, n + 1), dtype=np.float64)
    integral[:, 0] = 0.0
    np.cumsum(weights[None, :] * increments, axis=1, out=integral[:, 1:])
    return integral


def power_kernel_weights(a: float, n: int, T: float, exact: bool = True) -> np.ndarray:
    """Step weights for the power kernel f(s) = (T - s)**a on [0, T].

    Two weightings of the same n steps, both used as
    I = sum_i w_i * (W_{t_{i+1}} - W_{t_i}).

    exact=False, the naive left-point rule:

        w_i = (T - t_i)**a,          dt * sum_i w_i**2 = a Riemann sum.

    exact=True, the root-mean-square weight reproducing the variance of the
    true integral over each step:

        v_i = ((T - t_i)**(2a + 1) - (T - t_{i+1})**(2a + 1)) / (2a + 1),
        w_i = sqrt(v_i / dt),        dt * sum_i w_i**2 = T**(2a + 1)/(2a + 1)

    exactly, because the v_i telescope. That closed form is the Ito isometry
    applied to (T - s)**a and needs 2a + 1 > 0.

    Why it matters: for a < 0 the kernel blows up at s = T, the left point is
    the smallest value it takes on the last step, and the naive sum loses
    variance at the rate n**(-(2a+1)) - at a = -0.4, n = 1024 it is still 22%
    short. The exact weights are the seed of the hybrid scheme's near-singular
    block (notebook 04). They match the law of the integral exactly but not
    its joint law with W: Cov(w_i * dW_i, dW_i) = w_i * dt, where the true
    step integral has covariance integral (T - s)**a ds. Repairing that joint
    law is the hybrid scheme's job, not this function's.

    Parameters
    ----------
    a : float
        Kernel exponent, a > -0.5 so the variance integral converges.
    n : int
        Number of steps, n >= 1.
    T : float
        Horizon, T > 0.
    exact : bool, default True
        True for the telescoping weights, False for the left-point rule.

    Returns
    -------
    numpy.ndarray
        Float64, shape (n,), deterministic: the weight of step i = 0..n-1.

    Raises
    ------
    ValueError
        If a <= -0.5, n < 1 or T <= 0.

    Notes
    -----
    Cost: O(n), no random numbers. Bookwork - no equation number is cited.
    """
    if a <= -0.5:
        raise ValueError(f"a must be > -0.5, got {a}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")

    grid = np.linspace(0.0, T, n + 1)
    left, right = grid[:-1], grid[1:]
    if not exact:
        return (T - left) ** a

    dt = T / n
    v = ((T - left) ** (2 * a + 1) - (T - right) ** (2 * a + 1)) / (2 * a + 1)
    return np.sqrt(v / dt)


def power_kernel_integral(
    a: float, n: int, T: float, m: int, seed: int, exact: bool = True
) -> np.ndarray:
    """Simulate integral_0^T (T - s)**a dW_s, exactly or by the naive rule.

    Draws the same Brownian increments as `brownian` (rng.standard_normal(
    (m, n)) scaled by sqrt(dt)) and contracts them with
    `power_kernel_weights(a, n, T, exact)`. With exact=True the result is
    exactly N(0, T**(2a+1)/(2a+1)); with exact=False it is the left-point sum,
    N(0, dt * sum_i (T - t_i)**(2a)), whose variance is short of the target
    and stays short - see `power_kernel_weights`.

    Only the value at T is returned, because the integrand depends on T: the
    kernel (T - s)**a is a different function for a different endpoint, so
    there is no running version of this integral. That is exactly why the
    Riemann-Liouville process of notebook 01 must be rebuilt at every time
    point.

    Parameters
    ----------
    a : float
        Kernel exponent, a > -0.5.
    n : int
        Number of steps, n >= 1.
    T : float
        Horizon, T > 0.
    m : int
        Number of independent samples, m >= 1.
    seed : int
        Seed for numpy.random.default_rng.
    exact : bool, default True
        Passed to `power_kernel_weights`.

    Returns
    -------
    numpy.ndarray
        Float64, shape (m,): one draw of the integral per entry.

    Raises
    ------
    ValueError
        If a <= -0.5, n < 1, m < 1 or T <= 0.

    Notes
    -----
    Cost: O(m * n) time and memory. Bookwork - no equation number is cited.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    weights = power_kernel_weights(a, n, T, exact)

    rng = np.random.default_rng(seed)
    dt = T / n
    increments = np.sqrt(dt) * rng.standard_normal((m, n))
    return increments @ weights
