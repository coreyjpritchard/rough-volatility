"""Rough Bergomi, in the forward coordinate.

Bayer, Friz and Gatheral (2016). On a forward curve ``xi0``,

    v_t = xi0(t) exp(nu sqrt(2H) Y_t - nu^2 t^{2H} / 2),
    Y_t = int_0^t (t - s)^{H - 1/2} dW_s,
    dZ / Z = sqrt(v) (rho dW + sqrt(1 - rho^2) dW_perp),    Z_0 = 1.

``Y`` is the Riemann-Liouville process from :mod:`roughvol.volterra`. The
factor ``sqrt(2H)`` makes ``nu sqrt(2H) Y`` have variance ``nu^2 t^{2H}``, so
the exponential is a martingale and ``E[v_t] = xi0(t)`` when ``Y`` has its
exact variance. ``nu`` is the vol of vol. The name ``eta`` is not used here.

``rho`` is at most 0. The constructor rejects a positive value. The fitter
box in :meth:`RoughBergomi.param_spec` stops at 0 as well.

Draw order. :func:`roughvol.rng.spawn` makes four children of the seed.
Child 0 is unused (kept for a regime stream). Child 1 is the volatility
driver and is handed to :func:`roughvol.volterra.hybrid_scheme_rng`, which
draws, at each step, the components ``(dW, I_1, ..., I_kappa)``, path index
fastest. Child 2 is the spot: at each step, one standard normal per path,
scaled by ``sqrt(dt)``, and nothing else. Child 3 is unused (kept for a
surrogate stream). :meth:`RoughBergomi.volterra_components` is that same
volatility draw when the caller already holds child 1. It does not spawn.
The return is ``(dW, I_1, ..., I_kappa)`` on each step.
:meth:`RoughBergomi.volterra_increments` is the ``dW`` column of that array.

The spot step is the left-point exponential. With ``v`` frozen at the left
end of the step, the multiplier

    exp(sqrt(v) (rho dW + sqrt(1 - rho^2) dW_perp) - v dt / 2)

has conditional mean 1 for any ``rho`` in ``[-1, 0]``. :func:`log_multiplier`
is that exponent. :func:`gauss_hermite_multiplier_mean` integrates it.

The forward variance curve, for ``u > t``,

    xi_t(u) = xi0(u) exp( nu sqrt(2H) I_t(u) - nu^2 (u^{2H} - (u - t)^{2H}) / 2 ),
    I_t(u) = int_0^t (u - s)^{H - 1/2} dW_s.

On each step the integral is the covariance regression of the kernel onto
that step's Wiener increment: the weight times ``dW``. At a tenor equal to
the clock the state stores ``v`` instead, and a tenor already behind the
clock is NaN. ``E[xi_t(u)] = xi0(u)`` when the integral has its exact
variance. The variance swap from ``t`` to ``T`` is ``int_t^T xi_t(u) du``,
left Riemann on the time grid.

``price`` is conditional Monte Carlo. Given the volatility path, ``log Z_T``
is Gaussian with mean ``rho int sqrt(v) dW - int v / 2`` and variance
``(1 - rho^2) int v``. The estimator is the Black price of that conditional
law. Gassiat (2019) is the reference the build plan cites for infinite
moments of ``Z`` when ``|rho|`` is small; this module does not re-derive
that. The conditional estimator stays bounded in normalised units either way.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss

from roughvol.black import implied_vol
from roughvol.black import price as black_price
from roughvol.rng import generator, spawn
from roughvol.types import (
    ParamSpec,
    Paths,
    PriceResult,
    from_unconstrained_value,
    to_unconstrained_value,
)
from roughvol.volterra import hybrid_scheme_rng

Xi0 = float | Callable[[np.ndarray], np.ndarray]
_KAPPA = (1, 2)


def log_multiplier(
    v: np.ndarray,
    dW: np.ndarray,
    dW_perp: np.ndarray,
    rho: float,
    dt: float,
) -> np.ndarray:
    """Exponent of the left-point exponential spot step.

    ``dW`` and ``dW_perp`` are Wiener increments, variance ``dt``, not
    standard normals. ``v`` is the variance at the left end of the step.
    """
    v = np.asarray(v, dtype=np.float64)
    dW = np.asarray(dW, dtype=np.float64)
    dW_perp = np.asarray(dW_perp, dtype=np.float64)
    rho = float(rho)
    orthogonal = math.sqrt(1.0 - rho * rho)
    return np.sqrt(np.maximum(v, 0.0)) * (rho * dW + orthogonal * dW_perp) - 0.5 * v * dt


def gauss_hermite_multiplier_mean(
    v: np.ndarray | float,
    rho: float,
    dt: float,
    *,
    n: int = 32,
) -> np.ndarray:
    """Gaussian integral of ``exp(log_multiplier)`` on one step.

    Tensor Gauss-Hermite, ``n`` nodes on each of the two increments. The
    value is the one-step conditional mean of ``Z``. It equals 1 when the
    exponent is the exponential martingale.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    nodes, weights = hermgauss(int(n))
    z = math.sqrt(2.0) * nodes
    weight = weights / math.sqrt(math.pi)
    v = np.asarray(v, dtype=np.float64)
    scale = math.sqrt(float(dt))
    incr = log_multiplier(
        v[..., None, None],
        scale * z[:, None],
        scale * z[None, :],
        float(rho),
        float(dt),
    )
    quad = weight[:, None] * weight[None, :]
    return np.sum(np.exp(incr) * quad, axis=(-2, -1))


def _spec() -> list[ParamSpec]:
    return [
        ParamSpec("H", lower=0.01, upper=0.49, default=0.1, transform="logit"),
        ParamSpec("nu", lower=1e-4, upper=4.0, default=1.5, transform="log"),
        ParamSpec("rho", lower=-0.999, upper=0.0, default=-0.7, transform="logit"),
    ]


@dataclass(frozen=True)
class RoughBergomi:
    """Rough Bergomi. Defaults are enough for ``RoughBergomi()``.

    ``xi0`` is a positive level, or a callable of an array of times returning
    the same shape. ``xi_tenors`` switches the forward curve on. ``kappa``
    is 1 or 2. ``n_steps`` is what ``price`` uses from 0 to ``T``.
    """

    H: float = 0.1
    nu: float = 1.5
    rho: float = -0.7
    xi0: Xi0 = 0.04
    n_paths: int = 4_000
    n_steps: int = 64
    seed: int = 0
    kappa: int = 1
    xi_tenors: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.H < 1.0:
            raise ValueError(f"H must be in (0, 1), got {self.H}")
        if self.nu < 0.0:
            raise ValueError(f"nu must be >= 0, got {self.nu}")
        if self.rho < -1.0 or self.rho > 0.0:
            raise ValueError(f"rho must be in [-1, 0], got {self.rho}")
        if self.kappa not in _KAPPA:
            raise ValueError(f"kappa must be one of {list(_KAPPA)}, got {self.kappa}")
        if self.n_paths < 2:
            raise ValueError(f"n_paths must be at least 2, got {self.n_paths}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")
        if not callable(self.xi0) and float(self.xi0) <= 0.0:
            raise ValueError(f"xi0 must be positive, got {self.xi0}")
        if self.xi_tenors is not None:
            tenors = np.array(self.xi_tenors, dtype=np.float64, copy=True)
            if tenors.ndim != 1 or tenors.size < 1:
                raise ValueError(f"xi_tenors must be 1-d and non-empty, got shape {tenors.shape}")
            if np.any(np.diff(tenors) <= 0.0):
                raise ValueError("xi_tenors must be strictly increasing")
            object.__setattr__(self, "xi_tenors", tenors)

    def initial_state(self) -> dict[str, np.ndarray]:
        """Variance at 0, and the forward curve on ``xi_tenors`` when set."""
        v0 = float(_xi0_at(self.xi0, np.zeros(1))[0])
        state = {"v": np.array(v0, dtype=np.float64)}
        if self.xi_tenors is not None:
            state["xi"] = _xi0_at(self.xi0, self.xi_tenors)
        return state

    @classmethod
    def param_spec(cls) -> list[ParamSpec]:
        """Fitter coordinates ``H``, ``nu``, ``rho``. ``rho`` stops at 0.

        The constructor still accepts ``H = 1/2`` and ``nu = 0``, which sit
        outside this box. Those are the Black-Scholes and Brownian checks.
        """
        return _spec()

    def to_unconstrained(self) -> np.ndarray:
        """``(H, nu, rho)`` in unconstrained coordinates."""
        specs = _spec()
        return np.array(
            [
                to_unconstrained_value(self.H, specs[0]),
                to_unconstrained_value(self.nu, specs[1]),
                to_unconstrained_value(self.rho, specs[2]),
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_unconstrained(cls, x: np.ndarray) -> RoughBergomi:
        """Inverse of :meth:`to_unconstrained`. ``xi0`` stays at its default."""
        raw = np.asarray(x, dtype=np.float64).reshape(-1)
        specs = _spec()
        if raw.size != len(specs):
            raise ValueError(f"expected {len(specs)} coordinates, got {raw.size}")
        values = {
            spec.name: from_unconstrained_value(float(coord), spec)
            for spec, coord in zip(specs, raw, strict=True)
        }
        return cls(**values)

    def price(self, F, K, T, df=1.0, kind="call", *, state=None) -> np.ndarray:
        """Cash price by conditional Monte Carlo. No standard error on the array."""
        return self.price_with_error(F, K, T, df, kind, state=state).value

    def price_with_error(
        self,
        F,
        K,
        T,
        df=1.0,
        kind="call",
        *,
        state=None,
        n_paths: int | None = None,
        seed: int | np.random.SeedSequence | None = None,
    ) -> PriceResult:
        """Conditional Black price, and the standard error across paths."""
        started = time.perf_counter()
        n_paths = self.n_paths if n_paths is None else int(n_paths)
        seed = self.seed if seed is None else seed
        T = float(T)
        F = _scalar(F, "F")
        df = _scalar(df, "df")
        K_in = np.asarray(K, dtype=np.float64)
        if T < 0.0:
            raise ValueError(f"T must be non-negative, got {T}")
        if T == 0.0:
            value = _intrinsic(F, np.atleast_1d(K_in), df, kind).reshape(K_in.shape)
            stderr = np.zeros(K_in.shape, dtype=np.float64)
            return PriceResult(
                value=value,
                stderr=stderr,
                n_evals=0,
                wall_time=time.perf_counter() - started,
            )
        conditional, _payoff = self.mixing_and_payoff(
            F, K_in, T, df, kind, state=state, n_paths=n_paths, seed=seed
        )
        value = conditional.mean(axis=0).reshape(K_in.shape)
        stderr = (conditional.std(axis=0, ddof=1) / math.sqrt(n_paths)).reshape(K_in.shape)
        return PriceResult(
            value=value,
            stderr=stderr,
            n_evals=n_paths,
            wall_time=time.perf_counter() - started,
        )

    def mixing_and_payoff(
        self,
        F,
        K,
        T,
        df=1.0,
        kind="call",
        *,
        state=None,
        n_paths: int | None = None,
        seed: int | np.random.SeedSequence | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-path conditional Black price and per-path payoff.

        Both arrays have shape ``(m, n_strikes)`` and use one set of paths.
        The conditional column is ``E[payoff | volatility path]`` for the
        discrete model.
        """
        n_paths = self.n_paths if n_paths is None else int(n_paths)
        seed = self.seed if seed is None else seed
        if n_paths < 2:
            raise ValueError(f"n_paths must be at least 2, got {n_paths}")
        T = float(T)
        F = _scalar(F, "F")
        df = _scalar(df, "df")
        times = np.linspace(0.0, T, self.n_steps + 1)
        paths, mu, total_var = self._draw(n_paths, times, seed=seed, state0=state)
        conditional = conditional_prices(F, K, T, df, kind, mu, total_var)
        payoff = _terminal_payoff(F, paths.Z[:, -1], K, df, kind)
        return conditional, payoff

    def atm_skew(
        self,
        T: float,
        *,
        n_paths: int | None = None,
        seed: int | np.random.SeedSequence | None = None,
    ) -> float:
        """Slope of implied volatility in log-moneyness, at the money.

        Nine strikes on ``[-h, h]`` with ``h = max(0.15 sqrt(T), 0.01)``,
        a quadratic fit, and the linear coefficient. Negative when the
        smile falls as the strike rises.
        """
        maturity = float(T)
        n_paths = self.n_paths if n_paths is None else int(n_paths)
        half = max(0.15 * math.sqrt(maturity), 0.01)
        log_money = np.linspace(-half, half, 9)
        strikes = np.exp(log_money)
        result = self.price_with_error(1.0, strikes, maturity, n_paths=n_paths, seed=seed)
        iv = implied_vol(result.value, 1.0, strikes, maturity)
        if not np.all(np.isfinite(iv)):
            raise ValueError(f"implied volatility is not finite at T={maturity}: {iv}")
        return float(np.polyfit(log_money, np.asarray(iv, dtype=np.float64), 2)[1])

    def simulate(
        self,
        n_paths: int,
        times: np.ndarray,
        *,
        seed: int | np.random.SeedSequence,
        state0=None,
    ) -> Paths:
        """Unit-mean ``Z`` on a uniform grid that starts at 0.

        ``state['v']`` has shape ``(m, n + 1)``. ``state['xi']`` is present
        when ``xi_tenors`` is set, with shape ``(m, n + 1, d)``.
        """
        paths, _mu, _total_var = self._draw(int(n_paths), times, seed=seed, state0=state0)
        return paths

    def volterra_components(
        self,
        n_paths: int,
        times: np.ndarray,
        *,
        seed: int | np.random.SeedSequence,
        H: float,
    ) -> np.ndarray:
        """Hybrid-scheme components ``(dW, I_1, ..., I_kappa)`` on each step.

        Shape ``(n_paths, n_steps, kappa + 1)``. ``kappa`` is this model's.
        ``seed`` is already the volatility child. The draw is
        :func:`roughvol.volterra.hybrid_scheme_rng` on ``generator(seed)``,
        with no further spawn. Last-axis index 0 is ``dW``, the Wiener
        increment on that step, variance equal to the step length. Index
        ``k >= 1`` is ``I_k``, the exact integral of the power kernel at
        lag ``k``. At ``H = 1/2`` every ``I_k`` equals ``dW``. The kernel
        weights are not returned.

        ``times`` is the uniform grid :meth:`simulate` accepts. When ``H``
        equals this model's ``H``, these are the components that build
        ``Y`` inside that call.

        A caller rebuilds ``Y`` with a vol of vol on each step. Let ``w``
        be :func:`roughvol.volterra.power_kernel_weights` for this ``H``,
        the step count, the horizon and this ``kappa``. Its first
        ``kappa`` entries are 0. Write ``nu_j`` for the vol of vol on step
        ``j``, and ``k`` for the lag. The contribution of step ``j`` that
        lands at time index ``j + k``, for ``j + k <= n_steps``, is

        ``nu_j * components[:, j, k]`` when ``k <= kappa``, and
        ``nu_j * w[k - 1] * components[:, j, 0]`` when ``k > kappa``.

        Sum the contributions on each time. Column 0 stays 0. A constant
        ``nu`` factors out of that sum, and dividing by ``nu`` returns the
        Riemann-Liouville path inside ``simulate``. The compensator in the
        variance exponential is then ``nu**2 * t**(2H) / 2``. It is the
        exact variance of the integral, the one :meth:`simulate` uses, not
        the scheme's sample variance.

        When ``nu_j`` changes with the step, the same exact compensator on
        the grid is ``sum_j nu_j**2 * (k**(2H) - (k - 1)**(2H)) * dt**(2H) / 2``,
        with ``k`` the number of steps from ``j`` to the evaluation time
        and ``dt`` the step length. Replacing ``I_1`` by the lag-1 Riemann
        weight keeps only ``2H / (H + 1/2)**2`` of ``Var(I_1)``. At
        ``H = 0.1`` that ratio is ``5/9``, so ``4/9`` of each step's
        own-term variance is missing and ``E[v]`` falls short of ``xi0``.
        """
        return self._hybrid(n_paths, times, seed=seed, H=H).components

    def volterra_increments(
        self,
        n_paths: int,
        times: np.ndarray,
        *,
        seed: int | np.random.SeedSequence,
        H: float,
    ) -> np.ndarray:
        """Wiener increments, column 0 of :meth:`volterra_components`.

        Shape ``(n_paths, n_steps)``. Column ``k`` is
        ``W(times[k + 1]) - W(times[k])``, with variance equal to the step
        length. ``dW`` alone does not rebuild ``Y``: the near-field
        integrals ``I_1, ..., I_kappa`` are on :meth:`volterra_components`.
        """
        return self.volterra_components(n_paths, times, seed=seed, H=H)[..., 0]

    def _hybrid(self, n_paths: int, times: np.ndarray, *, seed, H: float):
        times = _uniform_times(times)
        n_paths = int(n_paths)
        if n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        n = times.size - 1
        horizon = float(times[-1])
        return hybrid_scheme_rng(
            float(H), n, horizon, n_paths, generator(seed), kappa=self.kappa
        )

    def _draw(self, n_paths: int, times: np.ndarray, *, seed, state0):
        times = _uniform_times(times)
        if n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        xi0 = _xi0_from_state(self, state0)
        n = times.size - 1
        horizon = float(times[-1])
        children = spawn(seed, 4)
        vol = hybrid_scheme_rng(
            self.H, n, horizon, n_paths, generator(children[1]), kappa=self.kappa
        )
        level = _xi0_at(xi0, times)
        v = _variance_from_driver(self.H, self.nu, level, times, vol.Y)
        spot = generator(children[2])
        Z, mu, total_var = _spot_and_mixing(v, vol.dW, times, self.rho, spot)
        state: dict[str, np.ndarray] = {"v": v}
        if self.xi_tenors is not None:
            state["xi"] = _forward_curve(
                self.H, self.nu, xi0, times, vol.dW, v, np.asarray(self.xi_tenors)
            )
            state["xi_tenors"] = np.asarray(self.xi_tenors, dtype=np.float64)
        return Paths(times=times, Z=Z, state=state), mu, total_var


def euler_bergomi(
    model: RoughBergomi,
    n_paths: int,
    times: np.ndarray,
    *,
    seed: int | np.random.SeedSequence,
) -> tuple[Paths, np.ndarray, np.ndarray]:
    """Full-truncation Euler of ``dv = nu v dW`` at ``H = 1/2``.

    The update is ``v <- v + nu max(v, 0) dW``. The spot step is the same
    left-point exponential as :meth:`RoughBergomi.simulate`, and the spawn
    tree is the same four children, so a paired comparison uses the same
    normals. The return is ``(paths, mu, total_var)`` for the conditional
    estimator. ``xi0`` must be a constant: a shaped forward curve is not
    this diffusion.
    """
    if abs(model.H - 0.5) > 1e-12:
        raise ValueError(f"euler_bergomi is the H = 1/2 diffusion, got H = {model.H}")
    if callable(model.xi0):
        raise ValueError("euler_bergomi takes a flat xi0")
    times = _uniform_times(times)
    n_paths = int(n_paths)
    if n_paths < 1:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    n = times.size - 1
    children = spawn(seed, 4)
    vol_rng = generator(children[1])
    spot_rng = generator(children[2])
    dW = np.empty((n_paths, n), dtype=np.float64)
    v = np.empty((n_paths, n + 1), dtype=np.float64)
    v[:, 0] = float(model.xi0)
    for j in range(n):
        dt = float(times[j + 1] - times[j])
        shock = vol_rng.standard_normal(n_paths)
        dW[:, j] = math.sqrt(dt) * shock
        v[:, j + 1] = v[:, j] + model.nu * np.maximum(v[:, j], 0.0) * dW[:, j]
    Z, mu, total_var = _spot_and_mixing(v, dW, times, model.rho, spot_rng)
    return Paths(times=times, Z=Z, state={"v": v}), mu, total_var


def variance_swap(paths: Paths) -> np.ndarray:
    """Left Riemann ``int_t^T xi_t(u) du`` on the time grid. Shape ``(m, n + 1)``.

    ``state['xi']`` tenors must be the time grid. Column ``i`` is the swap
    seen at ``times[i]``. The last column is 0.
    """
    xi = _xi_on_time_grid(paths)
    dt = np.diff(paths.times)
    weighted = np.zeros_like(xi)
    weighted[:, :, :-1] = xi[:, :, :-1] * dt[None, None, :]
    tail = np.cumsum(weighted[:, :, ::-1], axis=2)[:, :, ::-1]
    index = np.arange(paths.times.size)
    return tail[:, index, index]


def realised_variance(paths: Paths) -> np.ndarray:
    """Left Riemann ``int_t^T v_u du``. Shape ``(m, n + 1)``. The last column is 0."""
    v = paths.state["v"]
    dt = np.diff(paths.times)
    incr = v[:, :-1] * dt[None, :]
    out = np.zeros_like(v)
    out[:, :-1] = np.cumsum(incr[:, ::-1], axis=1)[:, ::-1]
    return out


def _variance_from_driver(H, nu, level, times, Y) -> np.ndarray:
    power = np.zeros_like(times)
    positive = times > 0.0
    power[positive] = times[positive] ** (2.0 * H)
    exponent = nu * math.sqrt(2.0 * H) * Y - 0.5 * nu * nu * power
    return level * np.exp(exponent)


def _spot_and_mixing(v, dW, times, rho, spot_rng):
    m, n_cols = v.shape
    n = n_cols - 1
    Z = np.ones((m, n_cols), dtype=np.float64)
    log_z = np.zeros(m, dtype=np.float64)
    mu = np.zeros(m, dtype=np.float64)
    integrated = np.zeros(m, dtype=np.float64)
    for j in range(n):
        dt = float(times[j + 1] - times[j])
        v_left = v[:, j]
        step = dW[:, j]
        perp = math.sqrt(dt) * spot_rng.standard_normal(m)
        log_z = log_z + log_multiplier(v_left, step, perp, rho, dt)
        Z[:, j + 1] = np.exp(log_z)
        sqrt_v = np.sqrt(np.maximum(v_left, 0.0))
        mu = mu + rho * sqrt_v * step - 0.5 * v_left * dt
        integrated = integrated + v_left * dt
    total_var = (1.0 - rho * rho) * integrated
    return Z, mu, total_var


def _forward_curve(H, nu, xi0, times, dW, v, tenors) -> np.ndarray:
    """``xi`` of shape ``(m, n + 1, d)``."""
    m = dW.shape[0]
    d = tenors.size
    weights = _step_weights(H, times, tenors)
    incr = dW[:, :, None] * weights[None, :, :]
    integral = np.zeros((m, times.size, d), dtype=np.float64)
    integral[:, 1:, :] = np.cumsum(incr, axis=1)
    gap = tenors[None, :] - times[:, None]
    tol = 1e-12 * max(1.0, float(times[-1]), float(np.max(np.abs(tenors))))
    ahead = gap > tol
    safe = np.where(ahead, gap, 1.0)
    u_power = tenors ** (2.0 * H)
    compensator = np.where(
        ahead,
        0.5 * nu * nu * (u_power[None, :] - safe ** (2.0 * H)),
        0.0,
    )
    exponent = nu * math.sqrt(2.0 * H) * integral - compensator[None, :, :]
    level = _xi0_at(xi0, tenors)
    xi = level[None, None, :] * np.exp(exponent)
    on_clock = np.abs(gap) <= tol
    if np.any(on_clock):
        rows, cols = np.nonzero(on_clock)
        xi[:, rows, cols] = v[:, rows]
    behind = gap < -tol
    if np.any(behind):
        xi = np.where(behind[None, :, :], np.nan, xi)
    return xi


def _step_weights(H: float, times: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Covariance weight of ``(u - s)^{H - 1/2}`` on each step. Shape ``(n, d)``.

    The weight is ``int_{step} (u - s)^{alpha} ds / dt``. Times the Wiener
    increment, that is the regression of the kernel integral onto ``dW``.
    A step that finishes after the tenor gets weight 0.
    """
    alpha = H - 0.5
    t0 = times[:-1]
    t1 = times[1:]
    dt = t1 - t0
    left = tenors[None, :] - t0[:, None]
    right = tenors[None, :] - t1[:, None]
    if abs(alpha) < 1e-12:
        weights = np.ones((t0.size, tenors.size), dtype=np.float64)
    else:
        power = alpha + 1.0
        integral = (np.maximum(left, 0.0) ** power - np.maximum(right, 0.0) ** power) / power
        weights = integral / dt[:, None]
    return np.where(right >= -1e-14, weights, 0.0)


def conditional_prices(F, K, T, df, kind, mu, total_var) -> np.ndarray:
    """Black price of each path's conditional law. Shape ``(m, n_strikes)``.

    ``mu`` is ``E[log Z_T | volatility]`` and ``total_var`` is the conditional
    variance of ``log Z_T``. The forward is ``F exp(mu + total_var / 2)``.
    """
    K_flat = np.atleast_1d(np.asarray(K, dtype=np.float64))
    sigma = np.sqrt(np.maximum(total_var, 0.0) / T)
    forward = np.asarray(F, dtype=np.float64) * np.exp(mu + 0.5 * total_var)
    return black_price(forward[:, None], K_flat[None, :], T, sigma[:, None], df, kind)


def _terminal_payoff(F, Z, K, df, kind) -> np.ndarray:
    K_flat = np.atleast_1d(np.asarray(K, dtype=np.float64))
    forward = F * Z[:, None]
    call = np.maximum(forward - K_flat[None, :], 0.0)
    put = np.maximum(K_flat[None, :] - forward, 0.0)
    return df * _select_kind(call, put, kind, K_flat.shape)


def _intrinsic(F, K, df, kind) -> np.ndarray:
    call = np.maximum(F - K, 0.0)
    put = np.maximum(K - F, 0.0)
    return df * _select_kind(call, put, kind, K.shape)


def _select_kind(call, put, kind, shape) -> np.ndarray:
    if isinstance(kind, str):
        if kind == "call":
            return call
        if kind == "put":
            return put
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    kinds = np.asarray(kind)
    if kinds.shape != shape:
        raise ValueError(f"kind shape {kinds.shape} does not match strikes {shape}")
    if not np.all((kinds == "call") | (kinds == "put")):
        raise ValueError(f"kind must be 'call' or 'put', got {kinds}")
    return np.where(kinds == "call", call, put)


def _xi0_at(xi0: Xi0, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if callable(xi0):
        raw = np.asarray(xi0(t), dtype=np.float64)
    else:
        raw = np.full(t.shape, float(xi0), dtype=np.float64)
    if raw.shape != t.shape:
        raise ValueError(f"xi0 returned shape {raw.shape}, expected {t.shape}")
    if np.any(~np.isfinite(raw) | (raw <= 0.0)):
        raise ValueError("xi0 must be finite and positive")
    return raw


def _xi0_from_state(model: RoughBergomi, state) -> Xi0:
    if not state:
        return model.xi0
    unknown = set(state) - {"v", "xi", "xi_tenors"}
    if unknown:
        raise ValueError(f"state keys must be 'v' and 'xi', got {sorted(unknown)}")
    if "xi" not in state:
        v0 = np.asarray(state["v"], dtype=np.float64)
        if v0.shape not in ((),) and v0.size != 1:
            raise ValueError(f"state['v'] must be a scalar level, got shape {v0.shape}")
        return float(v0)
    curve = np.asarray(state["xi"], dtype=np.float64)
    if curve.ndim == 0 or curve.shape == ():
        level = float(curve)
        _check_v_matches(state, level)
        return level
    if curve.ndim != 1:
        raise ValueError(f"state['xi'] must be a scalar or a 1-d curve, got shape {curve.shape}")
    if model.xi_tenors is None or model.xi_tenors.shape != curve.shape:
        raise ValueError(
            "state['xi'] length must match xi_tenors, "
            f"got {curve.shape} and {None if model.xi_tenors is None else model.xi_tenors.shape}"
        )
    if np.any(curve <= 0.0):
        raise ValueError("state['xi'] must be positive")
    tenors = np.asarray(model.xi_tenors, dtype=np.float64)
    _check_v_matches(state, float(np.interp(0.0, tenors, curve)))

    def xi0(t, tenors=tenors, curve=curve):
        return np.interp(np.asarray(t, dtype=np.float64), tenors, curve)

    return xi0


def _check_v_matches(state, level: float) -> None:
    if state is None or "v" not in state:
        return
    v0 = np.asarray(state["v"], dtype=np.float64)
    if v0.size != 1:
        raise ValueError(f"state['v'] must be a scalar level, got shape {v0.shape}")
    if abs(float(v0) - level) > 1e-8 * max(1.0, abs(level)):
        raise ValueError(f"state['v'] is {float(v0)} and the curve at 0 is {level}")


def _uniform_times(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or times[0] != 0.0:
        raise ValueError("times must be 1-d, start at 0, and contain at least one step")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    gaps = np.diff(times)
    if float(np.max(np.abs(gaps - gaps[0]))) > 1e-12 * max(1.0, float(times[-1])):
        raise ValueError("times must be equally spaced")
    return times


def _xi_on_time_grid(paths: Paths) -> np.ndarray:
    if "xi" not in paths.state:
        raise ValueError("variance_swap needs state['xi']")
    xi = paths.state["xi"]
    tenors = paths.state.get("xi_tenors")
    if tenors is None:
        raise ValueError("variance_swap needs state['xi_tenors']")
    same_grid = tenors.shape == paths.times.shape and np.allclose(
        tenors, paths.times, rtol=0.0, atol=1e-12
    )
    if not same_grid:
        raise ValueError("variance_swap needs xi_tenors equal to the time grid")
    return xi


def _scalar(value, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    return float(array)
