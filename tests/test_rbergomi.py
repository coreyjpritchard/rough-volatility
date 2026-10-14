"""Rough Bergomi: martingales, the forward curve, conditional Monte Carlo, skew.

The one-step spot multiplier is integrated by 32-point Gauss-Hermite on each
increment. That integral is the exact martingale check: it must be 1 to 1e-12.

Monte Carlo checks use the z-rule. Paths inside one replication share a
seed, so the grid keeps the max bound, and the mean of z is judged on R
replications. Sample sizes are on the constants below.

The skew bar was written before the first run of ``test_skew_exponent``.
The tolerance stays 0.05 and applies to the mean of 12 seeds. One seed is
noisier than that band.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from roughvol.black import implied_vol
from roughvol.black import price as black_price
from roughvol.rbergomi import (
    RoughBergomi,
    conditional_prices,
    euler_bergomi,
    gauss_hermite_multiplier_mean,
    log_multiplier,
    realised_variance,
    variance_swap,
)
from roughvol.rng import generator, spawn
from roughvol.volterra import hybrid_scheme_rng
from tests.zrule import check_replication_means, check_z

SEED = 20261014
M = 2_000
N = 64
R = 4
T_GRID = 1.0

# Skew exponent at H = 0.1. Target H - 1/2, tolerance 0.05, on T in [1/52, 1/4].
# Strike window half-width max(0.15 sqrt(T), 0.01), nine log-moneynesses, quadratic fit.
SKEW_H = 0.1
SKEW_NU = 1.5
SKEW_RHO = -0.7
SKEW_XI0 = 0.04
SKEW_T_MIN = 1.0 / 52.0
SKEW_T_MAX = 0.25
SKEW_TOL = 0.05
SKEW_PATHS = 8_000
SKEW_STEPS = 128
SKEW_MATURITIES = 6
SKEW_REPS = 12


def _shared(rows):
    centres = []
    for z in rows:
        failure = check_z(np.asarray(z, dtype=np.float64), mean=False, fraction=False)
        assert failure is None, failure
        centres.append(float(np.mean(z)))
    failure = check_replication_means(np.array(centres))
    assert failure is None, failure


def _mean_z(sample, target):
    sample = np.asarray(sample, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    se = sample.std(axis=0, ddof=1) / math.sqrt(sample.shape[0])
    return (sample.mean(axis=0) - target) / se


def _atm_skew(model: RoughBergomi, maturity: float, n_paths: int, seed: int) -> float:
    slope = model.atm_skew(maturity, n_paths=n_paths, seed=seed)
    if not slope < 0.0:
        raise AssertionError(f"ATM skew at T={maturity} is {slope}, expected negative")
    return slope


def test_black_atm_call_matches_the_normal_integral():
    # F = K = 1, sigma = 0.2, T = 1, df = 1: N(d1) - N(d2) with d1 = 0.1.
    assert black_price(1.0, 1.0, 1.0, 0.2) == pytest.approx(0.07965567455405798, abs=1e-14)
    assert implied_vol(0.07965567455405798, 1.0, 1.0, 1.0) == pytest.approx(0.2, abs=1e-10)


def test_one_step_multiplier_is_exactly_a_martingale():
    variances = np.array([1e-6, 0.04, 1.0])
    worst = 0.0
    for rho in (-1.0, -0.95, -0.7, -0.2, 0.0):
        for dt in (1e-4, 1.0 / 64.0, 0.25, 1.0):
            got = gauss_hermite_multiplier_mean(variances, rho, dt, n=32)
            worst = max(worst, float(np.max(np.abs(got - 1.0))))
    assert worst < 1e-12


def test_nu_zero_matches_black_scholes():
    """Flat xi0 = 0.04, nu = 0, rho = -0.7. m = 4,000 paths, 5 steps, R = 4.

    The conditional estimator is not constant, because rho is not 0. The
    standard error is the one ``price_with_error`` reports. Nine nodes:
    three strikes times three maturities.
    """
    strikes = np.array([0.8, 1.0, 1.2])
    maturities = (0.25, 0.5, 1.0)
    sigma = math.sqrt(0.04)
    rows = []
    for r in range(R):
        z_parts = []
        for i, maturity in enumerate(maturities):
            model = RoughBergomi(
                nu=0.0, rho=-0.7, xi0=0.04, n_steps=5, n_paths=4_000, seed=SEED + r
            )
            result = model.price_with_error(
                1.0, strikes, maturity, n_paths=4_000, seed=SEED + 100 * r + i
            )
            ref = black_price(1.0, strikes, maturity, sigma)
            z_parts.append((result.value - ref) / result.stderr)
        rows.append(np.concatenate(z_parts))
    _shared(rows)


def test_nu_zero_and_rho_zero_is_exact_black():
    model = RoughBergomi(nu=0.0, rho=0.0, xi0=0.04, n_steps=4, n_paths=16)
    strikes = np.array([0.7, 1.0, 1.3])
    got = model.price(1.0, strikes, 1.0)
    ref = black_price(1.0, strikes, 1.0, 0.2)
    assert np.max(np.abs(got - ref)) < 1e-12


def test_state_xi_replaces_the_forward_curve():
    tenors = np.array([0.0, 1.0])
    model = RoughBergomi(nu=0.0, rho=0.0, xi0=0.04, n_steps=4, n_paths=16, xi_tenors=tenors)
    got = model.price(1.0, 1.0, 1.0, state={"xi": np.array([0.09, 0.09])})
    assert got == pytest.approx(float(black_price(1.0, 1.0, 1.0, 0.3)), abs=1e-12)


def test_spot_and_variance_martingales_and_forward_curve():
    """E[Z_t] = 1 and E[v_t] = xi0(t) on 64 steps, and E[xi_t(u)] = xi0(u).

    m = 2,000, R = 4, H = 0.1, nu = 1.5, kappa = 1, T = 1.
    rho = -0.9 so that 1 / (1 - rho^2) = 5.26 and the second moment of Z is
    inside the range Gassiat's condition leaves finite. xi0(t) = 0.03 + 0.02 t.
    The standard error of each node is the sample standard deviation over the
    m paths, divided by sqrt(m). t = 0 is exact and is checked separately.
    """

    def xi0(t):
        return 0.03 + 0.02 * np.asarray(t, dtype=np.float64)

    tenors = np.array([0.25, 0.5, 1.0, 1.5])
    times = np.linspace(0.0, T_GRID, N + 1)
    z_rows = []
    v_rows = []
    xi_rows = []
    for r in range(R):
        model = RoughBergomi(
            H=0.1,
            nu=1.5,
            rho=-0.9,
            xi0=xi0,
            kappa=1,
            n_paths=M,
            n_steps=N,
            xi_tenors=tenors,
        )
        paths = model.simulate(M, times, seed=SEED + 1_000 + r)
        if r == 0:
            gap = np.max(np.abs(paths.state["xi"][:, 0, :] - xi0(tenors)))
            assert gap < 1e-12
        z_rows.append(_mean_z(paths.Z[:, 1:], np.ones(N)))
        v_rows.append(_mean_z(paths.state["v"][:, 1:], xi0(times[1:])))
        xi = paths.state["xi"][:, 1:, :]
        target = np.broadcast_to(xi0(tenors), (N, tenors.size))
        usable = times[1:, None] <= tenors[None, :] + 1e-12
        xi_rows.append(_mean_z(xi[:, usable], target[usable]))
    _shared(z_rows)
    _shared(v_rows)
    _shared(xi_rows)


def test_h_half_forward_curve_equals_spot_variance():
    tenors = np.array([0.3, 0.5, 1.0, 1.25])
    times = np.linspace(0.0, 1.0, 17)
    model = RoughBergomi(
        H=0.5, nu=1.2, rho=-0.3, xi0=0.04, n_paths=32, n_steps=16, xi_tenors=tenors
    )
    paths = model.simulate(32, times, seed=SEED + 7)
    xi = paths.state["xi"]
    v = paths.state["v"]
    for i, t in enumerate(times):
        for d, u in enumerate(tenors):
            if u > t + 1e-12:
                assert np.max(np.abs(xi[:, i, d] - v[:, i])) < 1e-10
            elif u < t - 1e-12:
                assert np.all(np.isnan(xi[:, i, d]))


def test_variance_swap_matches_realised_variance():
    """Left Riemann on the time grid. m = 2,000, n = 32, R = 4, H = 0.1.

    At H = 1/2 the swap equals v_t times the time left, to 1e-10, because
    xi_t(u) = v_t. The z-rule is the H = 0.1 comparison of the swap with
    realised variance, node by node in time. The standard error is the
    sample standard deviation of the pathwise difference, over sqrt(m).
    """
    n = 32
    times = np.linspace(0.0, 1.0, n + 1)
    flat = RoughBergomi(H=0.5, nu=0.8, rho=-0.4, xi0=0.04, n_paths=64, n_steps=n, xi_tenors=times)
    exact = flat.simulate(64, times, seed=SEED + 8)
    remaining = times[-1] - times
    swap = variance_swap(exact)
    assert np.max(np.abs(swap - exact.state["v"] * remaining)) < 1e-10

    rows = []
    for r in range(R):
        model = RoughBergomi(
            H=0.1, nu=1.5, rho=-0.7, xi0=0.04, n_paths=M, n_steps=n, xi_tenors=times.copy()
        )
        paths = model.simulate(M, times, seed=SEED + 2_000 + r)
        # The last step uses v on both sides, so the difference is identically 0.
        tail = realised_variance(paths)[:, -2] - variance_swap(paths)[:, -2]
        assert np.max(np.abs(tail)) < 1e-12
        diff = realised_variance(paths)[:, :-2] - variance_swap(paths)[:, :-2]
        rows.append(_mean_z(diff, np.zeros(n - 1)))
    _shared(rows)


def test_h_half_hybrid_matches_euler_prices():
    """Same normals, full-truncation Euler against the hybrid scheme.

    m = 2,000 paths, 256 steps, T = 1, R = 4, strikes 0.9, 1 and 1.1.
    nu = 1.5, rho = -0.7, xi0 = 0.04. The z score is the paired difference
    of conditional prices, standard error std(diff) / sqrt(m).
    """
    n_steps = 256
    strikes = np.array([0.9, 1.0, 1.1])
    rows = []
    for r in range(R):
        seed = SEED + 3_000 + r
        model = RoughBergomi(H=0.5, nu=1.5, rho=-0.7, xi0=0.04, n_steps=n_steps, n_paths=M, kappa=1)
        hybrid, _payoff = model.mixing_and_payoff(1.0, strikes, 1.0, n_paths=M, seed=seed)
        _paths, mu, total_var = euler_bergomi(
            model, M, np.linspace(0.0, 1.0, n_steps + 1), seed=seed
        )
        euler = conditional_prices(1.0, strikes, 1.0, 1.0, "call", mu, total_var)
        diff = hybrid - euler
        se = diff.std(axis=0, ddof=1) / math.sqrt(M)
        rows.append(diff.mean(axis=0) / se)
    _shared(rows)


def test_conditional_monte_carlo_matches_plain_where_variance_is_finite():
    """rho = -0.85, rho^2 = 0.7225 > 1/2. m = 2,000, 32 steps, T = 0.5, R = 4.

    Strikes 0.8, 1 and 1.2. The z score uses the paired difference of the
    conditional price and the payoff, standard error std(diff) / sqrt(m).
    The conditional standard error is at most the plain one on each strike.
    """
    strikes = np.array([0.8, 1.0, 1.2])
    rows = []
    for r in range(R):
        model = RoughBergomi(H=0.1, nu=1.5, rho=-0.85, xi0=0.04, n_steps=32, n_paths=M)
        conditional, payoff = model.mixing_and_payoff(
            1.0, strikes, 0.5, n_paths=M, seed=SEED + 4_000 + r
        )
        se_cond = conditional.std(axis=0, ddof=1)
        se_plain = payoff.std(axis=0, ddof=1)
        assert np.all(se_cond <= se_plain)
        diff = conditional - payoff
        se = diff.std(axis=0, ddof=1) / math.sqrt(M)
        rows.append(diff.mean(axis=0) / se)
    _shared(rows)


def _skew_seed(rep: int, index: int) -> int:
    """Seed for replication ``rep`` and maturity ``index`` on the skew grid.

    Replication 0 is the original single draw: seed 20261014 + 5000 + index.
    """
    return SEED + 5_000 + 100 * rep + index


def _skew_slopes(model: RoughBergomi, maturities: np.ndarray) -> np.ndarray:
    slopes = []
    for rep in range(SKEW_REPS):
        skew = np.array(
            [
                _atm_skew(model, float(maturity), SKEW_PATHS, _skew_seed(rep, i))
                for i, maturity in enumerate(maturities)
            ]
        )
        slopes.append(float(np.polyfit(np.log(maturities), np.log(np.abs(skew)), 1)[0]))
    return np.asarray(slopes, dtype=np.float64)


def test_skew_exponent():
    """|psi| ~ T^{H - 1/2} on T in [1/52, 1/4] at H = 0.1.

    m = 8,000 paths, 128 steps, nu = 1.5, rho = -0.7, xi0 = 0.04.
    Six maturities on a geometric grid, repeated on 12 seeds. One seed has
    standard deviation 0.0181 in this sample, so a band of 0.05 is under
    three standard deviations and is not a check on a single draw. The
    prewritten bar of 0.05 applies to the mean.
    """
    model = RoughBergomi(
        H=SKEW_H,
        nu=SKEW_NU,
        rho=SKEW_RHO,
        xi0=SKEW_XI0,
        n_steps=SKEW_STEPS,
        n_paths=SKEW_PATHS,
        kappa=1,
    )
    maturities = np.geomspace(SKEW_T_MIN, SKEW_T_MAX, SKEW_MATURITIES)
    slopes = _skew_slopes(model, maturities)
    mean = float(slopes.mean())
    sd = float(slopes.std(ddof=1))
    target = SKEW_H - 0.5
    assert slopes.size == SKEW_REPS
    assert sd > 0.0
    assert abs(mean - target) < SKEW_TOL, (
        f"mean skew exponent {mean:.4f} (sd {sd:.4f}, {SKEW_REPS} seeds) "
        f"on T in [{SKEW_T_MIN:.6f}, {SKEW_T_MAX}], "
        f"target {target:.4f}, tolerance {SKEW_TOL}, slopes {slopes}"
    )


def test_param_spec_keeps_rho_non_positive():
    model = RoughBergomi()
    specs = {spec.name: spec for spec in RoughBergomi.param_spec()}
    assert specs["rho"].upper == 0.0
    assert specs["rho"].transform == "logit"
    back = RoughBergomi.from_unconstrained(model.to_unconstrained())
    assert back.H == pytest.approx(model.H, abs=1e-12)
    assert back.nu == pytest.approx(model.nu, abs=1e-12)
    assert back.rho <= 0.0
    assert back.rho == pytest.approx(model.rho, abs=1e-12)
    high = RoughBergomi.from_unconstrained(np.array([0.0, 0.0, 40.0]))
    assert high.rho <= 0.0


def test_positive_rho_is_rejected():
    with pytest.raises(ValueError, match="rho"):
        RoughBergomi(rho=0.1)
    with pytest.raises(ValueError, match="equally spaced"):
        RoughBergomi().simulate(4, np.array([0.0, 0.1, 0.4]), seed=1)


def test_volatility_driver_is_spawn_child_1():
    times = np.linspace(0.0, 0.5, 9)
    model = RoughBergomi(H=0.1, nu=1.5, rho=-0.4, xi0=0.05, kappa=2, n_paths=6, n_steps=8)
    paths = model.simulate(6, times, seed=11)
    again = model.simulate(6, times, seed=11)
    assert np.array_equal(paths.Z, again.Z)
    hybrid = hybrid_scheme_rng(0.1, 8, 0.5, 6, generator(spawn(11, 4)[1]), kappa=2)
    power = np.zeros_like(times)
    power[1:] = times[1:] ** 0.2
    v = 0.05 * np.exp(1.5 * math.sqrt(0.2) * hybrid.Y - 0.5 * 1.5**2 * power)
    assert np.allclose(paths.state["v"], v)
    spot = generator(spawn(11, 4)[2])
    log_z = np.zeros(6)
    Z = np.ones_like(paths.Z)
    for j in range(8):
        dt = float(times[j + 1] - times[j])
        perp = math.sqrt(dt) * spot.standard_normal(6)
        log_z = log_z + log_multiplier(v[:, j], hybrid.dW[:, j], perp, -0.4, dt)
        Z[:, j + 1] = np.exp(log_z)
    assert np.allclose(paths.Z, Z)


def test_entry_point_loads_rough_bergomi():
    from importlib.metadata import entry_points

    matches = [ep for ep in entry_points(group="volsurface.models") if ep.name == "rbergomi"]
    assert len(matches) == 1
    assert matches[0].load() is RoughBergomi
    assert matches[0].load()().rho <= 0.0


def test_volsurface_create_resolves_rbergomi():
    """Skipped when volsurface, or its plugins module, is not installed."""
    plugins = pytest.importorskip("volsurface.plugins")

    model = plugins.create("rbergomi")
    assert isinstance(model, RoughBergomi)
    assert model.price(1.0, np.array([1.0]), 0.25).shape == (1,)


def test_volterra_increments_match_simulate_dw():
    """The volatility child, with no further spawn, is the dW inside simulate.

    Parent seed 11. The method receives child 1. H = 0.1 uses the Cholesky
    path at kappa = 2. H = 1/2 uses the single-normal path at kappa = 1.
    """
    times = np.linspace(0.0, 0.5, 9)
    parent = 11
    for H, kappa in ((0.1, 2), (0.5, 1)):
        model = RoughBergomi(
            H=H, nu=1.5, rho=-0.4, xi0=0.05, kappa=kappa, n_paths=6, n_steps=8
        )
        paths = model.simulate(6, times, seed=parent)
        child = spawn(parent, 4)[1]
        dW = model.volterra_increments(6, times, seed=child, H=H)
        again = model.volterra_increments(6, times, seed=child, H=H)
        hybrid = hybrid_scheme_rng(H, 8, 0.5, 6, generator(child), kappa=kappa)
        assert dW.shape == (6, 8)
        assert np.array_equal(dW, again)
        assert np.array_equal(dW, hybrid.dW)
        power = np.zeros_like(times)
        power[1:] = times[1:] ** (2.0 * H)
        v = 0.05 * np.exp(1.5 * math.sqrt(2.0 * H) * hybrid.Y - 0.5 * 1.5**2 * power)
        assert np.allclose(paths.state["v"], v)
        spot = generator(spawn(parent, 4)[2])
        log_z = np.zeros(6)
        Z = np.ones_like(paths.Z)
        for j in range(8):
            dt = float(times[j + 1] - times[j])
            perp = math.sqrt(dt) * spot.standard_normal(6)
            log_z = log_z + log_multiplier(v[:, j], dW[:, j], perp, -0.4, dt)
            Z[:, j + 1] = np.exp(log_z)
        assert np.allclose(paths.Z, Z)


def test_volterra_increment_variance_equals_dt():
    """Second moment of dW equals the step length. m = 2,000, n = 64, R = 4.

    A centred normal has Var(X^2) = 2 dt^2, so the standard error of the
    mean second moment is dt * sqrt(2 / m). The grid shares paths, so each
    replication keeps the max bound and the mean of z is judged on R.
    """
    model = RoughBergomi(H=0.1, kappa=1)
    n_steps = 64
    times = np.linspace(0.0, 1.0, n_steps + 1)
    dt = 1.0 / n_steps
    rows = []
    for r in range(R):
        dW = model.volterra_increments(M, times, seed=SEED + 7_000 + r, H=0.1)
        assert dW.shape == (M, n_steps)
        moment = np.mean(dW**2, axis=0)
        rows.append((moment - dt) / (dt * np.sqrt(2.0 / M)))
    _shared(rows)


def test_volterra_increments_rejects_non_increasing_times():
    model = RoughBergomi()
    with pytest.raises(ValueError, match="strictly increasing"):
        model.volterra_increments(4, np.array([0.0, 0.3, 0.2, 0.5]), seed=1, H=0.1)
    with pytest.raises(ValueError, match="1-d"):
        model.volterra_increments(4, np.array([[0.0, 0.25], [0.5, 1.0]]), seed=1, H=0.1)
