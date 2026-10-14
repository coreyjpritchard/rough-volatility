# Rough Bergomi

The figure `docs/figures/rvb_skew.png` is the absolute at-the-money skew of rough Bergomi against maturity, for H in {0.05, 0.1, 0.2, 0.5}. Vol of vol is nu = 1.5, correlation rho = -0.7, and the forward variance is flat at 0.04. Each point is the mean of 12 seeds. On T in [1/52, 1/4] the mean log-log slopes are -0.460, -0.410, -0.323 and -0.017. The short-dated claim is a slope of H - 1/2, which is -0.450, -0.400, -0.300 and 0.

## The model

Bayer, Friz and Gatheral (2016), in the forward coordinate. Y is the Riemann-Liouville process from the hybrid scheme, with Var(Y_t) = t^{2H} / (2H) when the draw is exact:

    v_t = xi0(t) exp(nu sqrt(2H) Y_t - nu^2 t^{2H} / 2)
    dZ / Z = sqrt(v) (rho dW + sqrt(1 - rho^2) dW_perp),    Z_0 = 1

The factor sqrt(2H) puts the variance of nu sqrt(2H) Y back to nu^2 t^{2H}, so the exponential is a martingale and E[v_t] = xi0(t). nu is the vol of vol. rho is at most 0: the constructor rejects a positive value, and the fitter coordinate stops at 0. The fitter box for H is (0.01, 0.49). The constructor still accepts H = 1/2 and nu = 0, which are the Brownian and Black-Scholes checks.

xi0 is a positive level, or a function of time. A host can also pass state["xi"], a scalar or a curve on xi_tenors. That starts a fresh model on the curve.

## The forward variance curve

For u > t,

    xi_t(u) = xi0(u) exp( nu sqrt(2H) I_t(u) - nu^2 (u^{2H} - (u - t)^{2H}) / 2 )
    I_t(u) = int_0^t (u - s)^{H - 1/2} dW_s

On each time step the integral is replaced by one number times that step's Wiener increment. The number is the integral of the kernel over the step, divided by dt, which is the covariance regression of the kernel integral onto dW. A tenor equal to the clock stores v. A tenor behind the clock is missing. At t = 0 the integral is 0, so xi_0(u) = xi0(u) with no simulation error. On the run in the log that gap is 0.

At H = 1/2 the kernel is 1, so I_t(u) = W_t = Y_t and the compensator is nu^2 t / 2. For a flat xi0 this is v_t at every tenor ahead of the clock. On 32 paths and 16 steps the max absolute gap is 0.

The variance swap from t to T is the integral of xi_t. On the time grid that integral is a left Riemann sum. The last step uses v on both the swap and the realised side, so the difference is 0 and it has no standard error. It is checked to 1e-12 and left out of the z-rule.

## Pricing

Given the volatility path, log Z_T in the discrete model is Gaussian. Its mean is rho times the sum of sqrt(v) dW, minus half the sum of v dt. Its variance is (1 - rho^2) times the sum of v dt. The estimator is the Black price of that conditional law. The orthogonal draw still builds the path Z, and the plain payoff is kept for the comparison where both variances are finite.

The build plan cites Gassiat (2019) for infinite moments of Z when |rho| is small enough that the moment order exceeds 1 / (1 - rho^2). This note does not re-derive that. The conditional estimator is the one `price` returns in any case. The plain payoff is compared with it only at rho = -0.85, where rho^2 = 0.7225 > 1/2.

The one-step check is not Monte Carlo. With v frozen, the multiplier exp(sqrt(v) (rho dW + sqrt(1 - rho^2) dW_perp) - v dt / 2) has mean 1. A tensor Gauss-Hermite rule, 32 nodes on each increment, returns that mean. The worst absolute error on the grid in the log is 4.441e-16.

## The skew

psi is the slope of implied volatility against log-moneyness at the money. The fit uses nine strikes on [-h, h], h = max(0.15 sqrt(T), 0.01), and a quadratic. The test, written before the run, asks that the slope of log|psi| on log T, for six geometric maturities in [1/52, 1/4] at H = 0.1, lie within 0.05 of -0.4. One seed is too noisy for that band: across 12 seeds the standard deviation is 0.0181, and the mean slope is -0.4096, with 8,000 paths and 128 steps. The bar of 0.05 applies to that mean. The same 12 seeds are the H = 0.1 markers on the figure. Replication 0 alone, the original draw, has slope -0.4163.

H = 0.2 on that same window has mean slope -0.3232, which is 0.023 steeper than -0.300. The standard error of the mean is 0.005. A single seed fitted -0.3364, a gap of 0.036, which is one draw and not the gap. The power is asymptotic. A tighter check would end the window before 1/4. H = 0.5 has mean slope -0.017, which is the diffusive end of the same plot. The standard deviation there is 0.065, because the skew is small and the log-log slope is noisy.
