# Two routes to H

The figure `docs/figures/rvc_routes.png` is the Hurst exponent by route. The left panel is the Gatheral, Jaisson and Rosenbaum estimator on three realised-variance series, against the longest lag, with the Monte Carlo standard error as the bar. The right panel is H = 1/2 - alpha from the S&P 500 implied-volatility skew, on five maturity windows. There is no line at a true H.

## Realised variance

For x = log sigma on a regular grid,

    m(q, Delta) = mean |x_{t+Delta} - x_t|^q.

If the increments scale like Delta^H, then m(q, Delta) scales like Delta^{q H}. The slope of log m on log Delta, divided by q, is the estimate. q = 2 throughout the figure. The ordinary least squares standard error treats the lags as independent. They share one series, so that standard error is too small. The number in the figure is the standard deviation of the same estimator on 40 fractional Brownian paths of the same length, at the estimated H.

On Oxford-Man rv5, 5,552 days, lags 1 to 50, the estimate is 0.1475 and the Monte Carlo standard error is 0.0134. Widening the window to lags 1 to 200 moves the estimate to 0.1225. The Parkinson and Garman-Klass range series, 6,731 days, stay near 0.08 to 0.09 on every window in the figure. The CSV next to the figure holds every cell.

## Vol of vol from the intercept

On one regime, log m(2, Delta) = a + 2 H log Delta, with Delta in grid steps. If log sigma = (nu_per_day / 2) B and B is standard fractional Brownian motion in those units, Var(B_t) = t^{2H}, then

    m(2, Delta) = (nu_per_day / 2)^2 Delta^{2H},    nu_per_day = 2 exp(a / 2).

The factor 1/2 is log sigma = log v / 2, so nu_per_day is the vol of vol of an increment of variance, on a grid whose step is one trading day. The regression sees E[(log v_{t+Delta} - log v_t)^2], not Var(log v_t). On rv5 at lags 1 to 50 this intercept gives nu_per_day = 0.699732 with Monte Carlo standard error 0.0133, at H = 0.147535. In rough Bergomi the same increment, once time is in years, has standard deviation nu sqrt(2 H c_H) dt^H. The constant is c_H = Gamma(H + 1/2) Gamma(2 - 2H) / (2 H Gamma(3/2 - H)), the reciprocal square of the Mandelbrot-Van Ness constant. One year is 252 trading days, so nu = nu_per_day * 252^H / sqrt(2 H c_H) = 1.328276. At this H the factor sqrt(2 H c_H) is 1.191, and the level scaling nu_per_day * 252^H = 1.582057 is a different quantity. At H = 1/2 the factor is 1 and the two scalings agree. It is not a fit of rough Bergomi to option prices.

A two-regime series is two long blocks. Each block is estimated on its own run, so a lag that would cross the join is not used.

## The skew

|psi| = A T^{-alpha}, and H = 1/2 - alpha. The fit is a regression of log|psi| on log T. It has no path to resample, so the Monte Carlo standard error on this route is NaN. The uncertainty to report with the all-expiry number is the range across windows, 0.115277 to 0.207468, alongside 0.167175.

The chain is the CBOE SPX snapshot of 14 October 2026, 40 expiries, as tidied in the volsurface snapshot file. The forward and the discount factor at each expiry are the put-call parity regression. Out-of-the-money prices are inverted, and the at-the-money skew is the linear coefficient of a quadratic in log-moneyness on |k| <= max(0.15 sqrt(T), 0.01). This package does not import the host. The script that builds the term structure does, and only there.

On all 40 expiries, |psi| = 0.305112 T^{-0.332825}, so H = 0.167175. The same points on other windows:

| Window | Expiries | H |
|---|---|---|
| all | 40 | 0.167175 |
| 1/252 to 1/12 | 20 | 0.206983 |
| 1/52 to 1/4 | 28 | 0.207468 |
| 1/12 to 1 | 18 | 0.115277 |
| 1/52 to 1 | 35 | 0.161886 |

The window 1/52 to 1/4 sits 0.040 above the all-expiry fit. The window 1/12 to 1 sits 0.052 below it. The two front expiries have discount factors above 1. Dropping every expiry shorter than a week, which drops those two, moves H by 0.005, to 0.161886. A later fit can cap the discount factor at 1 and should be reported beside this table.
