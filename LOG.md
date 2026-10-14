# Log

Dated entries, newest first. What was learned, what surprised, open questions.

**Current state:** session 00 built, not yet worked through. The hybrid scheme is in
`roughvol/volterra.py` (RV-a, not yet merged). **Next:** session 01, when does the
estimator lie.

## 2026-10-14: RV-a, hybrid scheme for the Riemann-Liouville kernel

**Found.** Y_t = int_0^t (t - s)^{H - 1/2} dW_s has Var(Y_t) = t^{2H} / (2H). The
hybrid scheme of Bennedsen, Lunde and Pakkanen (2017), kappa in {0, 1, 2}, default 1,
matches that variance by the z-rule. Grid: n = 64 steps on [0, 1], m = 2,000 paths,
R = 4 replications, H in {0.05, 0.1, 0.3, 0.5}. The same grid matches an exact
Cholesky draw on the means and the second moments, for H in {0.05, 0.1, 0.3}. At
H = 1/2 the scheme is Brownian motion pathwise: the increments of Y are the Wiener
increments, to 1e-12. The kappa-block covariance matches the integral closed form to
1e-12, including the hypergeometric cross entry at kappa = 2 against a quadrature.

The covariance figure (`docs/figures/rva_covariance.png`, H = 0.1, T = 1, n from 16
to 512) is deterministic. Root-mean-square covariance error at n = 64:

| kappa | RMS | rel. error of Var(Y_1) | slope of log RMS | slope of log abs rel. var |
|---|---|---|---|---|
| 0 | 1.214e-1 | -0.1940 | -0.700 | -0.200 |
| 1 | 5.234e-3 | -5.895e-4 | -0.681 | -0.198 |
| 2 | 2.854e-3 | -1.800e-4 | -0.660 | -0.194 |

kappa = 1 is a factor of 23 below kappa = 0 in RMS at n = 64, and kappa = 2 is a
factor of 1.83 below kappa = 1. The RMS slopes sit together. The terminal variance
error has slope -2H = -0.2, which is the rate in their Theorem 2.5. The covariance
RMS is a different functional, so its slope is not that rate. Raising kappa changes
the level.

**Surprised.** The archive repository was not readable from this build, so
`power_kernel_weights` and `wiener_integral` follow the paper's equations (2.15),
(3.5) and (3.7) under those names, rather than a line copy of `roughvol/paths.py`.

**Open.** The scheme's covariance is not the covariance of rough Bergomi's spot.
That is RV-b.

## 2026-10-08: session 00, what is H for the S&P 500

**Found.** On Oxford-Man 5-minute realised variance (`.SPX` `rv5`, 2000-01-03 to
2022-02-25, 5,552 days), with log σ = ½ log RV and lags 1 to 50 trading days:

| Source | Days | Ĥ (q = 2) | Multi-q Ĥ (q = 0.5 to 3) | Regression SE | Monte Carlo SE |
|---|---|---|---|---|---|
| Oxford-Man rv5, 2000-01-03 to 2022-02-25 | 5,552 | 0.148 | 0.149 | 0.001 | 0.009 |
| Parkinson range, Yahoo ^GSPC, 2000-01-03 to 2026-10-07 | 6,731 | 0.090 | 0.092 | 0.001 | 0.007 |
| Garman-Klass, Yahoo ^GSPC, 2000-01-03 to 2026-10-07 | 6,731 | 0.092 | 0.094 | 0.001 | 0.007 |

The Monte Carlo SE is the spread of Ĥ over 40 fBm paths of the same length and H. The
Oxford-Man figure sits at the top of the 0.1 to 0.14 range GJR report. ζ_q / q runs from
0.146 (q = 0.5) to 0.151 (q = 3): close to monofractal, as GJR found. Restricting the
range estimators to the Oxford-Man window (to 2022-02-25) changes them by 0.001.

**Surprised 1: the daily estimator biases Ĥ down, not up.** The brief for this session
expected daily estimators to push Ĥ up. On this data they pull it from 0.149 to 0.092.
Mechanism: a range estimate is σ² times a noisy factor, so log σ̂ = log σ + ε with ε
roughly independent across days, which adds 2 Var(ε) to m(2, Δ) at every lag and
flattens the log-log line. Simulated check: fBm with H = 0.15 plus iid noise in log σ of
standard deviation 0.1, 0.2, 0.4 gives Ĥ = 0.143, 0.130, 0.096. The upward bias GJR
discuss is a different effect: daily realised variance averages spot variance over the
day and smooths the path. Both biases exist; here noise dominates. Reframe logged:
"daily estimators bias H up" became "noise biases H down, intraday averaging biases it
up; which wins depends on the estimator". This is the first question for session 01.

**Surprised 2: the regression SE is far too small.** OLS on the log-log points treats
them as independent; they share one series. Over 200 fBm paths of 5,552 steps (q = 2,
lags 1 to 50) the spread of Ĥ was 6 times the mean regression SE at H = 0.1, 9 times at
H = 0.3, 12 times at 0.5, 20 times at 0.7. The app reports both and recommends the Monte
Carlo SE. `tests/test_estimate.py::test_regression_se_understates_sampling_error` pins
this down.

**Controls.** On fBm (4,096 steps, 40 paths, lags 1 to 50) mean Ĥ was 0.050, 0.149,
0.498, 0.925 for true H = 0.05, 0.15, 0.5, 0.95; bias appears only near H = 1. On Heston
variance (κ = 2, θ = 0.04, ξ = 0.3, 40 paths of 4,096 days) mean Ĥ was 0.469 at lags 1
to 10 days, 0.438 at 1 to 50 and 0.360 at 1 to 200. Mean reversion pulls Ĥ down at
long lags, never to 0.15. On SPX the same widening (lags 1 to 200) moves Ĥ from 0.148 to
0.122.

**Implementation notes.** Davies-Harte circulant embedding of fGn was nonnegative
definite for every H and n tried (H up to 0.99, n from 2 to 4,096), so the Cholesky
fallback is a guard only. Yahoo's ^GSPC open equals the previous close on 95 to 98% of
days in 2000 to 2005, 43% in 2006, 16% in 2007, about 1% in 2008 to 2010, 9 to 26% in
2011 to 2013 and almost never from 2014. That contaminates Garman-Klass; Parkinson
avoids the open. App slider responses measured at 0.2 to 0.5 seconds at worst-case lag ranges.

**Open questions.**
1. How much of the gap between 0.149 and 0.092 is noise, and how much of 0.149 itself is
   the intraday averaging bias? A simulation with known spot H, a 5-minute observation
   grid and the two estimators would split them (session 01).
2. Is 0.149 stable across subperiods (2000 to 2007, 2008 to 2012, 2013 to 2022) and
   across the other 30 indices in the Oxford-Man file? The file is cached only for SPX;
   the zip is on the Internet Archive.
3. Fukasawa, Takabatake and Westphal (2019) estimate H with an explicit model of the
   measurement error. What does their estimator give on this rv5 series?
