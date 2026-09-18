# Roadmap

Six notebooks, one question each, in a deliberate order: ground truth first, estimation second,
falsification third, market data fourth, the model fifth, option prices last. Nothing is claimed
about markets until the estimator that would make the claim has been shown to fail on a null model,
and every number carries an interval. **Status: in progress**; boxes tick as sections merge.

## `00_warmup` — why is `H = 1/2` a modelling choice, not a fact?

Optional; nothing later depends on it, only on the functions it introduces.

- [ ] Quadratic variation, p-variation, what "rough" means
- [ ] Wiener integrals, the Itô isometry
- [ ] Implied volatility, and the at-the-money skew Black–Scholes lacks

## `01_fractional_brownian_motion` — what does `H` do to a path?

- [ ] Covariance, self-similarity, and exact simulation by Cholesky
- [ ] What `H` changes: increment correlation, regularity, p-variation
- [ ] Davies–Harte circulant embedding: exact paths, `O(n log n)`
- [ ] The Riemann–Liouville Volterra process, and what it gives up

## `02_estimating_h` — can I measure `H`, and when does the measurement lie?

- [ ] The structure-function estimator, with honest intervals
- [ ] Monofractality tested, not assumed
- [ ] A smooth null model: Heston, spot versus integrated variance
- [ ] Fact or artefact — noise making a diffusive model look rough

## `03_realised_volatility_es` — what does the data say, and how sure am I?

The only notebook needing market data: one-minute bars of the continuous front-month E-mini
S&P 500 future (ES), 2019–2026, from a private Databento archive, with Interactive Brokers as an
independent cross-check. The series is an unadjusted splice, so every within-contract return is a
true return, and the single return spanning each roll is dropped rather than repaired. The sample
begins five years after Gatheral, Jaisson & Rosenbaum's ends, which makes this an out-of-sample
re-test of their finding. Raw bars are licensed and never committed.

- [ ] Data provenance: the archive, the roll, and a cross-check against a second source
- [ ] Realised variance, the signature plot on real one-minute bars, and why five minutes
- [ ] `Ĥ` on ES, and the sensitivity table behind it
- [ ] Bootstrap intervals, rolling estimates, regime splits
- [ ] Roughness is not memory

## `04_rough_bergomi` — how do I simulate a rough model without fooling myself?

- [ ] Forward variance and the compensator keeping `E[V_t] = ξ₀(t)`
- [ ] The hybrid scheme: the `κ`-block and its joint covariance
- [ ] The hybrid scheme: the convolution tail by FFT, benchmarked on paths
- [ ] Paths, non-Markovianity, a martingale audit

## `05_the_smile` — does roughness show up in option prices?

- [ ] Conditional Monte Carlo pricing, implied-volatility inversion
- [ ] The short-maturity skew power law, and why diffusive models lack it
- [ ] The case against, and an audit of what this repository does not show

## What is established, and what is contested

**Established.** The mathematics of fractional Brownian motion and of Riemann–Liouville Volterra
processes, including `Var(W̃_t) = t^{2H}`; the hybrid scheme's convergence rate, where the number of
exactly-integrated steps improves the constant and not the rate (Bennedsen, Lunde & Pakkanen 2017,
Thm 2.5); that rough Bergomi's price is a martingale exactly when the correlation is non-positive,
and that its moments of order `m > 1/(1 − ρ²)` are infinite (Gassiat 2019) — which is why no price
here is reported with a naive Monte Carlo error bar; and Gatheral, Jaisson & Rosenbaum's
`H ≈ 0.142` for the S&P.

**Contested.** Whether the measured roughness of realised volatility is a property of volatility or
an artefact of estimating it from a noisy proxy (Cont & Das 2024; Fukasawa, Takabatake & Westphal
2022; Bolko et al. 2023). Whether the at-the-money skew term structure really follows a power law
(Guyon & El Amrani 2022; Rømer 2022). Whether one rough factor earns its keep against several
Markovian ones (Abi Jaber & El Euch 2019). One comparison is shown rather than cited: Heston's skew
tending to a finite constant at short maturity is demonstrated here by simulation.

Rough Heston, Hawkes foundations, joint SPX/VIX and full surface calibration are out of scope, and
are named as such rather than approximated here.

## References

- Mandelbrot & Van Ness (1968), *Fractional Brownian motions, fractional noises and applications*,
  SIAM Review 10(4), 422–437.
- Davies & Harte (1987), *Tests for Hurst effect*, Biometrika 74(1), 95–101.
- Comte & Renault (1998), *Long memory in continuous-time stochastic volatility models*,
  Mathematical Finance 8(4), 291–323.
- Andersen & Bollerslev (1998), *Answering the skeptics*, International Economic Review 39(4), 885–905.
- Gatheral, Jaisson & Rosenbaum (2018), *Volatility is rough*, Quantitative Finance 18(6), 933–949;
  arXiv:1410.3394.
- Liu, Patton & Sheppard (2015), *Does anything beat 5-minute RV?*, Journal of Econometrics 187(1),
  293–311.
- Bayer, Friz & Gatheral (2016), *Pricing under rough volatility*, Quantitative Finance 16(6),
  887–904; SSRN 2554754.
- Fukasawa (2017), *Short-time at-the-money skew and rough fractional volatility*, Quantitative
  Finance 17(2), 189–198; arXiv:1501.06980.
- Bennedsen, Lunde & Pakkanen (2017), *Hybrid scheme for Brownian semistationary processes*, Finance
  and Stochastics 21, 931–965; arXiv:1507.03004.
- McCrickerd & Pakkanen (2018), *Turbocharging Monte Carlo pricing for the rough Bergomi model*,
  Quantitative Finance 18(11), 1877–1886; arXiv:1708.02563.
- Gassiat (2019), *On the martingale property in the rough Bergomi model*, Electronic
  Communications in Probability 24; arXiv:1811.10935.
- Abi Jaber & El Euch (2019), *Multifactor approximation of rough volatility models*, SIAM Journal
  on Financial Mathematics 10(2), 309–349; arXiv:1801.10359.
- Guyon & El Amrani (2022), *Does the term-structure of equity at-the-money skew really follow a
  power law?*, SSRN 4174538; Risk, 2023.
- Rømer (2022), *Empirical analysis of rough and classical stochastic volatility models to the SPX
  and VIX markets*, Quantitative Finance 22(10), 1805–1838.
- Fukasawa, Takabatake & Westphal (2022), *Consistent estimation for fractional stochastic
  volatility*, Mathematical Finance 32(4), 1086–1132; arXiv:1905.04852.
- Bolko, Christensen, Pakkanen & Veliyev (2023), *A GMM approach to estimate the roughness of
  stochastic volatility*, Journal of Econometrics 235(2), 745–778; arXiv:2010.04610.
- Cont & Das (2024), *Rough volatility: fact or artefact?*, Sankhyā B 86(1), 191–223;
  arXiv:2203.13820.
