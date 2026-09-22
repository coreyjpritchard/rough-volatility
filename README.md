# Volatility is rough

[![CI](https://github.com/coreyjpritchard/rough-volatility/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/coreyjpritchard/rough-volatility/actions/workflows/ci.yml)

Classical stochastic volatility models such as Heston drive the variance with a diffusion, so the
volatility path has the roughness of Brownian motion, whose Hurst exponent is `H = 1/2`. Gatheral,
Jaisson and Rosenbaum (2018) measured realised volatility across indices and time periods and found
that increments of log-volatility scale like those of a fractional Brownian motion with `H` of
order 0.1. In a model with this property the variance process is not Markovian, and when spot and
volatility are correlated the short-dated at-the-money skew follows the power law `T^(H - 1/2)`. In
models whose volatility is a diffusion the skew instead tends to a finite constant as `T` goes to 0.

This repository works through that result in six notebooks, written as lecture notes for traders.
We estimate `H` from data, simulate processes with a given `H`, and price options under a rough
model to see what the smile does. Each section states the question it answers, gives the
definitions and results with their sources, checks them numerically, and ends with exercises and
worked solutions.

## Progress

Notebook 00 is complete, and notebook 01 is next. [ROADMAP.md](ROADMAP.md) gives the plan section
by section, and its boxes are ticked as sections are merged.

| Notebook | Question | Sections done |
| --- | --- | --- |
| [`00_warmup`](00_warmup.ipynb) | Why is `H = 1/2` a modelling choice? | 3 of 3 |
| `01_fractional_brownian_motion` | What does `H` do to a path? | 0 of 4 (next) |
| `02_estimating_h` | Can we measure `H`, and when is the estimate biased? | 0 of 4 |
| `03_realised_volatility_es` | What is `H` for ES futures, and how precisely can we estimate it? | 0 of 5 |
| `04_rough_bergomi` | How do we simulate a rough model, and how do we check the simulation? | 0 of 4 |
| `05_the_smile` | Does roughness appear in option prices? | 0 of 3 |

### Notebook 00

Notebook 00 shows that `H = 1/2` is a property of the Brownian driver that every classical model
inherits, and builds the numerical tools the later notebooks use.

1. **Quadratic variation, and what "rough" means.** We begin with the P&L of a delta-hedged
   option, which is a gamma-weighted integral of realised minus implied variance. We then show that
   the quadratic variation of Brownian motion over `[0, T]` is `T`, and that its `p`-variation tends
   to zero for `p > 2` and to infinity for `p < 2`. The critical index `p = 1/H = 2` is the Hurst
   exponent of Brownian motion in another form.
2. **The Itô isometry.** We prove the isometry for a step integrand and use it to compute the
   variance of the Riemann-Liouville integral `∫ (t - s)^(H - 1/2) dW_s` that drives rough Bergomi.
   At `H = 0.1` a left-point Riemann sum on 512 steps loses 25% of that variance, and the error
   decays only like `n^(-2H)` as the grid is refined. `roughvol.paths.power_kernel_integral`
   removes the error by giving each step its exact variance.
3. **Implied volatility and the at-the-money skew.** We invert Black-Scholes prices for implied
   volatility and measure the at-the-money skew by a central difference. Black-Scholes has a flat
   smile at every maturity, so its skew is zero, while the S&P 500 skew grows roughly like
   `τ^(-0.4)` as the time to expiry `τ` goes to 0 (Bayer, Friz and Gatheral 2016).

### The `roughvol` package

The notebooks import their numerical code from `roughvol`. Its functions are typed and hold no
state, and each function that draws random numbers takes an explicit integer seed, so a notebook, a
test and a script produce the same numbers.

| Module | Contents |
| --- | --- |
| `paths` | Brownian paths, Wiener integrals of a deterministic integrand, and the power-kernel (Riemann-Liouville) integral with exact step variances |
| `variation` | Realised quadratic variation, `p`-variation, and sums of `X dX` evaluated at an interpolated point in each step |
| `bs` | Black-Scholes prices, vega and implied volatility |
| `smile` | The at-the-money skew of a smile |
| `plotting` | The figure style shared by the notebooks |
| `ibkr` | The Interactive Brokers connection, used only by notebook 03 |

The tests check each function against a known result, such as `E[QV] = T`, the Itô isometry and
put-call parity, and every statistical test states its tolerance in standard errors. On every push
and pull request, CI runs `ruff`, the fast tests, and notebook 00 from a clean kernel.

## Layout

```
00_warmup.ipynb   notebook 00
ROADMAP.md        the plan, section by section
roughvol/         the package the notebooks import
tests/            tests of each function against known results
```

Each later notebook is added at the top level when its first section is merged.

## Setup

```bash
conda create -n projects python=3.11 -y
conda activate projects
pip install -e ".[dev]"
```

Tests:

```bash
pytest                                                # everything
pytest -m "not slow"                                  # skip long Monte Carlo runs
pytest tests/test_wiener.py::test_isometry_variance   # one test
pytest --nbmake 00_warmup.ipynb                       # run the notebook
```

Every simulation is seeded, and each notebook runs from top to bottom in a clean kernel.

## Data

Only notebook 03 uses market data. It will use one-minute bars of the continuous front-month E-mini
S&P 500 future (ES), 2019 to 2026, from a private Databento archive. The series is an unadjusted
splice of contracts, so every return within a contract is a true return, and the one return that
spans each roll is dropped. We do not difference-adjust the series, because shifting the price
level rescales every log return and realised variance inherits the square of that error.

Interactive Brokers, through `ib_async`, provides an independent cross-check on overlapping
sessions, with TWS or IB Gateway running locally and connection settings in `.env` (see
`.env.example`). All IB access is in `roughvol/ibkr.py`, and a test checks that importing the
numerical modules does not load it.

Raw bars are licensed and are never committed, so the repository will contain the loaders and not
the data. Every other notebook, and CI, runs on simulated paths.

## References

- Gatheral, Jaisson & Rosenbaum (2018), *Volatility is rough*, Quantitative Finance 18(6), 933-949.
- Bayer, Friz & Gatheral (2016), *Pricing under rough volatility*, Quantitative Finance 16(6), 887-904.
- Bennedsen, Lunde & Pakkanen (2017), *Hybrid scheme for Brownian semistationary processes*,
  Finance and Stochastics 21, 931-965.

The full reading list, and what in this literature is established and what is contested, is in
[ROADMAP.md](ROADMAP.md).
