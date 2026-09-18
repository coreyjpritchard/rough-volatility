# Volatility is rough

Realised volatility does not behave like the diffusive process classical stochastic volatility
models assume. Log-volatility increments scale like those of a fractional Brownian motion with a
Hurst exponent well below `1/2` — around `H ~ 0.1` across indices and across time periods. That one
fact changes the shape of the model: the variance process is no longer Markovian, the smile
steepens as maturity shortens, and the short-dated ATM skew picks up the power law `T^(H - 1/2)`
that diffusive models cannot produce at all.

This repository works through that from the ground up: estimate `H` from data, simulate the
processes that have it, then price under them and look at what the smile does.

**Status: in progress.** Packaging and the test harness are in place; the notebook is being
written section by section.

## Contents

Six notebooks, one question each, starting from intuition, simulating, plotting, and bringing in
the mathematics once the picture is on the screen. The section-by-section plan is in
[ROADMAP.md](ROADMAP.md).

1. `00_warmup.ipynb` — why is `H = 1/2` a modelling choice, not a fact?
2. `01_fractional_brownian_motion.ipynb` — what does `H` do to a path?
3. `02_estimating_h.ipynb` — can I measure `H`, and when does the measurement lie?
4. `03_realised_volatility_es.ipynb` — what does the data say, and how sure am I?
5. `04_rough_bergomi.ipynb` — how do I simulate a rough model without fooling myself?
6. `05_the_smile.ipynb` — does roughness show up in option prices?

## References

- Gatheral, Jaisson & Rosenbaum (2018), *Volatility is rough*, Quantitative Finance 18(6), 933-949.
- Bayer, Friz & Gatheral (2016), *Pricing under rough volatility*, Quantitative Finance 16(6), 887-904.
- Bennedsen, Lunde & Pakkanen (2017), *Hybrid scheme for Brownian semistationary processes*,
  Finance and Stochastics 21, 931-965.

## Layout

```
00_warmup.ipynb
01_fractional_brownian_motion.ipynb
02_estimating_h.ipynb
03_realised_volatility_es.ipynb
04_rough_bergomi.ipynb
05_the_smile.ipynb
roughvol/     estimators, simulation schemes, plotting, IB data access
tests/        property tests: an estimator run on simulated fBm must recover the known H
data/         parquet cache of IB history (untracked)
```

The notebook imports from `roughvol`. The functions there are pure and typed and hold no state, so
the same code runs in a notebook, a test or a script.

## Setup

```bash
conda create -n projects python=3.11 -y
conda activate projects
pip install -e ".[dev]"
```

Tests:

```bash
pytest                                              # everything
pytest -m "not slow"                                # skip long Monte Carlo runs
pytest tests/test_hurst.py::test_recovers_known_H   # one test
```

Every simulation is seeded, and the notebook runs top to bottom from a clean kernel.

## Data

Only one notebook needs market data. It runs on one-minute bars of the continuous front-month
E-mini S&P 500 future (ES), 2019–2026, from a private Databento archive. The series is an unadjusted
splice of contracts: every within-contract return is a true return, and the single return spanning
each roll is dropped rather than repaired. A difference-adjusted series would not do — shifting the
price level rescales every log return, and realised variance inherits the square of that error.

Interactive Brokers, via `ib_async`, is the independent cross-check on overlapping sessions, with
TWS or IB Gateway running locally; connection settings live in `.env` (see `.env.example`). All IB
access is confined to one module. Loaders read a parquet cache under `data/` and touch a network
only when asked to refresh. The notebook records which contract, bar size and session were used,
and why.

Raw bars are licensed and never committed: this repository ships the loaders, not the data. Every
other notebook runs on simulated paths, and so does CI.
