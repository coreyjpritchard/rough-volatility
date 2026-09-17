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

`01_volatility_is_rough.ipynb` — starts from intuition, simulates, plots, and brings in the
mathematics once the picture is on the screen.

1. Realised variance from intraday bars, and the scaling of log-volatility increments.
2. Estimating the Hurst exponent, with the estimator checked against simulated fBm of known `H`.
3. Simulating fractional Brownian motion, and the Volterra kernel behind it.
4. The rough Bergomi model via the hybrid scheme.
5. Implied volatility under rough volatility, and the short-maturity skew.

## References

- Gatheral, Jaisson & Rosenbaum (2018), *Volatility is rough*, Quantitative Finance 18(6), 933-949.
- Bayer, Friz & Gatheral (2016), *Pricing under rough volatility*, Quantitative Finance 16(6), 887-904.
- Bennedsen, Lunde & Pakkanen (2017), *Hybrid scheme for Brownian semistationary processes*,
  Finance and Stochastics 21, 931-965.

## Layout

```
01_volatility_is_rough.ipynb
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

Intraday history comes from Interactive Brokers via `ib_async`, with TWS or IB Gateway running
locally; connection settings live in `.env` (see `.env.example`). All IB access is confined to one
module, which reads a parquet cache under `data/` and only calls the API when asked to refresh.
The notebook records which contract, bar size and `whatToShow` were used, and why.

Raw IB history is not redistributable and is never committed: this repository ships the loader,
not the data.
