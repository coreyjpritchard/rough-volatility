# rough-volatility

**S&P 500 realised volatility is rough: Ĥ = 0.148 ± 0.010, against 0.5 for Brownian volatility.**
Gatheral, Jaisson and Rosenbaum (2018) estimator, q = 2, lags 1 to 50 trading days, on
5-minute realised variance (`rv5`) from the Oxford-Man realised library, 3 January 2000 to
25 February 2022 (5,552 days). The ± is the Monte Carlo standard error over 1,000
fractional Brownian paths of the same length. The SPX implied volatility skew gives an
independent H = 0.167 (0.115 to 0.207 across maturity windows) from one chain, 14 October 2026.

![S&P 500 volatility is rough: log m(q, Δ) against log Δ, and the slopes ζ_q against q](docs/hurst_spx.png)

## Results

| Result | Number | Figure and source |
|---|---|---|
| Hurst exponent, Oxford-Man rv5, q = 2, lags 1 to 50 | 0.148 ± 0.010 | `docs/hurst_spx.png`, `docs/make_readme_figure.py` |
| Same series, multi-q slope (q = 0.5 to 3) | 0.149 | `docs/hurst_spx.png` |
| Hurst exponent from the SPX skew, \|ψ\| = 0.305 T^-0.333, all 40 expiries | 0.167 (windows: 0.115 to 0.207) | `docs/figures/rvc_routes.png`, `scripts/figures_rvc.py` |
| Rough Bergomi vol of vol from the rv5 intercept, joint bootstrap of (H, intercept) | ν = 1.328 ± 0.067 | `docs/figures/rvc_nu.csv`, `docs/notes/hurst-routes.md` |
| Hybrid scheme: covariance error of κ = 0 over κ = 1, H = 0.1, n = 16 to 512 | 23 to 24 times | `docs/figures/rva_covariance.png`, `scripts/figures_rva.py` |
| Simulated rough Bergomi ATM skew slope, 1 week to 3 months, at H = 0.05, 0.1, 0.2, 0.5 | -0.46, -0.41, -0.32, -0.02 (theory H - 1/2) | `docs/figures/rvb_skew.png`, `scripts/figures_rvb.py` |

H by source and lag window, q = 2, Monte Carlo standard error in brackets:

| Source | 1 to 10 | 1 to 20 | 1 to 50 | 1 to 100 | 1 to 200 |
|---|---|---|---|---|---|
| rv5, n = 5,552 | 0.1583 (0.0071) | 0.1508 (0.0078) | 0.1475 (0.0105) | 0.1426 (0.0132) | 0.1225 (0.0154) |
| Parkinson, n = 6,731 | 0.0793 (0.0049) | 0.0844 (0.0052) | 0.0900 (0.0068) | 0.0920 (0.0089) | 0.0830 (0.0107) |
| Garman-Klass, n = 6,731 | 0.0882 (0.0052) | 0.0866 (0.0052) | 0.0916 (0.0067) | 0.0925 (0.0087) | 0.0824 (0.0105) |

Every cell is in `docs/figures/rvc_routes.csv`; dated findings are in [LOG.md](LOG.md).

## What is in the repository

```
roughvol/fbm.py         exact fractional Brownian motion (Davies-Harte, Cholesky fallback)
roughvol/volterra.py    Riemann-Liouville process by the hybrid scheme (kappa = 0, 1, 2)
roughvol/rbergomi.py    rough Bergomi: variance, forward curve, conditional Monte Carlo
roughvol/black.py       Black price and implied volatility for the conditional estimator
roughvol/estimate.py    GJR Hurst estimator, skew route, vol of vol from the intercept, Monte Carlo SE
roughvol/realised.py    S&P 500 daily realised variance: Oxford-Man rv5, Parkinson, Garman-Klass
roughvol/heston_vol.py  Heston variance paths (full truncation Euler), a control with H = 1/2
roughvol/rng.py         SeedSequence children for a model run
roughvol/types.py       Paths, PriceResult and ParamSpec, duck-typed by a host
sessions/               marimo apps, one per session (static export in docs/)
scripts/, docs/         figure scripts, figures, CSVs and notes
tests/                  pytest; the mathematical tests take an `impl` fixture
data/                   committed caches: spx_rv5_oxford_man.parquet, spx_ohlc_yahoo.parquet
```

Session 00 asks what H does to a path and what H is for the S&P 500:
[`sessions/00_what_is_rough.py`](sessions/00_what_is_rough.py), static export
[`docs/00_what_is_rough.html`](docs/00_what_is_rough.html). The next sessions are in
[ROADMAP.md](ROADMAP.md).

The entry point group `volsurface.models` registers `rbergomi` as
`roughvol.rbergomi:RoughBergomi`, so a pricing host can discover the model without this
package importing the host.

## Reproduce

With [uv](https://docs.astral.sh/uv/) and `make`:

```bash
make reproduce     # uv sync from uv.lock, tests, then every figure and CSV from its seed
```

That runs `uv sync --extra dev`, `pytest -m "not slow"` and the four figure scripts
(`docs/make_readme_figure.py`, `scripts/figures_rva.py`, `figures_rvb.py`, `figures_rvc.py`).
The data are cached in `data/` and the SPX skew term structure in
`docs/figures/rvc_spx_term.csv`, so nothing is downloaded. The interactive app:

```bash
uv run marimo run sessions/00_what_is_rough.py
```

## Limits

- Ĥ depends on the lag window: 0.158 at lags 1 to 10, 0.122 at lags 1 to 200.
- The daily range estimators (Parkinson, Garman-Klass) are biased low by their
  measurement noise, about 0.09 against 0.15 from rv5.
- The skew route uses one day's chain and has no standard error; the spread across
  maturity windows is the honest range.
- ν comes from the realised-variance intercept. It is not a fit of rough Bergomi to option prices.
- The Oxford-Man library was withdrawn in 2022; the copy used is the last public file on
  the Internet Archive.
- Microstructure noise and the intraday averaging bias are not yet separated (session 01).

## Data

- **Oxford-Man realised library v0.3** (Heber, Lunde, Shephard and Sheppard 2009), from
  `https://web.archive.org/web/20220301022212id_/https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip`.
  Only the `.SPX` `rv5` column is kept (100 KB); `roughvol.realised.build_oxford_man_cache`
  rebuilds it.
- **Yahoo Finance `^GSPC`** daily open, high, low and close from 2000, via yfinance
  (`uv sync --extra data`); `roughvol.realised.build_yahoo_cache` refreshes it.
- **CBOE SPX option chain**, 14 October 2026, reduced to the at-the-money skew per expiry.

## References

- Gatheral, Jaisson, Rosenbaum (2018). Volatility is rough. Quantitative Finance 18(6).
- Bayer, Friz, Gatheral (2016). Pricing under rough volatility. Quantitative Finance 16(6).
- Bennedsen, Lunde, Pakkanen (2017). Hybrid scheme for Brownian semistationary processes. Finance and Stochastics 21(4).
- Heber, Lunde, Shephard, Sheppard (2009). Oxford-Man Institute's realized library, version 0.3.
- Davies, Harte (1987). Tests for Hurst effect. Biometrika 74(1).
- Fukasawa, Takabatake, Westphal (2019). Is volatility rough? arXiv preprint.

To cite this repository, see [CITATION.cff](CITATION.cff). MIT licence.

Built with the help of AI coding tools; all results are reproducible from the scripts in this repo.
