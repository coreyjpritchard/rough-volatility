"""Hurst estimates by realised-variance route, with window sensitivity.

Writes ``docs/figures/rvc_routes.png``, ``docs/figures/rvc_routes.csv`` and
``docs/figures/rvc_nu.csv``.

Realised-variance routes are Oxford-Man rv5, the Parkinson range and
Garman-Klass, each on lag windows 1 to 10, 20, 50, 100 and 200 trading days.
The error bar is the Monte Carlo standard error: 1,000 fractional Brownian
paths of the same length as the series, at the estimated H. There is no
reference line at a true H.

The rough Bergomi vol of vol and its joint standard error, from the same
paths, go to ``docs/figures/rvc_nu.csv``.

The same script can apply the implied-skew route (H = 1/2 - alpha, via
``roughvol.estimate.hurst_from_skew``) to an option chain the user supplies:

    python scripts/figures_rvc.py path/to/chain.parquet

That writes the term structure and windowed H estimates under
``data/cache/`` (gitignored). Those outputs are not committed: this
repository does not publish implied-vol results from option quotes.
Implied volatilities and the at-the-money quadratic are computed by
volsurface, imported only in that optional path.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.estimate import hurst_from_skew, lag_window_sensitivity, regime_estimate
from roughvol.realised import load_spx_rv, log_vol

SEED = 20261014
REPS = 1000
LAGS = ((1, 10), (1, 20), (1, 50), (1, 100), (1, 200))
SOURCES = ("oxford-man", "parkinson", "garman-klass")
LABELS = {
    "oxford-man": "rv5",
    "parkinson": "Parkinson",
    "garman-klass": "Garman-Klass",
}
# Maturity windows in years. None is every expiry. Chosen before the fit.
SKEW_WINDOWS: tuple[tuple[str, tuple[float, float] | None], ...] = (
    ("all expiries", None),
    ("1/252 to 1/12", (1.0 / 252.0, 1.0 / 12.0)),
    ("1/52 to 1/4", (1.0 / 52.0, 0.25)),
    ("1/12 to 1", (1.0 / 12.0, 1.0)),
    ("1/52 to 1", (1.0 / 52.0, 1.0)),
)
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
CACHE_DIR = ROOT / "data" / "cache"


def user_term_structure(parquet: Path):
    """ATM level and skew per expiry. Imports the host only inside this function."""
    import pandas as pd
    from volsurface.chains import market_implied_vols
    from volsurface.smile import market_term_structure

    snapshot = pd.read_parquet(parquet)
    return market_term_structure(market_implied_vols(snapshot))


def write_local_skew(parquet: Path) -> None:
    """Fit the skew route on a user-supplied chain. Writes only under data/cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    term = user_term_structure(parquet)
    term_path = CACHE_DIR / "implied_skew_term.csv"
    term.to_csv(term_path, index=False)
    print(f"wrote {term_path} rows={len(term)} (local, not committed)")
    maturities = term["T"].to_numpy(dtype=np.float64)
    skew = term["skew"].to_numpy(dtype=np.float64)
    rows = []
    for label, window in SKEW_WINDOWS:
        fit = hurst_from_skew(maturities, skew, window)
        rows.append(
            {
                "route": "implied-skew",
                "window": label,
                "H": fit.H,
                "se_mc": fit.se_mc,
                "n_obs": fit.n_obs,
                "q": "",
            }
        )
        print(
            f"skew {label} H={fit.H:.8f} n={fit.n_obs} "
            f"T={fit.lags[0]:.6f} to {fit.lags[-1]:.6f}"
        )
    csv_path = CACHE_DIR / "implied_skew_routes.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["route", "window", "H", "se_mc", "n_obs", "q"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} (local, not committed)")


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: python scripts/figures_rvc.py [path/to/chain.parquet]")
    parquet = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    series = {name: log_vol(load_spx_rv(name)) for name in SOURCES}
    for index, name in enumerate(SOURCES):
        x = series[name]
        window_fits = lag_window_sensitivity(x, LAGS, seed=SEED + 100 * index, reps=REPS)
        for fit in window_fits:
            rows.append(
                {
                    "route": name,
                    "window": f"{int(fit.lags[0])} to {int(fit.lags[-1])}",
                    "H": fit.H,
                    "se_mc": fit.se_mc,
                    "n_obs": fit.n_obs,
                    "q": fit.q,
                }
            )
            print(
                f"{name} lags {int(fit.lags[0])}-{int(fit.lags[-1])} "
                f"H={fit.H:.6f} se_mc={fit.se_mc:.6f} n={fit.n_obs}"
            )
    nu_fit = regime_estimate(series["oxford-man"], range(1, 51), seed=SEED + 7, reps=REPS)
    print(
        f"oxford-man nu_per_day={nu_fit.nu_per_day:.6f} "
        f"se_mc_nu_per_day={nu_fit.se_mc_nu_per_day:.6f} "
        f"rbergomi_nu={nu_fit.to_rbergomi_nu():.6f} "
        f"H={nu_fit.H:.6f} se_mc_H={nu_fit.se_mc_H:.6f} n={nu_fit.n_obs}"
    )
    print(
        f"oxford-man joint: se_mc_nu={nu_fit.se_mc_nu:.6f} "
        f"corr(H, nu_per_day)={nu_fit.corr_mc_H_nu_per_day:.6f} reps={REPS}"
    )
    nu_path = OUT_DIR / "rvc_nu.csv"
    with nu_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "route",
                "lags",
                "H",
                "se_mc_H",
                "nu_per_day",
                "se_mc_nu_per_day",
                "corr_mc_H_nu_per_day",
                "rbergomi_nu",
                "se_mc_nu",
                "reps",
                "seed",
                "n_obs",
            ]
        )
        writer.writerow(
            [
                "oxford-man",
                "1 to 50",
                nu_fit.H,
                nu_fit.se_mc_H,
                nu_fit.nu_per_day,
                nu_fit.se_mc_nu_per_day,
                nu_fit.corr_mc_H_nu_per_day,
                nu_fit.to_rbergomi_nu(),
                nu_fit.se_mc_nu,
                REPS,
                SEED + 7,
                nu_fit.n_obs,
            ]
        )

    csv_path = OUT_DIR / "rvc_routes.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["route", "window", "H", "se_mc", "n_obs", "q"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for offset, name in zip((0.96, 1.0, 1.04), SOURCES, strict=True):
        subset = [row for row in rows if row["route"] == name]
        high = np.array([int(row["window"].split(" to ")[1]) for row in subset]) * offset
        ax.errorbar(
            high,
            [row["H"] for row in subset],
            yerr=[row["se_mc"] for row in subset],
            marker="o",
            ls="",
            capsize=3,
            label=LABELS[name],
        )
    highs = [high for _low, high in LAGS]
    ax.set_xscale("log")
    ax.set_xticks(highs, [str(h) for h in highs])
    ax.minorticks_off()
    ax.set_ylim(0.0, 0.25)
    ax.set_xlabel("longest lag in the fit, trading days (lags start at 1)")
    ax.set_ylabel("H estimate (q = 2)")
    ax.set_title("Realised variance, bars = Monte Carlo SE", loc="left", fontsize=10)
    ax.legend(frameon=False, title="series", title_fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Every realised-variance route puts H far below the Brownian 0.5: "
        "about 0.15 from rv5, 0.08 to 0.09 from range estimators",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.01,
        "S&P 500 daily series; rv5 is Oxford-Man 5-minute realised variance (2000 to 2022), "
        "Parkinson and Garman-Klass are Yahoo Finance ^GSPC daily ranges (2000 to 2026).\n"
        f"Bars are the spread over {REPS:,} fractional Brownian paths. "
        "The implied-skew route is available on a chain you supply "
        "(`python scripts/figures_rvc.py path/to/chain.parquet`); "
        "those results are not published here.",
        fontsize=7.5,
        color="#52514e",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    png_path = OUT_DIR / "rvc_routes.png"
    fig.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")

    if parquet is not None:
        write_local_skew(parquet)


if __name__ == "__main__":
    main()
