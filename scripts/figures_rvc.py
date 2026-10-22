"""Hurst estimates by route, with window sensitivity.

Writes ``docs/figures/rvc_routes.png``, ``docs/figures/rvc_routes.csv`` and
``docs/figures/rvc_spx_term.csv``.

Realised-variance routes are Oxford-Man rv5, the Parkinson range and
Garman-Klass, each on lag windows 1 to 10, 20, 50, 100 and 200 trading days.
The error bar is the Monte Carlo standard error: 1,000 fractional Brownian
paths of the same length as the series, at the estimated H. There is no
reference line at a true H.

The rough Bergomi vol of vol and its joint standard error, from the same
paths, go to ``docs/figures/rvc_nu.csv``.

The skew route refits the SPX chain of 14 October 2026. The parquet is the
volsurface snapshot ``data/snapshots/SPX_2026-10-14.parquet``. Pass its path
as the first argument to rebuild the term structure. With no argument the
committed ``docs/figures/rvc_spx_term.csv`` is refitted, so the script runs
without volsurface. Implied volatilities and the at-the-money quadratic
are the host's: parity forward, out-of-the-money price, half-width
``max(0.15 sqrt(T), 0.01)``. This package does not import that host.
``H = 1/2 - alpha`` is ``roughvol.estimate.hurst_from_skew``.
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
    ("all 40 expiries", None),
    ("1/252 to 1/12", (1.0 / 252.0, 1.0 / 12.0)),
    ("1/52 to 1/4", (1.0 / 52.0, 0.25)),
    ("1/12 to 1", (1.0 / 12.0, 1.0)),
    ("1/52 to 1", (1.0 / 52.0, 1.0)),
)
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def spx_term_structure(parquet: Path):
    """ATM level and skew per expiry. Imports the host only inside this function."""
    import pandas as pd
    from volsurface.chains import market_implied_vols
    from volsurface.smile import market_term_structure

    snapshot = pd.read_parquet(parquet)
    return market_term_structure(market_implied_vols(snapshot))


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: python scripts/figures_rvc.py [path/to/SPX_2026-10-14.parquet]")
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
            ["route", "lags", "H", "se_mc_H", "nu_per_day", "se_mc_nu_per_day",
             "corr_mc_H_nu_per_day", "rbergomi_nu", "se_mc_nu", "reps", "seed", "n_obs"]
        )
        writer.writerow(
            ["oxford-man", "1 to 50", nu_fit.H, nu_fit.se_mc_H, nu_fit.nu_per_day,
             nu_fit.se_mc_nu_per_day, nu_fit.corr_mc_H_nu_per_day, nu_fit.to_rbergomi_nu(),
             nu_fit.se_mc_nu, REPS, SEED + 7, nu_fit.n_obs]
        )

    term_path = OUT_DIR / "rvc_spx_term.csv"
    if parquet is not None:
        term = spx_term_structure(parquet)
        term.to_csv(term_path, index=False)
        print(f"wrote {term_path} rows={len(term)}")
    else:
        import pandas as pd

        term = pd.read_csv(term_path)
        print(f"read {term_path} rows={len(term)}")
    maturities = term["T"].to_numpy(dtype=np.float64)
    skew = term["skew"].to_numpy(dtype=np.float64)
    skew_rows = []
    for label, window in SKEW_WINDOWS:
        fit = hurst_from_skew(maturities, skew, window)
        skew_rows.append((label, fit))
        rows.append(
            {
                "route": "spx-skew",
                "window": label,
                "H": fit.H,
                "se_mc": fit.se_mc,
                "n_obs": fit.n_obs,
                "q": "",
            }
        )
        print(f"skew {label} H={fit.H:.8f} n={fit.n_obs} T={fit.lags[0]:.6f} to {fit.lags[-1]:.6f}")

    csv_path = OUT_DIR / "rvc_routes.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["route", "window", "H", "se_mc", "n_obs", "q"])
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
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

    ax = axes[1]
    labels = [f"{label}\n({fit.n_obs} expiries)" for label, fit in skew_rows]
    values = [fit.H for _label, fit in skew_rows]
    ax.plot(range(len(values)), values, "s", ms=7, color="#52514e", ls="")
    for i, value in enumerate(values):
        ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.set_ylim(0.0, 0.25)
    ax.set_xlabel("maturity window, years")
    ax.set_ylabel("H = 1/2 - alpha, from |skew| = A T^(-alpha)")
    ax.set_title("SPX implied skew, one chain, no standard error", loc="left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "Every route puts H far below the Brownian 0.5: about 0.15 from rv5, "
        "0.08 to 0.09 from range estimators, 0.12 to 0.21 from the SPX skew",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.01,
        "Left: S&P 500 daily series; rv5 is Oxford-Man 5-minute realised variance (2000 to 2022), "
        "Parkinson and Garman-Klass are Yahoo Finance ^GSPC daily ranges (2000 to 2026).\n"
        f"Bars are the spread over {REPS:,} fractional Brownian paths. Right: CBOE SPX chain, "
        "14 October 2026, at-the-money skew per expiry.",
        fontsize=7.5,
        color="#52514e",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    png_path = OUT_DIR / "rvc_routes.png"
    fig.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
