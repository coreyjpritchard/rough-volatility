"""Hurst estimates by route, with window sensitivity.

Writes ``docs/figures/rvc_routes.png``, ``docs/figures/rvc_routes.csv`` and
``docs/figures/rvc_spx_term.csv``.

Realised-variance routes are Oxford-Man rv5, the Parkinson range and
Garman-Klass, each on lag windows 1 to 10, 20, 50, 100 and 200 trading days.
The error bar is the Monte Carlo standard error: 40 fractional Brownian paths
of the same length as the series, at the estimated H. There is no reference
line at a true H.

The skew route refits the SPX chain of 14 October 2026. The parquet is the
volsurface snapshot ``data/snapshots/SPX_2026-10-14.parquet``. Pass its path
as the first argument. Implied volatilities and the at-the-money quadratic
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
REPS = 40
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
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/figures_rvc.py path/to/SPX_2026-10-14.parquet")
    parquet = Path(sys.argv[1])
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

    term = spx_term_structure(parquet)
    term_path = OUT_DIR / "rvc_spx_term.csv"
    term.to_csv(term_path, index=False)
    print(f"wrote {term_path} rows={len(term)}")
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

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    ax = axes[0]
    for name in SOURCES:
        subset = [row for row in rows if row["route"] == name]
        high = [int(row["window"].split(" to ")[1]) for row in subset]
        ax.errorbar(
            high,
            [row["H"] for row in subset],
            yerr=[row["se_mc"] for row in subset],
            marker="o",
            capsize=3,
            label=LABELS[name],
        )
    ax.set_xlabel("longest lag (trading days)")
    ax.set_ylabel("H estimate")
    ax.set_title("Realised variance, Monte Carlo SE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = [label for label, _fit in skew_rows]
    values = [fit.H for _label, fit in skew_rows]
    ax.plot(range(len(values)), values, marker="o")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("H = 1/2 - alpha")
    ax.set_title("SPX skew, by maturity window")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = OUT_DIR / "rvc_routes.png"
    fig.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
