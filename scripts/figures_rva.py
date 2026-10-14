"""Covariance error of the hybrid scheme against the number of steps.

Writes ``docs/figures/rva_covariance.png`` and ``docs/figures/rva_covariance.csv``.
The error is deterministic: it is the root-mean-square difference between the
scheme's covariance and the exact Riemann-Liouville covariance on the grid
t_i = i/n, i = 1..n, with T = 1 and H = 0.1. kappa = 0 is the Riemann sum.
kappa = 1 and 2 keep that many lags exact.

Bennedsen, Lunde and Pakkanen (2017), Theorem 2.5: the mean-square rate is
n^{-2H} and does not depend on kappa. kappa changes the constant.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.volterra import hybrid_covariance, riemann_liouville_covariance

H = 0.1
T = 1.0
NS = (16, 32, 64, 128, 256, 512)
KAPPAS = (0, 1, 2)
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def rms_error(n: int, kappa: int) -> tuple[float, float]:
    """RMS covariance error on the grid, and the relative variance error at T."""
    times = np.linspace(0.0, T, n + 1)
    exact = riemann_liouville_covariance(H, times)
    scheme = hybrid_covariance(H, n, T, kappa=kappa)
    diff = scheme[1:, 1:] - exact[1:, 1:]
    rms = float(np.sqrt(np.mean(diff**2)))
    true_var = float(exact[-1, -1])
    rel_var = float((scheme[-1, -1] - true_var) / true_var)
    return rms, rel_var


def main() -> None:
    rows = []
    for kappa in KAPPAS:
        for n in NS:
            rms, rel_var = rms_error(n, kappa)
            rows.append({"H": H, "T": T, "n": n, "kappa": kappa, "rms": rms, "rel_var_T": rel_var})
            print(f"kappa={kappa} n={n} rms={rms:.6e} rel_var_T={rel_var:.6e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "rva_covariance.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["H", "T", "n", "kappa", "rms", "rel_var_T"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for kappa in KAPPAS:
        subset = [row for row in rows if row["kappa"] == kappa]
        ax.loglog(
            [row["n"] for row in subset],
            [row["rms"] for row in subset],
            marker="o",
            label=f"kappa = {kappa}",
        )
    ax.set_xlabel("number of steps n")
    ax.set_ylabel("covariance RMS error")
    ax.set_title(f"Hybrid scheme, H = {H}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png_path = OUT_DIR / "rva_covariance.png"
    fig.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")

    print("log-log slope of rms against n")
    for kappa in KAPPAS:
        subset = [row for row in rows if row["kappa"] == kappa]
        log_n = np.log([row["n"] for row in subset])
        log_rms = np.log([row["rms"] for row in subset])
        slope = float(np.polyfit(log_n, log_rms, 1)[0])
        print(f"kappa={kappa} slope={slope:.4f}")


if __name__ == "__main__":
    main()
