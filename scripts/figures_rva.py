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

    by_kappa = {k: np.array([row["rms"] for row in rows if row["kappa"] == k]) for k in KAPPAS}
    ratio = by_kappa[0] / by_kappa[1]
    print(f"kappa 0 / kappa 1 error ratio: {ratio.min():.1f} to {ratio.max():.1f}")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for kappa in KAPPAS:
        name = "Riemann sum" if kappa == 0 else f"{kappa} lag{'s' if kappa > 1 else ''} exact"
        ax.loglog(NS, by_kappa[kappa], marker="o", label=f"kappa = {kappa} ({name})")
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlabel("number of time steps n on [0, 1]")
    ax.set_ylabel("RMS error of the covariance matrix")
    ax.set_title(
        f"Keeping the first lag exact cuts the covariance error {ratio.min():.0f} to "
        f"{ratio.max():.0f} times\n"
        f"Hybrid scheme for a Riemann-Liouville process, H = {H}, T = {T:g}",
        loc="left",
        fontsize=10,
    )
    ax.legend(frameon=False)
    ax.grid(True, which="major", alpha=0.3)
    fig.text(
        0.01,
        0.01,
        "Deterministic: scheme covariance against the exact covariance on t_i = i/n. The log-log "
        "slope is about -0.7\nfor every kappa, so kappa lowers the constant and not the rate. "
        "Scheme of Bennedsen, Lunde and Pakkanen (2017).",
        fontsize=7.5,
        color="#52514e",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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
