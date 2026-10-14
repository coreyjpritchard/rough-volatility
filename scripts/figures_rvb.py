"""ATM skew term structure of rough Bergomi.

Writes ``docs/figures/rvb_skew.png`` and ``docs/figures/rvb_skew.csv``.
H in {0.05, 0.1, 0.2, 0.5}, nu = 1.5, rho = -0.7, xi0 = 0.04.
Maturities are six geometric points from 1/52 to 1/4, then 1/2 and 1.
Each point is a conditional Monte Carlo price, 8,000 paths, 128 steps,
nine log-moneynesses on [-h, h] with h = max(0.15 sqrt(T), 0.01), and the
slope of a quadratic in log-moneyness at the money.

The H = 0.1 points on [1/52, 1/4] use the same seeds as
``tests/test_rbergomi.py::test_skew_exponent`` (seed 20261014 + 5000 + index).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roughvol.rbergomi import RoughBergomi

SEED = 20261014

H_VALUES = (0.05, 0.1, 0.2, 0.5)
NU = 1.5
RHO = -0.7
XI0 = 0.04
N_PATHS = 8_000
N_STEPS = 128
SHORT = np.geomspace(1.0 / 52.0, 0.25, 6)
MATURITIES = np.concatenate([SHORT, np.array([0.5, 1.0])])
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def _seed(H: float, index: int) -> int:
    if abs(H - 0.1) < 1e-12 and index < SHORT.size:
        return SEED + 5_000 + index
    return SEED + 20_000 + int(round(1000 * H)) + index


def main() -> None:
    rows = []
    for H in H_VALUES:
        model = RoughBergomi(
            H=H, nu=NU, rho=RHO, xi0=XI0, n_steps=N_STEPS, n_paths=N_PATHS, kappa=1
        )
        for index, maturity in enumerate(MATURITIES):
            skew = model.atm_skew(float(maturity), n_paths=N_PATHS, seed=_seed(H, index))
            rows.append(
                {
                    "H": H,
                    "T": float(maturity),
                    "skew": skew,
                    "abs_skew": abs(skew),
                    "n_paths": N_PATHS,
                    "n_steps": N_STEPS,
                }
            )
            print(f"H={H} T={maturity:.6f} skew={skew:.6f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "rvb_skew.csv"
    fields = ["H", "T", "skew", "abs_skew", "n_paths", "n_steps"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for H in H_VALUES:
        subset = [row for row in rows if row["H"] == H]
        short = [row for row in subset if row["T"] <= 0.25 + 1e-12]
        log_t = np.log([row["T"] for row in short])
        log_s = np.log([row["abs_skew"] for row in short])
        slope = float(np.polyfit(log_t, log_s, 1)[0])
        ax.loglog(
            [row["T"] for row in subset],
            [row["abs_skew"] for row in subset],
            marker="o",
            label=f"H = {H}, slope {slope:.3f}",
        )
        print(f"H={H} slope on [1/52, 1/4] = {slope:.6f}, target {H - 0.5:.6f}")
    ax.set_xlabel("maturity T")
    ax.set_ylabel("absolute ATM skew")
    ax.set_title("Rough Bergomi skew, nu = 1.5, rho = -0.7")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png_path = OUT_DIR / "rvb_skew.png"
    fig.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
