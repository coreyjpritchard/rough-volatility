"""ATM skew term structure of rough Bergomi.

Writes ``docs/figures/rvb_skew.png`` and ``docs/figures/rvb_skew.csv``.
H in {0.05, 0.1, 0.2, 0.5}, nu = 1.5, rho = -0.7, xi0 = 0.04.
Maturities are six geometric points from 1/52 to 1/4, then 1/2 and 1.
Each point is a conditional Monte Carlo price, 8,000 paths, 128 steps,
nine log-moneynesses on [-h, h] with h = max(0.15 sqrt(T), 0.01), and the
slope of a quadratic in log-moneyness at the money.

Twelve seeds. The plotted skew is the mean of the absolute skew. The legend
is the mean of the per-seed log-log slopes on [1/52, 1/4]. At H = 0.1 that
mean is what ``tests/test_rbergomi.py::test_skew_exponent`` checks.
Replication 0 on that window uses seed 20261014 + 5000 + 100 * rep + index.
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
REPS = 12
SHORT = np.geomspace(1.0 / 52.0, 0.25, 6)
MATURITIES = np.concatenate([SHORT, np.array([0.5, 1.0])])
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def _seed(H: float, rep: int, index: int) -> int:
    if abs(H - 0.1) < 1e-12 and index < SHORT.size:
        return SEED + 5_000 + 100 * rep + index
    return SEED + 20_000 + int(round(1000 * H)) + 100 * rep + index


def main() -> None:
    rows = []
    for H in H_VALUES:
        model = RoughBergomi(
            H=H, nu=NU, rho=RHO, xi0=XI0, n_steps=N_STEPS, n_paths=N_PATHS, kappa=1
        )
        slopes = []
        by_maturity: list[list[float]] = [[] for _ in MATURITIES]
        for rep in range(REPS):
            skew = np.array(
                [
                    model.atm_skew(float(maturity), n_paths=N_PATHS, seed=_seed(H, rep, index))
                    for index, maturity in enumerate(MATURITIES)
                ]
            )
            for index, value in enumerate(skew):
                by_maturity[index].append(float(value))
            short = skew[: SHORT.size]
            slopes.append(float(np.polyfit(np.log(SHORT), np.log(np.abs(short)), 1)[0]))
        slopes_arr = np.asarray(slopes, dtype=np.float64)
        mean_slope = float(slopes_arr.mean())
        sd_slope = float(slopes_arr.std(ddof=1))
        print(
            f"H={H} mean slope on [1/52, 1/4] = {mean_slope:.6f} "
            f"sd={sd_slope:.6f} target {H - 0.5:.6f}"
        )
        for index, maturity in enumerate(MATURITIES):
            sample = np.asarray(by_maturity[index], dtype=np.float64)
            mean_skew = float(sample.mean())
            rows.append(
                {
                    "H": H,
                    "T": float(maturity),
                    "skew": mean_skew,
                    "abs_skew": float(np.mean(np.abs(sample))),
                    "skew_sd": float(sample.std(ddof=1)),
                    "n_paths": N_PATHS,
                    "n_steps": N_STEPS,
                    "n_seeds": REPS,
                    "mean_slope": mean_slope,
                    "slope_sd": sd_slope,
                }
            )
            print(f"H={H} T={maturity:.6f} mean skew={mean_skew:.6f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "rvb_skew.csv"
    fields = [
        "H",
        "T",
        "skew",
        "abs_skew",
        "skew_sd",
        "n_paths",
        "n_steps",
        "n_seeds",
        "mean_slope",
        "slope_sd",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for H in H_VALUES:
        subset = [row for row in rows if row["H"] == H]
        T_all = np.array([row["T"] for row in subset])
        skew = np.array([row["abs_skew"] for row in subset])
        slope = float(subset[0]["mean_slope"])
        line = ax.loglog(T_all, skew, "o", ms=5, label=f"H = {H}: fitted slope {slope:.2f}")[0]
        anchor = T_all[0]
        guide = SHORT
        guide_skew = skew[0] * (guide / anchor) ** (H - 0.5)
        ax.loglog(guide, guide_skew, "--", lw=1, color=line.get_color())
    ticks = [1 / 52, 1 / 12, 0.25, 0.5, 1.0]
    ax.set_xticks(ticks, ["1 week", "1 month", "3 months", "6 months", "1 year"])
    ax.minorticks_off()
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0], ["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.set_xlabel("maturity T")
    ax.set_ylabel("|ATM skew|, d sigma_imp / d log K")
    ax.set_title(
        "Rough Bergomi: the short-dated skew steepens as H falls, slope close to H - 1/2",
        loc="left",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8.5, title="points: simulated; dashed: slope H - 1/2",
              title_fontsize=8.5)
    ax.grid(True, which="major", alpha=0.3)
    fig.text(
        0.01,
        0.01,
        f"Simulated, not market data: nu = {NU}, rho = {RHO}, xi0 = {XI0}, {N_STEPS} steps, "
        f"kappa = 1, {N_PATHS:,} paths x {REPS} seeds.\n"
        "Points are the mean |skew| over seeds; slopes are fitted on 1 week to 3 months.",
        fontsize=7.5,
        color="#52514e",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    png_path = OUT_DIR / "rvb_skew.png"
    fig.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
