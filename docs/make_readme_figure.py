"""README figure: docs/hurst_spx.png.

Run from the repository root: python docs/make_readme_figure.py

The headline is the q = 2 estimate on lags 1 to 50, with the Monte Carlo
standard error over 1,000 fractional Brownian paths of the same length and H
(seed 20261014, the same draw as the lags 1 to 50 cell of
docs/figures/rvc_routes.csv). The right panel is the slope over every moment
order q at once, which is a separate estimate.
"""

# %%
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from roughvol import estimate, realised  # noqa: E402

OUT = Path(__file__).resolve().parent / "hurst_spx.png"
QS = (0.5, 1.0, 1.5, 2.0, 3.0)
LAGS = np.arange(1, 51)
SEED = 20261014
REPS = 1000
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
SHADES = ["#b5d0f2", "#8ab5ea", "#5c97e0", "#2a78d6", "#1a4f91"]  # one hue, light to dark

# %%
rv_om = realised.load_spx_rv("oxford-man")
rv_pk = realised.load_spx_rv("parkinson")
x_om = realised.log_vol(rv_om)
x_pk = realised.log_vol(rv_pk)
head = estimate.hurst_estimate(x_om, 2.0, LAGS, seed=SEED, reps=REPS)
fit_om = estimate.hurst_gjr_multi(x_om, QS, LAGS)
fit_pk = estimate.hurst_gjr_multi(x_pk, QS, LAGS)
print(f"Oxford-Man q = 2 H = {head.H:.6f}, Monte Carlo SE = {head.se_mc:.6f} ({REPS} paths)")
print(f"Oxford-Man all-q H = {fit_om.H:.6f}, Parkinson all-q H = {fit_pk.H:.6f}")


def _span(series) -> str:
    first, last = series.index[0], series.index[-1]
    return f"{first:%d %b %Y} to {last:%d %b %Y}"


# %%
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
ld = np.log(LAGS)
for f, c in zip(fit_om.fits, SHADES, strict=True):
    ax1.plot(ld, np.log(f.m), "o", ms=3, color=c)
    ax1.plot(ld, f.intercept + f.slope * ld, "-", lw=1.5, color=c)
    ax1.text(-0.1, np.log(f.m[0]), f"q = {f.q:g}", ha="right", va="center", color=MUTED)
ax1.set_xlim(-0.75, ld[-1] + 0.1)
ax1.set_xlabel("log Δ, lag in trading days")
ax1.set_ylabel("log m(q, Δ), mean |Δ log σ|^q")
ax1.set_title("Oxford-Man rv5: straight lines in log Δ, one per q", loc="left", fontsize=10)
ax1.legend(
    handles=[
        Line2D([], [], ls="", marker="o", ms=4, color=BLUE, label="measured moment"),
        Line2D([], [], ls="-", lw=1.5, color=BLUE, label="least-squares line"),
    ],
    frameon=False,
    loc="lower right",
)

qq = np.linspace(0, 3.2, 10)
ax2.plot(qq, 0.5 * qq, "--", lw=1.2, color=MUTED, label="Brownian volatility, H = 0.5")
for fit, c, label in [
    (
        fit_om,
        BLUE,
        f"Oxford-Man rv5, fit over all q: H = {fit_om.H:.3f}\n"
        f"(headline {head.H:.3f} uses q = 2 only)",
    ),
    (fit_pk, ORANGE, f"Parkinson range, fit over all q: H = {fit_pk.H:.3f}\n(biased low)"),
]:
    ax2.plot(qq, fit.H * qq, "-", lw=2, color=c, label=label)
    ax2.plot(fit.qs, fit.zetas, "o", ms=6, color=c, mec="white", mew=1.5)
ax2.set_xlim(0, 3.2)
ax2.set_ylim(0, 1.65)
ax2.set_xlabel("moment order q")
ax2.set_ylabel(r"slope $\zeta_q$ of the left panel")
ax2.set_title(r"Slopes are linear in q: $\zeta_q = qH$", loc="left", fontsize=10)
ax2.legend(frameon=False, loc="upper left", fontsize=8.5)
for ax in (ax1, ax2):
    ax.grid(color="#ececea", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    f"S&P 500 volatility is rough: H = {head.H:.3f} ± {head.se_mc:.3f} (q = 2, lags 1 to 50), "
    "far below the 0.5 of Brownian volatility",
    x=0.01,
    ha="left",
    fontsize=11.5,
    color=INK,
)
fig.text(
    0.01,
    0.01,
    f"Data: Oxford-Man realised library v0.3, S&P 500 5-minute realised variance, {_span(rv_om)} "
    f"({len(x_om):,} days). Parkinson: Yahoo Finance ^GSPC daily range, {_span(rv_pk)}.\n"
    f"Points: measured moments and slopes. Lines: least-squares fits. ± is the Monte Carlo "
    f"standard error over {REPS:,} fractional Brownian paths of the same length.",
    fontsize=7.5,
    color=MUTED,
    va="bottom",
)
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
