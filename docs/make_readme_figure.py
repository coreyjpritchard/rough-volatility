"""Static version of Session 00, panel 2, for the README: docs/hurst_spx.png.

Run from the repository root: python docs/make_readme_figure.py
"""

# %%
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roughvol import estimate, realised  # noqa: E402

OUT = Path(__file__).resolve().parent / "hurst_spx.png"
QS = (0.5, 1.0, 1.5, 2.0, 3.0)
LAGS = np.arange(1, 51)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
SHADES = ["#b5d0f2", "#8ab5ea", "#5c97e0", "#2a78d6", "#1a4f91"]  # one hue, light to dark

# %%
x_om = realised.log_vol(realised.load_spx_rv("oxford-man"))
x_pk = realised.log_vol(realised.load_spx_rv("parkinson"))
fit_om = estimate.hurst_gjr_multi(x_om, QS, LAGS)
fit_pk = estimate.hurst_gjr_multi(x_pk, QS, LAGS)
print(f"Oxford-Man multi-q H = {fit_om.H:.3f}, Parkinson multi-q H = {fit_pk.H:.3f}")

# %%
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.5, 1]})
ld = np.log(LAGS)
for f, c in zip(fit_om.fits, SHADES, strict=True):
    ax1.plot(ld, np.log(f.m), "o", ms=3, color=c)
    ax1.plot(ld, f.intercept + f.slope * ld, "-", lw=1.5, color=c)
    ax1.text(-0.1, np.log(f.m[0]), f"q = {f.q:g}", ha="right", va="center", color=MUTED)
ax1.set_xlim(-0.75, ld[-1] + 0.1)
ax1.set_xlabel("log Δ (trading days)")
ax1.set_ylabel("log m(q, Δ)")
ax1.set_title("S&P 500, Oxford-Man rv5, 2000 to 2022", loc="left", fontsize=10, color=INK)

qq = np.linspace(0, 3.2, 10)
for fit, c, name in [(fit_om, BLUE, "Oxford-Man rv5"), (fit_pk, ORANGE, "Parkinson range")]:
    ax2.plot(qq, fit.H * qq, "-", lw=2, color=c, label=f"{name}: H = {fit.H:.3f}")
    ax2.plot(fit.qs, fit.zetas, "o", ms=6, color=c, mec="white", mew=1.5)
ax2.set_xlim(0, 3.2)
ax2.set_ylim(0, None)
ax2.set_xlabel("q")
ax2.set_ylabel(r"$\zeta_q$ (slope of the left panel)")
ax2.set_title(r"$\zeta_q = qH$", loc="left", fontsize=10, color=INK)
ax2.legend(frameon=False, loc="upper left")
for ax in (ax1, ax2):
    ax.grid(color="#ececea", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
