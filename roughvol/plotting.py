import matplotlib as mpl

ACCENT = "#2E86C1"
ACCENT_ALT = "#C0392B"
GRID_ALPHA = 0.3
FIGSIZE = (10.0, 6.0)
FIGSIZE_WIDE = (16.0, 6.0)
FIGSIZE_DIAG = (8.0, 6.0)


def style() -> None:
    """Apply this repository's matplotlib defaults, in place and idempotently.

    Sets figure.figsize (10, 6); figure.dpi and savefig.dpi 100, a cap, since
    executed outputs are committed and a notebook must stay under ~3 MB;
    axes.grid on with grid.alpha 0.3; axes.titlesize 12; axes.labelsize 10;
    lines.linewidth 1.2; ACCENT first in the colour cycle. Line alpha is left
    alone — pass alpha=0.8 at the call site.

    Mutates matplotlib.rcParams and nothing else; call once per notebook. Wide
    and diagnostic figures pass figsize explicitly.
    """
    colors = [ACCENT, ACCENT_ALT, *mpl.rcParamsDefault["axes.prop_cycle"].by_key()["color"]]
    mpl.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "lines.linewidth": 1.2,
            "axes.prop_cycle": mpl.cycler(color=colors),
        }
    )
