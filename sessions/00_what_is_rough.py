import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="00 What is rough?")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from roughvol import estimate, realised
    from roughvol.fbm import fbm
    from roughvol.heston_vol import heston_variance

    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9893"

    def style(fig, height):
        fig.update_layout(
            template="plotly_white",
            height=height,
            margin=dict(l=50, r=20, t=40, b=40),
            showlegend=False,
            font=dict(size=12),
        )
        fig.update_xaxes(gridcolor="#ececea", zeroline=False)
        fig.update_yaxes(gridcolor="#ececea", zeroline=False)
        return fig

    return (
        BLUE,
        GREY,
        ORANGE,
        estimate,
        fbm,
        go,
        heston_variance,
        make_subplots,
        mo,
        np,
        realised,
        style,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 00 What does H do to a path, and what is H for the S&P 500?
    Three panels. Move the sliders first, read the text under each figure second.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Roughness on a slider
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    h1 = mo.ui.slider(0.05, 0.95, 0.01, value=0.10, label="H", show_value=True)
    nu1 = mo.ui.slider(0.1, 3.0, 0.1, value=1.5, label="ν (vol of log vol)", show_value=True)
    return h1, nu1


@app.cell(hide_code=True)
def _(BLUE, GREY, ORANGE, fbm, h1, make_subplots, mo, np, nu1, style):
    _n = 2**12
    _t = np.linspace(0.0, 1.0, _n + 1)
    _B = fbm(h1.value, _n, 1.0, 1, seed=7)[0]
    _dB = np.diff(_B)
    _fig1 = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            "fBm path B<sup>H</sup><sub>t</sub>",
            "its increments",
            "exp(ν B<sup>H</sup><sub>t</sub>): a stand-in volatility path",
        ),
        vertical_spacing=0.14,
    )
    _fig1.add_scatter(x=_t, y=_B, line=dict(color=BLUE, width=1), row=1, col=1)
    _fig1.add_scatter(x=_t[1:], y=_dB, line=dict(color=GREY, width=0.8), row=1, col=2)
    _fig1.add_scatter(
        x=_t, y=np.exp(nu1.value * _B), line=dict(color=ORANGE, width=1), row=2, col=1
    )
    _fig1.update_xaxes(title_text="t", row=2, col=1)
    style(_fig1, 560)
    _rho = 2 ** (2 * h1.value - 1) - 1
    mo.vstack(
        [
            mo.hstack([h1, nu1], justify="start", gap=2),
            _fig1,
            mo.md(
                f"Step Δ = 1/{_n}. Increment standard deviation Δ<sup>H</sup> = "
                f"{(1 / _n) ** h1.value:.2g}. Lag-one correlation of increments "
                f"2<sup>2H−1</sup> − 1 = {_rho:+.2f}."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What you are looking at.** One path of fractional Brownian motion (fBm) on
    $[0, 1]$ with 4,096 steps, its increments, and $\exp(\nu B^H_t)$: what a volatility path
    would look like if log volatility were this fBm. The random numbers are fixed, so moving
    $H$ changes only how rough the path is, not where it goes.

    **What H means.** fBm is the Gaussian process whose increments scale like $\Delta^H$:

    $$\mathbb{E}\big[(B^H_{t+\Delta} - B^H_t)^2\big] = \Delta^{2H}.$$

    Small $H$ means a short step already carries a large move, so the path is jagged at
    every zoom level. For $H > 1/2$ increments are positively correlated and moves persist;
    for $H < 1/2$ they are negatively correlated and moves tend to reverse. The covariance is
    $\operatorname{Cov}(B^H_s, B^H_t) = \tfrac12 (s^{2H} + t^{2H} - |t - s|^{2H})$.

    **H = 1/2 is Brownian motion.** The lag-one correlation $2^{2H-1} - 1$ is zero, the
    increments are independent, and $\Delta^{1/2}$ is the square-root-of-time rule.

    **H = 0.1 is what volatility looks like.** Set $H = 0.1$ and $\nu = 1.5$. The orange line
    has calm stretches broken by sharp bursts, and the bursts have bursts inside them.
    Gatheral, Jaisson and Rosenbaum (2018), *Volatility is rough*, found that the log of S&P
    500 realised volatility behaves like fBm with $H$ near 0.1. Panel 2 measures it.

    The sampler is exact: Davies and Harte (1987) circulant embedding,
    `roughvol.fbm.fbm`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Measuring H on the S&P 500
    """)
    return


@app.cell(hide_code=True)
def _(realised):
    LOGVOL = {k: realised.log_vol(realised.load_spx_rv(k)) for k in realised.SOURCES}
    DATES = {k: realised.load_spx_rv(k).index for k in realised.SOURCES}
    return DATES, LOGVOL


@app.cell(hide_code=True)
def _(mo, realised):
    src2 = mo.ui.dropdown(
        options={v: k for k, v in realised.SOURCES.items()},
        value=realised.SOURCES["oxford-man"],
        label="data",
    )
    q2 = mo.ui.slider(0.5, 3.0, 0.1, value=2.0, label="q", show_value=True)
    lags2 = mo.ui.range_slider(1, 250, 1, value=[1, 50], label="lags (days)", show_value=True)
    return lags2, q2, src2


@app.cell(hide_code=True)
def _(BLUE, DATES, GREY, LOGVOL, ORANGE, estimate, lags2, make_subplots, mo, np, q2, src2, style):
    _x = LOGVOL[src2.value]
    _lo, _hi = (int(v) for v in lags2.value)
    _hi = max(_hi, _lo + 2)
    _lags = np.arange(_lo, _hi + 1)
    _qs = (0.5, 1.0, 1.5, 2.0, 3.0)
    fit2 = estimate.hurst_gjr(_x, q2.value, _lags)
    multi2 = estimate.hurst_gjr_multi(_x, _qs, _lags)
    _mcse = estimate.monte_carlo_se(fit2.H, len(_x) - 1, q2.value, _lags, reps=40, seed=0)

    _fig2 = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=("log m(q, Δ) against log Δ", "ζ<sub>q</sub> against q"),
        horizontal_spacing=0.1,
    )
    _ld = np.log(_lags)
    for _f in multi2.fits:
        _fig2.add_scatter(
            x=_ld,
            y=np.log(_f.m),
            mode="markers",
            marker=dict(color=GREY, size=4, opacity=0.5),
            hovertemplate=f"q = {_f.q}<br>Δ = %{{customdata}}<extra></extra>",
            customdata=_lags,
            row=1,
            col=1,
        )
    _fig2.add_scatter(
        x=_ld,
        y=np.log(fit2.m),
        mode="markers",
        marker=dict(color=BLUE, size=7),
        customdata=_lags,
        hovertemplate="Δ = %{customdata} days<br>log m = %{y:.3f}<extra></extra>",
        row=1,
        col=1,
    )
    _fig2.add_scatter(
        x=_ld,
        y=fit2.intercept + fit2.slope * _ld,
        line=dict(color=ORANGE, width=2),
        hoverinfo="skip",
        row=1,
        col=1,
    )
    _fig2.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        text=f"q = {q2.value:g}: slope ζ<sub>q</sub> = {fit2.slope:.3f}, Ĥ = {fit2.H:.3f}",
        showarrow=False,
        font=dict(color="#52514e", size=13),
    )
    _qq = np.linspace(0, 3.2, 20)
    _fig2.add_scatter(
        x=_qq,
        y=multi2.H * _qq,
        line=dict(color=ORANGE, width=2),
        hoverinfo="skip",
        row=1,
        col=2,
    )
    _fig2.add_scatter(
        x=multi2.qs,
        y=multi2.zetas,
        mode="markers",
        marker=dict(color=GREY, size=8),
        hovertemplate="q = %{x}<br>ζ = %{y:.3f}<extra></extra>",
        row=1,
        col=2,
    )
    _fig2.add_scatter(
        x=[q2.value],
        y=[fit2.slope],
        mode="markers",
        marker=dict(color=BLUE, size=10),
        hovertemplate="q = %{x}<br>ζ = %{y:.3f}<extra></extra>",
        row=1,
        col=2,
    )
    _fig2.update_xaxes(title_text="log Δ (days)", row=1, col=1)
    _fig2.update_xaxes(title_text="q", range=[0, 3.2], row=1, col=2)
    _fig2.update_yaxes(range=[0, None], row=1, col=2)
    style(_fig2, 400)

    _figts = style(make_subplots(rows=1, cols=1), 170)
    _figts.add_scatter(x=DATES[src2.value], y=_x, line=dict(color=BLUE, width=0.6))
    _figts.update_layout(margin=dict(t=10, b=20), yaxis_title="log σ")

    mo.vstack(
        [
            mo.hstack([src2, q2, lags2], justify="start", gap=2),
            _fig2,
            mo.md(
                f"**Ĥ = ζ<sub>q</sub> / q = {fit2.H:.3f}** at q = {q2.value:g}, lags "
                f"{_lo} to {_hi} days. Regression SE {fit2.se_H:.3f}; Monte Carlo SE "
                f"{_mcse:.3f}. Multi-q Ĥ (slope of ζ<sub>q</sub> on q) = **{multi2.H:.3f}**. "
                f"{len(_x):,} days, {DATES[src2.value][0]:%Y-%m-%d} to "
                f"{DATES[src2.value][-1]:%Y-%m-%d}."
            ),
            _figts,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the figure shows.** Each blue dot is

    $$m(q, \Delta) = \frac{1}{N - \Delta}\sum_t
    \big|\log \sigma_{t+\Delta} - \log \sigma_t\big|^q,$$

    the average size of a $\Delta$-day move in log volatility, against the lag $\Delta$ on
    log scales. If log volatility scales like $\Delta^H$ the dots lie on a line of slope
    $\zeta_q = qH$, so $\hat H = \zeta_q / q$. Grey dots are the other values of $q$. On the
    right are the slopes $\zeta_q$ for five values of $q$: if they lie on a line through the
    origin, one $H$ describes every moment, and the slope of that line is the multi-q
    $\hat H$. This is the estimator of Gatheral, Jaisson and Rosenbaum (2018).
    The strip below the figure is the series itself, $\log \sigma_t = \tfrac12 \log
    \mathrm{RV}_t$.

    **The data.** Oxford-Man rv5 is the sum of squared 5-minute returns over each trading
    day, from the Oxford-Man Institute realised library, the data GJR used. The library
    closed in 2022; this is its last public file, from the Internet Archive, and runs to
    25 February 2022. The two daily estimators use only the day's high and low (Parkinson)
    or open, high, low and close (Garman-Klass) of `^GSPC` from Yahoo Finance, 2000 to
    yesterday. Before 2006 Yahoo's open is almost always the previous close, which
    contaminates Garman-Klass; Parkinson avoids the open.

    **The result.** GJR report $H$ of about 0.1 to 0.14 for the S&P 500, depending on the
    period. Here, on Oxford-Man rv5 with $q = 2$ and lags 1 to 50 days, $\hat H = 0.148$.
    That is rough: Brownian volatility would give 0.5.

    **Which error bar to read.** The regression SE treats the dots as independent, but
    every dot is computed from the same series, so it is several times too small. The
    Monte Carlo SE is the spread of $\hat H$ across 40 simulated fBm paths with the same
    length and the same $H$. Read that one: about 0.01 at the default settings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Try this: set the lag range to 1 to 200 days. Does Ĥ move? Why?": mo.md(
                """
                On Oxford-Man it falls from 0.148 to about 0.12, and beyond 50 days the dots
                bend below the line. Volatility mean-reverts: over months, log σ cannot
                wander as far as a pure power law allows, so m(q, Δ) grows more slowly. GJR
                model log volatility as a fractional Ornstein-Uhlenbeck process, which looks
                like fBm at short lags and is stationary at long ones. Roughness is a
                statement about short lags, so short lags are where to measure it.
                """
            ),
            "Try this: switch to the Parkinson range estimator. Is daily data rougher?": mo.md(
                """
                Ĥ drops to about 0.09. The volatility is the same; the measurement is worse.
                A range estimate is the true variance times a noisy factor, so
                log σ̂ = log σ + ε, with ε roughly independent from day to day. That adds the
                constant 2 Var(ε) to m(2, Δ) at every lag. A constant lifts the small values
                at short lags proportionally more than the large ones at long lags, so the
                line flattens and Ĥ is biased down. An opposite bias exists too: a daily
                realised variance averages spot variance over the day, which smooths the
                path and pushes Ĥ up (GJR discuss this smoothing). On this data the noise
                wins. Session 01 separates the two.
                """
            ),
            "Try this: move q from 0.5 to 3. Does Ĥ change?": mo.md(
                """
                Barely: on Oxford-Man rv5 with lags 1 to 50 it stays between 0.146 and 0.151.
                The ζ points sit on a straight line through the origin, so a single H
                describes small moves (q = 0.5) and large moves (q = 3) alike. A
                multifractal model would give a ζ curve that bends downwards. GJR use this to
                argue for fBm rather than a multifractal for log volatility.
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Does the estimator find H when we know it?
    """)
    return


@app.cell(hide_code=True)
def _(heston_variance, np):
    # 40 Heston variance paths of 4,096 trading days, simulated once at load.
    _v = heston_variance(2**12, 2**12 / 252, 40, seed=11)
    HESTON_LOGVOL = 0.5 * np.log(_v)
    return (HESTON_LOGVOL,)


@app.cell(hide_code=True)
def _(mo):
    model3 = mo.ui.radio(
        options=["fBm with chosen H", "Heston variance (true H = 1/2)"],
        value="fBm with chosen H",
        label="model",
        inline=True,
    )
    h3 = mo.ui.slider(0.05, 0.95, 0.01, value=0.15, label="true H (fBm)", show_value=True)
    lags3 = mo.ui.range_slider(1, 250, 1, value=[1, 50], label="lags (days)", show_value=True)
    return h3, lags3, model3


@app.cell(hide_code=True)
def _(
    BLUE,
    GREY,
    HESTON_LOGVOL,
    ORANGE,
    estimate,
    fbm,
    h3,
    lags3,
    make_subplots,
    mo,
    model3,
    np,
    style,
):
    _is_fbm = model3.value.startswith("fBm")
    _true = h3.value if _is_fbm else 0.5
    _paths = fbm(h3.value, 2**12, 1.0, 40, seed=3) if _is_fbm else HESTON_LOGVOL
    _lo, _hi = (int(v) for v in lags3.value)
    _hi = max(_hi, _lo + 2)
    _lags = np.arange(_lo, _hi + 1)
    _fits = [estimate.hurst_gjr(p, 2.0, _lags) for p in _paths]
    _h = np.array([f.H for f in _fits])
    _f0 = _fits[0]

    _fig3 = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.4, 0.3, 0.3],
        subplot_titles=("simulated log σ, path 1", "log m(2, Δ), path 1", "Ĥ over 40 paths"),
        horizontal_spacing=0.08,
    )
    _fig3.add_scatter(y=_paths[0], line=dict(color=BLUE, width=0.6), row=1, col=1)
    _ld = np.log(_lags)
    _fig3.add_scatter(
        x=_ld,
        y=np.log(_f0.m),
        mode="markers",
        marker=dict(color=BLUE, size=6),
        customdata=_lags,
        hovertemplate="Δ = %{customdata}<extra></extra>",
        row=1,
        col=2,
    )
    _fig3.add_scatter(
        x=_ld,
        y=_f0.intercept + _f0.slope * _ld,
        line=dict(color=ORANGE, width=2),
        hoverinfo="skip",
        row=1,
        col=2,
    )
    _fig3.add_scatter(
        x=_ld,
        y=_f0.intercept + 2 * _true * (_ld - _ld.mean()) + _f0.slope * _ld.mean(),
        line=dict(color=GREY, width=1.5, dash="dot"),
        hoverinfo="skip",
        row=1,
        col=2,
    )
    _fig3.add_histogram(
        x=_h,
        marker=dict(color=BLUE, line=dict(color="white", width=1)),
        nbinsx=15,
        row=1,
        col=3,
    )
    _fig3.add_vline(x=_true, line=dict(color=GREY, width=2, dash="dot"), row=1, col=3)
    _fig3.update_xaxes(title_text="day", row=1, col=1)
    _fig3.update_xaxes(title_text="log Δ", row=1, col=2)
    _fig3.update_xaxes(title_text="Ĥ", row=1, col=3)
    style(_fig3, 360)

    mo.vstack(
        [
            mo.hstack([model3, h3, lags3], justify="start", gap=2),
            _fig3,
            mo.md(
                f"True H = **{_true:.2f}**. Path 1: Ĥ = {_f0.H:.3f}, regression SE "
                f"{_f0.se_H:.3f}. Across 40 paths: mean Ĥ = **{_h.mean():.3f}**, standard "
                f"deviation {_h.std(ddof=1):.3f}. Dotted grey: the line with the true slope "
                "2H; orange: the fitted line."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Why we trust the S&P 500 number.** Each simulated path has 4,096 days, about the
    length of the real series, and is observed without error. On fBm the estimator
    returns the $H$ you set from 0.05 to about 0.9, with a spread across paths of about 0.01
    at $H = 0.15$: the Monte Carlo SE quoted in panel 2. Near $H = 0.95$ it reads a little
    low (0.925 on average), because strongly persistent paths make the sample mean of
    $m(q, \Delta)$ converge slowly. The regression SE of a single
    path is much smaller than that spread, which is why panel 2 does not rely on it.

    **The Heston control.** Heston variance is driven by Brownian motion, so its true
    $H$ is 1/2. At lags of 1 to 10 days the estimator gives about 0.47; widen the lags and
    it falls further, because mean reversion (here over half a year) bends the line, the
    same effect as the 200-day experiment in panel 2. What it never gives is 0.15. A
    classical model observed exactly does not look rough to this estimator, so 0.15 on the
    S&P 500 is a property of the data, not of the method.

    **What this does not settle.** Real realised variance is an estimate with error, and
    panel 2 already shows that error moving $\hat H$ from 0.15 to 0.09. If noise can push
    the estimate down, could a smooth volatility observed with noise look rough? Fukasawa,
    Takabatake and Westphal (2019), *Is volatility rough?*, take this question seriously
    and build an estimator that models the measurement error. Session 01 adds noise to the
    simulations in this panel and asks when the estimator lies.
    """)
    return


if __name__ == "__main__":
    app.run()
