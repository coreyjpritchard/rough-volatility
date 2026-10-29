"""The data caches load and look like daily S&P 500 variance.

The Oxford-Man cache is committed. The Yahoo cache is built locally, so its tests skip
when the file is absent; the range estimators are also checked on synthetic OHLC.
"""

import numpy as np
import pandas as pd
import pytest

from roughvol import realised

needs_yahoo = pytest.mark.skipif(
    not realised.YAHOO_FILE.exists(),
    reason="data/spx_ohlc_yahoo.parquet is built locally: python -m roughvol.realised yahoo",
)


def _check_daily_variance(rv, min_days=5000):
    assert len(rv) > min_days
    assert rv.index.is_monotonic_increasing
    assert (rv > 0).all()
    # Median daily variance between (0.3% daily vol)^2 and (2% daily vol)^2.
    assert 0.003**2 < rv.median() < 0.02**2
    assert np.isfinite(realised.log_vol(rv)).all()


@pytest.mark.parametrize(
    "source",
    [s if s == "oxford-man" else pytest.param(s, marks=needs_yahoo) for s in realised.SOURCES],
)
def test_load_spx_rv(source):
    _check_daily_variance(realised.load_spx_rv(source))


def test_oxford_man_date_range():
    rv = realised.load_spx_rv("oxford-man")
    assert str(rv.index[0].date()) == "2000-01-03"
    assert str(rv.index[-1].date()) == "2022-02-25"
    assert len(rv) == 5552


def _synthetic_ohlc(n=2000, daily_vol=0.01, seed=0):
    """Daily OHLC from a driftless log price observed at 78 points a day."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, daily_vol / np.sqrt(78), size=(n, 78))
    path = np.cumsum(steps, axis=1)
    start = np.concatenate([[0.0], np.cumsum(path[:-1, -1])])
    log_open = start
    log_close = start + path[:, -1]
    log_high = np.maximum(start[:, None] + path, start[:, None]).max(axis=1)
    log_low = np.minimum(start[:, None] + path, start[:, None]).min(axis=1)
    dates = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100 * np.exp(log_open),
            "high": 100 * np.exp(log_high),
            "low": 100 * np.exp(log_low),
            "close": 100 * np.exp(log_close),
        }
    )


@pytest.mark.parametrize("source", ["parkinson", "garman-klass"])
def test_range_estimators_on_synthetic_ohlc(tmp_path, monkeypatch, source):
    path = tmp_path / "spx_ohlc_yahoo.parquet"
    _synthetic_ohlc().to_parquet(path, index=False)
    monkeypatch.setattr(realised, "YAHOO_FILE", path)
    assert realised.yahoo_available()
    rv = realised.load_spx_rv(source)
    _check_daily_variance(rv, min_days=1500)
    # Both estimators are unbiased for a continuous path; sampling at 78 points a day
    # biases the range low, so the mean sits a little under 1e-4.
    assert 0.6e-4 < rv.mean() < 1.1e-4


def test_missing_yahoo_file_says_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setattr(realised, "YAHOO_FILE", tmp_path / "absent.parquet")
    assert not realised.yahoo_available()
    with pytest.raises(FileNotFoundError, match="python -m roughvol.realised yahoo"):
        realised.load_spx_rv("parkinson")
    # The Oxford-Man source does not need the Yahoo file.
    assert len(realised.load_spx_rv("oxford-man")) == 5552
