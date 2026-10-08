"""Daily realised variance of the S&P 500 from two sources.

Source A, "oxford-man": the Oxford-Man Institute realised library v0.3 (Heber, Lunde,
Shephard and Sheppard 2009), the data behind Gatheral, Jaisson and Rosenbaum (2018). The
library was withdrawn in 2022. The last public file was retrieved on 2026-10-08 from the
Internet Archive snapshot

    https://web.archive.org/web/20220301022212id_/https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip

(zip sha256 f620ad530e35e4e28c667cfd53a9b68e8cac42133bdf3e734614fffc58f87be1, one CSV of
57 MB). We keep symbol .SPX, column rv5: the sum of squared 5-minute returns over the
trading day, open to close, no overnight return. 5,552 days, 2000-01-03 to 2022-02-25.
The slim cache is data/spx_rv5_oxford_man.parquet (columns date, rv5).

Source B, "parkinson" and "garman-klass": daily range-based variance from the open, high,
low and close of ^GSPC on Yahoo Finance via yfinance, 2000-01-03 onwards, cached in
data/spx_ohlc_yahoo.parquet. Two estimators of the day's variance:

    Parkinson (1980):     (log H/L)^2 / (4 log 2)
    Garman-Klass (1980):  0.5 (log H/L)^2 - (2 log 2 - 1) (log C/O)^2

Known quirk: Yahoo often reports the ^GSPC open as the previous close (95 to 98% of days
in 2000 to 2005, 43% in 2006, 16% in 2007, about 1% in 2008 to 2010, 9 to 26% in 2011 to
2013, almost never from 2014). Garman-Klass then mixes the overnight move into its
(log C/O)^2 term. Parkinson uses only the high and low and is the
cleaner of the two on this data.

A range estimator uses four numbers per day where rv5 uses about 78 returns, so it is far
noisier. Session 00 shows what that does to H.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OXFORD_MAN_URL = (
    "https://web.archive.org/web/20220301022212id_/"
    "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip"
)
OXFORD_MAN_FILE = DATA_DIR / "spx_rv5_oxford_man.parquet"
YAHOO_FILE = DATA_DIR / "spx_ohlc_yahoo.parquet"

Source = Literal["oxford-man", "parkinson", "garman-klass"]
SOURCES: dict[str, str] = {
    "oxford-man": "Oxford-Man rv5 (5-minute returns)",
    "parkinson": "Parkinson range (daily high, low)",
    "garman-klass": "Garman-Klass (daily open, high, low, close)",
}


def parkinson(ohlc: pd.DataFrame) -> pd.Series:
    """Parkinson daily variance (log H/L)^2 / (4 log 2) from columns high, low."""
    hl = np.log(ohlc["high"] / ohlc["low"])
    return (hl**2 / (4 * np.log(2))).rename("rv")


def garman_klass(ohlc: pd.DataFrame) -> pd.Series:
    """Garman-Klass daily variance from columns open, high, low, close."""
    hl = np.log(ohlc["high"] / ohlc["low"])
    co = np.log(ohlc["close"] / ohlc["open"])
    return (0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).rename("rv")


def load_spx_rv(source: Source = "oxford-man") -> pd.Series:
    """Daily realised variance of the S&P 500 as a Series indexed by date.

    Reads the committed caches in data/. Days with nonpositive variance are dropped, so
    0.5 * log of the result is always finite. Consecutive entries are consecutive trading
    days in the source.
    """
    if source == "oxford-man":
        rv = pd.read_parquet(OXFORD_MAN_FILE).set_index("date")["rv5"].rename("rv")
    elif source in ("parkinson", "garman-klass"):
        ohlc = pd.read_parquet(YAHOO_FILE).set_index("date")
        rv = parkinson(ohlc) if source == "parkinson" else garman_klass(ohlc)
    else:
        raise ValueError(f"unknown source {source!r}; choose from {list(SOURCES)}")
    rv = rv[rv > 0].dropna()
    rv.index = pd.DatetimeIndex(rv.index)
    return rv


def log_vol(rv: pd.Series) -> np.ndarray:
    """log sigma = 0.5 * log(realised variance), as a float array."""
    return 0.5 * np.log(rv.to_numpy(dtype=np.float64))


def build_oxford_man_cache(url: str = OXFORD_MAN_URL, out: Path = OXFORD_MAN_FILE) -> Path:
    """Download the Oxford-Man zip and write the slim .SPX (date, rv5) parquet."""
    with urllib.request.urlopen(url, timeout=300) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(z.open(name), usecols=["Unnamed: 0", "Symbol", "rv5"])
    spx = df[df["Symbol"] == ".SPX"]
    dates = pd.to_datetime(spx["Unnamed: 0"].str[:10])
    slim = pd.DataFrame({"date": dates.to_numpy(), "rv5": spx["rv5"].to_numpy()})
    slim = slim.sort_values("date").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(out, index=False)
    return out


def build_yahoo_cache(start: str = "2000-01-01", out: Path = YAHOO_FILE) -> Path:
    """Download ^GSPC daily OHLC from Yahoo Finance and write (date, open, high, low, close)."""
    import yfinance as yf

    d = yf.download("^GSPC", start=start, auto_adjust=False, progress=False)
    d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
    d = d[["open", "high", "low", "close"]].dropna()
    d.index.name = "date"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.reset_index().to_parquet(out, index=False)
    return out
