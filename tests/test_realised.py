"""The committed data caches load and look like daily S&P 500 variance."""

import numpy as np
import pytest

from roughvol import realised


@pytest.mark.parametrize("source", list(realised.SOURCES))
def test_load_spx_rv(source):
    rv = realised.load_spx_rv(source)
    assert len(rv) > 5000
    assert rv.index.is_monotonic_increasing
    assert (rv > 0).all()
    # Median daily variance between (0.3% daily vol)^2 and (2% daily vol)^2.
    assert 0.003**2 < rv.median() < 0.02**2
    assert np.isfinite(realised.log_vol(rv)).all()


def test_oxford_man_date_range():
    rv = realised.load_spx_rv("oxford-man")
    assert str(rv.index[0].date()) == "2000-01-03"
    assert str(rv.index[-1].date()) == "2022-02-25"
    assert len(rv) == 5552
