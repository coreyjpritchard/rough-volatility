"""Tests for plotting.style()."""

import matplotlib
import pytest


@pytest.fixture(autouse=True)
def _restore_rcparams():
    original = dict(matplotlib.rcParams)
    yield
    matplotlib.rcParams.update(original)
    # Drop any keys style() may have added that weren't in the original set.
    for key in list(matplotlib.rcParams):
        if key not in original:
            del matplotlib.rcParams[key]


def test_style_is_idempotent_and_sets_defaults(plotting_mod):
    plotting_mod.style()
    plotting_mod.style()
    assert tuple(matplotlib.rcParams["figure.figsize"]) == (10.0, 6.0)
    assert matplotlib.rcParams["axes.grid"] is True
    assert matplotlib.rcParams["grid.alpha"] == 0.3
    cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle[0] == plotting_mod.ACCENT
