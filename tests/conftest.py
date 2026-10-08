"""Shared fixtures. `impl` points the mathematical tests at an implementation package.

    pytest                      # tests roughvol
    pytest --impl exercises     # tests Corey's own exercises/fbm.py, exercises/estimate.py, ...

`impl.fbm`, `impl.estimate` and `impl.heston_vol` resolve to modules of the chosen
package. A test whose module is missing is skipped, so exercises can be written one file
at a time. The --impl option itself is registered in the root conftest.py, because
pytest only reads options from conftest files it loads before collection.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Impl:
    def __init__(self, package: str) -> None:
        self.package = package

    def __getattr__(self, name: str):
        try:
            return importlib.import_module(f"{self.package}.{name}")
        except ModuleNotFoundError:
            pytest.skip(f"{self.package}.{name} not implemented yet")

    def __repr__(self) -> str:
        return f"impl({self.package})"


@pytest.fixture(scope="session")
def impl(request: pytest.FixtureRequest) -> _Impl:
    return _Impl(request.config.getoption("--impl"))
