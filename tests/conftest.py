"""Parametrise the maths test modules over roughvol and, if present, exercises.

exercises is an optional top-level package Corey may add later to work the
same tests against his own from-scratch implementation. Its absence must make
the parametrised tests skip cleanly, not fail, so conftest resolves each
module lazily inside the fixture rather than importing it at collection time.
"""

import importlib

import pytest

_IMPLS = ["roughvol"]
try:
    importlib.import_module("exercises")
    _IMPLS.append("exercises")
except ImportError:
    pass


@pytest.fixture(params=_IMPLS, scope="session")
def impl(request):
    return request.param


def _resolve(impl, name):
    try:
        return importlib.import_module(f"{impl}.{name}")
    except ImportError:
        pytest.skip(f"{impl}.{name} not importable")


@pytest.fixture
def paths_mod(impl):
    return _resolve(impl, "paths")


@pytest.fixture
def variation_mod(impl):
    return _resolve(impl, "variation")


@pytest.fixture
def plotting_mod(impl):
    return _resolve(impl, "plotting")
