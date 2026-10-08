"""Registers --impl at the repository root, where pytest reads options before collection.

The `impl` fixture that uses it lives in tests/conftest.py.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--impl",
        default="roughvol",
        help="package whose modules the mathematical tests import (default: roughvol)",
    )
