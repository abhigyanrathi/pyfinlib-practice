"""Smoke test: the package imports and exposes a version string."""

import pyfinlib_practice


def test_package_imports() -> None:
    assert pyfinlib_practice.__version__
