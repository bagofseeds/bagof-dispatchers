"""Smoke tests."""

import importlib


def test_package_is_importable() -> None:
    """The package should be importable after installation."""
    module = importlib.import_module("bagof.dispatchers")
    assert module is not None
