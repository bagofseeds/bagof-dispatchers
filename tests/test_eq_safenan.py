"""Tests for `eq_safenan` (NaN-safe comparison), numpy-free at the root."""

# stdlib
import inspect
from fractions import Fraction

# dependencies
import pytest

# local
from bagof.dispatchers.core import eq_safenan


def test_two_nans_compare_equal_after_mapping() -> None:
    nan = float("nan")
    assert nan != nan
    assert eq_safenan(nan) == eq_safenan(nan)


def test_the_marker_is_recognisable() -> None:
    assert repr(eq_safenan(float("nan"))) == "<NaN>"


@pytest.mark.parametrize("value", [1, 1.0, 0, -3.5, "x", None, Fraction(1, 2)])
def test_non_nan_values_are_unchanged(value: object) -> None:
    assert eq_safenan(value) is value


def test_a_complex_nan_is_left_unequal_to_itself() -> None:
    # Only real numbers are recognised; a complex NaN is returned as is.
    cnan = complex(float("nan"), 0.0)
    mapped = eq_safenan(cnan)
    assert mapped is cnan


def test_no_numpy_import_at_the_root() -> None:
    # `numbers.Real` covers numpy scalars via ABC registration, so the root
    # relation must not import numpy.
    from bagof.dispatchers.core import _introspect

    source = inspect.getsource(_introspect)
    assert "import numpy" not in source


def test_numpy_nan_is_recognised_if_available() -> None:
    np = pytest.importorskip("numpy")
    assert eq_safenan(np.float64("nan")) == eq_safenan(np.float64("nan"))
    assert eq_safenan(np.float32(1.5)) == eq_safenan(np.float32(1.5))
