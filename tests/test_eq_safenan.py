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


def test_a_real_nan_complex_compares_equal_to_itself_after_mapping() -> None:
    cnan = complex(float("nan"), 0.0)
    assert cnan != cnan
    assert eq_safenan(cnan) == eq_safenan(cnan)


def test_an_imag_nan_complex_compares_equal_to_itself_after_mapping() -> None:
    cnan = complex(0.0, float("nan"))
    assert cnan != cnan
    assert eq_safenan(cnan) == eq_safenan(cnan)


def test_complex_nans_with_distinct_non_nan_parts_stay_distinct() -> None:
    a = complex(float("nan"), 1.0)
    b = complex(float("nan"), 2.0)
    assert eq_safenan(a) != eq_safenan(b)


def test_a_non_nan_complex_is_unchanged() -> None:
    value = complex(1, 2)
    assert eq_safenan(value) is value


def test_a_mapped_real_nan_differs_from_a_mapped_complex_nan() -> None:
    assert eq_safenan(float("nan")) != eq_safenan(complex(float("nan"), 0.0))


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
