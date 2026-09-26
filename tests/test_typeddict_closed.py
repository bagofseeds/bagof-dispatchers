"""PEP 728 ``closed`` / ``extra_items`` at the value level.

`ishintstance` reads a `TypedDict`'s treatment of keys beyond the declared
ones: an open one allows them, a `closed=True` one rejects them, and an
`extra_items=` one checks each extra value against that type.

The `closed=` and `extra_items=` class keywords, and the `__closed__` /
`__extra_items__` attributes they set, exist only on new-enough
`typing_extensions`. Tests that need them are skipped where they cannot be
expressed, rather than failing.
"""

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers.core import ishintstance


def _closed_supported() -> bool:
    """Whether this `typing_extensions` can express a closed `TypedDict`."""
    try:

        class _Probe(tx.TypedDict, closed=True):
            a: int

    except Exception:
        return False
    return getattr(_Probe, "__closed__", None) is True


def _extra_items_supported() -> bool:
    """Whether this `typing_extensions` can express `extra_items=`."""
    try:

        class _Probe(tx.TypedDict, extra_items=int):
            a: int

    except Exception:
        return False
    return getattr(_Probe, "__extra_items__", None) is int


CLOSED = pytest.mark.skipif(
    not _closed_supported(),
    reason="typing_extensions here cannot express closed=True",
)
EXTRA = pytest.mark.skipif(
    not _extra_items_supported(),
    reason="typing_extensions here cannot express extra_items=",
)


# --- open (the default): extras allowed, unchanged ---------------------


class Open(tx.TypedDict):
    a: int


def test_open_allows_extra_keys() -> None:
    assert ishintstance({"a": 1, "b": "anything"}, Open) is True


def test_open_matches_exact_keys() -> None:
    assert ishintstance({"a": 1}, Open) is True


def test_open_rejects_missing_required_key() -> None:
    assert ishintstance({"b": 2}, Open) is False


def test_open_rejects_wrong_value_type() -> None:
    assert ishintstance({"a": "not-an-int"}, Open) is False


# --- closed=True: extras rejected --------------------------------------


@CLOSED
def test_closed_rejects_extra_key() -> None:
    class Closed(tx.TypedDict, closed=True):
        a: int

    assert ishintstance({"a": 1, "b": 2}, Closed) is False


@CLOSED
def test_closed_matches_exact_keys() -> None:
    class Closed(tx.TypedDict, closed=True):
        a: int

    assert ishintstance({"a": 1}, Closed) is True


@CLOSED
def test_closed_rejects_missing_required_key() -> None:
    class Closed(tx.TypedDict, closed=True):
        a: int

    assert ishintstance({}, Closed) is False


@CLOSED
def test_closed_still_checks_declared_value_types() -> None:
    class Closed(tx.TypedDict, closed=True):
        a: int

    assert ishintstance({"a": "x"}, Closed) is False


@CLOSED
def test_closed_honours_notrequired() -> None:
    class Closed(tx.TypedDict, closed=True):
        a: int
        b: tx.NotRequired[str]

    # The optional declared key may be absent...
    assert ishintstance({"a": 1}, Closed) is True
    # ...present with the right type...
    assert ishintstance({"a": 1, "b": "y"}, Closed) is True
    # ...present with the wrong type is rejected...
    assert ishintstance({"a": 1, "b": 2}, Closed) is False
    # ...and an *undeclared* extra key is still rejected.
    assert ishintstance({"a": 1, "c": 3}, Closed) is False


# --- extra_items=SomeType: typed extras --------------------------------


@EXTRA
def test_extra_items_accepts_matching_extra_value() -> None:
    class TD(tx.TypedDict, extra_items=int):
        a: int

    assert ishintstance({"a": 1, "b": 2}, TD) is True


@EXTRA
def test_extra_items_rejects_mismatching_extra_value() -> None:
    class TD(tx.TypedDict, extra_items=int):
        a: int

    assert ishintstance({"a": 1, "b": "not-an-int"}, TD) is False


@EXTRA
def test_extra_items_allows_no_extras() -> None:
    class TD(tx.TypedDict, extra_items=int):
        a: int

    assert ishintstance({"a": 1}, TD) is True


@EXTRA
def test_extra_items_still_checks_declared_value_types() -> None:
    class TD(tx.TypedDict, extra_items=int):
        a: int

    assert ishintstance({"a": "x", "b": 2}, TD) is False


@EXTRA
def test_extra_items_never_rejects_any_extra() -> None:
    class TD(tx.TypedDict, extra_items=tx.Never):
        a: int

    # No value satisfies `Never`, so any extra key is refused...
    assert ishintstance({"a": 1, "b": 2}, TD) is False
    # ...while exactly the declared keys still match.
    assert ishintstance({"a": 1}, TD) is True
