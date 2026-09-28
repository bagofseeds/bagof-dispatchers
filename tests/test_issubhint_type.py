"""Tests for `issubhint` with a `type[...]` super-hint."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers.core import ishintstance, issubhint
from bagof.dispatchers.core._relation import _issubtype

# (hint, superhint, expected)
TYPE_CASES = [
    # Arguments are compared as classes.
    (tx.Type[bool], tx.Type[int], True),
    (tx.Type[int], tx.Type[bool], False),
    (tx.Type[int], tx.Type[int], True),
    # Every `type[...]` is a subhint of the bare `type`...
    (tx.Type[int], tx.Type, True),
    (tx.Type[int], type, True),
    # ... but the bare `type` is not a subhint of a parametrised one.
    (tx.Type, tx.Type[int], False),
    (type, tx.Type[int], False),
    # A non-type hint is never a subhint of a `type[...]`.
    (int, tx.Type[int], False),
    (tx.List[int], tx.Type[int], False),
    # `Annotated` is transparent on both sides.
    (tx.Annotated[tx.Type[bool], "meta"], tx.Type[int], True),
    (tx.Type[bool], tx.Annotated[tx.Type[int], "meta"], True),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    TYPE_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in TYPE_CASES],
)
def test_issubhint_type(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_issubtype_rejects_a_non_type_superhint() -> None:
    with pytest.raises(TypeError, match="is not a type"):
        _issubtype(tx.Type[int], int)


# --- type[...] read through the full relation (0.2.0) -------------------
#
# `type[...]` now compares its argument through `issubhint`, so `type[Any]`,
# `type[Union]`, `type[TypeVar]` and `type[Exact[C]]` each behave correctly
# rather than degenerating.

_T = tx.TypeVar("_T")


def test_ishintstance_type_any() -> None:
    # Every class is a `type[Any]` (Q4 fix; used to be all False).
    assert ishintstance(int, tx.Type[tx.Any]) is True
    assert ishintstance(str, tx.Type[tx.Any]) is True
    assert ishintstance(1, tx.Type[tx.Any]) is False


def test_ishintstance_type_free_typevar() -> None:
    assert ishintstance(int, tx.Type[_T]) is True


def test_ishintstance_type_union() -> None:
    assert ishintstance(int, tx.Type[tx.Union[int, str]]) is True
    assert ishintstance(bytes, tx.Type[tx.Union[int, str]]) is False


def test_ishintstance_type_exact() -> None:
    assert ishintstance(int, tx.Type[Exact[int]]) is True
    assert ishintstance(bool, tx.Type[Exact[int]]) is False


def test_issubhint_type_any_and_union() -> None:
    assert issubhint(tx.Type[int], tx.Type[tx.Any]) is True
    assert issubhint(tx.Type[int], tx.Type[tx.Union[int, str]]) is True
    assert issubhint(tx.Type[bytes], tx.Type[tx.Union[int, str]]) is False


def test_issubhint_type_exact_leaf() -> None:
    assert issubhint(tx.Type[Exact[int]], tx.Type[int]) is True
    assert issubhint(tx.Type[int], tx.Type[Exact[int]]) is False
