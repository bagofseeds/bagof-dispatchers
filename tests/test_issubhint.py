"""Tests for `issubhint`, focused on Union super-hints."""

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers.core import issubhint

# (hint, superhint, expected)
UNION_CASES = [
    # A plain type is a subhint of a union that contains it.
    (int, tx.Union[int, str], True),
    (str, tx.Union[int, str], True),
    (bytes, tx.Union[int, str], False),
    # Subclasses count (bool is a subclass of int).
    (bool, tx.Union[int, str], True),
    # Optional[X] is Union[X, None].
    (int, tx.Optional[int], True),
    (type(None), tx.Optional[int], True),
    (bytes, tx.Optional[int], False),
    # Parametrised members.
    (tx.List[int], tx.Union[tx.List[int], str], True),
    (tx.List[str], tx.Union[tx.List[int], str], False),
    # Union hint vs union super-hint.
    (tx.Union[int, str], tx.Union[int, str, bytes], True),
    (tx.Union[int, str, bytes], tx.Union[int, str], False),
    (tx.Union[int, str], tx.Union[int, str], True),
    # The bare `tx.Union` is not a subhint of a parametrised union.
    (tx.Union, tx.Union[int, str], False),
]

IDS = [f"{h}<:{s}" for h, s, _ in UNION_CASES]


@pytest.mark.parametrize("hint,superhint,expected", UNION_CASES, ids=IDS)
def test_issubhint_union(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_plain_type_is_subhint_of_containing_union() -> None:
    # Regression: a non-union hint used to always fail against a union.
    assert issubhint(int, tx.Union[int, str]) is True


def test_bare_union_hint_does_not_raise() -> None:
    # Regression: this used to raise TypeError instead of returning a bool.
    assert issubhint(tx.Union, tx.Union[int, str]) is False


def test_non_union_superhint_is_unaffected() -> None:
    assert issubhint(int, int) is True
    assert issubhint(int, str) is False
    assert issubhint(bool, int) is True
    assert issubhint(int, tx.Any) is True
