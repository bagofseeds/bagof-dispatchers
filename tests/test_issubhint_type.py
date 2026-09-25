"""Tests for `issubhint` with a `type[...]` super-hint."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers.core import issubhint
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
