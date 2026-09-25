"""Tests for `issubhint` with a `TypeVar` on either side."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers.core import issubhint
from bagof.dispatchers.core._relation import _issubtypevar

ANY_T = tx.TypeVar("ANY_T")
INT_T = tx.TypeVar("INT_T", bound=int)
BOOL_T = tx.TypeVar("BOOL_T", bound=bool)
NUM_T = tx.TypeVar("NUM_T", int, str)
NARROW_T = tx.TypeVar("NARROW_T", bool, str)
WIDE_T = tx.TypeVar("WIDE_T", int, str, bytes)

# --- a typevar as the super-hint ---------------------------------------

# (hint, superhint, expected)
SUPER_CASES = [
    # An unconstrained typevar stands for anything.
    (int, ANY_T, True),
    (tx.List[int], ANY_T, True),
    (ANY_T, ANY_T, True),
    # A bound accepts the bound and its subclasses.
    (int, INT_T, True),
    (bool, INT_T, True),
    (str, INT_T, False),
    (tx.Annotated[int, "meta"], INT_T, True),
    # Bound against bound: the narrower one is the subhint.
    (BOOL_T, INT_T, True),
    (INT_T, BOOL_T, False),
    # An unconstrained typevar is not a subhint of a bounded one.
    (ANY_T, INT_T, False),
    (ANY_T, NUM_T, False),
    # Constraints accept anything that satisfies one of them.
    (int, NUM_T, True),
    (str, NUM_T, True),
    (bool, NUM_T, True),
    (bytes, NUM_T, False),
    # Constrained against constrained: every constraint must be covered.
    (NARROW_T, NUM_T, True),
    (NUM_T, NARROW_T, False),
    (NUM_T, WIDE_T, True),
    (WIDE_T, NUM_T, False),
    # A bounded typevar against a constrained one, and the reverse.
    (INT_T, NUM_T, True),
    (NUM_T, INT_T, False),
    # Anything is a subhint of an unconstrained typevar.
    (NUM_T, ANY_T, True),
    (INT_T, ANY_T, True),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    SUPER_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in SUPER_CASES],
)
def test_issubhint_typevar_superhint(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


# --- a typevar as the hint ---------------------------------------------

# (hint, superhint, expected)
HINT_CASES = [
    # A bounded typevar stands for its bound.
    (INT_T, int, True),
    (INT_T, object, True),
    (INT_T, str, False),
    (BOOL_T, int, True),
    (INT_T, tx.Union[int, str], True),
    # A constrained typevar must satisfy the super-hint through *every*
    # constraint...
    (NUM_T, tx.Union[int, str], True),
    (NUM_T, tx.Union[int, str, bytes], True),
    (NUM_T, int, False),
    (NARROW_T, tx.Union[int, str], True),
    # ... and against a non-union super-hint, each one on its own.
    (tx.TypeVar("BOTH_T", bool, int), int, True),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    HINT_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in HINT_CASES],
)
def test_issubhint_typevar_hint(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_issubtypevar_rejects_a_non_typevar_superhint() -> None:
    with pytest.raises(TypeError, match="is not a TypeVar"):
        _issubtypevar(int, int)


def test_an_annotated_typevar_is_a_subhint_of_itself() -> None:
    # `Annotated` is transparent, so this is the exact-match case.
    assert issubhint(tx.Annotated[ANY_T, "meta"], ANY_T) is True
