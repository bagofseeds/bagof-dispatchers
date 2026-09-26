"""Tests for `issubhint` with a `Literal` on either side."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers.core import issubhint
from bagof.dispatchers.core._relation import _issubliteral

L = tx.Literal

# --- a Literal as the super-hint ----------------------------------------

# (hint, superhint, expected)
LITERAL_CASES = [
    # A literal is a subhint of one that contains all of its values.
    (L[1], L[1, 2], True),
    (L[1, 2], L[1, 2, 3], True),
    (L[1, 2], L[1], False),
    (L["a"], L["a", "b"], True),
    (L["a"], L["b"], False),
    (L[None], L[None, 1], True),
    # A mixed literal needs every value present.
    (L[1, "a"], L[1, "a", 2], True),
    (L[1, "a"], L[1, 2], False),
    # Every literal is a subhint of the bare `Literal`...
    (L[1], L, True),
    (L, L, True),
    # ... but the bare `Literal` is not a subhint of a parametrised one.
    (L, L[1, 2], False),
    # A non-literal hint is never a subhint of a literal.
    (int, L[1, 2], False),
    (tx.Union[int, str], L[1, 2], False),
    # `Annotated` is transparent on both sides.
    (tx.Annotated[L[1], "meta"], L[1, 2], True),
    (L[1], tx.Annotated[L[1, 2], "meta"], True),
    # PEP 586 makes literal matching type-aware (issue #6): `1 == True` and
    # `1 == 1.0` are True in Python, but they denote different literals.
    (L[True], L[1], False),
    (L[1], L[True], False),
    (L[1], L[1.0], False),
    (L[1.0], L[1], False),
    # ... while same-type values still match.
    (L[True], L[True, False], True),
    (L[1], L[1, 2], True),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    LITERAL_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in LITERAL_CASES],
)
def test_issubhint_literal(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_issubliteral_rejects_a_non_literal_superhint() -> None:
    with pytest.raises(TypeError, match="is not a Literal"):
        _issubliteral(L[1], int)


# --- a Literal as the hint, against a non-Literal super-hint ------------

# (hint, superhint, expected)
LITERAL_HINT_CASES = [
    # A literal is a subhint of a class every one of its values belongs
    # to, whatever the class's own subhint relation to the literal would
    # otherwise be dispatched on.
    (L[1], int, True),
    (L[1], object, True),
    (L[1], str, False),
    (L["a"], tx.Union[int, str], True),
    (L[None], type(None), True),
    # Every value has to match, not just one of them.
    (L[1, "a"], int, False),
    (L[1, "a"], tx.Union[int, str], True),
    # PEP 586 keeps the value's own type, so `Literal[True]` is an `int`
    # (`bool` is a subclass of `int`) but `Literal[1]` is not a `bool`.
    (L[True], int, True),
    (L[1], bool, False),
    # A bare, unparametrised `Literal` has no values to check and stays
    # False against everything but `Literal` itself and `Any`.
    (L, int, False),
    (L, object, False),
    (L, tx.Union[int, str], False),
    (L, L, True),
    (L, tx.Any, True),
    # `Annotated` is still transparent.
    (tx.Annotated[L[1], "meta"], int, True),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    LITERAL_HINT_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in LITERAL_HINT_CASES],
)
def test_issubhint_literal_as_hint(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_a_nan_literal_is_a_subhint_of_itself() -> None:
    # `eq_safenan` keeps a NaN literal equal to itself even though
    # `nan == nan` is False, so the type-aware compare must still hold.
    nan = float("nan")
    assert issubhint(L[nan], L[nan]) is True


def test_literal_hint_against_typevar_uses_the_typevar_branch() -> None:
    # `issubhint` checks for a typevar super-hint before it checks for a
    # `Literal` hint, so a `Literal` against a typevar is resolved through
    # the typevar's bound/constraints rather than through the values check
    # above.
    bound = tx.TypeVar("bound", bound=int)
    assert issubhint(L[1], bound) is True
    assert issubhint(L[1], tx.TypeVar("unbound")) is True
    constrained = tx.TypeVar("constrained", int, str)
    assert issubhint(L[1], constrained) is True
