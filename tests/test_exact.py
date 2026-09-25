"""Tests for `Exact[C]` in the relation."""

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers import Exact
from bagof.dispatchers.core import ishintstance, issubhint
from bagof.dispatchers.core._exact import EXACT, exact_target, is_exact


def test_exact_spelling_and_detection() -> None:
    assert Exact[int] == tx.Annotated[int, EXACT]
    assert is_exact(Exact[int]) is True
    assert is_exact(int) is False
    assert is_exact(tx.Annotated[int, "meta"]) is False
    assert exact_target(Exact[int]) is int


# --- issubhint against an Exact super-hint ------------------------------

# (hint, superhint, expected)
EXACT_SUPER_CASES = [
    # `q <= Exact[C]` iff `q` is equivalent to `C`.
    (int, Exact[int], True),
    (bool, Exact[int], False),
    (tx.Annotated[int, "meta"], Exact[int], True),
    (str, Exact[int], False),
    # `Exact[D] <= Exact[C]` iff `D` and `C` are the same type.
    (Exact[int], Exact[int], True),
    (Exact[bool], Exact[int], False),
    (Exact[int], Exact[bool], False),
]


@pytest.mark.parametrize("hint,superhint,expected", EXACT_SUPER_CASES)
def test_issubhint_exact_superhint(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


# --- Exact as the sub-hint: `Exact[C] < C` -----------------------------

# (hint, superhint, expected)
EXACT_SUB_CASES = [
    (Exact[int], int, True),
    (Exact[int], object, True),
    (Exact[bool], int, True),
    (Exact[int], bool, False),
    (Exact[int], str, False),
    (Exact[int], tx.Union[int, str], True),
    (Exact[int], tx.Any, True),
]


@pytest.mark.parametrize("hint,superhint,expected", EXACT_SUB_CASES)
def test_issubhint_exact_subhint(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_exact_of_a_typevar_equivalent_type() -> None:
    # A bound typevar is equivalent to its bound, so it matches `Exact` of
    # that bound.
    bound = tx.TypeVar("bound", bound=int)
    assert issubhint(bound, Exact[int]) is True


# --- ishintstance with Exact -------------------------------------------


@pytest.mark.parametrize(
    "obj,hint,expected",
    [
        (1, Exact[int], True),
        (True, Exact[int], False),  # a bool is not exactly an int
        (True, Exact[bool], True),
        ("a", Exact[int], False),
        (None, Exact[None], True),
        (None, Exact[type(None)], True),
        (1, Exact[type(None)], False),
        # `Annotated` metadata around the value's own type is transparent.
        (1, tx.Annotated[Exact[int], "meta"], True),
    ],
)
def test_ishintstance_exact(
    obj: tx.Any, hint: tx.Any, expected: bool
) -> None:
    assert ishintstance(obj, hint) is expected
