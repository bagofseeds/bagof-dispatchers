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

# `Exact[C]` is a *leaf* subtype of `C`: nothing ordinary is below it, so
# neither `C` nor a subclass of `C` is `<= Exact[C]`.

# (hint, superhint, expected)
EXACT_SUPER_CASES = [
    # `C` is NOT `<= Exact[C]`: an ordinary `int` is not exactly an int.
    (int, Exact[int], False),
    (bool, Exact[int], False),
    (tx.Annotated[int, "meta"], Exact[int], False),
    (str, Exact[int], False),
    # `Exact[D] <= Exact[C]` iff `D` and `C` are the same type.
    (Exact[int], Exact[int], True),
    (Exact[bool], Exact[int], False),
    (Exact[int], Exact[bool], False),
    # A `Literal` is below `Exact[C]` iff every value has type exactly `C`.
    (tx.Literal[1], Exact[int], True),
    (tx.Literal[1, 2], Exact[int], True),
    (tx.Literal[True], Exact[int], False),  # its type is bool, not int
    (tx.Literal[True], Exact[bool], True),
    (tx.Literal[1], Exact[bool], False),
    (tx.Literal["a"], Exact[int], False),
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


def test_a_bound_typevar_is_not_below_exact() -> None:
    # A bound typevar stands for `int` *and its subclasses*, so it is not a
    # leaf below `Exact[int]` -- only an exactly-int value is.
    bound = tx.TypeVar("bound", bound=int)
    assert issubhint(bound, Exact[int]) is False
    # But `Exact` as the sub-hint still passes through to the bound.
    assert issubhint(Exact[int], bound) is True


# --- the relation stays a preorder with Exact present ------------------

# A corpus that mixes ordinary classes, `Exact` and `Literal`. Bare
# (unparametrised) `Union`/`Literal` are deliberately excluded: the relation
# is documented as non-transitive through those.
# `Literal[1]` and `Literal[True]` are deliberately not both present: Python
# compares `1 == True`, a value-level quirk of literal matching that is
# orthogonal to `Exact` and would otherwise break transitivity here.
_CORPUS = [
    object,
    int,
    bool,
    str,
    Exact[int],
    Exact[bool],
    Exact[str],
    tx.Literal[1],
    tx.Literal["x"],
]


def test_exact_relation_is_reflexive() -> None:
    for hint in _CORPUS:
        assert issubhint(hint, hint) is True, hint


def test_exact_relation_is_transitive() -> None:
    for a in _CORPUS:
        for b in _CORPUS:
            if not issubhint(a, b):
                continue
            for c in _CORPUS:
                if issubhint(b, c):
                    assert issubhint(a, c) is True, (a, b, c)


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


def test_ishintstance_exact_through_a_typevar_bound() -> None:
    # Exactness reached through a bound must be honoured at the value level,
    # just as it is at the hint level -- `True` is a `bool`, not an exact int.
    T = tx.TypeVar("T", bound=Exact[int])
    assert ishintstance(1, T) is True
    assert ishintstance(True, T) is False
    assert issubhint(bool, T) is False
