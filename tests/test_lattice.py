"""Tests for the dispatch-internal lattice layer (`_lattice.py`)."""

# stdlib
import warnings

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._lattice import (
    equivalent,
    is_instance,
    is_value_dependent,
    mro_index,
    solve_typevar,
    typevar_consistent,
)
from bagof.dispatchers.core import UNSET, issubhint

# --- a shared corpus for the preorder / equivalence laws ---------------


class _Animal:
    pass


class _Dog(_Animal):
    pass


@tx.runtime_checkable
class _Sized(tx.Protocol):
    def __len__(self) -> int: ...


_T = tx.TypeVar("_T")
_TBOUND = tx.TypeVar("_TBOUND", bound=int)
_TCONSTR = tx.TypeVar("_TCONSTR", int, str)

# ~40 hints spanning classes, ABCs, unions, optionals, literals, the tuple /
# list / dict families, `Callable` pairs, `TypeVar`s and `Exact`. It
# deliberately leaves out the two documented quirks so the laws stay clean:
# no `Literal[True]` beside `Literal[1]` (issue #6), and no `TypeVar` bounded
# by `Exact[...]` used as a sub-hint (issue #7).
CORPUS = [
    # plain classes and the diamond of ABCs / nominal bases
    object,
    int,
    bool,
    str,
    bytes,
    float,
    _Animal,
    _Dog,
    # ABCs
    tx.Sequence,
    tx.Iterable,
    tx.Mapping,
    # unions and optionals
    tx.Union[int, str],
    tx.Union[bool, int],
    tx.Union[int, str, bytes],
    tx.Optional[int],
    tx.Union[tx.List[int], tx.List[str]],
    # literals (no bool/int collision -- issue #6)
    tx.Literal[1],
    tx.Literal[1, 2],
    tx.Literal["a"],
    tx.Literal["a", "b"],
    # the tuple / list / dict families
    tx.List[int],
    tx.List[bool],
    tx.List[object],
    list,
    tx.Dict[str, int],
    tx.Dict[str, bool],
    dict,
    tx.Tuple[int],
    tx.Tuple[int, str],
    tx.Tuple[int, ...],
    tuple,
    # Callable pairs (contravariant params, covariant return). The `...`
    # wildcard list is deliberately left out: like `Any`, it is
    # consistent-with every list without being a true top or bottom, so it
    # is not part of this strict preorder.
    tx.Callable[[int], str],
    tx.Callable[[bool], str],
    tx.Callable[[int], bool],
    # TypeVars
    _T,
    _TBOUND,
    _TCONSTR,
    # Exact leaves
    Exact[int],
    Exact[str],
    # Any / None
    tx.Any,
    type(None),
]


@pytest.fixture(autouse=True)
def _quiet_unknown_hint_warnings() -> tx.Iterator[None]:
    """The relation warns once per unknown form; the laws do not care."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --- equivalence laws --------------------------------------------------


def test_equivalence_is_reflexive() -> None:
    for hint in CORPUS:
        assert equivalent(hint, hint) is True, hint


def test_equivalence_is_symmetric() -> None:
    for a in CORPUS:
        for b in CORPUS:
            assert equivalent(a, b) == equivalent(b, a), (a, b)


def test_equivalence_matches_its_definition() -> None:
    # `a ≡ b` iff each is a sub-hint of the other.
    for a in CORPUS:
        for b in CORPUS:
            expected = issubhint(a, b) and issubhint(b, a)
            assert equivalent(a, b) is expected, (a, b)


def test_equivalence_is_transitive() -> None:
    for a in CORPUS:
        for b in CORPUS:
            if not equivalent(a, b):
                continue
            for c in CORPUS:
                if equivalent(b, c):
                    assert equivalent(a, c) is True, (a, b, c)


# --- the order is a preorder -------------------------------------------


def test_order_is_reflexive() -> None:
    for hint in CORPUS:
        assert issubhint(hint, hint) is True, hint


def test_order_is_transitive() -> None:
    for a in CORPUS:
        for b in CORPUS:
            if not issubhint(a, b):
                continue
            for c in CORPUS:
                if issubhint(b, c):
                    assert issubhint(a, c) is True, (a, b, c)


# --- is_instance delegates to the Exact-aware value check --------------


@pytest.mark.parametrize(
    "value,hint,expected",
    [
        (1, int, True),
        (True, int, True),
        (True, Exact[int], False),
        (1, Exact[int], True),
        (1, tx.Literal[1, 2], True),
        (3, tx.Literal[1, 2], False),
        ([1], tx.List[str], True),  # items are not inspected
        ("x", tx.Union[int, str], True),
        (1, tx.Union[list, str], False),
    ],
)
def test_is_instance(value: tx.Any, hint: tx.Any, expected: bool) -> None:
    assert is_instance(value, hint) is expected


# --- mro_index ---------------------------------------------------------


class _B:
    pass


class _C:
    pass


class _D(_B, _C):
    pass


def test_mro_index_orders_a_diamond() -> None:
    # `class D(B, C)` -> `D.__mro__` is `(D, B, C, object)`, so B precedes C.
    assert mro_index(_B, _D) == 1
    assert mro_index(_C, _D) == 2
    assert mro_index(_B, _D) < mro_index(_C, _D)


def test_mro_index_of_the_value_type_itself_is_zero() -> None:
    assert mro_index(_D, _D) == 0


def test_mro_index_treats_exact_as_its_target() -> None:
    assert mro_index(Exact[_B], _D) == mro_index(_B, _D)


@pytest.mark.parametrize(
    "hint",
    [
        _Sized,  # a protocol satisfied structurally, not in the MRO
        tx.Union[_B, _C],  # a union
        tx.Literal[1],  # a literal
        tx.List[int],  # a parametrised generic
        tx.Type[_B],  # type[...]
        _T,  # a TypeVar
    ],
)
def test_mro_index_gives_no_refinement(hint: tx.Any) -> None:
    assert mro_index(hint, _D) is None


def test_mro_index_of_a_class_absent_from_the_mro() -> None:
    assert mro_index(str, _D) is None


# --- repeated TypeVar solving ------------------------------------------


def test_greatest_element_solves_to_the_maximum() -> None:
    assert solve_typevar((int, bool), _T) is int
    assert typevar_consistent((int, bool), _T) is True
    # Order does not matter: the greatest element is found either way.
    assert solve_typevar((bool, int), _T) is int


def test_incomparable_classes_have_no_greatest_element() -> None:
    assert solve_typevar((int, str), _T) is UNSET
    assert typevar_consistent((int, str), _T) is False


def test_a_single_class_always_solves() -> None:
    assert solve_typevar((str,), _T) is str
    assert typevar_consistent((str,), _T) is True


def test_a_bound_typevar_uses_the_greatest_element() -> None:
    assert solve_typevar((int, bool), _TBOUND) is int


def test_constrained_typevar_same_constraint_is_consistent() -> None:
    # `bool` solves the constraint set as `int`, so `(int, bool)` agrees.
    assert solve_typevar((int, bool), _TCONSTR) is int
    assert typevar_consistent((int, bool), _TCONSTR) is True


def test_constrained_typevar_different_constraints_is_inconsistent() -> None:
    assert solve_typevar((int, str), _TCONSTR) is UNSET
    assert typevar_consistent((int, str), _TCONSTR) is False


def test_a_class_matching_no_constraint_is_inconsistent() -> None:
    assert solve_typevar((bytes,), _TCONSTR) is UNSET


def test_no_positions_stand_for_the_full_upper_bound() -> None:
    assert solve_typevar((), _T) is tx.Any
    assert solve_typevar((), _TBOUND) is int
    assert solve_typevar((), _TCONSTR) == tx.Union[int, str]


# --- value dependence --------------------------------------------------


@pytest.mark.parametrize(
    "hint",
    [
        tx.Literal[1],
        tx.Literal["a", "b"],
        tx.Type[int],
    ],
)
def test_value_dependent_hints(hint: tx.Any) -> None:
    assert is_value_dependent(hint) is True


def test_a_typeddict_is_value_dependent() -> None:
    class Movie(tx.TypedDict):
        title: str
        year: int

    assert is_value_dependent(Movie) is True


@pytest.mark.parametrize(
    "hint",
    [
        int,
        tx.List[int],
        tx.Union[int, str],
        Exact[int],
        type,  # a bare `type` is decided by the type of the value alone
        object,
    ],
)
def test_type_dependent_hints(hint: tx.Any) -> None:
    assert is_value_dependent(hint) is False
