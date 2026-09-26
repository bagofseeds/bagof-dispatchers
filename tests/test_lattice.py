"""Tests for the dispatch-internal lattice layer (`_lattice.py`)."""

# stdlib
import collections.abc
import sys
import typing
import warnings

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._lattice import (
    equivalent,
    is_value_dependent,
    mro_index,
    solve_typevar,
    typevar_consistent,
)
from bagof.dispatchers.core import UNSET, UnknownHintWarning, issubhint

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
def _no_unknown_hint_warnings() -> tx.Iterator[None]:
    """Fail if the corpus trips an ``UnknownHintWarning``.

    An unrecognised hint is treated as ``Any`` by the relation, which would
    make the preorder and equivalence laws trivially true. Turning only that
    warning into an error keeps a future corpus addition honest, while any
    other warning is left to behave as usual.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownHintWarning)
        yield


# --- equivalence laws --------------------------------------------------


def test_equivalence_is_reflexive() -> None:
    for hint in CORPUS:
        assert equivalent(hint, hint) is True, hint


def test_equivalence_is_symmetric() -> None:
    for a in CORPUS:
        for b in CORPUS:
            assert equivalent(a, b) == equivalent(b, a), (a, b)


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


class _Seq(collections.abc.Sequence):
    def __getitem__(self, index: int) -> int:
        return 0

    def __len__(self) -> int:
        return 0


def test_mro_index_of_a_bare_abc_alias_is_spelling_independent() -> None:
    # A bare `Sequence` alias refines to its origin,
    # `collections.abc.Sequence`, whichever spelling names it -- so all three
    # agree on the MRO position.
    nominal = mro_index(collections.abc.Sequence, _Seq)
    assert nominal is not None and nominal > 0
    assert mro_index(typing.Sequence, _Seq) == nominal
    assert mro_index(tx.Sequence, _Seq) == nominal


class _MyList(list):
    pass


def test_mro_index_of_a_bare_list_alias_is_spelling_independent() -> None:
    # `List.__mro__` position matches the bare `list`'s.
    assert mro_index(list, _MyList) == 1
    assert mro_index(typing.List, _MyList) == 1
    assert mro_index(tx.List, _MyList) == 1


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


def test_constrained_typevar_solution_is_order_independent() -> None:
    # `(bool, str)` is under `object` but not both under `int`, so the answer
    # is `object` -- and it must not depend on which order the constraints
    # were declared in (mypy's rule: the first constraint all classes share).
    t_io = tx.TypeVar("_TIO", int, object)
    t_oi = tx.TypeVar("_TOI", object, int)
    assert solve_typevar((bool, str), t_io) is object
    assert solve_typevar((bool, str), t_oi) is object


def test_no_positions_stand_for_the_full_upper_bound() -> None:
    assert solve_typevar((), _T) is tx.Any
    assert solve_typevar((), _TBOUND) is int
    assert solve_typevar((), _TCONSTR) == tx.Union[int, str]


# --- value dependence --------------------------------------------------


_TV_LITERAL_BOUND = tx.TypeVar("_TV_LITERAL_BOUND", bound=tx.Literal[1, 2])
_TV_LITERAL_CONSTR = tx.TypeVar("_TV_LITERAL_CONSTR", tx.Literal[1], str)


@pytest.mark.parametrize(
    "hint",
    [
        tx.Literal[1],
        tx.Literal["a", "b"],
        tx.Type[int],
        # A union descends into its members ...
        tx.Optional[tx.Literal["a"]],
        tx.Union[tx.Literal[1], str],
        tx.Union[tx.Type[int], tx.Type[str]],
        # ... and a TypeVar into its upper bound.
        _TV_LITERAL_BOUND,
        _TV_LITERAL_CONSTR,
    ],
)
def test_value_dependent_hints(hint: tx.Any) -> None:
    assert is_value_dependent(hint) is True


def test_a_concrete_typeddict_is_value_dependent() -> None:
    # A concrete TypedDict dispatches on the mapping's shape, so the call
    # cache must carry the value at a TypedDict-typed argument.
    class Movie(tx.TypedDict):
        title: str
        year: int

    assert is_value_dependent(Movie) is True


def test_the_bare_typeddict_marker_is_type_dependent() -> None:
    # The bare marker names no fields, so its value-level check is type-only
    # (a plain dict is not a TypedDict) and the value adds nothing.
    assert is_value_dependent(tx.TypedDict) is False


def test_a_parametrised_generic_typeddict_is_value_dependent() -> None:
    # A generic `TypedDict` subscripted with a type argument (`GTD[int]`) is a
    # typing alias, not a `TypedDict` class -- so the classifier must read its
    # origin, not the alias, to see the shape it dispatches on. Missing this
    # keys the argument by type only while dispatch reads the shape.
    T = tx.TypeVar("T")

    class GTD(tx.TypedDict, typing.Generic[T]):
        a: T

    assert is_value_dependent(GTD[int]) is True
    # And through a union, which descends into its members.
    assert is_value_dependent(tx.Optional[GTD[int]]) is True


@pytest.mark.parametrize(
    "hint",
    [
        tx.Tuple[tx.Literal[1]],  # container items are never inspected
        tx.List[tx.Literal[1]],
    ],
)
def test_nested_container_args_are_not_value_dependent(hint: tx.Any) -> None:
    assert is_value_dependent(hint) is False


@pytest.mark.skipif(
    sys.version_info >= (3, 9),
    reason="typing.Literal is a distinct object from tx.Literal only < 3.9",
)
def test_the_typing_literal_spelling_is_value_dependent() -> None:
    # Before 3.9, `typing.Literal` and `typing_extensions.Literal` are two
    # distinct objects; the classifier must recognise both spellings.
    assert is_value_dependent(typing.Literal[1]) is True


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
