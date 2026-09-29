"""Tests for the dispatch-internal lattice layer (`_lattice.py`)."""

# stdlib
import collections.abc
import dataclasses
import numbers
import sys
import typing
import warnings

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import Between, Exact, Hint, Super
from bagof.dispatchers._lattice import (
    equivalent,
    is_value_dependent,
    overlaps,
    solve_typevar,
    typevar_consistent,
)
from bagof.dispatchers.core import (
    UNSET,
    UnknownHintWarning,
    issubhint,
    mro_index,
)

# --- a shared corpus for the preorder / equivalence laws ---------------


class _Animal:
    pass


class _Dog(_Animal):
    pass


class _Puppy(_Dog):
    pass


@tx.runtime_checkable
class _Sized(tx.Protocol):
    def __len__(self) -> int: ...


_T = tx.TypeVar("_T")
_TBOUND = tx.TypeVar("_TBOUND", bound=int)
_TBOUND_OBJ = tx.TypeVar("_TBOUND_OBJ", bound=object)
_TCONSTR = tx.TypeVar("_TCONSTR", int, str)
_P = tx.ParamSpec("_P")
_Ts = tx.TypeVarTuple("_Ts")
_T_co = tx.TypeVar("_T_co", covariant=True)
_T_contra = tx.TypeVar("_T_contra", contravariant=True)


class _Arr(tx.Generic[tx.Unpack[_Ts]]):
    """A user class parametrised by a `TypeVarTuple`, for the corpus laws."""


class _Src(tx.Generic[_T_co]):
    """A covariant user generic, for the variance corpus (#50)."""


class _Snk(tx.Generic[_T_contra]):
    """A contravariant user generic, for the variance corpus (#50)."""


class _Box(tx.Generic[_T]):
    """An invariant (unflagged) user generic, for the variance corpus (#50)."""


class _Cell(tx.Generic[_T]):
    """A second invariant user generic, for the variance corpus (#50)."""


_A = tx.TypeVar("_A")
_B_ = tx.TypeVar("_B_")


class _IntBox(_Box[int]):
    """A class declaring `_Box[int]` through its base (#50, V5)."""


class _Sub(_Box[_T]):
    """A generic passing its argument on to `_Box` (#50, V5)."""


class _Pair(tx.Generic[_A, _B_]):
    """An invariant two-parameter user generic (#50, V5)."""


class _Flip(_Pair[_B_, _A], tx.Generic[_A, _B_]):
    """`_Pair` with its parameters swapped: `_Flip[X, Y]` is `_Pair[Y, X]`."""


class _Child(tx.List[int]):
    """A list subclass declaring `List[int]` through its base (#50, V5)."""


class _Two(tx.List[_A], tx.Container[_B_]):
    """Reaches `Container` through both bases (#64): `_Two[int, str]` is a
    `Container[int]` through `List`, and a `Container[str]` through its own
    base. A type checker rejects it; the relation accepts either."""


class _Mid(_Box[_T]):
    """Passes its argument on to `_Box`, for the diamond below (#64)."""


class _MidInt(_Mid[int]):
    """`_Mid[int]`, so `_Box[int]`."""


class _MidStr(_Mid[str]):
    """`_Mid[str]`, so `_Box[str]`."""


class _MidBoth(_MidInt, _MidStr):
    """Reaches `_Mid` twice, with other arguments: below both `_Box`es."""


def _pep585_rows() -> tx.List[tx.Any]:
    """The `_Two` rows again, written against `list[...]` (3.9+ only)."""
    if sys.version_info < (3, 9):
        return []

    class _Two585(list[_A], tx.Container[_B_]):  # type: ignore[misc]
        """`_Two`, with its `list` base spelled as a PEP 585 alias."""

    return [_Two585[int, str], _Two585[bool, str]]


class _Movie(tx.TypedDict):
    """A concrete TypedDict, for the `TypedDict <= dict` corpus rows (#19)."""

    title: str
    year: int


@tx.runtime_checkable
class _HasName(tx.Protocol):
    """A runtime protocol with a data member, for the #56 corpus rows."""

    name: str


@tx.runtime_checkable
class _HasNameAge(_HasName, tx.Protocol):
    """A sub-protocol of `_HasName`, adding a second data member."""

    age: int


class _Named:
    """Declares `_HasName`'s member itself, as a class attribute."""

    name = "named"


@dataclasses.dataclass
class _NamedAged:
    """Declares both of `_HasNameAge`'s members, as dataclass fields."""

    name: str
    age: int


class _Unnamed:
    """Annotates `name` but never sets it: declares it, as a type checker
    reads it."""

    name: str

# ~40 hints spanning classes, ABCs, unions, optionals, literals, the tuple /
# list / dict families, `Callable` pairs, `TypeVar`s and `Exact`. It
# deliberately leaves out one documented quirk so the laws stay clean: no
# `TypeVar` bounded by `Exact[...]` used as a sub-hint (issue #7). The
# `Literal[True]`/`Literal[1]` overlap is included now that the value
# comparison is type-aware.
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
    # literals, including the bool/int value overlap: a type-aware compare
    # keeps `Literal[True]` and `Literal[1]` distinct, so the laws hold.
    tx.Literal[1],
    tx.Literal[1, 2],
    tx.Literal[True],
    tx.Literal[False],
    tx.Literal[True, False],
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
    # variance families (#50): a covariant (`_Src`), contravariant (`_Snk`)
    # and invariant (`_Box`/`_Cell`) user generic, and stdlib containers of
    # each variance, over subtype-related arguments and the per-family ends --
    # a free `T`, `Any`, and the bottom `Never`. Invariance is equality and
    # contravariance is reversal, both of which preserve the preorder, so the
    # reflexive / transitive laws must still hold over all of these.
    _Src[int],
    _Src[bool],
    _Src[_T],
    _Src[tx.Any],
    _Src[tx.Never],
    _Snk[int],
    _Snk[bool],
    _Snk[_T],
    _Snk[tx.Any],
    _Snk[tx.Never],
    _Box[int],
    _Box[bool],
    _Box[_T],
    _Box[tx.Any],
    _Box[tx.Never],
    # bounded / constrained `TypeVar` slots (#50, V5): solved on the super
    # side at an invariant slot, read by the bound at a covariant or
    # contravariant one.
    _Box[_TBOUND],
    _Box[_TBOUND_OBJ],
    _Box[_TCONSTR],
    _Box[tx.Union[int, str]],
    _Box[str],
    _Src[_TBOUND],
    _Snk[_TBOUND],
    _Cell[int],
    tx.Sequence[int],
    tx.Sequence[bool],
    tx.Mapping[str, int],
    tx.Mapping[str, bool],
    # declared parametrisations (#50, V5): classes and parametrisations whose
    # origin differs from the super-hint's, read through their bases --
    # `_IntBox` is `_Box[int]`, `_Sub[bool]` is `_Box[bool]`, `_Flip[int, str]`
    # is `_Pair[str, int]`, and `_Child` is `List[int]` (so a
    # `Sequence[int]`, but not a `List[bool]`).
    _IntBox,
    _Sub[bool],
    _Sub[int],
    _Pair[str, int],
    _Pair[int, str],
    _Flip[int, str],
    _Child,
    # multi-base walks (#64): a class reaching `Container` through two bases
    # with other arguments, and a diamond reaching `_Mid` twice. Each is below
    # every parametrisation it reaches, so `_MidBoth <= _MidStr <= _Mid[str]
    # <= _Box[str]` must give `_MidBoth <= _Box[str]` for transitivity.
    _Two[int, str],
    _Two[bool, str],
    tx.Container[int],
    tx.Container[bool],
    tx.Container[str],
    tx.Container[object],
    _Mid[int],
    _Mid[str],
    _MidInt,
    _MidStr,
    _MidBoth,
    *_pep585_rows(),
    # TypedDict rows (#19): the bare marker ("any TypedDict") and a concrete
    # TypedDict both sit strictly below `dict` -- every TypedDict value is a
    # dict, but a plain `dict` is neither. The marker is reduced to `dict` only
    # against a plain-class super-hint, so it must still be ordered correctly
    # when merely *contained* in one (a union member, an `Annotated` wrapper)
    # or written in the other spelling -- these rows are that regression.
    tx.TypedDict,
    typing.TypedDict,
    tx.Optional[tx.TypedDict],
    tx.Annotated[tx.TypedDict, "m"],
    _Movie,
    # runtime protocols with data members (#56): a protocol, a sub-protocol,
    # a class declaring the member, a dataclass declaring both, and a class
    # that only annotates it (so is below the protocol, not the
    # sub-protocol).
    _HasName,
    _HasNameAge,
    _Named,
    _NamedAged,
    _Unnamed,
    tx.Optional[_HasName],
    tx.Tuple[int],
    tx.Tuple[int, str],
    tx.Tuple[int, ...],
    tx.Tuple[()],
    tuple,
    # variadic tuples (PEP 646): a `*Ts` run, its prefix/suffix variants, and
    # `Tuple[Any, ...]` which is equivalent to `Tuple[*Ts]`. The preorder /
    # equivalence laws over these are the transitivity regression for the
    # tuple-shape matcher.
    tx.Tuple[int, tx.Unpack[_Ts]],
    tx.Tuple[tx.Unpack[_Ts], int],
    tx.Tuple[tx.Unpack[_Ts]],
    tx.Tuple[int, tx.Unpack[_Ts], str],
    tx.Tuple[tx.Any, ...],
    # a user `Generic[*Ts]` pair, ordered through the same tuple-shape path
    _Arr[int, str],
    _Arr[int, tx.Unpack[_Ts]],
    # Callable pairs (contravariant params, covariant return). `...` and a
    # bare `ParamSpec` are the top of parameter lists, and a `Concatenate`
    # prefix sits between the fixed lists and that top -- all part of the
    # preorder now that the relation is transitive through them (the row-flip
    # in RFC 11.1; issue #32).
    tx.Callable[[int], str],
    tx.Callable[[bool], str],
    tx.Callable[[int], bool],
    tx.Callable[..., str],
    tx.Callable[_P, str],
    tx.Callable[tx.Concatenate[int, _P], str],
    tx.Callable[tx.Concatenate[bool, _P], str],
    tx.Callable[tx.Concatenate[int, str, _P], str],
    tx.Callable[[int, tx.Unpack[_Ts]], str],
    # TypeVars
    _T,
    _TBOUND,
    _TCONSTR,
    # Exact leaves
    Exact[int],
    Exact[bool],
    Exact[str],
    # Type with an Exact argument, an identity leaf within the Type position.
    tx.Type[Exact[int]],
    # Hint forms, whose values are hints: ordered only among themselves, and a
    # `Hint[Exact[C]]` is a leaf under `Hint[C]` just as `Exact[C]` is under
    # `C`. Below only `Any`, above nothing ordinary.
    Hint[int],
    Hint[bool],
    Hint[tx.Any],
    Hint[Exact[int]],
    # Lower bounds inside `Type` and `Hint` (0.3.0): ordered contravariantly
    # by their bound, above the matching `Exact` form, and apart from a plain
    # argument.
    tx.Type[Super[int]],
    tx.Type[Super[bool]],
    Hint[Super[int]],
    Hint[Super[bool]],
    # Intervals inside `Type` and `Hint` (0.3.0): ordered by inclusion.
    tx.Type[Between[_Dog, _Animal]],
    tx.Type[Between[_Puppy, _Dog]],
    Hint[Between[bool, numbers.Integral]],
    # Any / None, and the bottom on its own -- below every hint above,
    # `Exact[C]` included (#54)
    tx.Any,
    type(None),
    tx.Never,
    tx.NoReturn,
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


# --- overlap between unordered Type and Hint forms (0.3.0) ------------


def _lts(c: tx.Any) -> tx.Any:
    return tx.Type[Super[c]]


def _lte(c: tx.Any) -> tx.Any:
    return tx.Type[Exact[c]]


def _lhs(c: tx.Any) -> tx.Any:
    return Hint[Super[c]]


def _ltb(lower: tx.Any, upper: tx.Any) -> tx.Any:
    return tx.Type[Between[lower, upper]]


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (tx.Type[_Animal], _lts(_Dog), True),
        (tx.Type[_Puppy], _lts(_Dog), False),
        (_lts(_Dog), _lts(int), True),  # `object` is above both
        (Hint[int], _lhs(bool), True),
        (Hint[str], _lhs(bool), False),
        (int, _lts(int), False),
        (tx.Type[_Animal], Hint[Super[_Dog]], False),
        (tx.Type[_Animal], tx.Type[_Dog], False),  # no lower bound involved
        (_lte(_Dog), _lts(_Dog), False),  # the order decides an `Exact` pair
        (tx.Type[_Animal], _lts(tx.Any), False),  # an empty interval
        # The only shared end is a union, which no class value can be.
        (tx.Type[tx.Union[_Dog, int]], _lts(tx.Union[_Dog, int]), False),
        (tx.Annotated[tx.Type[_Animal], "m"], _lts(_Dog), True),
        (tx.Type, _lts(_Dog), True),  # ordered too, and `_Dog` is in both
        # A union is read through its members, and a `TypeVar` as its bound.
        (tx.Type[_Animal], tx.Optional[_lts(_Dog)], True),
        (tx.Type[_Animal], tx.TypeVar("_TVS", bound=_lts(_Dog)), True),
        (tx.Type[_Puppy], tx.Optional[_lts(_Dog)], False),
        # An interval overlaps wherever one of its ends lies in the other.
        (_ltb(_Dog, _Animal), tx.Type[_Dog], True),
        (_ltb(_Dog, _Animal), tx.Type[_Puppy], False),
        (_ltb(_Dog, _Animal), _lts(_Animal), True),
        (Hint[Between[bool, int]], _lhs(numbers.Integral), False),
    ],
    ids=repr,
)
def test_overlaps(a: tx.Any, b: tx.Any, expected: bool) -> None:
    assert overlaps(a, b) is expected
    assert overlaps(b, a) is expected


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


def test_the_bottom_is_below_every_hint_and_only_a_bottom_below_it() -> None:
    bottoms = (tx.Never, tx.NoReturn)
    for hint in CORPUS:
        assert issubhint(tx.Never, hint) is True, hint
        assert issubhint(hint, tx.Never) is (hint in bottoms), hint


def test_the_data_protocol_rows_are_ordered() -> None:
    """The #56 rows are not trivially incomparable: the laws above bite."""
    assert issubhint(_HasNameAge, _HasName) is True
    assert issubhint(_HasName, _HasNameAge) is False
    assert issubhint(_Named, _HasName) is True
    assert issubhint(_Named, _HasNameAge) is False
    assert issubhint(_NamedAged, _HasNameAge) is True
    assert issubhint(_NamedAged, _HasName) is True
    assert issubhint(_Unnamed, _HasName) is True
    assert issubhint(_HasName, _Unnamed) is False
    assert issubhint(_Unnamed, _HasNameAge) is False
    assert issubhint(_Unnamed, tx.Optional[_HasName]) is True
    assert issubhint(_Named, tx.Optional[_HasName]) is True


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
        # A `Callable` is matched shallowly at the value level -- its value's
        # own signature is never inspected -- so no `Callable` form is
        # value-dependent, `ParamSpec` / `Concatenate` lists included. Pinning
        # this keeps a future value-level `P` solve from silently changing the
        # cache key.
        tx.Callable[[int], str],
        tx.Callable[..., str],
        tx.Callable[_P, str],
        tx.Callable[tx.Concatenate[int, _P], str],
        # A variadic tuple / callable dispatches on the value's *type* alone
        # (its items are never inspected), so none is value-dependent.
        tx.Tuple[int, tx.Unpack[_Ts]],
        tx.Tuple[tx.Unpack[_Ts]],
        tx.Unpack[_Ts],
        tx.Callable[[int, tx.Unpack[_Ts]], str],
    ],
)
def test_type_dependent_hints(hint: tx.Any) -> None:
    assert is_value_dependent(hint) is False


# --- repeated ParamSpec solving ----------------------------------------


def _shape(*prefix: tx.Any) -> tx.Any:
    """A closed parameter-list shape from a fixed prefix, for the solver."""
    from bagof.dispatchers.core._relation import _ParamShape

    return _ParamShape(tuple(prefix), None)


def _open_shape(*prefix: tx.Any) -> tx.Any:
    """An open (`...`-tailed) parameter-list shape, for the solver."""
    from bagof.dispatchers.core._relation import _ParamShape

    return _ParamShape(tuple(prefix), Ellipsis)


def test_solve_paramspec_greatest_element() -> None:
    from bagof.dispatchers._lattice import (
        paramspec_consistent,
        solve_paramspec,
    )

    # `([int])` and `([bool])` -> `([bool])` (contravariant: int <: bool as a
    # parameter, so the bool-list is the greater).
    assert solve_paramspec([_shape(int), _shape(bool)]) == _shape(bool)
    assert paramspec_consistent([_shape(int), _shape(bool)]) is True
    # Order does not matter.
    assert solve_paramspec([_shape(bool), _shape(int)]) == _shape(bool)


def test_solve_paramspec_incomparable_is_unset() -> None:
    from bagof.dispatchers._lattice import (
        paramspec_consistent,
        solve_paramspec,
    )

    assert solve_paramspec([_shape(int), _shape(str)]) is UNSET
    assert paramspec_consistent([_shape(int), _shape(str)]) is False


def test_solve_paramspec_closed_and_open_solve_to_open() -> None:
    from bagof.dispatchers._lattice import solve_paramspec

    # A closed list is a sub-hint of an open one with a matching prefix, so the
    # open list is the greatest element.
    assert solve_paramspec([_shape(int), _open_shape(int)]) == _open_shape(int)


def test_solve_paramspec_no_slots_is_the_open_top() -> None:
    from bagof.dispatchers._lattice import (
        paramspec_consistent,
        solve_paramspec,
    )

    assert solve_paramspec([]) == _open_shape()
    assert paramspec_consistent([]) is True


def test_solve_paramspec_single_tail_always_solves() -> None:
    from bagof.dispatchers._lattice import (
        paramspec_consistent,
        solve_paramspec,
    )

    assert solve_paramspec([_shape(int, str)]) == _shape(int, str)
    assert paramspec_consistent([_shape(int, str)]) is True


def test_paramspec_captures_reads_a_top_level_callable() -> None:
    from bagof.dispatchers._lattice import paramspec_captures

    P = tx.ParamSpec("P")
    # Query `Callable[[int, str], int]` at a slot `Callable[P, int]`: `P`
    # captures the whole list `([int, str])`.
    captures = dict(
        (id(p), tail)
        for p, tail in paramspec_captures(
            tx.Callable[[int, str], int], tx.Callable[P, int]
        )
    )
    assert captures[id(P)] == _shape(int, str)
    # A slot whose list does not end in a ParamSpec captures nothing.
    assert list(
        paramspec_captures(
            tx.Callable[[int], int], tx.Callable[[int], int]
        )
    ) == []
    # A nested (non-top-level) Callable is not read.
    assert list(
        paramspec_captures(
            tx.Optional[tx.Callable[[int], int]],
            tx.Optional[tx.Callable[P, int]],
        )
    ) == []
    # A bare `Callable` slot (no parameter list) captures nothing.
    assert list(
        paramspec_captures(tx.Callable[[int], int], tx.Callable)
    ) == []
    # A query whose list does not match the slot captures nothing (the empty
    # list is too short for the committed `int` prefix).
    assert list(
        paramspec_captures(
            tx.Callable[[], int], tx.Callable[tx.Concatenate[int, P], int]
        )
    ) == []


def test_paramspec_captures_with_a_concatenate_slot() -> None:
    from bagof.dispatchers._lattice import paramspec_captures

    P = tx.ParamSpec("P")
    # Slot `Callable[Concatenate[int, P], int]`, query `Callable[[int, str],
    # int]`: `P` captures the leftover `([str])`.
    captures = dict(
        (id(p), tail)
        for p, tail in paramspec_captures(
            tx.Callable[[int, str], int],
            tx.Callable[tx.Concatenate[int, P], int],
        )
    )
    assert captures[id(P)] == _shape(str)


# --- repeated TypeVarTuple solving -------------------------------------


def _run(*prefix: tx.Any) -> tx.Any:
    """A closed run shape (a fixed sequence of elements), for the solver."""
    from bagof.dispatchers.core._relation import _TupleShape

    return _TupleShape(tuple(prefix), None, (), None)


def _open_run(*prefix: tx.Any) -> tx.Any:
    """An open run shape (a `*Ts`-style run), for the solver."""
    from bagof.dispatchers.core._relation import _TupleShape

    return _TupleShape(tuple(prefix), tx.Any, (), None)


def test_solve_typevartuple_greatest_element() -> None:
    from bagof.dispatchers._lattice import (
        solve_typevartuple,
        typevartuple_consistent,
    )

    # `(int,)` and `(bool,)` -> `(int,)` (covariant: bool <: int, so the
    # int-run is the greater).
    assert solve_typevartuple([_run(int), _run(bool)]) == _run(int)
    assert typevartuple_consistent([_run(int), _run(bool)]) is True
    # Order does not matter.
    assert solve_typevartuple([_run(bool), _run(int)]) == _run(int)


def test_solve_typevartuple_incomparable_is_unset() -> None:
    from bagof.dispatchers._lattice import (
        solve_typevartuple,
        typevartuple_consistent,
    )

    assert solve_typevartuple([_run(int), _run(str)]) is UNSET
    assert typevartuple_consistent([_run(int), _run(str)]) is False


def test_solve_typevartuple_different_arities_is_unset() -> None:
    from bagof.dispatchers._lattice import solve_typevartuple

    assert solve_typevartuple([_run(int), _run(int, int)]) is UNSET


def test_solve_typevartuple_closed_and_open_solve_to_open() -> None:
    from bagof.dispatchers._lattice import solve_typevartuple

    # A closed run is a sub-run of an open one with a matching prefix.
    assert solve_typevartuple([_run(int), _open_run(int)]) == _open_run(int)


def test_solve_typevartuple_no_slots_is_the_open_run() -> None:
    from bagof.dispatchers._lattice import (
        solve_typevartuple,
        typevartuple_consistent,
    )

    assert solve_typevartuple([]) == _open_run()
    assert typevartuple_consistent([]) is True


def test_solve_typevartuple_single_run_always_solves() -> None:
    from bagof.dispatchers._lattice import solve_typevartuple

    assert solve_typevartuple([_run(int, str)]) == _run(int, str)


def test_typevartuple_captures_reads_a_top_level_tuple() -> None:
    from bagof.dispatchers._lattice import typevartuple_captures

    Ts = tx.TypeVarTuple("Ts")
    # Query `Tuple[int, str, bytes]` at slot `Tuple[int, *Ts]`: `Ts` captures
    # the leftover run `(str, bytes)`.
    captures = dict(
        (id(t), run)
        for t, run in typevartuple_captures(
            tx.Tuple[int, str, bytes], tx.Tuple[int, tx.Unpack[Ts]]
        )
    )
    assert captures[id(Ts)] == _run(str, bytes)
    # A closed slot (no `*Ts` run) captures nothing.
    assert list(
        typevartuple_captures(tx.Tuple[int], tx.Tuple[int])
    ) == []
    # A `Tuple[X, ...]` slot has no `TypeVarTuple`, so it captures nothing.
    assert list(
        typevartuple_captures(tx.Tuple[int], tx.Tuple[int, ...])
    ) == []
    # A nested (non-top-level) tuple is not read.
    assert list(
        typevartuple_captures(
            tx.Optional[tx.Tuple[int, str]],
            tx.Optional[tx.Tuple[int, tx.Unpack[Ts]]],
        )
    ) == []
    # A query whose shape does not match the slot captures nothing.
    assert list(
        typevartuple_captures(
            tx.Tuple[str], tx.Tuple[int, tx.Unpack[Ts]]
        )
    ) == []
    # A bare `tuple` query (no arguments) captures nothing.
    assert list(
        typevartuple_captures(tuple, tx.Tuple[int, tx.Unpack[Ts]])
    ) == []
    # A non-tuple query captures nothing.
    assert list(
        typevartuple_captures(int, tx.Tuple[int, tx.Unpack[Ts]])
    ) == []


def test_typevartuple_captures_with_a_suffix_slot() -> None:
    from bagof.dispatchers._lattice import typevartuple_captures

    Ts = tx.TypeVarTuple("Ts")
    # Slot `Tuple[int, *Ts, bytes]`, query `Tuple[int, str, str, bytes]`: `Ts`
    # captures the middle `(str, str)`.
    captures = dict(
        (id(t), run)
        for t, run in typevartuple_captures(
            tx.Tuple[int, str, str, bytes],
            tx.Tuple[int, tx.Unpack[Ts], bytes],
        )
    )
    assert captures[id(Ts)] == _run(str, str)
