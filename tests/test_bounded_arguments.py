"""Tests for `Super[C]` and `Between[L, U]` as type arguments of generics.

At an invariant slot, a bound names the range of arguments the slot
accepts: `List[Super[int]]` is every `List[Y]` whose argument `Y` is `int`
or above it. At a covariant or contravariant slot, the variance already
widens a plain argument, so a bound is accepted only when the variance
says the same thing and is refused when the variance would ignore one of
its ends.
"""

# stdlib
import collections.abc as cabc
import numbers
import sys

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers import (
    AmbiguousMethodError,
    Between,
    Exact,
    Function,
    Hint,
    NoMethodError,
    Signature,
    Super,
    dispatch,
)
from bagof.dispatchers._lattice import is_declaration_dependent, overlaps
from bagof.dispatchers.core import ishintstance, issubhint
from bagof.dispatchers.core._bounds import (
    _Lower,
    bound_ends,
    slot_bounds,
    slot_kind,
)
from bagof.dispatchers.core._introspect import (
    _CONTRAVARIANT,
    _COVARIANT,
    _INVARIANT,
)
from bagof.dispatchers.core._relation import clear_relation_cache

B, S, E = Between, Super, Exact
N = tx.Never
Integral, Real = numbers.Integral, numbers.Real
L = tx.List

T = tx.TypeVar("T")
T_co = tx.TypeVar("T_co", covariant=True)
T_contra = tx.TypeVar("T_contra", contravariant=True)
TB = tx.TypeVar("TB", bound=int)
TC = tx.TypeVar("TC", int, str)
T_S = tx.TypeVar("T_S", bound=Super[int])


class Animal:
    pass


class Dog(Animal):
    pass


class Puppy(Dog):
    pass


class Box(tx.Generic[T]):
    pass


class Src(tx.Generic[T_co]):
    pass


class Snk(tx.Generic[T_contra]):
    pass


class Row(tx.Sequence[T]):
    """An invariant parameter that reaches `Sequence`'s covariant one."""

    def __getitem__(self, index: tx.Any) -> tx.Any:
        raise IndexError(index)

    def __len__(self) -> int:
        return 0


class W(Snk[T]):
    """An invariant parameter that reaches `Snk`'s contravariant one."""


class IntList(tx.List[int]):
    pass


class BoolList(tx.List[bool]):
    pass


class IntegralList(tx.List[Integral]):
    pass


class ObjList(tx.List[object]):
    pass


class StrList(tx.List[str]):
    pass


class AnyList(tx.List[tx.Any]):
    pass


class LL(tx.List[tx.List[int]]):
    pass


class LLS(tx.List[tx.List[S[int]]]):
    pass


class ObjDict(tx.Dict[str, object]):
    pass


class BoolDict(tx.Dict[str, bool]):
    pass


def _fn(annotation: tx.Any) -> tx.Any:
    def f(xs: annotation) -> None: ...

    return f


def _table(
    columns: tx.Sequence[tx.Any], rows: tx.Sequence[tx.Tuple[tx.Any, str]]
) -> tx.List[tx.Tuple[tx.Any, tx.Any, tx.Optional[bool]]]:
    """Spell a truth table as `(row, column, expected)` cases.

    A `T` or an `F` is the expected answer, and an `E` means the relation
    raises a `TypeError`.
    """
    answers = {"T": True, "F": False, "E": None}
    return [
        (row, column, answers[line[index]])
        for row, line in rows
        for index, column in enumerate(columns)
    ]


def _check(sub: tx.Any, sup: tx.Any, expected: tx.Optional[bool]) -> None:
    if expected is None:
        with pytest.raises(TypeError):
            issubhint(sub, sup)
    else:
        assert issubhint(sub, sup) is expected


# --- readers -----------------------------------------------------------


def test_slot_bounds_reads_each_argument() -> None:
    assert slot_bounds(B[bool, Integral]) == (bool, Integral)
    assert slot_bounds(S[int]) == (int, tx.Any)
    assert slot_bounds(tx.Any) == (N, tx.Any)
    assert slot_bounds(T) == (N, tx.Any)
    assert slot_bounds(TB) == (N, int)
    assert slot_bounds(int) == (int, int)
    assert slot_bounds(E[int]) == (E[int], E[int])
    assert slot_bounds(tx.Annotated[TB, "metadata"]) == (N, int)
    assert bound_ends(S[int]) == (int, tx.Any)
    assert bound_ends(B[N, int]) == (N, int)


_KINDS = [
    (B[N, int], "ok", "redundant", "conflicting"),
    (S[int], "ok", "conflicting", "redundant"),
    (B[int, tx.Any], "ok", "conflicting", "redundant"),
    (B[bool, int], "ok", "conflicting", "conflicting"),
    (S[N], "ok", "redundant", "redundant"),
]


@pytest.mark.parametrize("bound, invariant, covariant, contra", _KINDS)
def test_slot_kind_table(
    bound: tx.Any, invariant: str, covariant: str, contra: str
) -> None:
    assert slot_kind(bound, _INVARIANT) == invariant
    assert slot_kind(bound, _COVARIANT) == covariant
    assert slot_kind(bound, _CONTRAVARIANT) == contra


# --- the order at an invariant slot ------------------------------------

# The arguments of the invariant table's columns and rows, each row with its
# answers. The last column and row are the bare generic.
_INVARIANT_COLUMNS = [int, B[N, int], B[bool, Integral], S[int], S[bool], TB]
_INVARIANT_ROWS = [
    (int, "TTTTTTTT"),
    (bool, "FTTFTTTT"),
    (B[N, int], "FTFFFTTT"),
    (B[bool, Integral], "FFTFTFTT"),
    (S[int], "FFFTTFTT"),
    (S[bool], "FFFFTFTT"),
    (TB, "FTFFFTTT"),
    (tx.Any, "FFFFFFTT"),
    (T, "FFFFFFTT"),
]


def _invariant_table(generic: tx.Any) -> tx.List[tx.Any]:
    columns = [generic[a] for a in _INVARIANT_COLUMNS + [tx.Any]] + [generic]
    rows = [(generic[a], line) for a, line in _INVARIANT_ROWS]
    return _table(columns, rows + [(generic, "FFFFFFFT")])


@pytest.mark.parametrize(
    "sub, sup, expected",
    _invariant_table(L) + _invariant_table(Box),
    ids=repr,
)
def test_issubhint_invariant_slot_table(
    sub: tx.Any, sup: tx.Any, expected: bool
) -> None:
    _check(sub, sup, expected)


_EQUIVALENT = [
    (L[B[N, tx.Any]], L[tx.Any]),
    (L[B[int, int]], L[int]),
    (L[B[N, int]], L[TB]),
    (Box[B[N, tx.Any]], Box[T]),
    (L[S[tx.Type[int]]], L[tx.Type[S[int]]]),
]


@pytest.mark.parametrize("a, b", _EQUIVALENT, ids=repr)
def test_equivalent_slot_arguments(a: tx.Any, b: tx.Any) -> None:
    assert issubhint(a, b) and issubhint(b, a)


def test_exact_is_an_ordinary_argument() -> None:
    # `Exact[int]` names the exact hint `int`, a narrower argument than
    # `int` and one that no range written with plain ends contains.
    assert issubhint(L[E[int]], L[int]) is False
    assert issubhint(L[E[int]], L[B[int, int]]) is False
    assert issubhint(L[E[int]], L[S[bool]]) is False
    assert issubhint(L[E[int]], L[B[N, int]]) is True
    assert issubhint(L[E[int]], L[TB]) is True
    assert issubhint(L[S[int]], L[E[int]]) is False


def test_constrained_typevar_at_an_invariant_slot() -> None:
    assert issubhint(L[B[int, int]], L[TC]) is True
    assert issubhint(L[B[bool, Integral]], L[TC]) is False
    assert issubhint(L[TC], L[B[N, object]]) is True
    assert issubhint(L[TC], L[S[bool]]) is False


# --- covariant and contravariant slots ---------------------------------

_COVARIANT_COLUMNS = [
    tx.Sequence[int],
    tx.Sequence[B[N, int]],
    tx.Sequence[Integral],
    tx.Sequence[tx.Any],
    tx.Sequence[S[int]],
]
_COVARIANT_ROWS = [
    (tx.Sequence[bool], "TTTTE"),
    (tx.Sequence[int], "TTTTE"),
    (tx.Sequence[B[N, int]], "TTTTE"),
    (Row[int], "TTTTE"),
    (Row[B[N, int]], "TTTTE"),
    (Row[TB], "TTTTE"),
    (Row[S[int]], "FFFTE"),
    (Row[B[bool, Integral]], "FFTTE"),
    (Src[S[int]], "EEEEE"),
]


@pytest.mark.parametrize(
    "sub, sup, expected",
    _table(_COVARIANT_COLUMNS, _COVARIANT_ROWS),
    ids=repr,
)
def test_issubhint_covariant_slot_table(
    sub: tx.Any, sup: tx.Any, expected: bool
) -> None:
    _check(sub, sup, expected)


_CONTRAVARIANT_COLUMNS = [
    Snk[int],
    Snk[S[int]],
    Snk[bool],
    Snk[object],
    Snk,
    Snk[B[N, int]],
]
_CONTRAVARIANT_ROWS = [
    (Snk[int], "TTTFTE"),
    (Snk[S[int]], "TTTFTE"),
    (Snk[object], "TTTTTE"),
    (Snk[bool], "FFTFTE"),
    (W[S[int]], "TTTFTE"),
    (W[B[bool, Integral]], "FFTFTE"),
    (W[B[N, int]], "FFFFTE"),
]


@pytest.mark.parametrize(
    "sub, sup, expected",
    _table(_CONTRAVARIANT_COLUMNS, _CONTRAVARIANT_ROWS),
    ids=repr,
)
def test_issubhint_contravariant_slot_table(
    sub: tx.Any, sup: tx.Any, expected: bool
) -> None:
    _check(sub, sup, expected)


def test_redundant_bound_reads_as_plain() -> None:
    for bounded, plain in [
        (Src[B[N, int]], Src[int]),
        (Snk[S[int]], Snk[int]),
        (tx.Mapping[str, B[N, int]], tx.Mapping[str, int]),
        (tx.Sequence[S[N]], tx.Sequence[tx.Any]),
    ]:
        assert issubhint(bounded, plain) and issubhint(plain, bounded)


def test_an_open_argument_reaches_a_base_as_the_range_it_stands_for() -> None:
    # At `W`'s invariant slot `Any` and a `TypeVar` stand for every argument
    # they admit, and they keep standing for them at `Snk`'s contravariant
    # slot, the way the equivalent `Between[Never, ...]` does.
    for open_, bounded in [(tx.Any, B[N, tx.Any]), (TB, B[N, int])]:
        assert issubhint(W[open_], W[bounded]) is True
        assert issubhint(W[bounded], W[open_]) is True
        for sup in (Snk[int], Snk[bool], Snk[N], Snk):
            assert issubhint(W[open_], sup) is issubhint(W[bounded], sup)
    assert issubhint(W[tx.Any], Snk[int]) is False
    assert issubhint(W[tx.Any], Snk[N]) is True
    # `Snk[T]` and `Snk[Any]` sit at the bottom of the contravariant order,
    # so the family of every `W[Y]` is never below them.
    assert issubhint(W[T], Snk[T]) is False
    assert issubhint(W[tx.Any], Snk[tx.Any]) is False
    # The covariant reading is unchanged: the range's upper end.
    assert issubhint(Row[TB], tx.Sequence[int]) is True
    assert issubhint(Row[tx.Any], tx.Sequence[int]) is False
    # An open argument inside another hint is handed over as written.

    class Nested(tx.Sequence[tx.List[T]]):
        def __getitem__(self, index: tx.Any) -> tx.Any:
            raise IndexError(index)

        def __len__(self) -> int:
            return 0

    assert issubhint(Nested[tx.Any], tx.Sequence[tx.List[tx.Any]]) is True
    assert issubhint(Nested[tx.Any], tx.Sequence[tx.List[int]]) is False


def test_a_constrained_typevar_reaches_a_base_as_each_constraint() -> None:
    assert issubhint(W[str], W[TC]) is True
    assert issubhint(W[TC], Snk[int]) is False
    assert issubhint(W[TC], Snk[N]) is True
    assert issubhint(Row[TC], tx.Sequence[tx.Union[int, str]]) is True
    assert issubhint(Row[TC], tx.Sequence[int]) is False

    class Two(Snk[T], tx.Generic[T, T_co]):
        pass

    # The same variable is given the same constraint wherever it appears.
    assert issubhint(Two[TC, TC], Snk[N]) is True


# --- Dict and Mapping --------------------------------------------------

_MAPPINGS = [
    (tx.Dict[str, int], tx.Dict[str, S[int]], True),
    (tx.Dict[str, bool], tx.Dict[str, S[int]], False),
    (tx.Dict[object, int], tx.Dict[S[str], int], True),
    (tx.Dict[str, S[int]], tx.Dict[str, S[bool]], True),
    (tx.Dict[str, B[N, int]], tx.Mapping[str, int], True),
    (tx.Dict[str, S[int]], tx.Mapping[str, int], False),
    (tx.Dict[str, S[int]], tx.Mapping[str, tx.Any], True),
    (tx.Dict[object, int], tx.Mapping[S[str], int], True),
    (tx.Mapping[str, B[N, int]], tx.Mapping[str, int], True),
    (tx.Mapping[str, int], tx.Mapping[str, B[N, int]], True),
    (tx.Mapping[str, S[int]], tx.Mapping[str, tx.Any], None),
    (tx.Mapping[str, int], tx.Mapping[str, S[int]], None),
]


@pytest.mark.parametrize("sub, sup, expected", _MAPPINGS, ids=repr)
def test_dict_and_mapping_slots(
    sub: tx.Any, sup: tx.Any, expected: tx.Optional[bool]
) -> None:
    _check(sub, sup, expected)


# --- nested bounds, unions, Type and Hint ------------------------------


def test_nested_bounds_do_not_propagate() -> None:
    wide = L[B[N, L[S[int]]]]
    assert issubhint(L[L[int]], L[L[S[int]]]) is False
    assert issubhint(L[L[S[int]]], L[L[S[int]]]) is True
    assert issubhint(L[L[int]], wide) is True
    assert issubhint(L[L[bool]], wide) is False
    assert issubhint(L[L[object]], wide) is True
    assert issubhint(L[L[S[int]]], wide) is True


def test_union_arguments() -> None:
    assert issubhint(L[int], tx.Union[L[S[int]], L[str]]) is True
    assert issubhint(L[bool], tx.Union[L[S[int]], L[str]]) is False
    ranged = L[B[tx.Union[int, str], object]]
    assert issubhint(L[tx.Union[int, str]], ranged) is True
    assert issubhint(L[int], ranged) is False


def test_hint_and_type_of_a_bounded_generic() -> None:
    family = Hint[L[S[int]]]
    for hint in (L[int], L[Integral], L[object], L[S[int]], L[B[int, object]]):
        assert ishintstance(hint, family) is True, hint
    for hint in (L[bool], L[tx.Any], list):
        assert ishintstance(hint, family) is False, hint

    above = Hint[S[L[int]]]
    for hint in (L[int], L[B[N, int]], L[TB], L[S[int]], tx.Sequence[int]):
        assert ishintstance(hint, above) is True, hint
    assert ishintstance(object, above) is True
    assert ishintstance(tx.Any, above) is True
    assert ishintstance(L[bool], above) is False

    classes = tx.Type[L[S[int]]]
    assert ishintstance(IntList, classes) is True
    assert ishintstance(ObjList, classes) is True
    assert ishintstance(BoolList, classes) is False
    assert ishintstance(list, classes) is False


def test_a_bound_inside_a_callable_parameter_generic_is_fine() -> None:
    wide = tx.Callable[[L[S[int]]], None]
    assert issubhint(wide, tx.Callable[[L[int]], None]) is True
    assert issubhint(wide, tx.Callable[[L[bool]], None]) is False
    assert issubhint(tx.Callable[[L[object]], None], wide) is False
    assert ishintstance(print, wide) is True


# --- values ------------------------------------------------------------

_VALUE_COLUMNS = [
    L[S[int]],
    L[B[N, int]],
    L[B[bool, Integral]],
    L[L[S[int]]],
    L[B[N, L[S[int]]]],
]
_VALUE_ROWS = [
    ([1], "TTTTT"),
    (AnyList(), "TTTTT"),
    (IntList(), "TTTFF"),
    (BoolList(), "FTTFF"),
    (IntegralList(), "TFTFF"),
    (ObjList(), "TFFFF"),
    (StrList(), "FFFFF"),
    (LL(), "FFFFT"),
    (LLS(), "FFFTT"),
]


@pytest.mark.parametrize(
    "value, hint, expected",
    _table(_VALUE_COLUMNS, _VALUE_ROWS),
    ids=repr,
)
def test_ishintstance_declared_values_table(
    value: tx.Any, hint: tx.Any, expected: bool
) -> None:
    assert ishintstance(value, hint) is expected


_BOX_ROWS = [
    (Box(), "TTT"),
    (Box[tx.Any](), "TTT"),
    (Box[TB](), "TTT"),
    (Box[int](), "TTT"),
    (Box[bool](), "FTT"),
    (Box[Integral](), "TFT"),
    (Box[object](), "TFF"),
    (Box[str](), "FFF"),
    (Box[S[int]](), "TFF"),
    (Box[B[N, int]](), "FTF"),
]


@pytest.mark.parametrize(
    "value, hint, expected",
    _table([Box[S[int]], Box[B[N, int]], Box[B[bool, Integral]]], _BOX_ROWS),
    ids=repr,
)
def test_a_box_declares_what_it_holds(
    value: tx.Any, hint: tx.Any, expected: bool
) -> None:
    assert ishintstance(value, hint) is expected


def test_variant_slots_read_what_a_value_declares() -> None:
    assert ishintstance(IntList(), tx.Sequence[B[N, int]]) is True
    assert ishintstance(IntegralList(), tx.Sequence[B[N, int]]) is False
    assert ishintstance(Snk[int](), Snk[S[int]]) is True
    assert ishintstance(Snk[object](), Snk[S[int]]) is True
    assert ishintstance(Snk[bool](), Snk[S[int]]) is False
    assert ishintstance(Snk(), Snk[S[int]]) is True
    assert ishintstance(ObjDict(), tx.Dict[str, S[int]]) is True
    assert ishintstance(BoolDict(), tx.Dict[str, S[int]]) is False
    for value in ({}, ObjDict(), BoolDict()):
        with pytest.raises(TypeError, match="puts a lower bound"):
            ishintstance(value, tx.Mapping[str, S[int]])


def test_ishintstance_refuses_a_conflicting_bound_even_for_a_plain_list(
) -> None:
    for hint in (tx.Sequence[S[int]], Src[B[bool, int]], Snk[B[N, int]]):
        with pytest.raises(TypeError, match="bound on argument 1"):
            ishintstance([], hint)


def test_value_and_slot_levels_do_not_mix() -> None:
    # On a value, a bound reads the value's own class; as a type argument it
    # reads the argument that the value declares.
    assert ishintstance(IntList(), L[S[int]]) is True
    assert ishintstance(IntList(), S[list]) is False
    assert ishintstance(list(), S[list]) is True
    with pytest.raises(TypeError, match="cannot bound a value with"):
        dispatch(_fn(S[L[int]]))
    assert dispatch(_fn(L[S[int]]))(IntList()) is None


# --- refusals ----------------------------------------------------------

_P = tx.ParamSpec("_P")


class Hook(tx.Generic[_P, T]):
    pass


_REFUSED = [
    (
        tx.Sequence[S[int]],
        "Super[int] puts a lower bound on argument 1 of Sequence, whose "
        "type parameter is covariant, so Sequence[Super[int]] would accept "
        "every Sequence. Write Sequence without an argument to accept every "
        "Sequence, or Sequence[int] to accept Sequence[int] and the "
        "parametrisations below it.",
    ),
    (
        tx.Sequence[B[bool, int]],
        "Between[bool, int] puts a lower bound on argument 1 of Sequence, "
        "whose type parameter is covariant, so the lower bound bool changes "
        "nothing there: Sequence[int] already accepts every parametrisation "
        "below int. Write Sequence[int].",
    ),
    (
        Snk[B[N, int]],
        "Between[Never, int] puts an upper bound on argument 1 of Snk, whose "
        "type parameter is contravariant, so Snk[Between[Never, int]] would "
        "accept every Snk. Write Snk without an argument to accept every "
        "Snk, or Snk[int] to accept Snk[int] and the parametrisations above "
        "it.",
    ),
    (
        Snk[B[bool, int]],
        "Between[bool, int] puts an upper bound on argument 1 of Snk, whose "
        "type parameter is contravariant, so the upper bound int changes "
        "nothing there: Snk[bool] already accepts every parametrisation "
        "above bool. Write Snk[bool].",
    ),
    (
        tx.Mapping[str, S[int]],
        "Super[int] puts a lower bound on argument 2 of Mapping, whose type "
        "parameter is covariant, so Mapping[str, Super[int]] would mean the "
        "same as Mapping[str, Any]. Write Mapping[str, Any] to accept every "
        "such Mapping, or Mapping[str, int] to accept Mapping[str, int] and "
        "the parametrisations below it.",
    ),
    (
        tx.Generator[int, B[N, int], None],
        "Between[Never, int] puts an upper bound on argument 2 of "
        "Generator, whose type parameter is contravariant, so "
        "Generator[int, Between[Never, int], None] would mean the same as "
        "Generator[int, Never, None]. Write Generator[int, Never, None] to "
        "accept every such Generator, or Generator[int, int, None] to "
        "accept Generator[int, int, None] and the parametrisations above "
        "it.",
    ),
    (
        L[tx.Union[S[int], str]],
        "Super[int] cannot be a member of a union, or the bound of a "
        "TypeVar, inside a type argument of List: a bound there has to be "
        "the whole argument, so that List can read it as the range of "
        "arguments it accepts. Write Union[List[Super[int]], List[B]] in "
        "place of List[Union[Super[int], B]], and List[Super[int]] in place "
        "of List[T] with T bounded by Super[int].",
    ),
    (
        tx.AbstractSet[S[int]],
        "Super[int] puts a lower bound on argument 1 of AbstractSet, whose "
        "type parameter is covariant, so AbstractSet[Super[int]] would "
        "accept every AbstractSet. Write AbstractSet without an argument to "
        "accept every AbstractSet, or AbstractSet[int] to accept "
        "AbstractSet[int] and the parametrisations below it.",
    ),
    (tx.FrozenSet[S[int]], "argument 1 of FrozenSet, whose type"),
    (L[T_S], "Super[int] cannot be a member of a union, or the bound of"),
    (Box[tx.Optional[S[int]]], "inside a type argument of Box"),
    (
        Hook[int, S[int]],
        "Super[int] cannot be an argument of Hook: the variance of its type "
        "parameters cannot be read, so a bound there has no meaning. Write "
        "a plain argument instead.",
    ),
    (Hook[S[int], int], "cannot be an argument of Hook"),
    (tx.Tuple[S[int]], "Super[int] cannot be an element of Tuple[...]"),
    (tx.Tuple[B[N, int], ...], "cannot be an element of Tuple[...]"),
    (tx.Callable[[S[int]], None], "cannot be a parameter or the return"),
    (tx.Callable[..., B[N, int]], "cannot be a parameter or the return"),
    (L[S], "Super needs a bound"),
    (L[tx.TypeVar("_TCS", S[int], str)], "cannot be a constraint of a"),
]


@pytest.mark.parametrize("hint, message", _REFUSED, ids=repr)
def test_registration_and_relation_messages(
    hint: tx.Any, message: str
) -> None:
    origin = tx.get_origin(hint)
    other = {
        tuple: tx.Tuple[int],
        cabc.Callable: tx.Callable[[int], None],
    }.get(origin, object)
    calls = [
        lambda: dispatch(_fn(hint)),
        lambda: Function("f").register((hint,))(lambda xs: xs),
        lambda: Signature.from_hints(hint),
        lambda: issubhint(hint, other),
        lambda: issubhint(hint, hint),
    ]
    if isinstance(origin, type) and origin not in (tuple, cabc.Callable):
        calls.append(lambda: ishintstance(object(), hint))
    for call in calls:
        with pytest.raises(TypeError) as info:
            call()
        assert message in str(info.value)
    with pytest.raises(TypeError) as info:
        dispatch(_fn(hint))
    assert str(info.value).startswith("'xs' of f: ")


def test_a_star_args_parameter_is_checked_too() -> None:
    def f(*xs: tx.Sequence[S[int]]) -> None: ...

    with pytest.raises(TypeError, match="^'xs' of f: Super.int. puts"):
        dispatch(f)


def test_a_forward_reference_is_checked_once_resolved() -> None:
    namespace: tx.Dict[str, tx.Any] = {}
    exec("def f(xs: 'Later'): pass", namespace)
    sig = Signature.from_callable(namespace["f"])
    namespace["Later"] = tx.Sequence[S[int]]
    with pytest.raises(TypeError, match="'xs' of f: Super.int. puts a lower"):
        sig._settle()

    namespace = {}
    exec("def f(xs: 'Later'): pass", namespace)
    sig = Signature.from_callable(namespace["f"])
    namespace["Later"] = L[S[int]]
    sig._settle()
    assert not sig._deferred


def test_an_interval_emptied_by_a_forward_reference_is_refused() -> None:
    # What `List[Between[int, "Later"]]` becomes once `Later` resolves to
    # `bool`: the interval is never built by `Between`.
    emptied = L[tx.Annotated[bool, _Lower(int)]]
    with pytest.raises(TypeError, match=r"Between\[int, bool\] is empty"):
        dispatch(_fn(emptied))


def test_a_bound_nested_in_a_slot_bound_is_refused() -> None:
    # What `List[Between[Never, "Later"]]` becomes once `Later` resolves to
    # a hint holding a bound.
    nested = L[tx.Annotated[tx.Optional[S[int]], _Lower(N)]]
    for call in (
        lambda: issubhint(nested, L[int]),
        lambda: dispatch(_fn(nested)),
    ):
        needle = "cannot appear inside an Exact, Super or Between form"
        with pytest.raises(TypeError, match=needle):
            call()


def test_a_class_writing_a_conflicting_bound_in_its_base_is_refused() -> None:
    class Written(tx.Sequence[S[int]]):
        def __getitem__(self, index: tx.Any) -> tx.Any:
            raise IndexError(index)

        def __len__(self) -> int:
            return 0

    class Below(Written):
        pass

    needle = (
        "Written derives from Sequence[Super[int]]: Super[int] puts a lower "
        "bound on argument 1 of Sequence"
    )
    for call in (
        lambda: issubhint(Written, tx.Sequence[int]),
        lambda: issubhint(Below, tx.Sequence[tx.Any]),
        lambda: ishintstance(Written(), tx.Sequence[int]),
    ):
        with pytest.raises(TypeError) as info:
            call()
        assert str(info.value).startswith(needle)
    # A bare super-hint asks nothing of the arguments.
    assert issubhint(Written, tx.Sequence) is True


def test_a_bound_reaching_a_base_inside_a_union_is_refused() -> None:
    class Either(tx.Sequence[tx.Union[T, str]]):
        def __getitem__(self, index: tx.Any) -> tx.Any:
            raise IndexError(index)

        def __len__(self) -> int:
            return 0

    assert issubhint(Either[int], tx.Sequence[tx.Union[int, str]]) is True
    with pytest.raises(TypeError, match="inside a type argument of Sequence"):
        issubhint(Either[S[int]], tx.Sequence[tx.Any])


def test_a_bound_reaching_a_base_of_unreadable_variance_is_refused() -> None:
    class Hooked(Hook[int, T]):
        pass

    with pytest.raises(TypeError, match="cannot be an argument of Hook"):
        issubhint(Hooked[S[int]], Hook[int, int])


_Ts = tx.TypeVarTuple("_Ts")


class Run(tx.Generic[tx.Unpack[_Ts]]):
    pass


def test_a_bound_compared_positionally_is_refused() -> None:
    # `Run` is a base written without arguments, so nothing maps
    # `Unmapped`'s argument onto `Run`'s, and the two are compared by
    # position with no variance to read the bound by.
    class Unmapped(Run, tx.Generic[T]):
        pass

    assert issubhint(Unmapped[bool], Run[int]) is True
    for sup in (Run[int], Run[tx.Unpack[tx.Tuple[int, ...]]]):
        with pytest.raises(TypeError, match="cannot be an argument of Run"):
            issubhint(Unmapped[S[int]], sup)
    # Fixed arguments of different lengths are never paired, so the bound is
    # not read, as `Dict[str, Super[int]]` is not read against `Container`.
    assert issubhint(Unmapped[S[int]], Run[int, str]) is False
    assert issubhint(tx.Dict[str, S[int]], tx.Container[int]) is False


_ACCEPTED = [
    L[S[int]],
    tx.Dict[str, B[N, int]],
    Box[B[bool, Integral]],
    tx.Type[L[S[int]]],
    Hint[L[S[int]]],
    tx.Optional[L[S[int]]],
    tx.Callable[[L[S[int]]], None],
    tx.Type[S[L[S[int]]]],
    tx.Sequence[B[N, int]],
    Snk[S[int]],
    tx.Mapping[S[str], int],
    L[B[tx.Literal[1], int]],
    L[tx.TypeVar("_TL", bound=L[S[int]])],
    tx.Sequence[tx.Union[L[S[int]], str]],
]


@pytest.mark.parametrize("hint", _ACCEPTED, ids=repr)
def test_registration_accepts_bounds_where_they_can_be_read(
    hint: tx.Any,
) -> None:
    Function("f").register(_fn(hint))
    Signature.from_hints(hint)


def test_a_signature_mixes_both_levels() -> None:
    f = Function("f")

    @f.register
    def _mixed(x: S[Dog], xs: L[S[int]]) -> str:
        return "mixed"

    assert f(Animal(), IntList()) == "mixed"
    assert f(Dog(), ObjList()) == "mixed"
    with pytest.raises(NoMethodError):
        f(Puppy(), IntList())
    with pytest.raises(NoMethodError):
        f(Dog(), BoolList())


# --- dispatch ----------------------------------------------------------


def test_dispatch_on_a_bounded_list() -> None:
    f = Function("f")

    @f.register
    def _above(xs: L[S[int]]) -> str:
        return "above int"

    @f.register
    def _exactly(xs: L[int]) -> str:
        return "int"

    @f.register
    def _below(xs: L[B[N, bool]]) -> str:
        return "bool or below"

    assert f.ambiguities() == []

    assert f(IntList()) == "int"
    assert f(ObjList()) == "above int"
    assert f(IntegralList()) == "above int"
    assert f(BoolList()) == "bool or below"
    with pytest.raises(NoMethodError):
        f(StrList())
    # A plain list declares nothing, so it fits every method, and the two
    # that are not ordered against each other tie.
    with pytest.raises(AmbiguousMethodError):
        f([1])


def test_dispatch_bounded_vs_upper_is_ambiguous_and_priority_settles() -> None:
    f = Function("f")

    @f.register
    def _upper(xs: L[B[N, int]]) -> str:
        return "int or below"

    @f.register
    def _lower(xs: L[S[bool]]) -> str:
        return "bool or above"

    assert len(f.ambiguities()) == 1

    # `bool` and `int` lie in both ranges.
    for value in (BoolList(), IntList()):
        with pytest.raises(AmbiguousMethodError):
            f(value)
    assert f(ObjList()) == "bool or above"
    with pytest.raises(NoMethodError):
        f(StrList())

    g = Function("g")

    @g.register(priority=1)
    def _upper_first(xs: L[B[N, int]]) -> str:
        return "int or below"

    @g.register
    def _lower_second(xs: L[S[bool]]) -> str:
        return "bool or above"

    assert g.ambiguities() == []

    assert g(IntList()) == "int or below"


_OVERLAPS = [
    (L[B[N, int]], L[S[bool]], True),
    (L[S[int]], L[B[N, bool]], False),
    (tx.Dict[str, B[N, int]], tx.Dict[S[str], S[bool]], True),
    (L[B[N, int]], tx.MutableSequence[S[bool]], True),
    (tx.MutableSequence[S[bool]], L[B[N, int]], True),
    (L[S[int]], tx.MutableSequence[B[N, bool]], False),
    (L[int], L[str], False),
    (L[L[int]], L[L[str]], False),
    (tx.Sequence[L[S[int]]], tx.Sequence[L[B[N, int]]], True),
    (tx.Type[L[S[int]]], tx.Type[L[B[N, int]]], True),
    (Hint[L[S[int]]], Hint[L[B[N, int]]], True),
    (Hint[L[S[int]]], Hint[L[B[N, bool]]], False),
    (Box[TB], Box[S[bool]], True),
    (Box[TC], Box[B[Integral, object]], False),
    (Box[TC], Box[B[bool, int]], True),
    (L[S[int]], tx.Sequence[str], False),
    (Row[S[int]], tx.Sequence[str], False),
    (Row[S[int]], tx.Sequence[object], True),
    (L[S[int]], Box[S[int]], False),
    # A class that fills in its base's argument itself has no argument of its
    # own to vary.
    (IntList, L[B[N, bool]], False),
    (L[E[int]], L[S[bool]], False),
    (L[E[int]], L[B[N, int]], True),
    (tx.Callable[[L[S[int]]], None], tx.Callable[[L[int]], None], False),
]


@pytest.mark.parametrize("a, b, expected", _OVERLAPS, ids=repr)
def test_overlaps_slots(a: tx.Any, b: tx.Any, expected: bool) -> None:
    assert overlaps(a, b) is expected
    assert overlaps(b, a) is expected


@pytest.mark.skipif(sys.version_info < (3, 9), reason="PEP 585 aliases")
def test_pep_585_spellings() -> None:
    class GL(list[T]):
        pass

    assert issubhint(list[int], list[S[int]]) is True
    assert issubhint(list[bool], list[S[int]]) is False
    assert ishintstance(GL[int](), GL[S[int]]) is True
    assert ishintstance(GL[bool](), GL[S[int]]) is False
    assert issubhint(GL[S[int]], L[S[bool]]) is True
    assert overlaps(list[B[N, int]], list[S[bool]]) is True


def test_bounded_generic_renders() -> None:
    sig = Signature.from_hints(
        L[S[int]],
        tx.Dict[str, B[N, int]],
        tx.Callable[[L[S[int]]], None],
        tx.Callable[..., L[S[int]]],
        tx.Dict[T, S[int]],
    )
    text = repr(sig)
    assert "Dict[~T, Super[int]]" in text
    assert "List[Super[int]]" in text
    assert "Dict[str, Between[Never, int]]" in text
    assert "Callable[[List[Super[int]]], NoneType]" in text
    assert "Callable[..., List[Super[int]]]" in text


def test_a_bounded_argument_is_declaration_dependent() -> None:
    assert is_declaration_dependent(L[S[int]]) is True
    assert is_declaration_dependent(Box[B[bool, Integral]]) is True


def test_the_relation_cache_keeps_bounded_arguments_apart() -> None:
    clear_relation_cache()
    assert issubhint(L[int], L[S[int]]) is True
    assert issubhint(L[int], L[S[bool]]) is True
    assert issubhint(L[int], L[B[bool, bool]]) is False
    assert issubhint(L[int], L[int]) is True


# --- laws --------------------------------------------------------------

_ARGUMENTS = [
    int,
    bool,
    Integral,
    object,
    str,
    tx.Any,
    T,
    TB,
    TC,
    E[int],
    S[int],
    S[bool],
    S[Integral],
    B[N, int],
    B[N, bool],
    B[bool, Integral],
    B[int, int],
    B[str, str],
    B[N, tx.Any],
    B[N, tx.Union[int, str]],
]
_SWEEP = (
    [L[a] for a in _ARGUMENTS]
    + [Box[a] for a in _ARGUMENTS]
    + [W[a] for a in _ARGUMENTS]
    + [Row[a] for a in _ARGUMENTS if a is not TC]
    + [tx.Dict[str, a] for a in (int, bool, object, S[int], B[N, int])]
    + [tx.Mapping[str, a] for a in (int, bool, object, tx.Any, B[N, int])]
    + [tx.Sequence[a] for a in (int, bool, Integral, tx.Any, B[N, int])]
    + [Src[a] for a in (int, bool, tx.Any, T, B[N, int], B[N, Integral])]
    + [Snk[a] for a in (int, bool, object, tx.Any, T, TB, S[int], S[bool])]
    + [L[L[S[int]]], L[L[int]], L[L[object]], L[B[N, L[S[int]]]]]
    + [IntList, BoolList, ObjList, StrList, LL, LLS, list, Box, Snk, object]
)
_VALUES = [
    [1],
    IntList(),
    BoolList(),
    IntegralList(),
    ObjList(),
    StrList(),
    LL(),
    LLS(),
    Box(),
    Box[int](),
    Box[bool](),
    Box[object](),
    Box[S[int]](),
    Box[B[bool, Integral]](),
    Snk[int](),
    Snk[bool](),
    Snk[object](),
    W[int](),
    W[str](),
    W[S[int]](),
    Row[int](),
    Row[S[int]](),
    ObjDict(),
    BoolDict(),
]


@pytest.fixture(scope="module")
def order() -> tx.Dict[tx.Tuple[int, int], bool]:
    return {
        (i, j): issubhint(a, b)
        for i, a in enumerate(_SWEEP)
        for j, b in enumerate(_SWEEP)
    }


def test_slot_order_is_a_preorder(
    order: tx.Dict[tx.Tuple[int, int], bool],
) -> None:
    indices = range(len(_SWEEP))
    for i in indices:
        assert order[i, i] is True, _SWEEP[i]
    for i in indices:
        for j in indices:
            if not order[i, j]:
                continue
            for k in indices:
                if order[j, k]:
                    assert order[i, k], (_SWEEP[i], _SWEEP[j], _SWEEP[k])


def test_slot_value_consistency(
    order: tx.Dict[tx.Tuple[int, int], bool],
) -> None:
    member = {
        (v, i): ishintstance(value, hint)
        for v, value in enumerate(_VALUES)
        for i, hint in enumerate(_SWEEP)
    }
    for (i, j), below in order.items():
        if not below:
            continue
        for v, value in enumerate(_VALUES):
            if member[v, i]:
                assert member[v, j], (value, _SWEEP[i], _SWEEP[j])
