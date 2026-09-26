"""Relation-level tests for `TypeVarTuple` / `Unpack` in tuples and callables.

These exercise the sub-hint relation over `Tuple[..., *Ts]` shapes and the
`Callable[[int, *Ts], R]` open tail directly (`issubhint`), plus the
tuple-shape classifier and matcher. The dispatch-level tests (joint solving,
ambiguity, registration rejection) live in `test_typevartuple_dispatch.py`.

Every hint is spelled with `tx.Unpack[Ts]`, never the `*Ts` star syntax, which
is a `SyntaxError` below 3.11 (and this suite -- like the docstring harness --
runs on 3.8). One 3.11+-only test checks the star syntax orders identically.
"""

# stdlib
import sys
import warnings

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers.core import UnknownHintWarning, ishintstance, issubhint
from bagof.dispatchers.core._relation import (
    _CLOSED_TUPLE_MATCH,
    _match_tuple,
    _tuple_shape,
    _TupleShape,
)

T = tx.Tuple
C = tx.Callable
U = tx.Unpack
Ts = tx.TypeVarTuple("Ts")
Us = tx.TypeVarTuple("Us")
P = tx.ParamSpec("P")


@pytest.fixture(autouse=True)
def _no_unknown_hint_warnings() -> tx.Iterator[None]:
    """Fail if any row trips an ``UnknownHintWarning``.

    An `Unpack[Ts]` or `*Ts` is recognised, so nothing here should ever be
    treated as an unknown `Any`. Turning that warning into an error keeps the
    relation honest.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownHintWarning)
        yield


# --- the semantics table, both directions ------------------------------

# (hint, superhint, expected) -- straight from RFC 0001 §2.1 / the Phase-8(b)
# table. `Tuple[()] <= Tuple[*Ts]` is version-dependent (issue #36) and tested
# separately below.
TABLE = [
    (T[int, str], T[int, U[Ts]], True),          # Ts captures (str,)
    (T[int], T[int, U[Ts]], True),               # zero-run allowed
    (T[int, U[Ts]], T[U[Ts]], True),             # shorter fixed prefix wider
    (T[U[Ts]], T[int, U[Ts]], False),            # longer fixed prefix stricter
    (T[int, U[Ts], str], T[int, U[Ts]], True),   # shorter suffix wider
    (T[int, U[Ts]], T[int, U[Ts], str], False),  # longer suffix stricter
    (T[int, U[Ts]], T[U[Ts], int], False),       # prefix vs suffix: no order
    (T[U[Ts], int], T[int, U[Ts]], False),
    (T[int, U[Ts]], T[int, ...], False),         # *Ts vs ... incomparable
    (T[int, ...], T[int, U[Ts]], False),
    (T[int, ...], T[U[Ts]], True),               # Tuple[*Ts] == Tuple[Any,...]
    (T[U[Ts]], T[tx.Any, ...], True),
    (T[tx.Any, ...], T[U[Ts]], True),
    (T[U[Ts]], tuple, True),                      # Tuple[*Ts] <= tuple
    (tuple, T[U[Ts]], False),                     # bare tuple holds anything
    (T[bool, U[Ts]], T[int, U[Ts]], True),        # elements covariant
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    TABLE,
    ids=[f"{h}<:{s}" for h, s, _ in TABLE],
)
def test_semantics_table(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_preserved_fixed_ellipsis_rows() -> None:
    # The tuple-shape rewrite must not change the fixed/ellipsis rows: a closed
    # tuple whose middle does not satisfy the run is not a sub-hint (`str <=
    # int` fails), and the classic chain still holds.
    assert issubhint(T[int, str], T[int, ...]) is False
    assert issubhint(T[str, int], T[int, ...]) is False
    assert issubhint(T[int], T[int, ...]) is True
    assert issubhint(T[int, ...], T[int]) is False
    assert issubhint(T[bool, ...], T[int, ...]) is True


def test_reflexive_over_variadic_tuples() -> None:
    for hint in (
        T[int, U[Ts]],
        T[U[Ts]],
        T[int, U[Ts], str],
        T[U[Ts], int],
        T[int, ...],
    ):
        assert issubhint(hint, hint) is True, hint


def test_bare_typevartuple_run_behaves_like_any_element() -> None:
    # `Tuple[*Ts]` accepts any tuple's elements, so any fixed tuple is a
    # sub-hint of it (a run of zero or more `Any`).
    assert issubhint(T[int, str, bytes], T[U[Ts]]) is True
    assert issubhint(T[int], T[U[Ts]]) is True


# --- issubhint / ishintstance on a lone Unpack[Ts] ----------------------


def test_lone_unpack_typevartuple_is_any() -> None:
    # A standalone `Unpack[Ts]` (the `*args: *Ts` tail read as one slot)
    # behaves like `Any`: everything is a sub-hint of it, it is a sub-hint only
    # of `Any` and itself, and every value is an instance of it -- no warning.
    assert issubhint(int, U[Ts]) is True
    assert issubhint(U[Ts], int) is False
    assert issubhint(U[Ts], tx.Any) is True
    assert ishintstance(123, U[Ts]) is True
    assert ishintstance("x", U[Ts]) is True


# --- user Generic[Unpack[Ts]] ------------------------------------------


class _Arr(tx.Generic[U[Ts]]):
    pass


def test_user_generic_unpack_is_ordered() -> None:
    # A user class parametrised by a `TypeVarTuple` is ordered through the same
    # tuple-shape path, for free (`_issubclasshint` -> `_issubargs`).
    assert issubhint(_Arr[int, str], _Arr[int, U[Ts]]) is True
    assert issubhint(_Arr[int], _Arr[int, U[Ts]]) is True
    assert issubhint(_Arr[U[Ts]], _Arr[int, U[Ts]]) is False
    assert issubhint(_Arr[bool, U[Ts]], _Arr[int, U[Ts]]) is True


# --- Unpack[Tuple[...]] splicing ---------------------------------------


def test_unpack_fixed_tuple_flattens() -> None:
    # `Tuple[int, *Tuple[str, int]]` is exactly `Tuple[int, str, int]`.
    assert issubhint(T[int, U[T[str, int]]], T[int, str, int]) is True
    assert issubhint(T[int, str, int], T[int, U[T[str, int]]]) is True
    assert issubhint(T[int, U[T[str, int]]], T[int, str, bool]) is False


def test_unpack_unbounded_tuple_becomes_a_run() -> None:
    # `Tuple[*Tuple[str, ...]]` is `Tuple[str, ...]`.
    assert issubhint(T[U[T[str, ...]]], T[str, ...]) is True
    assert issubhint(T[str, ...], T[U[T[str, ...]]]) is True
    # open not <= fixed
    assert issubhint(T[U[T[str, ...]]], T[str, str]) is False


# --- the classifier and matcher directly -------------------------------


def test_tuple_shape_classifier() -> None:
    assert _tuple_shape((int, str)) == _TupleShape((int, str), None, (), None)
    assert _tuple_shape(()) == _TupleShape((), None, (), None)
    assert _tuple_shape((int, Ellipsis)) == _TupleShape((), int, (), None)
    assert _tuple_shape((int, U[Ts], str)) == _TupleShape(
        (int,), tx.Any, (str,), Ts
    )
    assert _tuple_shape((U[Ts],)) == _TupleShape((), tx.Any, (), Ts)
    # A fixed unpacked tuple flattens; an unbounded one opens.
    assert _tuple_shape((int, U[T[str, int]])) == _TupleShape(
        (int, str, int), None, (), None
    )
    assert _tuple_shape((U[T[str, ...]],)) == _TupleShape((), str, (), None)


def test_tuple_shape_normalises_the_3_8_phantom() -> None:
    # `((),)` -- what `get_args(Tuple[()])` reports on 3.8-3.10 -- means "no
    # elements", not a one-element tuple whose element is `()`.
    assert _tuple_shape(((),)) == _TupleShape((), None, (), None)


def test_match_tuple_closed_sentinel() -> None:
    assert _match_tuple(
        _tuple_shape((int,)), _tuple_shape((int,))
    ) is _CLOSED_TUPLE_MATCH
    assert _match_tuple(_tuple_shape((int,)), _tuple_shape((str,))) is None


def test_match_tuple_captures_the_run() -> None:
    captured = _match_tuple(
        _tuple_shape((int, str, bytes)), _tuple_shape((int, U[Ts]))
    )
    assert captured == _TupleShape((str, bytes), None, (), None)


def test_match_tuple_prefix_covariance_can_fail() -> None:
    # Both open, the fixed prefix element does not satisfy the super's.
    assert issubhint(T[str, U[Ts]], T[int, U[Ts]]) is False


def test_match_tuple_extra_prefix_must_satisfy_the_run() -> None:
    # Both open, an extra fixed prefix element fails the super's run.
    assert issubhint(T[str, U[Ts]], T[int, ...]) is False


def test_match_tuple_extra_suffix_must_satisfy_the_run() -> None:
    # Both open with a concrete run and a suffix: an extra leading suffix
    # element must satisfy the run. Built as shapes directly (`Tuple[int, ...,
    # X]` degenerate forms are awkward to spell).
    sub = _TupleShape((), int, (str, bytes), None)
    sup = _TupleShape((), int, (bytes,), None)
    assert _match_tuple(sub, sup) is None
    # ... and when the extra leading suffix does satisfy it, the match holds.
    ok_sub = _TupleShape((), int, (int, bytes), None)
    assert _match_tuple(ok_sub, sup) is not None


def test_tuple_shape_unpacked_unmodelled_form_warns() -> None:
    # `Tuple[int, *Unpack[TypedDict]]` -- an unpack of something that is not a
    # `TypeVarTuple` nor a tuple -- degrades its run to `Any` and warns once.
    from bagof.dispatchers.core import _relation

    class _Opts(tx.TypedDict):
        a: int

    element = U[_Opts]
    _relation._WARNED_UNKNOWN.discard(_relation._warn_key(element))
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(UnknownHintWarning):
            shape = _tuple_shape((int, element))
    assert shape == _TupleShape((int,), tx.Any, (), None)


# --- single-unpack enforcement -----------------------------------------


def test_two_open_runs_in_one_tuple_is_rejected() -> None:
    # PEP 646 allows a single unpack; typing does not reject a second at
    # runtime, so the classifier does.
    with pytest.raises(TypeError):
        _tuple_shape((U[Ts], U[Us]))
    with pytest.raises(TypeError):
        _tuple_shape((U[Ts], Ellipsis))


def test_leading_ellipsis_with_no_element_is_rejected() -> None:
    # A `...` run needs a preceding element to repeat.
    with pytest.raises(TypeError):
        _tuple_shape((Ellipsis, int))


# --- Callable[[int, *Ts], R] rides the open-tail (8c) path -------------


def test_callable_unpack_tail_chain() -> None:
    # `Callable[[int, str], R] < Callable[[int, *Ts], R] == Callable[
    #  Concatenate[int, P], R] < Callable[..., R]`.
    fixed = C[[int, str], int]
    tail = C[[int, U[Ts]], int]
    concat = C[tx.Concatenate[int, P], int]
    top = C[..., int]
    assert issubhint(fixed, tail) is True
    assert issubhint(tail, fixed) is False
    assert issubhint(tail, concat) is True and issubhint(concat, tail) is True
    assert issubhint(tail, top) is True
    assert issubhint(top, tail) is False
    # The committed prefix stays contravariant: `int` is not below `bool`, so a
    # fixed `[bool]` list is not a sub-hint of the open `[int, *Ts]`.
    assert issubhint(C[[bool], int], C[[int, U[Ts]], int]) is False
    # ... but `[int]` is (a callable taking `int` stands in for one taking
    # `bool` and more).
    assert issubhint(C[[int], int], C[[bool, U[Ts]], int]) is True


def test_callable_middle_unpack_with_suffix_degrades() -> None:
    # `Callable[[int, *Ts, str], R]` degrades to an open `Concatenate[int,...]`
    # shape (the suffix dropped) -- so a fixed `[int]` list is a sub-hint of
    # it, and it equals `Concatenate[int, P]`.
    mid = C[[int, U[Ts], str], int]
    assert issubhint(C[[int], int], mid) is True
    assert issubhint(mid, C[[int], int]) is False
    concat = C[tx.Concatenate[int, P], int]
    assert issubhint(mid, concat) is True and issubhint(concat, mid) is True


# --- Tuple[()] phantom, pre-3.11 only ----------------------------------


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="Tuple[()] reports empty args on 3.11+ (#36), not the phantom",
)
def test_empty_tuple_phantom_orders_below_a_run() -> None:
    # On 3.8-3.10 `Tuple[()]` reports `((),)`; normalised to no elements, it is
    # the empty tuple -- a valid zero-length run.
    assert issubhint(T[()], T[U[Ts]]) is True
    assert issubhint(T[()], T[int, ...]) is True     # zero ints
    assert issubhint(T[()], T[int, U[Ts]]) is False  # needs one element


# --- Tuple[()] vs bare tuple, every interpreter (#36) -------------------

# `Tuple[()]` is the empty-tuple type. Its arguments read as the phantom
# `((),)` on 3.8-3.10 and as genuinely empty `()` on 3.11+; the empty form
# used to be mistaken for a bare, unparametrised `Tuple`/`tuple`, so every
# tuple hint was wrongly judged a sub-hint of `Tuple[()]`. These rows must
# hold on every interpreter.
EMPTY_TUPLE_TABLE = [
    (T[int], T[()], False),        # a 1-tuple is not the empty tuple
    (T[()], T[()], True),          # reflexive
    (T[()], tuple, True),          # the empty tuple is a tuple
    (T[()], T, True),              # ... and a sub-hint of bare `Tuple`
    (tuple, T[()], False),         # a bare tuple may hold anything
    (T, T[()], False),
    (T[()], T[int], False),        # the empty tuple has no first element
    (T[()], T[U[Ts]], True),       # a valid zero-length run
    (T[()], T[int, ...], True),    # zero ints
    (T[()], T[int, U[Ts]], False),  # needs one element
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    EMPTY_TUPLE_TABLE,
    ids=[f"{h}<:{s}" for h, s, _ in EMPTY_TUPLE_TABLE],
)
def test_empty_tuple_relation(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_bare_tuple_still_accepts_any_parametrisation() -> None:
    # The fix must not disturb a bare `Tuple`/`tuple` superhint: it constrains
    # nothing, so any tuple parametrisation is a sub-hint of it.
    assert issubhint(T[int], T) is True
    assert issubhint(T[int], tuple) is True
    assert issubhint(T[int, str], tuple) is True
    assert issubhint(T[int, ...], T) is True
    assert issubhint(T[U[Ts]], tuple) is True


# --- 3.11+ star syntax parity ------------------------------------------


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="`*Ts` star syntax is a SyntaxError below 3.11",
)
def test_star_syntax_orders_identically() -> None:
    # The `*Ts` star syntax must order exactly as `Unpack[Ts]`. Built through
    # `exec` so the module still parses on 3.8.
    namespace = {"tx": tx, "Ts": Ts, "T": T}
    exec(
        "star = T[int, *Ts]\n"
        "star_open = T[*Ts]\n",
        namespace,
    )
    star = namespace["star"]
    star_open = namespace["star_open"]
    assert issubhint(T[int, str], star) is True
    assert issubhint(T[int], star) is True
    assert issubhint(star, star_open) is True
    assert issubhint(star_open, star) is False
    # It orders the same as the `Unpack[Ts]` spelling.
    assert issubhint(star, T[int, U[Ts]]) is True
    assert issubhint(T[int, U[Ts]], star) is True
