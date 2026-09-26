"""Dispatch-level tests for `TypeVarTuple` / `Unpack` / `*Ts` (Phase 8b).

These exercise `TypeVarTuple` solving through the whole engine: value calls
(shallow tuple matching), hint-level `resolve` (where a repeated `TypeVarTuple`
and a `*args: *Ts` tail are solved jointly), registration rejection of a bare
`Ts` / top-level `Unpack[Ts]` parameter, and the ambiguity behaviour of
incomparable tuple shapes. The relation- and lattice-level unit tests live in
`test_relation_tuple_variadic.py` and `test_lattice.py`.

Every hint is spelled `tx.Unpack[Ts]`, never `*Ts` -- the star syntax is a
`SyntaxError` below 3.11, and this suite runs on 3.8.
"""

# stdlib
import warnings

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers import AmbiguousMethodError, Function
from bagof.dispatchers._signature import Signature

T = tx.Tuple
C = tx.Callable
U = tx.Unpack
Ts = tx.TypeVarTuple("Ts")
Us = tx.TypeVarTuple("Us")
Tv = tx.TypeVar("Tv")
P = tx.ParamSpec("P")


def _quiet(function, *fns):  # noqa: ANN001, ANN002, ANN202
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for fn in fns:
            function.register(fn)


# --- case 1: fixed / variadic / open tuples form a chain ---------------


def _fixed(x: T[int, str]) -> str:
    return "fixed"


def _variadic(x: T[int, U[Ts]]) -> str:
    return "variadic"


def _open(x: T[U[Ts]]) -> str:
    return "open"


@pytest.mark.parametrize(
    "order",
    [
        (_fixed, _variadic, _open),
        (_open, _variadic, _fixed),
        (_variadic, _fixed, _open),
        (_open, _fixed, _variadic),
    ],
)
def test_fixed_variadic_open_chain_is_unambiguous(order: tx.Any) -> None:
    f = Function("f")
    _quiet(f, *order)
    assert f.ambiguities() == []
    # The value level is shallow: every tuple matches all three by
    # `isinstance(v, tuple)`, so the most specific (fixed) always wins whatever
    # the tuple's real length.
    assert f((1, "a")) == "fixed"
    assert f((1, 2, 3)) == "fixed"
    assert f(()) == "fixed"


def test_dropping_fixed_falls_to_variadic() -> None:
    f = Function("f")
    _quiet(f, _variadic, _open)
    assert f.ambiguities() == []
    assert f((1, 2)) == "variadic"


def test_dropping_variadic_falls_to_open() -> None:
    f = Function("f")
    _quiet(f, _open)
    assert f((1, 2)) == "open"


# --- case 2: incomparable tuple shapes are ambiguous, not warned -------


def test_prefix_vs_suffix_is_ambiguous_but_not_warned() -> None:
    f = Function("f")

    def a(x: T[int, U[Ts]]) -> str:
        return "prefix"

    def b(x: T[U[Ts], int]) -> str:
        return "suffix"

    _quiet(f, a, b)  # no registration warning
    assert f.ambiguities() == []
    with pytest.raises(AmbiguousMethodError):
        f((1, 2))


def test_variadic_vs_ellipsis_is_ambiguous_but_not_warned() -> None:
    f = Function("f")

    def a(x: T[int, U[Ts]]) -> str:
        return "star"

    def b(x: T[int, ...]) -> str:
        return "ellipsis"

    _quiet(f, a, b)
    assert f.ambiguities() == []
    with pytest.raises(AmbiguousMethodError):
        f((1, 2))


# --- case 3: joint solving of a repeated TypeVarTuple via resolve ------


def test_repeated_typevartuple_slots() -> None:
    f = Function("f")

    def m(a: T[int, U[Ts]], b: T[str, U[Ts]]) -> str:
        return "m"

    f.register(m)
    # Same captured run at both slots -> applies.
    assert f.resolve(T[int, bool], T[str, bool]).name == "m"
    # A greatest element exists ((int,) & (bool,) -> (int,)) -> applies.
    assert f.resolve(T[int, bool], T[str, int]).name == "m"
    # Incomparable captured runs ((int,) vs (str,)) -> no method.
    assert f.resolve(T[int, int], T[str, str], default=None) is None
    # Different arities have no greatest element -> no method.
    assert (
        f.resolve(T[int, int, int], T[str, int], default=None) is None
    )
    # A closed run and an open run solve to the open one.
    assert f.resolve(T[int, U[Us]], T[str, int]).name == "m"


def test_repeated_typevartuple_distinct_vars_are_independent() -> None:
    f = Function("f")

    def m(a: T[U[Ts]], b: T[U[Us]]) -> str:
        return "m"

    f.register(m)
    # `Ts` and `Us` are solved independently, so nothing has to agree.
    assert f.resolve(T[int], T[str, bytes]).name == "m"
    assert f.resolve(T[int, str], T[bool]).name == "m"


# --- case 4: a Tuple[*Ts] slot and *args: *Ts share the run -----------


def test_tuple_slot_and_star_args_solved_together() -> None:
    f = Function("f")

    def m(t: T[U[Ts]], *args: U[Ts]) -> str:
        return "m"

    f.register(m)
    # The tuple captures (int, str); the two extra positionals are (int, str)
    # too -> consistent.
    assert f.resolve(T[int, str], int, str).name == "m"
    # The tuple captures (int); no extra positionals -> the empty *args run is
    # inconsistent with (int), so the call is rejected (matching mypy).
    assert f.resolve(T[int], default=None) is None
    # The tuple captures (int, str) but *args captures only (int) -> the runs
    # disagree.
    assert f.resolve(T[int, str], int, default=None) is None
    # A longer tuple with matching extras stays consistent.
    assert f.resolve(T[int, str, bytes], int, str, bytes).name == "m"


def test_star_args_ts_alone_accepts_any_positionals() -> None:
    f = Function("f")

    def m(*args: U[Ts]) -> str:
        return "m"

    f.register(m)
    assert f.resolve(int, str, bytes).name == "m"
    assert f.resolve().name == "m"


# --- case 5: nested occurrences apply but are not solved ---------------


def test_nested_typevartuple_applies_and_does_not_crash() -> None:
    f = Function("f")

    def opt(x: tx.Optional[T[U[Ts]]]) -> str:
        return "opt"

    f.register(opt)
    # A nested (non-top-level) `Tuple[*Ts]` is not jointly solved; it still
    # applies and does not crash.
    assert f.resolve(T[int, str]).name == "opt"
    assert f.resolve(type(None)).name == "opt"


# --- case 6: *args: *Ts vs *args -- the deliberate opposite of *args: T -


def test_star_ts_vs_star_is_ambiguous_for_any_count() -> None:
    # `*args: *Ts` does NOT group in the §3 tie-break (an `Unpack[Ts]` is not a
    # `TypeVar`), so it never beats an untyped `*args`, for any count.
    f = Function("f")

    def variadic(*args: U[Ts]) -> str:
        return "ts"

    def plain(*args) -> str:  # noqa: ANN002
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register(variadic)
        f.register(plain)
    for count in (0, 1, 3):
        with pytest.raises(AmbiguousMethodError):
            f(*([1] * count))


def test_star_typevar_vs_star_still_resolves_for_two_or_more() -> None:
    # The contrast pin: `*args: T` DOES group (a `TypeVar`), so it beats an
    # untyped `*args` for 2+ arguments and ties (ambiguous) for 0 or 1.
    f = Function("f")

    def typed(*args: Tv) -> str:
        return "t"

    def plain(*args) -> str:  # noqa: ANN002
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register(typed)
        f.register(plain)
    assert f(1, 2) == "t"
    assert f(1, 2, 3) == "t"
    for count in (0, 1):
        with pytest.raises(AmbiguousMethodError):
            f(*([1] * count))


# --- case 7: registration rejections -----------------------------------


def test_bare_typevartuple_parameter_is_rejected() -> None:
    f = Function("f")

    def bad(x: Ts) -> str:  # a TypeVarTuple is not a value hint
        return "bad"

    with pytest.raises(TypeError):
        f.register(bad)


def test_top_level_unpack_typevartuple_parameter_is_rejected() -> None:
    f = Function("f")

    def bad(x: U[Ts]) -> str:
        return "bad"

    with pytest.raises(TypeError):
        f.register(bad)


def test_bare_typevartuple_on_star_args_is_rejected() -> None:
    f = Function("f")

    def bad(*args: Ts) -> str:  # must be *args: Unpack[Ts]
        return "bad"

    with pytest.raises(TypeError):
        f.register(bad)


def test_two_unpacks_in_one_tuple_is_rejected() -> None:
    f = Function("f")

    def bad(x: T[U[Ts], U[Us]]) -> str:
        return "bad"

    with pytest.raises(TypeError):
        f.register(bad)


def test_two_unpacks_in_a_callable_list_is_rejected() -> None:
    f = Function("f")

    def bad(x: C[[U[Ts], U[Us]], int]) -> str:
        return "bad"

    with pytest.raises(TypeError):
        f.register(bad)


def test_two_unpacks_nested_in_a_callable_list_is_rejected() -> None:
    f = Function("f")

    def bad(x: C[[T[U[Ts], U[Us]]], int]) -> str:
        return "bad"

    with pytest.raises(TypeError):
        f.register(bad)


def test_two_unpacks_nested_in_a_generic_is_rejected() -> None:
    f = Function("f")

    def bad(x: tx.List[T[U[Ts], U[Us]]]) -> str:
        return "bad"

    with pytest.raises(TypeError):
        f.register(bad)


def test_overlay_typevartuple_hint_is_rejected() -> None:
    f = Function("f")

    def impl(x) -> str:  # noqa: ANN001
        return "impl"

    with pytest.raises(TypeError):
        f.register({"x": Ts})(impl)


def test_overlay_star_args_unpack_is_accepted() -> None:
    f = Function("f")

    def impl(*args) -> str:  # noqa: ANN002
        return "impl"

    # `Unpack[Ts]` targeting the `*args` catch-all is valid.
    f.register({"args": U[Ts]})(impl)
    assert f.methods[0].signature.varargs == U[Ts]


def test_deferred_typevartuple_parameter_is_rejected() -> None:
    # A forward reference that only resolves to a `TypeVarTuple` on first use
    # is refused there, the same as one written outright.
    namespace = {}  # type: dict
    exec("def f(x: 'TT'): pass", namespace)
    sig = Signature.from_callable(namespace["f"])
    assert sig._deferred
    namespace["TT"] = tx.TypeVarTuple("TT")
    with pytest.raises(TypeError):
        sig.applies_to_values((1,), {})


# --- case 9: *Ts does not enter the §3 grouping tie-break --------------


def test_typevartuple_does_not_group_in_the_specificity_tiebreak() -> None:
    # A method repeating one `Ts` at two `Tuple` slots does NOT beat one with
    # two distinct `TypeVarTuple`s: the repeated-grouping tie-break (Phase 7)
    # is a `TypeVar` rule and a `TypeVarTuple` is not a `TypeVar`, so the two
    # land equivalent hints and stay ambiguous.
    f = Function("f")

    def repeated(a: T[U[Ts]], b: T[U[Ts]]) -> str:
        return "repeated"

    def distinct(a: T[U[Us]], b: T[U[Ts]]) -> str:
        return "distinct"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register(repeated)
        f.register(distinct)
    with pytest.raises(AmbiguousMethodError):
        f.resolve(T[int], T[int])
