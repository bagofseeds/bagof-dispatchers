"""Dispatch-level tests for `ParamSpec` / `Concatenate` (Phase 8c).

These exercise `ParamSpec` solving through the whole engine: value calls
(shallow `Callable` matching), hint-level `resolve` (where a repeated
`ParamSpec` is solved jointly), registration rejection of a bare `ParamSpec` /
`Concatenate` parameter, and the ambiguity behaviour of incomparable open
prefixes. The relation- and lattice-level unit tests live in
`test_relation_callable.py` and `test_lattice.py`.
"""

# stdlib
import warnings

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers import (
    AmbiguousMethodError,
    Function,
)
from bagof.dispatchers._signature import Signature

C = tx.Callable
Concat = tx.Concatenate
P = tx.ParamSpec("P")
Q = tx.ParamSpec("Q")

# Callable values of different arity; their own signatures are never inspected
# at the value level, so all three match every `Callable[...]` slot.
def _g(x):  # noqa: ANN001, ANN202
    return 0


def _g2(x, y):  # noqa: ANN001, ANN202
    return 0


def _quiet(function, *fns):  # noqa: ANN001, ANN002, ANN202
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for fn in fns:
            function.register(fn)


# --- case 1: fixed / concat / open form a chain ------------------------


def _fixed(fn: C[[int], int]) -> str:
    return "fixed"


def _concat(fn: C[Concat[int, P], int]) -> str:
    return "concat"


def _open(fn: C[P, int]) -> str:
    return "open"


@pytest.mark.parametrize(
    "order",
    [
        (_fixed, _concat, _open),
        (_open, _concat, _fixed),
        (_concat, _fixed, _open),
        (_open, _fixed, _concat),
    ],
)
def test_fixed_concat_open_chain_is_unambiguous(order: tx.Any) -> None:
    f = Function("f")
    # No RuntimeWarning in any registration order, and no listed ambiguity.
    _quiet(f, *order)
    assert f.ambiguities() == []
    # The value level is shallow: every callable matches all three, so the
    # most specific (fixed) always wins, whatever the argument's real arity.
    assert f(_g) == "fixed"
    assert f(_g2) == "fixed"
    assert f(print) == "fixed"


def test_dropping_fixed_falls_to_concat() -> None:
    f = Function("f")
    _quiet(f, _concat, _open)
    assert f.ambiguities() == []
    assert f(_g) == "concat"


def test_dropping_concat_falls_to_open() -> None:
    f = Function("f")
    _quiet(f, _open)
    assert f(_g) == "open"


# --- case 2: prefix contravariance and incomparable prefixes -----------


def test_int_prefix_beats_bool_prefix() -> None:
    f = Function("f")

    def a(fn: C[Concat[int, P], int]) -> str:
        return "int"

    def b(fn: C[Concat[bool, P], int]) -> str:
        return "bool"

    _quiet(f, a, b)
    # `Concatenate[int, P]` is the more specific (contravariant prefix).
    assert f(_g) == "int"


def test_incomparable_prefixes_are_ambiguous_but_not_warned() -> None:
    f = Function("f")

    def a(fn: C[Concat[int, P], int]) -> str:
        return "int"

    def b(fn: C[Concat[str, Q], int]) -> str:
        return "str"

    # Incomparable open prefixes: no registration warning ...
    _quiet(f, a, b)
    assert f.ambiguities() == []
    # ... but a value call matching both has no most specific method.
    with pytest.raises(AmbiguousMethodError):
        f(_g)


def test_longer_prefix_beats_shorter() -> None:
    f = Function("f")

    def a(fn: C[Concat[int, str, P], int]) -> str:
        return "long"

    def b(fn: C[Concat[int, Q], int]) -> str:
        return "short"

    _quiet(f, a, b)
    assert f(_g) == "long"


# --- case 3: `*args: P.args` degrades, and binding decides -------------


def test_paramspec_args_kwargs_degrade_and_bind() -> None:
    f = Function("f")

    def open_m(fn: C[P, int], *args: P.args, **kwargs: P.kwargs) -> str:
        return "open"

    def fixed_m(fn: C[[int], int], x: int) -> str:
        return "fixed"

    _quiet(f, open_m, fixed_m)
    # `*args: P.args` reads as an `Any` tail -- it neither groups nor crashes.
    assert f.methods[0].signature.varargs is tx.Any
    # `f(g, 1)`: both bind; the fixed method is more specific at both slots.
    assert f(_g, 1) == "fixed"
    # `f(g, 1, 2)`: only the `*args` method binds a second positional.
    assert f(_g, 1, 2) == "open"


# --- case 4: joint solving of a repeated ParamSpec via `resolve` -------


def test_repeated_paramspec_bare_slots() -> None:
    f = Function("f")

    def m(a: C[P, int], b: C[P, str]) -> str:
        return "m"

    f.register(m)
    # Same captured list at both slots -> applies.
    assert f.resolve(C[[int], int], C[[int], str]).name == "m"
    # Incomparable captured lists ([int] vs [str]) -> no method.
    assert f.resolve(C[[int], int], C[[str], str], default=None) is None
    # A greatest element exists ([int] & [bool] -> [bool]) -> applies.
    assert f.resolve(C[[int], int], C[[bool], str]).name == "m"
    # Different arities have no greatest element -> no method.
    assert (
        f.resolve(C[[int, str], int], C[[int], str], default=None) is None
    )
    # An open query at both bare-P slots: closed <= open, greatest is the open.
    assert f.resolve(C[Q, int], C[[int], str]).name == "m"


def test_repeated_paramspec_distinct_vars_are_independent() -> None:
    f = Function("f")

    def m(a: C[P, int], b: C[Q, str]) -> str:
        return "m"

    f.register(m)
    # `P` and `Q` are solved independently, so nothing has to agree.
    assert f.resolve(C[[int], int], C[[int], str]).name == "m"
    assert f.resolve(C[[int], int], C[[str], str]).name == "m"
    assert f.resolve(C[[int, str], int], C[[int], str]).name == "m"


def test_repeated_paramspec_with_a_concatenate_slot() -> None:
    f = Function("f")

    def m(a: C[Concat[int, P], int], b: C[P, str]) -> str:
        return "m"

    f.register(m)
    # a captures the leftover `[str]`; b captures `[str]` -> agree.
    assert f.resolve(C[[int, str], int], C[[str], str]).name == "m"
    # a captures `[str]`; b captures `[int, str]` -> disagree.
    assert (
        f.resolve(C[[int, str], int], C[[int, str], str], default=None)
        is None
    )
    # The query does not even match the `Concatenate[int, P]` slot (a `bool`
    # prefix is not contravariantly below `int`).
    assert (
        f.resolve(C[[bool, str], int], C[[str], str], default=None) is None
    )


# --- case 5: nested / union occurrences apply but are not solved -------


def test_nested_paramspec_occurrences_apply_and_do_not_crash() -> None:
    f = Function("f")

    def opt(x: tx.Optional[C[P, int]]) -> str:
        return "opt"

    f.register(opt)
    # A nested (non-top-level) `Callable[P, int]` is not jointly solved; it
    # still applies and does not crash.
    assert f.resolve(C[[int], int]).name == "opt"
    assert f.resolve(type(None)).name == "opt"

    g = Function("g")

    def nested(x: C[P, C[P, int]]) -> str:
        return "nested"

    g.register(nested)
    # The `ParamSpec` in the return is not part of `P`; the outer slot still
    # applies.
    assert g.resolve(C[[int], C[[str], int]]).name == "nested"


# --- case 6: a bare ParamSpec / Concatenate parameter is refused -------


def test_bare_paramspec_parameter_is_rejected() -> None:
    f = Function("f")

    def bad(p: P) -> str:  # a ParamSpec is not a value hint
        return "bad"

    with pytest.raises(TypeError):
        f.register(bad)


def test_concatenate_parameter_is_rejected() -> None:
    f = Function("f")

    def bad(c: Concat[int, P]) -> str:
        return "bad"

    with pytest.raises(TypeError):
        f.register(bad)


def test_overlay_paramspec_hint_is_rejected() -> None:
    f = Function("f")

    def impl(x) -> str:  # noqa: ANN001
        return "impl"

    with pytest.raises(TypeError):
        f.register({"x": P})(impl)


def test_deferred_paramspec_parameter_is_rejected() -> None:
    # A forward reference that only resolves to a `ParamSpec` on first use is
    # refused there, the same as one written outright.
    namespace = {}  # type: dict
    exec("def f(p: 'PS'): pass", namespace)
    sig = Signature.from_callable(namespace["f"])
    assert sig._deferred
    namespace["PS"] = tx.ParamSpec("PS")
    with pytest.raises(TypeError):
        sig.applies_to_values((_g,), {})


# --- case 8: distinct ParamSpecs are distinct methods ------------------


def test_distinct_paramspecs_do_not_replace() -> None:
    f = Function("f")

    def a(fn: C[P, int]) -> str:
        return "p"

    def b(fn: C[Q, int]) -> str:
        return "q"

    f.register(a)
    # `Callable[P, int]` and `Callable[Q, int]` are spelled differently, so
    # the second is a *new* method, not a replacement. The two are equivalent
    # (both the top), so they are ambiguous -- but that is an ambiguity
    # warning, never a replacement warning, and both methods are kept.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f.register(b)
    assert len(f.methods) == 2
    assert not any("replacing" in str(w.message) for w in caught)
    assert not f.methods[0].signature.same_as(f.methods[1].signature)


# --- ParamSpec identity does not enter the §3 grouping tie-break --------


def test_paramspec_does_not_group_in_the_specificity_tiebreak() -> None:
    # A method repeating one `P` at two `Callable` slots does NOT beat one with
    # two distinct `ParamSpec`s: the repeated-grouping tie-break (Phase 7) is
    # a `TypeVar` rule and a `ParamSpec` is not a `TypeVar`, so the two land
    # equivalent hints and stay ambiguous.
    f = Function("f")

    def repeated(a: C[P, int], b: C[P, str]) -> str:
        return "repeated"

    def distinct(a: C[Q, int], b: C[P, str]) -> str:
        return "distinct"

    _quiet_ambiguous = warnings.catch_warnings()
    with _quiet_ambiguous:
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register(repeated)
        f.register(distinct)
    with pytest.raises(AmbiguousMethodError):
        f.resolve(C[[int], int], C[[int], str])
