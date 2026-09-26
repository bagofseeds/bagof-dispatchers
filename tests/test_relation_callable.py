"""Tests for `Callable` variance in `issubhint`."""

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers.core import ishintstance, issubhint

C = tx.Callable

# (hint, superhint, expected)
CALLABLE_CASES = [
    # Parameters are contravariant: a callable taking a wider argument can
    # stand in for one taking a narrower argument.
    (C[[int], str], C[[bool], str], True),
    (C[[bool], str], C[[int], str], False),
    (C[[int], str], C[[int], str], True),
    # The return type is covariant.
    (C[[int], bool], C[[int], int], True),
    (C[[int], int], C[[int], bool], False),
    # Both at once.
    (C[[int], bool], C[[bool], int], True),
    (C[[bool], int], C[[int], bool], False),
    # `Callable[..., R]` is the *top* of parameter lists: a fixed list is a
    # sub-hint of it, but it is not a sub-hint of any fixed list (RFC 11.1,
    # the transitivity fix -- an open list may be called with arguments a
    # fixed one does not accept).
    (C[[int], str], C[..., str], True),
    (C[..., str], C[[int], str], False),
    (C[..., bool], C[..., int], True),
    (C[..., int], C[..., bool], False),
    # Arity must match for fixed lists.
    (C[[int, str], bool], C[[int], bool], False),
    (C[[int], bool], C[[int, str], bool], False),
    (C[[int, str], bool], C[[int, str], bool], True),
    # A bare `Callable` constrains nothing; a parametrised one is not a
    # sub-hint of a bare callable's... other way round.
    (C[[int], str], C, True),
    (C, C[[int], str], False),
    # A callable is a plain object.
    (C[[int], str], object, True),
    # A non-callable hint is not a sub-hint of a Callable.
    (int, C[[int], str], False),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    CALLABLE_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in CALLABLE_CASES],
)
def test_issubhint_callable(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_paramspec_and_ellipsis_are_the_top() -> None:
    # A bare `ParamSpec` parameter list and `...` are both the *top* of
    # parameter lists (RFC 11.1): `Callable[P, R] ≡ Callable[..., R]`.
    P = tx.ParamSpec("P")
    Q = tx.ParamSpec("Q")
    assert issubhint(C[P, int], C[Q, int]) is True
    assert issubhint(C[P, int], C[..., int]) is True
    assert issubhint(C[..., int], C[P, int]) is True
    # The top is *not* a sub-hint of a fixed list, but a fixed list is a
    # sub-hint of the top.
    assert issubhint(C[P, int], C[[int], int]) is False
    assert issubhint(C[[int], int], C[P, int]) is True
    # The return type stays covariant even with a ParamSpec parameter list.
    assert issubhint(C[P, bool], C[P, int]) is True
    assert issubhint(C[P, int], C[P, bool]) is False
    # `Concatenate[X, P]` is a contravariant fixed prefix followed by an open
    # tail: it sits between the fixed lists (below) and the top (above).
    concat = C[tx.Concatenate[int, P], str]
    # A fixed list of matching arity is a sub-hint of the open prefix, its
    # prefix compared contravariantly.
    assert issubhint(C[[int], str], concat) is True
    assert issubhint(C[[bool], str], concat) is False  # int not <: bool
    assert issubhint(C[[int], str], C[tx.Concatenate[bool, P], str]) is True
    # An open prefix is not a sub-hint of a fixed list (it may be called with
    # more arguments than the fixed list accepts).
    assert issubhint(concat, C[[int], str]) is False
    # Two open prefixes: contravariant, and the longer prefix is the more
    # specific (a sub-hint of the shorter).
    assert issubhint(concat, C[tx.Concatenate[bool, P], str]) is True
    assert issubhint(C[tx.Concatenate[bool, P], str], concat) is False
    # Every open list is a sub-hint of the top.
    assert issubhint(concat, C[..., str]) is True
    assert issubhint(C[..., str], concat) is False


def test_a_function_value_is_a_callable_instance() -> None:
    assert ishintstance(print, C[[int], str]) is True
    assert ishintstance(42, C[[int], str]) is False


def test_a_callable_class_is_a_callable() -> None:
    # R2: a callable *class* -- `type`, a function type, `Type[C]`, or a class
    # with `__call__` -- is a sub-hint of a bare `Callable`.
    import types

    class Caller:
        def __call__(self) -> int:
            return 0

    assert issubhint(type, C) is True
    assert issubhint(types.FunctionType, C) is True
    assert issubhint(tx.Type[int], C) is True
    assert issubhint(Caller, C) is True
    # A callable class has no parameter list, so it cannot stand in for a
    # parametrised `Callable[...]`.
    assert issubhint(type, C[[int], str]) is False


def test_callable_param_shape_classifies_paramspec_shapes() -> None:
    # Direct unit test of the shape classifier, which also exercises the
    # <3.10 spellings (a flattened `Concatenate` list, a bare `ParamSpec` as
    # a one-element list, and a `ParamSpec` erased to `[]`) that cannot arise
    # from a real alias on 3.10+ but come out of `typing_extensions` there.
    from bagof.dispatchers.core._relation import (
        _callable_param_shape,
        _ParamShape,
    )

    P = tx.ParamSpec("P")
    assert _callable_param_shape(None, ...) == _ParamShape((), Ellipsis)
    assert _callable_param_shape(None, P) == _ParamShape((), P)
    assert _callable_param_shape(None, [int, str]) == _ParamShape(
        (int, str), None
    )
    # A genuine zero-argument list is closed, not a wildcard.
    assert _callable_param_shape(None, []) == _ParamShape((), None)
    # An unknown shape degrades to the open top rather than raising.
    assert _callable_param_shape(None, object()) == _ParamShape((), Ellipsis)
    # `ParamSpecArgs` / `ParamSpecKwargs` classify as the open top.
    assert _callable_param_shape(None, P.args) == _ParamShape((), Ellipsis)
    assert _callable_param_shape(None, P.kwargs) == _ParamShape((), Ellipsis)
    # <3.10: `Concatenate[X, P]` flattened to `[X, ..., P]` (ParamSpec last).
    assert _callable_param_shape(None, [int, P]) == _ParamShape((int,), P)
    # <3.10 (tx>=4.13): a bare `P` is the one-element list `[P]`.
    assert _callable_param_shape(None, [P]) == _ParamShape((), P)
    # A list ending in `...` (a flattened open tail) is open.
    assert _callable_param_shape(None, [int, ...]) == _ParamShape(
        (int,), Ellipsis
    )
    # <3.10: a bare `P` erased to `[]`, surviving only in `__parameters__`.
    assert _callable_param_shape(C[P, int], []) == _ParamShape((), P)


def test_bare_typeguard_markers_dispatch_as_bool() -> None:
    # The bare (unsubscripted) `TypeGuard` / `TypeIs` markers map to `bool`.
    assert issubhint(tx.TypeGuard, bool) is True
    assert issubhint(tx.TypeIs, bool) is True


def test_concatenate_prefix_edge_cases() -> None:
    P = tx.ParamSpec("P")
    Q = tx.ParamSpec("Q")
    # A fixed-arity sub of matching arity *is* a sub-hint of an open
    # (Concatenate) super (RFC 11.1, the transitivity fix).
    assert issubhint(C[[int], str], C[tx.Concatenate[int, P], str]) is True
    # A fixed list too short for the committed prefix cannot match.
    assert issubhint(C[[int], str], C[tx.Concatenate[int, Q], str]) is True
    assert issubhint(C[[], str], C[tx.Concatenate[int, P], str]) is False
    # Two open prefixes: the *longer* prefix is the more specific, so it is a
    # sub-hint of the shorter -- not the other way round.
    assert (
        issubhint(
            C[tx.Concatenate[int, str, P], bool],
            C[tx.Concatenate[int, Q], bool],
        )
        is True
    )
    assert (
        issubhint(
            C[tx.Concatenate[int, Q], bool],
            C[tx.Concatenate[int, str, P], bool],
        )
        is False
    )
    # `Concatenate[int, ...]` (3.10+ only) equals `Concatenate[int, P]`: both
    # are an open list with the same committed prefix.
    import sys

    if sys.version_info[:2] >= (3, 10):
        assert issubhint(
            C[tx.Concatenate[int, ...], str],
            C[tx.Concatenate[int, P], str],
        ) is True
        assert issubhint(
            C[tx.Concatenate[int, P], str],
            C[tx.Concatenate[int, ...], str],
        ) is True
