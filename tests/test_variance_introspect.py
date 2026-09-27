"""Tests for the variance-introspection helpers.

These cover the additions of #50 phase V1: reading a `TypeVar`'s declared
variance, the per-position variance of a generic's origin, and -- the
load-bearing one -- the pin that keeps the vendored standard-library table
equal to what CPython's `typing` declares. The pin is only meaningful on
Python 3.8, where `typing.List` and friends still expose `__parameters__`;
from 3.9 on the special aliases expose nothing, so it skips there.
"""

# stdlib
import collections
import sys
import typing
from collections import abc

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers.core._introspect import (
    _CONTRAVARIANT,
    _COVARIANT,
    _INVARIANT,
    _STDLIB_VARIANCE,
    _generic_variances,
    _typevar_variance,
)

# --- _typevar_variance -------------------------------------------------


@pytest.mark.parametrize("typevar", [typing.TypeVar, tx.TypeVar])
def test_typevar_variance_flags(typevar: tx.Any) -> None:
    # Both the `typing` and the `typing_extensions` spelling read the same.
    assert _typevar_variance(typevar("T_co", covariant=True)) == _COVARIANT
    assert (
        _typevar_variance(typevar("T_contra", contravariant=True))
        == _CONTRAVARIANT
    )
    assert _typevar_variance(typevar("T")) == _INVARIANT


@pytest.mark.parametrize("typevar", [typing.TypeVar, tx.TypeVar])
def test_typevar_variance_bound_is_invariant(typevar: tx.Any) -> None:
    # A bound says nothing about variance -- an unflagged bounded typevar is
    # invariant, exactly as an unbounded one.
    assert _typevar_variance(typevar("T", bound=int)) == _INVARIANT


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="PEP 695 `class Box[T]` syntax needs Python 3.12+",
)
def test_typevar_variance_infer_variance_is_invariant() -> None:
    # A PEP 695 auto-variance typevar's variance is unknowable at runtime, so
    # it is read as invariant.
    namespace = {}  # type: tx.Dict[str, tx.Any]
    exec("class Box[T]: pass", namespace)
    typevar = namespace["Box"].__type_params__[0]
    assert getattr(typevar, "__infer_variance__", False) is True
    assert _typevar_variance(typevar) == _INVARIANT


# --- _generic_variances: user generics ---------------------------------


def test_generic_variances_user_unflagged_is_invariant() -> None:
    t = tx.TypeVar("t")

    class Box(tx.Generic[t]):
        pass

    assert _generic_variances(Box) == (_INVARIANT,)


def test_generic_variances_user_covariant() -> None:
    t_co = tx.TypeVar("t_co", covariant=True)

    class Box(tx.Generic[t_co]):
        pass

    assert _generic_variances(Box) == (_COVARIANT,)


def test_generic_variances_user_contravariant() -> None:
    t_contra = tx.TypeVar("t_contra", contravariant=True)

    class Box(tx.Generic[t_contra]):
        pass

    assert _generic_variances(Box) == (_CONTRAVARIANT,)


def test_generic_variances_user_two_params_mixed() -> None:
    k = tx.TypeVar("k")
    v_co = tx.TypeVar("v_co", covariant=True)

    class Pair(tx.Generic[k, v_co]):
        pass

    assert _generic_variances(Pair) == (_INVARIANT, _COVARIANT)


def test_generic_variances_paramspec_is_none() -> None:
    # A `ParamSpec` fills a parameter position but carries no variance. On
    # Python 3.8 it is a `TypeVar` subclass, so this guards the exclusion.
    p = tx.ParamSpec("p")

    class Fn(tx.Generic[p]):
        pass

    assert _generic_variances(Fn) is None


def test_generic_variances_typevartuple_is_none() -> None:
    ts = tx.TypeVarTuple("ts")

    class Many(tx.Generic[tx.Unpack[ts]]):
        pass

    assert _generic_variances(Many) is None


def test_generic_variances_non_generic_is_none() -> None:
    # A plain class exposes no `__parameters__`.
    class Plain:
        pass

    assert _generic_variances(Plain) is None
    assert _generic_variances(int) is None


# --- _generic_variances: stdlib origins --------------------------------


@pytest.mark.parametrize(
    "origin, expected",
    [
        (list, (_INVARIANT,)),
        (set, (_INVARIANT,)),
        (frozenset, (_COVARIANT,)),
        (dict, (_INVARIANT, _INVARIANT)),
        (type, (_COVARIANT,)),
        (abc.Sequence, (_COVARIANT,)),
        (abc.Mapping, (_INVARIANT, _COVARIANT)),
        (abc.MutableMapping, (_INVARIANT, _INVARIANT)),
        (abc.Generator, (_COVARIANT, _CONTRAVARIANT, _COVARIANT)),
        (collections.deque, (_INVARIANT,)),
    ],
)
def test_generic_variances_stdlib(
    origin: tx.Any, expected: tx.Tuple[str, ...]
) -> None:
    assert _generic_variances(origin) == expected


# --- the pin: table equals CPython's `typing` --------------------------


# The standard-library generics the table covers, by their `typing` name.
# `Tuple` and `Callable` are excluded: they expose no `__parameters__` and are
# ordered by their own dedicated paths in the relation.
_STDLIB_NAMES = (
    "List", "Dict", "Set", "FrozenSet", "Sequence", "MutableSequence",
    "Mapping", "MutableMapping", "AbstractSet", "MutableSet", "Collection",
    "Container", "Iterable", "Iterator", "Reversible", "KeysView",
    "ValuesView", "ItemsView", "Generator", "Coroutine", "Awaitable",
    "AsyncIterable", "AsyncIterator", "AsyncGenerator", "Deque",
    "DefaultDict", "OrderedDict", "Counter", "ChainMap", "Type",
)


def _live_stdlib_variance() -> tx.Dict[tx.Any, tx.Tuple[str, ...]]:
    """Regenerate the table from the live `typing` module.

    Reads each alias's declared variance off its `__parameters__` and keys
    the result by the runtime origin, exactly as the vendored table is keyed.
    Only meaningful where the aliases still expose `__parameters__` (3.8).
    """
    table = {}  # type: tx.Dict[tx.Any, tx.Tuple[str, ...]]
    for name in _STDLIB_NAMES:
        alias = getattr(typing, name)
        params = alias.__parameters__
        variances = tuple(_typevar_variance(param) for param in params)
        placeholders = tuple(tx.Any for _ in params)
        subscripted = alias[
            placeholders if len(placeholders) != 1 else placeholders[0]
        ]
        origin = tx.get_origin(subscripted)
        table[origin] = variances
    return table


def test_vendored_table_matches_typing() -> None:
    if sys.version_info >= (3, 9):
        pytest.skip("typing aliases expose no __parameters__ on 3.9+")
    # On 3.8 the vendored table must equal what `typing` declares, key for
    # key, so any drift from the spec's reference implementation is caught.
    assert _live_stdlib_variance() == _STDLIB_VARIANCE
