"""Most-specific single-key lookup over a hint-keyed mapping.

[`resolve_hint`][] answers the question "which entry of this mapping
best describes this hint?" It is the successor to
`bagof-core-magic`'s `get_from_registry`. Where that older lookup
summed a made-up type distance, this one uses the sub-hint relation,
[`issubhint`][], directly: the best entry is the mapping key that
accepts the query and is the most specific key that does. When two
equally specific keys both accept the query, and the query is a class,
the tie is broken by the query's MRO, the same refinement described in
RFC 0001 §2.2 step 3 that the value-dispatch path uses. A tie that the
MRO cannot break is ambiguous, and the caller chooses how to handle it:
raise, warn, or ignore.

The mapping's values are whatever the caller stored; its keys are type
hints. Keys are compared through the relation rather than by equality,
so a `#!python Union` key matches any of its members, a
`#!python List[int]` key matches a `#!python List[bool]` query, and an
[`Any`][typing.Any] key is the catch-all that every query reaches. A
key that cannot be hashed still works; it is simply never chosen as an
exact match.

This module stays inside [`core`][], because `bagof-core-magic` reuses
it directly, so it depends only on the relation and never on the
dispatch engine built on top of it.
"""

# stdlib
import warnings

# dependencies
import typing_extensions as tx

# local
from ._compat import UnknownHintWarning
from ._exact import exact_target, is_exact
from ._introspect import (
    _looks_like_class,
    _typing_spelling,
    get_origin_uw,
    mro_index,
    normalise_hint,
)
from ._relation import issubhint
from ._sentinels import UNSET

__all__ = ["resolve_hint"]


def resolve_hint(
    hint: tx.Any,
    mapping: tx.Mapping[tx.Any, tx.Any],
    *,
    default: tx.Any = UNSET,
    ambiguity: str = "raise",
) -> tx.Any:
    """Return the value stored under the most specific key that accepts `hint`.

    A key `#!python K` accepts `hint` when
    `#!python issubhint(hint, K)` holds: the query is a sub-hint of the
    key, so any value described by `hint` is also described by `K`.
    Among the accepting keys, the best one is the most specific, the
    key that is itself a sub-hint of every other accepting key. An
    exact match, where the query equals a key, always wins outright,
    and an [`Exact`][bagof.dispatchers.Exact]`[C]` key is additionally
    reachable by a query equivalent to `C`.

    When several keys are equally specific and the query is a class,
    the tie is broken by the query's MRO, following RFC 0001 §2.2 step
    3: the key whose class is the nearest base of the query wins. So
    `#!python {Enum, str}` resolves `#!python class Color(str, Enum)`
    to `str`, and a diamond `#!python D(B, C)` resolves
    `#!python {B, C}` to `B`, in both cases regardless of the order the
    keys were registered in. A tie the MRO cannot break, because the
    query is not a class or because the class keys are equidistant in
    the MRO, falls to the `ambiguity` parameter.

    !!! example
        ```pycon
        >>> registry = {int: "number", object: "any"}
        >>> resolve_hint(bool, registry)
        'number'
        >>> resolve_hint(str, registry)
        'any'
        >>> resolve_hint(list, {int: "n"}, default="?")
        '?'
        ```

    Parameters
    ----------
    hint
        The query hint. A bare `#!python None` is read as
        `#!python NoneType`, matching the rest of the relation.
    mapping
        The registry, keyed by type hint. Values are arbitrary.
    default
        Returned when no key accepts the query. When left at
        [`UNSET`][], a [`NoMethodError`][bagof.dispatchers.NoMethodError]
        is raised instead.
    ambiguity
        What to do when two equally specific keys both accept the
        query. `#!python "raise"`, the default, raises
        [`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError].
        `#!python "warn"` picks the first such key in the mapping's
        order and emits a [`RuntimeWarning`][]. `#!python "ignore"`
        picks it silently.

    Returns
    -------
    Any
        The stored value, or `default` when nothing accepts the query.
    """
    hint = normalise_hint(hint)

    # An exact key -- the query written as one of the keys -- always wins,
    # before any relation is read. This is the only way a `Union`, `Literal`
    # or `TypeVar` key (which the relation compares by structure, not by an
    # ordinary specificity) is reachable by an identical query.
    exact = _exact_key(hint, mapping)
    if exact is not UNSET:
        return mapping[exact]

    accepting = [key for key in mapping if _accepts(hint, key)]
    if not accepting:
        if default is not UNSET:
            return default
        raise _no_key(hint, mapping)

    best = [
        key
        for key in accepting
        if not any(
            other is not key and _strictly_below(other, key)
            for other in accepting
        )
    ]
    if len(best) == 1:
        return mapping[best[0]]

    # Two or more equally specific keys accept the query. When the query is a
    # class, refine by argument MRO (RFC 0001 §2.2 step 3): keep only the keys
    # nearest in the query's MRO -- so `{Enum, str}` picks `str` for
    # `class Color(str, Enum)`, and the diamond `D(B, C)` picks `B`, both
    # order-independently, matching the value-dispatch tie-break. When several
    # keys tie at that nearest position (duplicate spellings of one class), the
    # tie is restricted to just them, so a farther key never wins on order.
    nearest = _mro_nearest(hint, best)
    if nearest is not None:
        best = nearest
        if len(best) == 1:
            return mapping[best[0]]

    # Nothing separates them -- a non-class query, or several equidistant keys.
    # Keep the mapping's own order so the choice is repeatable.
    winner = next(key for key in mapping if key in best)
    if ambiguity == "raise":
        raise _ambiguous_keys(hint, best)
    if ambiguity == "warn":
        warnings.warn(
            f"{_render(hint)} matches "
            + ", ".join(_render(key) for key in best)
            + " equally well; using the first that was registered. Add a more "
            "specific key to disambiguate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return mapping[winner]


def _accepts(hint: tx.Any, key: tx.Any) -> bool:
    """Report whether registry `key` describes every value that `hint` does.

    Ordinarily this is just `#!python issubhint(hint, key)`. An
    [`Exact`][bagof.dispatchers.Exact]`[C]` key is also reachable by a
    query equivalent to `C`: an exact-`C` registration answers a
    plain-`C` lookup too, a lookup convenience that the relation itself
    does not grant (RFC 0001 §4).

    A key that is not a usable type hint, such as an exotic or
    malformed entry one of the sibling bags happens to have registered,
    or a bare string forward reference, makes the relation raise a
    [`TypeError`][]. That is read here as "does not accept", so one bad
    key never fails the whole lookup. It is not swallowed silently,
    though: an [`UnknownHintWarning`][] names the key, so a key that
    should have matched is not lost without trace. Any other error,
    such as a user metaclass whose `#!python __subclasscheck__` raises,
    is a genuine fault and is left to propagate.
    """
    try:
        target = normalise_hint(key)
        if issubhint(hint, target):
            return True
        if is_exact(target):
            inner = normalise_hint(exact_target(target))
            return issubhint(hint, inner) and issubhint(inner, hint)
        return False
    except TypeError:
        _warn_unusable_key(key)
        return False


def _strictly_below(a: tx.Any, b: tx.Any) -> bool:
    """Report whether key `a` is strictly more specific than key `b`.

    A [`TypeError`][] raised by the relation, meaning one of the keys
    is not a usable hint, is read here as "not comparable", so it never
    propagates out of the lookup. The keys compared here have already
    passed through [`_accepts`][], which warns about a bad one, so no
    second warning is emitted. Any other error is a genuine fault and
    is left to propagate.
    """
    try:
        na, nb = normalise_hint(a), normalise_hint(b)
        return issubhint(na, nb) and not issubhint(nb, na)
    except TypeError:
        return False


def _mro_nearest(
    hint: tx.Any, best: tx.Sequence[tx.Any]
) -> tx.Optional[tx.List[tx.Any]]:
    """Return the nearest-MRO subset of `best` for a class query, or `None`.

    This applies the RFC 0001 §2.2 step-3 refinement to a tie between
    equally specific keys: when the query is a class, only the keys
    whose class is the nearest base of the query in its MRO are kept.
    A single-key result is a clean MRO win, such as `{Enum, str}`
    resolving to `str` or the diamond `D(B, C)` resolving to `B`. A
    multi-key result is an irreducible tie between duplicate spellings
    of the nearest class, such as `typing.Sequence` and
    `collections.abc.Sequence`, which the caller then resolves among
    those keys alone, so a farther key never wins simply because of
    registration order.

    A parametrised generic query is read as its origin class, so
    `#!python G[int]` refines the same way `G` does. This keeps the
    answer independent of the interpreter version: a `#!python G[int]`
    alias satisfies `#!python isinstance(_, type)` on 3.9 and 3.10 but
    not from 3.11 on, so testing that directly would refine
    inconsistently across versions.

    The return value is `#!python None`, leaving the whole tie to the
    caller's order fallback, when the query is not a class, or when any
    candidate names no position in the query's MRO: a
    `#!python Union`, a `#!python Protocol` or an ABC satisfied only by
    registration, or a parametrised generic key.
    """
    query = hint
    if not _looks_like_class(query):
        origin = get_origin_uw(query)
        if not _looks_like_class(origin):
            return None
        query = origin
    ranked = []
    for key in best:
        index = mro_index(key, query)
        if index is None:
            return None
        ranked.append((index, key))
    nearest = min(index for index, _ in ranked)
    return [key for index, key in ranked if index == nearest]


def _warn_unusable_key(key: tx.Any) -> None:
    """Warn that a registry key is not a usable hint and was skipped."""
    warnings.warn(
        f"registry key {_render(key)!r} is not a usable type hint; it is "
        f"skipped, so the lookup ignores it. Store a real type hint as the "
        f"key, or remove it.",
        UnknownHintWarning,
        stacklevel=2,
    )


def _exact_key(hint: tx.Any, mapping: tx.Mapping[tx.Any, tx.Any]) -> tx.Any:
    """Return the key `hint` is stored under by equality, or `UNSET`.

    A `#!python Union` or `#!python Literal` compares order-insensitively
    and a `#!python TypeVar` compares by identity, which together give
    exactly the "same hint" test this needs. A new-style generic such
    as `#!python list[int]` is also tried in its `typing` spelling,
    `#!python List[int]`. A key that cannot be hashed is simply not an
    exact key, so the lookup falls through to the general case rather
    than raising.
    """
    for candidate in (hint, _typing_spelling(hint)):
        try:
            if candidate in mapping:
                return candidate
        except TypeError:
            pass
    return UNSET


def _no_key(
    hint: tx.Any, mapping: tx.Mapping[tx.Any, tx.Any]
) -> tx.Any:
    """Build a [`NoMethodError`][] for a query nothing in the mapping
    accepts.
    """
    # Imported here, not at module load: the error type lives in the dispatch
    # layer above `core`, and `core` must import from nothing above it.
    from .._errors import NoMethodError

    if not mapping:
        message = (
            f"no entry matching {_render(hint)}: the registry is empty."
        )
    else:
        message = (
            f"no entry matching {_render(hint)}. None of the "
            f"{len(mapping)} registered keys describes it."
        )
    return NoMethodError(message, call=hint, candidates=tuple(mapping))


def _ambiguous_keys(hint: tx.Any, keys: tx.Sequence[tx.Any]) -> tx.Any:
    """Build an [`AmbiguousMethodError`][] for equally specific
    accepting keys.
    """
    from .._errors import AmbiguousMethodError

    listed = ", ".join(_render(key) for key in keys)
    message = (
        f"{_render(hint)} is matched equally well by {listed}, and there is "
        f"nothing to choose between them. Register a more specific key, or "
        f'resolve with ambiguity="warn" to take the first.'
    )
    return AmbiguousMethodError(message, call=hint, candidates=tuple(keys))


def _render(hint: tx.Any) -> str:
    """Return a short, readable spelling of a hint for use in a message."""
    if isinstance(hint, type):
        return hint.__name__
    text = str(hint)
    return text.replace("typing_extensions.", "").replace("typing.", "")
