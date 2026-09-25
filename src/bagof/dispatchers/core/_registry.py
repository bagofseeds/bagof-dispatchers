"""Most-specific single-key lookup over a hint-keyed mapping.

[`resolve_hint`][bagof.dispatchers.core.resolve_hint] answers "which entry of
this mapping best describes this hint?" -- the successor to
`bagof-core-magic`'s `get_from_registry`. Where the old lookup summed a made-up
type distance, this one uses the sub-hint relation
([`issubhint`][bagof.dispatchers.core.issubhint]) directly: the best entry is
the mapping key that accepts the query and is the most specific such key. When
two equally specific keys both accept the query the lookup is ambiguous, and
the caller chooses -- raise, warn or ignore.

The mapping's *values* are whatever the caller stored; the key is a type hint.
Keys are compared by the relation, so a `#!python Union` key matches any of its
members, a `#!python List[int]` key matches a `#!python List[bool]` query, and
an [`Any`][typing.Any] key is the catch-all every query reaches. A key that
cannot be hashed still works -- it is simply not an exact-match key.

This module stays inside [`core`][bagof.dispatchers.core]: it is what
`bagof-core-magic` reuses, so it depends only on the relation and never on the
dispatch engine.
"""

# stdlib
import warnings

# dependencies
import typing_extensions as tx

# local
from ._compat import UnknownHintWarning
from ._exact import exact_target, is_exact
from ._introspect import _typing_spelling, normalise_hint
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
    """The value the most-specific key that accepts `hint` is stored under.

    A key `#!python K` *accepts* `hint` when `#!python issubhint(hint, K)` --
    the query is a sub-hint of the key, so a value described by `hint` is also
    described by `K`. Among the accepting keys, the best is the most specific
    one: the key that is a sub-hint of every other accepting key. An exact
    match (the query is a key, by equality) always wins, and a
    [`Exact`][bagof.dispatchers.Exact]`[C]` key is additionally reachable by a
    query equivalent to `C`.

    Parameters
    ----------
    hint
        The query hint. A bare `#!python None` is read as
        `#!python NoneType`, matching the rest of the relation.
    mapping
        The registry, keyed by type hint. Values are arbitrary.
    default
        Returned when no key accepts the query. When left at
        [`UNSET`][bagof.dispatchers.core.UNSET] a
        [`NoMethodError`][bagof.dispatchers.NoMethodError] is raised instead.
    ambiguity
        What to do when two equally specific keys both accept the query:
        `#!python "raise"` (the default) raises
        [`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError];
        `#!python "warn"` picks the first such key in the mapping's order and
        emits a [`RuntimeWarning`][]; `#!python "ignore"` picks it silently.

    Returns
    -------
    Any
        The stored value, or `default` when nothing accepts the query.

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

    # Two or more equally specific keys accept the query and nothing separates
    # them. Keep the mapping's own order so the choice is repeatable.
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
    """Whether the registry `key` describes every value the query describes.

    Ordinarily this is `#!python issubhint(hint, key)`. A
    [`Exact`][bagof.dispatchers.Exact]`[C]` key is also reachable by a query
    equivalent to `C`: an exact-`C` registration answers a plain-`C` lookup,
    the lookup convenience the relation itself does not grant (RFC 0001 §4).

    A key that is not a usable type hint -- an exotic or malformed entry one of
    the sibling bags happens to have registered, or a bare string forward
    reference -- makes the relation raise a [`TypeError`][], which is read as
    "does not accept" so one bad key never fails the whole lookup. It is not
    swallowed silently, though: an [`UnknownHintWarning`][] names the key, so a
    key that should have matched is not lost without trace. Any other error --
    a user metaclass whose `#!python __subclasscheck__` raises, say -- is a
    genuine fault and is left to propagate.
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
    """Whether key `a` is strictly more specific than key `b`.

    A [`TypeError`][] from the relation -- a key that is not a usable hint --
    is read as "not comparable", so it never propagates out of the lookup. The
    keys compared here have already been accepted by
    [`_accepts`][bagof.dispatchers.core._registry._accepts] (which warns for a
    bad one), so no second warning is emitted. Any other error is a genuine
    fault and is left to propagate.
    """
    try:
        na, nb = normalise_hint(a), normalise_hint(b)
        return issubhint(na, nb) and not issubhint(nb, na)
    except TypeError:
        return False


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
    """The key `hint` is stored under by equality, or `UNSET`.

    A `#!python Union` / `#!python Literal` compares order-insensitively and a
    `#!python TypeVar` by identity, which is exactly the "same hint" test. A
    new-style generic (`#!python list[int]`) is also tried in its `typing`
    spelling (`#!python List[int]`). A key that cannot be hashed is simply not
    an exact key, so the lookup falls through rather than raising.
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
    """A [`NoMethodError`][] for a query nothing in the mapping accepts."""
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
    """An [`AmbiguousMethodError`][] for equally specific accepting keys."""
    from .._errors import AmbiguousMethodError

    listed = ", ".join(_render(key) for key in keys)
    message = (
        f"{_render(hint)} is matched equally well by {listed}, and there is "
        f"nothing to choose between them. Register a more specific key, or "
        f'resolve with ambiguity="warn" to take the first.'
    )
    return AmbiguousMethodError(message, call=hint, candidates=tuple(keys))


def _render(hint: tx.Any) -> str:
    """A short, readable spelling of a hint for a message."""
    if isinstance(hint, type):
        return hint.__name__
    text = str(hint)
    return text.replace("typing_extensions.", "").replace("typing.", "")
