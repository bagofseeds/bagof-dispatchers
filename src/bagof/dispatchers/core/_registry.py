"""Looking a hint up in a mapping keyed by other hints.

[`resolve_hint`][] answers a single question: among the keys of a
mapping that are themselves type hints, which one best describes a given
query hint? It replaces `bagof-core-magic`'s older `get_from_registry`,
which scored candidates with an ad hoc notion of type distance; this
lookup instead goes straight through the sub-hint relation,
[`issubhint`][]. The best key is whichever key accepts the query and is
the most specific among the keys that do. If two equally specific keys
both accept the query and the query is a class, the tie is broken by the
query's own MRO, the same refinement RFC 0001 §2.2 step 3 describes for
value dispatch; a tie the MRO cannot settle is left to the caller, who
chooses whether it is raised, warned about, or resolved silently.

A mapping's values may be anything the caller wants to store against a
hint. Its keys are compared through the subtype relation rather than by
equality, so a `#!python Union` key matches any of its members, a
`#!python List[int]` key matches a `#!python List[bool]` query, and an
[`Any`][typing.Any] key acts as a catch-all that every query reaches. A
key that happens not to be hashable still participates in this
comparison-based lookup; it is only ruled out from the identity-based
exact match.

This module is kept inside [`core`][] because `bagof-core-magic` depends
on it directly, so it is written to need nothing beyond the subtype
relation itself, never the dispatch engine built on top of that layer.
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

    A key `#!python K` accepts `hint` when `#!python issubhint(hint, K)`
    holds, meaning the query is a sub-hint of the key and so every value
    the query describes is also described by `K`. The best key among
    those that accept the query is the most specific one, the accepting
    key that is itself a sub-hint of every other accepting key. A key
    equal to the query wins outright regardless of specificity, and an
    [`Exact`][bagof.dispatchers.Exact]`[C]` key is reachable this way
    too, by a query equivalent to `C` itself.

    When several keys are equally specific and the query is a class, the
    tie is broken using the query's MRO, following RFC 0001 §2.2 step 3:
    whichever key's class sits nearest to the query in that MRO wins.
    `#!python {Enum, str}` therefore resolves
    `#!python class Color(str, Enum)` to `str`, and a diamond
    `#!python D(B, C)` resolves `#!python {B, C}` to `B`, in both cases
    independently of the order the keys happened to be registered in. A
    tie the MRO cannot settle, because the query is not a class or
    because the competing keys sit at the same distance in it, is handed
    to the `ambiguity` parameter.

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
        The query hint. A bare `#!python None` is treated as
        `#!python NoneType`, as it is everywhere else in the relation.
    mapping
        The registry to search, keyed by type hint; its values can be
        anything the caller chose to store.
    default
        The value to return when no key accepts the query. Left at its
        default of [`UNSET`][], such a query raises
        [`NoMethodError`][bagof.dispatchers.NoMethodError] instead.
    ambiguity
        What to do when two equally specific keys both accept the query
        and neither the query nor the mapping breaks the tie.
        `#!python "raise"`, the default, raises
        [`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError].
        `#!python "warn"` instead picks whichever tied key comes first in
        the mapping's own order and emits a [`RuntimeWarning`][].
        `#!python "ignore"` makes that same choice without a warning.

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

    In the ordinary case this is simply
    `#!python issubhint(hint, key)`. An
    [`Exact`][bagof.dispatchers.Exact]`[C]` key is given one extra way to
    be reached: a query equivalent to `C` accepts it too. A registration
    made against exact `C` therefore also answers a lookup for plain
    `C`, a convenience this function adds on top of what the relation
    itself grants (RFC 0001 §4).

    A key that turns out not to be a usable type hint, such as a
    malformed entry left behind by another package or a bare string
    forward reference, makes the relation raise a [`TypeError`][]. That
    is treated here as the key simply not accepting the query, so one bad
    key never brings down the whole lookup, though it is not passed over
    in silence either: an [`UnknownHintWarning`][] names it, so a key
    that should have matched does not disappear without a trace. Any
    other exception, such as one raised by a user-defined metaclass's own
    `#!python __subclasscheck__`, is a genuine fault and is allowed to
    propagate.
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

    A [`TypeError`][] from the relation, meaning one of the two keys is
    not a usable hint, is treated as the two keys simply being
    incomparable, so it never escapes this lookup. No second warning is
    raised for it here, since both keys have already passed through
    [`_accepts`][], which warns about an unusable one on its own. Any
    other exception is a genuine fault and is allowed to propagate.
    """
    try:
        na, nb = normalise_hint(a), normalise_hint(b)
        return issubhint(na, nb) and not issubhint(nb, na)
    except TypeError:
        return False


def _mro_nearest(
    hint: tx.Any, best: tx.Sequence[tx.Any]
) -> tx.Optional[tx.List[tx.Any]]:
    """Narrow `best` to whichever keys sit nearest the query in its MRO.

    This carries out the RFC 0001 §2.2 step-3 refinement for a tie
    between equally specific keys: when the query is a class, only the
    keys whose class is the nearest base of the query in that class's MRO
    survive. A single surviving key is a clean win by MRO distance, the
    way `{Enum, str}` resolves to `str` or a diamond `D(B, C)` resolves to
    `B`. More than one surviving key means an irreducible tie between
    duplicate spellings of the same nearest class, such as
    `typing.Sequence` and `collections.abc.Sequence`; the caller resolves
    that remaining tie among just those keys, so a key farther up the MRO
    never wins purely because of registration order.

    A parametrised generic query is read through its origin class, so
    `#!python G[int]` is refined exactly as `G` would be. Doing so keeps
    the answer stable across interpreter versions, since a
    `#!python G[int]` alias satisfies `#!python isinstance(_, type)` on
    3.9 and 3.10 but not from 3.11 onward, and testing that directly
    would make the refinement version-dependent.

    `None` is returned, leaving the whole tie to the caller's
    registration-order fallback, whenever the query is not a class at
    all. The same happens whenever any candidate key names no position
    in the query's MRO, as with a `#!python Union`, a
    `#!python Protocol` or ABC satisfied only through registration, or a
    parametrised generic key.
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
    """Warn that a registry key is not a usable hint and has been skipped."""
    warnings.warn(
        f"registry key {_render(key)!r} is not a usable type hint; it is "
        f"skipped, so the lookup ignores it. Store a real type hint as the "
        f"key, or remove it.",
        UnknownHintWarning,
        stacklevel=2,
    )


def _exact_key(hint: tx.Any, mapping: tx.Mapping[tx.Any, tx.Any]) -> tx.Any:
    """Find the key `hint` is stored under by equality, or return `UNSET`.

    Ordinary equality already does the right thing here: a
    `#!python Union` or `#!python Literal` compares order-insensitively,
    and a `#!python TypeVar` compares by identity, together giving the
    "is this the same hint" test this lookup needs. A new-style generic
    such as `#!python list[int]` is also tried under its `typing`
    spelling, `#!python List[int]`, so either spelling finds the same
    entry. A key that is not hashable is simply treated as not an exact
    key, letting the lookup fall through to the general case rather than
    raising.
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
    """Build a [`NoMethodError`][] for a query that no key accepts."""
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
    """Build an [`AmbiguousMethodError`][] for a tie between accepting keys."""
    from .._errors import AmbiguousMethodError

    listed = ", ".join(_render(key) for key in keys)
    message = (
        f"{_render(hint)} is matched equally well by {listed}, and there is "
        f"nothing to choose between them. Register a more specific key, or "
        f'resolve with ambiguity="warn" to take the first.'
    )
    return AmbiguousMethodError(message, call=hint, candidates=tuple(keys))


def _render(hint: tx.Any) -> str:
    """Render `hint` briefly and readably, for use inside an error message."""
    if isinstance(hint, type):
        return hint.__name__
    text = str(hint)
    return text.replace("typing_extensions.", "").replace("typing.", "")
