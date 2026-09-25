"""Version-pinned typing constants and special-form recognition.

Several typing constructs are classes on some Python versions and not on
others -- [`Any`][typing.Any] became one in 3.11,
[`Annotated`][typing.Annotated] was one through 3.12 but not from 3.13, and
[`Union`][typing.Union] became one in 3.14 -- so
`#!python isinstance(hint, type)` silently gives different answers across the
versions this package supports. The names here pin those answers instead of
inheriting them.
"""

# stdlib
import typing

# dependencies
import typing_extensions as tx

if tx.TYPE_CHECKING:
    from types import NoneType, UnionType
else:
    try:
        from types import NoneType, UnionType
    except ImportError:  # pragma: no cover  -- Python < 3.10
        NoneType = type(None)
        UnionType = tx.Union


UNION_TYPES = (
    (tx.Union,) if UnionType is tx.Union else (tx.Union, UnionType)
)
"""The union spellings this package understands."""


class UnknownHintWarning(RuntimeWarning):
    """A type hint the relation does not recognise is treated as `Any`.

    An unrecognised or future construct is opaque: it is accepted by
    everything as a super-hint and is a sub-hint only of itself and
    [`Any`][typing.Any], so a method annotated with it stays reachable
    rather than silently never firing. The warning is emitted once per
    form.
    """


def spellings(name: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """Every distinct object a typing form is spelled as.

    A construct such as `Unpack` exists on both [`typing`][] and
    `typing_extensions`, and on some versions `#!python typing.X is not
    tx.X`, so an identity check against a single spelling misses the other.
    Returns the objects both modules provide under `name`, without
    duplicates -- ready to test a hint against with `is`.

    !!! example
        ```pycon
        >>> from bagof.dispatchers.core._compat import spellings
        >>> tx.Any in spellings("Any")
        True
        ```
    """
    found = ()  # type: tx.Tuple[tx.Any, ...]
    for module in (tx, typing):
        obj = getattr(module, name, None)
        if obj is not None and not any(obj is each for each in found):
            found += (obj,)
    return found


_SPECIAL_FORMS = (tx.Any, tx.Optional, tx.Literal, tx.Annotated) + UNION_TYPES
"""The typing constructs that must never be treated as classes."""

# The typing markers a structural special-form must *not* swallow: real
# classes a user can subclass or instance-check.
_GENERIC_MARKERS = spellings("Generic")
_PROTOCOL_MARKERS = spellings("Protocol")


def is_special_form(hint: tx.Any) -> bool:
    """Whether a hint is a typing construct rather than a class.

    The explicit [`_SPECIAL_FORMS`][] tuple is the fast path; a structural
    fallback then recognises any construct that lives in `typing` /
    `typing_extensions` and has become a class on this version (or a future
    one), so no such form is mistaken for a subclassable class.
    """
    # Identity, not `in`: `==` on typing objects can be surprising.
    if any(hint is form for form in _SPECIAL_FORMS):
        return True
    return _is_typing_class_form(hint)


def _is_typing_class_form(hint: tx.Any) -> bool:
    """A class living in `typing`/`typing_extensions` that is not a real,
    subclassable class (`Generic`, `Protocol`, `TypedDict`)."""
    if not isinstance(hint, type):
        return False
    module = getattr(hint, "__module__", None)
    if module not in ("typing", "typing_extensions"):
        return False
    if any(hint is marker for marker in _GENERIC_MARKERS):
        return False
    if any(hint is marker for marker in _PROTOCOL_MARKERS):
        return False
    return not is_typeddict_marker(hint)


# `typing.TypedDict` and `typing_extensions.TypedDict` are distinct objects
# on every Python this package supports, and a class built from one never
# mentions the other in its `__orig_bases__`. Both spellings describe the
# same thing, so treat them interchangeably throughout -- `typing` is
# imported for this identity check alone, never for annotations (which go
# through `tx`, per the house style).
_TYPEDDICT_MARKERS = tuple(
    marker
    for marker in (tx.TypedDict, getattr(typing, "TypedDict", None))
    if marker is not None
)


def is_typeddict_marker(cls: tx.Any) -> bool:
    """Whether `cls` is `TypedDict` itself, in either spelling."""
    return any(cls is marker for marker in _TYPEDDICT_MARKERS)


def canonical_typeddict(cls: tx.Any) -> tx.Any:
    """Collapse either `TypedDict` spelling to the canonical one."""
    return tx.TypedDict if is_typeddict_marker(cls) else cls
