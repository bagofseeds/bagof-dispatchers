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


_SPECIAL_FORMS = (tx.Any, tx.Optional, tx.Literal, tx.Annotated) + UNION_TYPES
"""The typing constructs that must never be treated as classes."""


def is_special_form(hint: tx.Any) -> bool:
    """Whether a hint is a typing construct rather than a class."""
    # Identity, not `in`: `==` on typing objects can be surprising.
    return any(hint is form for form in _SPECIAL_FORMS)


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
