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
import abc
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


# Every spelling of the forms the relation reads by identity: on 3.8-3.10
# `typing_extensions` ships its own `Any`/`Literal`, distinct from `typing`'s,
# so a single-object check misses the other spelling.
_ANY_FORMS = spellings("Any")
_LITERAL_FORMS = spellings("Literal")
_ANNOTATED_FORMS = spellings("Annotated")
_OPTIONAL_FORMS = spellings("Optional")

_SPECIAL_FORMS = (
    _ANY_FORMS
    + _OPTIONAL_FORMS
    + _LITERAL_FORMS
    + _ANNOTATED_FORMS
    + UNION_TYPES
)
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
    if is_typeddict_marker(hint):
        return False
    # Real, checkable classes that merely live in `typing` -- a Protocol
    # (`SupportsInt`, `SupportsIndex`, ...), any `Generic` subclass
    # (`typing.IO`), or an ABC (`Buffer` on <=3.11) -- are not special
    # forms: they can be subclass/instance-checked, so swallowing them
    # would make the relation silently reject them.
    if getattr(hint, "_is_protocol", False):
        return False
    mro = getattr(hint, "__mro__", ())
    if any(marker in mro for marker in _GENERIC_MARKERS):
        return False
    if isinstance(hint, abc.ABCMeta):
        return False
    return True


# The `TypeVar` family: objects that are hints in a signature but not classes.
_TYPEVAR_FAMILY = tuple(
    form
    for name in ("TypeVar", "ParamSpec", "ParamSpecArgs", "ParamSpecKwargs",
                 "TypeVarTuple")
    for form in (getattr(tx, name, None),)
    if isinstance(form, type)
)


def is_plausible_hint(obj: tx.Any) -> bool:
    """Whether `obj` could be a type hint, rather than an obvious non-hint.

    A hint is a class, a typing special form, a member of the
    [`TypeVar`][typing.TypeVar] family, a [`ForwardRef`][typing.ForwardRef],
    or any object defined in `typing` / `typing_extensions` -- which covers a
    future construct of that shape too. An obvious non-hint -- a plain value
    such as a number, string or container instance, or a plain function -- is
    not, and the relation raises rather than treating it as `Any`.
    """
    if isinstance(obj, type):
        return True
    if is_special_form(obj):
        return True
    if _TYPEVAR_FAMILY and isinstance(obj, _TYPEVAR_FAMILY):
        return True
    forward_ref = getattr(tx, "ForwardRef", None)
    if isinstance(forward_ref, type) and isinstance(obj, forward_ref):
        return True
    module = getattr(obj, "__module__", None)
    return module in ("typing", "typing_extensions")


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
