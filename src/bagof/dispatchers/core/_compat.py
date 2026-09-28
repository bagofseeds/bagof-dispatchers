"""Version-pinned typing constants and special-form recognition.

Several typing constructs are classes on some Python versions and plain
objects on others. [`Any`][typing.Any] became a class in 3.11,
[`Annotated`][typing.Annotated] was one through 3.12 and stopped being
one from 3.13, and [`Union`][typing.Union] became one in 3.14. Because
of this, `#!python isinstance(hint, type)` gives a different answer for
the same construct depending on which Python version is running. The
names defined here pin the answer that matters, a special form or not,
instead of letting it drift with the interpreter.
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
    """Warns that an unrecognised type hint is being treated as `Any`.

    The subtype relation treats a construct it does not recognise, such
    as one introduced by a future Python version, as opaque: every hint
    is accepted as a super-hint of it, and it is a sub-hint only of
    itself and of [`Any`][typing.Any]. This keeps a method annotated
    with the unrecognised construct reachable, rather than letting it
    silently never fire. The warning is raised once per distinct
    construct.
    """


def spellings(name: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """Return every distinct object a typing form is spelled as.

    A construct such as `Unpack` exists on both [`typing`][] and
    `typing_extensions`, and on some Python versions
    `#!python typing.X is not tx.X`, so checking a hint's identity
    against a single spelling misses the other one. This returns the
    objects both modules provide under `name`, with duplicates removed,
    so that every spelling can be tested against with `is`.

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

# Every spelling of `Unpack`. On 3.11 `typing.Unpack is not tx.Unpack`, and the
# star syntax `Tuple[int, *Ts]` yields `typing.Unpack[Ts]` while the
# `tx.Unpack[Ts]` spelling yields the `typing_extensions` one, so an identity
# check must accept both.
_UNPACK_FORMS = spellings("Unpack")

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
    """Report whether `hint` is a typing construct rather than a class.

    Checking `hint` against the explicit [`_SPECIAL_FORMS`][] tuple is
    the fast path. A structural fallback then recognises any construct
    that lives in `typing` or `typing_extensions` and has become a
    class on the running Python version, or would on a future one, so
    that such a form is never mistaken for a class a user could
    subclass.
    """
    # Identity, not `in`: `==` on typing objects can be surprising.
    if any(hint is form for form in _SPECIAL_FORMS):
        return True
    return _is_typing_class_form(hint)


def _is_typing_class_form(hint: tx.Any) -> bool:
    """Report whether `hint` is a typing special form disguised as a class.

    A special form such as `Generic`, `Protocol` or `TypedDict` lives
    in `typing` or `typing_extensions` and, on some Python versions, is
    implemented as a class. This tells such a form apart from a real,
    subclassable class that merely happens to live in the same module.
    """
    if not isinstance(hint, type):
        return False
    module = getattr(hint, "__module__", None)
    if module not in ("typing", "typing_extensions"):
        return False
    if any(hint is marker for marker in _GENERIC_MARKERS):
        return False
    if any(hint is marker for marker in _PROTOCOL_MARKERS):
        return False
    if is_typeddict_marker(hint):  # pragma: no cover  -- 3.8 only
        # On Python 3.8 `typing.TypedDict` is a real class
        # (`class TypedDict(dict, metaclass=_TypedDictMeta)`), so it reaches
        # here through the `isinstance(hint, type)` guard above and must not be
        # taken for a class-shaped special form. From 3.9 on (and on the
        # single-version coverage job) the marker is a function, excluded by
        # that guard, so this line is not reached there.
        return False
    # Real, checkable classes that merely live in `typing`, such as a
    # Protocol (`SupportsInt`, `SupportsIndex`, ...), a `Generic`
    # subclass (`typing.IO`), or an ABC (`Buffer` on 3.11 and earlier),
    # are not special forms: they can be subclass- or instance-checked,
    # so treating them as special forms would make the relation
    # silently reject them.
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
    """Report whether `obj` could plausibly be a type hint.

    A hint is a class, a typing special form, a member of the
    [`TypeVar`][typing.TypeVar] family, a [`ForwardRef`][typing.ForwardRef],
    or any other object defined in `typing` or `typing_extensions`,
    which also covers a future construct of the same shape. An obvious
    non-hint, such as a plain value (a number, a string, a container
    instance) or an ordinary function, is not plausible, and the
    relation raises for it rather than silently treating it as `Any`.
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
    """Report whether `cls` is `TypedDict` itself, in either spelling."""
    return any(cls is marker for marker in _TYPEDDICT_MARKERS)


def canonical_typeddict(cls: tx.Any) -> tx.Any:
    """Return `cls`, collapsing either spelling of `TypedDict` to one."""
    return tx.TypedDict if is_typeddict_marker(cls) else cls
