"""Telling a typing construct apart from a class, across Python versions.

Whether a given typing construct happens to be implemented as a class is
not stable across Python releases: [`Any`][typing.Any] only became one
in 3.11, [`Annotated`][typing.Annotated] was one up through 3.12 and
stopped being one in 3.13, and [`Union`][typing.Union] became one in
3.14. A plain `#!python isinstance(hint, type)` check therefore answers
differently for the same piece of code depending on which interpreter
runs it. Everything in this module exists to give one fixed answer to
"is this a special form or a class a user could subclass", regardless of
which representation the running interpreter happens to use.
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
"""Every object that can appear as the origin of a union hint.

On a Python old enough that the `X | Y` syntax does not exist, this
holds only `Union` itself; from 3.10 on, where `X | Y` produces its own
`UnionType` object distinct from `Union`, it holds both.
"""


class UnknownHintWarning(RuntimeWarning):
    """Raised when a hint the subtype relation cannot identify is met.

    A hint that this package does not recognise, for instance a
    construct added by a Python release newer than the one this package
    was written against, is not rejected outright. It is instead treated
    as equivalent to [`Any`][typing.Any]: every other hint is accepted as
    broader than it, and it is narrower than nothing except itself and
    `Any`. A method annotated with such a hint therefore stays reachable
    by dispatch rather than becoming permanently unmatchable, and this
    warning is raised once for each distinct construct met this way, so
    that the fallback does not pass unnoticed.
    """


def spellings(name: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """Collect the distinct objects that both typing modules export as `name`.

    A construct such as `Unpack` is defined in both [`typing`][] and
    `typing_extensions`, and on some Python versions the two are not the
    same object (`#!python typing.X is not tx.X`), so an identity check
    written against only one of them misses hints that use the other.
    Looking `name` up on both modules and keeping the results distinct
    by identity gives every object a caller needs to compare against
    with `is` to recognise the construct under either spelling.

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

    `hint` is checked first against the fixed [`_SPECIAL_FORMS`][] tuple,
    which covers the constructs this module already knows about by
    identity. Anything that passes that check without matching falls
    through to a structural test for a typing construct that happens to
    be implemented as a class on the running interpreter, so that such a
    construct is never mistaken for an ordinary class a user could
    subclass.
    """
    # Identity, not `in`: `==` on typing objects can be surprising.
    if any(hint is form for form in _SPECIAL_FORMS):
        return True
    return _is_typing_class_form(hint)


def _is_typing_class_form(hint: tx.Any) -> bool:
    """Report whether `hint` is a class-shaped special form, not a real class.

    `Generic`, `Protocol`, `TypedDict` and similar markers are, on
    certain Python versions, implemented as classes even though they
    behave as typing constructs rather than as types a value can belong
    to. This distinguishes that case from a genuine class that a caller
    could subclass or instance-check against, even one that happens to
    live in the `typing` or `typing_extensions` module itself.
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


# Every spelling of `TypeAliasType`, the runtime object behind a PEP 695
# `type X = ...` alias, from both typing modules.
_TYPE_ALIAS_TYPES = spellings("TypeAliasType")


def _is_newtype(x: tx.Any) -> bool:
    """Report whether `x` is a `NewType`, under any of its runtime forms."""
    return callable(x) and hasattr(x, "__supertype__")


def _is_type_alias_type(x: tx.Any) -> bool:
    """Report whether `x` is a PEP 695 `type X = ...` alias, in either
    spelling.

    Both an instance check and a duck-typed fallback are tried, because a
    native 3.12 `type X = ...` alias is not an instance of
    `typing_extensions.TypeAliasType`, even though every alias, whichever
    way it was spelled, carries `__value__` and `__type_params__`.
    """
    for alias_type in _TYPE_ALIAS_TYPES:
        try:
            if isinstance(x, alias_type):
                return True
        except TypeError:  # pragma: no cover  -- not a class on this version
            pass
    return hasattr(x, "__value__") and hasattr(x, "__type_params__")


def ishint(x: tx.Any) -> bool:
    """Report whether `x` is a type hint, rather than an ordinary value.

    A type hint is anything that can legitimately stand as an annotation
    or as an argument to the subtype relation, which includes far more
    than a class alone. A bare [`None`][] counts, standing for
    [`NoneType`][types.NoneType], as does a string or a
    [`ForwardRef`][typing.ForwardRef] naming a type not yet resolved. A
    class counts, and so does a parametrised generic such as
    `#!python List[int]`, which carries a typing origin. A recognised
    special form such as [`Union`][typing.Union] or
    [`Literal`][typing.Literal] counts too, and indeed any object defined
    in [`typing`][] or `typing_extensions`, which leaves room for a
    construct those modules add in a future release. Finally, the
    [`TypedDict`][tx.TypedDict] marker, a [`NewType`][typing.NewType], and
    a PEP 695 `#!python type X = ...` alias each count.

    An everyday value that was never meant to be a hint, such as a
    number, a container instance, an ordinary function, or a sentinel
    object, fails this check. A bare string is the one deliberate
    exception: it counts as a hint here, since it may be a forward
    reference, even though most callers that reject non-hints reject a
    bare string too.

    !!! example
        ```pycon
        >>> ishint(int)
        True
        >>> ishint(tx.Union)
        True
        >>> ishint("Foo")
        True
        >>> ishint(1)
        False
        ```
    """
    if x is None:
        return True
    if isinstance(x, (str, tx.ForwardRef)):
        return True
    if isinstance(x, type):
        return True
    if tx.get_origin(x) is not None:
        return True
    if is_special_form(x) or type(x).__module__ in (
        "typing",
        "typing_extensions",
    ):
        return True
    return (
        is_typeddict_marker(x)
        or _is_newtype(x)
        or _is_type_alias_type(x)
    )


def is_plausible_hint(obj: tx.Any) -> bool:
    """Report whether `obj` is shaped like something that could be a hint.

    This is [`ishint`][] with the one bare-string case excluded, since a
    caller that rejects an implausible hint outright, such as a method
    registration, has no namespace in which to resolve a forward
    reference and so cannot accept a bare string name either. Every other
    hint [`ishint`][] recognises counts as plausible here too. An everyday
    value that was never meant to be a hint fails this check, so that the
    caller can raise a clear error for it rather than register a method
    that silently never matches.
    """
    return ishint(obj) and not isinstance(obj, str)


class SameObject:
    """A wrapper around an object that makes a dict key it by identity.

    `typing`'s own equality merges some hints that dispatch needs to tell
    apart. On Python 3.8, `#!python Literal[1] == Literal[True]` holds with
    equal hashes, and so does `#!python Box[Literal[1]] == Box[Literal[True]]`,
    so a plain `#!python ==`-keyed cache can collapse two distinct hints onto
    one entry and hand back the wrong answer once `typing`'s own subscription
    cache has evicted the earlier object. Wrapping an object in `SameObject`
    keys it by [`id`][] instead, which never merges two distinct objects and
    is always hashable. The wrapper holds onto the object itself, so its
    identity cannot be reassigned to something else while a cache entry that
    depends on it is alive.

    Because the key is identity, two equal but separately built objects, such
    as `#!python Literal[1]` created twice, count as different keys. The only
    consequence is a missed cache hit and a fresh computation, never a wrong
    result.
    """

    __slots__ = ("obj",)

    def __init__(self, obj: tx.Any) -> None:
        self.obj = obj

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other: tx.Any) -> bool:
        if not isinstance(other, SameObject):
            return NotImplemented
        return self.obj is other.obj


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
    """Report whether `cls` is the `TypedDict` marker, from either module."""
    return any(cls is marker for marker in _TYPEDDICT_MARKERS)


def canonical_typeddict(cls: tx.Any) -> tx.Any:
    """Return `cls` unchanged, unless it is a `TypedDict` marker.

    A `TypedDict` marker imported from `typing` and one imported from
    `typing_extensions` are different objects standing for the same
    thing, so a caller that compares by identity needs them collapsed to
    a single representative first. `cls` passes through untouched
    whenever it is not one of those two markers.
    """
    return tx.TypedDict if is_typeddict_marker(cls) else cls
