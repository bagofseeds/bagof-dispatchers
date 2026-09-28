"""Hint introspection helpers.

This module collects small, version-safe wrappers around [`typing`][]
that never raise when given a value that is not a type, together with
helpers that unwrap the transparent wrappers, such as
[`Annotated`][typing.Annotated] and [`TypeVar`][typing.TypeVar], that
the subtype relation looks straight through.
"""

# stdlib
import collections
import contextlib
import functools
import inspect
import math
import numbers
import re
import sys
import types
from collections import abc

# dependencies
import typing_extensions as tx

# local
from ._compat import (
    _ANY_FORMS,
    _LITERAL_FORMS,
    UNION_TYPES,
    NoneType,
    canonical_typeddict,
    is_special_form,
    is_typeddict_marker,
    spellings,
)
from ._exact import exact_target, is_exact
from ._sentinels import UNSET


def _looks_like_class(x: tx.Any) -> bool:
    """Report whether `x` is a real class, not a parametrised generic alias.

    `#!python isinstance(list[int], type)` is `#!python True` on Python
    3.9 and 3.10, so a bare `#!python isinstance(x, type)` mistakes
    `#!python list[int]` for a class. A real class has no typing origin,
    which tells the two apart on every supported version.
    """
    return isinstance(x, type) and tx.get_origin(x) is None


# --- origins and arguments ---------------------------------------------


def safe_get_origin(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Any:
    """Return a hint's origin, without raising on a hint that has none.

    On request, it first unwraps a wrapper such as
    [`Annotated`][typing.Annotated] before reading the origin.

    !!! note
        Unlike [`typing.get_origin`][], this returns the hint itself,
        rather than `None`, when the hint is not a generic type.
    """
    if unwrap:
        hint = _unwrap(hint, origin=unwrap)
    origin = tx.get_origin(hint)
    if origin is None:
        return hint
    return origin


def get_origin_uw(hint: tx.Any) -> tx.Any:
    """Return a hint's origin, unwrapping `Annotated` first.

    Returns the hint itself, rather than `None`, when it is not a
    generic type.
    """
    return safe_get_origin(hint, unwrap=tx.Annotated)


def safe_get_args(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Tuple[tx.Any, ...]:
    """Return a hint's type arguments, without raising on a plain type.

    Returns an empty tuple when the hint is not a generic type. On
    request, it first unwraps a wrapper such as
    [`Annotated`][typing.Annotated] before reading the arguments.
    """
    hint = _unwrap(hint, origin=unwrap)
    return tx.get_args(hint)


def get_args_uw(hint: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """Return a hint's type arguments, unwrapping `Annotated` first.

    Returns an empty tuple when the hint is not a generic type.
    """
    return safe_get_args(hint, unwrap=tx.Annotated)


# --- unwrapping --------------------------------------------------------


def unwrap(hint: tx.Any, origin: tx.Any = (tx.Annotated,)) -> tx.Any:
    """Strip a hint's origin away, if that origin is one of `origin`.

    When [`TypeVar`][typing.TypeVar] is among the origins to unwrap, a
    type variable is replaced by its default, by the union of its
    constraints, or by its bound, trying each in that order.

    !!! example
        ```pycon
        >>> from typing import Annotated
        >>> unwrap(Annotated[int, "meta"])
        <class 'int'>
        >>> unwrap(Annotated[Annotated[str, 1], 2])
        <class 'str'>
        >>> unwrap(int)  # unchanged
        <class 'int'>
        ```
    """
    if origin is None:
        origin = ()
    if isinstance(origin, str) or not isinstance(origin, abc.Sequence):
        # A `str` is a `Sequence`, but a single hint - not a list of them.
        origin = (origin,)
    if safe_get_origin(hint) in origin:
        args = tx.get_args(hint)
        if args:
            return unwrap(args[0], origin=origin)
        # A bare, unsubscripted special form (e.g. `Annotated`, with no
        # argument to unwrap to) falls through opaquely rather than raising an
        # `IndexError` -- so it behaves as an unknown, `Any`-like hint (RFC
        # 0001 §2.1) both here and everywhere `unwrap`/`safe_get_origin` is a
        # step in the relation.
        return hint
    if tx.TypeVar in origin and safe_isinstance(hint, tx.TypeVar):
        return unwrap(_unwrap_typevar(hint), origin=origin)
    return hint


_unwrap = unwrap  # alias for convenience


def _unwrap_typevar(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    origin = get_origin_uw(hint)
    if origin in __reentrant:
        # A cycle (e.g. two typevars defaulting to each other). Returning
        # the typevar would send `unwrap` straight back in here, so answer
        # what an uninformative typevar answers.
        return tx.Any
    __reentrant += (origin,)
    if not safe_isinstance(origin, tx.TypeVar):
        return hint
    if getattr(origin, "__default__", tx.NoDefault) is not tx.NoDefault:
        return _unwrap_typevar(origin.__default__, __reentrant=__reentrant)
    if getattr(origin, "__constraints__", ()):
        return tx.Union[origin.__constraints__]
    if getattr(origin, "__bound__", None) is not None:
        return _unwrap_typevar(origin.__bound__, __reentrant=__reentrant)
    return tx.Any


# --- normalisation -----------------------------------------------------


# The transparent qualifiers that wrap a hint without changing which
# values it accepts, so the relation looks straight through them.
_QUALIFIER_FORMS = (
    spellings("Required")
    + spellings("NotRequired")
    + spellings("ReadOnly")
    + spellings("Final")
    + spellings("ClassVar")
)

_TYPE_ALIAS_TYPES = spellings("TypeAliasType")

# The bare, unparametrised tuple spellings. Used to tell a bare `Tuple` /
# `tuple` from a subscripted alias whose arguments happen to be empty -- the
# empty-tuple type `Tuple[()]` (mirrors `_relation._is_subscripted_tuple`,
# kept here to avoid importing from `_relation`, which imports this module).
_BARE_TUPLE_FORMS = spellings("Tuple") + (tuple,)

# A generous cap: each pass either resolves one wrapper (strictly reducing
# the hint) or leaves it untouched, so a handful of passes always settles.
_MAX_NORMALISE_STEPS = 100


def _is_type_alias_type(x: tx.Any) -> bool:
    """Report whether `x` is a PEP 695 `type X = ...` alias, in either
    spelling.

    This checks duck typing as well as the instance check: a native
    3.12 `type X = ...` alias is not an instance of
    `typing_extensions.TypeAliasType`, but every alias, in either
    spelling, carries `__value__` and `__type_params__`.
    """
    for alias_type in _TYPE_ALIAS_TYPES:
        try:
            if isinstance(x, alias_type):
                return True
        except TypeError:  # pragma: no cover  -- not a class on this version
            pass
    return hasattr(x, "__value__") and hasattr(x, "__type_params__")


def resolve_alias(hint: tx.Any) -> tx.Any:
    """Resolve a PEP 695 `type X = ...` alias to the hint it stands for.

    A bare alias becomes its value. A subscripted generic alias, such
    as `#!python L[int]` for `#!python type L[T] = list[T]`, has its
    type arguments substituted in first. An alias that stands for
    another alias is followed all the way to the end, and a reference
    cycle stops rather than recursing forever. A hint that is not an
    alias at all is returned unchanged.
    """
    return _resolve_alias(hint, ())


def _resolve_alias(hint: tx.Any, seen: tx.Tuple[tx.Any, ...]) -> tx.Any:
    # The subscripted case comes first: `L[int]` forwards `__value__` from
    # its origin `L`, so it would otherwise be mistaken for a bare alias and
    # its type arguments dropped.
    origin = tx.get_origin(hint)
    if _is_type_alias_type(origin):
        alias = origin
        sub_args = tx.get_args(hint)  # type: tx.Tuple[tx.Any, ...]
    elif _is_type_alias_type(hint):
        alias = hint
        sub_args = ()
    else:
        return hint
    if any(alias is each for each in seen):
        # A recursive alias -- stop rather than loop, leaving the origin in
        # place to be matched structurally. A cycle arises with native PEP 695
        # `type X = ... X ...` syntax (3.12+), and with any duck-typed alias
        # whose `__value__` points back at itself, so it is reachable on every
        # version.
        return hint
    value = alias.__value__
    if sub_args:
        # `type L[T] = list[T]`; `L[int]` fills `T` in through typing's own
        # subscription: `(list[T])[int]` is `list[int]`.
        key = sub_args if len(sub_args) > 1 else sub_args[0]
        try:
            value = value[key]
        except Exception:  # pragma: no cover  -- a value that refuses args
            return hint
    return _resolve_alias(value, seen + (alias,))


def _is_newtype(x: tx.Any) -> bool:
    """Report whether `x` is a `NewType`, in any of its runtime forms."""
    return callable(x) and hasattr(x, "__supertype__")


def resolve_newtype(hint: tx.Any) -> tx.Any:
    """Resolve a [`NewType`][typing.NewType] to its supertype, recursively.

    A `NewType` is a distinct name for an existing type; for dispatch it
    behaves exactly as that type, so it is followed to the underlying hint.
    A hint that is not a `NewType` is returned unchanged.
    """
    seen = ()  # type: tx.Tuple[tx.Any, ...]
    while _is_newtype(hint):
        if any(hint is each for each in seen):  # pragma: no cover
            break
        seen += (hint,)
        hint = hint.__supertype__
    return hint


def _strip_qualifier(hint: tx.Any) -> tx.Any:
    """Drop a transparent qualifier such as `Required`, `Final`, or
    `ClassVar`.
    """
    origin = tx.get_origin(hint)
    if origin is not None and any(origin is q for q in _QUALIFIER_FORMS):
        args = tx.get_args(hint)
        if args:
            return args[0]
    return hint


def normalise_hint(hint: tx.Any) -> tx.Any:
    """Put a hint into its canonical form.

    A bare [`None`][] means [`NoneType`][types.NoneType] as a hint, so
    it is replaced by it. A PEP 695 `type X = ...` alias is resolved to
    the hint it stands for. A [`NewType`][typing.NewType] is resolved
    to its supertype. The transparent qualifiers
    [`Required`][typing.Required], [`NotRequired`][typing.NotRequired],
    [`ReadOnly`][typing.ReadOnly], [`Final`][typing.Final], and
    [`ClassVar`][typing.ClassVar] are unwrapped to the hint they wrap.

    These steps are applied repeatedly until the hint settles, so that
    an alias which expands to a qualified `NewType` is fully resolved
    rather than resolved only one layer deep. Every other hint is
    returned unchanged.

    !!! note
        Only a bare `None` is replaced this way. A `None` inside a
        hint keeps its own meaning: `#!python Literal[None]` describes
        the literal `None` value, not a type. [`Annotated`][typing.Annotated]
        is not stripped here, because its metadata can carry an
        exactness marker that the relation reads.

    !!! example
        ```pycon
        >>> normalise_hint(None)
        <class 'NoneType'>
        >>> normalise_hint(int)
        <class 'int'>
        ```
    """
    for _ in range(_MAX_NORMALISE_STEPS):
        if hint is None:
            hint = NoneType
        resolved = _strip_qualifier(resolve_newtype(resolve_alias(hint)))
        if resolved is hint:
            return hint
        hint = resolved
    return hint  # pragma: no cover  -- the cap is never reached in practice


# --- TypedDict ---------------------------------------------------------


def is_typeddict(cls: tx.Any) -> bool:
    """Report whether `cls` is a [`TypedDict`][tx.TypedDict] or a subclass.

    !!! tip
        This differs from [`typing.is_typeddict`][tx.is_typeddict],
        which returns `#!python False` for [`TypedDict`][tx.TypedDict]
        itself; this function returns `#!python True` for it.
    """
    if is_typeddict_marker(cls):
        return True
    return tx.is_typeddict(cls)


@functools.lru_cache(maxsize=None)
def typeddict_required_keys(cls: tx.Any) -> tx.FrozenSet[str]:
    """Return the required keys of a [`TypedDict`][tx.TypedDict].

    Where the class has `__required_keys__`, this reads it directly;
    it is the only source that accounts for
    [`Required`][typing.Required] and [`NotRequired`][typing.NotRequired]
    (nested inside [`Annotated`][typing.Annotated] or not) and for
    inheriting from bases declared with a different `total=`.

    Where the class does not have `__required_keys__`, this falls back
    to `__total__`: [`typing.TypedDict`][] gained `__required_keys__`
    only in Python 3.9, and before that a key's requiredness came from
    the class's `total=` alone, since per-key `Required` and
    `NotRequired` did not yet exist.

    !!! warning
        On older Pythons, a [`typing.TypedDict`][] that mixes `total=`
        across its bases is read only through the subclass's own
        `total=`, which cannot be correct for every key: the standard
        library records neither which class declared a key nor a
        usable link back to the base, since a subclass has no
        `__orig_bases__` and its `__mro__` reaches only [`dict`][], so
        the per-key answer is simply not recoverable.

        The error runs in both directions, not only the safe one. A
        `total=True` key inherited into a `total=False` subclass is
        reported optional, so a value missing it is accepted when it
        should fail, as when `{}` is matched against a shape with a
        required key. A `total=False` key inherited into a
        `total=True` subclass is reported required, so a valid value
        is rejected. Use [`typing_extensions.TypedDict`][tx.TypedDict],
        which reimplements the class precisely and records
        `__required_keys__` on every version, when this distinction
        matters.

    !!! example
        ```pycon
        >>> class Movie(TypedDict):
        ...     title: str
        ...     year: NotRequired[int]
        >>> typeddict_required_keys(Movie)
        frozenset({'title'})
        ```
    """
    keys = getattr(cls, "__required_keys__", None)
    if keys is not None:
        return frozenset(keys)
    annotations = getattr(cls, "__annotations__", {})
    if getattr(cls, "__total__", True):
        return frozenset(annotations)
    return frozenset()


@functools.lru_cache(maxsize=None)
def typeddict_field_hints(cls: tx.Any) -> tx.Dict[str, tx.Any]:
    """Map each declared field of a [`TypedDict`][tx.TypedDict] to its hint.

    The mapping covers every key the class declares, its own and those
    inherited from [`TypedDict`][tx.TypedDict] bases, each mapped to the
    hint written for it. The [`Required`][typing.Required] and
    [`NotRequired`][typing.NotRequired] qualifier is left on the hint;
    reading it off is [`typeddict_required_keys`][]'s job, and a caller
    that only wants the value type can simply let the relation look
    through the qualifier.

    A string annotation is resolved against the class's own module
    where possible, using [`typing.get_type_hints`][tx.get_type_hints].
    Where a forward reference cannot be resolved, the raw name is kept
    for that field alone, so the caller sees the name rather than
    nothing, while every sibling field that can be resolved still is.

    !!! example
        ```pycon
        >>> class Movie(TypedDict):
        ...     title: str
        ...     year: NotRequired[int]
        >>> sorted(typeddict_field_hints(Movie))
        ['title', 'year']
        ```
    """
    if is_typeddict_marker(cls):
        # The bare marker declares no fields of its own.
        return {}
    try:
        return dict(tx.get_type_hints(cls, include_extras=True))
    except Exception:
        # `get_type_hints` is all-or-nothing: a single unresolvable forward
        # reference makes it raise for the whole class. Returning the raw
        # `__annotations__` would then hand back *every* field unresolved --
        # under `from __future__ import annotations` all of them are strings --
        # so a sibling field with a perfectly readable hint would go
        # unchecked. Resolve each field on its own instead.
        return _resolve_fields_individually(cls)


def _resolve_fields_individually(cls: tx.Any) -> tx.Dict[str, tx.Any]:
    """Resolve a TypedDict's fields one at a time, keeping what resolves.

    This is the fallback [`typeddict_field_hints`][] uses in place of
    its own all-or-nothing resolution: each annotation is evaluated
    against the class's own module, the ones that resolve are kept, and
    the ones that do not are left as their raw name, a string, for the
    caller to skip.
    """
    module = sys.modules.get(getattr(cls, "__module__", None))
    globalns = getattr(module, "__dict__", {})
    hints = {}  # type: tx.Dict[str, tx.Any]
    for key, raw in getattr(cls, "__annotations__", {}).items():
        hints[key] = _resolve_one_annotation(raw, globalns)
    return hints


def _resolve_one_annotation(
    raw: tx.Any, globalns: tx.Mapping[str, tx.Any]
) -> tx.Any:
    """Evaluate one annotation against `globalns`, or keep its raw name.

    A `raw` value that is already a hint object, an unquoted
    annotation, is returned unchanged. A string or
    [`ForwardRef`][typing.ForwardRef] is evaluated the way
    [`typing.get_type_hints`][tx.get_type_hints] would; when its name is
    not defined there, the raw form is returned instead, so that field
    can be skipped without losing the whole class.
    """
    if isinstance(raw, tx.ForwardRef):
        source = raw.__forward_arg__  # type: tx.Any
    elif isinstance(raw, str):
        source = raw
    else:
        return raw
    try:
        return eval(source, dict(globalns))  # noqa: S307 -- as get_type_hints
    except Exception:
        return raw


def _all_orig_bases(cls: type, _self: bool = True) -> tx.Tuple[type, ...]:
    """Return all original bases of a type, including the type itself."""
    if not is_typeddict(cls):
        return ()
    bases = (cls,) if _self else ()
    for base in getattr(cls, "__orig_bases__", ()):
        if is_typeddict_marker(base):
            # Appended once, canonically, at the end.
            continue
        bases += (base,) + _all_orig_bases(base, _self=False)
    if _self:
        # Always terminate with the canonical marker rather than trusting
        # `__orig_bases__` to contain one. `typing.TypedDict` records no
        # `__orig_bases__` at all on a sub-subclass, and the two spellings
        # never appear in each other's bases -- so deriving this from the
        # declared bases alone misses a typeddict that plainly is one.
        bases += (tx.TypedDict,)
    return bases


# --- safe isinstance / issubclass --------------------------------------


def safe_issubclass(subcls: tx.Any, cls: tx.Any) -> bool:
    """Report whether `subcls` is a subclass of `cls`, without raising.

    !!! warning
        When `cls` is a [`TypedDict`][tx.TypedDict], this looks at
        `subcls`'s `__orig_bases__` rather than its `__bases__`. A
        plain [`dict`][] is not a subclass of a
        [`TypedDict`][tx.TypedDict]; the relation holds only the other
        way round.

    !!! example
        ```pycon
        >>> safe_issubclass(bool, int)
        True
        >>> safe_issubclass(bool, (str, int))  # a tuple, like `issubclass`
        True
        >>> safe_issubclass(int, "not a type")  # no error
        False
        ```
    """
    if isinstance(cls, tuple):
        return any(safe_issubclass(subcls, each) for each in cls)
    if is_typeddict(cls):
        return canonical_typeddict(cls) in _all_orig_bases(subcls)
    if is_special_form(cls) or is_special_form(subcls):
        # A typing construct may be a real class on a recent Python
        # (`Any` from 3.11, `Union` from 3.14), so `issubclass` would
        # answer it - differently than on the versions before.
        return False
    if _looks_like_class(subcls) and _looks_like_class(cls):
        try:
            return issubclass(subcls, cls)
        except TypeError:
            # A non-`runtime_checkable` Protocol refuses `issubclass`.
            # Answer False rather than letting it raise out of the relation.
            return False
    return False


def safe_isinstance(obj: tx.Any, cls: tx.Any) -> bool:
    """Report whether `obj` is an instance of `cls`, without raising.

    !!! warning
        A [`TypedDict`][tx.TypedDict] cannot be instance-checked at
        all. Python refuses `#!python isinstance(value, SomeTypedDict)`
        outright, and a `TypedDict` leaves no trace on the dict it
        describes, so there is nothing to recognise at runtime. This
        function therefore answers [`False`][] for one; validate the
        shape of the dict instead.

    !!! example
        ```pycon
        >>> safe_isinstance(1, int)
        True
        >>> safe_isinstance(1, (str, int))  # a tuple, like `isinstance`
        True
        >>> safe_isinstance(1, "not a type")  # no error
        False
        ```
    """
    if isinstance(cls, tuple):
        return any(safe_isinstance(obj, each) for each in cls)
    if is_typeddict(cls):
        return safe_issubclass(type(obj), cls)
    if isinstance(cls, type) and not any(cls is form for form in _ANY_FORMS):
        return isinstance(obj, cls)
    return False


# --- classifiers -------------------------------------------------------


def issubclassable(cls: tx.Any) -> bool:
    """Report whether `cls` is a type, or is [`TypedDict`][tx.TypedDict].

    !!! tip
        This differs from `#!python isinstance(cls, type)` in that it
        returns [`True`][] for [`TypedDict`][tx.TypedDict] and its
        subclasses, even though they are not technically types.

    !!! note
        A typing construct such as [`Any`][typing.Any],
        [`Union`][typing.Union], or [`Literal`][typing.Literal] is never
        subclassable, on any Python version. Some of them are classes
        on recent Pythons, `Any` from 3.11 and `Union` from 3.14, so
        `#!python isinstance(hint, type)` answers differently across
        the versions this package supports.
    """
    if is_special_form(cls):
        return False
    if is_typeddict_marker(cls):
        return True
    return _looks_like_class(cls)


def issubscriptable(x: tx.Any) -> bool:
    """Report whether `x` can be subscripted with `#!python x[...]`.

    This is `#!python True` when `x` is a type that has
    `__class_getitem__`, or an instance that has `__getitem__`, and
    `#!python False` otherwise.
    """
    is_class = _looks_like_class(x)
    if is_class and hasattr(x, "__class_getitem__"):
        return True
    if not is_class and hasattr(x, "__getitem__"):
        return True
    return False


# --- concrete types ----------------------------------------------------


def get_concrete_type(hint: tx.Any, fallback: type = UNSET) -> tx.Type[tx.Any]:
    """Return a concrete, instantiable type for a type hint.

    If the hint is annotated, its [`Annotated`][typing.Annotated]
    wrapper is removed first. If the resulting hint has an origin, the
    origin is used. If it is a [`TypeVar`][typing.TypeVar] instead, its
    default value is used if it has one; otherwise the first of its
    constraints is used if it has any; otherwise its bound is used if
    it has one; otherwise the `fallback` type is used if one was
    provided; and failing all of that, a [`TypeError`][] is raised.

    Once resolved this way, a concrete, non-abstract type is returned
    as is. Anything else falls back to `fallback` if one was provided,
    or raises a [`TypeError`][] if not.

    !!! note
        A constrained type variable has no single concrete type, since
        it stands for the union of its constraints, so its first
        constraint is taken as a stand-in.

    !!! example
        ```pycon
        >>> get_concrete_type(List[int])
        <class 'list'>
        >>> get_concrete_type(TypeVar("T", int, str))
        <class 'int'>
        ```
    """
    origin = safe_get_origin(hint, unwrap=(tx.Annotated, tx.TypeVar))
    if _is_concrete_type(origin):
        return origin
    concrete = _first_concrete_constraint(hint)
    if concrete is not None:
        return concrete
    if safe_isinstance(fallback, type):
        return fallback
    raise TypeError(
        f"Cannot get concrete type for hint {hint} (of type {type(hint)}) "
        f"and fallback {fallback} (of type {type(fallback)})."
    )


def _is_concrete_type(hint: tx.Any) -> bool:
    """Report whether a hint is a class that can actually be instantiated."""
    if is_special_form(hint):
        # `Union` is a class from python 3.14 on, but instantiating it
        # is still meaningless.
        return False
    return safe_isinstance(hint, type) and not inspect.isabstract(hint)


def _first_concrete_constraint(hint: tx.Any) -> tx.Optional[type]:
    """Return the first concrete constraint of a constrained typevar,
    if any.
    """
    typevar = unwrap(hint, tx.Annotated)
    if not safe_isinstance(typevar, tx.TypeVar):
        return None
    for constraint in getattr(typevar, "__constraints__", ()):
        origin = safe_get_origin(constraint, unwrap=(tx.Annotated,))
        if _is_concrete_type(origin):
            return origin
    return None


# --- variance ----------------------------------------------------------


# The three variances a generic's parameter position can have (PEP 484).
# Plain strings rather than an enum: they are only ever compared and
# carried, and a string reads straight in a table and a test.
_COVARIANT = "covariant"
_CONTRAVARIANT = "contravariant"
_INVARIANT = "invariant"


# The members of the `TypeVar` family that are not plain type variables:
# a ParamSpec or a TypeVarTuple fills a parameter position but has no
# variance of its own. On Python 3.8, typing_extensions backports both
# as subclasses of TypeVar, so a bare isinstance(p, tx.TypeVar) accepts
# them there too; this tuple is what tells them apart on every version.
_NON_TYPE_PARAMS = tuple(
    form
    for name in ("ParamSpec", "TypeVarTuple")
    for form in (getattr(tx, name, None),)
    if isinstance(form, type)
)


def _is_plain_typevar(param: tx.Any) -> bool:
    """Report whether `param` is an ordinary `TypeVar`, not a
    `ParamSpec` or `*Ts`.

    A [`ParamSpec`][typing.ParamSpec] or
    [`TypeVarTuple`][typing.TypeVarTuple] carries no variance of its
    own, so a generic that has one in a parameter position has no
    readable per-position variance at all.
    """
    return isinstance(param, tx.TypeVar) and not (
        _NON_TYPE_PARAMS and isinstance(param, _NON_TYPE_PARAMS)
    )


def _typevar_variance(tv: tx.Any) -> str:
    """Return the variance a [`TypeVar`][typing.TypeVar] declares, per PEP 484.

    This returns `#!python "covariant"` for a
    `#!python TypeVar(..., covariant=True)`,
    `#!python "contravariant"` for a
    `#!python TypeVar(..., contravariant=True)`, and
    `#!python "invariant"` otherwise. The invariant case covers both an
    unflagged type variable, which PEP 484 makes invariant by default,
    and a PEP 695 `#!python class Box[T]` variable, whose variance is
    inferred by the type checker and cannot be read at runtime, so it
    is simply taken as invariant here.

    Every attribute is read with [`getattr`][], because the `TypeVar`
    family differs across Python 3.8 through 3.13 and between
    [`typing`][] and `typing_extensions`.

    !!! example
        ```pycon
        >>> _typevar_variance(tx.TypeVar("T_co", covariant=True))
        'covariant'
        >>> _typevar_variance(tx.TypeVar("T"))
        'invariant'
        ```
    """
    if getattr(tv, "__infer_variance__", False) is True:
        # PEP 695 auto-variance: the type checker decides, so at runtime it
        # is the owner's call, taken as invariant.
        return _INVARIANT
    if getattr(tv, "__covariant__", False):
        return _COVARIANT
    if getattr(tv, "__contravariant__", False):
        return _CONTRAVARIANT
    return _INVARIANT


# The per-position variance of the standard-library generics, keyed by the
# runtime origin [`tx.get_origin`][] returns for each.
#
# GENERATED from CPython 3.8's `typing` module, the typing spec's reference
# implementation: on 3.8 `typing.List`, `typing.Sequence`, ... still expose
# `__parameters__` whose `TypeVar`s carry the spec variance, while from 3.9
# on the special aliases expose nothing. Variance is version-invariant, so
# these values hold on every supported Python. `tests/
# test_variance_introspect.py` regenerates this from the live `typing` on
# 3.8 and asserts it equals this table, so any drift is caught rather than
# silently trusted.
#
# `Tuple` and `Callable` are deliberately absent: they carry no
# `__parameters__` and are ordered by their own dedicated paths (tuple shape,
# and contravariant parameters with a covariant return) in `_relation.py`.
_STDLIB_VARIANCE = {
    list: (_INVARIANT,),
    set: (_INVARIANT,),
    frozenset: (_COVARIANT,),
    dict: (_INVARIANT, _INVARIANT),
    type: (_COVARIANT,),
    collections.deque: (_INVARIANT,),
    collections.defaultdict: (_INVARIANT, _INVARIANT),
    collections.OrderedDict: (_INVARIANT, _INVARIANT),
    collections.Counter: (_INVARIANT,),
    collections.ChainMap: (_INVARIANT, _INVARIANT),
    abc.Sequence: (_COVARIANT,),
    abc.MutableSequence: (_INVARIANT,),
    abc.Set: (_COVARIANT,),
    abc.MutableSet: (_INVARIANT,),
    abc.Mapping: (_INVARIANT, _COVARIANT),
    abc.MutableMapping: (_INVARIANT, _INVARIANT),
    abc.Collection: (_COVARIANT,),
    abc.Container: (_COVARIANT,),
    abc.Iterable: (_COVARIANT,),
    abc.Iterator: (_COVARIANT,),
    abc.Reversible: (_COVARIANT,),
    abc.KeysView: (_INVARIANT,),
    abc.ValuesView: (_COVARIANT,),
    abc.ItemsView: (_INVARIANT, _COVARIANT),
    abc.Generator: (_COVARIANT, _CONTRAVARIANT, _COVARIANT),
    abc.Coroutine: (_COVARIANT, _CONTRAVARIANT, _COVARIANT),
    abc.Awaitable: (_COVARIANT,),
    abc.AsyncIterable: (_COVARIANT,),
    abc.AsyncIterator: (_COVARIANT,),
    abc.AsyncGenerator: (_COVARIANT, _CONTRAVARIANT),
}  # type: tx.Dict[tx.Any, tx.Tuple[str, ...]]


# The runtime type of a PEP 585 alias (`#!python list[T]`, `#!python
# dict[str, T]`, `#!python collections.abc.Mapping[K, V]`, and a
# subscripted subclass of one, `#!python GL[int]`); `None` on Python 3.8,
# which has none.
_PEP585_ALIAS = getattr(types, "GenericAlias", None)


def _own_orig_bases(cls: type) -> tx.Tuple[tx.Any, ...]:
    """Return the parametrised bases `cls` itself was written with, else `()`.

    This reads the class's own namespace directly, rather than using
    attribute access, because attribute access would instead find a
    parent's `__orig_bases__`, which describes the parent's bases, not
    the class's own.
    """
    written = vars(cls).get("__orig_bases__")
    return written if isinstance(written, tuple) else ()


def _is_pep585_alias(hint: tx.Any) -> bool:
    """Report whether `hint` is a PEP 585 alias, `#!python list[T]`,
    not `List[T]`.

    A subclass of the runtime alias type counts too, such as
    `#!python collections.abc.Callable[[T], int]`.
    """
    return _PEP585_ALIAS is not None and isinstance(hint, _PEP585_ALIAS)


def _class_parameters(cls: type) -> tx.Tuple[tx.Any, ...]:
    """Return the type variables the class `cls` takes, in order.

    A [`Generic`][typing.Generic] subclass lists them itself, as
    `#!python __parameters__`. A class whose only generic bases are PEP
    585 aliases, such as `#!python class GL(list[T])` or
    `#!python class GD(dict[str, T])`, has no `Generic` in its MRO and
    lists none there, even though `#!python GL[int]` is still what an
    instance built from it records. Its parameters are instead
    collected the way `Generic` would collect them: the type variables
    its own PEP 585 bases mention, in order of first appearance.
    `#!python class Sub(GL[int])` mentions none and takes none, and a
    class written without a parametrised base takes none either.
    """
    params = getattr(cls, "__parameters__", None)
    if isinstance(params, tuple):
        return params
    collected = []  # type: tx.List[tx.Any]
    for base in _own_orig_bases(cls):
        if not _is_pep585_alias(base):
            continue
        for param in base.__parameters__:
            if not any(param is seen for seen in collected):
                collected.append(param)
    return tuple(collected)


def _generic_variances(origin: tx.Any) -> tx.Optional[tx.Tuple[str, ...]]:
    """Return the per-position variance of a generic's origin, or `None`.

    A standard-library origin, such as `#!python list` or
    [`collections.abc.Sequence`][], is looked up in the spec-derived
    [`_STDLIB_VARIANCE`][] table. A user-defined generic is instead read
    live off its `#!python __parameters__`, where each declared
    [`TypeVar`][typing.TypeVar] gives its own position's variance. A
    class whose generic bases are PEP 585 aliases, such as
    `#!python class GL(list[T])`, which has no `#!python __parameters__`
    of its own, is read the same way, off the type variables those
    bases mention ([`_class_parameters`][]). In every case, each type
    variable gives the variance it declares, not the variance of the
    slot it fills: an unflagged `T` is invariant wherever it goes, as in
    `#!python class GL(List[T])`, and a `T_co` written into
    `#!python list`'s invariant slot, which a type checker reports as an
    error in the class, is still taken at its word as covariant, the
    same way a checker treats it once the error has been reported.
    Anything else, such as an origin with a
    [`ParamSpec`][typing.ParamSpec] or
    [`TypeVarTuple`][typing.TypeVarTuple] in a parameter position, or
    one with no readable parameters at all, returns `#!python None`,
    leaving the caller to fall back to its own default.

    !!! example
        ```pycon
        >>> _generic_variances(list)
        ('invariant',)
        >>> import collections.abc
        >>> _generic_variances(collections.abc.Sequence)
        ('covariant',)
        >>> T_co = tx.TypeVar("T_co", covariant=True)
        >>> class Box(tx.Generic[T_co]): pass
        >>> _generic_variances(Box)
        ('covariant',)
        ```

    The result is memoised per origin. A class whose metaclass defines
    `#!python __eq__` without `#!python __hash__` cannot key the memo,
    and is read afresh on every call instead. The call cache cannot key
    such a class either, so a dispatch on one is resolved again on
    every call too.
    """
    try:
        hash(origin)
    except TypeError:
        # An unhashable class -- its metaclass defines `__eq__` alone -- can
        # key neither the memo nor the standard-library table, which holds
        # only hashable origins.
        return _declared_variances(origin)
    return _memoised_variances(origin)


@functools.lru_cache(maxsize=None)
def _memoised_variances(origin: tx.Any) -> tx.Optional[tx.Tuple[str, ...]]:
    """Compute [`_generic_variances`][] for a hashable origin, memoised."""
    if origin in _STDLIB_VARIANCE:
        return _STDLIB_VARIANCE[origin]
    return _declared_variances(origin)


def _declared_variances(origin: tx.Any) -> tx.Optional[tx.Tuple[str, ...]]:
    """Compute [`_generic_variances`][] for a non-stdlib origin.

    Each position gives the variance its own type variable declares;
    see [`_generic_variances`][] for the full rule.
    """
    if not isinstance(origin, type):
        return None
    params = _class_parameters(origin)
    if params and all(_is_plain_typevar(param) for param in params):
        return tuple(_typevar_variance(param) for param in params)
    return None


def _reads_declared_arguments(origin: tx.Any) -> bool:
    """Report whether a value's declared arguments are read against
    `origin[...]`.

    A parametrisation of a class whose parameters line up one per
    argument with a readable variance, whether a user generic such as
    `#!python Box[int]` or a standard-library one such as
    `#!python List[int]` or `#!python Sequence[int]`, is checked against
    what a value declares: the parametrisation an instance of a
    [`Generic`][typing.Generic] subclass was built from, or the one its
    class was written against. `#!python Type[C]`, a `TypedDict`, and
    the shape-typed `Tuple` and `Callable` keep their own checks
    instead, and so does a `ParamSpec` or `TypeVarTuple` generic.

    Both the value check and the call cache ask this question, so they
    always agree on which arguments can depend on a declaration.

    !!! example
        ```pycon
        >>> import collections.abc
        >>> _reads_declared_arguments(collections.abc.Sequence)
        True
        >>> _reads_declared_arguments(tuple), _reads_declared_arguments(type)
        (False, False)
        ```
    """
    return (
        _looks_like_class(origin)
        and origin is not type
        and not is_typeddict(origin)
        and _generic_variances(origin) is not None
    )


# --- eq_safenan --------------------------------------------------------


class _NaN:
    """The value that [`eq_safenan`][] maps every real NaN to."""

    def __repr__(self) -> str:
        return "<NaN>"


_NAN = _NaN()


def eq_safenan(x: tx.Any) -> tx.Any:
    """Map a value to a form that compares equal across NaNs.

    Since `#!python float("nan") != float("nan")`, comparing values
    that may contain NaN with `==` is unsafe. Apply this function to
    both operands before comparing them: every real NaN value is mapped
    to one shared sentinel, so that two NaNs compare equal, while every
    other value is returned unchanged.

    !!! note
        Only real numbers are recognised. A complex NaN is returned
        unchanged, and so still compares unequal to itself. A numpy
        scalar is recognised through its [`numbers.Real`][] ABC
        registration, so no numpy import is needed here.

    !!! example
        ```pycon
        >>> nan = float("nan")
        >>> nan == nan
        False
        >>> eq_safenan(nan) == eq_safenan(nan)
        True
        ```
    """
    if isinstance(x, numbers.Real) and math.isnan(x):
        return _NAN
    return x


# --- type <-> hint -----------------------------------------------------


_TYPE2HINT_NAMES = (
    (dict, "Dict"),
    (frozenset, "FrozenSet"),
    (list, "List"),
    (set, "Set"),
    (tuple, "Tuple"),
    (type, "Type"),
    (abc.AsyncGenerator, "AsyncGenerator"),
    (abc.AsyncIterable, "AsyncIterable"),
    (abc.AsyncIterator, "AsyncIterator"),
    (abc.Awaitable, "Awaitable"),
    (abc.Callable, "Callable"),
    (abc.Collection, "Collection"),
    (abc.Container, "Container"),
    (abc.Coroutine, "Coroutine"),
    (abc.Generator, "Generator"),
    (abc.Hashable, "Hashable"),
    (abc.ItemsView, "ItemsView"),
    (abc.Iterable, "Iterable"),
    (abc.Iterator, "Iterator"),
    (abc.KeysView, "KeysView"),
    (abc.Mapping, "Mapping"),
    (abc.MappingView, "MappingView"),
    (abc.MutableMapping, "MutableMapping"),
    (abc.MutableSequence, "MutableSequence"),
    (abc.MutableSet, "MutableSet"),
    (abc.Reversible, "Reversible"),
    (abc.Sequence, "Sequence"),
    (abc.Set, "AbstractSet"),
    (abc.Sized, "Sized"),
    (abc.ValuesView, "ValuesView"),
    (collections.ChainMap, "ChainMap"),
    (collections.Counter, "Counter"),
    (collections.OrderedDict, "OrderedDict"),
    (collections.defaultdict, "DefaultDict"),
    (collections.deque, "Deque"),
    (contextlib.AbstractContextManager, "ContextManager"),
    (contextlib.AbstractAsyncContextManager, "AsyncContextManager"),
    (re.Match, "Match"),
    (re.Pattern, "Pattern"),
)
"""
The type hint each non-subscriptable type maps to, by name.

This is an explicit table rather than a derivation from the type's own
name, because the capitalisation does not follow a simple rule:
`defaultdict` becomes `DefaultDict`, `frozenset` becomes `FrozenSet`,
and `abc.Set` becomes `AbstractSet`.
"""

_TYPE2HINT = {
    cls: getattr(tx, name)
    for cls, name in _TYPE2HINT_NAMES
    if hasattr(tx, name)
}
"""
[`_TYPE2HINT_NAMES`][], resolved against the running `typing_extensions`.

An entry whose hint the running version does not provide, such as
`ByteString` after its removal, is simply left out, so the type is
returned unchanged rather than raising an error at import time.
"""


def type2hint(x: tx.Any) -> tx.Any:
    """Convert a type to a subscriptable type hint.

    If `x` is a type that does not have `__class_getitem__`, this looks
    up its corresponding type hint; on Python 3.8, for example,
    `#!python type2hint(list)` returns [`typing.List`][tx.List].
    Anything else is returned unchanged.

    !!! example
        ```pycon
        >>> type2hint(frozenset)
        typing.FrozenSet
        >>> type2hint(3)  # not a type, so unchanged
        3
        ```
    """
    if issubscriptable(x):
        return x
    try:
        return _TYPE2HINT.get(x, x)
    except TypeError:
        # Unhashable: cannot be a key, so there is nothing to look up.
        return x


def _typing_spelling(hint: tx.Any) -> tx.Any:
    """Rewrite `list[int]` as `List[int]`, recursively, or leave
    `hint` unchanged.

    A parameterised builtin or ABC generic, such as `list[int]` or
    `dict[str, int]`, is a different object from its `typing` twin,
    `List[int]` or `Dict[str, int]`, and does not compare equal to it,
    so a registry keyed one way misses a query written the other way.
    Rewriting the new-style form into the `typing` spelling lets the two
    meet.

    The rewrite reaches all the way down, so a new-style generic nested
    inside a `Union`, `Optional`, `Annotated`, `Callable`, or another
    generic is rewritten too. `Literal` is left alone, since its
    arguments are values rather than types. `list[int]` does not exist
    before Python 3.9, so there is nothing to rewrite on that version.
    """
    origin = tx.get_origin(hint)
    if origin is None or any(origin is form for form in _LITERAL_FORMS):
        return hint
    args = tx.get_args(hint)
    try:
        if not args:
            # `tuple[()]` / `Tuple[()]` (the empty-tuple type) reports no
            # arguments from Python 3.11 on, yet still differs from a bare,
            # unparametrised `Tuple`. Rewrite only the subscripted empty-tuple
            # form to the `typing` spelling; a bare `Tuple` (a known bare form,
            # or one carrying no `__args__`) is returned unchanged.
            subscripted = (
                origin is tuple
                and not any(hint is form for form in _BARE_TUPLE_FORMS)
                and hasattr(hint, "__args__")
            )
            return tx.Tuple[()] if subscripted else hint
        if origin is tx.Annotated:
            # `(type, *metadata)`: rewrite the type, keep the metadata.
            inner = _typing_spelling(args[0])
            if inner == args[0]:
                return hint
            return tx.Annotated[(inner, *args[1:])]
        if origin is abc.Callable and len(args) == 2:
            # `(parameters, return)`: the parameters are a list of types,
            # or `...` / a `ParamSpec`, which are left whole.
            params, ret = args
            if isinstance(params, list):
                params = [_typing_spelling(each) for each in params]
            return tx.Callable[(params, _typing_spelling(ret))]
        spelled = tuple(_typing_spelling(arg) for arg in args)
        if origin in UNION_TYPES:
            return tx.Union[spelled]
        # A builtin/abc container gets its `typing` spelling; any other
        # generic (a user `Generic`) is rebuilt on its own origin.
        typing_origin = _TYPE2HINT.get(origin, origin)
        return typing_origin[spelled if len(spelled) > 1 else spelled[0]]
    except Exception:
        # A rebuild that fails -- a user origin that refuses these
        # arguments (`types.GenericAlias(SomeClass, (int,))` over a class with
        # no `__class_getitem__`), an exotic `Callable` form -- leaves the
        # hint as it was, to be matched by its origin instead.
        return hint


# --- MRO refinement ----------------------------------------------------


def mro_index(hint: tx.Any, value_type: type) -> tx.Optional[int]:
    """Return where `hint`'s class sits in `value_type`'s MRO, or `None`.

    This drives the MRO tie-break described in RFC 0001 §2.2: when two
    hints are otherwise incomparable, the one whose class names a more
    derived base of `value_type` wins, so the diamond
    `#!python D(B, C)` resolves to `B`, exactly as
    [`functools.singledispatch`][functools.singledispatch] does. Value
    dispatch reads this against an argument's runtime type;
    [`resolve_hint`][bagof.dispatchers.core.resolve_hint] reads it
    against a class query to break a tie between equally specific class
    keys.

    A refinement is defined only when the hint names a single ordinary
    class that is a nominal base of `value_type`. An
    [`Exact`][bagof.dispatchers.Exact]`[C]` hint counts as `C`. A bare
    class returns its own index in `#!python value_type.__mro__`. A
    bare, unparametrised alias counts as its origin class, in whichever
    spelling it was written, so `#!python List` and `#!python list`
    name the same position, as do `#!python Sequence` and
    `#!python collections.abc.Sequence`. A class that is not in the
    MRO at all, such as a `#!python Protocol` or an ABC satisfied
    structurally or by registration rather than by inheritance, gives
    no refinement and returns `#!python None`, as does a
    `#!python Union`, a `#!python Literal`, a `#!python type[...]`, or
    any other parametrised generic.

    !!! example
        ```pycon
        >>> class B: pass
        >>> class C: pass
        >>> class D(B, C): pass
        >>> mro_index(B, D) < mro_index(C, D)   # D resolves to B
        True
        ```

    Returns
    -------
    int or None
        The index of the hint's class in `#!python value_type.__mro__`,
        where `0` is `value_type` itself, or `#!python None` when no
        refinement applies.
    """
    hint = normalise_hint(hint)
    if is_exact(hint):
        cls = normalise_hint(exact_target(hint))
    else:
        cls = unwrap(hint, tx.Annotated)
    # Only a bare class names a position in the MRO. A `TypeVar` or any
    # non-class does not refine at all.
    if not _looks_like_class(cls):
        # A bare typing alias with no arguments is equivalent to its origin
        # class, whichever spelling it was written in -- `typing.Sequence`
        # and `collections.abc.Sequence` name the same MRO position. A
        # *parametrised* generic (`List[int]`, `type[C]`) carries arguments
        # that constrain more than the class does, so it names no position.
        if get_args_uw(cls):
            return None
        cls = get_origin_uw(cls)
        if not _looks_like_class(cls):
            # A union, literal or bare `Callable` has no plain-class origin.
            return None
    mro = getattr(value_type, "__mro__", ())
    for index, base in enumerate(mro):
        if base is cls:
            return index
    return None
