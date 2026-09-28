"""Small building blocks for reading and comparing type hints.

This module gathers version-safe wrappers around [`typing`][] that
answer sensibly, rather than raising, when handed a value that turns out
not to be a type at all. Alongside them sit helpers that strip away the
transparent wrappers, such as [`Annotated`][typing.Annotated] and
[`TypeVar`][typing.TypeVar], through which the subtype relation always
looks straight to what lies underneath.
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
    _OPTIONAL_FORMS,
    UNION_TYPES,
    NoneType,
    _is_newtype,
    _is_type_alias_type,
    canonical_typeddict,
    is_special_form,
    is_typeddict_marker,
    spellings,
)
from ._exact import Exact, exact_target, is_exact
from ._hint import Hint
from ._sentinels import UNSET


def _looks_like_class(x: tx.Any) -> bool:
    """Report whether `x` is a genuine class, not a parametrised generic alias.

    `#!python isinstance(list[int], type)` comes back `#!python True` on
    Python 3.9 and 3.10, so a bare `#!python isinstance(x, type)` check
    would mistake `#!python list[int]` for a class. A genuine class has
    no typing origin of its own, and checking for one distinguishes the
    two cases consistently across every supported version.
    """
    return isinstance(x, type) and tx.get_origin(x) is None


# --- origins and arguments ---------------------------------------------


def safe_get_origin(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Any:
    """Find a hint's origin, without raising when the hint has none.

    Passing `unwrap` strips a wrapper such as
    [`Annotated`][typing.Annotated] off the hint first, before its
    origin is read.

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
    """Find a hint's origin, unwrapping `Annotated` first.

    The hint itself is returned, rather than `None`, when it is not a
    generic type.
    """
    return safe_get_origin(hint, unwrap=tx.Annotated)


def safe_get_args(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Tuple[tx.Any, ...]:
    """Find a hint's type arguments, without raising on a plain type.

    An empty tuple comes back when the hint is not a generic type.
    Passing `unwrap` strips a wrapper such as
    [`Annotated`][typing.Annotated] off the hint first, before its
    arguments are read.
    """
    hint = _unwrap(hint, origin=unwrap)
    return tx.get_args(hint)


def get_args_uw(hint: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """Find a hint's type arguments, unwrapping `Annotated` first.

    An empty tuple comes back when the hint is not a generic type.
    """
    return safe_get_args(hint, unwrap=tx.Annotated)


# --- unwrapping --------------------------------------------------------


def unwrap(hint: tx.Any, origin: tx.Any = (tx.Annotated,)) -> tx.Any:
    """Strip a hint down to its argument, when its origin is one of `origin`.

    When [`TypeVar`][typing.TypeVar] is listed among the origins to
    unwrap, a type variable is replaced by whichever of its default, the
    union of its constraints, or its bound comes first and actually
    exists, tried in that order.

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

# The bare, unparametrised tuple spellings. Used to tell a bare `Tuple` /
# `tuple` from a subscripted alias whose arguments happen to be empty -- the
# empty-tuple type `Tuple[()]` (mirrors `_relation._is_subscripted_tuple`,
# kept here to avoid importing from `_relation`, which imports this module).
_BARE_TUPLE_FORMS = spellings("Tuple") + (tuple,)

# A generous cap: each pass either resolves one wrapper (strictly reducing
# the hint) or leaves it untouched, so a handful of passes always settles.
_MAX_NORMALISE_STEPS = 100


def resolve_alias(hint: tx.Any) -> tx.Any:
    """Resolve a PEP 695 `type X = ...` alias to the hint it stands for.

    A bare alias resolves to its stored value directly. A subscripted
    generic alias, such as `#!python L[int]` for
    `#!python type L[T] = list[T]`, has its type arguments substituted in
    first. An alias standing for another alias is followed all the way
    through to the end, and a reference cycle stops resolution rather
    than recursing without limit. A hint that is not an alias at all
    comes back unchanged.
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


def resolve_newtype(hint: tx.Any) -> tx.Any:
    """Resolve a [`NewType`][typing.NewType] to its supertype, recursively.

    A `NewType` gives an existing type a distinct name, and dispatch
    treats it exactly as that underlying type, so this follows the chain
    down to it. A hint that is not a `NewType` at all comes back
    unchanged.
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


# Every bare `Type` spelling, used to recognise `Exact[Type]` before it is
# subscripted.
_TYPE_FORMS = spellings("Type")


def _lower_exact(hint: tx.Any) -> tx.Any:
    """Rewrite an outer `Exact` around `Type` or `Hint` as an inner one.

    `Exact[C]` marks the type `C` as the exact match for a value, but the
    same intent can be written the other way round, wrapping the whole of
    a `Type` or `Hint` form. This rewrites the outer spelling to the
    inner one, so that the relation only ever meets the inner form.
    `#!python Exact[Type[int]]` becomes `#!python Type[Exact[int]]`,
    `#!python Exact[Hint[int]]` becomes `#!python Hint[Exact[int]]`, and a
    bare `#!python Exact[Type]` becomes `#!python Exact[type]`. Any other
    `Exact` hint, including a plain `#!python Exact[int]`, is already in
    its canonical form and comes back unchanged.
    """
    target = exact_target(hint)
    origin = tx.get_origin(target)
    args = tx.get_args(target)
    if origin is type and args:
        return tx.Type[Exact[args[0]]]
    if origin is Hint and args:
        return Hint[Exact[args[0]]]
    if any(target is form for form in _TYPE_FORMS):
        # A bare `Exact[Type]`: an exact match for any class, which is just
        # an exact match for `type` itself.
        return Exact[type]
    return hint


def normalise_hint(hint: tx.Any) -> tx.Any:
    """Put a hint into its canonical form.

    A bare [`None`][] is understood as a hint to mean
    [`NoneType`][types.NoneType], and is replaced by it. A PEP 695
    `type X = ...` alias is resolved to the hint it stands for, a
    [`NewType`][typing.NewType] is resolved to its supertype, and the
    transparent qualifiers [`Required`][typing.Required],
    [`NotRequired`][typing.NotRequired], [`ReadOnly`][typing.ReadOnly],
    [`Final`][typing.Final], and [`ClassVar`][typing.ClassVar] are
    unwrapped down to the hint each one wraps. A bare, unsubscripted
    [`Optional`][typing.Optional] is read as a bare
    [`Union`][typing.Union], since both stand for "some union", and an
    [`Exact`][bagof.dispatchers.Exact] wrapping a whole
    [`Type`][typing.Type] or [`Hint`][bagof.dispatchers.Hint] is rewritten
    to carry the `Exact` on the inner type instead, so that
    `#!python Exact[Type[int]]` becomes `#!python Type[Exact[int]]`.

    These steps repeat until the hint stops changing, so an alias that
    expands into a qualified `NewType` is resolved all the way through
    rather than only one layer deep. Any hint none of these steps
    touches comes back unchanged.

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
        if is_exact(hint):
            hint = _lower_exact(hint)
        elif any(hint is form for form in _OPTIONAL_FORMS):
            # A bare `Optional`, with no argument, means the same as a bare
            # `Union`: both stand for "some union".
            hint = tx.Union
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
        itself; here, the bare marker itself counts too.
    """
    if is_typeddict_marker(cls):
        return True
    return tx.is_typeddict(cls)


@functools.lru_cache(maxsize=None)
def typeddict_required_keys(cls: tx.Any) -> tx.FrozenSet[str]:
    """Find the required keys of a [`TypedDict`][tx.TypedDict].

    When the class carries `__required_keys__`, that attribute is read
    directly, since it is the only source that correctly accounts for
    [`Required`][typing.Required] and [`NotRequired`][typing.NotRequired]
    (whether or not nested inside [`Annotated`][typing.Annotated]) and
    for inheriting from bases declared with a different `total=`.

    When the class has no `__required_keys__` at all, this falls back to
    reading `__total__` instead: [`typing.TypedDict`][] only gained
    `__required_keys__` in Python 3.9, and before that a key's
    requiredness came from the class's own `total=` alone, since
    per-key `Required` and `NotRequired` did not yet exist.

    !!! warning
        On older Pythons, a [`typing.TypedDict`][] that mixes `total=`
        settings across its bases can only be read through the
        subclass's own `total=`, which is not correct for every key.
        A subclass has no `__orig_bases__`, and its `__mro__` reaches
        only [`dict`][], so the standard library records neither which
        class declared a given key nor a usable link back to that base,
        leaving the per-key answer simply unrecoverable.

        This error can go in either direction. A `total=True` key
        inherited into a `total=False` subclass is reported as optional,
        so a value missing it is wrongly accepted, as when `{}` is
        matched against a shape that actually requires that key. A
        `total=False` key inherited into a `total=True` subclass is
        reported as required, so an otherwise valid value is wrongly
        rejected. Use [`typing_extensions.TypedDict`][tx.TypedDict],
        which reimplements the class precisely and records
        `__required_keys__` correctly on every version, whenever this
        distinction matters.

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

    The mapping covers every key the class declares, both its own and
    those inherited from [`TypedDict`][tx.TypedDict] bases, each paired
    with the hint written for it. Any [`Required`][typing.Required] or
    [`NotRequired`][typing.NotRequired] qualifier is left in place on the
    hint; reading it off is [`typeddict_required_keys`][]'s job instead,
    and a caller who only wants the value type can simply let the
    relation look straight through the qualifier.

    A string annotation is resolved against the class's own module where
    possible, using [`typing.get_type_hints`][tx.get_type_hints]. When a
    forward reference cannot be resolved, the raw name is kept for that
    one field alone, so the caller sees the name rather than nothing at
    all, while every sibling field that can be resolved still is.

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
    """Resolve a TypedDict's fields one at a time, keeping whatever resolves.

    [`typeddict_field_hints`][] falls back to this in place of its
    otherwise all-or-nothing resolution. Each annotation is evaluated
    separately against the class's own module, every one that resolves
    is kept as resolved, and every one that does not is left as its raw
    string name, for the caller to recognise and skip.
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
    """Evaluate one annotation against `globalns`, or fall back to its
    raw name.

    A `raw` value that is already a hint object, an annotation that was
    never quoted, comes back unchanged. A string or
    [`ForwardRef`][typing.ForwardRef] is evaluated the same way
    [`typing.get_type_hints`][tx.get_type_hints] would; when the name it
    names is not defined there, the raw, unevaluated form is returned
    instead, so this one field can be skipped without losing the rest of
    the class.
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
    """Collect every original base of a type, including the type itself."""
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
    """Report whether `subcls` is a subclass of `cls`, without ever raising.

    !!! warning
        When `cls` is a [`TypedDict`][tx.TypedDict], this looks at
        `subcls`'s `__orig_bases__` rather than its `__bases__`. A
        plain [`dict`][] is not treated as a subclass of a
        [`TypedDict`][tx.TypedDict] here; the relation only holds the
        other way round.

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
    """Report whether `obj` is an instance of `cls`, without ever raising.

    !!! warning
        A [`TypedDict`][tx.TypedDict] cannot be instance-checked at
        all: Python refuses `#!python isinstance(value, SomeTypedDict)`
        outright, and a `TypedDict` leaves no trace on the dict it
        describes for anything to recognise at runtime. This function
        therefore answers [`False`][] for one; check the shape of the
        dict directly instead.

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
        answers [`True`][] for [`TypedDict`][tx.TypedDict] and its
        subclasses too, even though none of them is technically a type.

    !!! note
        A typing construct such as [`Any`][typing.Any],
        [`Union`][typing.Union], or [`Literal`][typing.Literal] is never
        subclassable, whichever Python version is running. Some of these
        become classes on recent Pythons, `Any` from 3.11 and `Union`
        from 3.14, so `#!python isinstance(hint, type)` alone would
        answer differently across the versions this package supports.
    """
    if is_special_form(cls):
        return False
    if is_typeddict_marker(cls):
        return True
    return _looks_like_class(cls)


def issubscriptable(x: tx.Any) -> bool:
    """Report whether `x` can be subscripted, written as `#!python x[...]`.

    This comes back `#!python True` when `x` is a type defining
    `__class_getitem__`, or an instance defining `__getitem__`, and
    `#!python False` for everything else.
    """
    is_class = _looks_like_class(x)
    if is_class and hasattr(x, "__class_getitem__"):
        return True
    if not is_class and hasattr(x, "__getitem__"):
        return True
    return False


# --- concrete types ----------------------------------------------------


def get_concrete_type(hint: tx.Any, fallback: type = UNSET) -> tx.Type[tx.Any]:
    """Find a concrete, instantiable type standing in for a type hint.

    Any [`Annotated`][typing.Annotated] wrapper is removed from the hint
    first. If what remains has an origin, that origin is used. If it is
    instead a [`TypeVar`][typing.TypeVar], the first of the following
    that applies is used: its default, the first of its constraints, its
    bound, or the `fallback` type when one was given. Failing all of
    those, a [`TypeError`][] is raised.

    Whatever is arrived at this way is returned as is when it turns out
    to be concrete and non-abstract. Anything else falls back to
    `fallback`, when one was given, or otherwise raises a
    [`TypeError`][].

    !!! note
        A constrained type variable has no single concrete type of its
        own, since it stands for the union of its constraints, so its
        first constraint is taken as a stand-in for it.

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
    """Report whether a hint is a class that could actually be instantiated."""
    if is_special_form(hint):
        # `Union` is a class from python 3.14 on, but instantiating it
        # is still meaningless.
        return False
    return safe_isinstance(hint, type) and not inspect.isabstract(hint)


def _first_concrete_constraint(hint: tx.Any) -> tx.Optional[type]:
    """Find the first concrete constraint of a constrained typevar, if any."""
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
    """Report whether `param` is an ordinary `TypeVar`, not a `ParamSpec`
    or `*Ts`.

    A [`ParamSpec`][typing.ParamSpec] or
    [`TypeVarTuple`][typing.TypeVarTuple] carries no variance of its own,
    so a generic with one occupying a parameter position has no
    per-position variance there for anything to read.
    """
    return isinstance(param, tx.TypeVar) and not (
        _NON_TYPE_PARAMS and isinstance(param, _NON_TYPE_PARAMS)
    )


def _typevar_variance(tv: tx.Any) -> str:
    """Find the variance a [`TypeVar`][typing.TypeVar] declares, per PEP 484.

    `#!python "covariant"` comes back for a
    `#!python TypeVar(..., covariant=True)`, `#!python "contravariant"`
    for a `#!python TypeVar(..., contravariant=True)`, and
    `#!python "invariant"` for everything else. That invariant case
    covers both an unflagged type variable, which PEP 484 makes
    invariant by default, and a PEP 695 `#!python class Box[T]`
    variable, whose variance a type checker infers and which cannot be
    read at runtime at all, so it is simply taken to be invariant here.

    Every attribute is read with [`getattr`][] rather than directly,
    because the `TypeVar` family's shape differs across Python 3.8
    through 3.13 and between [`typing`][] and `typing_extensions`.

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
    """List the type variables that class `cls` takes, in order.

    A [`Generic`][typing.Generic] subclass lists its own type variables
    as `#!python __parameters__`. A class whose only generic bases are
    PEP 585 aliases, such as `#!python class GL(list[T])` or
    `#!python class GD(dict[str, T])`, has no `Generic` in its MRO and so
    lists none there, even though `#!python GL[int]` is still what an
    instance built from it records. Its parameters are instead collected
    the way `Generic` itself would: the type variables its own PEP 585
    bases mention, in the order they first appear. A class such as
    `#!python class Sub(GL[int])`, which mentions no type variable of its
    own, takes none, and neither does a class with no parametrised base
    at all.
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
    """Find the per-position variance of a generic's origin, or `None`.

    A standard-library origin, such as `#!python list` or
    [`collections.abc.Sequence`][], is looked up directly in the
    spec-derived [`_STDLIB_VARIANCE`][] table. A user-defined generic is
    instead read live off its `#!python __parameters__`, where each
    declared [`TypeVar`][typing.TypeVar] gives the variance of its own
    position. A class whose generic bases are PEP 585 aliases, such as
    `#!python class GL(list[T])`, and which therefore has no
    `#!python __parameters__` of its own, is read the same way, off the
    type variables those bases mention ([`_class_parameters`][]). In
    every case, a type variable contributes the variance it declares for
    itself, not the variance of the slot it happens to fill. An
    unflagged `T` is invariant wherever it is written, as in
    `#!python class GL(List[T])`. A `T_co` written into
    `#!python list`'s invariant slot is something a type checker would
    flag as an error in the class, yet it is still taken at its word as
    covariant here, the same way a checker treats it once that error has
    already been reported. Anything else, such as an origin with a
    [`ParamSpec`][typing.ParamSpec] or
    [`TypeVarTuple`][typing.TypeVarTuple] occupying a parameter position,
    or one with no readable parameters at all, returns
    `#!python None`, leaving the caller to apply its own default.

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
    `#!python __eq__` without `#!python __hash__` cannot key that memo,
    and is read afresh on every call instead. The dispatch call cache
    cannot key such a class either, so a dispatch involving one is
    likewise resolved anew on every call.
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
    """Compute [`_generic_variances`][] for a hashable origin, and cache it."""
    if origin in _STDLIB_VARIANCE:
        return _STDLIB_VARIANCE[origin]
    return _declared_variances(origin)


def _declared_variances(origin: tx.Any) -> tx.Optional[tx.Tuple[str, ...]]:
    """Compute [`_generic_variances`][] for an origin outside the standard
    library.

    Each position takes the variance its own type variable declares;
    see [`_generic_variances`][] for the complete rule.
    """
    if not isinstance(origin, type):
        return None
    params = _class_parameters(origin)
    if params and all(_is_plain_typevar(param) for param in params):
        return tuple(_typevar_variance(param) for param in params)
    return None


def _reads_declared_arguments(origin: tx.Any) -> bool:
    """Report whether a value's declared arguments are checked against
    `origin[...]`.

    A parametrisation of a class whose parameters line up one to one
    with a readable variance, whether a user-defined generic such as
    `#!python Box[int]` or a standard-library one such as
    `#!python List[int]` or `#!python Sequence[int]`, is checked against
    what a value itself declares. That declaration is the parametrisation
    an instance of a [`Generic`][typing.Generic] subclass was built with,
    or the one its class was written against. `#!python Type[C]`, a
    `TypedDict`, and the
    shape-typed `Tuple` and `Callable` each keep their own separate
    checks instead, and so does any generic parameterised by a
    `ParamSpec` or `TypeVarTuple`.

    Both the value check and the call cache rely on this same question,
    so the two always agree on which arguments can even depend on a
    declaration.

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
        and origin is not Hint
        and not is_typeddict(origin)
        and _generic_variances(origin) is not None
    )


# --- eq_safenan --------------------------------------------------------


class _NaN:
    """The shared value [`eq_safenan`][] maps every real NaN onto."""

    def __repr__(self) -> str:
        return "<NaN>"


_NAN = _NaN()


class _ComplexNaN:
    """The marker tagging the key [`eq_safenan`][] builds for a complex NaN."""

    def __repr__(self) -> str:
        return "<complexNaN>"


_COMPLEX_NAN = _ComplexNaN()


def eq_safenan(x: tx.Any) -> tx.Any:
    """Map a value to a form that compares equal to any other NaN.

    Comparing values that might be NaN with plain `==` is unsafe, since
    `#!python float("nan") != float("nan")` even for the very same
    object. Applying this function to both operands before comparing
    them sidesteps that: every real NaN maps onto one shared sentinel,
    so two NaNs end up comparing equal, while every other value passes
    through unchanged.

    !!! note
        A complex number carrying a NaN in either its real or its
        imaginary part is normalised component by component, so a
        NaN-bearing complex compares equal to itself, and two complex
        values compare equal when their non-NaN parts match and NaN
        falls in the same position. A numpy scalar is recognised through
        its registration under the [`numbers.Real`][] ABC, so this needs
        no numpy import of its own.

    !!! example
        ```pycon
        >>> nan = float("nan")
        >>> nan == nan
        False
        >>> eq_safenan(nan) == eq_safenan(nan)
        True
        >>> cnan = complex(float("nan"), 2.0)
        >>> cnan == cnan
        False
        >>> eq_safenan(cnan) == eq_safenan(cnan)
        True
        ```
    """
    if isinstance(x, numbers.Real) and math.isnan(x):
        return _NAN
    # Mirror numpy.array_equal(a, b, equal_nan=True): normalise a complex
    # NaN part by part. numbers.Real is a subclass of numbers.Complex, so
    # the real check above runs first and only genuine complex values reach
    # here; recursing on each part reuses the real path, which also covers a
    # numpy complex scalar without importing numpy.
    if isinstance(x, numbers.Complex) and (
        math.isnan(x.real) or math.isnan(x.imag)
    ):
        return (_COMPLEX_NAN, eq_safenan(x.real), eq_safenan(x.imag))
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
The type hint each non-subscriptable type maps to, named explicitly.

This is written out as a table rather than derived from each type's own
name, because the capitalisation follows no simple rule: `defaultdict`
becomes `DefaultDict`, `frozenset` becomes `FrozenSet`, and `abc.Set`
becomes `AbstractSet`.
"""

_TYPE2HINT = {
    cls: getattr(tx, name)
    for cls, name in _TYPE2HINT_NAMES
    if hasattr(tx, name)
}
"""
[`_TYPE2HINT_NAMES`][], resolved against the running `typing_extensions`.

An entry whose hint the running version no longer provides, such as
`ByteString` after its removal, is simply left out, so that type is
returned unchanged rather than raising an error at import time.
"""


def type2hint(x: tx.Any) -> tx.Any:
    """Convert a type to a subscriptable type hint standing in for it.

    When `x` has no `__class_getitem__` of its own, this looks up the
    type hint that corresponds to it; on Python 3.8, for instance,
    `#!python type2hint(list)` returns [`typing.List`][tx.List].
    Anything else comes back unchanged.

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
    """Rewrite `list[int]` as `List[int]`, recursively, leaving other
    hints alone.

    A parameterised builtin or ABC generic, such as `list[int]` or
    `dict[str, int]`, is a different object from its `typing` twin,
    `List[int]` or `Dict[str, int]`, and the two do not compare equal, so
    a registry keyed under one spelling would miss a query written under
    the other. Rewriting the new-style form into the `typing` spelling
    lets the two meet on common ground.

    The rewrite reaches all the way down into a hint's structure, so a
    new-style generic nested inside a `Union`, `Optional`, `Annotated`,
    `Callable`, or another generic is rewritten too. `Literal` is left
    untouched, since its arguments are values rather than types.
    `list[int]` does not exist before Python 3.9, so on that version
    there is nothing here to rewrite in the first place.
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
    """Find where `hint`'s class sits in `value_type`'s MRO, or `None`.

    This is what drives the MRO tie-break described in RFC 0001 §2.2:
    when two hints are otherwise incomparable, the one whose class names
    a more derived base of `value_type` wins, so a diamond such as
    `#!python D(B, C)` resolves to `B`, exactly as
    [`functools.singledispatch`][functools.singledispatch] does. Value
    dispatch checks this against an argument's runtime type, while
    [`resolve_hint`][bagof.dispatchers.core.resolve_hint] checks it
    against a class query to break a tie between equally specific class
    keys.

    A refinement is only defined when the hint names a single ordinary
    class that is a nominal base of `value_type`. An
    [`Exact`][bagof.dispatchers.Exact]`[C]` hint is treated as `C`
    itself. A bare class returns its own index in
    `#!python value_type.__mro__`. A bare, unparametrised alias is
    treated as its origin class regardless of which spelling it was
    written in, so `#!python List` and `#!python list` name the same
    position, as do `#!python Sequence` and
    `#!python collections.abc.Sequence`. A class that does not appear in
    the MRO at all, such as a `#!python Protocol` or an ABC satisfied
    structurally or through registration rather than inheritance, gives
    no refinement and returns `#!python None`. The same is true of a
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
