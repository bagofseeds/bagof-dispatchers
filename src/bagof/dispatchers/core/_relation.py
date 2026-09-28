"""The subtype relation over hints: [`issubhint`][] and [`ishintstance`][].

Together, these two functions decide everything dispatch needs to know
about how hints relate: [`issubhint`][] orders two hints by which one
describes a broader set of values, and [`ishintstance`][] tells whether
one particular value falls inside the set a hint describes. Every other
name in this module exists to make one of those two questions
answerable for a specific kind of hint, from a plain class through
unions, literals, generics, protocols, and `TypedDict`.
"""

# stdlib
import builtins
import dataclasses
import functools
import inspect
import re
import sys
import types
import warnings
import weakref
from collections import abc

# dependencies
import typing_extensions as tx

# local
from ._compat import (
    _UNPACK_FORMS,
    UNION_TYPES,
    UnknownHintWarning,
    is_plausible_hint,
    is_special_form,
    is_typeddict_marker,
    spellings,
)
from ._exact import exact_target, is_exact
from ._introspect import (
    _CONTRAVARIANT,
    _COVARIANT,
    _NON_TYPE_PARAMS,
    _all_orig_bases,
    _class_parameters,
    _generic_variances,
    _is_pep585_alias,
    _is_plain_typevar,
    _looks_like_class,
    _own_orig_bases,
    _reads_declared_arguments,
    eq_safenan,
    get_args_uw,
    get_origin_uw,
    is_typeddict,
    normalise_hint,
    safe_get_args,
    safe_get_origin,
    safe_issubclass,
    typeddict_field_hints,
    typeddict_required_keys,
    unwrap,
)

# --- known non-class forms ---------------------------------------------

# Bottom types: a value is never one, and only a bottom is a sub-hint of a
# bottom.
_NEVER_FORMS = spellings("Never") + spellings("NoReturn")

# Forms that stand in, for dispatch, for an ordinary class.
_LITERALSTRING_FORMS = spellings("LiteralString")
_TYPEGUARD_FORMS = spellings("TypeGuard") + spellings("TypeIs")

# `Concatenate` shapes a `Callable` parameter list can carry: a fixed,
# contravariant prefix followed by an open `ParamSpec` tail (RFC 11.1).
_CONCATENATE_FORMS = spellings("Concatenate")

# Every spelling of the forms the relation reads by identity. On 3.8-3.10
# `typing_extensions` ships its own `Any` / `Literal`, distinct objects from
# `typing`'s, so a single-object `is` check silently misses the other.
_ANY_FORMS = spellings("Any")
_LITERAL_FORMS = spellings("Literal")

# Bare, unparametrised tuple spellings: the plain `tuple` class and every
# `Tuple` special-form object. Used to tell a bare tuple (which accepts any
# parametrisation) from the empty-tuple type `Tuple[()]`, which does not.
_BARE_TUPLE_FORMS = spellings("Tuple") + (tuple,)


def _is_any(hint: tx.Any) -> bool:
    """Report whether `hint` is `Any`, in any spelling."""
    return any(hint is form for form in _ANY_FORMS)


def _is_literal(origin: tx.Any) -> bool:
    """Report whether `origin` is the `Literal` form, in any spelling."""
    return any(origin is form for form in _LITERAL_FORMS)


def _is_never(hint: tx.Any) -> bool:
    """Report whether `hint` is a bottom type, `Never` or `NoReturn`."""
    return any(hint is form for form in _NEVER_FORMS)


def _is_unpack(hint: tx.Any) -> bool:
    """Report whether `hint` is an `Unpack[...]`, in any spelling.

    On Python 3.11, `typing.Unpack is not typing_extensions.Unpack`, and
    the star syntax `Tuple[int, *Ts]` produces one of those spellings
    while `tx.Unpack[Ts]` produces the other, so the origin has to be
    tested against both.
    """
    return any(safe_get_origin(hint) is form for form in _UNPACK_FORMS)


def _is_unpacked_typevartuple(hint: tx.Any) -> bool:
    """Report whether `hint` is `Unpack[Ts]` for a `TypeVarTuple` `Ts`."""
    if not _is_unpack(hint):
        return False
    args = tx.get_args(hint)
    return bool(args) and isinstance(args[0], tx.TypeVarTuple)


def _known_form(hint: tx.Any) -> tx.Any:
    """Map a known non-class form to the class it dispatches as.

    `LiteralString` dispatches as [`str`][], and `TypeGuard[...]` or
    `TypeIs[...]` dispatches as [`bool`][]. An unpacked
    `TypeVarTuple`, `*Ts`, stands for an open run of `Any` elements, so
    taken on its own, as when an `*args: *Ts` tail is read as a single
    slot, it dispatches as [`Any`][typing.Any]. Every other hint passes
    through unchanged.
    """
    if any(hint is form for form in _LITERALSTRING_FORMS):
        return str
    if any(hint is form for form in _TYPEGUARD_FORMS):
        return bool
    origin = safe_get_origin(hint)
    if any(origin is form for form in _TYPEGUARD_FORMS):
        return bool
    if _is_unpacked_typevartuple(hint):
        return tx.Any
    return hint


def _typevar_upper(tv: tx.Any) -> tx.Any:
    """Find the upper bound a `TypeVar` dispatches as.

    This is the variable's bound, the union of its constraints, or
    [`Any`][typing.Any], whichever applies, and never its PEP 696
    default, which is a static-checker fallback that dispatch ignores
    outright: `#!python TypeVar("T", bound=float, default=int)`
    dispatches as `float`, not `int`.
    """
    constraints = getattr(tv, "__constraints__", ())
    if constraints:
        return tx.Union[constraints]
    bound = getattr(tv, "__bound__", None)
    if bound is not None:
        return bound
    return tx.Any


def _equivalent(a: tx.Any, b: tx.Any) -> bool:
    """Report whether two hints accept exactly the same values."""
    return issubhint(a, b) and issubhint(b, a)


_WARNED_UNKNOWN = set()  # type: set


def _warn_key(hint: tx.Any) -> tx.Any:
    """Build a stable, hashable key that identifies an unknown form.

    A future form and every hint built from it, such as `Unpack[Ts]` and
    `Unpack[Us]`, share a single origin, so the key is that origin, or
    the form itself when it has none, which deduplicates the warning per
    form rather than per individual `repr`.
    """
    origin = safe_get_origin(hint)
    try:
        hash(origin)
    except TypeError:
        # An unhashable origin cannot key the set, so fall back to the form's
        # own type.
        origin = None
    return origin if origin is not None else id(type(hint))


def _warn_unknown(hint: tx.Any) -> None:
    """Warn, once per form, that `hint` is unrecognised and treated as
    `Any`.
    """
    key = _warn_key(hint)
    if key in _WARNED_UNKNOWN:
        return
    _WARNED_UNKNOWN.add(key)
    try:
        shown = repr(hint)
    except Exception:
        # A hint whose own `repr` raises still has to be named in the warning.
        shown = object.__repr__(hint)
    warnings.warn(
        f"Type hint {shown} is not recognised; treating it as `Any` for "
        "dispatch.",
        UnknownHintWarning,
        stacklevel=3,
    )


def _not_a_hint_message(obj: tx.Any) -> str:
    """Compose the error text for an object that is not a usable type hint."""
    if isinstance(obj, str):
        # A bare string is a forward reference, which cannot be resolved
        # without the namespace it was written in -- not available here.
        return (
            f"Cannot use the string {obj!r} as a type hint: a forward "
            "reference needs the namespace it was written in to be "
            "resolved, which is not available here. Pass the type itself, "
            "not its name."
        )
    kind = type(obj).__name__
    return (
        f"Expected a type hint, but got {obj!r} (of type {kind}), which is "
        "not a type or a typing construct."
    )


def ishintstance(obj: tx.Any, hint: tx.Any) -> bool:
    """Report whether `obj` belongs to the set of values `hint` describes.

    `ishintstance` extends [`isinstance`][] to the full vocabulary of
    type hints rather than only classes, so its second argument can be
    a union, a literal, a generic, a protocol, or any other construct
    this package understands. What counts as membership then depends on
    the specific kind of hint given.

    A [`type`][] or [`Type[...]`][tx.Type] hint requires `obj` to be a
    class itself, and a valid subclass of whatever the hint names. A
    [`Literal`][tx.Literal] hint requires `obj` to equal one of its
    listed values, with the type checked alongside the value, so
    `#!python True` does not satisfy `#!python Literal[1]` even though
    `#!python True == 1` in Python. A [`Union`][tx.Union] hint is
    satisfied whenever `obj` satisfies any one of its members.

    A [`TypedDict`][tx.TypedDict] hint describes the shape of a mapping
    rather than its class: `obj` must be a [`dict`][] carrying every
    required key, with each key it does carry holding a value of that
    field's declared type. Keys beyond the ones the `TypedDict` declares
    are handled according to that `TypedDict`'s own policy, since an
    ordinary open one allows them, a `closed=True` one rejects them, and
    one declared with `extra_items=` checks each of them against that
    type. A field that is itself a nested `TypedDict` or container is
    checked the same way, recursively.

    A [`runtime_checkable`][typing.runtime_checkable] protocol that
    declares data members, such as one with `#!python name: str`, is
    checked structurally, in the same spirit as [`isinstance`][].
    `obj`'s class either names the protocol among its own bases, or
    every declared data member is present on `obj`, whether set directly
    on the instance or supplied by the class through an annotation, a
    class attribute, a property, or a slot, never through a
    `#!python ClassVar`; every method, in either case, must also be
    defined somewhere on the class. An annotated member counts as present
    even on an
    instance that never actually set it, a case where [`isinstance`][]
    itself would find nothing, because that mirrors how a type checker
    reads the same annotation and keeps this check aligned with
    [`issubhint`][]. A member the protocol declares as a
    `#!python ClassVar` is read only from the class, and only a
    `#!python ClassVar` annotation is treated that way. None of this
    lookup runs the value's own code: a property is never called, and
    `__getattr__` is never invoked. A protocol declaring only methods,
    with no data members at all, is instead checked directly against the
    value's type.

    A parametrised generic, such as `#!python List[int]` or
    `#!python Box[int]`, first requires `obj` to be an instance of its
    class. It only goes on to check the type arguments when `obj`
    actually declares them, which happens for an instance built as
    `#!python Box[int]()` or for one of a class written against a
    parametrised base such as `#!python class IntList(List[int])`. Those
    declared arguments are then compared the same way
    [`issubhint`][] compares them, so an invariant position demands
    exactly the same type. `#!python Box[int]()` does not satisfy
    `#!python Box[object]`, and `#!python IntList` does not satisfy
    `#!python List[object]`, even though it does satisfy
    `#!python List[Any]`, plain `#!python list`, and
    `#!python Sequence[object]`.

    Every other hint falls back to `#!python issubhint(type(obj), hint)`.

    !!! note
        `#!python Box[int]()` only writes its record onto the instance
        once `__init__` returns, so a call dispatched on `self` from
        inside `__init__` still sees a value that declares nothing yet.
        An instance that has no way to hold that record at all, such as
        one of a class with `__slots__` and no `__dict__`, or a frozen
        dataclass, never declares its arguments either.

    !!! warning
        A container's item types are never checked here. A plain
        `#!python [1, 2]` declares no type arguments of its own, so this
        function treats it as a valid `#!python List[str]`; Python
        itself refuses `#!python isinstance(x, list[int])` for the same
        reason. Checking the items would mean iterating them, which is
        left to the caller to decide on; `bagof.validators` is what does
        it.

    !!! example
        ```pycon
        >>> ishintstance(1, int)
        True
        >>> ishintstance(1, Literal[1, 2])
        True
        >>> ishintstance(bool, Type[int])
        True
        ```
    """
    hint = normalise_hint(hint)
    # `Exact[C]` first, before the `Annotated` metadata is unwrapped: the
    # value's type must be exactly `C`.
    if is_exact(hint):
        return type(obj) is normalise_hint(exact_target(hint))
    hint = _known_form(hint)
    if _is_never(hint):
        # A bottom type describes no value.
        return False
    if _is_any(hint):
        return True
    # Resolve typevars to their bound/constraints (never their default), so
    # a typevar behaves exactly like the hint it stands for.
    hint = unwrap(hint, tx.Annotated)
    while isinstance(hint, tx.TypeVar):
        upper = normalise_hint(_typevar_upper(hint))
        # Exactness can be reached through a bound (`TypeVar(bound=Exact[C])`),
        # so re-check before the `Annotated` wrapper is stripped.
        if is_exact(upper):
            return type(obj) is normalise_hint(exact_target(upper))
        hint = unwrap(upper, tx.Annotated)
    hint = _known_form(hint)
    if _is_any(hint):
        return True
    origin_uw = get_origin_uw(hint)
    if origin_uw is type:
        return _ishintstance_type(obj, hint)
    if _is_literal(origin_uw):
        return _ishintstance_literal(obj, hint)
    if origin_uw in UNION_TYPES:
        args = get_args_uw(hint)
        if args:
            return any(ishintstance(obj, arg) for arg in args)
    if is_typeddict_marker(origin_uw):
        # The bare `TypedDict` marker: a value is one iff its type is a
        # `TypedDict` (`safe_issubclass` reads it structurally). Without
        # this it would fall to the opaque rule and accept everything.
        # The marker names no fields, so there is no shape to check; a plain
        # `dict` is not one of it.
        return safe_issubclass(type(obj), origin_uw)
    if is_typeddict(origin_uw):
        # A concrete `TypedDict` describes the *shape* of a mapping, so a
        # value is one when it has the declared keys with the declared value
        # types -- not when its type is nominally the `TypedDict` (a `dict`
        # literal never is). The bare marker was handled just above, so only
        # a `TypedDict` with fields reaches here.
        return _ishintstance_typeddict(obj, origin_uw)
    if isinstance(origin_uw, type):
        # A container's items are never looked at: that means iterating
        # them, which is the caller's business, not an instance check's. So
        # the origin is checked, and the arguments only when the value
        # *declares* them (#50, V5) -- `Box[int]()` records `Box[int]`, and an
        # instance of `class Child(List[int])` is one of `List[int]` by its
        # class. A plain `[1]` declares nothing: `type([1])` is `list`, never
        # `List[int]`, so any list matches every `List[...]`.
        members = _data_protocol_members(origin_uw)
        if members is not None:
            # A runtime-checkable protocol with data members is decided by
            # what the value holds, which `issubclass` refuses to answer
            # (#56).
            if not _ishintstance_protocol(obj, origin_uw, members):
                return False
        elif not safe_issubclass(type(obj), origin_uw):
            return False
        if not tx.get_args(hint):
            return True
        declared = _declared_parametrisation(obj, origin_uw)
        return declared is None or issubhint(declared, hint)
    return issubhint(type(obj), hint)


def _literal_value_eq(a: tx.Any, b: tx.Any) -> bool:
    """Report the type-aware equality of two `Literal` values, per PEP 586.

    Python itself treats `#!python 1 == True` and `#!python 1 == 1.0` as
    true, yet PEP 586 considers those different literal values, so their
    types are compared alongside their values here. [`eq_safenan`][]
    ensures a NaN literal still compares equal to itself.
    """
    return type(a) is type(b) and eq_safenan(a) == eq_safenan(b)


def _ishintstance_literal(obj: tx.Any, hint: tx.Any) -> bool:
    """Report whether a value matches one of a `Literal`'s listed values."""
    # Both the type and the value must match. Python compares `True == 1`
    # and `1 == 1.0` as equal, but PEP 586 makes literal matching
    # type-aware, so `Literal[1]` must reject `True` and `1.0`.
    return any(_literal_value_eq(arg, obj) for arg in get_args_uw(hint))


def _ishintstance_type(obj: tx.Any, hint: tx.Any) -> bool:
    """Report whether `obj` is a class matching a `type[...]` hint."""
    # Unwrap the hint, do *not* take its origin: the origin of `type[T]`
    # is the bare `type`, whose `get_args` is always empty, which would
    # make every `type[T]` behave like an unparametrised `type`.
    hint_uw = unwrap(hint)
    if safe_get_origin(hint_uw) is not type:
        # Invalid superhint -> error
        raise TypeError(f"Hint {hint} is not a type[]")
    args_uw = tx.get_args(hint_uw)
    if not args_uw:
        # hint is `type` (or `tx.Type`), so any type is valid
        return isinstance(obj, type)
    # hint is `type[T]` (or `tx.Type[T]`), so check obj is a subclass of T
    return isinstance(obj, type) and _issubclass_origin(obj, args_uw[0])


# --- runtime-checkable protocols with data members (#56) ---------------
#
# Python refuses `issubclass()` against a protocol that declares a data member
# (`name: str`), because whether a value has one is a property of the
# instance, not of its class. So such a protocol is read here member by
# member, in two halves:
#
# * the value level (`_ishintstance_protocol`) reads each **instance**
#   variable off the value itself, and each **method** and each
#   **`ClassVar`** member off the value's class -- the way a method-only
#   protocol is already decided, by type. Only the instance variables depend
#   on the instance, so the call cache keys such a position on the value's
#   type and which of those it has (`_present_data_members`, the one reader
#   the check and the key share);
# * the hint level (`_declares_protocol`) asks whether a class declares
#   every member, much as a type checker asks it of a structural subtype
#   (a read-only member counts here, where a type checker refuses it).
#
# Both halves read what a class declares as a type checker does
# (`_class_reading`), each kind exclusively: `name: str` anywhere in a
# class's MRO declares an instance variable, set in `__init__` or not, and
# so does a value in a class body; only `name: ClassVar[str]` declares a
# class variable. That puts a class that annotates `name` below a
# protocol asking for `name`, so -- for a value of a sub-hint to stay a value
# of the super-hint -- it also makes `name` present on every instance of the
# class, where Python's own `isinstance` looks for the attribute and finds
# none on an instance that never set it.


class _ProtocolMembers(tx.NamedTuple):
    """A runtime-checkable protocol's members, grouped by where each is read.

    `data` lists its instance variables, which are read off a value
    itself: its instance `__dict__`, its class, or its class's
    annotations. `methods` and `class_variables` (the data members the
    protocol declares as `ClassVar`) are instead read off the value's
    class alone. All three tuples are sorted, so a caller keying on them
    always sees the same order.
    """

    data: tx.Tuple[str, ...]
    methods: tx.Tuple[str, ...]
    class_variables: tx.Tuple[str, ...]


# A marker for "no such attribute", distinct from any value a class can hold
# (`None` included, which means something to a protocol method).
_ABSENT = object()


def _data_protocol_members(cls: tx.Any) -> tx.Optional[_ProtocolMembers]:
    """Find the members of `cls`, if it is a runtime protocol with data
    members.

    `#!python None` comes back for anything else: a class that is not a
    protocol at all, including a concrete class that merely inherits
    one; a protocol that is not
    [`runtime_checkable`][typing.runtime_checkable], which Python
    refuses to instance-check and so does this relation; and a protocol
    whose members are all methods, which Python decides purely from the
    class, and which the relation keeps deciding the same way.
    """
    if not isinstance(cls, type):
        return None
    if not getattr(cls, "_is_runtime_protocol", False):
        # The flag `runtime_checkable` sets, and the one Python's own
        # `isinstance` reads: a sub-protocol inherits it.
        return None
    try:
        hash(cls)
    except TypeError:
        # A class whose metaclass defines `__eq__` alone is unhashable and
        # cannot key the memo: it is read afresh on each call. The call
        # cache cannot key such a class either, so a dispatch on one is
        # resolved again on every call anyway.
        return _read_protocol_members.__wrapped__(cls)
    return _read_protocol_members(cls)


@functools.lru_cache(maxsize=None)
def _read_protocol_members(cls: type) -> tx.Optional[_ProtocolMembers]:
    """Compute [`_data_protocol_members`][] for a runtime-checkable class.

    A member counts as a method when the protocol holds a callable under
    its name, and as a data member otherwise, whether declared through a
    bare annotation, a property, or a plain default value, following the
    same split Python itself makes. A data member counts as a class
    variable when the protocol's nearest annotation of it is `ClassVar`.
    """
    if not tx.is_protocol(cls):
        return None
    names = sorted(tx.get_protocol_members(cls))
    data = [name for name in names if not callable(getattr(cls, name, None))]
    if not data:
        return None
    kinds = {}  # type: tx.Dict[str, int]
    for base in reversed(cls.__mro__):
        # Nearest last, so a sub-protocol's annotation wins over its base's.
        kinds.update(_own_annotation_kinds(base))
    return _ProtocolMembers(
        data=tuple(
            name for name in data if kinds.get(name) != _CLASS_VARIABLE
        ),
        methods=tuple(name for name in names if name not in data),
        class_variables=tuple(
            name for name in data if kinds.get(name) == _CLASS_VARIABLE
        ),
    )


def _class_attribute(cls: type, name: str) -> tx.Any:
    """Find what `cls` defines under `name` anywhere in its MRO, or `_ABSENT`.

    Each class's own namespace is read directly, so a property is found
    as an object rather than called, and none of the class's code ever
    runs.
    """
    for base in cls.__mro__:
        namespace = base.__dict__
        if name in namespace:
            return namespace[name]
    return _ABSENT


def _has_method(cls: type, name: str) -> bool:
    """Report whether `cls` defines the protocol method `name`.

    A class that sets the name to `#!python None` is treated as
    declaring that it does not support that method, matching how Python
    reads a protocol method itself.
    """
    found = _class_attribute(cls, name)
    return found is not _ABSENT and found is not None


def _present_data_members(
    obj: tx.Any, names: tx.Sequence[str]
) -> tx.Tuple[bool, ...]:
    """Report, for each name in `names`, whether `obj` has that instance
    variable.

    A name counts as present when it appears in the instance's own
    `__dict__`, or when the value's class declares it an instance
    variable anywhere in its MRO ([`_class_reading`][]). Such a
    declaration comes through an annotation (`#!python name: str`), or
    through a class attribute, a property, a slot, or a method that the
    same class does not annotate as `ClassVar`. None of the value's code
    runs to answer this: a
    property is found rather than called, and `__getattr__` is never
    invoked, matching how Python's own `isinstance` reads a protocol
    member from 3.12 on, even though earlier versions of `typing` call
    `hasattr` instead. Two differences from plain `isinstance` are
    deliberate here, and both are shared with the type checker's own
    reading and with the hint-level check in
    [`_declares_protocol`][], so that the two stay in agreement. An
    annotated name counts as present even on an instance that never set
    it, and a `ClassVar` never counts as an instance variable, so a class
    attribute declared that way is never treated as present.

    Both the value check and the call cache read this same function, so
    a cache key always reflects exactly what the check itself reads. It
    runs on every call the cache answers for such an argument, so
    whatever only depends on the class is computed once per class
    instead ([`_class_reading`][]). That includes how the value's
    attributes are looked up (an ordinary instance's own `__dict__`,
    [`inspect.getattr_static`][] for a class read as a value, or the
    class alone for anything else), and which names the class declares
    at all. A declared name is present on
    every instance of that class, so its entry in the key is identical
    for all of them, and reading it costs no more than a set lookup.
    """
    cls = type(obj)
    reading = _class_reading(cls)
    declared = reading.instance_variables
    if reading.lookup == _PLAIN:
        try:
            own = object.__getattribute__(obj, "__dict__")
        except AttributeError:
            # `__slots__` without a `__dict__`: only the class can hold it.
            own = _NO_ATTRIBUTES
        # A loop, not a comprehension: before 3.12 a comprehension builds a
        # function on each call, which costs more than the reading here.
        present = []
        for name in names:
            present.append(name in own or name in declared)
        return tuple(present)
    if reading.lookup == _CLASS_ONLY:
        return tuple([name in declared for name in names])
    return tuple(
        [name in declared or _found_statically(obj, name) for name in names]
    )


def _holds_class_variables(cls: type, names: tx.Sequence[str]) -> bool:
    """Report whether `cls` declares every `ClassVar` member listed in `names`.

    A class variable lives on the class itself, so it is read off the
    class alone, both at the value level and the hint level: a
    `ClassVar` annotation anywhere in the MRO counts, whether or not it
    carries a value. A plain class attribute, an instance variable's
    annotation, and an attribute an instance sets on itself all fail to
    count, since a type checker would reject each of them as a
    `ClassVar` member.
    """
    declared = _class_reading(cls).class_variables
    return all(name in declared for name in names)


# The namespace of an instance that has none of its own.
_NO_ATTRIBUTES = frozenset()  # type: tx.FrozenSet[str]


def _found_statically(obj: tx.Any, name: str) -> bool:
    """Report whether [`inspect.getattr_static`][] can find `name` on `obj`."""
    try:
        inspect.getattr_static(obj, name)
    except (AttributeError, TypeError):
        # Absent. A `TypeError` is defensive: from 3.13 `getattr_static`
        # memoises on the classes it walks, and raises for one it cannot
        # hash -- a case `_class_reading` already routes around for the
        # value's own class.
        return False
    return True


# How an instance of a class has its attributes looked up (`_class_reading`):
# its own `__dict__` and then what its class declares;
# `inspect.getattr_static`, for a class read as a value; or what its class
# declares alone.
_PLAIN = 0
_STATIC = 1
_CLASS_ONLY = 2


class _ClassReading(tx.NamedTuple):
    """What reading an instance of a class requires, computed once per class.

    `lookup` says how the instance's attributes must be looked up: one
    of `_PLAIN`, `_STATIC`, or `_CLASS_ONLY`. `instance_variables` and
    `class_variables` list the names the class declares as each kind,
    anywhere in its MRO.
    """

    lookup: int
    instance_variables: tx.FrozenSet[str]
    class_variables: tx.FrozenSet[str]


# `_class_reading`'s memo. Its keys are the types of the values dispatched
# on, so they are held weakly: a class made and dropped at runtime is not
# kept alive for the life of the process.
_CLASS_READINGS = weakref.WeakKeyDictionary()  # type: weakref.WeakKeyDictionary


def _class_reading(cls: type) -> _ClassReading:
    """Compute, and cache, how [`_present_data_members`][] should read `cls`.

    The lookup mode is `_PLAIN` for an ordinary class
    ([`_reads_instance_dict`][]); `_STATIC`, meaning
    `inspect.getattr_static`, for a metaclass, since its instances are
    themselves classes holding attributes across their own bases and
    their metaclass; and `_CLASS_ONLY` for a class that redefines
    `__dict__`, whose instances' own attributes cannot be read without
    running its code.

    The declared names are collected from every class in the MRO
    ([`_read_class`][]), and a name counts as declared when any one of
    them declares it. A subclass therefore always declares everything
    its bases do, and an instance of a class beneath a protocol remains
    an instance of that protocol too. This result is cached per class, so
    an attribute or annotation added to a class after it was first
    dispatched on will not be picked up.

    A class that cannot key this cache, because its metaclass makes it
    unhashable, is recomputed fresh on every call instead. Such a class
    also cannot key `getattr_static`'s own internal cache, which raises
    `TypeError` for an instance of it from Python 3.13 on, matching how
    Python's own `isinstance` behaves against a protocol on every
    version. So an instance of that kind of metaclass falls back to
    `_CLASS_ONLY`, reading through what its class declares alone.
    """
    try:
        return _CLASS_READINGS[cls]
    except KeyError:
        reading = _read_class(cls, _STATIC)
        _CLASS_READINGS[cls] = reading
        return reading
    except TypeError:
        return _read_class(cls, _CLASS_ONLY)


def _read_class(cls: type, metaclass_lookup: int) -> _ClassReading:
    """Compute [`_class_reading`][] directly, without consulting the cache.

    `metaclass_lookup` is the lookup mode to use when `cls` turns out to
    be a metaclass. Each class in the MRO is read the way a type checker
    would read it. An instance variable is declared by any annotation
    that is not a `ClassVar`, an `InitVar`, or a `KW_ONLY` (which declare
    nothing at all), and also by any name the class's namespace holds,
    such as a class attribute, a property, a slot, or a method, that it
    does not annotate. A value written in the class body then becomes
    that instance variable's default, and a class variable is declared by
    each `ClassVar` annotation, whether or not it carries a value.

    Both kinds are collected across the whole MRO at once, never decided
    class by class in isolation. A subclass that redeclares an inherited
    instance variable as a `ClassVar` still keeps it as one, since a type
    checker would reject that kind of override, so a class always sits
    below whatever its bases sit below.
    """
    instance_variables = set()  # type: tx.Set[str]
    class_variables = set()  # type: tx.Set[str]
    for base in cls.__mro__:
        kinds = _own_annotation_kinds(base)
        for name, kind in kinds.items():
            if kind == _INSTANCE_VARIABLE:
                instance_variables.add(name)
            elif kind == _CLASS_VARIABLE:
                class_variables.add(name)
        for name in base.__dict__:
            if name not in kinds:
                instance_variables.add(name)
    if _reads_instance_dict(cls):
        lookup = _PLAIN
    elif issubclass(cls, type):
        lookup = metaclass_lookup
    else:
        lookup = _CLASS_ONLY
    return _ClassReading(
        lookup, frozenset(instance_variables), frozenset(class_variables)
    )


# --- what a class's annotations declare --------------------------------
#
# A type checker reads `name: str` in a class body as declaring an instance
# variable, whether or not a value follows or `__init__` sets it;
# `name: ClassVar[str]` as declaring a class variable; and a dataclass's
# `InitVar[...]` / `KW_ONLY` as declaring no attribute at all. Each satisfies
# a protocol member of its own kind only.
#
# An annotation that is still text (`from __future__ import annotations`) is
# parsed, not executed: the dotted name it starts with is looked up, by
# dictionary lookups alone, where the evaluated annotation would find it,
# and the marker it names is recognised by identity -- so an alias
# (`CV = ClassVar`) is read as what it stands for, and a class of the user's
# that happens to be called `InitVar` as the class it is. Nothing is called,
# and nothing but the leading name is looked up.

_INSTANCE_VARIABLE = 0
_CLASS_VARIABLE = 1
_NOT_AN_ATTRIBUTE = 2

_CLASSVAR_FORMS = spellings("ClassVar")
_ANNOTATED_FORMS = spellings("Annotated")

# `KW_ONLY` is new in 3.10; before that, nothing can be annotated with it.
_KW_ONLY = getattr(dataclasses, "KW_ONLY", _ABSENT)

# The kinds a marker declares, by the name it is written with: for an
# annotation still text whose leading name cannot be looked up.
_MARKER_KINDS = {
    "ClassVar": _CLASS_VARIABLE,
    "InitVar": _NOT_AN_ATTRIBUTE,
    "KW_ONLY": _NOT_AN_ATTRIBUTE,
}

# The dotted name an annotation written as text starts with, and whether a
# `[` follows it.
_LEADING_NAME = re.compile(r"\s*((?:\w+\s*\.\s*)*\w+)\s*(\[?)")


if tx.TYPE_CHECKING:
    from annotationlib import Format, get_annotations
else:
    try:
        from annotationlib import Format, get_annotations
    except ImportError:  # pragma: no cover  -- Python < 3.14
        Format = get_annotations = None


def _own_annotations(cls: type) -> tx.Mapping[str, tx.Any]:
    """Find the annotations written in the body of `cls` itself, by name.

    Base classes' annotations are excluded; a caller wanting those walks
    the MRO on its own. Up through Python 3.13 these are read straight
    from the class's own namespace; from 3.14 on, where annotations are
    evaluated lazily, they are read through `annotationlib` instead,
    where an undefined name comes back as a forward reference rather
    than raising, and an annotation that cannot be evaluated at all is
    simply treated as absent.
    """
    if get_annotations is None:  # pragma: no cover  -- Python < 3.14
        annotations = cls.__dict__.get("__annotations__")
        # `type` holds the descriptor that serves every class's
        # `__annotations__`, not annotations of its own.
        return annotations if isinstance(annotations, dict) else {}
    try:
        return get_annotations(cls, format=Format.FORWARDREF)
    except Exception:  # noqa: BLE001 -- the class's own code raised
        return {}


def _own_annotation_kinds(cls: type) -> tx.Dict[str, int]:
    """Find what each annotation in the body of `cls` declares, by name.

    A `TypedDict`'s annotations declare its keys, not attributes, since
    any instance of it is a plain `dict`, so none of them declares
    anything by this function's reckoning.
    """
    if is_typeddict(cls):
        return {}
    return {
        name: _annotation_kind(annotation, cls)
        for name, annotation in _own_annotations(cls).items()
    }


def _annotation_kind(annotation: tx.Any, cls: tx.Any = None) -> int:
    """Find what one annotation, written in the body of `cls`, declares.

    The result names an instance variable, a class variable, or no
    attribute at all. `Annotated[...]` is looked through first. An
    annotation still held as text, whether a string or a forward
    reference, is read by the leading name it starts with
    ([`_text_annotation_kind`][]).
    """
    if isinstance(annotation, tx.ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        return _text_annotation_kind(annotation, cls)
    annotation = unwrap(annotation)
    return _marker_kind(safe_get_origin(annotation) or annotation)


def _marker_kind(marker: tx.Any) -> int:
    """Find what an annotation whose outermost form is `marker` declares."""
    if any(marker is form for form in _CLASSVAR_FORMS):
        return _CLASS_VARIABLE
    if (
        marker is dataclasses.InitVar
        or isinstance(marker, dataclasses.InitVar)
        or marker is _KW_ONLY
    ):
        return _NOT_AN_ATTRIBUTE
    return _INSTANCE_VARIABLE


def _text_annotation_kind(text: str, cls: tx.Any = None) -> int:
    """Compute [`_annotation_kind`][] for an annotation held as text in `cls`.

    The dotted name the text starts with is looked up wherever
    evaluating the annotation would find it ([`_look_up_marker`][]) and
    then recognised by identity; `Annotated[...]` is looked through
    along the way. A name that cannot be looked up falls back to being
    read by its last dotted part instead, so `typing.ClassVar[int]` and
    `ClassVar[int]` are read the same way. The text itself is never
    evaluated.
    """
    match = _LEADING_NAME.match(text)
    if match is None:
        return _INSTANCE_VARIABLE
    parts = [part.strip() for part in match.group(1).split(".")]
    marker = _look_up_marker(parts, cls)
    if marker is _ABSENT:
        annotated = parts[-1] == "Annotated"
        kind = _MARKER_KINDS.get(parts[-1], _INSTANCE_VARIABLE)
    else:
        annotated = any(marker is form for form in _ANNOTATED_FORMS)
        kind = _marker_kind(marker)
    if annotated and match.group(2):
        return _text_annotation_kind(text[match.end() :], cls)
    return kind


def _look_up_marker(parts: tx.Sequence[str], cls: tx.Any) -> tx.Any:
    """Find what the dotted name `parts`, written in the body of `cls`, names.

    The first part is looked up in the class's own namespace, then in
    its module's, and finally in the builtins, following the same order
    evaluating the annotation would; each further part is then looked up
    in whichever module the name resolved so far names. Only plain
    dictionaries are ever read here, never an attribute fetch, so none
    of the user's code runs. `_ABSENT` comes back when a part cannot be
    found, or when a name before the last one turns out not to be a
    module.
    """
    namespaces = []  # type: tx.List[tx.Mapping[str, tx.Any]]
    if isinstance(cls, type):
        namespaces.append(cls.__dict__)
        module = sys.modules.get(cls.__dict__.get("__module__"))
        if module is not None:
            namespaces.append(module.__dict__)
    namespaces.append(builtins.__dict__)
    found = _ABSENT
    for namespace in namespaces:
        found = namespace.get(parts[0], _ABSENT)
        if found is not _ABSENT:
            break
    for part in parts[1:]:
        if not isinstance(found, types.ModuleType):
            return _ABSENT
        found = found.__dict__.get(part, _ABSENT)
    return found


def _reads_instance_dict(cls: type) -> bool:
    """Report whether an instance of `cls` has its attributes looked up
    plainly.

    This comes back `#!python True` for an ordinary class, whose
    instances hold their attributes in their own `__dict__`, read
    through the standard descriptor that runs no code, together with
    what the class's MRO declares. It comes back `#!python False` for a
    metaclass, since its instances are themselves classes whose
    attributes come from their own bases and their own metaclass, and
    for a class that redefines `__dict__`, such as with a property,
    where a plain read would actually run code.
    """
    if issubclass(cls, type):
        return False
    for base in cls.__mro__:
        entry = base.__dict__.get("__dict__", _ABSENT)
        if entry is _ABSENT:
            continue
        if not (
            type(entry) is types.GetSetDescriptorType
            and entry.__objclass__ is base
        ):
            return False
    return True


def _ishintstance_protocol(
    obj: tx.Any, proto: type, members: _ProtocolMembers
) -> bool:
    """Report whether `obj` is an instance of the data-member protocol `proto`.

    A class that names `proto` among its own bases already counts,
    exactly as Python's `isinstance` would count it, regardless of what
    its instances actually hold. Otherwise, every method has to be
    defined on the value's class, every `ClassVar` member has to be
    declared there ([`_holds_class_variables`][]), and every instance
    variable has to be present on the value itself
    ([`_present_data_members`][]).

    Methods are read off the class the same way a method-only protocol
    is decided, so only the instance variables actually depend on the
    particular instance, and those are the only part the call cache
    needs to key on. A method set only on the instance is therefore not
    counted here, unlike how Python's own `isinstance` would count it,
    and neither is a `ClassVar` member, which a type checker would
    likewise reject.
    """
    cls = type(obj)
    if any(base is proto for base in cls.__mro__):
        return True
    return (
        all(_has_method(cls, name) for name in members.methods)
        and _holds_class_variables(cls, members.class_variables)
        and all(_present_data_members(obj, members.data))
    )


def _declares_protocol(
    cls: tx.Any, proto: type, members: _ProtocolMembers
) -> bool:
    """Report whether `cls` is a structural subtype of the data protocol
    `proto`.

    This asks at the hint level the question
    [`_ishintstance_protocol`][] asks at the value level, and does so
    much as a type checker would. A class naming `proto` among its own
    bases is below it, including a sub-protocol of it. Another protocol
    that does not name it is never below it, even one listing the same
    members, since a protocol stands only for what it explicitly says it
    extends. Any other class has to define every method, and not as
    `#!python None`, and has to declare every data member of the
    matching kind somewhere in its MRO ([`_read_class`][]). An instance
    variable is declared through an annotation such as
    `#!python name: str`, including a dataclass field whether or not
    `__init__` actually sets it, or through a class attribute, a
    property, or a slot not annotated `ClassVar`; a class variable is
    declared through a `ClassVar` annotation alone.

    An annotation that is never assigned a value still promises nothing
    at runtime, so the value level treats it as present too
    ([`_present_data_members`][]), which keeps every instance of `cls` an
    instance of `proto`, the one guarantee this ordering has to uphold.
    Both levels read from the same per-class record, gathered across the
    whole MRO, so a subclass always declares whatever its bases declare.

    Unlike a type checker, a read-only member, such as a property with
    no setter or `#!python name: Final = ...`, still counts here as a
    protocol's settable instance variable, since the value genuinely has
    it and dispatch only ever reads it, never writes to it.
    """
    if not isinstance(cls, type):
        return False
    if any(base is proto for base in cls.__mro__):
        return True
    if tx.is_protocol(cls):
        return False
    if not all(_has_method(cls, name) for name in members.methods):
        return False
    if not _holds_class_variables(cls, members.class_variables):
        return False
    declared = _class_reading(cls).instance_variables
    return all(name in declared for name in members.data)


def _protocol_below_protocol(
    sub: type, members: _ProtocolMembers, sup: type
) -> bool:
    """Report whether the data protocol `sub` is below the method-only
    protocol `sup`.

    `sup` is runtime-checkable here, and `members` belongs to `sub`. A
    protocol naming `sup` among its own bases is below it outright. So
    is one whose methods, read off the class the same way the value
    level reads them, already cover every member `sup` requires, since
    that makes the class of any instance of `sub` pass Python's own
    check for `sup` as well. A data member of `sub` never stands in for
    a method of `sup`, even though Python's own `issubclass` would let
    an annotation do exactly that, because such a member might be set
    only on an instance, in which case the instance's class alone does
    not actually satisfy `sup`.
    """
    if any(base is sup for base in sub.__mro__):
        return True
    return set(tx.get_protocol_members(sup)) <= set(members.methods)


# A unique marker for "no such attribute". Distinct from any value a
# `TypedDict` could carry, so an identity check never confuses it with one.
_NO_EXTRA_ITEMS = object()


def _typeddict_extra_policy(td: tx.Any) -> tx.Tuple[str, tx.Any]:
    """Find how a `TypedDict` treats keys beyond the ones it declares.

    `("open", None)` comes back when extra keys are allowed, the
    default; `("closed", None)` when extra keys are rejected outright;
    and `("typed", hint)` when extra keys are allowed but each one's
    value must satisfy `hint`.

    A `TypedDict` written `closed=True` reports `"closed"`, and one
    written `extra_items=SomeType` reports `("typed", SomeType)`.
    Closedness is inherited, so a subclass of a closed or
    `extra_items=` `TypedDict` is itself closed or typed even without
    repeating the keyword; the base chain is walked nearest first, and
    the first class along it that states a policy wins. A class built by
    an older `typing_extensions`, unable to express either concept, has
    no readable closedness and is reported `"open"`, the permissive
    default, so the check never fails purely because of it.
    """
    # `extra_items=SomeType` records the type on the declaring class's own
    # `__extra_items__`; when none was given `typing_extensions` leaves a
    # sentinel there (or, predating the feature, no attribute). `__closed__`
    # is likewise set from a class's *own* keyword only, and a `TypedDict`'s
    # base is not in its MRO -- so read the policy off each class in the base
    # chain rather than off `td` alone, nearest-first.
    no_extra = getattr(tx, "NoExtraItems", _NO_EXTRA_ITEMS)
    for base in _all_orig_bases(td):
        cls = safe_get_origin(base) or base
        extra = getattr(cls, "__extra_items__", _NO_EXTRA_ITEMS)
        if extra is not _NO_EXTRA_ITEMS and extra is not no_extra:
            # `extra_items=Never` arrives here too, and needs no special case:
            # no value satisfies `Never`, so any extra key is rejected --
            # exactly what `extra_items=Never` means (like `closed=True`).
            return "typed", extra
        if getattr(cls, "__closed__", None) is True:
            return "closed", None
    return "open", None


# --- TypedDict well-formedness (PEP 728) -------------------------------


def _skippable_extra_hint(hint: tx.Any) -> bool:
    """Report whether an `extra_items` or field hint cannot be judged here.

    An unresolvable forward reference, whether a bare string or a
    [`ForwardRef`][typing.ForwardRef], carries no actual type to compare
    against, and a missing hint shows up as `#!python None`. Either way,
    the well-formedness of a key or policy involving it is left
    unjudged rather than guessed at, exactly as the value-level shape
    check itself skips such a field.
    """
    return hint is None or isinstance(hint, (str, tx.ForwardRef))


def _is_any_hint(hint: tx.Any) -> bool:
    """Report whether a field or `extra_items` hint is `Any`, seen through its
    qualifiers.

    The transparent qualifiers `Required`, `NotRequired`, `ReadOnly`,
    `Final`, and `ClassVar` are stripped away first, via
    `normalise_hint`, along with any [`Annotated`][typing.Annotated]
    wrapper, before `_is_any` is asked. An `Any`-typed key or
    `extra_items` accepts every value, so it can never break a base's
    contract and is always treated as compatible.
    """
    return _is_any(unwrap(normalise_hint(hint), tx.Annotated))


# PEP 484 §Numeric numeric tower: `bool` promotes to `int`, `int` to `float`,
# `float` to `complex`. A type checker treats a lower type as assignable to a
# wider one, and `_numeric_consistent` reads a hint to its rank here.
_NUMERIC_RANK = {bool: 0, int: 1, float: 2, complex: 3}


def _numeric_consistent(sub: tx.Any, sup: tx.Any) -> bool:
    """Report whether `sub` promotes to `sup` under the PEP 484 numeric tower.

    `bool`, `int`, `float`, and `complex` form a promotion chain, and a
    type checker treats each as assignable wherever a wider one is
    expected, such as passing `int` where `float` is wanted. Both hints
    are reduced to their bare builtin type first, through
    `normalise_hint` and any `Annotated` wrapper; the result comes back
    `#!python True` only when both are numeric and `sub` sits at or
    below `sup` in that chain. `bool` promoting to `int` is already a
    genuine subclass relationship and so would hold on its own
    regardless; including it here just keeps this check self-contained.
    """
    sub = unwrap(normalise_hint(sub), tx.Annotated)
    sup = unwrap(normalise_hint(sup), tx.Annotated)
    return (
        sub in _NUMERIC_RANK
        and sup in _NUMERIC_RANK
        and _NUMERIC_RANK[sub] <= _NUMERIC_RANK[sup]
    )


def _own_extra_policy(cls: tx.Any) -> tx.Tuple[str, tx.Any]:
    """Find the policy a class declares itself, ignoring whatever it inherits.

    A `TypedDict`'s base is absent from its MRO, and `__closed__` and
    `__extra_items__` are each set only from the class's own keyword, so
    a plain attribute read already gives just that class's own
    declaration. `("typed", hint)` comes back when the class wrote
    `extra_items=hint`, `("closed", None)` when it wrote `closed=True`,
    `("reopened", None)` when it wrote `closed=False`, and
    `("none", None)` when the class declared no policy of its own at
    all.
    """
    no_extra = getattr(tx, "NoExtraItems", _NO_EXTRA_ITEMS)
    extra = getattr(cls, "__extra_items__", _NO_EXTRA_ITEMS)
    if extra is not _NO_EXTRA_ITEMS and extra is not no_extra:
        return "typed", extra
    closed = getattr(cls, "__closed__", None)
    if closed is True:
        return "closed", None
    if closed is False:
        return "reopened", None
    return "none", None


def _typeddict_bases(cls: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """List the direct `TypedDict` bases of a class, by origin, skipping
    the marker.

    A parametrised base, such as `Base[int]`, is read through its
    origin `Base`, the same way [`_typeddict_extra_policy`][] reaches a
    generic base.
    """
    bases = ()  # type: tx.Tuple[tx.Any, ...]
    for base in getattr(cls, "__orig_bases__", ()):
        if is_typeddict_marker(base):
            continue
        origin = safe_get_origin(base) or base
        if is_typeddict(origin) and not is_typeddict_marker(origin):
            bases += (origin,)
    return bases


def _inherited_extra_policy(
    cls: tx.Any,
) -> tx.Tuple[str, tx.Any, tx.Any]:
    """Find the `("open"/"closed"/"typed", hint, base)` policy `cls` inherits.

    This is the effective policy of `cls`'s nearest constrained base,
    returned together with that base, so that a violation can name where
    the contract actually came from. `cls`'s own declaration plays no
    part here.
    """
    for base in _typeddict_bases(cls):
        policy, hint = _typeddict_extra_policy(base)
        if policy != "open":
            return policy, hint, base
    return "open", None, None


def _malformed_class_reason(cls: tx.Any) -> tx.Optional[str]:
    """Find why `cls` violates a constrained base's PEP 728 contract, or
    `None`.

    `cls`'s own declaration is compared against the policy it inherits
    from its nearest closed or `extra_items` base. The first key or
    policy found that would let a value of `cls` carry something that
    base refuses is reported, since the nominal hint order would still
    call `cls` a sub-hint of that base and dispatch would then go wrong.
    A closed base is treated as `extra_items=Never`, since it admits no
    extra key at all, which turns adding a key to it and adding an
    `extra_items`-incompatible key into the very same check.

    Every key `cls` carries, whether from its own body or from any base,
    not only the ones it declares directly, is checked against the
    constrained base. A key the base does not declare, whose value type
    it would not admit, is a dispatch-unsound diamond, the same problem a
    sibling open base contributing a key a closed base rejects would
    cause. An `Any`-typed key or `extra_items` accepts every value, so it
    can never break the contract and is always treated as compatible.
    """
    inherited, inherited_hint, base = _inherited_extra_policy(cls)
    if inherited == "open":
        # An open base constrains nothing, so nothing about `cls` can break it.
        return None
    cname = getattr(cls, "__name__", str(cls))
    bname = getattr(base, "__name__", str(base))
    # A closed base admits no extra items: read it as `extra_items=Never`.
    inherited_extra = tx.Never if inherited == "closed" else inherited_hint
    own, own_hint = _own_extra_policy(cls)
    if own == "reopened":
        # `closed=False` widens a closed base back to open, or a typed base to
        # open -- either way broader than the inherited contract.
        return (
            f"{cname} is a malformed TypedDict: reopens "
            f"{'closed ' if inherited == 'closed' else ''}base {bname} "
            "with closed=False"
        )
    if (
        own == "typed"
        and not _skippable_extra_hint(own_hint)
        and not _is_any_hint(own_hint)
    ):
        # Setting `extra_items` is allowed only when it narrows the inherited
        # one; against a closed base (Never) nothing but Never narrows.
        if not issubhint(own_hint, inherited_extra):
            if inherited == "closed":
                return (
                    f"{cname} is a malformed TypedDict: sets extra_items on "
                    f"closed base {bname}"
                )
            return (
                f"{cname} is a malformed TypedDict: widens the extra_items of "
                f"base {bname}"
            )
    field_hints = typeddict_field_hints(cls)
    base_keys = set(typeddict_field_hints(base))
    for key in sorted(set(field_hints) - base_keys):
        key_hint = field_hints.get(key)
        if _skippable_extra_hint(key_hint) or _is_any_hint(key_hint):
            continue
        # The added key's value type must be assignable to the base's
        # `extra_items` type: the nominal relation, plus numeric-tower
        # promotion (`int` under `extra_items=float`), which a type checker
        # accepts under PEP 484 §Numeric.
        compatible = issubhint(
            key_hint, inherited_extra
        ) or _numeric_consistent(key_hint, inherited_extra)
        if not compatible:
            if inherited == "closed":
                return (
                    f"{cname} is a malformed TypedDict: adds key {key!r} to "
                    f"closed base {bname}"
                )
            return (
                f"{cname} is a malformed TypedDict: adds key {key!r} whose "
                f"type is not compatible with the extra_items of base {bname}"
            )
    return None


def _typeddict_chain(td: tx.Any) -> tx.List[tx.Any]:
    """List every `TypedDict` class in `td`'s inheritance, `td` first, by
    origin.

    Origins are resolved along the way, so `Base[int]` becomes `Base`,
    and each class is visited only once, so a diamond in the inheritance
    is never walked twice.
    """
    chain = []  # type: tx.List[tx.Any]
    stack = [td]
    while stack:
        current = stack.pop(0)
        origin = safe_get_origin(current) or current
        if is_typeddict_marker(origin) or not is_typeddict(origin):
            continue
        if any(origin is seen for seen in chain):
            continue
        chain.append(origin)
        stack.extend(getattr(origin, "__orig_bases__", ()))
    return chain


def _malformed_typeddict_reason(td: tx.Any) -> tx.Optional[str]:
    """Find why a `TypedDict` `td` would break dispatch soundness, or `None`.

    This flags only a dispatch-unsound subset of `TypedDict`s, not
    everything a type checker would forbid. Specifically, it flags one
    whose nominal hint order (since `issubhint` and `ishintstance` both
    treat a concrete `TypedDict` nominally) would disagree with its own
    value-level shape check, breaking the rule that `v in Sub` together
    with `Sub <= Base` must imply `v in Base`. Such a class is refused at
    registration rather than left free to mis-dispatch later. The shapes
    caught this way are a key added to a closed base, a key whose value
    type a base's `extra_items` does not admit (including one
    contributed by a sibling open base in a diamond), a widened
    `extra_items`, and a closed base reopened through `extra_items=` or
    `closed=False`.

    Checking added-key and `extra_items` compatibility treats two cases
    as consistent to match how a type checker reads them, rather than
    relying purely on this library's own nominal relation. A top-level
    `Any` is accepted, since an `Any`-typed key or `extra_items` admits
    every value and so can never carry something a base refuses. The
    numeric tower is accepted too, so a key typed `int` under
    `extra_items=float`, or `int` or `float` under `complex`, is
    considered consistent, because a type checker promotes it under PEP
    484 §Numeric and rejecting it here would be surprising.

    Deeper gradual-consistency cases that a type checker also accepts
    stay strict here, as a documented limitation: nested `Any`, as in
    `List[Any]` against `List[int]`, a `Callable[..., R]` argument list,
    and non-runtime protocols are all judged strictly by the nominal
    relation instead.

    A well-formed `TypedDict` returns `#!python None`, whether it is a
    plain subclass of a closed base, one that narrows `extra_items`, or a
    subclass of an open base. So does anything that is not a concrete
    `TypedDict` at all, and any class built by a `typing_extensions` too
    old to record closedness, since its policy simply cannot be read and
    so nothing about it can be judged malformed.
    """
    if not is_typeddict(td) or is_typeddict_marker(td):
        return None
    for cls in _typeddict_chain(td):
        reason = _malformed_class_reason(cls)
        if reason is not None:
            return reason
    return None


def _ishintstance_typeddict(obj: tx.Any, td: tx.Any) -> bool:
    """Report whether a value has the shape a `TypedDict` describes.

    A value matches when it is a [`dict`][], holds every required key,
    and, for each declared key it does hold, carries a value satisfying
    that field's hint, checked through `ishintstance` so that a nested
    `TypedDict` or container field is read the same way any other value
    would be. `Required`, `NotRequired`, and the class's `total=` setting
    are all honoured through `typeddict_required_keys`.

    Only a genuine [`dict`][] is accepted, never some other
    [`Mapping`][collections.abc.Mapping]. This keeps the value level
    consistent with the hint level, where `TypedDict <= dict` holds.
    Every `TypedDict`-shaped value must also be a valid `dict`, so a
    non-`dict` mapping that matched the shape but was not itself a
    `dict` would break the rule that `v in S` together with `S <= T`
    implies `v in T`.

    Keys beyond the declared ones are treated exactly as the
    `TypedDict` itself specifies, following PEP 728. An open
    `TypedDict`, the default, allows extra keys, so a `dict` carrying
    keys beyond the declared ones still matches as long as the declared
    keys check out; such a `TypedDict` names a minimum shape rather than
    a closed one, the same way pydantic's `TypeAdapter` reads it. A
    closed `TypedDict`, written `closed=True`, rejects any extra key
    outright, so a value carrying one it does not declare never matches.
    A `TypedDict` written `extra_items=SomeType` allows extra keys but
    checks each one's value against `SomeType` through `ishintstance`,
    the same way a declared field is checked, so `extra_items=Never`
    ends up admitting no extra key at all, since no value satisfies
    `Never`.

    Two unrelated `TypedDict`s that happen to share a shape both can
    satisfy are not ordered against each other by this check, so a value
    matching both dispatches to neither on its own: selection raises
    `AmbiguousMethodError` instead, the same outcome two equally
    matched `Protocol`s produce, per RFC 0001 §5.
    """
    if not isinstance(obj, dict):
        return False
    for key in typeddict_required_keys(td):
        if key not in obj:
            return False
    field_hints = typeddict_field_hints(td)
    for key, field_hint in field_hints.items():
        if key not in obj:
            continue
        if isinstance(field_hint, (str, tx.ForwardRef)):
            # A forward reference that could not be resolved -- a bare string
            # on some versions, a `ForwardRef` on others. Its value type
            # cannot be read here, so the present value is accepted rather
            # than raised on; but the field is reported, the same way an
            # unrecognised hint is reported elsewhere, so a shape that silently
            # skips a check is not mistaken for one that passed it.
            _warn_unknown(field_hint)
            continue
        if not ishintstance(obj[key], field_hint):
            return False
    # Keys the `TypedDict` does not declare are constrained only when it is
    # closed or carries an `extra_items=` type. An open `TypedDict` (the
    # common case) skips this loop entirely.
    #
    # NOTE (hint-level relation): the value check above reads *inherited*
    # closedness (via `_typeddict_extra_policy`), but the hint-level
    # `issubhint` on a concrete `TypedDict` is purely nominal
    # (`safe_issubclass`) and does not. For every *well-formed* PEP 728
    # `TypedDict` this stays sound -- a subclass may not add keys to a closed
    # base, nor a key whose value type is incompatible with a base's
    # `extra_items`, so a subclass value never carries a key its closed/typed
    # base would refuse. A subclass that breaks those rules (which a type
    # checker rejects, but the runtime still lets you build) could carry such
    # a key. Rather than make the relation closedness-aware, such a *malformed*
    # `TypedDict` is refused at registration (`_malformed_typeddict_reason`),
    # so the relation stays nominal and total and only well-formed shapes reach
    # it (#42).
    policy, extra_hint = _typeddict_extra_policy(td)
    if policy != "open":
        for key in obj:
            if key in field_hints:
                continue
            if policy == "closed":
                return False
            if isinstance(extra_hint, (str, tx.ForwardRef)):
                _warn_unknown(extra_hint)
                continue
            if not ishintstance(obj[key], extra_hint):
                return False
    return True


def issubhint(hint: tx.Any, superhint: tx.Any) -> bool:
    """Report whether `hint` is a sub-hint of `superhint`.

    One hint is a sub-hint of another exactly when every value the first
    describes is also described by the second, which makes `issubhint`
    the type-level counterpart of asking whether every instance of one
    class is also an instance of another.

    Where a hint is generic, each of its argument positions is compared
    according to the variance that generic declares for it, following
    PEP 484. A covariant position, such as a `#!python Sequence`'s
    element type, carries the relation forward unchanged, so
    `#!python Sequence[bool]` is a sub-hint of
    `#!python Sequence[int]`. A contravariant position reverses the
    relation instead. An invariant position, such as a mutable container
    like `#!python list`, or an unflagged user `#!python TypeVar`,
    demands that the two arguments describe exactly the same values, so
    `#!python List[bool]` is not a sub-hint of `#!python List[int]`
    even though `bool` is a subtype of `int`. A free `#!python TypeVar`
    or `#!python Any` standing in the super-hint's position still acts
    as a top that an invariant position may widen to. A hint given with
    no arguments at all is never a sub-hint of one that has them, since a
    bare `#!python list` might hold anything and so cannot stand in for
    the narrower `#!python List[int]`.

    !!! note
        An unparametrised `#!python Union` or `#!python Literal` asks a
        different question when it appears as the super-hint: not
        "does every value here belong to some member", but "is this
        very hint one of those forms". `#!python issubhint(int, Union)`
        is therefore `#!python False`, since a plain `#!python int` is
        not itself a union, even though
        `#!python issubhint(int, Union[int, str])` is `#!python True`.
        This is what lets either form be used as a `#!python BOUND`, and
        it is also why the relation is not transitive through a bare
        `#!python Union`.

    !!! note
        A class counts as a sub-hint of a
        [`runtime_checkable`][typing.runtime_checkable] protocol
        declaring data members, such as one with `#!python name: str`,
        whenever it declares those members itself, read much as a type
        checker would read it. Either the class names the protocol among
        its own bases, or it declares each member as that member's own
        kind. An ordinary member is declared through an annotation such
        as `#!python name: str` (including a dataclass field), or
        through a class attribute, a property, or a slot, but never
        through a `#!python ClassVar`. A member the protocol itself
        declares as `#!python ClassVar` can only be declared back
        through a matching `#!python ClassVar` annotation. An annotation
        counts whether or not the attribute is ever actually assigned,
        and [`ishintstance`][] counts it the same way on every instance,
        to keep the two checks aligned. Unlike a type checker, a
        read-only member, such as a property with no setter or one
        typed `#!python Final`, still counts as satisfying an ordinary
        member here.

    !!! example
        ```pycon
        >>> from typing import List, Sequence, Union
        >>> issubhint(bool, int)
        True
        >>> issubhint(Union[int, str], Union[int, str, bytes])
        True
        >>> issubhint(Sequence[bool], Sequence[int])  # Sequence covariant
        True
        >>> issubhint(List[bool], List[int])  # list invariant
        False
        >>> issubhint(list, List[int])  # a bare list may hold anything
        False
        >>> issubhint(int, str)
        False
        ```
    """
    hint, superhint = normalise_hint(hint), normalise_hint(superhint)

    # A bottom (`Never`/`NoReturn`) holds no values, so it is a sub-hint of
    # every hint -- `Exact[C]` included, which is why this comes first.
    if _is_never(hint):
        return True

    # `Exact` first, before any `Annotated` metadata is unwrapped. `Exact[C]`
    # is a *leaf* subtype of `C`: an exactly-`C` value is a `C`, so
    # `Exact[C] <= C`, but neither `C` nor any subclass of `C` is exactly-`C`,
    # so nothing ordinary is `<= Exact[C]`. Keeping it a proper leaf is what
    # makes `<=` a preorder (reflexive and transitive) with `Exact` present.
    if is_exact(hint):
        target = normalise_hint(exact_target(hint))
        if is_exact(superhint):
            # `Exact[D] <= Exact[C]` iff `D` and `C` are the same type.
            supertarget = normalise_hint(exact_target(superhint))
            return _equivalent(target, supertarget)
        # A super-hint that *contains* `Exact` -- a union with an `Exact`
        # member, or a typevar bounded/constrained by one -- must distribute
        # first, so the exactness is matched member by member rather than lost
        # by reducing to `issubhint(target, superhint)`.
        sup_origin = get_origin_uw(superhint)
        if sup_origin in UNION_TYPES and get_args_uw(superhint):
            return any(
                issubhint(hint, arg) for arg in get_args_uw(superhint)
            )
        if isinstance(sup_origin, tx.TypeVar):
            return _issubtypevar(hint, superhint)
        # `Exact[D] <= P` iff `D <= P` (an exactly-`D` value is a `D`).
        return issubhint(target, superhint)
    if is_exact(superhint):
        target = normalise_hint(exact_target(superhint))
        # A parametrised union sub-hint distributes member by member, so a
        # union that is equivalent to a `Literal` (e.g.
        # `Union[Literal[1], Literal[2]]` == `Literal[1, 2]`) is ordered
        # against `Exact[C]` the same way that `Literal` is -- keeping the
        # relation transitive.
        if get_origin_uw(hint) in UNION_TYPES and get_args_uw(hint):
            return all(
                issubhint(member, superhint) for member in get_args_uw(hint)
            )
        # The only ordinary hints below `Exact[C]` are `Literal`s whose every
        # value has type exactly `C`: `Literal[1] <= Exact[int]`, but
        # `Literal[True]` (a `bool`) does not.
        if _is_literal(get_origin_uw(hint)):
            args = get_args_uw(hint)
            return bool(args) and all(type(arg) is target for arg in args)
        return False

    hint, superhint = _known_form(hint), _known_form(superhint)

    # Only a bottom is a sub-hint of a bottom (a bottom sub-hint has already
    # returned above).
    if _is_never(superhint):
        return False

    # shortcircuits
    if _is_any(superhint):
        return True

    if hint is superhint:
        return True

    # Unwrap superhint origin
    origin_uw = get_origin_uw(superhint)

    if _is_any(origin_uw):
        return True

    if isinstance(origin_uw, tx.TypeVar):
        return _issubtypevar(hint, superhint)

    if isinstance(hint, tx.TypeVar):
        # Read the typevar's bound/constraints (never its default) so it is
        # checked against the superhint exactly as the hint it stands for.
        # The superhint-is-a-typevar case is already handled above.

        # For constraints, each constraint must be a subhint of the
        # superhint
        constraints = getattr(hint, "__constraints__", ())
        if constraints:
            if origin_uw in UNION_TYPES:
                # Against a union, ask about the union the typevar
                # stands for: each constraint on its own is not a union,
                # but their combination is.
                return issubhint(_typevar_upper(hint), superhint)
            return all(
                issubhint(constraint, superhint)
                for constraint in constraints
            )

        # For bounds, the bound must be a subhint of the superhint
        return issubhint(_typevar_upper(hint), superhint)

    if not _is_literal(origin_uw) and _is_literal(get_origin_uw(hint)):
        # The hint is a Literal and the superhint is not, so the class,
        # union and NoneType branches below cannot see the literal's
        # values - they only ever compare origins. A Literal is a subhint
        # here iff every one of its values is valid for the superhint, the
        # same question ishintstance already answers for a single value.
        # A bare, unparametrised Literal has no values and stays False.
        args = get_args_uw(hint)
        return bool(args) and all(
            ishintstance(arg, superhint) for arg in args
        )

    if origin_uw in UNION_TYPES:
        return _issubunion(hint, superhint)

    # The dual of the Literal sub-hint rule above: the super-hint is now known
    # not to be a union, so a *parametrised* union sub-hint is a subhint iff
    # every one of its members is (`Union[bool, int] <= int`,
    # `Union[int, str] <= object`). A bare, unparametrised `Union` has no
    # members and falls through to the branches below.
    if get_origin_uw(hint) in UNION_TYPES and get_args_uw(hint):
        return all(
            issubhint(member, superhint) for member in get_args_uw(hint)
        )

    if _is_literal(origin_uw):
        return _issubliteral(hint, superhint)

    if origin_uw is type(None):
        return _issubnone(hint, superhint)

    if origin_uw is type:
        return _issubtype(hint, superhint)

    if origin_uw is abc.Callable:
        return _issubcallable(hint, superhint)

    if is_typeddict_marker(origin_uw):
        # The bare `TypedDict` marker as a super-hint: `safe_issubclass`
        # reads it structurally. Without this it would fall to the opaque
        # rule below and accept everything.
        return _issubclasshint(hint, superhint, origin_uw)

    if is_special_form(origin_uw):
        # A recognised special form with no branch of its own and nothing left
        # to check -- a bare, unsubscripted `Annotated` (its origin is itself,
        # not a class), or a future class-shaped construct -- is opaque:
        # permissive like `Any`, so a hint or registry key written with it
        # stays reachable (RFC 0001 §2.1) rather than being mistaken for a
        # subclassable class below. A subscripted form has already resolved to
        # its inner origin above, so only a bare one reaches here.
        return True

    # A bare `TypedDict` marker on the hint side means "any TypedDict". Every
    # TypedDict-shaped value is a `dict`, so the marker is a sub-hint of `dict`
    # (and of whatever `dict` is a sub-hint of, e.g. `Mapping`, `object`); rank
    # it as `dict` here. This is reached only after the union, `TypeVar`,
    # literal and marker-as-superhint branches above have had their turn, so a
    # super-hint that merely *contains* the marker (`Optional[TypedDict]`, a
    # `TypeVar` bound by it, an `Annotated` wrapper, the other spelling) still
    # matches through those; the redirect applies only against a plain class
    # super-hint. A plain `dict` is not a sub-hint of the marker (that goes
    # through the marker-as-superhint branch above and is `False`), so the
    # marker stays strictly below `dict` -- which makes `TypedDict` the unique
    # most-specific key over `dict` in `resolve_hint` (RFC 0001 §8.1).
    if is_typeddict_marker(get_origin_uw(hint)):
        return issubhint(dict, superhint)

    if isinstance(origin_uw, type):
        return _issubclasshint(hint, superhint, origin_uw)

    # A recognised typing construct with no branch of its own -- or a future
    # form -- is opaque: treated as `Any` so a method annotated with it stays
    # reachable, and reported once. An object that is plainly *not* a hint (a
    # value, a plain function) is a caller error, so it raises instead.
    if is_plausible_hint(superhint):
        _warn_unknown(superhint)
        return True
    raise TypeError(_not_a_hint_message(superhint))


def _issubclasshint(hint: tx.Any, superhint: tx.Any, origin: type) -> bool:
    """Report whether `hint` is a sub-hint of a `superhint` whose origin
    is a class.
    """
    # Compare origins, not the hints themselves: a parametrised alias
    # (`List[int]`) and an `Annotated` wrapper are not instances of
    # `type`, so handing either to `safe_issubclass` directly would
    # answer False for every one of them.
    hint_uw = unwrap(hint)
    if not _issubclass_origin(get_origin_uw(hint_uw), origin):
        return False

    if origin is tuple:
        # A tuple superhint reads its arguments as a shape, and the empty-tuple
        # type `Tuple[()]` must be told from a bare, unparametrised
        # `Tuple`/`tuple`: the two report the same empty arguments on 3.11+, so
        # the "no arguments constrains nothing" rule below would wrongly accept
        # every tuple as a sub-hint of `Tuple[()]`.
        return _issubtuplehint(hint_uw, unwrap(superhint))

    superargs = safe_get_args(unwrap(superhint))
    if not superargs:
        # An unparametrised superhint constrains nothing further.
        return True

    variances = _generic_variances(origin)
    if get_origin_uw(hint_uw) is not origin:
        # Differing origins: read the sub-hint as the parametrisations of the
        # super-hint's origin it declares through its bases (#50, V5), so
        # `class IntBox(Box[int])` is compared as `Box[int]` and `Flip[int,
        # str]` for `class Flip(Pair[B, A], Generic[A, B])` as `Pair[str,
        # int]`. A class reaching the origin through several bases is below
        # the super-hint when any of them is, and the walk stops at the
        # first that is. When no base maps onto the origin the arguments are
        # compared positionally, as before.
        mapped = False
        for args in _as_base_args(hint_uw, origin):
            if _issubclassargs(args, superargs, variances):
                return True
            mapped = True
        if mapped:
            return False
    return _issubclassargs(safe_get_args(hint_uw), superargs, variances)


def _issubclassargs(
    args: tx.Tuple[tx.Any, ...],
    superargs: tx.Tuple[tx.Any, ...],
    variances: tx.Optional[tx.Tuple[str, ...]],
) -> bool:
    """Report whether a class hint's arguments fit a super-hint's, slot by
    slot.

    `variances` belongs to the super-hint's origin, obtained from
    [`_generic_variances`][].
    """
    if not args:
        # `list` cannot stand in for `List[int]`: it may hold anything.
        return False
    if variances is not None and len(variances) == len(args) == len(superargs):
        # Compare each position by the variance the super-hint's origin
        # declares for it (#50): a covariant slot forwards the relation, a
        # contravariant slot reverses it, an invariant slot demands equality.
        return all(
            _issubslot(arg, superarg, variance)
            for arg, superarg, variance in zip(args, superargs, variances)
        )
    # No readable per-position variance -- a `ParamSpec`/`TypeVarTuple` origin,
    # an arity mismatch, or an origin the table does not cover -- keeps the
    # covariant argument comparison.
    return _issubargs(args, superargs)


def _issubclass_origin(sub: tx.Any, origin: tx.Any) -> bool:
    """Report whether the class `sub` is below the class `origin`, for the
    relation.

    This behaves like [`safe_issubclass`][], except where a
    runtime-checkable protocol with data members is involved, a case
    Python's own `issubclass` either refuses outright as the super side
    or reads too loosely as the sub side; see
    [`_declares_protocol`][] and [`_protocol_below_protocol`][] for how
    that case is handled instead.
    """
    members = _data_protocol_members(origin)
    if members is not None:
        return _declares_protocol(sub, origin, members)
    # From here on `origin` is not a data protocol: a method-only runtime
    # protocol is the one case a data protocol `sub` is read member-wise.
    sub_members = _data_protocol_members(sub)
    if (
        sub_members is not None
        and getattr(origin, "_is_runtime_protocol", False)
        and tx.is_protocol(origin)
    ):
        return _protocol_below_protocol(sub, sub_members, origin)
    return safe_issubclass(sub, origin)


def _issubslot(sub: tx.Any, sup: tx.Any, variance: str) -> bool:
    """Report whether a sub-side argument fits a super-side one at one slot.

    `variance` is the declared variance of a generic parameter position,
    one of the string constants defined in [`_introspect`][], read off
    the super-hint's origin. Following PEP 484, a covariant position
    requires `sub` to be a sub-hint of `sup`, so a covariant container
    narrows along with its item type, as in
    `Sequence[bool] <= Sequence[int]`; a `TypeVar` here is read as its
    bound or constraints, exactly what solving it would give, so
    `G[A] <= G[T]` for some `T <= B` holds exactly when `A <= B`. A
    contravariant position requires `sup` to be a sub-hint of `sub`
    instead, so a consumer of `int` can stand in for a consumer of
    `bool`; a `TypeVar` is read as its bound here too, and
    [`_issubslot_invariant`][] explains why it is never solved this way.
    An invariant position is handled by [`_issubslot_invariant`][]:
    either the two sides accept exactly the same values, or the super
    side is a `TypeVar` that can be solved to the sub side, or the super
    side is a top such as `Any` or a free `TypeVar`.
    """
    if variance == _COVARIANT:
        return issubhint(sub, sup)
    if variance == _CONTRAVARIANT:
        # Solving a `TypeVar` here would ask whether the two sides *overlap*
        # (`Snk[bool] <= Snk[T <= int]` needs some `T` below both `bool` and
        # `int`), which is not transitive -- `D(B, C)` is below both `B` and
        # `C`, which are not related -- and cannot be decided over an open
        # class hierarchy. So a contravariant slot keeps reading a `TypeVar`
        # as its bound, which keeps the order a preorder (#50, V5).
        return issubhint(sup, sub)
    return _issubslot_invariant(sub, sup)


def _issubslot_invariant(sub: tx.Any, sup: tx.Any) -> bool:
    """Report whether `G[sub] <= G[sup]` holds at an invariant slot.

    A `TypeVar` on the super side stands for some single type within its
    bound or constraints, the way a type checker would solve it; on the
    sub side, though, it stands for a whole family of types, which no
    single type can contain. When `sup` is `Any` or a free `TypeVar`, it
    acts as the top that the slot may widen to under gradual consistency,
    so a generic fallback overload still sits above every specialisation
    of it. When `sup` is a bounded `TypeVar` `T <= B`, `T` can be solved
    to `sub` exactly when `sub <= B`, reading a `TypeVar` `sub` by its
    own bound. So `Box[bool] <= Box[T <= int]` holds, and
    `Box[T1 <= B1] <= Box[T2 <= B2]` holds exactly when `B1 <= B2`. When
    `sup` is a constrained `TypeVar`, `T` is solved to one of its
    constraints, so `sub` must be equivalent to one of them, or, when
    `sub` is itself constrained, each of its constraints must be
    equivalent to one of `sup`'s. In every other case `sup` is a
    concrete type, and `sub` must be equivalent to it; a `TypeVar`
    `sub` never satisfies this, since `Box[T <= int]` is not a
    `Box[int]`, because `T` could still turn out to be `bool`.

    This relation is transitive, since every one of these rules reduces
    either to `<=` or to equivalence against the super side's bound or
    constraints, and both of those chain.
    """
    if sub is sup or issubhint(tx.Any, sup):
        return True
    sub_uw = unwrap(normalise_hint(sub), tx.Annotated)
    sup_uw = unwrap(normalise_hint(sup), tx.Annotated)
    sub_is_typevar = isinstance(sub_uw, tx.TypeVar)
    if isinstance(sup_uw, tx.TypeVar):
        constraints = getattr(sup_uw, "__constraints__", ())
        if not constraints:
            # Bounded (a free one is the top, above): `sub` within the bound.
            return issubhint(sub, sup)
        if sub_is_typevar:
            members = getattr(sub_uw, "__constraints__", ())
            if not members:
                # A bounded or free family is not one constraint.
                return False
        else:
            members = (sub,)
        return all(
            any(_equivalent(member, each) for each in constraints)
            for member in members
        )
    if sub_is_typevar:
        return False
    return _equivalent(sub, sup)


# --- declared parametrisations (#50, V5) -------------------------------

# The markers a generic lists its own type variables with. A base written
# `Generic[T]` / `Protocol[T]` says which variables the class takes, not what
# it inherits, so the base walk below never follows (or subscripts) one.
_PARAMETER_MARKERS = spellings("Generic") + spellings("Protocol")


def _own_bases(cls: type) -> tx.Tuple[tx.Any, ...]:
    """List the bases `cls` was written with, parametrised where they were.

    `__orig_bases__` is read directly off the class's own namespace,
    since ordinary attribute access would instead find a parent's,
    describing the parent's bases rather than this class's own. A class
    written with no parametrised base has none of its own, so its plain
    `__bases__` serve as the answer instead.
    """
    return _own_orig_bases(cls) or cls.__bases__


def _free_parameters(node: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """List the type variables a generic class or alias takes, or `()`.

    A class is read through [`_class_parameters`][], so one written
    against a PEP 585 base, such as `#!python class GL(list[T])`, takes
    whichever variables that base mentions; an alias, such as
    `#!python Box[T]` or `#!python list[T]`, simply lists its own.
    """
    if _looks_like_class(node):
        return _class_parameters(node)
    params = getattr(node, "__parameters__", ())
    return params if isinstance(params, tuple) else ()


def _filled_bases(
    node: tx.Any, cls: type
) -> tx.Optional[tx.Tuple[tx.Any, ...]]:
    """List `cls`'s written bases, with `node`'s arguments filled in.

    `node` is `cls` itself or a parametrisation of it, such as
    `Sub[bool]`. Each base that mentions one of `cls`'s type variables is
    subscripted with whatever `node` gives that variable, using typing's
    own substitution, where `Box[T][bool]` becomes `Box[bool]`, the
    same mechanism that resolves a generic type alias; a base that
    mentions none of those variables, such as `Box[int]` or a plain
    class, is kept exactly as written. Arguments are paired to variables
    by identity rather than by position, because a class can list its
    variables in a different order than one of its bases does, as in
    `class Flip(Pair[B, A], Generic[A, B])`.

    `#!python None` comes back when `node` leaves the bases
    undetermined: a generic class written with no arguments at all, a
    bare `Sub`, or a `ParamSpec` or `TypeVarTuple` generic, whose
    arguments do not pair one to one with its variables. A base whose
    substitution raises an error is simply dropped from the result.
    """
    params = _free_parameters(cls)
    if not params:
        return _own_bases(cls)
    args = tx.get_args(node)
    if len(args) != len(params) or not all(
        _is_plain_typevar(param) for param in params
    ):
        return None
    filled_in = dict(zip(params, args))
    bases = ()  # type: tx.Tuple[tx.Any, ...]
    for base in _own_bases(cls):
        origin = tx.get_origin(base)
        if origin is None:
            # A plain class base mentions no variable.
            bases += (base,)
            continue
        if any(origin is marker for marker in _PARAMETER_MARKERS):
            continue
        base_params = _free_parameters(base)
        if not base_params:
            bases += (base,)
            continue
        fill = tuple(filled_in.get(param, param) for param in base_params)
        try:
            bases += (base[fill if len(fill) > 1 else fill[0]],)
        except Exception:
            # A base that refuses its arguments maps onto nothing; the walk
            # carries on through the others.
            continue
    return bases


def _as_base_args(
    hint: tx.Any, target: type
) -> tx.Iterator[tx.Tuple[tx.Any, ...]]:
    """Yield `hint` re-expressed as parametrisations of `target`, as their
    arguments.

    `hint` here is a class or a parametrised generic whose origin is a
    subclass of `target`. Each value yielded is what `hint` fills
    `target`'s parameters with, worked out by walking the bases each
    class was written with, `__orig_bases__`, filling in each class's
    own arguments along the way.

    `class IntBox(Box[int])` is written as `Box[int]`, so `IntBox`
    yields `(int,)`. `class Sub(Box[T])` passes its own argument
    through, so `Sub[bool]` yields `(bool,)`, and a subclass written with
    no parametrised base of its own, such as `class Leaf(IntBox)`, is
    followed through its plain bases instead. Every base is followed, so
    a class reaching `target` along two separate paths yields both. In a
    diamond `class D(A, B)` with `class A(Box[int])` and
    `class B(Box[str])`, `D` yields `(int,)` and then `(str,)`, and the
    same happens for `class Two(List[T], Container[U])`, whose
    `Two[int, str]` is a `Container[int]` through `List` and a
    `Container[str]` through its own base. A type checker would reject
    such a class outright, while this relation instead accepts a hint
    satisfying either reading. A standard-library class reached along the
    way, such as `class Child(List[int])` reaching `List[int]`, is
    compared positionally against a standard-library `target` it
    subclasses, such as `Sequence`, since two standard-library origins
    are always compared that way.

    `hint`'s own arguments are yielded directly when its origin is
    `target` itself and it actually has arguments. Nothing is yielded at
    all when nothing maps onto `target`, as when no base reaches it with
    arguments, for instance for a `collections.Counter`, a runtime
    subclass of `dict` recording no parametrised base, or for a generic
    class written with no arguments; the caller then falls back to its
    own positional comparison.

    This is a generator that walks depth first, following each class's
    bases in the order they are listed. A caller that only needs one
    answer can stop the walk there, and a class whose first base maps
    onto `target` yields its answer after as many steps as it is deep.
    Each node, a class together with the arguments it was reached with,
    is walked only once, and each answer is yielded only once
    ([`_first_time`][]); a hierarchy that reaches `target` along several
    paths carrying different arguments, such as a diamond repeated at
    every level, still produces as many answers as there are paths, so a
    caller that needs all of them has to walk them all.
    """
    pending = [hint]
    walked = {}  # type: tx.Dict[tx.Tuple[int, ...], tx.Any]
    given = {}  # type: tx.Dict[tx.Tuple[int, ...], tx.Any]
    while pending:
        node = pending.pop()
        cls = safe_get_origin(node)
        args = tx.get_args(node)
        if cls is target:
            if args and _first_time(args, args, given):
                yield args
            continue
        if not _looks_like_class(cls) or not _first_time(
            (cls,) + args, node, walked
        ):
            # A class reached again with the same arguments has already
            # given what it maps to. One reached with other arguments -- in
            # a diamond, `class A(Mid[int])` and `class B(Mid[str])` -- is
            # walked again, so each of its parametrisations maps.
            continue
        if args and "__orig_bases__" not in vars(cls):
            # A parametrised standard-library class (`List[int]`, `Dict[K,
            # V]`), which records no parametrised base: its arguments line up
            # with a standard-library `target` it subclasses and that takes as
            # many (`Sequence`, `Mapping`) -- the positional reading two such
            # origins are given. One that takes a different number
            # (`Dict[K, V]` against `Iterable`, `Tuple[int, int]` against
            # `Sequence`) cannot be paired up, so nothing maps.
            variances = _generic_variances(target)
            if (
                variances is not None
                and len(variances) == len(args)
                and safe_issubclass(cls, target)
                and _first_time(args, args, given)
            ):
                yield args
            continue
        bases = _filled_bases(node, cls)
        if bases is not None:
            # Depth first, the first-listed base on top.
            pending.extend(reversed(bases))


def _first_time(
    parts: tx.Tuple[tx.Any, ...],
    holder: tx.Any,
    met: tx.Dict[tx.Tuple[int, ...], tx.Any],
) -> bool:
    """Record `parts` in `met` by identity, and report whether they are new.

    The key is built from each part's `id`, so no argument's own
    `__eq__` or `__hash__` ever runs, and two equal but separately built
    objects, such as `list[int]` written twice, count as different,
    which simply means that node gets read a second time and yields the
    same answers again. `holder`, either the node or the answer itself,
    is stored as the entry's value, so every object whose `id` appears
    in a key stays alive, and so keeps that `id` meaningful, for as long
    as the walk runs.
    """
    key = tuple(map(id, parts))
    if key in met:
        return False
    met[key] = holder
    return True


def _is_fully_declared(args: tx.Optional[tx.Sequence[tx.Any]]) -> bool:
    """Report whether declared arguments actually say what each parameter
    holds.

    `#!python False` comes back for no arguments at all, and for
    arguments that mention a type variable anywhere inside them, as
    `class Child(List[T])` leaves `T` open, or that mention
    [`Any`][typing.Any], or a name not yet resolved, as when
    `Box["int"]()` records `Box[ForwardRef('int')]`. Each of these
    leaves what the value actually holds undeclared, so none of them may
    be used to narrow which parametrisations it matches.
    """
    if not args:
        return False
    for arg in args:
        # The `TypeVar` family first: on 3.8 the `ParamSpec` backport is a
        # `list`, and would otherwise be read as a parameter list.
        if isinstance(arg, tx.TypeVar) or _is_any(arg):
            return False
        if isinstance(arg, (str, tx.ForwardRef)):
            return False
        if _NON_TYPE_PARAMS and isinstance(arg, _NON_TYPE_PARAMS):
            return False
        if isinstance(arg, (list, tuple)):
            # A `Callable`'s parameter list.
            if arg and not _is_fully_declared(arg):
                return False
            continue
        inner = tx.get_args(arg)
        if inner and not _is_literal(tx.get_origin(arg)):
            if not _is_fully_declared(inner):
                return False
    return True


# `_may_record_parametrisation`'s memo: the answer for each class, keyed by
# the class's `id`. It is read on every call at a declaration-dependent
# argument, and a plain `dict` read is the cheapest there is; keying by `id`
# holds no reference to the class, so a class made and dropped at runtime is
# not kept alive, as `_LOOKUPS` does not keep one alive either. The weak
# reference kept beside each answer drops the entry when its class is
# collected -- on CPython, before that `id` can be given to another object,
# since the callback runs as the class is freed. That guarantee is
# CPython's: PyPy defers weakref callbacks, so there it does not hold.
_RECORDERS = {}  # type: tx.Dict[int, bool]
_RECORDER_REFS = {}  # type: tx.Dict[int, weakref.ref]


def _may_record_parametrisation(cls: type) -> bool:
    """Report, and cache, whether an instance of `cls` is asked for
    `__orig_class__`.

    Calling a subscripted generic class records that subscription on the
    instance it builds. Two kinds of class can be subscripted this way:
    a [`Generic`][typing.Generic] subclass, as in
    `#!python Box[int]()`, and a class written against a PEP 585 alias,
    such as `#!python class GL(list[T])`. Even though the second kind
    has no `Generic` in its MRO at all, its `#!python GL[int]()` is
    still recorded by the runtime alias type. An instance of any other
    class, such as a builtin
    container or a lazy proxy whose `__getattr__` performs real work, is
    never probed for this.

    Both the value check ([`_declared_parametrisation`][]) and the call
    cache's own key (`_declared_key`) ask this same question, so the key
    always covers exactly what the check reads.
    """
    try:
        return _RECORDERS[id(cls)]
    except KeyError:
        pass
    answer = _records_parametrisation(cls)
    key = id(cls)
    _RECORDER_REFS[key] = weakref.ref(cls, functools.partial(_forget, key))
    _RECORDERS[key] = answer
    return answer


def _forget(key: int, _ref: tx.Any) -> None:
    """Drop [`_may_record_parametrisation`][]'s cached answer for a
    collected class.
    """
    _RECORDERS.pop(key, None)
    _RECORDER_REFS.pop(key, None)


def _records_parametrisation(cls: type) -> bool:
    """Compute [`_may_record_parametrisation`][]."""
    if issubclass(cls, tx.Generic):
        return True
    return any(
        _is_pep585_alias(base)
        for each in cls.__mro__
        for base in _own_orig_bases(each)
    )


def _orig_class(obj: tx.Any) -> tx.Any:
    """Find the parametrisation `obj` was built from, or `#!python None`.

    Calling a subscripted user generic, such as `Box[int]()`, or
    `GL[int]()` for `class GL(list[T])`, records that subscription on
    the new instance as `__orig_class__`. It is absent from a builtin
    container and from any instance built by calling the bare class. It
    is also absent from an instance typing simply cannot write it onto,
    such as one of a class with `__slots__` and no `__dict__`, or of a
    frozen dataclass, where typing quietly swallows the resulting
    `FrozenInstanceError`. It is also only written after `__init__`
    returns, so a dispatch on `self` from inside `__init__` still sees
    an instance that declares nothing yet. Whatever the attribute holds
    is trusted only once it turns out to be a parametrisation of a class
    `obj` is genuinely an instance of.
    """
    try:
        declared = obj.__orig_class__
    except Exception:
        # Absent (`AttributeError`), or a `__getattr__` that raises.
        return None
    if not tx.get_args(declared):
        return None
    if not safe_issubclass(type(obj), safe_get_origin(declared)):
        return None
    return declared


def _declared_parametrisation(obj: tx.Any, origin: type) -> tx.Any:
    """Find what `obj` declares itself to be, as a parametrisation of `origin`.

    `origin` is the class a parametrised hint, `G[args]`, is written on,
    and `type(obj)` is already known to be a subclass of it. The result
    is a hint suitable for comparing against `G[args]` through the
    relation, worked out by trying each of the following in order.

    First, the instance's own `__orig_class__`, such as `Box[int]` for
    `Box[int]()`, is used when `obj` is an instance of a `Generic`
    subclass, or of a class written against a PEP 585 alias, such as
    `GL[int]()` for `class GL(list[T])`
    ([`_may_record_parametrisation`][]). This only applies provided that
    record declares every argument of `origin` along every base reaching
    it, whether `origin` is a user generic or a standard-library one
    ([`_declares_arguments`][]); that is what makes `Row[int]()` for
    `class Row(Sequence[T])` a `Sequence[int]`. Failing that,
    `type(obj)` itself is used when the class declares every argument of
    `origin` along every base reaching it, as `class Child(List[int])`
    does. Failing both, the result is `#!python None`, meaning the
    value declares nothing and only its origin can be checked.

    Only an origin whose parameters line up one readable argument each
    is even considered here ([`_reads_declared_arguments`][]). `Tuple`
    and `Callable`, whose argument lists describe shapes rather than
    positions, a `ParamSpec` or `TypeVarTuple` generic such as
    `Hook[[int]]`, and any origin whose parameters cannot be read at all,
    all keep to the shallow check instead.
    """
    if not _reads_declared_arguments(origin):
        return None
    cls = type(obj)
    if _may_record_parametrisation(cls):
        # Only an instance of a class that can be subscripted into a record
        # is asked -- the same gate the call cache applies before reading it
        # (`_declared_key`), so the key always covers what the check reads.
        declared = _orig_class(obj)
        if declared is not None and _declares_arguments(declared, origin):
            return declared
    if _declares_arguments(cls, origin):
        return cls
    return None


def _declares_arguments(hint: tx.Any, origin: type) -> bool:
    """Report whether `hint` says what `origin`'s parameters hold, through
    its bases.

    Every parametrisation of `origin` reached through the bases
    ([`_as_base_args`][]) has to be fully declared for this to hold. The
    hint is then compared against the super-hint through the relation,
    which accepts it as long as any one of those parametrisations fits.
    One that leaves an argument open, as when
    `class Two(List[T], Container[U])` built as `Two[Any, str]()`
    reaches `Container[Any]` through `List`, makes the whole value
    count as undeclared, so only its origin is checked from then on.
    Reading it through its other base alone would wrongly reject it as a
    `Container[bytes]`, whereas it is correctly accepted as a
    `Collection[bytes]`, which sits below that.
    """
    declared = False
    for args in _as_base_args(hint, origin):
        if not _is_fully_declared(args):
            return False
        declared = True
    return declared


def _is_subscripted_tuple(hint: tx.Any) -> bool:
    """Report whether a tuple hint is subscripted, as `Tuple[int]` or
    `Tuple[()]`.

    This tells a subscripted tuple apart from a bare, unparametrised
    `Tuple` or `tuple`. The empty-tuple type `Tuple[()]` reports its
    arguments as the phantom `#!python ((),)` on Python 3.8 through 3.10
    and as a genuinely empty `#!python ()` from 3.11 on, so an empty
    argument list alone cannot distinguish it from a bare `Tuple`. A
    subscripted alias with no arguments still carries an `__args__`
    attribute that a bare form lacks, and that distinction is the
    load-bearing fallback for a genuinely empty argument list, covering
    both the PEP 585 `#!python tuple[()]` spelling on 3.9 and 3.10 and
    every empty-args form from 3.11 on.
    """
    if any(hint is form for form in _BARE_TUPLE_FORMS):
        return False
    if tx.get_args(hint):
        # `Tuple[int]`, or the `Tuple[()]` phantom `((),)` on 3.8-3.10.
        return True
    # A genuinely empty `Tuple[()]` carries `__args__` on 3.11+; a bare `Tuple`
    # (excluded by identity above) does not.
    return hasattr(hint, "__args__")


def _issubtuplehint(hint_uw: tx.Any, superhint_uw: tx.Any) -> bool:
    """Report whether a tuple hint is a sub-hint of a tuple superhint."""
    if not _is_subscripted_tuple(superhint_uw):
        # A bare `Tuple`/`tuple` constrains nothing: any tuple is a sub-hint.
        return True
    if not _is_subscripted_tuple(hint_uw):
        # A bare `tuple` may hold anything, so it cannot stand for a
        # parametrised tuple (including the empty-tuple type `Tuple[()]`).
        return False
    return _issubargs(safe_get_args(hint_uw), safe_get_args(superhint_uw))


def _issubargs(
    args: tx.Tuple[tx.Any, ...], superargs: tx.Tuple[tx.Any, ...]
) -> bool:
    """Check a hint's arguments against a superhint's, treating each as
    covariant.

    This serves as the covariant fallback for a generic with no readable
    per-position variance, such as one with a `ParamSpec` or
    `TypeVarTuple` origin, or one with mismatched arity; an origin whose
    variance is known is instead compared slot by slot, each according to
    its own declared variance, in [`_issubclasshint`][].

    The arguments are read as tuple shapes made of a fixed prefix, an
    optional open run, and a fixed suffix, so that a trailing ellipsis,
    as in `Tuple[int, ...]`, and an unpacked `TypeVarTuple`, as in
    `Tuple[int, *Ts]`, are both ordered the same way. A plain, fully
    fixed argument list, such as `List[int]` or `Dict[str, int]`, is
    simply a closed shape, and is compared element by element as before.
    """
    return _issubtupleshape(_tuple_shape(args), _tuple_shape(superargs))


def _issubnone(hint: tx.Any, superhint: tx.Any) -> bool:
    """Report whether a hint is a sub-hint of `NoneType`, the super-hint
    here.
    """
    none_uw = get_origin_uw(superhint)
    if none_uw is not type(None):
        raise TypeError(f"nonehint {superhint} is not a NoneType")
    origin_uw = get_origin_uw(hint)
    return origin_uw is type(None)


def _issubliteral(hint: tx.Any, superhint: tx.Any) -> bool:
    """Report whether a hint is a sub-hint of a `Literal` super-hint."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if not _is_literal(safe_get_origin(superhint_uw)):
        # Superhint is not a literal -> error
        raise TypeError(f"Super-hint {superhint} is not a Literal")
    if not _is_literal(safe_get_origin(hint_uw)):
        # Hint is not a Literal, cannot be a subhint
        return False
    # !! We use tx.get_origin instead of safe_get_origin
    # !! to differentiate tx.Literal (origin is None)
    # !! from tx.Literal[()] (origin is tx.Literal)
    if not tx.get_origin(superhint_uw):
        # All literals are subhints of `tx.Literal`
        return True
    if not tx.get_origin(hint_uw):
        # tx.Literal is not a subhint of tx.Literal[...]
        return False
    # Check that every arg of hint matches one of superhint's, type-aware:
    # `Literal[1]` is not a sub-hint of `Literal[True]` or `Literal[1.0]`
    # even though `1 == True == 1.0` in Python (PEP 586).
    args = safe_get_args(hint_uw)
    superargs = safe_get_args(superhint_uw)
    return all(
        any(_literal_value_eq(a, s) for s in superargs) for a in args
    )


def _issubtypevar(hint: tx.Any, superhint: tx.Any) -> bool:
    """Report whether a hint is a sub-hint of a `TypeVar` super-hint."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if not isinstance(superhint_uw, tx.TypeVar):
        # Invalid superhint -> error
        raise TypeError(f"Super-hint {superhint} is not a TypeVar")
    if hint_uw is superhint_uw:
        # Exact match
        return True
    if getattr(superhint_uw, "__constraints__", ()):
        # If constraints, check that hint is a subhint of one of them
        constraints = superhint_uw.__constraints__
        for constraint in constraints:
            if issubhint(hint, constraint):
                return True
        # A constrained typevar is equivalent to the union of its
        # constraints, so a union hint is a subhint when every member is a
        # subhint of some constraint (`Union[C1, C2] <= TypeVar(_, C1, C2)`,
        # the symmetric partner of `TypeVar(...) <= Union[...]`).
        if get_origin_uw(hint) in UNION_TYPES:
            members = get_args_uw(hint)
            if members:
                return all(
                    any(issubhint(member, constraint)
                        for constraint in constraints)
                    for member in members
                )
        # Else, if hint is a TypeVar, check that all its constraints are
        # subhints of one of the superhint's constraints
        if isinstance(hint_uw, tx.TypeVar):
            subconstraints = getattr(hint_uw, "__constraints__", ())
            if not subconstraints:
                return False
            return all(
                any(
                    issubhint(subconstraint, constraint)
                    for constraint in constraints
                )
                for subconstraint in subconstraints
            )
        # Otherwise, constraints do not match
        return False
    elif getattr(superhint_uw, "__bound__", None) is not None:
        # If bound, check that hint is a subhint of the bound
        bound = superhint_uw.__bound__
        if issubhint(hint, bound):
            return True
        # Else, if hint is a TypeVar, check that its bound is a subhint of
        # the superhint's bound
        if isinstance(hint_uw, tx.TypeVar):
            subbound = getattr(hint_uw, "__bound__", None)
            if subbound is None:
                return False
            return issubhint(subbound, bound)
        # Otherwise, bound does not match
        return False
    else:
        # Unconstrained TypeVar -> any hint is a subhint
        return True


def _issubunion(hint: tx.Any, superhint: tx.Any) -> bool:
    """Report whether a hint is a sub-hint of a `Union` super-hint."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) not in UNION_TYPES:
        # Invalid superhint -> error
        raise TypeError(f"union {superhint} is not a Union type")
    # !! We use tx.get_origin instead of safe_get_origin
    # !! to differentiate tx.Union (origin is None)
    # !! from tx.Union[...] (origin is tx.Union)
    if not tx.get_origin(superhint_uw):
        # Every union - and only a union - is a subhint of the bare
        # `tx.Union`, the same rule `_issubliteral` applies for a bare
        # `Literal`. A *parametrised* union is different: a plain type
        # is a subhint of one that contains it, which the member logic
        # below works out.
        return safe_get_origin(hint_uw) in UNION_TYPES
    # Collect the hint's member hints. A hint is a subhint of the union if
    # each of its members is a subhint of one of the union's members:
    #   * a parametrised union contributes its arguments;
    #   * the bare `tx.Union` is not a subhint of a parametrised union;
    #   * any other hint (e.g. `int`) is a single member, so that a plain
    #     type is a subhint of a union that contains it.
    if safe_get_origin(hint_uw) in UNION_TYPES:
        if not tx.get_origin(hint_uw):
            return False
        members = safe_get_args(hint_uw)
    else:
        members = (hint_uw,)
    superargs = safe_get_args(superhint_uw)
    return all(
        any(issubhint(member, superarg) for superarg in superargs)
        for member in members
    )


def _issubtype(hint: tx.Any, superhint: tx.Any) -> bool:
    """Report whether a hint is a sub-hint of a `type[...]` super-hint."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) is not type:
        # Invalid superhint -> error
        raise TypeError(f"superhint {superhint} is not a type")
    if safe_get_origin(hint_uw) is not type:
        # Hint is not a type, cannot be a subhint
        return False
    if not tx.get_args(superhint_uw):
        # All types are subhints of `tx.Type`
        return True
    if not tx.get_args(hint_uw):
        # tx.Type is not a subhint of tx.Type[...]
        return False
    # Check that the hint's arg is a subclass of the superhint's arg
    args = safe_get_args(hint_uw)
    superargs = safe_get_args(superhint_uw)
    return _issubclass_origin(args[0], superargs[0])


def _issubcallable(hint: tx.Any, superhint: tx.Any) -> bool:
    """Report whether a hint is a sub-hint of a `Callable[...]` super-hint.

    Parameters are compared contravariantly, and the return type
    covariantly. A `#!python ...` or a bare `ParamSpec` parameter list
    is the widest possible parameter list, describing every callable, so
    a fixed list is a sub-hint of it but never the reverse; a
    `#!python Concatenate[X, P]` list is instead a contravariant fixed
    prefix followed by an open tail, sitting between the fixed lists and
    that widest one, per RFC 11.1. A callable class, such as a function
    type, `#!python type`, `#!python Type[C]`, or a class defining
    `__call__`, has no parameter list of its own, so it can only stand
    in for an unparametrised `Callable`.
    """
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    superargs = safe_get_args(superhint_uw)
    hint_origin = get_origin_uw(hint_uw)
    if hint_origin is not abc.Callable:
        # A callable *class* has no parameter list: it can only stand in for a
        # bare `Callable`, exactly as `list` cannot stand in for `List[int]`.
        if not safe_issubclass(hint_origin, abc.Callable):
            return False
        return not superargs
    if not superargs:
        # A bare `Callable` constrains nothing further.
        return True
    args = safe_get_args(hint_uw)
    if not args:
        # A bare `callable` may accept anything: it cannot stand in for a
        # parametrised `Callable[...]`.
        return False
    # `(params, return)`: `params` is a list, `...`, or a ParamSpec form.
    sub_params, sub_ret = args[0], args[-1]
    sup_params, sup_ret = superargs[0], superargs[-1]
    # Return type is covariant.
    if not issubhint(sub_ret, sup_ret):
        return False
    return _issubparams(
        _callable_param_shape(hint_uw, sub_params),
        _callable_param_shape(superhint_uw, sup_params),
    )


class _ParamShape(tx.NamedTuple):
    """A `Callable` parameter list reduced to a shape, per RFC 11.1.

    A `_ParamShape` is a fixed, contravariant `prefix` followed by a
    `tail` describing how the list ends: `#!python None` for a closed,
    fixed-arity list such as `[int, str]`, or `#!python Ellipsis` or a
    [`ParamSpec`][typing.ParamSpec] for an open list, one that could be
    called with arbitrarily many further arguments.

    An open list sits at the top of every parameter list ordering, the
    way `#!python Tuple[X, ...]` sits above the fixed-length tuples: a
    fixed list is a sub-hint of it, but it is never a sub-hint of any
    fixed list in turn.
    """

    prefix: tx.Tuple[tx.Any, ...]
    tail: tx.Any


# A sentinel `_match_params` returns when a closed super matches: there is no
# tail to capture, but the result must be non-`None` to signal the match.
_CLOSED_MATCH = _ParamShape((), None)


def _callable_param_shape(alias: tx.Any, params: tx.Any) -> _ParamShape:
    """Classify a `Callable` parameter list into a `_ParamShape`, per RFC 11.1.

    `alias` is the whole `Callable[...]` hint, needed here to reach
    `#!python __parameters__` on an older Python that has already erased
    a `ParamSpec` out of the arguments themselves; `params` is that
    hint's parameter-list argument.
    """
    # `Ellipsis` and the `ParamSpec` forms name the open top. `ParamSpec` is
    # tested before `list`: on 3.8/3.9 a `ParamSpec` *is* a `list` subclass,
    # so the list branch would otherwise claim it.
    if params is Ellipsis:
        return _ParamShape((), Ellipsis)
    if isinstance(params, tx.ParamSpec):
        return _ParamShape((), params)
    if isinstance(params, (tx.ParamSpecArgs, tx.ParamSpecKwargs)):
        return _ParamShape((), Ellipsis)
    if any(safe_get_origin(params) is form for form in _CONCATENATE_FORMS):
        # `Concatenate[X1, ..., Xn, tail]`: the fixed prefix is everything but
        # the trailing element, which is a `ParamSpec` (or `...` on 3.10+).
        args = tx.get_args(params)
        return _ParamShape(tuple(args[:-1]), args[-1])
    seq = list(params) if isinstance(params, (list, tuple)) else None
    if seq is None:
        # An unknown shape degrades to the widest (open) list rather than
        # raising -- the opaque rule for a form the relation does not model.
        return _ParamShape((), Ellipsis)
    if seq and isinstance(seq[-1], tx.ParamSpec):
        # Below 3.10 `Concatenate[X, P]` is flattened to `[X, ..., P]`, and a
        # bare `P` is the one-element list `[P]` (typing_extensions >= 4.13).
        return _ParamShape(tuple(seq[:-1]), seq[-1])
    for index, element in enumerate(seq):
        if _is_unpacked_typevartuple(element):
            if index == len(seq) - 1:
                # `Callable[[int, *Ts], R]`: the `*Ts` tail is an open run, so
                # the list rides the same open-tail path a `ParamSpec` does,
                # carrying the `TypeVarTuple` as its tail.
                return _ParamShape(tuple(seq[:-1]), tx.get_args(element)[0])
            # `*Ts` in the *middle* of a list with a fixed suffix
            # (`Callable[[int, *Ts, str], R]`) is out of v1 scope: degrade to
            # an open `Concatenate[int, ...]`-shape, dropping the suffix.
            return _ParamShape(tuple(seq[:index]), Ellipsis)
    if seq and seq[-1] is Ellipsis:
        return _ParamShape(tuple(seq[:-1]), Ellipsis)
    if not seq:
        for parameter in getattr(alias, "__parameters__", ()):
            if isinstance(parameter, tx.ParamSpec):
                # Below 3.10 a bare `P` is erased to `[]`, surviving only in
                # `__parameters__` -- an open list, not a zero-argument one.
                return _ParamShape((), parameter)
    return _ParamShape(tuple(seq), None)


def _match_params(
    sub: _ParamShape, sup: _ParamShape
) -> tx.Optional[_ParamShape]:
    """Match one parameter-list shape against another, per RFC 11.1.

    `sub` describes the callables the left `Callable` accepts, and `sup`
    describes those the right one accepts; the match holds when every
    callable `sub` describes is also one `sup` describes, comparing the
    two committed prefixes contravariantly. A closed `sup` accepts
    exactly its own arity, so `sub` must be closed too, with a matching
    prefix length. An open `sup` accepts its committed prefix followed
    by anything at all, so `sub` only has to supply at least that
    prefix; the variable at `sup`'s tail then captures whatever remains
    of `sub`'s list, meaning its leftover prefix together with its own
    tail.

    The captured tail shape comes back when `sup` is open, a sentinel
    closed shape when `sup` is closed and the match succeeds, and
    `#!python None` when there is no match at all.
    """
    sub_closed = sub.tail is None
    sup_closed = sup.tail is None
    k = len(sub.prefix)
    m = len(sup.prefix)
    if sup_closed:
        if not sub_closed or k != m:
            return None
    elif k < m:
        return None
    # Compare the committed prefix contravariantly: each super-parameter must
    # be a sub-hint of the matching sub-parameter (a function taking `int` can
    # stand in for one taking `bool`).
    for i in range(m):
        if not issubhint(sup.prefix[i], sub.prefix[i]):
            return None
    if sup_closed:
        return _CLOSED_MATCH
    return _ParamShape(tuple(sub.prefix[m:]), sub.tail)


def _issubparams(sub: _ParamShape, sup: _ParamShape) -> bool:
    """Report whether one parameter-list shape stands in for another, per
    RFC 11.1.
    """
    return _match_params(sub, sup) is not None


class _TupleShape(tx.NamedTuple):
    """A tuple's arguments reduced to a shape, per RFC 0001 §11.1 and PEP 646.

    A tuple is described here as a fixed, front-aligned `prefix`, an
    optional open middle run, and a fixed, back-aligned `suffix`. The
    middle run is closed and fixed-arity, as in `Tuple[int, str]`, when
    `rep` is `#!python None`; otherwise `rep` is the element upper bound
    of an open run, `int` for `Tuple[int, ...]` or `Any` for
    `Tuple[int, *Ts]`.

    `var` is the `TypeVarTuple` an `*Ts` run stands for, or
    `#!python None` when there is none. Every fixed position is
    covariant, and an open run sits above every tuple of its shape, the
    way `Tuple[int, ...]` sits above every fixed-length tuple that
    begins with `int`.
    """

    prefix: tx.Tuple[tx.Any, ...]
    rep: tx.Any
    suffix: tx.Tuple[tx.Any, ...]
    var: tx.Any


# The sentinel `_match_tuple` returns when a closed super matches: nothing is
# captured, but the result must be non-`None` to signal the match.
_CLOSED_TUPLE_MATCH = _TupleShape((), None, (), None)


def _tuple_shape(args: tx.Tuple[tx.Any, ...]) -> _TupleShape:
    """Classify a tuple's arguments into a `_TupleShape`.

    `Tuple[()]` on Python 3.8 through 3.10 reports its arguments as the
    phantom `#!python ((),)`, which is normalised here to no elements at
    all, so the empty-tuple type reads the same way on every version. A
    second open run within one tuple is refused, since PEP 646 allows
    only a single unpack and `typing` itself does not enforce that at
    runtime, leaving the relation to enforce it here instead.
    """
    if args == ((),):
        # The 3.8-3.10 `Tuple[()]` phantom: a single empty-tuple element that
        # means "no elements", not a one-element tuple whose element is `()`.
        args = ()
    prefix = []  # type: tx.List[tx.Any]
    suffix = []  # type: tx.List[tx.Any]
    rep = None  # type: tx.Any
    var = None  # type: tx.Any
    open_seen = False
    index = 0
    while index < len(args):
        element = args[index]
        if element is Ellipsis:
            # `Tuple[X, ...]`: the preceding single element is the open run's
            # upper bound.
            if open_seen or not prefix:
                raise TypeError(
                    "a tuple may hold at most one open run of elements "
                    "(`...` or `*Ts`)"
                )
            rep = prefix.pop()
            open_seen = True
            index += 1
            continue
        if _is_unpack(element):
            if open_seen:
                raise TypeError(
                    "a tuple may hold at most one open run of elements "
                    "(`...` or `*Ts`)"
                )
            inner = tx.get_args(element)[0] if tx.get_args(element) else None
            if isinstance(inner, tx.TypeVarTuple):
                rep = tx.Any
                var = inner
                open_seen = True
            elif safe_get_origin(inner) is tuple:
                # `Tuple[int, *Tuple[str, int]]` / `Tuple[*Tuple[str, ...]]`:
                # splice the unpacked tuple's own shape in.
                spliced = _tuple_shape(tx.get_args(inner))
                if spliced.rep is None:
                    # A fixed unpacked tuple flattens into the prefix.
                    prefix.extend(spliced.prefix)
                else:
                    prefix.extend(spliced.prefix)
                    rep = spliced.rep
                    var = spliced.var
                    suffix.extend(spliced.suffix)
                    open_seen = True
            else:
                # An unpacked something the relation does not model: treat its
                # run as `Any`, and report the form once.
                _warn_unknown(element)
                rep = tx.Any
                open_seen = True
            index += 1
            continue
        (suffix if open_seen else prefix).append(element)
        index += 1
    return _TupleShape(tuple(prefix), rep, tuple(suffix), var)


def _match_tuple(
    sub: _TupleShape, sup: _TupleShape
) -> tx.Optional[_TupleShape]:
    """Match one tuple shape against another, comparing fixed positions
    covariantly.

    This is the tuple counterpart of [`_match_params`][]. The result is
    the shape a `*Ts` in `sup` would capture from `sub`, meaning the run
    `sub` supplies beyond `sup`'s fixed prefix and suffix; a sentinel
    closed shape when `sup` is closed and the match succeeds; or
    `#!python None` when there is no match at all. Every fixed position
    is compared covariantly, through `issubhint(sub_i, sup_i)`.
    """
    sub_open = sub.rep is not None
    sup_open = sup.rep is not None
    if not sup_open:
        # A closed super accepts exactly its own arity.
        if sub_open or len(sub.prefix) != len(sup.prefix):
            return None
        for element, superel in zip(sub.prefix, sup.prefix):
            if not issubhint(element, superel):
                return None
        return _CLOSED_TUPLE_MATCH
    p2, r2, q2 = sup.prefix, sup.rep, sup.suffix
    if not sub_open:
        # A closed sub against an open super: it must supply the whole fixed
        # prefix and suffix, and its middle elements must satisfy the run.
        elements = sub.prefix
        if len(elements) < len(p2) + len(q2):
            return None
        for i in range(len(p2)):
            if not issubhint(elements[i], p2[i]):
                return None
        for j in range(1, len(q2) + 1):
            if not issubhint(elements[-j], q2[-j]):
                return None
        middle = elements[len(p2): len(elements) - len(q2)]
        for element in middle:
            if not issubhint(element, r2):
                return None
        return _TupleShape(tuple(middle), None, (), None)
    # Both open: the sub's fixed prefix/suffix must cover the super's, its
    # extra fixed elements must satisfy the super's run, and `r1 <= r2` holds.
    p1, r1, q1 = sub.prefix, sub.rep, sub.suffix
    if len(p1) < len(p2) or len(q1) < len(q2):
        return None
    for i in range(len(p2)):
        if not issubhint(p1[i], p2[i]):
            return None
    for extra in p1[len(p2):]:
        if not issubhint(extra, r2):
            return None
    if not issubhint(r1, r2):
        return None
    for j in range(1, len(q2) + 1):
        if not issubhint(q1[-j], q2[-j]):
            return None
    for extra in q1[: len(q1) - len(q2)]:
        if not issubhint(extra, r2):
            return None
    return _TupleShape(
        tuple(p1[len(p2):]), r1, tuple(q1[: len(q1) - len(q2)]), None
    )


def _issubtupleshape(sub: _TupleShape, sup: _TupleShape) -> bool:
    """Report whether one tuple shape stands in for another, as a sub-hint."""
    return _match_tuple(sub, sup) is not None
