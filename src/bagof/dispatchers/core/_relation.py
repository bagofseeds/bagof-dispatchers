"""The hint-level subtype relation: `issubhint` and `ishintstance`."""

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
    """Whether `hint` is `Any`, in any spelling."""
    return any(hint is form for form in _ANY_FORMS)


def _is_literal(origin: tx.Any) -> bool:
    """Whether `origin` is the `Literal` form, in any spelling."""
    return any(origin is form for form in _LITERAL_FORMS)


def _is_never(hint: tx.Any) -> bool:
    """Whether `hint` is a bottom type (`Never`/`NoReturn`)."""
    return any(hint is form for form in _NEVER_FORMS)


def _is_unpack(hint: tx.Any) -> bool:
    """Whether `hint` is an `Unpack[...]`, in any spelling.

    On 3.11 `typing.Unpack is not typing_extensions.Unpack`, and the star
    syntax `Tuple[int, *Ts]` and `tx.Unpack[Ts]` produce the two different
    spellings -- so the origin is tested against both.
    """
    return any(safe_get_origin(hint) is form for form in _UNPACK_FORMS)


def _is_unpacked_typevartuple(hint: tx.Any) -> bool:
    """Whether `hint` is `Unpack[Ts]` for a `TypeVarTuple` `Ts`."""
    if not _is_unpack(hint):
        return False
    args = tx.get_args(hint)
    return bool(args) and isinstance(args[0], tx.TypeVarTuple)


def _known_form(hint: tx.Any) -> tx.Any:
    """Map a known non-class form to the class it dispatches as.

    `LiteralString` dispatches as [`str`][], and `TypeGuard[...]` /
    `TypeIs[...]` as [`bool`][]. An unpacked `TypeVarTuple` (`*Ts`) is an open
    run of `Any` elements, so on its own -- an `*args: *Ts` tail read as a
    single slot -- it dispatches as [`Any`][typing.Any]. Every other hint is
    returned unchanged.
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
    """The upper bound of a `TypeVar`, for dispatch.

    Its bound, the union of its constraints, or [`Any`][typing.Any] -- and
    **not** its PEP 696 default, which is a static-checker fallback that
    dispatch ignores (so `#!python TypeVar("T", bound=float, default=int)`
    dispatches as `float`).
    """
    constraints = getattr(tv, "__constraints__", ())
    if constraints:
        return tx.Union[constraints]
    bound = getattr(tv, "__bound__", None)
    if bound is not None:
        return bound
    return tx.Any


def _equivalent(a: tx.Any, b: tx.Any) -> bool:
    """Whether two hints accept exactly the same values."""
    return issubhint(a, b) and issubhint(b, a)


_WARNED_UNKNOWN = set()  # type: set


def _warn_key(hint: tx.Any) -> tx.Any:
    """A stable, hashable key identifying an unknown *form*.

    A future form and every hint built from it (`Unpack[Ts]`, `Unpack[Us]`,
    ...) share one origin, so the origin -- or the form itself when it has
    none -- keys the warning, deduping per form rather than per `repr`.
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
    """Warn once per form that `hint` is unrecognised and treated as `Any`."""
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
    """The error text for an object that is not a usable type hint."""
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
    """
    Like isinstance, but the second argument can be a type hint.

    * If `hint` is [`type`][] or [`Type[...]`][tx.Type], checks
      that `obj` is a type and that it is valid subclass of the hint argument.
    * If `hint` is a [`Literal`][tx.Literal], checks that `obj` is one of
      its values. The value must match in type as well: `#!python True` is
      not a valid `#!python Literal[1]`, even though `#!python True == 1`.
    * If `hint` is a [`Union`][tx.Union], checks `obj` against each of
      its members.
    * If `hint` is a [`TypedDict`][tx.TypedDict], checks the *shape* of
      `obj`: it must be a [`dict`][] that holds every required key, and each
      declared key it holds must carry a value of that field's type. Keys
      beyond the declared ones follow the `TypedDict`: an open one (the
      default) allows them, a `closed=True` one rejects them, and an
      `extra_items=` one checks each against that type. A nested `TypedDict`
      or container field is read recursively.
    * If `hint` is a [`runtime_checkable`][typing.runtime_checkable]
      protocol with **data members** (`#!python name: str`), checks the
      value itself, much as [`isinstance`][] does: its class names the
      protocol among its bases, or every data member is present on the
      value -- set on the instance, or declared by its class: an
      annotation, a class attribute, a property or a slot, but not a
      `#!python ClassVar` -- and every method is defined by its class. An
      annotated member counts even on an instance that never set it, where
      [`isinstance`][] finds nothing: that is how a type checker reads the
      annotation, and it keeps this check in step with
      [`issubhint`][bagof.dispatchers.core.issubhint]. A member the protocol
      declares `#!python ClassVar` is read off the class alone, and only a
      `#!python ClassVar` annotation declares it. Members are looked up
      without running the value's code: a property is not called and
      `__getattr__` is not asked. A protocol with methods only is checked
      on the value's type.
    * If `hint` is a parametrised generic (`#!python List[int]`,
      `#!python Box[int]`), checks that `obj` is an instance of its class,
      and checks the type arguments only when `obj` **declares** them: an
      instance built as `#!python Box[int]()`, or an instance of a class
      written against a parametrised base
      (`#!python class IntList(List[int])`). The arguments are then compared
      as [`issubhint`][bagof.dispatchers.core.issubhint] compares them, so
      an invariant position asks for the same type: `#!python Box[int]()` is
      not a `#!python Box[object]`, and an `#!python IntList` is not a
      `#!python List[object]` (it is a `#!python List[Any]`, a
      `#!python list` and a `#!python Sequence[object]`).

    !!! note
        `#!python Box[int]()` writes its record onto the instance only after
        `__init__` returns, so a call dispatched on `self` from inside
        `__init__` sees a value that declares nothing yet. An instance that
        cannot hold the record -- a class with `__slots__` and no
        `__dict__`, or a frozen dataclass -- never declares its arguments.
    * Otherwise, returns `#!python  issubhint(type(obj), hint)`.

    !!! warning
        A container's **item types are not checked**. A plain
        `#!python [1, 2]` declares no type arguments, so it is a valid
        `#!python List[str]` as far as this function is concerned. (Python
        itself refuses `#!python isinstance(x, list[int])` for the same
        reason.) Checking the items means iterating them, which is the
        caller's decision to make - `bagof.validators` does it.

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
    """Type-aware equality of two Literal values (PEP 586).

    ``1 == True`` and ``1 == 1.0`` are True in Python but denote different
    literal values, so the types must match too. ``eq_safenan`` keeps a NaN
    literal equal to itself.
    """
    return type(a) is type(b) and eq_safenan(a) == eq_safenan(b)


def _ishintstance_literal(obj: tx.Any, hint: tx.Any) -> bool:
    """Check that a value is one of a `Literal`'s values."""
    # Both the type and the value must match. Python compares `True == 1`
    # and `1 == 1.0` as equal, but PEP 586 makes literal matching
    # type-aware, so `Literal[1]` must reject `True` and `1.0`.
    return any(_literal_value_eq(arg, obj) for arg in get_args_uw(hint))


def _ishintstance_type(obj: tx.Any, hint: tx.Any) -> bool:
    """Like isinstance, but the second argument can be a type hint."""
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
    """A runtime-checkable protocol's members, split by where they are read.

    `data` are its instance variables, read off the value (its instance
    `__dict__`, its class, or its class's annotations). `methods` and
    `class_variables` -- the data members it declares `ClassVar` -- are read
    off the value's class. All are sorted, so a caller that keys on them
    always reads them in the same order.
    """

    data: tx.Tuple[str, ...]
    methods: tx.Tuple[str, ...]
    class_variables: tx.Tuple[str, ...]


# A marker for "no such attribute", distinct from any value a class can hold
# (`None` included, which means something to a protocol method).
_ABSENT = object()


def _data_protocol_members(cls: tx.Any) -> tx.Optional[_ProtocolMembers]:
    """The members of `cls` if it is a runtime protocol with data members.

    Returns `#!python None` for anything else: a class that is not a
    protocol (a concrete class that merely inherits one included), a protocol
    that is not [`runtime_checkable`][typing.runtime_checkable] -- Python
    refuses to instance-check it, and so does the relation -- and a protocol
    whose members are all methods, which Python decides from the class and
    the relation keeps reading by type.
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
    """[`_data_protocol_members`][] for a runtime-checkable class, memoised.

    A member is a method when the protocol holds a callable under its name,
    and a data member otherwise -- an annotation alone, a property, or a
    plain default -- the same split Python makes. A data member is a class
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
    """What `cls` defines under `name`, anywhere in its MRO, or `_ABSENT`.

    Read out of each class's own namespace, so a property is found, not
    called, and nothing of the class's own code runs.
    """
    for base in cls.__mro__:
        namespace = base.__dict__
        if name in namespace:
            return namespace[name]
    return _ABSENT


def _has_method(cls: type, name: str) -> bool:
    """Whether `cls` defines the protocol method `name`.

    A class that sets the name to `#!python None` declares that it does
    *not* support it, as Python reads a protocol method too.
    """
    found = _class_attribute(cls, name)
    return found is not _ABSENT and found is not None


def _present_data_members(
    obj: tx.Any, names: tx.Sequence[str]
) -> tx.Tuple[bool, ...]:
    """Whether the value `obj` has each instance variable in `names`, in order.

    A name is present when it is in the instance's own `__dict__`, or when
    the value's class declares it an instance variable anywhere in its MRO
    ([`_class_reading`][]): an annotation (`#!python name: str`), or a class
    attribute, a property, a slot or a method that the same class does not
    annotate `ClassVar`. Nothing of the value's code runs: a property is
    found, not called, and `__getattr__` is never asked, as Python's own
    `isinstance` reads a protocol member from 3.12 on (earlier versions of
    `typing` call `hasattr`). Two differences from `isinstance`, both the
    type checker's reading and the one the hint level
    ([`_declares_protocol`][]) takes, so that the two agree: an annotated
    name is present on an instance that never set it, and a `ClassVar` is
    not an instance variable, so a class attribute declared one is not
    present.

    This is the one reader both the value check and the call cache use, so
    the key always covers what the check reads. It runs on every call the
    cache answers at such an argument, so what depends on the class alone
    is worked out once per class ([`_class_reading`][]): how the value's
    attributes are looked up -- an ordinary instance reads its own
    `__dict__`, a class read as a value takes [`inspect.getattr_static`][],
    and any other is read through its class alone -- and which names the
    class declares. A declared name is present on every instance, so its
    entry in the key is the same for all of them, and reading it costs a
    set lookup.
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
    """Whether `cls` declares every `ClassVar` member in `names`.

    A class variable lives on the class, so it is read off the class alone,
    at the value level and the hint level alike: a `ClassVar` annotation
    anywhere in the MRO, with a value or not. A plain class attribute, an
    instance variable's annotation, and an attribute an instance sets on
    itself do not count -- a type checker rejects each of them for a
    `ClassVar` member.
    """
    declared = _class_reading(cls).class_variables
    return all(name in declared for name in names)


# The namespace of an instance that has none of its own.
_NO_ATTRIBUTES = frozenset()  # type: tx.FrozenSet[str]


def _found_statically(obj: tx.Any, name: str) -> bool:
    """Whether [`inspect.getattr_static`][] finds `name` on `obj`."""
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
    """What reading an instance of a class needs, worked out once per class.

    `lookup` says how its attributes are looked up (`_PLAIN`, `_STATIC` or
    `_CLASS_ONLY`); `instance_variables` and `class_variables` are the names
    the class declares, anywhere in its MRO, as each kind.
    """

    lookup: int
    instance_variables: tx.FrozenSet[str]
    class_variables: tx.FrozenSet[str]


# `_class_reading`'s memo. Its keys are the types of the values dispatched
# on, so they are held weakly: a class made and dropped at runtime is not
# kept alive for the life of the process.
_CLASS_READINGS = weakref.WeakKeyDictionary()  # type: weakref.WeakKeyDictionary


def _class_reading(cls: type) -> _ClassReading:
    """How [`_present_data_members`][] reads an instance of `cls`, memoised.

    Its lookup is `_PLAIN` for an ordinary class
    ([`_reads_instance_dict`][]); `_STATIC` -- `inspect.getattr_static` --
    for a metaclass, whose instances are classes holding attributes of
    their own in their bases and their metaclass; and `_CLASS_ONLY` for a
    class that redefines `__dict__`, whose instances' own attributes cannot
    be read without running its code.

    Its declared names are read from each class of its MRO
    ([`_read_class`][]): a name counts when *any* of them declares it, so a
    subclass declares everything its bases do, and an instance of a class
    below a protocol is always an instance of the protocol too. The memo is
    read once per class: an attribute or an annotation added to a class
    after it was first dispatched on is not seen.

    A class that cannot key the memo -- one whose metaclass makes it
    unhashable -- is worked out afresh on each call. It cannot key
    `getattr_static`'s own memo either, which raises `TypeError` for an
    instance of it from 3.13 on (as Python's own `isinstance` does against a
    protocol, on every version), so an instance of such a metaclass is read
    through what its class declares alone, `_CLASS_ONLY`, too.
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
    """[`_class_reading`][], unmemoised.

    `metaclass_lookup` is the lookup for a metaclass. Each class of the MRO
    declares, as a type checker reads it:

    * an instance variable for each of its annotations that is not a
      `ClassVar` (nor an `InitVar` or `KW_ONLY`, which declare nothing), and
      for each name its namespace holds -- a class attribute, a property, a
      slot, a method -- that it does not annotate: a value in the class
      body is an instance variable's default;
    * a class variable for each `ClassVar` annotation, with a value or not.

    The two are gathered over the whole MRO, never decided class by class:
    a subclass that redeclares an inherited instance variable `ClassVar`
    keeps it (a type checker rejects the override), so a class is always
    below whatever its bases are below.
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
    """The annotations written in the body of `cls` itself, by name.

    Not its bases': the caller walks the MRO. Read from the class's own
    namespace up to 3.13; from 3.14, where annotations are evaluated lazily,
    through `annotationlib` -- a name that is not defined comes back as a
    forward reference rather than raising, and an annotation that cannot be
    evaluated at all reads as none.
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
    """What each annotation in the body of `cls` declares, by name.

    A `TypedDict`'s annotations declare its keys, not attributes: an
    instance is a plain `dict`, so they declare nothing here.
    """
    if is_typeddict(cls):
        return {}
    return {
        name: _annotation_kind(annotation, cls)
        for name, annotation in _own_annotations(cls).items()
    }


def _annotation_kind(annotation: tx.Any, cls: tx.Any = None) -> int:
    """What one annotation, written in the body of `cls`, declares: an
    instance or a class variable, or no attribute.

    `Annotated[...]` is looked through. An annotation that is still text --
    a string, or a forward reference -- is read by the name it starts with
    ([`_text_annotation_kind`][]).
    """
    if isinstance(annotation, tx.ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        return _text_annotation_kind(annotation, cls)
    annotation = unwrap(annotation)
    return _marker_kind(safe_get_origin(annotation) or annotation)


def _marker_kind(marker: tx.Any) -> int:
    """What an annotation whose outermost form is `marker` declares."""
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
    """[`_annotation_kind`][] for an annotation written as text in `cls`.

    The dotted name the text starts with is looked up where the evaluated
    annotation would find it ([`_look_up_marker`][]) and recognised by
    identity; `Annotated[...]` is looked through. A name that cannot be
    looked up is read by its last part instead -- `typing.ClassVar[int]`
    and `ClassVar[int]` alike. The text is never evaluated.
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
    """What the dotted name `parts`, written in the body of `cls`, names.

    The first part is looked up in the class's own namespace, then its
    module's, then the builtins -- where evaluating the annotation would
    look -- and each further part in the module the name so far names.
    Only dictionaries are read: no attribute is fetched, so nothing of the
    user's code runs. `_ABSENT` when a part is not found, or when a name
    before the last one is not a module.
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
    """Whether an instance of `cls` has its attributes looked up plainly.

    True for an ordinary class: an instance's attributes are its own
    `__dict__` -- read through the standard descriptor, which runs no code
    -- and its class's MRO. False for a metaclass, whose instances are
    classes (their attributes are their bases' and their own metaclass's),
    and for a class that redefines `__dict__` (a property, say), which a
    plain read would run.
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
    """Whether `obj` is an instance of the data-member protocol `proto`.

    A class that names `proto` among its bases is one, as Python's
    `isinstance` counts it, whatever its instances hold. Otherwise every
    method must be defined by the value's class, every `ClassVar` member be
    declared by it ([`_holds_class_variables`][]), and every instance
    variable be present on the value ([`_present_data_members`][]).

    Methods are read off the class, as a method-only protocol is decided,
    so only the instance variables depend on the instance -- and those are
    all the call cache keys on. A method set on the instance alone is
    therefore not counted, where Python's `isinstance` would count it, and
    neither is a `ClassVar` member, which a type checker rejects too.
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
    """Whether the class `cls` is a structural subtype of the data protocol
    `proto`, much as a type checker reads it.

    The hint-level twin of [`_ishintstance_protocol`][]:

    * a class that names `proto` among its bases is below it, a sub-protocol
      included, as it is at the value level;
    * another protocol that does not is not, even when it lists the same
      members -- a protocol stands for what it says it extends;
    * any other class must define every method (not as `#!python None`),
      and declare every data member as its kind anywhere in its MRO
      ([`_read_class`][]): an instance variable by an annotation
      (`#!python name: str`, a dataclass field included, whether or not
      `__init__` sets it) or by a class attribute, a property or a slot not
      annotated `ClassVar`; a class variable by a `ClassVar` annotation
      alone.

    An annotation that is never set promises nothing at runtime, so the
    value level counts it as present too ([`_present_data_members`][]):
    every instance of `cls` is then an instance of `proto`, which is the
    one thing the order must guarantee. Both levels read the same
    per-class record, gathered over the whole MRO, so a subclass declares
    what its bases do.

    Unlike a type checker, a read-only member -- a property without a
    setter, or `#!python name: Final = ...` -- counts for a protocol's
    (settable) instance variable: the value has it, and dispatch only
    reads it.
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
    """Whether the data protocol `sub` is below the method-only protocol `sup`.

    `sup` is runtime-checkable, and `members` are `sub`'s. A protocol that
    names `sup` among its bases is below it. So is one whose methods cover
    every member of `sup` -- read the way the value level reads them, off
    the class, so the class of an instance of `sub` passes Python's check
    for `sup` too. A data member of `sub` never stands in for a method of
    `sup` (Python's own `issubclass` would let an annotation do so): it may
    be set on the instance alone, and then the instance's class does not
    satisfy `sup`.
    """
    if any(base is sup for base in sub.__mro__):
        return True
    return set(tx.get_protocol_members(sup)) <= set(members.methods)


# A unique marker for "no such attribute". Distinct from any value a
# `TypedDict` could carry, so an identity check never confuses it with one.
_NO_EXTRA_ITEMS = object()


def _typeddict_extra_policy(td: tx.Any) -> tx.Tuple[str, tx.Any]:
    """How a `TypedDict` treats keys beyond the ones it declares.

    Returns one of:

    * `("open", None)` -- extra keys are allowed (the default).
    * `("closed", None)` -- extra keys are rejected.
    * `("typed", hint)` -- extra keys are allowed, but each such key's value
      must satisfy `hint`.

    A `TypedDict` written `closed=True` reports `"closed"`; one written
    `extra_items=SomeType` reports `("typed", SomeType)`. Closedness is
    **inherited**: a subclass of a closed (or `extra_items=`) `TypedDict` is
    itself closed/typed even when it does not repeat the keyword, so the base
    chain is walked nearest-first and the first class that states a policy
    wins. A class built by an older `typing_extensions` that cannot express
    either -- so whose closedness cannot be read -- is reported `"open"`, the
    permissive default, so the check never fails on it.
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
    """Whether an `extra_items` / field hint cannot be judged here.

    An unresolvable forward reference -- a bare string or a
    [`ForwardRef`][typing.ForwardRef] -- carries no type to compare, and a
    missing hint is `#!python None`. In either case the well-formedness of a
    key or policy against it is left unjudged rather than guessed at, exactly
    as the value-level shape check skips such a field.
    """
    return hint is None or isinstance(hint, (str, tx.ForwardRef))


def _is_any_hint(hint: tx.Any) -> bool:
    """Whether a field / `extra_items` hint is `Any`, through its qualifiers.

    Looks through the transparent qualifiers (`Required` / `NotRequired` /
    `ReadOnly` / `Final` / `ClassVar`, via `normalise_hint`) and any
    [`Annotated`][typing.Annotated] wrapper, then asks `_is_any`. An
    `Any`-typed key or `extra_items` accepts every value, so it can never break
    a base's contract and is treated as compatible.
    """
    return _is_any(unwrap(normalise_hint(hint), tx.Annotated))


# PEP 484 §Numeric numeric tower: `bool` promotes to `int`, `int` to `float`,
# `float` to `complex`. A type checker treats a lower type as assignable to a
# wider one, and `_numeric_consistent` reads a hint to its rank here.
_NUMERIC_RANK = {bool: 0, int: 1, float: 2, complex: 3}


def _numeric_consistent(sub: tx.Any, sup: tx.Any) -> bool:
    """Whether `sub` promotes to `sup` under the PEP 484 numeric tower.

    `bool`, `int`, `float` and `complex` form a promotion chain, and a type
    checker treats each as assignable to any wider one (`int` where `float` is
    expected). Both hints are read to their bare builtin type (through
    `normalise_hint` and any `Annotated` wrapper); the result is `#!python
    True` only when both are numeric and `sub` sits at or below `sup` in that
    chain. `bool` -> `int` is a real subclass and already consistent; listing
    it keeps the check self-contained.
    """
    sub = unwrap(normalise_hint(sub), tx.Annotated)
    sup = unwrap(normalise_hint(sup), tx.Annotated)
    return (
        sub in _NUMERIC_RANK
        and sup in _NUMERIC_RANK
        and _NUMERIC_RANK[sub] <= _NUMERIC_RANK[sup]
    )


def _own_extra_policy(cls: tx.Any) -> tx.Tuple[str, tx.Any]:
    """The policy a class declares *itself*, ignoring what it inherits.

    A `TypedDict`'s base is not in its MRO, and `__closed__` and
    `__extra_items__` are set from the class's own keyword alone, so a plain
    attribute read gives the class's own declaration. Returns one of:

    * `("typed", hint)` -- the class wrote `extra_items=hint`;
    * `("closed", None)` -- the class wrote `closed=True`;
    * `("reopened", None)` -- the class wrote `closed=False`;
    * `("none", None)` -- the class declared no policy of its own.
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
    """The direct `TypedDict` bases of a class, by origin, skipping the marker.

    A parametrised base (`Base[int]`) is read through its origin (`Base`), the
    same way [`_typeddict_extra_policy`][] reaches a generic base.
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
    """The `("open"/"closed"/"typed", hint, base)` policy `cls` inherits.

    The effective policy of `cls`'s nearest constrained base, together with
    that base -- so a violation can name where the contract came from. `cls`'s
    own declaration is not read here.
    """
    for base in _typeddict_bases(cls):
        policy, hint = _typeddict_extra_policy(base)
        if policy != "open":
            return policy, hint, base
    return "open", None, None


def _malformed_class_reason(cls: tx.Any) -> tx.Optional[str]:
    """Why `cls` violates a constrained base's PEP 728 contract, or `None`.

    Reads `cls`'s own declaration and the policy it inherits from its nearest
    closed / `extra_items` base, and reports the first key or policy that would
    let a value of `cls` carry something that base refuses -- which the nominal
    hint order would still call a sub-hint, so it would mis-dispatch. A closed
    base is read as `extra_items=Never` (it admits no extra key), so adding a
    key to it and adding an `extra_items`-incompatible key are the one check.

    Every key `cls` carries -- from its own body **or any base**, not only the
    keys it declares itself -- is checked against the constrained base: a key
    the base does not declare, whose value type it does not admit, is a
    dispatch-unsound diamond (a sibling open base contributing a key a closed
    base would reject). An `Any`-typed key or `extra_items` accepts every
    value, so it can never break the contract and is treated as compatible.
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
    """Every `TypedDict` class in `td`'s inheritance, `td` first, by origin.

    Origins are resolved (`Base[int]` -> `Base`) and each class is visited
    once, so a diamond is not walked twice.
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
    """Why a `TypedDict` `td` would break dispatch soundness, or `None`.

    This rejects a **dispatch-sound subset**, not everything a type checker
    forbids: it names a `TypedDict` whose nominal hint order (`issubhint` /
    `ishintstance` are nominal on a concrete `TypedDict`) would disagree with
    the value-level shape check, breaking `v in Sub and Sub <= Base => v in
    Base`. Such a class is refused at registration rather than left to
    mis-dispatch. The shapes named are: a key added to a closed base, a key
    whose value type a base's `extra_items` does not admit (including a key
    from a sibling open base in a diamond), a widened `extra_items`, and a
    reopened closed base (`extra_items=` or `closed=False`).

    For the added-key / `extra_items` compatibility check, two cases are read
    as consistent to match a type checker rather than resting purely on this
    library's nominal relation:

    * **top-level `Any` is accepted** -- an `Any`-typed key or `extra_items`
      admits every value, so it can never carry something a base refuses.
    * **the numeric tower is accepted** -- a key typed `int` under
      `extra_items=float` (or `int` / `float` under `complex`) is consistent,
      because a type checker promotes it under PEP 484 §Numeric and rejecting
      it would be surprising.

    The deeper gradual-consistency cases a type checker also accepts stay
    strict here, as a documented residual: nested `Any` (`List[Any]` under
    `List[int]`), a `Callable[..., R]` argument list, and non-runtime
    protocols are all judged by the nominal relation.

    A well-formed `TypedDict` -- a plain subclass of a closed base, one that
    narrows `extra_items`, or a subclass of an open base -- returns `#!python
    None`. So does anything that is not a concrete `TypedDict`, and any class
    built by a `typing_extensions` too old to record closedness (its policy
    cannot be read, so nothing can be judged malformed).
    """
    if not is_typeddict(td) or is_typeddict_marker(td):
        return None
    for cls in _typeddict_chain(td):
        reason = _malformed_class_reason(cls)
        if reason is not None:
            return reason
    return None


def _ishintstance_typeddict(obj: tx.Any, td: tx.Any) -> bool:
    """Check that a value has the shape a `TypedDict` describes.

    A value matches when it is a [`dict`][], holds every **required** key, and
    every declared key it *does* hold carries a value that satisfies that
    field's hint (checked through `ishintstance`, so a nested `TypedDict` or a
    container field is read the same way as any other value). `Required` /
    `NotRequired` and the class's `total=` are honoured through
    `typeddict_required_keys`.

    Only a `dict` is accepted, not any [`Mapping`][collections.abc.Mapping].
    This keeps the value level in step with the hint level, where
    `TypedDict <= dict`: since every `TypedDict`-shaped value must also be a
    valid `dict`, a non-`dict` mapping that matched the shape but is not a
    `dict` would break `v in S and S <= T => v in T`.

    Keys **beyond** the declared ones are treated as the `TypedDict` itself
    says (PEP 728):

    * An **open** `TypedDict` -- the default -- allows extra keys: a `dict`
      with keys beyond the declared ones still matches, so long as the
      declared keys check out. Such a `TypedDict` names a minimum shape, not a
      closed one (as `pydantic`'s `TypeAdapter` reads it too).
    * A **closed** `TypedDict`, written `closed=True`, rejects any extra key:
      a value carrying a key it does not declare does not match.
    * A `TypedDict` written `extra_items=SomeType` allows extra keys but
      checks each one's value against `SomeType` (through `ishintstance`, the
      same way a declared field is read). `extra_items=Never` therefore admits
      no extra key at all, since no value satisfies `Never`.

    Two *unrelated* `TypedDict`s that happen to share a satisfiable shape are
    not ordered by this check, so a value matching both dispatches to neither
    on its own: selection raises `AmbiguousMethodError`, the same outcome two
    equally-matched `Protocol`s give (RFC 0001 §5).
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
    """
    Check that a hint is a sub-hint for another hint.

    A hint is a valid subhint if all values that are valid for the hint
    are also valid for the superhint.

    Each argument position is compared by the **variance the generic
    declares for it** (PEP 484): a covariant position (a
    `#!python Sequence`'s item) forwards the relation, so
    `#!python Sequence[bool]` is a sub-hint of `#!python Sequence[int]`; a
    contravariant one reverses it; an invariant one -- a mutable container
    like `#!python list`, or an unflagged user `#!python TypeVar` -- demands
    that the two arguments accept the same values, so `#!python List[bool]`
    is **not** a sub-hint of `#!python List[int]`. A free `#!python TypeVar`
    or `#!python Any` on the super side is a top an invariant position may
    still widen to. A hint with no arguments is *not* a subhint of one that
    has them - a bare `#!python list` may hold anything, so it cannot stand
    in for a `#!python List[int]`.

    !!! note
        An **unparametrised** `#!python Union` or `#!python Literal` asks
        a different question: *is this hint one of those?* So
        `#!python issubhint(int, Union)` is `#!python False` (an
        `#!python int` is not a union) even though
        `#!python issubhint(int, Union[int, str])` is `#!python True`.
        This makes them usable as a `#!python BOUND`, and it is why the
        relation is not transitive through a bare `#!python Union`.

    !!! note
        A class is a sub-hint of a
        [`runtime_checkable`][typing.runtime_checkable] protocol with data
        members (`#!python name: str`) when it **declares** them, much as
        a type checker reads it: the class names the protocol among its
        bases, or declares each member as the member's kind. An ordinary
        member is declared by an annotation such as `#!python name: str`
        (a dataclass field included) or by a class attribute, a property or
        a slot -- but not by a `#!python ClassVar`. A member the protocol
        declares `#!python ClassVar` is declared by a `#!python ClassVar`
        annotation alone. An annotation counts whether or not the attribute
        is ever set, and
        [`ishintstance`][bagof.dispatchers.core.ishintstance] counts it on
        every instance too. Unlike a type checker, a read-only member (a
        property without a setter, a `#!python Final`) counts for an
        ordinary one.

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
    """Check that a hint is a sub-hint for a hint whose origin is a class."""
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
    """Whether a class hint's arguments fit a super-hint's, slot by slot.

    `variances` is the super-hint's origin's, from [`_generic_variances`][].
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
    """Whether the class `sub` is below the class `origin`, for the relation.

    [`safe_issubclass`][], except where a runtime-checkable protocol with
    data members is involved, which `issubclass` refuses (as the super side)
    or reads too loosely (as the sub side) -- see [`_declares_protocol`][]
    and [`_protocol_below_protocol`][].
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
    """Whether a sub-side argument fits a super-side one at one slot (#50).

    `variance` is a generic parameter position's declared variance (the string
    constants from [`_introspect`][], read off the super-hint's origin). Per
    PEP 484:

    * **covariant** -- `sub` must be a sub-hint of `sup`, so a covariant
      container narrows with its item (`Sequence[bool] <= Sequence[int]`). A
      `TypeVar` is read as its bound (or its constraints), which is already
      what solving it would give: `G[A] <= G[T]` for some `T <= B` exactly
      when `A <= B`.
    * **contravariant** -- `sup` must be a sub-hint of `sub`, so a consumer of
      `int` stands in for a consumer of `bool`. A `TypeVar` is read as its
      bound here too; see [`_issubslot_invariant`][] for why it is not solved.
    * **invariant** -- see [`_issubslot_invariant`][]: the two must accept the
      same values, or the super side is a `TypeVar` that can be solved to the
      sub side, or a top (`Any`, a free `TypeVar`).
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
    """Whether `G[sub] <= G[sup]` at an invariant slot (#50, V5).

    A `TypeVar` on the super side stands for *some* type within its bound or
    constraints, as a type checker solves it; on the sub side it stands for a
    whole family, which a single type cannot contain:

    * `sup` is `Any` or a free `TypeVar` -- the top the slot may widen to
      (gradual consistency), so a generic-fallback overload stays above every
      specialisation;
    * `sup` is a bounded `TypeVar` `T <= B` -- `T` can be solved to `sub`
      exactly when `sub <= B` (a `TypeVar` `sub` read by its own bound), so
      `Box[bool] <= Box[T <= int]`, and `Box[T1 <= B1] <= Box[T2 <= B2]` iff
      `B1 <= B2`;
    * `sup` is a constrained `TypeVar` -- `T` is solved to one constraint, so
      `sub` must be equivalent to one of them (a constrained `sub`: each of its
      constraints to one of them);
    * otherwise `sup` is a concrete type, and `sub` must be equivalent to it.
      A `TypeVar` `sub` never is: `Box[T <= int]` is not a `Box[int]`, since
      `T` may be `bool`.

    This is transitive: each rule reduces to `<=` or to equivalence against
    the super side's bound or constraints, which chain.
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
    """The bases `cls` was written with, parametrised where they were.

    `__orig_bases__` is read off the class's *own* namespace: an attribute
    read would find a parent's, which describes the parent's bases, not these.
    A class written without a parametrised base has none of its own, and its
    plain `__bases__` are the answer.
    """
    return _own_orig_bases(cls) or cls.__bases__


def _free_parameters(node: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """The type variables a generic class or alias takes (else `()`).

    A class is read through [`_class_parameters`][], so one written against
    a PEP 585 base (`#!python class GL(list[T])`) takes the variables that
    base mentions; an alias (`#!python Box[T]`, `#!python list[T]`) lists
    its own.
    """
    if _looks_like_class(node):
        return _class_parameters(node)
    params = getattr(node, "__parameters__", ())
    return params if isinstance(params, tuple) else ()


def _filled_bases(
    node: tx.Any, cls: type
) -> tx.Optional[tx.Tuple[tx.Any, ...]]:
    """`cls`'s written bases, with `node`'s arguments filled in.

    `node` is `cls` itself or a parametrisation of it (`Sub[bool]`). Each base
    that mentions one of `cls`'s type variables is subscripted with what
    `node` gives that variable -- typing's own substitution, `Box[T][bool]` is
    `Box[bool]`, the mechanism that resolves a generic type alias too -- and
    one that mentions none (`Box[int]`, a plain class) is kept as written.
    The arguments are paired by variable, not by position, because a class
    may list its variables in another order than a base does (`class
    Flip(Pair[B, A], Generic[A, B])`).

    Returns `#!python None` when `node` leaves the bases undetermined: a
    generic class written without arguments (a bare `Sub`), or a
    `ParamSpec` / `TypeVarTuple` generic, whose arguments do not pair one to
    one with its variables. A base whose substitution raises is dropped.
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
    """`hint` re-expressed as parametrisations of `target`: their arguments.

    `hint` is a class or a parametrised generic whose origin is a subclass of
    `target`. Each answer is what `hint` fills `target`'s parameters with,
    read through the bases each class was written with (`__orig_bases__`),
    filling in each class's own arguments as it goes:

    * `class IntBox(Box[int])` is `Box[int]`, so `IntBox` gives `(int,)`;
    * `class Sub(Box[T])` passes its argument on, so `Sub[bool]` gives
      `(bool,)`, and a subclass written without a parametrised base (`class
      Leaf(IntBox)`) is followed through its plain bases;
    * every base is followed, so a class that reaches `target` along two
      paths gives both: in a diamond `class D(A, B)` with `class A(Box[int])`
      and `class B(Box[str])`, `D` gives `(int,)` and then `(str,)`, and so
      does `class Two(List[T], Container[U])`, whose `Two[int, str]` is a
      `Container[int]` through `List` and a `Container[str]` through its own
      base. A type checker rejects such a class, and the relation accepts a
      hint that any of them satisfies;
    * a standard-library class reached on the way (`class Child(List[int])`
      reaches `List[int]`) is read positionally against a standard-library
      `target` it subclasses (`Sequence`), as two such origins always are.

    `hint`'s own arguments are the answer when its origin *is* `target` and
    it has any. Nothing is yielded when nothing maps -- no base reaches
    `target` with arguments, as for a `collections.Counter` (a runtime
    subclass of `dict` that records no parametrised base), or a generic class
    written without arguments -- and the caller then keeps its positional
    comparison.

    A generator, walking depth first with each class's bases in the order
    they are listed, so a caller that needs one answer stops the walk there
    and a class whose first base maps gives it after as many steps as it is
    deep. Each node -- a class with the arguments it was reached with -- is
    walked once, and each answer given once ([`_first_time`][]); a hierarchy
    that reaches `target` along many paths with other arguments on each (a
    diamond at every level) still has as many answers as paths, and a caller
    that needs all of them walks them all.
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
    """Record `parts` in `met` by identity; whether they were new there.

    The key is the `id` of each part, so no argument's own `__eq__` or
    `__hash__` runs, and two equal objects built apart (`list[int]` twice)
    count as different: the node is then read a second time, which gives
    the same answers again. `holder` -- the node, or the answer -- is stored
    as the entry's value, so every object whose `id` is in a key stays alive,
    and keeps its `id`, for as long as the walk runs.
    """
    key = tuple(map(id, parts))
    if key in met:
        return False
    met[key] = holder
    return True


def _is_fully_declared(args: tx.Optional[tx.Sequence[tx.Any]]) -> bool:
    """Whether declared arguments say what each parameter holds.

    `#!python False` for no arguments at all, and for arguments that mention a
    type variable (`class Child(List[T])` leaves `T` open),
    [`Any`][typing.Any], or a name not yet resolved (`Box["int"]()` records
    `Box[ForwardRef('int')]`) anywhere inside them: each leaves what the value
    holds undeclared, so none may narrow which parametrisations it matches.
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
    """Whether an instance of `cls` is asked for `__orig_class__`, memoised.

    Calling a subscripted generic class records the subscription on the
    instance it builds, and two classes can be subscripted that way: a
    [`Generic`][typing.Generic] subclass (`#!python Box[int]()`), and a
    class written against a PEP 585 alias (`#!python class GL(list[T])`,
    whose `#!python GL[int]()` is recorded by the runtime alias type), which
    has no `Generic` in its MRO. An instance of any other class -- a builtin
    container, a lazy proxy whose `__getattr__` does work -- is never probed.

    The value check ([`_declared_parametrisation`][]) and the call cache's
    key (`_declared_key`) both ask this, so the key always covers what the
    check reads.
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
    """Drop `_may_record_parametrisation`'s answer for a collected class."""
    _RECORDERS.pop(key, None)
    _RECORDER_REFS.pop(key, None)


def _records_parametrisation(cls: type) -> bool:
    """[`_may_record_parametrisation`][], worked out."""
    if issubclass(cls, tx.Generic):
        return True
    return any(
        _is_pep585_alias(base)
        for each in cls.__mro__
        for base in _own_orig_bases(each)
    )


def _orig_class(obj: tx.Any) -> tx.Any:
    """The parametrisation `obj` was built from, or `#!python None`.

    Calling a subscripted user generic -- `Box[int]()`, or `GL[int]()` for
    `class GL(list[T])` -- records the subscription on the new instance as
    `__orig_class__`. It is absent from a builtin container, from any
    instance built by calling the bare class, and from one typing cannot
    write it onto: a class built with `__slots__` and no `__dict__`, and a
    frozen dataclass (typing swallows the `FrozenInstanceError`). It is also
    written only *after* `__init__` returns, so a dispatch on `self` from
    inside `__init__` sees an instance that declares nothing yet. Whatever
    the attribute holds is only trusted when it is a parametrisation of a
    class `obj` is an instance of.
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
    """What `obj` declares itself to be, as a parametrisation of `origin`.

    `origin` is the class a parametrised hint (`G[args]`) is written on, and
    `type(obj)` is already known to be a subclass of it. The answer is a hint
    to compare with `G[args]` through the relation. Returns, in order:

    1. the instance's `__orig_class__` (`Box[int]` for `Box[int]()`), when
       `obj` is an instance of a `Generic` subclass or of a class written
       against a PEP 585 alias (`GL[int]()` for `class GL(list[T])`,
       [`_may_record_parametrisation`][]) and the record declares
       every argument of `origin` -- against a user generic or a
       standard-library one alike (`Row[int]()` for `class Row(Sequence[T])`
       is a `Sequence[int]`) -- along every base that reaches it
       ([`_declares_arguments`][]);
    2. else `type(obj)`, when the class declares every argument of `origin`
       along every base that reaches it (`class Child(List[int])`);
    3. else `#!python None`: the value declares nothing, and only its origin
       can be checked.

    Only an origin with one readable argument per parameter is read
    ([`_reads_declared_arguments`][]): `Tuple` and `Callable` (whose argument
    lists are shapes), a `ParamSpec` / `TypeVarTuple` generic (`Hook[[int]]`),
    and any origin whose parameters cannot be read keep the shallow check.
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
    """Whether `hint` says what `origin`'s parameters hold, through its bases.

    Every parametrisation of `origin` the bases reach ([`_as_base_args`][])
    has to be fully declared. The hint is then compared with the super-hint
    through the relation, which accepts it when any of them fits. One that
    leaves an argument open (`class Two(List[T], Container[U])` built as
    `Two[Any, str]()` reaches `Container[Any]` through `List`) makes the whole
    value undeclared, so only its origin is checked: reading it by its other
    base alone would reject it as a `Container[bytes]` while it is accepted as
    a `Collection[bytes]`, which is below that.
    """
    declared = False
    for args in _as_base_args(hint, origin):
        if not _is_fully_declared(args):
            return False
        declared = True
    return declared


def _is_subscripted_tuple(hint: tx.Any) -> bool:
    """Whether a tuple hint is subscripted (`Tuple[int]`, `Tuple[()]`).

    Told apart from a bare, unparametrised `Tuple`/`tuple`. The empty-tuple
    type `Tuple[()]` reports its arguments as the phantom `#!python ((),)` on
    Python 3.8-3.10 and as genuinely empty `#!python ()` on 3.11+, so an empty
    argument list alone cannot distinguish it from a bare `Tuple`. A
    subscripted alias with no arguments still carries an `__args__` attribute
    where a bare form does not -- the load-bearing fallback for a genuinely
    empty argument list, which includes the PEP 585 `#!python tuple[()]`
    spelling on 3.9-3.10 as well as every empty-args form on 3.11+.
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
    """Whether a tuple hint is a sub-hint of a tuple superhint."""
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
    """Check a hint's arguments against a superhint's, covariantly.

    The covariant fallback for a generic with no readable per-position
    variance (a `ParamSpec`/`TypeVarTuple` origin, or an arity mismatch); an
    origin whose variance is known is compared slot by slot (each by its
    declared variance) in [`_issubclasshint`][] instead.

    The arguments are read as tuple *shapes* (a fixed prefix, an optional open
    run, a fixed suffix) so a trailing ellipsis (`Tuple[int, ...]`) and an
    unpacked `TypeVarTuple` (`Tuple[int, *Ts]`) are ordered the same way. A
    plain, fully fixed argument list (`List[int]`, `Dict[str, int]`) is a
    closed shape, so it is compared element by element as before.
    """
    return _issubtupleshape(_tuple_shape(args), _tuple_shape(superargs))


def _issubnone(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for NoneType."""
    none_uw = get_origin_uw(superhint)
    if none_uw is not type(None):
        raise TypeError(f"nonehint {superhint} is not a NoneType")
    origin_uw = get_origin_uw(hint)
    return origin_uw is type(None)


def _issubliteral(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a Literal."""
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
    """Check that a hint is a sub-hint for a TypeVar."""
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
    """Check that a hint is a sub-hint for a Union."""
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
    """Check that a hint is a sub-hint for a type[...] hint."""
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
    """Check that a hint is a sub-hint for a `Callable[...]`.

    Parameters are compared contravariantly and the return type covariantly.
    A `#!python ...` or a bare `ParamSpec` parameter list is the **top** of
    parameter lists -- the widest, describing every callable -- so a fixed
    list is a sub-hint of it but not the other way round; a
    `#!python Concatenate[X, P]` list is a contravariant fixed prefix followed
    by an open tail, sitting between the fixed lists and the top (RFC 11.1). A
    callable *class* -- a function type, `#!python type`, `#!python Type[C]`,
    or a class with `__call__` -- has no parameter list, so it stands in only
    for an unparametrised `Callable`.
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
    """A `Callable` parameter list as a shape (RFC 11.1).

    A fixed, contravariant `prefix` followed by a `tail` saying how the list
    ends:

    * `#!python None` -- a **closed** (fixed-arity) list, e.g. `[int, str]`;
    * `#!python Ellipsis` or a [`ParamSpec`][typing.ParamSpec] -- an **open**
      list, one that may be called with arbitrarily many further arguments.

    An open list is the *top* of parameter lists, the way `#!python Tuple[X,
    ...]` tops the fixed-length tuples: a fixed list is a sub-hint of it, but
    it is not a sub-hint of any fixed list.
    """

    prefix: tx.Tuple[tx.Any, ...]
    tail: tx.Any


# A sentinel `_match_params` returns when a closed super matches: there is no
# tail to capture, but the result must be non-`None` to signal the match.
_CLOSED_MATCH = _ParamShape((), None)


def _callable_param_shape(alias: tx.Any, params: tx.Any) -> _ParamShape:
    """Classify a `Callable` parameter list into a `_ParamShape` (RFC 11.1).

    `alias` is the whole `Callable[...]` hint, needed to reach
    `#!python __parameters__` where an older Python has erased a `ParamSpec`
    out of the arguments; `params` is its parameter-list argument.
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
    """Match one parameter-list shape against another (RFC 11.1).

    `sub` describes the callables the left `Callable` accepts, `sup` those the
    right does; the match holds when every callable `sub` describes `sup`
    describes too, with the committed prefixes compared **contravariantly**.

    * A **closed** `sup` accepts exactly its own arity: `sub` must be closed
      too, with the same prefix length.
    * An **open** `sup` accepts its committed prefix and anything after: `sub`
      must supply at least that prefix. The variable at `sup`'s tail then
      captures the rest of `sub`'s list -- the leftover prefix plus `sub`'s own
      tail.

    Returns that captured tail shape when `sup` is open, a sentinel closed
    shape when `sup` is closed and matches, or `#!python None` on no match.
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
    """Whether one parameter-list shape stands in for another (RFC 11.1)."""
    return _match_params(sub, sup) is not None


class _TupleShape(tx.NamedTuple):
    """A tuple's arguments as a shape (RFC 0001 §11.1, PEP 646).

    A tuple is a fixed, front-aligned `prefix`, an optional open middle run,
    and a fixed, back-aligned `suffix`. The middle run is:

    * `#!python None` in `rep` -- a **closed** (fixed-arity) tuple, e.g.
      `Tuple[int, str]`;
    * otherwise `rep` is the element upper bound of the run -- an **open**
      tuple, `Tuple[int, ...]` (bound `int`) or `Tuple[int, *Ts]` (bound
      `Any`).

    `var` is the `TypeVarTuple` an `*Ts` run stands for, or `#!python None`.
    All fixed positions are covariant; an open run tops the tuples of its
    shape, as `Tuple[int, ...]` tops the fixed-length tuples beginning `int`.
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

    `Tuple[()]` on 3.8-3.10 reports its arguments as the phantom `#!python
    ((),)`; it is normalised here to no elements, so the empty-tuple type is
    read the same way on every version. A second open run in one tuple is
    refused (PEP 646 allows a single unpack; `typing` does not enforce it at
    runtime, so the relation does).
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
    """Match one tuple shape against another, covariantly (the tuple twin of
    `_match_params`).

    Returns the shape a `*Ts` in `sup` would capture from `sub` (the run `sub`
    supplies beyond `sup`'s fixed prefix and suffix), a sentinel closed shape
    when `sup` is closed and matches, or `#!python None` on no match. Every
    fixed position is compared covariantly (`issubhint(sub_i, sup_i)`).
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
    """Whether one tuple shape stands in for another."""
    return _match_tuple(sub, sup) is not None
