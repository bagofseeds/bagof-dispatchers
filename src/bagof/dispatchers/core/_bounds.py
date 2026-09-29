"""Intervals of classes or hints, read against a value's class or against
a class or hint passed in.

A `Super[C]` or a `Between[L, U]` describes a closed interval between a
lower and an upper bound. On a value parameter the interval holds the
classes a value's own class may have, and inside `Type[...]` or
`Hint[...]` it holds the classes or hints that may be passed. A plain `C`
is the interval from the bottom `Never` up to `C`,
[`Super`][bagof.dispatchers.Super]`[C]` is the interval from `C` up to
the top, and [`Between`][]`[L, U]` names both ends. This module holds
`Between`, the readers that turn a bound into its two ends, and the
refusals shared by every place where a bound can be written in a position
where it cannot be read.
"""

# stdlib
import functools
from collections import abc

# dependencies
import typing_extensions as tx

# local
from ._compat import _LITERAL_FORMS, UNION_TYPES, ishint, spellings
from ._exact import _ANNOTATED_ALIAS, Exact, exact_target, is_exact
from ._hint import hint_arg, is_hint_form
from ._introspect import get_args_uw, get_origin_uw, normalise_hint
from ._super import (
    SUPER,
    _render_target,
    bare_super_message,
    is_bare_super,
    is_super,
    super_target,
)

_ANY_FORMS = spellings("Any")
_NEVER_FORMS = spellings("Never") + spellings("NoReturn")


class _Lower:
    """The piece of `Annotated` metadata that carries a lower bound.

    [`Between`][]`[L, U]` is spelled `Annotated[U, _Lower(L)]`: the upper
    bound is the annotated type and the lower bound travels as metadata.
    Two markers are equal when their lower bounds are, so that two
    spellings of the same interval are the same hint. A lower bound that
    cannot be hashed makes the whole hint unhashable, which every cache
    in this package already tolerates.
    """

    __slots__ = ("lower",)

    def __init__(self, lower: tx.Any) -> None:
        self.lower = lower

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Lower):
            return NotImplemented
        return bool(self.lower == other.lower)

    def __hash__(self) -> int:
        return hash((_Lower, self.lower))

    def __repr__(self) -> str:
        return f"LOWER({_render_target(self.lower)})"


def _is_any(hint: tx.Any) -> bool:
    """Report whether `hint` is `Any`, in any spelling."""
    return any(hint is form for form in _ANY_FORMS)


def _is_forward(hint: tx.Any) -> bool:
    """Report whether `hint` holds a forward reference that is not resolved.

    A forward reference is written as a string, which `typing` turns into
    a [`ForwardRef`][typing.ForwardRef] when it is the argument of a
    generic form. It counts wherever it appears in `hint`, so
    `#!python Optional["Later"]` and `#!python List["Later"]` hold one
    as much as `#!python "Later"` does. The arguments of a `Literal` are
    values rather than hints, so a string among them is not a forward
    reference.
    """
    if isinstance(hint, str) or isinstance(
        getattr(hint, "__forward_arg__", None), str
    ):
        return True
    if isinstance(hint, (list, tuple)):
        # The parameter list of a `Callable`.
        return any(_is_forward(arg) for arg in hint)
    if get_origin_uw(hint) in _LITERAL_FORMS:
        return False
    return any(_is_forward(arg) for arg in get_args_uw(hint))


def nesting_message(outer: str, inner: tx.Any) -> str:
    """Compose the error for an `Exact` or a `Super` given a `Between`.

    `outer` names the form being applied, `"Exact"` or `"Super"`, and
    `inner` is the [`Between`][] hint it was given. The message explains
    that the three forms each describe the whole hint at their position,
    and names the spelling to use for each.
    """
    return (
        f"{outer}[...] cannot take {_render_target(inner)}: Exact, Super "
        "and Between cannot be nested, because each of them already "
        "describes the whole hint at its position. " + _SPELLINGS
    )


# The spellings that every refusal of a nested bound ends with.
_SPELLINGS = (
    "On a value, write Between[L, U] for a value whose class lies between "
    "L and U, Super[C] for a value whose class is C or a class above it, "
    "or Exact[C] for a value whose class is exactly C. Inside Type[...], "
    "write Type[Between[L, U]] for a class between L and U, Type[Super[C]] "
    "for C or any class above it, or Type[Exact[C]] for exactly C; the "
    "same holds inside Hint[...]."
)


def empty_interval_message(lower: tx.Any, upper: tx.Any) -> tx.Optional[str]:
    """Explain why no hint lies between `lower` and `upper`, if none does.

    The interval `Between[lower, upper]` holds every hint that is a
    super-hint of `lower` and a sub-hint of `upper`, so it is empty
    exactly when `lower` is not a sub-hint of `upper`. The message then
    says which of three mistakes was probably made: writing `Any` as a
    lower bound where `Never` was meant, writing the two bounds the wrong
    way round, or bounding by two unrelated hints. `None` comes back when
    the interval is not empty, and also when either bound is still a
    forward reference, whose emptiness can only be decided once it
    resolves.
    """
    if _is_forward(lower) or _is_forward(upper):
        return None
    # Imported here because the relation imports this module.
    from ._relation import issubhint

    if issubhint(lower, upper):
        return None
    shown_lower, shown_upper = _render_target(lower), _render_target(upper)
    if _is_any(lower):
        return (
            f"Between[Any, {shown_upper}] is empty: Any is the top of the "
            f"hint order, so nothing lies between Any and {shown_upper}. For "
            f"no lower bound write Between[Never, {shown_upper}], which "
            f"means the same as plain {shown_upper}."
        )
    if issubhint(upper, lower):
        return (
            f"Between[{shown_lower}, {shown_upper}] is empty: {shown_lower} "
            f"is not a sub-hint of {shown_upper}, so no hint lies between "
            f"them. Did you mean Between[{shown_upper}, {shown_lower}]?"
        )
    return (
        f"Between[{shown_lower}, {shown_upper}] is empty: {shown_lower} is "
        f"not a sub-hint of {shown_upper}, and {shown_upper} is not a "
        f"sub-hint of {shown_lower}, so no hint lies between them."
    )


if tx.TYPE_CHECKING:
    # As with `Super`, a type checker must accept every call the runtime
    # accepts. `Between[L, U]` is spelled `Union[L, U, Any]`: a generic alias
    # has to use each of its type variables, and the `Any` member makes
    # `Type[Between[L, U]]` accept any class object, so a checker never
    # rejects a class that the runtime would accept.
    _L = tx.TypeVar("_L")
    _U = tx.TypeVar("_U")
    Between: tx.TypeAlias = tx.Union[_L, _U, tx.Any]
else:

    class Between:
        """Accept what lies between two bounds.

        An ordinary annotation bounds what a parameter accepts from
        above, and [`Super`][bagof.dispatchers.Super] bounds it from
        below. `Between[L, U]` bounds it from both sides at once. On a
        value parameter, `#!python x: Between[Dog, Animal]` accepts a
        value whose class is `Dog`, `Animal`, or a class that derives from
        `Animal` and is itself a base of `Dog`. It refuses an instance of
        `Puppy`, a subclass of `Dog`, because `Puppy` lies below the lower
        bound, and it refuses a plain `object()`, because `object` lies
        above the upper bound. Inside [`Type`][typing.Type], where the
        value passed is itself a class, `#!python Type[Between[Dog, Animal]]`
        accepts the same classes `Dog` and `Animal` and those between
        them. Inside [`Hint`][bagof.dispatchers.Hint], where the value
        passed is a type hint, `#!python Hint[Between[int, numbers.Real]]`
        accepts the hints `#!python int`, `#!python numbers.Integral` and
        `#!python numbers.Real`, and refuses `#!python bool`, which lies
        below `#!python int`, as well as `#!python object` and
        `#!python Any`, which lie above `#!python numbers.Real`.

        Each of these spellings describes such an interval. A plain `C` is
        the interval `#!python Between[Never, C]`, since it accepts `C` and
        everything below it. `#!python Super[C]` is the interval from `C`
        to the top, which is `object` on a value and inside `Type`, and
        `Any` inside `Hint`. [`Exact`][]`[C]` names the single class or
        hint `C` as it is written, which is narrower than
        `#!python Between[C, C]`. On a value, `#!python Between[C, C]`
        accepts a value whose class is equivalent to `C`, meaning that
        each of the two classes counts as a subclass of the other, which
        `register` or `__subclasshook__` can arrange for a class other
        than `C`. `#!python Exact[C]` accepts a value whose class is `C`
        itself.

        On a value, the interval is compared against the value's class, so
        each bound must be a hint that a class can be compared against,
        such as a class, a union of classes, `Never` or `Any`. A hint that
        reads the value itself, such as a `Literal`, a `TypedDict` or a
        parametrised generic, is refused as a bound there. Inside `Type`
        or `Hint`, any hint can be a bound, so the same
        `#!python Between[Literal[1], int]` that is refused on a value is
        accepted inside `Hint[...]`.

        One interval is more specific than another when it lies inside
        it, so `#!python Between[Dog, Animal]` is more specific than
        `#!python Super[Dog]` and than `Animal`. On the other hand, `Dog`
        and `#!python Between[Dog, Animal]` are not ordered against each
        other, although both accept a value of class `Dog`, so registering
        methods for both warns that such a call is ambiguous, and a
        `priority` on either method settles it. The same holds inside
        `Type` and `Hint`.

        The lower bound must be a sub-hint of the upper bound. An empty
        interval such as `#!python Between[Animal, Dog]` is refused when
        it is written, with a message that suggests the reversed spelling
        when that one is not empty. `#!python Between[Any, U]` is refused
        as well unless `U` is `Any` itself, because nothing else lies
        above `Any`. The interval with no lower bound is spelled
        `#!python Between[Never, U]`, which means the same as plain `U`.
        A lower bound cannot be a quoted forward reference, because
        nothing would ever resolve it, but quoting the whole annotation
        works as usual.

        `Exact`, `Super` and `Between` cannot be nested inside one
        another, even through a union or a `TypeVar`, and inside `Type`
        or `Hint` a bound is always written inside the brackets, as in
        `#!python Type[Between[L, U]]`, never around them. A type checker
        reads `Between[L, U]` as `#!python Union[L, U, Any]`, which
        accepts every call the runtime accepts.

        !!! example
            ```pycon
            >>> from typing import Type
            >>> from bagof.dispatchers import Between
            >>> from bagof.dispatchers.core import ishintstance
            >>> class Animal: pass
            >>> class Dog(Animal): pass
            >>> class Puppy(Dog): pass
            >>> ishintstance(Dog(), Between[Dog, Animal])
            True
            >>> ishintstance(Puppy(), Between[Dog, Animal])
            False
            >>> ishintstance(Animal, Type[Between[Dog, Animal]])
            True
            >>> ishintstance(object, Type[Between[Dog, Animal]])
            False
            ```
        """

        def __class_getitem__(cls, item: tx.Any) -> tx.Any:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "Between[...] takes two type hints, a lower and an upper "
                    f"bound, as Between[L, U]; got {item!r}."
                )
            lower, upper = (
                type(None) if end is None else end for end in item
            )
            for end in (lower, upper):
                if not ishint(end):
                    raise TypeError(
                        f"Between[...] takes type hints as its bounds, got "
                        f"{end!r}."
                    )
            for end in (lower, upper):
                if (
                    is_unbounded_form(end)
                    or is_bound(end)
                    or is_exact(end)
                    or end is Exact
                ):
                    raise TypeError(
                        f"Between[...] cannot take {_render_target(end)} as "
                        "a bound: a bound is a plain hint, and Exact, Super "
                        "and Between cannot be nested. Write Between[L, U] "
                        "with plain L and U. " + _SPELLINGS
                    )
                found = find_bound(end)
                if found is not None:
                    raise TypeError(
                        misplaced_bound_message(found, endpoint_bound_message)
                    )
            if _is_forward(lower):
                shown = _render_target(lower)
                spelled = f"Between[{shown}, {_render_target(upper)}]"
                if isinstance(lower, str) or hasattr(lower, "__forward_arg__"):
                    what = f"the forward reference {shown!r}"
                    define = shown
                else:
                    what = f"{shown}, which holds a forward reference,"
                    define = f"every name that {shown} refers to"
                raise TypeError(
                    f"Between[...] cannot take {what} as its lower bound, "
                    "because a lower bound is kept as written and never "
                    f"resolved. Define {define} before the annotation is "
                    "read, or quote the whole annotation instead, as "
                    f"'Type[{spelled}]'."
                )
            message = empty_interval_message(lower, upper)
            if message is not None:
                raise TypeError(message)
            marker = _Lower(lower)
            try:
                return tx.Annotated[upper, marker]
            except TypeError:
                # A bare special form (`Between[Never, Union]`), which
                # `typing._type_check` refuses: build the alias directly.
                return _ANNOTATED_ALIAS(upper, (marker,))


def is_between(hint: tx.Any) -> bool:
    """Report whether `hint` was built with [`Between`][]`[L, U]`."""
    return any(
        isinstance(meta, _Lower) for meta in getattr(hint, "__metadata__", ())
    )


def between_bounds(hint: tx.Any) -> tx.Tuple[tx.Any, tx.Any]:
    """Return the bounds `(L, U)` of a [`Between`][]`[L, U]` hint.

    Calling this only makes sense once [`is_between`][]`(hint)` has been
    confirmed. Any other `Annotated` metadata that rides alongside the
    lower bound is discarded.
    """
    marker = next(
        meta for meta in hint.__metadata__ if isinstance(meta, _Lower)
    )
    return marker.lower, tx.get_args(hint)[0]


def is_bound(arg: tx.Any) -> bool:
    """Report whether `arg` is a `Super[C]` or a `Between[L, U]` hint.

    These are the two forms that give a hint a lower bound, and so the two
    that the relation reads as an interval rather than as an ordinary
    hint, whether they stand on a value or as the argument of `Type` or
    `Hint`.
    """
    for meta in getattr(arg, "__metadata__", ()):
        if meta is SUPER or isinstance(meta, _Lower):
            return True
    return False


def written_ends(bound: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """Return the hints a `Super[C]` or a `Between[L, U]` is written with.

    A `Between[L, U]` names its two bounds and a `Super[C]` names `C`,
    since the top it runs up to is implied by its position rather than
    written.
    """
    if is_super(bound):
        return (super_target(bound),)
    return between_bounds(bound)


def bounds_of(arg: tx.Any, top: tx.Any) -> tx.Tuple[tx.Any, tx.Any]:
    """Read a hint as a closed interval of classes or of hints.

    On a value, a hint describes which classes the value's own class may
    have. The values of `#!python Type[X]` are classes and the values of
    `#!python Hint[X]` are hints, and the argument `X` describes which of
    them are accepted. Either way the hint is read as the interval
    between a lower and an upper bound, returned as `(lower, upper)`. A
    `Between[L, U]` names both ends itself. A plain `X` accepts everything
    from the bottom `Never` up to `X`. A `Super[C]` accepts everything
    from `C` up to `top`, which is `object` on a value and for a `Type`
    argument, since every class is below it, and `Any` for a `Hint`
    argument, since every hint is below that. An `Exact[C]` read this way
    is the single point `(C, C)`, which is only correct when it is the
    narrower side of a comparison; as the wider side, an exact hint is
    never known to contain a lower bound, and the caller decides that
    case before reading any interval.

    A `TypeVar` is not read through its bound here. On the wider
    side it stays a plain upper bound, which the relation already orders
    correctly, and on the narrower side the caller reads the bound first.
    """
    if is_between(arg):
        return between_bounds(arg)
    if is_super(arg):
        return super_target(arg), top
    if is_exact(arg):
        target = exact_target(arg)
        return target, target
    return tx.Never, arg


def is_unbounded_form(hint: tx.Any) -> bool:
    """Report whether `hint` is a bound form written without its bounds.

    This is true of the unsubscripted `Super`, `SuperType`, `SuperHint`
    and [`Between`][], none of which names a range until it is given its
    bounds. Such a form is refused wherever a hint is read, with
    [`unbounded_form_message`][].
    """
    return is_bare_super(hint) or hint is Between


def unbounded_form_message(hint: tx.Any) -> str:
    """Compose the error for a bound form written without its bounds.

    `hint` is one that [`is_unbounded_form`][] reports. The message names
    the form and the spellings that give it its bounds.
    """
    if hint is Between:
        return (
            "Between needs two bounds: write Between[L, U] to accept a value "
            "whose class lies between L and U, Type[Between[L, U]] to accept "
            "a class between them, or Hint[Between[L, U]] to accept a hint "
            "between them."
        )
    return bare_super_message(hint)


def find_bound(hint: tx.Any) -> tx.Any:
    """Find a bound that stands inside `hint` as part of one set of values.

    A union holds the values of each of its members, a `TypeVar` stands
    for its bound or for one of its constraints, and a plain `Annotated`
    wrapper stands for the hint it wraps, so a bound reached through any
    of these describes part of the same set as `hint` itself. This
    returns the first `Super[C]` or `Between[L, U]` found that way, or
    the first unsubscripted form that [`is_unbounded_form`][] reports, and
    `None` when there is none. The search does not enter the argument of
    `Type` or `Hint`, nor the arguments of any other generic, because
    each of those positions is checked where it is compared.
    """
    if is_unbounded_form(hint):
        return hint
    if isinstance(hint, type):
        # A class is neither a bound nor made of other hints.
        return None
    try:
        return _find_bound_cached(hint)
    except TypeError:
        # An unhashable hint, such as one carrying a list as `Annotated`
        # metadata, cannot key the cache.
        return _find_bound(hint)


@functools.lru_cache(maxsize=4096)
def _find_bound_cached(hint: tx.Any) -> tx.Any:
    """Remember what [`_find_bound`][] finds, since the answer depends on
    nothing but the hint.
    """
    return _find_bound(hint)


def _find_bound(hint: tx.Any) -> tx.Any:
    """Search `hint` the way [`find_bound`][] describes."""
    hint = normalise_hint(hint)
    if is_bound(hint) or is_unbounded_form(hint):
        return hint
    if is_exact(hint):
        return None
    if isinstance(hint, tx.TypeVar):
        bound = getattr(hint, "__bound__", None)
        members = tuple(getattr(hint, "__constraints__", ()))
        if bound is not None:
            members += (bound,)
    elif get_origin_uw(hint) in UNION_TYPES:
        members = get_args_uw(hint)
    elif getattr(hint, "__metadata__", None) is not None:
        members = tx.get_args(hint)[:1]
    else:
        return None
    for member in members:
        found = find_bound(member)
        if found is not None:
            return found
    return None


def holds_bound(hint: tx.Any) -> bool:
    """Report whether [`find_bound`][] finds a bound inside `hint`."""
    return find_bound(hint) is not None


def misplaced_bound_message(
    found: tx.Any, message: tx.Callable[[tx.Any], str]
) -> str:
    """Compose the error for a bound that [`find_bound`][] found.

    An unsubscripted form is always refused for its missing bounds, and
    any other bound is refused with the message that `message` composes
    for the position it was found in.
    """
    if is_unbounded_form(found):
        return unbounded_form_message(found)
    return message(found)


def member_bound_message(bound: tx.Any) -> str:
    """Compose the error for a bound reached through a union member or a
    `TypeVar` inside the argument of `Type` or `Hint`.
    """
    shown = _render_target(bound)
    return (
        f"{shown} cannot be a member of a union, or the bound of a TypeVar, "
        "inside the argument of Type[...] or Hint[...]: a bound there has "
        "to be the whole argument, because the class or hint passed is "
        "compared against it as a class or hint rather than as a value. "
        f"Write Union[Type[{shown}], Type[B]] in place of "
        f"Type[Union[{shown}, B]], and Type[{shown}] in place of a Type[T] "
        f"whose T is bounded by {shown}; the same holds inside Hint[...]."
    )


def endpoint_bound_message(bound: tx.Any) -> str:
    """Compose the error for a bound found inside the bound of another."""
    return (
        f"{_render_target(bound)} cannot appear inside the bound of another "
        "Super or Between, even through a union or a TypeVar: Exact, Super "
        "and Between cannot be nested. Write the outer bound with plain "
        "hints."
    )


def constraint_bound_message(bound: tx.Any) -> str:
    """Compose the error for a bound written as a `TypeVar` constraint."""
    shown = _render_target(bound)
    return (
        f"{shown} cannot be a constraint of a TypeVar: a constrained "
        "variable is solved to the one constraint an argument's class is "
        "below, and a class is never below a bound, so such a variable "
        f'would apply to no call. Write TypeVar("T", bound={shown}) to '
        "bound the variable, or a Union of the alternatives."
    )


def argument_bound_message(bound: tx.Any, origin: tx.Any) -> str:
    """Compose the error for a bound written as a type argument of a
    generic class.
    """
    name = getattr(origin, "__name__", None) or _render_target(origin)
    return (
        f"{_render_target(bound)} is not supported as a type argument of "
        f"{name}[...]: a bound can stand on a value parameter, or as the "
        "argument of Type[...] or Hint[...], but not inside a generic's "
        "type arguments. Write a plain type argument instead."
    )


def tuple_bound_message(bound: tx.Any) -> str:
    """Compose the error for a bound written as an element of `Tuple`."""
    return (
        f"{_render_target(bound)} cannot be an element of Tuple[...]: a "
        "tuple element is covariant by the form itself, so a bound adds "
        "nothing that a plain element hint cannot say. Write a plain "
        "element hint instead."
    )


def callable_bound_message(bound: tx.Any) -> str:
    """Compose the error for a bound written in the signature of
    `Callable`.
    """
    return (
        f"{_render_target(bound)} cannot be a parameter or the return type "
        "of Callable[...]: Callable parameters are contravariant and its "
        "return type is covariant by the form itself, so a bound adds "
        "nothing that a plain hint cannot say. Write a plain hint instead."
    )


def slot_bound_message(bound: tx.Any, origin: tx.Any) -> str:
    """Compose the error for a bound written as a type argument of
    `origin`.

    A tuple element and a `Callable` parameter or return type each have a
    message of their own, and every other generic shares
    [`argument_bound_message`][].
    """
    if origin is tuple:
        return tuple_bound_message(bound)
    if origin is abc.Callable:
        return callable_bound_message(bound)
    return argument_bound_message(bound, origin)


def _is_never(hint: tx.Any) -> bool:
    """Report whether `hint` is a bottom type, `Never` or `NoReturn`."""
    return any(hint is form for form in _NEVER_FORMS)


def _wrapped_argument(end: tx.Any, form: str) -> tx.Any:
    """Return the argument of `end` when it is a `Type` or `Hint` form.

    `form` names the wrapper looked for, `"Type"` or `"Hint"`. A bottom or
    `Any` is returned unchanged, because it can stand beside either
    wrapper, and `None` comes back for anything else.
    """
    if _is_never(end) or _is_any(end):
        return end
    origin = get_origin_uw(end)
    args = get_args_uw(end)
    if form == "Type" and origin is type and args:
        return args[0]
    if form == "Hint" and is_hint_form(origin) and args:
        return hint_arg(end)
    return None


def value_bound_message(hint: tx.Any) -> tx.Optional[str]:
    """Explain why a bound cannot stand on a value parameter, if it cannot.

    `hint` is a `Super[C]` or a `Between[L, U]` met where a value is
    checked. On a value, a bound constrains the value's class, so each of
    its bounds must be a hint that a class can be compared against, and
    the upper bound of `Super[C]`, which is `object`, always is. The
    message refuses the first bound that holds another bound, and then
    the first bound that is read against a value rather than against a
    class. `None` comes back when the bound is legal on a value, and a
    bound that is still a forward reference is left to be checked once
    it resolves.
    """
    # Imported here because the relation imports this module.
    from ._relation import is_class_hint

    ends = written_ends(hint)
    for end in ends:
        found = find_bound(end)
        if found is not None:
            return misplaced_bound_message(found, endpoint_bound_message)
    for end in ends:
        if _is_forward(end) or is_class_hint(end):
            continue
        shown, shown_end = _render_target(hint), _render_target(end)
        advice = (
            "Inside Type[...] or Hint[...], where the value passed is itself "
            "a class or a hint, any hint can be a bound."
        )
        for form, what in (("Type", "class"), ("Hint", "hint")):
            inner = [_wrapped_argument(each, form) for each in ends]
            if len(ends) == 2 and all(each is not None for each in inner):
                lower, upper = (_render_target(each) for each in inner)
                advice = (
                    f"To bound the {what} passed to a {form}[...] parameter "
                    f"write {form}[Between[{lower}, {upper}]]."
                )
        return (
            f"{shown} cannot bound a value with {shown_end}: a bound on a "
            "value parameter constrains the value's class, so each bound "
            "has to be a hint that a class can be compared against, such as "
            f"a class, a union of classes, Never or Any. {shown_end} is "
            "matched against a value rather than against the value's class. "
            + advice
        )
    return None
