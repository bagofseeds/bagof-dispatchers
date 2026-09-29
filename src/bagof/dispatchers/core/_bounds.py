"""Intervals of classes or hints inside `Type[...]` and `Hint[...]`.

The argument of `Type[...]` or `Hint[...]` describes a set of classes or
hints, which the relation reads as a closed interval between a lower and
an upper bound. A plain `C` is the interval from the bottom `Never` up to
`C`, [`Super`][bagof.dispatchers.Super]`[C]` is the interval from `C` up
to the top, and [`Between`][]`[L, U]` names both ends. This module holds
`Between`, the readers that turn such an argument into its two ends, and
the refusals shared by every place where a bound can be written in the
wrong position.
"""

# dependencies
import typing_extensions as tx

# local
from ._compat import ishint, spellings
from ._exact import _ANNOTATED_ALIAS, Exact, exact_target, is_exact
from ._super import (
    _render_target,
    bare_super_message,
    is_bare_super,
    is_super,
    super_target,
)

_ANY_FORMS = spellings("Any")


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
    """Report whether `hint` is a forward reference that is not resolved yet.

    A forward reference is written as a string, which `Annotated` turns
    into a [`ForwardRef`][typing.ForwardRef] when it is the annotated
    type.
    """
    return isinstance(hint, str) or isinstance(
        getattr(hint, "__forward_arg__", None), str
    )


def _render_marked(hint: tx.Any) -> str:
    """Spell a hint the way an error message names it, markers included.

    An `Exact`, `Super` or `Between` hint is named by the spelling it was
    written with rather than by its `Annotated` form, and any other hint
    as [`_render_target`][] spells it.
    """
    if is_exact(hint):
        return f"Exact[{_render_target(exact_target(hint))}]"
    if is_super(hint):
        return f"Super[{_render_target(super_target(hint))}]"
    if is_between(hint):
        lower, upper = between_bounds(hint)
        return f"Between[{_render_target(lower)}, {_render_target(upper)}]"
    return _render_target(hint)


def nesting_message(outer: str, inner: tx.Any) -> str:
    """Compose the error for an `Exact` or a `Super` given a `Between`.

    `outer` names the form being applied, `"Exact"` or `"Super"`, and
    `inner` is the [`Between`][] hint it was given. The message explains
    that the three forms each describe a whole argument of `Type` or
    `Hint`, and names the spelling to use for each.
    """
    return (
        f"{outer}[...] cannot take {_render_marked(inner)}: Exact, Super "
        "and Between cannot be nested, because each of them already "
        "describes the whole argument of Type[...] or Hint[...]. Write "
        "Type[Between[L, U]] for a class between L and U, Type[Super[C]] "
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
            "inside Type[...] or Hint[...] means the same as plain "
            f"{shown_upper}."
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
        """Accept the classes or hints that lie between two bounds.

        An ordinary annotation bounds what a parameter accepts from
        above, and [`Super`][bagof.dispatchers.Super] bounds it from
        below. `Between[L, U]` bounds it from both sides at once, and
        like `Super` it is only meaningful inside
        [`Type`][typing.Type] or [`Hint`][bagof.dispatchers.Hint]. A
        parameter annotated `#!python Type[Between[Dog, Animal]]`
        accepts the classes `Dog` and `Animal`, together with any class
        that derives from `Animal` and is itself a base of `Dog`. It
        refuses `Puppy`, a subclass of `Dog`, because `Puppy` lies below
        the lower bound, and it refuses `object`, because `object` lies
        above the upper bound. In the same way,
        `#!python Hint[Between[bool, numbers.Integral]]` accepts the
        hints `#!python bool`, `#!python int` and
        `#!python numbers.Integral`, but neither `#!python object` nor
        `#!python Any`.

        Every spelling that can stand inside `Type[...]` or `Hint[...]`
        describes such an interval. A plain `C` is the interval
        `#!python Between[Never, C]`, since it accepts `C` and
        everything below it. `#!python Super[C]` is the interval from
        `C` to the top, which is `object` inside `Type` and `Any` inside
        `Hint`. [`Exact`][bagof.dispatchers.Exact]`[C]` stays apart from
        this family: it names the one hint `C` as it is written, which
        is narrower than `#!python Between[C, C]`, the interval of every
        hint equivalent to `C`.

        One interval is more specific than another when it lies inside
        it, so `#!python Type[Between[Dog, Animal]]` is more specific
        than `#!python Type[Super[Dog]]` and than
        `#!python Type[Animal]`. On the other hand,
        `#!python Type[Dog]` and `#!python Type[Between[Dog, Animal]]`
        are not ordered against each other, although both accept the
        class `Dog`, so registering methods for both warns that a call
        with `Dog` is ambiguous, and a `priority` on either method
        settles it.

        The lower bound must be a sub-hint of the upper bound. An empty
        interval such as `#!python Between[int, bool]` is refused when it
        is written, with a message that suggests the reversed spelling
        when that one is not empty. `#!python Between[Any, U]` is refused
        as well unless `U` is itself a top, because only a top lies above
        `Any`. The interval with no lower bound is spelled
        `#!python Between[Never, U]`, which inside `Type` or `Hint` means
        the same as plain `U`. A lower bound cannot be a quoted forward
        reference, because nothing would ever resolve it, but quoting the
        whole annotation works as usual.

        `Exact`, `Super` and `Between` cannot be nested inside one
        another, and `Between` is always written inside the brackets, as
        in `#!python Type[Between[L, U]]`, never around them. A value has
        a single concrete class, so an interval of classes cannot be
        checked on an ordinary value parameter, and registration refuses
        `Between` there, naming the parameter. A type checker reads
        `Between[L, U]` as `#!python Union[L, U, Any]`, which accepts
        every call the runtime accepts inside `Type[...]`.

        !!! example
            ```pycon
            >>> from typing import Type
            >>> from bagof.dispatchers import Between
            >>> from bagof.dispatchers.core import ishintstance
            >>> class Animal: pass
            >>> class Dog(Animal): pass
            >>> class Puppy(Dog): pass
            >>> ishintstance(Animal, Type[Between[Dog, Animal]])
            True
            >>> ishintstance(Puppy, Type[Between[Dog, Animal]])
            False
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
                if is_bare_bound(end) or is_exact(end) or end is Exact:
                    raise TypeError(
                        f"Between[...] cannot take {_render_marked(end)} as "
                        "a bound: a bound is a plain hint, and Exact, Super "
                        "and Between cannot be nested. Write Between[L, U] "
                        "with plain L and U; for exactly C write "
                        "Type[Exact[C]], and for C and everything above it "
                        "write Type[Super[C]]."
                    )
            if _is_forward(lower):
                shown = _render_target(lower)
                spelled = f"Between[{shown}, {_render_target(upper)}]"
                raise TypeError(
                    f"Between[...] cannot take the forward reference "
                    f"{shown!r} as its lower bound, because a lower bound "
                    "is kept as written and never resolved. Define "
                    f"{shown} before the annotation is read, or quote the "
                    f"whole annotation instead, as 'Type[{spelled}]'."
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

    These are the two forms that give the argument of `Type` or `Hint` a
    lower bound, and so the two that the relation reads as an interval
    rather than as an ordinary hint.
    """
    return is_super(arg) or is_between(arg)


def bounds_of(arg: tx.Any, top: tx.Any) -> tx.Tuple[tx.Any, tx.Any]:
    """Read the argument of a `Type` or `Hint` form as a closed interval.

    The values of `#!python Type[X]` are classes and the values of
    `#!python Hint[X]` are hints, and each argument describes which of
    them are accepted as the interval between a lower and an upper
    bound, returned as `(lower, upper)`. A `Between[L, U]` argument names
    both ends itself. A plain argument `X` accepts everything from the
    bottom `Never` up to `X`. A `Super[C]` argument accepts everything
    from `C` up to `top`, which is `object` for a `Type` argument, since
    every class is below it, and `Any` for a `Hint` argument, since every
    hint is below that. An `Exact[C]` argument read this way is the
    single point `(C, C)`, which is only correct when it is the narrower
    side of a comparison; as the wider side, an exact argument is never
    known to contain a lower bound, and the caller decides that case
    before reading any interval.

    A `TypeVar` argument is not read through its bound here. On the wider
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


def is_bare_bound(hint: tx.Any) -> bool:
    """Report whether `hint` is a bound standing outside `Type` or `Hint`.

    This is true of a `Super[C]` or a `Between[L, U]` hint itself, and of
    the unsubscripted `Super`, `SuperType`, `SuperHint` and [`Between`][].
    A bound is only meaningful as the immediate argument of `Type[...]` or
    `Hint[...]`, which read it before it could reach this check, so any of
    these met anywhere else is refused with [`bare_bound_message`][].
    """
    return is_bare_super(hint) or hint is Between or is_between(hint)


def bare_bound_message(hint: tx.Any) -> str:
    """Compose the error for a bound used outside `Type` or `Hint`.

    `hint` is a hint that [`is_bare_bound`][] reports. The message names
    what was written and the spelling to use instead.
    """
    if hint is Between:
        return (
            "Between needs two bounds and must sit inside Type[...] or "
            "Hint[...]: write Type[Between[L, U]] to accept a class between "
            "L and U, or Hint[Between[L, U]] to accept a hint between them."
        )
    if is_between(hint):
        lower, upper = (_render_target(end) for end in between_bounds(hint))
        shown = f"Between[{lower}, {upper}]"
        return (
            f"{shown} is only valid inside Type[...] or Hint[...]: a value "
            "has one concrete class, so an interval of classes cannot be "
            f"checked on a value parameter. Write Type[{shown}] to accept a "
            f"class between {lower} and {upper}, or Hint[{shown}] to accept "
            "a hint between them."
        )
    return bare_super_message(hint)
