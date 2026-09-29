"""A lower bound inside `Type[...]` or `Hint[...]`: a type and everything
above it.
"""

# dependencies
import typing_extensions as tx

# local
from ._compat import ishint
from ._exact import _ANNOTATED_ALIAS, exact_target, is_exact
from ._hint import Hint


class _SuperMarker:
    """The piece of `Annotated` metadata that tags a hint as [`Super`][]."""

    def __new__(cls) -> "_SuperMarker":
        if "_INSTANCE" not in cls.__dict__:
            cls._INSTANCE = object.__new__(cls)
        return cls._INSTANCE

    def __repr__(self) -> str:
        return "SUPER"


SUPER = _SuperMarker()
"""The sentinel that a [`Super`][] hint attaches as `Annotated` metadata."""


def _render_target(hint: tx.Any) -> str:
    """Spell a hint the way an error message names it.

    A class is named by its `__name__`, and any other hint by its string
    form with the `typing.` and `typing_extensions.` prefixes removed, so
    that a message reads `Super[int]` or `Super[List[int]]`.
    """
    if isinstance(hint, type):
        return hint.__name__
    text = str(hint)
    return text.replace("typing_extensions.", "").replace("typing.", "")


def combination_message(outer: str, inner: tx.Any) -> str:
    """Compose the error for an `Exact` and a `Super` applied to each other.

    `outer` names the form being applied, `"Exact"` or `"Super"`, and
    `inner` is the hint it was given, which already carries the other
    marker.
    """
    if is_exact(inner):
        shown = f"Exact[{_render_target(exact_target(inner))}]"
    else:
        shown = f"Super[{_render_target(super_target(inner))}]"
    return (
        f"{outer}[...] cannot take {shown}: Exact and Super cannot be "
        "combined, because Exact[C] names the single type C and leaves "
        "nothing above it to bound. Write Type[Exact[C]] for exactly C, or "
        "Type[Super[C]] for C and anything above it; the same holds inside "
        "Hint[...]."
    )


if tx.TYPE_CHECKING:
    # A type checker must accept every call the runtime accepts, and a
    # `Type[Super[Dog]]` parameter accepts `Animal` and `object`. Reading
    # `Super[C]` as `C`, the way `Exact[C]` is read, would reject those calls,
    # so `Super[C]` is spelled `Union[C, Any]` instead: the type variable is
    # used, which a checker requires of a generic alias, and the `Any` member
    # makes `Type[Super[C]]` accept any class object. The aliases are declared
    # with a PEP 613 annotation rather than a `# type:` comment, which pyright
    # does not read; this branch never runs, so the annotation is safe on 3.8.
    _T = tx.TypeVar("_T")
    Super: tx.TypeAlias = tx.Union[_T, tx.Any]
    SuperType: tx.TypeAlias = tx.Type[Super[_T]]
    SuperHint: tx.TypeAlias = Hint[Super[_T]]
else:

    class Super:
        """Bound a `Type` or `Hint` parameter from below.

        An ordinary annotation places an upper bound on what a parameter
        accepts: a parameter annotated `#!python Type[Animal]` accepts the
        class `Animal` and every class derived from it. `Super[C]` places
        a lower bound instead, and it is only meaningful inside
        [`Type`][typing.Type] or [`Hint`][]. A parameter annotated
        `#!python Type[Super[Dog]]` accepts the class `Dog` and every class
        that `Dog` derives from, such as `Animal` and `object`, but not a
        subclass of `Dog` and not an unrelated class. In the same way,
        `#!python Hint[Super[bool]]` accepts the hint `#!python bool`
        together with every hint above it, such as `#!python int`,
        `#!python Union[bool, str]`, and `#!python Any`.

        Which spellings are available therefore depends on where the hint
        appears. On an ordinary value parameter, a hint can be written as
        `C`, for `C` and its subclasses, or as
        [`Exact`][bagof.dispatchers.Exact]`[C]`, for `C` alone. Inside
        `Type[...]` or `Hint[...]`, where the value passed is itself a
        class or a hint, `Super[C]` is available as well. `Super` is
        refused on a value parameter because a value has a single concrete
        class, and a function written for the instances of some class must
        also accept the instances of its subclasses. A lower bound on a
        value would contradict that expectation and could not be checked,
        so registration refuses it and names the parameter. A bare
        `Super[C]` is refused wherever a hint is read, including when it
        is passed as a value to a `Hint[...]` parameter.

        `Super` is written inside the bracket, as in
        `#!python Type[Super[C]]`. Writing it around the whole form, as in
        `#!python Super[Type[C]]`, means the same thing and is normalised
        to the inner spelling. [`SuperType`][] and [`SuperHint`][] are
        shorter aliases for the two spellings.

        Lower bounds are ordered in the opposite direction to the classes
        they name. `#!python Type[Super[Animal]]` is more specific than
        `#!python Type[Super[Dog]]`, because every class above `Animal` is
        also above `Dog`, so the classes above `Animal` form the smaller
        set. `#!python Type[Exact[Dog]]` sits below both
        `#!python Type[Dog]` and `#!python Type[Super[Dog]]`, since the
        single class `Dog` belongs to each of them. A plain
        `#!python Type[Animal]` and `#!python Type[Super[Dog]]` are not
        ordered against each other, yet a call with the class `Dog` or
        `Animal` matches both, so registering methods for both warns about
        the ambiguity. A third method for `#!python Type[Exact[Dog]]`
        settles the call with `Dog`, and an explicit `priority` on either
        method settles every call the two share.

        `Super` and `Exact` cannot be combined, in either order, because
        an exact type has nothing above it to bound. A type checker reads
        `Super[C]` as `#!python Union[C, Any]`, which accepts every call
        the runtime accepts.

        !!! example
            ```pycon
            >>> from typing import Type
            >>> from bagof.dispatchers import Super
            >>> from bagof.dispatchers.core import ishintstance
            >>> Super[int]
            typing.Annotated[int, SUPER]
            >>> ishintstance(object, Type[Super[int]])
            True
            >>> ishintstance(bool, Type[Super[int]])
            False
            ```
        """

        def __class_getitem__(cls, item: tx.Any) -> tx.Any:
            if not ishint(item):
                raise TypeError(
                    f"Super[...] takes a type hint, got {item!r}."
                )
            if is_exact(item):
                raise TypeError(combination_message("Super", item))
            if is_super(item):
                # `Super[Super[C]]` bounds from below by the same `C`.
                return item
            try:
                return tx.Annotated[item, SUPER]
            except TypeError:
                # A bare special form (`Super[Union]`, `Super[Literal]`),
                # which `typing._type_check` refuses: build the alias directly.
                return _ANNOTATED_ALIAS(item, (SUPER,))

    class SuperType:
        """Accept a class and every class it derives from.

        `#!python SuperType[C]` is exactly `#!python Type[Super[C]]`, the
        spelling [`Super`][] describes, so a parameter annotated
        `#!python SuperType[Dog]` accepts the class `Dog` and each of its
        base classes up to `object`. The alias always needs its bound: a
        bare `#!python SuperType` is refused wherever a hint is read,
        because the unbounded reading would be every class, which plain
        `#!python type` already spells.

        !!! example
            ```pycon
            >>> from typing import Type
            >>> from bagof.dispatchers import Super, SuperType
            >>> SuperType[int] == Type[Super[int]]
            True
            ```
        """

        def __class_getitem__(cls, item: tx.Any) -> tx.Any:
            if not ishint(item):
                raise TypeError(
                    f"SuperType[...] takes a type hint, got {item!r}."
                )
            return tx.Type[Super[item]]

    class SuperHint:
        """Accept a type hint and every hint above it.

        `#!python SuperHint[X]` is exactly `#!python Hint[Super[X]]`, the
        spelling [`Super`][] describes, so a parameter annotated
        `#!python SuperHint[bool]` accepts the hint `#!python bool` and
        every hint that `#!python bool` is a sub-hint of, such as
        `#!python int` or `#!python Any`. The alias always needs its
        bound: a bare `#!python SuperHint` is refused wherever a hint is
        read, because the unbounded reading would be every hint, which
        plain [`Hint`][] already spells.

        !!! example
            ```pycon
            >>> from bagof.dispatchers import Hint, Super, SuperHint
            >>> SuperHint[int] == Hint[Super[int]]
            True
            ```
        """

        def __class_getitem__(cls, item: tx.Any) -> tx.Any:
            if not ishint(item):
                raise TypeError(
                    f"SuperHint[...] takes a type hint, got {item!r}."
                )
            return Hint[Super[item]]


def is_super(hint: tx.Any) -> bool:
    """Report whether `hint` was built with [`Super`][]`[C]`."""
    return any(meta is SUPER for meta in getattr(hint, "__metadata__", ()))


def super_target(hint: tx.Any) -> tx.Any:
    """Return the bound `C` from a [`Super`][]`[C]` hint.

    Calling this only makes sense once [`is_super`][]`(hint)` has been
    confirmed. The `SUPER` marker and any other `Annotated` metadata that
    rides alongside it are discarded.
    """
    return tx.get_args(hint)[0]


def is_bare_super(hint: tx.Any) -> bool:
    """Report whether `hint` is a lower bound standing outside `Type` or
    `Hint`.

    This is true of a `Super[C]` hint itself, and of the unsubscripted
    [`Super`][], [`SuperType`][] and [`SuperHint`][]. A lower bound is
    only meaningful as the immediate argument of `Type[...]` or
    `Hint[...]`, which read it before it could reach this check, so any
    of these met anywhere else is refused with
    [`bare_super_message`][].
    """
    return (
        hint is Super
        or hint is SuperType
        or hint is SuperHint
        or is_super(hint)
    )


def bare_super_message(hint: tx.Any) -> str:
    """Compose the error for a lower bound used outside `Type` or `Hint`.

    `hint` is a hint that [`is_bare_super`][] reports. The message names
    what was written and the spelling to use instead.
    """
    if hint is SuperType:
        return (
            "SuperType needs a bound: write SuperType[C], which is "
            "Type[Super[C]], to accept the class C or any class above it."
        )
    if hint is SuperHint:
        return (
            "SuperHint needs a bound: write SuperHint[X], which is "
            "Hint[Super[X]], to accept the hint X or any hint above it."
        )
    if hint is Super:
        return (
            "Super needs a bound and must sit inside Type[...] or "
            "Hint[...]: write Type[Super[C]] to accept the class C or any "
            "class above it, or Hint[Super[C]] to accept the hint C or any "
            "hint above it."
        )
    shown = _render_target(super_target(hint))
    return (
        f"Super[{shown}] is only valid inside Type[...] or Hint[...]: a "
        "value has one concrete class, so a lower bound on a value "
        f"parameter cannot be checked. Write Type[Super[{shown}]] to accept "
        f"the class {shown} or any class above it, or Hint[Super[{shown}]] "
        f"to accept the hint {shown} or any hint above it."
    )


def bounds_of(arg: tx.Any, top: tx.Any) -> tx.Tuple[tx.Any, tx.Any]:
    """Read the argument of a `Type` or `Hint` form as a closed interval.

    The values of `#!python Type[X]` are classes and the values of
    `#!python Hint[X]` are hints, and each argument describes which of
    them are accepted as the interval between a lower and an upper
    bound, returned as `(lower, upper)`. A plain argument `X` accepts
    everything from the bottom `Never` up to `X`. A `Super[C]` argument
    accepts everything from `C` up to `top`, which is `object` for a
    `Type` argument, since every class is below it, and `Any` for a
    `Hint` argument, since every hint is below that. An `Exact[C]`
    argument read this way is the single point `(C, C)`, which is only
    correct when it is the narrower side of a comparison; as the wider
    side, an exact argument is never known to contain a lower bound, and
    the caller decides that case before reading any interval.

    A `TypeVar` argument is not read through its bound here. On the wider
    side it stays a plain upper bound, which the relation already orders
    correctly, and on the narrower side the caller reads the bound first.
    """
    if is_super(arg):
        return super_target(arg), top
    if is_exact(arg):
        target = exact_target(arg)
        return target, target
    return tx.Never, arg
