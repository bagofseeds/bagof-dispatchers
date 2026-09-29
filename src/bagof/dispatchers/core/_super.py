"""A lower bound: a type and everything above it, on a value, a class or a
hint.
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

    A class is named by its `__name__`, a forward reference by the name
    it was written with, and any other hint by its string form with the
    `typing.` and `typing_extensions.` prefixes removed, so that a message
    reads `Super[int]` or `Super[List[int]]`.
    """
    if isinstance(hint, type):
        return hint.__name__
    if isinstance(hint, str):
        return hint
    forward = getattr(hint, "__forward_arg__", None)
    if isinstance(forward, str):
        return forward
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
        "nothing above it to bound. On a value, write Exact[C] for a value "
        "whose class is exactly C, or Super[C] for a value whose class is C "
        "or a class above it. Inside Type[...], write Type[Exact[C]] for "
        "exactly C, or Type[Super[C]] for C and anything above it; the same "
        "holds inside Hint[...]."
    )


if tx.TYPE_CHECKING:
    # A type checker must accept every call the runtime accepts, and a
    # `Type[Super[Dog]]` parameter accepts `Animal` and `object`. Reading
    # `Super[C]` as `C`, the way `Exact[C]` is read, would reject those calls,
    # so `Super[C]` is spelled `Union[C, Any]` instead: the type variable is
    # used, which a checker requires of a generic alias, and the `Any` member
    # makes `Type[Super[C]]` accept any class object. The aliases are declared
    # with a PEP 613 annotation rather than a PEP 484 type comment, which
    # pyright does not read; this branch never runs, so the annotation is safe
    # on 3.8.
    _T = tx.TypeVar("_T")
    Super: tx.TypeAlias = tx.Union[_T, tx.Any]
    SuperType: tx.TypeAlias = tx.Type[Super[_T]]
    SuperHint: tx.TypeAlias = Hint[Super[_T]]
else:

    class Super:
        """Bound a hint from below: accept a type and everything above it.

        An ordinary annotation places an upper bound on what a parameter
        accepts, so a parameter annotated `Animal` accepts an instance of
        `Animal` or of any class derived from it. `Super[C]` places a
        lower bound instead. On a value parameter, `#!python x: Super[Dog]`
        accepts a value whose class is `Dog` or a class that `Dog` derives
        from, such as `Animal` or `object`. It refuses an instance of a
        subclass of `Dog`, such as `Puppy`, and an instance of an unrelated
        class. Inside [`Type`][typing.Type], where the value passed is
        itself a class, `#!python Type[Super[Dog]]` accepts the class `Dog`
        and every class that `Dog` derives from. Inside [`Hint`][], where
        the value passed is a type hint, `#!python Hint[Super[int]]`
        accepts the hint `#!python int` together with every hint above it,
        such as `#!python numbers.Integral`, `#!python Union[int, str]` and
        `#!python Any`, but not `#!python bool`, which sits below it. As
        the type argument of an invariant generic, where a plain argument
        means exactly that type, `#!python List[Super[int]]` accepts a
        list declared to hold `#!python int` or a type above it, such as
        `#!python numbers.Integral` or `#!python object`, and refuses one
        declared to hold `#!python bool`. In each position `Super[C]` is
        the range that runs from `C` up to the top of the order, and
        [`Between`][bagof.dispatchers.Between] names both ends of such a
        range when the top should be bounded as well.

        On a value, `Super` belongs to a family of hints that compare the
        value's own class against a range of classes. A plain `C` accepts
        a value whose class is `C` or lies below it, and
        [`Exact`][bagof.dispatchers.Exact]`[C]` accepts a value whose class
        is `C` itself. `Super[C]` accepts a value whose class is `C` or lies
        above it, and `Between[L, U]` accepts a value whose class lies
        between `L` and `U`. Writing `Exact`, `Super` or `Between` on a
        value is a deliberate departure from substitutability, the usual
        expectation that a function written for a class also accepts the
        instances of its subclasses. A method for `#!python Super[Dog]`
        handles `Dog` and its ancestors and leaves every subclass of `Dog`
        to other methods, which suits a general fallback that should never
        capture the specialised classes. Because the bound is compared
        against the value's class, it must be a hint that a class can be
        compared against, such as a class, a union of classes, `Never` or
        `Any`. A hint that reads the value itself, such as a `Literal`, a
        `TypedDict` or a parametrised generic, is refused as the bound of a
        value. Inside `Type[...]` or `Hint[...]`, and as a type argument,
        the bound is compared against another hint, so any hint can be a
        bound there.

        As a type argument, `Super[C]` stands for every parametrisation
        whose argument lies at or above `C`. A value that declares its
        arguments, such as an instance of
        `#!python class IntList(List[int])`, matches when its declared
        argument lies in that range, and a value that declares nothing,
        such as a plain `#!python [1]`, matches as it matches every
        parametrisation of `#!python List`. Each bound names its own
        range and does not reach through the position around it, so
        `#!python List[List[Super[int]]]` accepts a list declared to hold
        `#!python List[Super[int]]` and not one declared to hold
        `#!python List[int]`. A lower bound is meaningful at an invariant
        position. At a covariant position, such as the argument of
        `#!python Sequence`, the variance already accepts every
        parametrisation below the argument, so a lower bound would accept
        every sequence, and it is refused with a message naming the plain
        spelling to write instead. At a contravariant position the
        variance already reads the argument as a lower bound, so
        `Super[C]` there means the same as plain `C` and is read that way.
        A bound is also refused as an element of a `#!python Tuple` and in
        the signature of a `#!python Callable`, whose positions have a
        fixed variance of their own.

        `Super` is written inside the bracket, as in
        `#!python Type[Super[C]]`. Writing it around the whole form, as in
        `#!python Super[Type[C]]`, means the same thing and is normalised
        to the inner spelling. [`SuperType`][] and [`SuperHint`][] are
        shorter aliases for the two spellings.

        Lower bounds are ordered in the opposite direction to the classes
        they name. `#!python Super[Animal]` is more specific than
        `#!python Super[Dog]`, because every class above `Animal` is also
        above `Dog`, so the classes above `Animal` form the smaller set.
        `#!python Exact[Dog]` sits below both `Dog` and
        `#!python Super[Dog]`, since the single class `Dog` belongs to each
        of them. A plain `Animal` and `#!python Super[Dog]` are not ordered
        against each other, yet a value of class `Dog` or `Animal` matches
        both, so registering methods for both warns about the ambiguity. A
        third method for `#!python Exact[Dog]` settles the call with a
        `Dog`, and an explicit `priority` on either method settles every
        call the two share. The same order holds inside `Type` and `Hint`,
        where `#!python Type[Exact[Dog]]` sits below both
        `#!python Type[Dog]` and `#!python Type[Super[Dog]]`. As a type
        argument, one range is more specific than another when it lies
        inside it, and a plain argument counts as a range holding only
        itself, so `#!python List[Dog]` and `#!python List[Super[Animal]]`
        are both more specific than `#!python List[Super[Dog]]`.

        `Super` and `Exact` cannot be combined, in either order, because
        an exact type has nothing above it to bound, and neither of them
        can be combined with `Between`. A type checker reads `Super[C]` as
        `#!python Union[C, Any]`, which accepts every call the runtime
        accepts. As a type argument of an invariant generic, mypy reads
        that union as a lower bound, as the runtime does, so it accepts a
        `#!python list[int]` or a `#!python list[object]` passed to a
        `#!python List[Super[int]]` parameter and rejects a
        `#!python list[bool]`. Inside the function body, a checker treats a
        parameter annotated `#!python Super[Dog]` as a `Dog` and checks
        attribute access against `Dog`, although the value may be an
        `Animal`. When that difference matters, the value is best treated
        as an `object` in the body.

        !!! example
            ```pycon
            >>> from typing import Type
            >>> from bagof.dispatchers import Super
            >>> from bagof.dispatchers.core import ishintstance
            >>> class Animal: pass
            >>> class Dog(Animal): pass
            >>> class Puppy(Dog): pass
            >>> Super[int]
            typing.Annotated[int, SUPER]
            >>> ishintstance(Animal(), Super[Dog])
            True
            >>> ishintstance(Puppy(), Super[Dog])
            False
            >>> ishintstance(object, Type[Super[int]])
            True
            >>> ishintstance(bool, Type[Super[int]])
            False
            >>> from typing import List
            >>> class IntList(List[int]): pass
            >>> class BoolList(List[bool]): pass
            >>> ishintstance(IntList(), List[Super[int]])
            True
            >>> ishintstance(BoolList(), List[Super[int]])
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
            # Imported here because `_bounds` imports this module.
            from ._bounds import (
                endpoint_bound_message,
                find_bound,
                is_between,
                misplaced_bound_message,
                nesting_message,
            )

            if is_between(item):
                raise TypeError(nesting_message("Super", item))
            if is_super(item):
                # `Super[Super[C]]` bounds from below by the same `C`.
                return item
            found = find_bound(item)
            if found is not None:
                # A bound reached through a union or a `TypeVar`.
                raise TypeError(
                    misplaced_bound_message(found, endpoint_bound_message)
                )
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
        `#!python SuperHint[int]` accepts the hint `#!python int` and
        every hint that `#!python int` is a sub-hint of, such as
        `#!python numbers.Integral` or `#!python Any`, but not the
        narrower `#!python bool`. The alias always needs its
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
    """Report whether `hint` is [`Super`][], [`SuperType`][] or
    [`SuperHint`][] written without a bound.

    Each of these names a lower bound only once it is subscripted, so an
    unsubscripted one is refused wherever a hint is read, with
    [`bare_super_message`][].
    """
    return hint is Super or hint is SuperType or hint is SuperHint


def bare_super_message(hint: tx.Any) -> str:
    """Compose the error for [`Super`][], [`SuperType`][] or
    [`SuperHint`][] written without a bound.

    `hint` is one that [`is_bare_super`][] reports. The message names the
    form and the spellings that give it a bound.
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
    return (
        "Super needs a bound: write Super[C] to accept a value whose class "
        "is C or a class above it, Type[Super[C]] to accept the class C or "
        "any class above it, or Hint[Super[C]] to accept the hint C or any "
        "hint above it."
    )
