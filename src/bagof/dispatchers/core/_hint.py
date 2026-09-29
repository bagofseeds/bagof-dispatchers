"""A hint whose values are themselves type hints.

[`Hint`][]`[X]` describes the set of type hints that are sub-hints of
`X`, so dispatch can select on a hint passed as an ordinary value rather
than on the value's runtime type. A function parameter annotated
`#!python Hint[int]` matches when the argument is a hint such as
`#!python int` or `#!python bool`, and one annotated
`#!python Hint[tx.Union]` matches when the argument is any union. This is
what lets a dispatcher treat a type hint as data, dispatching on the
relationships between hints that [`issubhint`][] already decides.
"""

# dependencies
import typing_extensions as tx

# local
from ._compat import ishint, spellings

_ANNOTATED_FORMS = spellings("Annotated")

_T_co = tx.TypeVar("_T_co", covariant=True)


class Hint(tx.Generic[_T_co]):
    """A hint matched by the type hints that are sub-hints of its argument.

    Where an ordinary annotation describes the values an argument may
    take, [`Hint`][]`[X]` describes the type *hints* an argument may be.
    An argument matches `#!python Hint[X]` when the argument is itself a
    hint and a sub-hint of `X`, decided by [`issubhint`][]. So
    `#!python Hint[int]` matches `#!python int` and `#!python bool` but
    not `#!python str`, and `#!python Hint[tx.Union]` matches any union.
    This lets a dispatcher select on a hint that is handed to it as a
    value, the way a registry or a schema builder passes types around at
    runtime.

    The argument may itself be [`Exact`][bagof.dispatchers.Exact]`[C]`,
    which narrows the match from "a sub-hint of `C`" to "the hint `C`
    itself". `#!python Hint[Exact[int]]` therefore matches only
    `#!python int`, not `#!python bool`, and `#!python Hint[Exact[tx.Union]]`
    matches only the bare `#!python Union` form.

    The argument may instead be [`Super`][bagof.dispatchers.Super]`[X]`,
    which turns the bound around: `#!python Hint[Super[int]]` matches
    the hint `#!python int` and every hint above it, such as
    `#!python numbers.Integral`, `#!python object`, and `#!python Any`.
    It matches neither `#!python bool`, which sits below `#!python int`,
    nor the unrelated `#!python str`. An argument written as
    [`Between`][bagof.dispatchers.Between]`[L, U]` bounds the match from
    both sides, so `#!python Hint[Between[int, numbers.Real]]` matches
    `#!python int`, `#!python numbers.Integral` and
    `#!python numbers.Real`, but neither `#!python bool`, which sits
    below `#!python int`, nor `#!python object`, which sits above
    `#!python numbers.Real`.

    `Hint` is a marker used only in annotations and is never
    instantiated. A bare `#!python Hint`, written with no argument, is
    read as `#!python Hint[Any]`, the hint matched by every type hint.

    !!! example
        ```pycon
        >>> from bagof.dispatchers import Hint
        >>> from bagof.dispatchers.core import ishintstance
        >>> ishintstance(int, Hint[int])
        True
        >>> ishintstance(bool, Hint[int])
        True
        >>> ishintstance(str, Hint[int])
        False
        >>> ishintstance(tx.Union[int, str], Hint[tx.Union])
        True
        ```
    """

    def __init__(self, *args: tx.Any, **kwargs: tx.Any) -> None:
        raise TypeError(
            "Hint is a type-hint marker used in annotations, such as "
            "Hint[int]; it cannot be instantiated."
        )

    def __class_getitem__(cls, item: tx.Any) -> tx.Any:
        if not ishint(item):
            raise TypeError(
                f"Hint[...] takes a type hint, got {item!r}."
            )
        try:
            return super().__class_getitem__(item)
        except TypeError:  # pragma: no cover  -- reached only before 3.10
            # A bare special form (`Hint[Union]`, `Hint[Literal]`) that the
            # Generic subscription refuses on the oldest interpreters (Python
            # 3.8): build the alias directly. Python 3.10 and later accept a
            # bare special form here, so this fallback is reached only before
            # 3.10.
            return _GENERIC_ALIAS(cls, (item,))


# The runtime type of a `Hint[...]` alias, computed once the class exists.
# `Hint[Union]` and the like construct one of these directly, sidestepping
# the `typing._type_check` that refuses a bare special form as an argument.
_GENERIC_ALIAS = type(Hint[int])


def _unwrap_annotated(hint: tx.Any) -> tx.Any:
    """Strip any `Annotated` wrappers off `hint`, looking through to its
    core.
    """
    while any(tx.get_origin(hint) is form for form in _ANNOTATED_FORMS):
        args = tx.get_args(hint)
        if not args:  # pragma: no cover  -- a subscripted Annotated has args
            return hint
        hint = args[0]
    return hint


def is_hint_form(hint: tx.Any) -> bool:
    """Report whether `hint` is the [`Hint`][] form, bare or parametrised.

    A bare `#!python Hint` and any `#!python Hint[X]` both count, whether
    or not the alias is wrapped in [`Annotated`][typing.Annotated].
    """
    hint = _unwrap_annotated(hint)
    return hint is Hint or tx.get_origin(hint) is Hint


def hint_arg(hint: tx.Any) -> tx.Any:
    """Return the argument `X` of a [`Hint`][]`[X]`, or `Any` for a bare
    `Hint`.

    `hint` is assumed to be a [`Hint`][] form already, as
    [`is_hint_form`][] reports. A bare `#!python Hint` carries no
    argument and stands for `#!python Hint[Any]`, so [`Any`][typing.Any]
    comes back for it.
    """
    args = tx.get_args(_unwrap_annotated(hint))
    return args[0] if args else tx.Any
