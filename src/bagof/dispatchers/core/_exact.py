"""A hint that matches only a type itself, never one of its subclasses."""

# dependencies
import typing_extensions as tx

# local
from ._compat import ishint


class _ExactMarker:
    """The piece of `Annotated` metadata that tags a hint as [`Exact`][]."""

    def __new__(cls) -> "_ExactMarker":
        if "_INSTANCE" not in cls.__dict__:
            cls._INSTANCE = object.__new__(cls)
        return cls._INSTANCE

    def __repr__(self) -> str:
        return "EXACT"


EXACT = _ExactMarker()
"""The sentinel that an [`Exact`][] hint attaches as `Annotated` metadata."""


# The runtime type of an `Annotated` alias. Building one of these directly
# sidesteps `typing._type_check`, which refuses a bare special form such as
# `Union` or `Literal` as an argument on every version -- so `Exact[Union]`
# has an alias to fall back to when `tx.Annotated[Union, EXACT]` will not
# construct.
_ANNOTATED_ALIAS = type(tx.Annotated[int, EXACT])


if tx.TYPE_CHECKING:
    # To a type checker, `Exact[C]` is just `C`: it is `Annotated[C, EXACT]`,
    # and the checker sees through the metadata. A generic alias, so
    # `Exact[int]` substitutes the type variable and stays a two-argument
    # `Annotated` -- `Exact = Annotated` would make `Exact[int]` a one-argument
    # `Annotated[int]`, which a checker rejects.
    _T = tx.TypeVar("_T")
    Exact: tx.TypeAlias = tx.Annotated[_T, EXACT]
else:

    class Exact:
        """Restrict a dispatch parameter to one type, excluding its subtypes.

        Ordinary dispatch treats a parameter annotated with a class as
        accepting that class and anything derived from it, so that one
        implementation can serve a base type and every subclass alike.
        `Exact[C]` asks for the opposite: an implementation with a
        parameter annotated `#!python Exact[int]` matches an argument
        whose type is exactly `int`, and not one whose type is `bool`,
        even though `bool` is an `int` subclass that an ordinary `int`
        parameter would gladly accept. `Exact` exists for the cases where
        a handler written for a base type would misbehave if a more
        specific subtype reached it.

        A type checker reads `Exact[C]` as plain `C`, since it is defined
        as `#!python Annotated[C, EXACT]` and a checker looks straight
        through `Annotated` metadata to the wrapped type.

        `Exact` composes with [`Type`][typing.Type] and
        [`Hint`][bagof.dispatchers.Hint]. Placing `Exact` inside the
        bracket narrows that position from matching a subtype to matching
        the type itself, so `#!python Type[Exact[int]]` matches the class
        `#!python int` but not `#!python bool`, and
        `#!python Hint[Exact[int]]` matches the hint `#!python int` but
        not `#!python bool`. Writing `Exact` around the whole form instead,
        as `#!python Exact[Type[int]]`, means the same thing and is
        normalised to the inner spelling `#!python Type[Exact[int]]`, which
        is the form to prefer.

        Inside those brackets, [`Super`][bagof.dispatchers.Super] is the
        mirror image of `Exact`: where `#!python Type[Exact[int]]` accepts
        the class `#!python int` alone, `#!python Type[Super[int]]` accepts
        `#!python int` together with every class above it, and
        [`Between`][bagof.dispatchers.Between] accepts the classes between
        two bounds. Neither can be combined with `Exact`, since an exact
        type leaves nothing above or below it to bound.

        !!! example
            ```pycon
            >>> from bagof.dispatchers import Exact
            >>> Exact[int]
            typing.Annotated[int, EXACT]
            ```
        """

        def __class_getitem__(cls, item: tx.Any) -> tx.Any:
            if not ishint(item):
                raise TypeError(
                    f"Exact[...] takes a type hint, got {item!r}."
                )
            # Imported here because `_super` and `_bounds` import this module.
            from ._bounds import (
                endpoint_bound_message,
                find_bound,
                is_between,
                misplaced_bound_message,
                nesting_message,
            )
            from ._super import combination_message, is_super

            if is_super(item):
                raise TypeError(combination_message("Exact", item))
            if is_between(item):
                raise TypeError(nesting_message("Exact", item))
            found = find_bound(item)
            if found is not None:
                # A bound reached through a union or a `TypeVar`.
                raise TypeError(
                    misplaced_bound_message(found, endpoint_bound_message)
                )
            try:
                return tx.Annotated[item, EXACT]
            except TypeError:
                # A bare special form (`Exact[Union]`, `Exact[Literal]`),
                # which `typing._type_check` refuses: build the alias directly.
                return _ANNOTATED_ALIAS(item, (EXACT,))


def is_exact(hint: tx.Any) -> bool:
    """Report whether `hint` was built with [`Exact`][]`[C]`."""
    return any(meta is EXACT for meta in getattr(hint, "__metadata__", ()))


def exact_target(hint: tx.Any) -> tx.Any:
    """Return the wrapped type `C` from an [`Exact`][]`[C]` hint.

    Calling this only makes sense once [`is_exact`][]`(hint)` has already
    been confirmed. The `EXACT` marker and any other `Annotated` metadata
    riding alongside it are discarded, leaving the plain type that the
    dispatch value is compared against.
    """
    return tx.get_args(hint)[0]
