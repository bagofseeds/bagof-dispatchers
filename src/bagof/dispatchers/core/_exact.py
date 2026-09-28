"""A hint that matches only a type itself, never one of its subclasses."""

# dependencies
import typing_extensions as tx


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


if tx.TYPE_CHECKING:
    # To a type checker, `Exact[C]` is just `C`: it is `Annotated[C, EXACT]`,
    # and the checker sees through the metadata. A generic alias, so
    # `Exact[int]` substitutes the type variable and stays a two-argument
    # `Annotated` -- `Exact = Annotated` would make `Exact[int]` a one-argument
    # `Annotated[int]`, which a checker rejects.
    _T = tx.TypeVar("_T")
    Exact = tx.Annotated[_T, EXACT]  # type: tx.TypeAlias
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

        !!! example
            ```pycon
            >>> from bagof.dispatchers import Exact
            >>> Exact[int]
            typing.Annotated[int, EXACT]
            ```
        """

        def __class_getitem__(cls, item: tx.Any) -> tx.Any:
            return tx.Annotated[item, EXACT]


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
