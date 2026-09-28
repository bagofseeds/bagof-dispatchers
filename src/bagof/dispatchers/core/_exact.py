"""`Exact[C]`, a hint that matches a type but excludes its subclasses."""

# dependencies
import typing_extensions as tx


class _ExactMarker:
    """The metadata value that marks an [`Exact`][] hint."""

    def __new__(cls) -> "_ExactMarker":
        if "_INSTANCE" not in cls.__dict__:
            cls._INSTANCE = object.__new__(cls)
        return cls._INSTANCE

    def __repr__(self) -> str:
        return "EXACT"


EXACT = _ExactMarker()
"""The metadata value carried by an [`Exact`][] hint."""


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
        """A hint that matches a type but not its subclasses.

        `Exact[C]` describes a value whose type is precisely `C`, so an
        overload registered for `#!python Exact[int]` does not fire for a
        `#!python bool`, even though `bool` is a subclass of `int` and an
        ordinary `#!python int` parameter would accept it. This is the
        reverse of the usual multiple-dispatch situation, where a base
        class is registered once and its subclasses share the overload:
        `Exact` is for the times a handler for a base type must not run
        on a more specific one.

        A type checker sees straight through `Exact[C]` to `C`, because
        `Exact[C]` is defined as `#!python Annotated[C, EXACT]`.

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
    """Report whether `hint` is an [`Exact`][]`[C]`."""
    return any(meta is EXACT for meta in getattr(hint, "__metadata__", ()))


def exact_target(hint: tx.Any) -> tx.Any:
    """Return the type `C` that an [`Exact`][]`[C]` hint wraps.

    The caller must already know that [`is_exact`][]`(hint)` holds. The
    `EXACT` marker and any other metadata attached to the hint are
    dropped, leaving the plain type.
    """
    return tx.get_args(hint)[0]
