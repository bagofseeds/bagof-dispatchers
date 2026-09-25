"""`Exact[C]`: dispatch on a type but not its subclasses."""

# dependencies
import typing_extensions as tx


class _ExactMarker:
    """The metadata object that marks an [`Exact`][] hint."""

    def __new__(cls) -> "_ExactMarker":
        if "_INSTANCE" not in cls.__dict__:
            cls._INSTANCE = object.__new__(cls)
        return cls._INSTANCE

    def __repr__(self) -> str:
        return "EXACT"


EXACT = _ExactMarker()
"""The sentinel that marks an [`Exact`][] hint's metadata."""


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
        """Match a type exactly, not its subclasses.

        `Exact[C]` describes a value whose type is exactly `C` -- `True` is
        not an `#!python Exact[int]`, even though it is an `#!python int`.
        It is the reverse of the usual need: a base class whose subclasses
        each want their own method, or an `#!python int` handler that must
        not fire for `#!python bool`.

        `Exact[C]` is spelled `#!python Annotated[C, EXACT]`, so a type
        checker still sees plain `C`.

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
    """Whether `hint` is an [`Exact`][]`[C]`."""
    return any(meta is EXACT for meta in getattr(hint, "__metadata__", ()))


def exact_target(hint: tx.Any) -> tx.Any:
    """The `C` inside an [`Exact`][]`[C]`.

    Assumes [`is_exact`][]`(hint)` -- returns the type the exactness
    applies to, with the `EXACT` marker and any other metadata dropped.
    """
    return tx.get_args(hint)[0]
