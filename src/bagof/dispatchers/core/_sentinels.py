"""Sentinel values shared across the package."""

# dependencies
import typing_extensions as tx


class Unset:
    """A singleton type whose sole instance stands for "no value given"."""

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # `cls.__dict__`, not `hasattr`: the latter finds an inherited
        # `_INSTANCE`, so a subclass would hand back the base's instance.
        if "_INSTANCE" not in cls.__dict__:
            cls._INSTANCE = object.__new__(cls)
        return cls._INSTANCE

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<UNSET>"

    def __str__(self) -> str:
        return "<UNSET>"


UNSET = Unset()
"""
The sentinel value that marks an argument as not having been supplied.

!!! note
    `UNSET` is distinct from [`None`][], which can be a meaningful value
    in its own right.
"""
