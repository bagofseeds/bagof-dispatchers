"""Sentinel values shared across the package."""

# dependencies
import typing_extensions as tx


class Unset:
    """A singleton whose only value is "no value was set"."""

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
A value that indicates that an argument was not set.

!!! note
    This is different from [`None`][], which may be a valid value.
"""
