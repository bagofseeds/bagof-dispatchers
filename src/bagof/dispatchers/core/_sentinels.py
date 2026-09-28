"""Marker objects that stand for the absence of a value."""

# dependencies
import typing_extensions as tx


class Unset:
    """A class restricted to a single instance, for building markers with.

    Every subclass gets an instance of its own the first time it is
    constructed, rather than reusing an ancestor's, so several distinct
    markers can each be written as a one-line subclass without extra
    bookkeeping.
    """

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
Marks a parameter that a caller left out of a call entirely.

!!! note
    `UNSET` means the argument was never given. It is a separate concept
    from [`None`][], which a caller can pass deliberately as a value in
    its own right.
"""
