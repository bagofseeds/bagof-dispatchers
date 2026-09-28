"""A single registered implementation: [`Method`][].

A [`Method`][] pairs one function with the [`Signature`][] that
dispatch reads off it, an optional `priority` used to break ties, and
the location where the function was defined, so that an error can point
back at the exact `#!python def`.
"""

# stdlib
import inspect
import os

# dependencies
import typing_extensions as tx

# local
from ._signature import Signature, _render_parameters

__all__ = ["Method"]


class Method:
    """One implementation registered under a function's name.

    A `Method` wraps a callable together with the [`Signature`][] that
    describes its parameters. Calling the method calls the wrapped
    function; its [`repr`][] shows the named signature and the location
    where it was defined, which is what a dispatch error uses to point
    a reader at the right line.

    !!! example
        ```pycon
        >>> def area(shape, scale=1.0): ...
        >>> method = Method(area)
        >>> method.priority
        0
        ```

    Parameters
    ----------
    function
        The callable this method runs.
    signature
        The signature dispatch compares candidates by. When omitted, it
        is read from `function` with [`Signature.from_callable`][].
    priority
        A tie-break applied before the type-based one: between two
        methods that are otherwise equally specific, the one with the
        higher priority wins. Defaults to `0`.

    Attributes
    ----------
    signature : Signature
        The method's signature.
    function : Callable
        The wrapped callable.
    priority : int
        The tie-break priority.
    filename : str
        The file the function was defined in, or `"<module>"` when it
        has no source file.
    lineno : int or None
        The line the function's `#!python def` starts on, if known.
    """

    __slots__ = (
        "function",
        "signature",
        "priority",
        "filename",
        "lineno",
    )

    def __init__(
        self,
        function: tx.Callable[..., tx.Any],
        signature: tx.Optional[Signature] = None,
        priority: int = 0,
    ) -> None:
        self.function = function
        self.signature = (
            signature
            if signature is not None
            else Signature.from_callable(function)
        )
        self.priority = priority
        self.filename, self.lineno = _source_location(function)

    def __call__(self, *args: tx.Any, **kwargs: tx.Any) -> tx.Any:
        """Call the wrapped function with the given arguments."""
        return self.function(*args, **kwargs)

    @property
    def name(self) -> str:
        """The name of the wrapped function."""
        return getattr(self.function, "__name__", "<function>")

    @property
    def location(self) -> str:
        """A short `file:line` string for where the function was defined."""
        base = os.path.basename(self.filename)
        if self.lineno is None:
            return base
        return f"{base}:{self.lineno}"

    def describe(
        self, highlight: tx.Optional[tx.Collection[tx.Any]] = None
    ) -> str:
        """Render the named signature and its definition site for an error.

        The rendering matches [`repr`][], except that each slot named
        in `highlight` has its hint marked with a leading `#!python !`,
        pointing out the offending argument in a dispatch error. A slot
        is named by its parameter name, or by `Parameter.VAR_POSITIONAL`
        or `Parameter.VAR_KEYWORD` for the `#!python *args` and
        `#!python **kwargs` catch-alls.
        """
        body = _render_parameters(self.signature, highlight)
        return f"{self.name}({body}) @ {self.location}"

    def __repr__(self) -> str:
        return self.describe()

    def __eq__(self, other: tx.Any) -> bool:
        if not isinstance(other, Method):
            return NotImplemented
        return (
            self.function is other.function
            and self.priority == other.priority
        )

    def __hash__(self) -> int:
        return hash((id(self.function), self.priority))


def _source_location(
    function: tx.Callable[..., tx.Any],
) -> tx.Tuple[str, tx.Optional[int]]:
    """Return the file and line a callable was defined at, best effort."""
    try:
        filename = inspect.getsourcefile(function) or "<module>"
    except (TypeError, OSError):
        # `TypeError` -- a callable with no source module (a builtin, a C
        # function). `OSError` -- a class or function whose module has no
        # `__file__`, as in the REPL, a Jupyter cell, `python -c`, or code
        # built with `exec`; there is no source file to name.
        filename = "<module>"
    code = getattr(function, "__code__", None)
    lineno = getattr(code, "co_firstlineno", None)
    return filename, lineno
