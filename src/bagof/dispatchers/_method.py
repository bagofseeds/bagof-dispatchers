"""A single registered implementation: [`Method`][].

A [`Method`][bagof.dispatchers._method.Method] pairs one function with the
[`Signature`][bagof.dispatchers.Signature] dispatch reads off it, an optional
`priority` for breaking ties, and where it was defined -- so an error can point
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
    """One implementation registered against a function name.

    A method wraps a callable together with the
    [`Signature`][bagof.dispatchers.Signature] its parameters describe.
    Calling the method calls the wrapped function; its
    [`repr`][repr] shows the named signature and where it was defined, which
    is what dispatch errors use to point a reader at the right line.

    Parameters
    ----------
    function
        The callable this method runs.
    signature
        The signature dispatch compares. When omitted it is read from
        `function` with
        [`Signature.from_callable`][bagof.dispatchers.Signature.from_callable].
    priority
        A tie-break applied before the type-based one: a higher priority wins
        between two otherwise equally specific methods. Defaults to `0`.

    Attributes
    ----------
    signature : Signature
        The method's signature.
    function : Callable
        The wrapped callable.
    priority : int
        The tie-break priority.
    filename : str
        The file the function was defined in, or `"<module>"` when it has no
        source file.
    lineno : int or None
        The line the function's `#!python def` starts on, if known.

    !!! example
        ```pycon
        >>> def area(shape, scale=1.0): ...
        >>> method = Method(area)
        >>> method.priority
        0
        ```
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
        """Call the wrapped function."""
        return self.function(*args, **kwargs)

    @property
    def name(self) -> str:
        """The wrapped function's name."""
        return getattr(self.function, "__name__", "<function>")

    @property
    def location(self) -> str:
        """A short `file:line` where the function was defined."""
        base = os.path.basename(self.filename)
        if self.lineno is None:
            return base
        return f"{base}:{self.lineno}"

    def describe(
        self, highlight: tx.Optional[tx.Collection[tx.Any]] = None
    ) -> str:
        """The named signature and where it was defined, for an error.

        The rendering matches [`repr`][repr], except that each slot named in
        `highlight` is marked with a leading `#!python !` on its hint -- the
        offending argument in a dispatch error. A slot is named by its
        parameter name, or by `Parameter.VAR_POSITIONAL` /
        `Parameter.VAR_KEYWORD` for the `#!python *args` / `#!python **kwargs`
        catch-alls.
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
    """The file and line a callable was defined at, best effort."""
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
