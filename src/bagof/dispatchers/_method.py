"""One implementation registered under a dispatched function's name.

A [`Method`][] wraps a single callable together with the
[`Signature`][] dispatch reads its parameter types from, and with an
optional priority for breaking ties that type alone cannot settle. It
also records the location where the callable was defined, so that a
dispatch error can point a reader straight at the `#!python def`
responsible.
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
    """A callable paired with the signature dispatch selects it by.

    Calling a `Method` calls the function it wraps, so a `Method` can
    stand in for the function itself once it has been registered. Its
    [`repr`][] prints the function's name together with its parameters
    and the file and line it was defined on, and a dispatch error reuses
    that same rendering to point a reader at the exact candidate that
    did, or did not, match.

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
        The callable this method runs when it is called or selected.
    signature
        The signature dispatch reads the parameter types from to decide
        whether this method matches a call, and how specific it is
        compared to another. When omitted, it is obtained from
        `function` with [`Signature.from_callable`][].
    priority
        A tie-break considered ahead of specificity: given two methods
        that a call matches equally well by type, the one with the
        higher priority is chosen. Defaults to `0`.

    Attributes
    ----------
    signature : Signature
        The method's signature.
    function : Callable
        The wrapped callable.
    priority : int
        The tie-break priority given at construction.
    filename : str
        The file the wrapped callable was defined in, or `"<module>"`
        when no source file could be found for it.
    lineno : int or None
        The line where the callable's `#!python def` begins, when that
        could be determined.
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
        """Run the wrapped function, forwarding every argument to it."""
        return self.function(*args, **kwargs)

    @property
    def name(self) -> str:
        """The wrapped function's `__name__`, or a placeholder if it has
        none.
        """
        return getattr(self.function, "__name__", "<function>")

    @property
    def location(self) -> str:
        """A short `file:line` string pointing at the function's definition."""
        base = os.path.basename(self.filename)
        if self.lineno is None:
            return base
        return f"{base}:{self.lineno}"

    def describe(
        self, highlight: tx.Optional[tx.Collection[tx.Any]] = None
    ) -> str:
        """Render the method's name, signature and definition site.

        This is the same rendering [`repr`][] produces, except that the
        hint of each parameter named in `highlight` is prefixed with a
        `#!python !`, marking it as the argument responsible for the
        method not matching in a dispatch error. A parameter is named
        either by its own name or, for the catch-all parameters, by
        `Parameter.VAR_POSITIONAL` or `Parameter.VAR_KEYWORD`.
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
    """Find the file and line a callable was defined at, where possible.

    A callable with no accessible source, such as a builtin or one
    defined interactively, has no meaningful location; `filename` falls
    back to `"<module>"` and `lineno` to `None` in that case, rather than
    letting the lookup raise.
    """
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
