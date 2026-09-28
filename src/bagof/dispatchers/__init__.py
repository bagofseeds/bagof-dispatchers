"""Hint-informed multiple dispatch for Python.

`bagof.dispatchers` chooses which of several registered implementations
of a function to run by looking at the runtime types of the arguments a
call actually receives. Among the implementations whose declared types
accept those arguments, it picks the most specific one, and raises a
clear error when two implementations are equally specific and neither
one is more applicable than the other. Dispatch is driven by ordinary
type hints, and understands the full hint vocabulary: unions, literals,
generics, `TypedDict`, `TypeVar`, variance, and more.

Register overloads with the [`dispatch`][] decorator, or build a
[`Dispatcher`][] of your own; calling the resulting function then runs
whichever overload matches best. [`Exact`][] lets an overload match a
type while excluding its subclasses. The hint-level subtype relation and
the introspection helpers it is built from, which are the foundation the
rest of the `bagof` family builds on, live in `bagof.dispatchers.core`.
The design RFC under `docs/rfc/` describes the underlying model in full.
"""

# local
from ._dispatcher import Dispatcher, dispatch
from ._errors import AmbiguousMethodError, DispatchError, NoMethodError
from ._function import Function
from ._method import Method
from ._signature import Parameter, Signature
from .core._exact import Exact

__all__ = [
    "dispatch",
    "Dispatcher",
    "Function",
    "Method",
    "Signature",
    "Parameter",
    "DispatchError",
    "NoMethodError",
    "AmbiguousMethodError",
    "Exact",
]
