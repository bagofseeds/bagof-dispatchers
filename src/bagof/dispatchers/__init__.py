"""Hint-informed multiple dispatch for Python.

`bagof.dispatchers` selects a registered function by the runtime types of its
arguments, choosing the most specific matching signature and raising a clear
error when the choice is ambiguous. Dispatch is driven by type hints and
understands the full hint vocabulary (unions, literals, generics, `TypedDict`,
`TypeVar`, variance, and more).

Register overloads with the ready-made
[`dispatch`][bagof.dispatchers.dispatch] decorator, or build your own
[`Dispatcher`][bagof.dispatchers.Dispatcher]; call the resulting function and
the most specific overload runs. [`Exact`][bagof.dispatchers.Exact] matches a
type without its subclasses. The hint-level subtype relation and introspection
helpers -- the shared root the rest of the `bagof` family builds on -- live
under `bagof.dispatchers.core`. See the design RFC in `docs/rfc/` for the
model.
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
