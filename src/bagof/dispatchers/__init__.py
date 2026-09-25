"""Hint-informed multiple dispatch for Python.

`bagof.dispatchers` selects a registered function by the runtime types of its
arguments, choosing the most specific matching signature and raising a clear
error when the choice is ambiguous. Dispatch is driven by type hints and
understands the full hint vocabulary (unions, literals, generics,
``TypedDict``, ``TypeVar``, variance, and more).

The rest of the public API (``dispatch``, ``Dispatcher``, ``Function``,
``Method``, ``Signature``, ``Parameter`` and the dispatch errors) is added in
later phases; the hint subtype relation and introspection helpers live under
``bagof.dispatchers.core``. See the design RFC in ``docs/rfc/`` for the model.
"""

# local
from .core._exact import Exact

__all__ = [
    "Exact",
]
