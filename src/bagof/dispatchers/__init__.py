"""Multiple dispatch for Python, driven by ordinary type hints.

A function built by `bagof.dispatchers` can hold several implementations
side by side, each written for a different combination of argument
types. Calling the function inspects the runtime types of the arguments
actually given, finds every implementation whose parameter hints accept
them, and runs whichever of those candidates describes the arguments
most specifically. When two candidates are equally specific and neither
is more applicable than the other, the call fails with a clear error
instead of picking one arbitrarily. The specificity comparison is not
limited to plain classes: unions, literals, generic containers,
`TypedDict`, `TypeVar` and the rest of the typing vocabulary all take
part in it.

The [`dispatch`][] decorator is the usual way to add an implementation
to such a function. [`Dispatcher`][] is the object underneath it, for
building one directly rather than through the decorator. [`Exact`][]
marks a parameter that must match a type precisely, so that an
implementation written for a base class does not also catch its
subclasses. [`Super`][] is its mirror image inside
[`Type`][typing.Type] and [`Hint`][]: it accepts a class or a hint
together with everything above it, rather than everything below. The
subtype relation over hints, and the lower-level introspection it is
built from, live separately in `bagof.dispatchers.core`, since the rest
of the `bagof` family depends on that comparison without needing
dispatch itself. The reasoning behind the model is written up in the
design RFC under `docs/rfc/`.
"""

# local
from ._dispatcher import Dispatcher, dispatch
from ._errors import AmbiguousMethodError, DispatchError, NoMethodError
from ._function import Function
from ._method import Method
from ._signature import Parameter, Signature
from .core._exact import Exact
from .core._hint import Hint
from .core._super import Super, SuperHint, SuperType

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
    "Hint",
    "Super",
    "SuperType",
    "SuperHint",
]
