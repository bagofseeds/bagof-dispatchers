# bagof-dispatchers

Hint-informed **multiple dispatch** for Python.

`bagof.dispatchers` lets you register several implementations of a function and
picks the right one from the **runtime types of the arguments**, choosing the
most specific matching signature and raising a clear error when the choice is
ambiguous — the way Julia's multiple dispatch does, driven by Python type hints.

```python
from bagof.dispatchers import dispatch

@dispatch
def area(shape: Circle) -> float:
    return 3.14159 * shape.r ** 2

@dispatch
def area(shape: Rect) -> float:
    return shape.w * shape.h

area(Circle(1))   # -> 3.14159
```

Signatures may use the full hint vocabulary — unions, literals, generics,
`TypedDict`, `TypeVar` (with variance), `Exact[...]` for exact-type matching —
and dispatch is **name-aware** (keyword and positional calls both resolve).

`bagof.dispatchers` is the low-level root of the `bagof` family: it owns the hint
subtype relation and introspection helpers (under `bagof.dispatchers.core`) that
the other bags build on. Its only dependency is
[`typing_extensions`](https://typing-extensions.readthedocs.io/), and it supports
Python 3.8 through current.

> **Status:** in development. The design is written up in
> [`docs/rfc/0001-hint-informed-multiple-dispatch.md`](docs/rfc/0001-hint-informed-multiple-dispatch.md);
> the public API is being implemented in phases.

## Part of `bagof`

`bagof` is a namespace package of small, focused tools. See the
[project overview](https://bagofseeds.github.io/bagof/).
