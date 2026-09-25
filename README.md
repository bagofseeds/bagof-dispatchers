# bagof-dispatchers

Hint-informed **multiple dispatch** for Python.

`bagof.dispatchers` lets you register several implementations of a function and
picks the right one from the **runtime types of the arguments**, choosing the
most specific matching signature and raising a clear error when the choice is
ambiguous — the way Julia's multiple dispatch does, driven by Python type hints.

Signatures may use the full hint vocabulary — unions, literals, generics,
`TypedDict`, `TypeVar` (with variance), `Exact[...]` for exact-type matching —
and dispatch is **name-aware**: keyword and positional calls both resolve.

## Getting started

Decorate each overload with `@dispatch`. A second `def` of the same name **adds
an overload** rather than replacing the name, and a call runs the most specific
one:

```pycon
>>> from bagof.dispatchers import dispatch
>>> @dispatch
... def area(width: int, height: int) -> int:
...     return width * height
>>> @dispatch
... def area(radius: float) -> float:
...     return 3 * radius * radius
>>> area(3, 4)
12
>>> area(2.0)
12.0
```

The name stays bound to a dispatched function you can keep calling; `@dispatch`
returns it.

## Registering overloads

`@dispatch` reads the signature from the `def` it decorates. To register
something that already exists, or to lay explicit hints over a function's
parameters, register onto the function directly. All of these do the same job —
add an overload:

#### A `def`

```pycon
>>> from bagof.dispatchers import dispatch
>>> @dispatch
... def describe(x: int) -> str:
...     return "an integer"
>>> describe(7)
'an integer'
```

#### An existing callable

```pycon
>>> from bagof.dispatchers import Function
>>> describe = Function("describe")
>>> def a_string(x: str) -> str:
...     return "a string"
>>> _ = describe.register(a_string)
>>> describe("hi")
'a string'
```

#### A class (on its `__init__`)

```pycon
>>> from bagof.dispatchers import Function
>>> box = Function("box")
>>> class IntBox:
...     def __init__(self, value: int):
...         self.value = value
...     def __repr__(self):
...         return "IntBox(%r)" % self.value
>>> _ = box.register(IntBox)
>>> box(5)
IntBox(5)
```

#### Explicit hints

```pycon
>>> from bagof.dispatchers import Function
>>> scale = Function("scale")
>>> @scale.register((int,), {"factor": int})
... def _(value, factor):
...     return value * factor
>>> scale(3, factor=4)
12
```

In the explicit-hints form the hints are laid over the wrapped function's own
parameters, keeping its names, kinds and defaults: **positional hints are a
tuple** (always a tuple, even for one — `(int,)`), **named hints a dict**, and
**keyword arguments are options** — only `priority`, a tie-break between
otherwise equally specific overloads. Named hints go in the dict, never as
keyword arguments.

## When two overloads are equally specific

Dispatch never guesses. If two overloads accept the call and neither is more
specific, it raises `AmbiguousMethodError` and shows the overload to define to
break the tie:

```pycon
>>> from bagof.dispatchers import Function, AmbiguousMethodError
>>> import warnings
>>> combine = Function("combine")
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")   # registration warns about the clash
...     @combine.register((float, object))
...     def _(a, b): return "left"
...     @combine.register((object, float))
...     def _(a, b): return "right"
>>> try:
...     combine(1.0, 2.0)
... except AmbiguousMethodError as error:
...     print(sorted(m.name for m in error.candidates))
['_', '_']
```

Give one overload a higher `priority`, or register the more specific one
(`(float, float)`), to resolve it.

## When nothing matches

A call no overload accepts raises `NoMethodError`, which is a `TypeError` — so
existing `except TypeError:` handling keeps working. It carries the function
name and the closest overloads:

```pycon
>>> from bagof.dispatchers import Function, NoMethodError, DispatchError
>>> area = Function("area")
>>> def rectangle(width: int, height: int) -> int:
...     return width * height
>>> _ = area.register(rectangle)
>>> try:
...     area("triangle")
... except NoMethodError as error:
...     print(error.function, isinstance(error, DispatchError), isinstance(error, TypeError))
area True True
```

## Exact types

By default an overload for `int` also accepts `bool`, since `bool` is a subclass
of `int`. `Exact[C]` accepts a value only when its type is **exactly** `C`:

```pycon
>>> from bagof.dispatchers import dispatch, Exact
>>> @dispatch
... def label(x: int) -> str:
...     return "an integer"
>>> @dispatch
... def label(x: Exact[bool]) -> str:
...     return "a boolean"
>>> label(5)
'an integer'
>>> label(True)
'a boolean'
```

To a type checker `Exact[int]` reads as plain `int`.

## Your own registry

`@dispatch` is a ready-made registry that keeps functions **separate per
module** — the same name in two modules is two independent functions. Build a
`Dispatcher` when you want a **shared** function several modules extend: a
`Dispatcher` keys a function by its qualified name alone, so every module
registering that name into the one instance extends a single function.

Reach a function to hand around through the `functions` namespace. It
**get-or-creates**, so a registry and its callers can name the same function in
any order:

```pycon
>>> from bagof.dispatchers import Dispatcher
>>> registry = Dispatcher()
>>> render = registry.functions.render      # an empty function, for now
>>> @registry
... def render(x: int) -> str:
...     return "int:%d" % x
>>> registry.functions["render"] is render is registry.functions.render
True
>>> render(7)
'int:7'
```

`functions` is a bare namespace with **no methods of its own**, so a function
may be named like anything — `register`, `items`, `map` — without colliding:

```pycon
>>> registry = Dispatcher()
>>> items = registry.functions["items"]
>>> @items.register
... def _(x: int) -> str:
...     return "one item"
>>> registry.functions.items(0)
'one item'
>>> "items" in registry.functions
True
```

Attribute access ignores names beginning with `_`, so a REPL or a tool probing
for dunders never mints an empty function; item access does not, so
`registry.functions["_x"]` still names one. For the same reason, do not register
an anonymous overload (`@dispatch def _`, or a `lambda`) directly on the
module-level `dispatch`: they all key the one `"_"` or `"<lambda>"` name and
collapse into a single function. Give each overload a real name, or overlay
hints onto a named `def`.

## Without a `def`

For a table of type-to-callable with no functions to write, build one from a
mapping. A tuple key gives one positional hint per element:

```pycon
>>> from bagof.dispatchers import Function
>>> size = Function.from_mapping({(int,): abs, (str,): len}, name="size")
>>> size(-3)
3
>>> size("abcd")
4
```

For an exotic shape — positional-only, `*args`, keyword-only — that a plain
tuple of hints cannot spell, pass a `Signature` as the key:

```pycon
>>> from bagof.dispatchers import Function, Signature
>>> total = Function.from_mapping({Signature.from_hints(int, int): lambda a, b: a + b})
>>> total(2, 3)
5
```

## Part of `bagof`

`bagof.dispatchers` is the low-level root of the `bagof` family: it owns the hint
subtype relation and introspection helpers (under `bagof.dispatchers.core`) that
the other bags build on. Its only dependency is
[`typing_extensions`](https://typing-extensions.readthedocs.io/), and it supports
Python 3.8 through current.

The design is written up in
[`docs/rfc/0001-hint-informed-multiple-dispatch.md`](docs/rfc/0001-hint-informed-multiple-dispatch.md).
`bagof` is a namespace package of small, focused tools; see the
[project overview](https://bagofseeds.github.io/bagof/).
