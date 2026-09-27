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

## Variance

A generic's type argument is compared by the **variance its parameter
declares**, following the typing spec. A covariant `TypeVar`
(`covariant=True`) orders a container the same way as its argument, so a more
specific argument gives a more specific — and preferred — overload:

```pycon
>>> import typing_extensions as tx
>>> from bagof.dispatchers import dispatch
>>> T_co = tx.TypeVar("T_co", covariant=True)
>>> class Source(tx.Generic[T_co]):
...     pass
>>> @dispatch
... def read(s: Source[int]) -> str:
...     return "ints"
>>> @dispatch
... def read(s: Source[bool]) -> str:
...     return "bools"
>>> read(Source())          # Source[bool] is the more specific overload
'bools'
```

A contravariant `TypeVar` (`contravariant=True`) reverses that order, so the
*wider* argument wins instead:

```pycon
>>> T_contra = tx.TypeVar("T_contra", contravariant=True)
>>> class Sink(tx.Generic[T_contra]):
...     pass
>>> @dispatch
... def write(s: Sink[int]) -> str:
...     return "ints"
>>> @dispatch
... def write(s: Sink[bool]) -> str:
...     return "bools"
>>> write(Sink())           # Sink[int] is the more specific overload
'ints'
```

A plain `TypeVar` — and every mutable standard-library container, such as
`list` — is **invariant**: `List[int]` and `List[bool]` describe different,
incomparable lists, so two such overloads both match a list and neither is
more specific. The call is ambiguous:

```pycon
>>> from bagof.dispatchers import Function, AmbiguousMethodError
>>> handle = Function("handle")
>>> @handle.register
... def _ints(xs: tx.List[int]) -> str:
...     return "ints"
>>> @handle.register
... def _bools(xs: tx.List[bool]) -> str:
...     return "bools"
>>> try:
...     handle([True, False])   # both match; neither is more specific
... except AmbiguousMethodError:
...     print("ambiguous")
ambiguous
```

Give one overload a higher `priority` (`@handle.register(priority=1)`) to
break the tie. Registering both is legitimate — a plain list declares no type
arguments, so `List[int]` and `List[bool]` genuinely both apply to it, while a
value that *does* declare them picks one (below) — so this is *not* reported
at registration; the ambiguity surfaces at the call, where a `priority`
resolves it.

A value that declares its type arguments is dispatched on them. An instance of
a class written against a parametrised base, and an instance built from a
parametrised user generic, both say what they hold:

```pycon
>>> class IntList(tx.List[int]):
...     pass
>>> handle(IntList([1, 2]))       # declares List[int]
'ints'
>>> T = tx.TypeVar("T")
>>> class Box(tx.Generic[T]):
...     pass
>>> unbox = Function("unbox")
>>> @unbox.register
... def _int(b: Box[int]) -> str:
...     return "int"
>>> @unbox.register
... def _str(b: Box[str]) -> str:
...     return "str"
>>> unbox(Box[int]()), unbox(Box[str]())
('int', 'str')
```

Dispatch never looks inside a container, so a value that declares nothing —
a plain `[1, 2]`, or a `Box()` built without arguments — still matches every
parameterisation, as the ambiguous `handle([True, False])` above shows.

`Sequence`, `frozenset`, `Iterable` and the other read-only containers are
covariant, so a `Sequence[bool]` overload stays more specific than a
`Sequence[int]` one — only the mutable containers are invariant. A generic
whose positions have *different* signs — such as
`Generator[Y_co, S_contra, R_co]` — can leave two parameterisations
incomparable (and so ambiguous): `Any` is the top of the covariant yield slot
but the *bottom* of the contravariant send slot, so `Generator[int, None,
None]` and `Generator[int, Any, Any]` order neither way. Variance changes only
how hints are **ordered**; which values match is decided by what each value
declares, as above.

If you use `bagof.hints`, its `typevars.co`, `typevars.contra` and
`typevars.inv` variables carry the matching flags and are read the same way.

## Dispatching on numbers

Dispatch matches the **runtime type** of a value, so a `float` overload accepts
only actual `float` values. An `int` is **not** a `float`: Python's numeric
tower (`int` → `float` → `complex`) is a static type-checking convention, and
dispatch does not apply it. Annotate a parameter `float` only when you mean an
actual float and nothing else:

```pycon
>>> from bagof.dispatchers import dispatch, NoMethodError
>>> @dispatch
... def halve(x: float) -> float:
...     return x / 2
>>> halve(2.0)
1.0
>>> try:
...     halve(3)          # an int is not a float
... except NoMethodError:
...     print("no overload for int")
no overload for int
```

To accept **any real number**, annotate with `numbers.Real` instead of `float`.
It matches `int`, `float`, `bool`, `fractions.Fraction`, and third-party reals
such as NumPy scalars — every type registered under the `numbers` abstract base
classes:

```pycon
>>> from bagof.dispatchers import dispatch
>>> from numbers import Real
>>> @dispatch
... def classify(x: Real) -> str:
...     return "a real number"
>>> classify(3)           # int
'a real number'
>>> classify(2.5)         # float
'a real number'
```

Use `numbers.Integral` for integers (`int`, `bool`, and NumPy ints),
`numbers.Complex` to also accept `complex`, or spell the set out explicitly with
`Union[int, float]`.

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
