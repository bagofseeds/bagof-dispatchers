---
icon: fontawesome/solid/list-ul
---

# Registering overloads

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
