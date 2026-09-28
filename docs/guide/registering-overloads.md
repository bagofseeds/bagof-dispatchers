---
icon: fontawesome/solid/list-ul
---

# Registering overloads

`@dispatch` reads its signature from the `def` it decorates, which covers the
common case of writing a fresh function for each overload. Registering
directly on the underlying
[`Function`][bagof.dispatchers.Function] covers the rest: an implementation
that already exists elsewhere, a class whose constructor should be
dispatched on, or a function whose own parameters carry no hints of their
own. Each of the forms below adds one overload.

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

A [`Function`][bagof.dispatchers.Function] built directly, and registered on
by calling [`register`][bagof.dispatchers.Function.register], reaches the
same overloads that `@dispatch` would have built, without requiring the
implementation to be defined as a fresh `def`:

```pycon
>>> from bagof.dispatchers import Function
>>> describe = Function("describe")
>>> def a_string(x: str) -> str:
...     return "a string"
>>> _ = describe.register(a_string)
>>> describe("hi")
'a string'
```

#### A class, on its constructor

Registering a class dispatches on its constructor's parameters, so calling
the resulting function builds an instance:

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

`register` also accepts hints directly, instead of an implementation. Passed
a tuple, a dict, or both, it returns a decorator that lays those hints over
the wrapped function's own parameters, keeping that function's names,
argument kinds, and defaults:

```pycon
>>> from bagof.dispatchers import Function
>>> scale = Function("scale")
>>> @scale.register((int,), {"factor": int})
... def _(value, factor):
...     return value * factor
>>> scale(3, factor=4)
12
```

Positional hints are always a tuple, even for a single hint (`(int,)`), and
named hints are always a dict. A keyword argument to `register` is not a
hint: the only one it recognizes is `priority`, which breaks a tie between
overloads that would otherwise be equally specific. Named hints belong in
the dict; passing one as a keyword argument to `register` is an error.
