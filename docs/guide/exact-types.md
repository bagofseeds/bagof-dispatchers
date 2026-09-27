---
icon: fontawesome/solid/bullseye
---

# Exact types

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
