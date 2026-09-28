# Exact types

By default, an overload registered for `int` also accepts `bool`, because
`bool` is a subclass of `int`. `Exact[C]` narrows a hint against that
default: it matches a value only when the value's type is exactly `C`, not
any of its subclasses:

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

To a static type checker, `Exact[int]` is indistinguishable from plain
`int`.
