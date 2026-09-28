---
icon: fontawesome/solid/circle-xmark
---

# When nothing matches

When no registered overload accepts a call, dispatch raises
[`NoMethodError`][bagof.dispatchers.NoMethodError]. Because
`NoMethodError` is also a `TypeError`, code that already catches
`TypeError` keeps working without modification. The error carries the
function's name and the overloads that came closest to matching:

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
