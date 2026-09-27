---
icon: fontawesome/solid/circle-xmark
---

# When nothing matches

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
