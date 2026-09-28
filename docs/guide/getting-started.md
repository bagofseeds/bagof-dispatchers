---
icon: fontawesome/solid/rocket
---

# Getting started

Register each overload by decorating its `def` with
[`dispatch`][bagof.dispatchers.dispatch]. A second `def` that shares the name
of an earlier one does not replace it; it adds another overload to the same
function, and a call runs whichever overload most specifically matches the
arguments:

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

After both definitions, the name `area` refers to a single dispatched
function rather than to either `def` on its own. `@dispatch` returns that
function, so `area` keeps working as an ordinary callable, except that it now
chooses among every overload registered under that name.
