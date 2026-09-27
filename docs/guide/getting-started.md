---
icon: fontawesome/solid/rocket
---

# Getting started

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
