---
icon: fontawesome/solid/triangle-exclamation
---

# When two overloads are equally specific

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
