---
icon: fontawesome/solid/triangle-exclamation
---

# When two overloads are equally specific

When two registered overloads both accept a call and neither is more
specific than the other, dispatch does not guess between them. It raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError], whose
message lists the overloads that tie:

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

To resolve a tie like this one, give one overload a higher `priority`, or
register a signature that is strictly more specific than both, such as
`(float, float)`.
