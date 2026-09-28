# When two overloads are equally specific

When two registered overloads both accept a call and neither is more
specific than the other, dispatch does not guess between them. It raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError], whose
message lists the overloads that tie:

```pycon
>>> from bagof.dispatchers import dispatch, AmbiguousMethodError
>>> import warnings
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")   # registration warns about the clash
...     @dispatch
...     def combine(a: float, b: object) -> str:
...         return "left"
...     @dispatch
...     def combine(a: object, b: float) -> str:
...         return "right"
>>> try:
...     combine(1.0, 2.0)
... except AmbiguousMethodError as error:
...     print([m.name for m in error.candidates])
['combine', 'combine']
```

Registering the second overload already warns about the clash with a
`RuntimeWarning`, since dispatch can tell, from the two signatures alone,
that some call will eventually tie between them. The example above
silences that warning only to keep the page's output focused on the
exception the call itself raises.

To resolve a tie like this one, give one overload a higher `priority`, or
register a signature that is strictly more specific than both, such as
`(float, float)`.
