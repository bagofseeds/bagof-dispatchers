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

## Inspecting the candidates of a call

An ambiguity error names the tied overloads only after a call has
failed. The same information can be read from a function beforehand,
without calling anything.
[`bestcandidates`][bagof.dispatchers.Function.bestcandidates] takes the
arguments of a call and returns the overloads that survive every step of
selection, including the tie-breaks on `priority` and on the argument's
own MRO. [`candidates`][bagof.dispatchers.Function.candidates] returns
every overload that accepts the call, ordered from the most specific to
the least specific, with the overloads that `bestcandidates` returns at
the front.

The function below has two overloads that tie when both arguments are
polygons, and a third overload that accepts any two shapes:

```pycon
>>> class Shape: pass
>>> class Polygon(Shape): pass
>>> class Square(Polygon): pass
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")   # registration warns about the clash
...     @dispatch
...     def overlap(a: Polygon, b: Shape) -> str:
...         return "polygon first"
...     @dispatch
...     def overlap(a: Shape, b: Polygon) -> str:
...         return "polygon second"
...     @dispatch
...     def overlap(a: Shape, b: Shape) -> str:
...         return "any shapes"
```

For two squares, `bestcandidates` returns both of the tied overloads.
These are exactly the overloads that the `AmbiguousMethodError` raised by
`#!python overlap(Square(), Square())` would list. `candidates` returns
the same two overloads, followed by the less specific overload for any
two shapes:

```pycon
>>> for method in overlap.bestcandidates(Square(), Square()):
...     print(method.signature)
Signature(a: Polygon, b: Shape)
Signature(a: Shape, b: Polygon)
>>> for method in overlap.candidates(Square(), Square()):
...     print(method.signature)
Signature(a: Polygon, b: Shape)
Signature(a: Shape, b: Polygon)
Signature(a: Shape, b: Shape)
```

When a single overload is more specific than every other one,
`bestcandidates` returns only that overload, and it is the overload that
the call runs:

```pycon
>>> for method in overlap.bestcandidates(Square(), Shape()):
...     print(method.signature)
Signature(a: Polygon, b: Shape)
>>> overlap(Square(), Shape())
'polygon first'
```

Neither method raises an error: a call that no overload accepts gives an
empty tuple from both of them. Each of the two also has an iterator form,
`itercandidates` and `iterbestcandidates`. For a call described by type
hints rather than by values, as in
[`resolve`][bagof.dispatchers.Function.resolve],
[`resolve_candidates`][bagof.dispatchers.Function.resolve_candidates] and
[`resolve_bestcandidates`][bagof.dispatchers.Function.resolve_bestcandidates]
answer the same two questions.
