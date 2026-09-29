# When two overloads are equally specific

When two registered overloads both accept a call and neither is more
specific than the other, dispatch does not guess between them. It raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError], whose
message lists the overloads that tie:

```pycon
>>> from bagof.dispatchers import dispatch, AmbiguousMethodError
>>> @dispatch
... def combine(a: float, b: object) -> str:
...     return "left"
>>> @dispatch
... def combine(a: object, b: float) -> str:
...     return "right"
>>> try:
...     combine(1.0, 2.0)
... except AmbiguousMethodError as error:
...     print([m.name for m in error.candidates])
['combine', 'combine']
```

Registering the two overloads raises nothing and warns about nothing. The
tie is reported by the call that meets it, and only a call whose
arguments match both overloads meets it. `#!python combine(1.0, "x")`,
for instance, matches only the first overload and runs it.

To resolve a tie like this one, give one overload a higher `priority`, or
register a signature that is strictly more specific than both, such as
`(float, float)`.

## Finding ambiguities before a call

A tie that no call has met yet can still be found.
[`ambiguities`][bagof.dispatchers.Function.ambiguities] lists each pair
of overloads that some call would match with nothing to choose between
them:

```pycon
>>> for first, second in combine.ambiguities():
...     print(first.signature, second.signature)
Signature(a: float, b: object) Signature(a: object, b: float)
```

The check takes every registered overload into account, as a call does.
Once an overload for `(float, float)` is registered, that overload wins
every call the pair has in common, so the pair is no longer reported,
whichever order the three overloads were registered in:

```pycon
>>> @dispatch
... def combine(a: float, b: float) -> str:
...     return "both"
>>> combine.ambiguities()
[]
>>> combine(1.0, 2.0)
'both'
```

`ambiguities` is meant to be run in a test suite, where
`#!python assert not combine.ambiguities()` fails as soon as a new
overload leaves some call without a most specific overload. The check
looks only at calls written the way the overloads' own parameters are,
so it does not find a tie that is reachable only by spreading arguments
into `*args`. It can also report a pair that no call actually trips
over, when several narrower overloads together win every call the pair
has in common but none of them wins all of those calls alone.

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
>>> @dispatch
... def overlap(a: Polygon, b: Shape) -> str:
...     return "polygon first"
>>> @dispatch
... def overlap(a: Shape, b: Polygon) -> str:
...     return "polygon second"
>>> @dispatch
... def overlap(a: Shape, b: Shape) -> str:
...     return "any shapes"
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
