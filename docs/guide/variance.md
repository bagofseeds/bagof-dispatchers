---
icon: fontawesome/solid/arrows-left-right
---

# Variance

A generic's type argument is compared by the **variance its parameter
declares**, following the typing spec. A covariant `TypeVar`
(`covariant=True`) orders a container the same way as its argument, so a more
specific argument gives a more specific — and preferred — overload:

```pycon
>>> import typing_extensions as tx
>>> from bagof.dispatchers import dispatch
>>> T_co = tx.TypeVar("T_co", covariant=True)
>>> class Source(tx.Generic[T_co]):
...     pass
>>> @dispatch
... def read(s: Source[int]) -> str:
...     return "ints"
>>> @dispatch
... def read(s: Source[bool]) -> str:
...     return "bools"
>>> read(Source())          # Source[bool] is the more specific overload
'bools'
```

A contravariant `TypeVar` (`contravariant=True`) reverses that order, so the
*wider* argument wins instead:

```pycon
>>> T_contra = tx.TypeVar("T_contra", contravariant=True)
>>> class Sink(tx.Generic[T_contra]):
...     pass
>>> @dispatch
... def write(s: Sink[int]) -> str:
...     return "ints"
>>> @dispatch
... def write(s: Sink[bool]) -> str:
...     return "bools"
>>> write(Sink())           # Sink[int] is the more specific overload
'ints'
```

A plain `TypeVar` — and every mutable standard-library container, such as
`list` — is **invariant**: `List[int]` and `List[bool]` describe different,
incomparable lists, so two such overloads both match a list and neither is
more specific. The call is ambiguous:

```pycon
>>> from bagof.dispatchers import Function, AmbiguousMethodError
>>> handle = Function("handle")
>>> @handle.register
... def _ints(xs: tx.List[int]) -> str:
...     return "ints"
>>> @handle.register
... def _bools(xs: tx.List[bool]) -> str:
...     return "bools"
>>> try:
...     handle([True, False])   # both match; neither is more specific
... except AmbiguousMethodError:
...     print("ambiguous")
ambiguous
```

Give one overload a higher `priority` (`@handle.register(priority=1)`) to
break the tie. Registering both is legitimate — a value carries no type
arguments, so `List[int]` and `List[bool]` genuinely both apply — so this is
*not* reported at registration; the ambiguity surfaces at the call, where a
`priority` resolves it.

`Sequence`, `frozenset`, `Iterable` and the other read-only containers are
covariant, so a `Sequence[bool]` overload stays more specific than a
`Sequence[int]` one — only the mutable containers are invariant. A generic
whose positions have *different* signs — such as
`Generator[Y_co, S_contra, R_co]` — can leave two parameterisations
incomparable (and so ambiguous): `Any` is the top of the covariant yield slot
but the *bottom* of the contravariant send slot, so `Generator[int, None,
None]` and `Generator[int, Any, Any]` order neither way. Variance changes only
how hints are **ordered**: a value carries no type arguments, so any list still
matches every `List[...]` at call time.

If you use `bagof.hints`, its `typevars.co`, `typevars.contra` and
`typevars.inv` variables carry the matching flags and are read the same way.
