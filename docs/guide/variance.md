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
break the tie. Registering both is legitimate — a plain list declares no type
arguments, so `List[int]` and `List[bool]` genuinely both apply to it, while a
value that *does* declare them picks one (below) — so this is *not* reported
at registration; the ambiguity surfaces at the call, where a `priority`
resolves it.

A value that declares its type arguments is dispatched on them. An instance of
a class written against a parametrised base, and an instance built from a
parametrised generic, both say what they hold:

```pycon
>>> class IntList(tx.List[int]):
...     pass
>>> handle(IntList([1, 2]))       # declares List[int]
'ints'
>>> T = tx.TypeVar("T")
>>> class Box(tx.Generic[T]):
...     pass
>>> unbox = Function("unbox")
>>> @unbox.register
... def _int(b: Box[int]) -> str:
...     return "int"
>>> @unbox.register
... def _str(b: Box[str]) -> str:
...     return "str"
>>> unbox(Box[int]()), unbox(Box[str]())
('int', 'str')
```

On Python 3.9 and later, a class written against a builtin generic works the
same way, without `Generic`. Its `T` is invariant, like `list`'s:

```python
class GL(list[T]):
    pass

handle(GL[int]())     # 'ints'
handle(GL[bool]())    # 'bools'
```

Dispatch never looks inside a container, so a value that declares nothing —
a plain `[1, 2]`, or a `Box()` built without arguments — still matches every
parameterisation, as the ambiguous `handle([True, False])` above shows. A
`Box[int]()` records its arguments only once `__init__` has returned, and an
instance with nowhere to record them (a class with `__slots__` and no
`__dict__`, or a frozen dataclass) declares nothing.

A declared argument follows the same variance as the hints do, so an
invariant position asks for the same type: a `Box[int]()` does not match
`Box[object]`, and an `IntList` does not match `List[object]` (it does match
`List[Any]`, `list` and `Sequence[object]`). A `TypeVar` in an invariant
position stands for any type it allows, as a type checker solves it: with
`TB = TypeVar("TB", bound=int)`, a `Box[TB]` overload accepts a `Box[int]()`
and a `Box[bool]()`, and is a fallback below which `Box[int]` is more
specific. In a contravariant position a `TypeVar` is read as its bound.

`Sequence`, `frozenset`, `Iterable` and the other read-only containers are
covariant, so a `Sequence[bool]` overload stays more specific than a
`Sequence[int]` one — only the mutable containers are invariant. A generic
whose positions have *different* signs — such as
`Generator[Y_co, S_contra, R_co]` — can leave two parameterisations
incomparable (and so ambiguous): `Any` is the top of the covariant yield slot
but the *bottom* of the contravariant send slot, so `Generator[int, None,
None]` and `Generator[int, Any, Any]` order neither way. Variance changes only
how hints are **ordered**; which values match is decided by what each value
declares, as above.

If you use `bagof.hints`, its `typevars.co`, `typevars.contra` and
`typevars.inv` variables carry the matching flags and are read the same way.
