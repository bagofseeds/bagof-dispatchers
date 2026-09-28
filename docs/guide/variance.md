# Variance

Two overloads can differ only in the type argument of a generic, such as
`Source[int]` against `Source[bool]`. Which one dispatch treats as more
specific then depends on the variance the generic's type parameter
declares, following the [typing
specification](https://typing.python.org/en/latest/spec/generics.html).

## Covariant and contravariant type variables

A covariant type variable, declared `covariant=True`, orders two
parametrisations of the generic the same way it orders their type
arguments. `Source[bool]` therefore counts as more specific than
`Source[int]`, simply because `bool` is more specific than `int`. A plain
`Source()`, which declares no argument of its own, matches both overloads,
and the more specific one wins:

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

A contravariant type variable, declared `contravariant=True`, reverses that
order, so between two overloads it is the wider argument that counts as
more specific:

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

## Invariant type variables

A plain, unflagged type variable is invariant, and so is every mutable
standard-library container, such as `list`. `List[int]` and `List[bool]`
describe different, incomparable kinds of list under an invariant
parameter, so two overloads written for them are equally specific rather
than one being more specific than the other. A plain list matches both, and
the call is ambiguous:

```pycon
>>> from bagof.dispatchers import AmbiguousMethodError
>>> @dispatch
... def handle(xs: tx.List[int]) -> str:
...     return "ints"
>>> @dispatch
... def handle(xs: tx.List[bool]) -> str:
...     return "bools"
>>> try:
...     handle([True, False])   # both match; neither is more specific
... except AmbiguousMethodError:
...     print("ambiguous")
ambiguous
```

Registering both overloads is legitimate: a plain list declares no type
arguments of its own, so `List[int]` and `List[bool]` genuinely both apply
to it. A value that does declare its type arguments, in contrast, picks
between them, as the next section shows. Dispatch therefore does not report
the ambiguity at registration; it surfaces only at the call, where giving
one overload a higher `priority` (`@handle.register(priority=1)`) resolves
it.

## Dispatching on a value's declared type argument

A value that declares its type arguments is dispatched on them rather than
left to match every parametrisation equally. Both an instance of a class
written against a parametrised base, and an instance built directly from a
parametrised generic, declare what they hold:

```pycon
>>> class IntList(tx.List[int]):
...     pass
>>> handle(IntList([1, 2]))       # declares List[int]
'ints'
>>> T = tx.TypeVar("T")
>>> class Box(tx.Generic[T]):
...     pass
>>> @dispatch
... def unbox(b: Box[int]) -> str:
...     return "int"
>>> @dispatch
... def unbox(b: Box[str]) -> str:
...     return "str"
>>> unbox(Box[int]()), unbox(Box[str]())
('int', 'str')
```

On Python 3.9 and later, a class written against a builtin generic through a
[PEP 585](https://peps.python.org/pep-0585/) alias works the same way,
without inheriting from `Generic` at all. It dispatches by its declared
parametrisation exactly as `Box` does above, and its unflagged `T` is
invariant, the same as it would be under `Generic`:

```python
class GL(list[T]):
    pass

handle(GL[int]())     # 'ints'
handle(GL[bool]())    # 'bools'
```

A class can also declare two different arguments for the same generic
through separate bases, which a type checker rejects as an error but
dispatch reads as both declarations at once. A diamond over `Box[int]` and
`Box[str]` therefore matches either overload, and the call is ambiguous
rather than resolved by taking the first base in the class's method
resolution order:

```pycon
>>> class Ints(Box[int]):
...     pass
>>> class Strs(Box[str]):
...     pass
>>> class Both(Ints, Strs):
...     pass
>>> try:
...     unbox(Both())
... except AmbiguousMethodError:
...     print("ambiguous")
ambiguous
```

The same reasoning applies to a class written `class Two(List[T],
Container[U])`: an instance built as `Two[int, str]()` is both a
`Container[int]` and a `Container[str]`. Nor does an `Ints` overload settle
the call for `Both()`. `Ints` is more specific than `Box[int]`, but not more
specific than `Box[str]`, and the method-resolution-order tie-break that
resolves a diamond of plain classes has no way to order parametrised
generics against each other. Giving one overload a `priority` is the way to
resolve this case too.

## What counts as a declared argument

Dispatch never looks inside a container to decide whether a value declares
a parametrisation; it only reads what the value recorded when it was built.
A value that declares nothing, such as a plain `[1, 2]` or a `Box()` built
with no argument, still matches every parametrisation, which is why the
`handle([True, False])` call above was ambiguous. A `Box[int]()` records its
argument only once `__init__` has returned. An instance with nowhere to
record it, such as one built from a class with `__slots__` and no
`__dict__`, or a frozen dataclass, declares nothing either.

A declared argument is compared using the same variance as the hints
themselves, so an invariant position requires exactly the same type on both
sides: a `Box[int]()` does not match `Box[object]`, and an `IntList` does
not match `List[object]`, though it does match `List[Any]`, plain `list`,
and `Sequence[object]`. A type variable that appears in an invariant
position stands for any type it allows, the way a type checker would solve
it. With `TB = TypeVar("TB", bound=int)`, a `Box[TB]` overload accepts both
a `Box[int]()` and a `Box[bool]()`, and acts as a fallback below which
`Box[int]` remains the more specific overload. In a contravariant position,
a type variable is read as its bound instead.

`Sequence`, `frozenset`, `Iterable`, and the other read-only containers are
covariant, so a `Sequence[bool]` overload stays more specific than a
`Sequence[int]` one; only the mutable containers are invariant. A generic
whose positions carry different variances, such as `Generator[Y_co,
S_contra, R_co]`, can leave two of its parametrisations incomparable, and
therefore ambiguous between overloads written for them. `Any` sits at the
top of the covariant yield slot but at the bottom of the contravariant send
slot, so `Generator[int, None, None]` and `Generator[int, Any, Any]` order
neither way. Variance changes only how hints are ordered against each
other; which values match which hints is decided by what each value
declares, exactly as described above.

The `typevars.co`, `typevars.contra`, and `typevars.inv` variables in
`bagof.hints`, where that package is used, carry the same variance flags and
are read the same way.
