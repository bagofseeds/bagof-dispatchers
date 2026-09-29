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

## Bounding a type argument

A type argument in an invariant position means exactly that type, so a
`List[int]` overload serves a list declared to hold `int`, and neither one
declared to hold `bool` nor one declared to hold `object`. `Super` and
`Between`, which bound a value's class on an ordinary parameter as
[exact types](exact-types.md) describes, can also stand as the whole type
argument of an invariant position. There they widen the single type into a
range of types, and the overload serves a value that declares any type in
that range. `List[Super[int]]` serves a list declared to hold `int` or a
type above it, such as `numbers.Integral` or `object`, and
`Box[Between[Never, numbers.Integral]]` serves a box declared to hold
`numbers.Integral` or a type below it, such as `int`:

```pycon
>>> import numbers
>>> from bagof.dispatchers import Between, NoMethodError, Super
>>> class ObjectList(tx.List[object]):
...     pass
>>> class BoolList(tx.List[bool]):
...     pass
>>> @dispatch
... def widest(xs: tx.List[Super[int]]) -> str:
...     return "int or wider"
>>> @dispatch
... def widest(xs: tx.List[bool]) -> str:
...     return "bools"
>>> widest(IntList()), widest(ObjectList()), widest(BoolList())
('int or wider', 'int or wider', 'bools')
>>> @dispatch
... def unpack(b: Box[Between[tx.Never, numbers.Integral]]) -> str:
...     return "integral or narrower"
>>> @dispatch
... def unpack(b: Box[object]) -> str:
...     return "objects"
>>> unpack(Box[int]()), unpack(Box[numbers.Integral]()), unpack(Box[object]())
('integral or narrower', 'integral or narrower', 'objects')
```

A value that declares nothing, such as a plain `[1]`, still matches every
parametrisation, so it matches both `widest` overloads. One range is more
specific than another when it lies inside it, and a single type counts as a
range of its own, so an overload for `List[int]` is more specific than the
one for `List[Super[int]]` and takes a list declared to hold `int`:

```pycon
>>> @dispatch
... def widest(xs: tx.List[int]) -> str:
...     return "exactly ints"
>>> widest(IntList()), widest(ObjectList())
('exactly ints', 'int or wider')
```

A covariant or contravariant position already widens its argument, which
changes what a bound can add there. `Sequence[int]` accepts every
parametrisation below it, so `Sequence[Between[Never, int]]` means exactly
what `Sequence[int]` means, and it is accepted and read that way. A lower
bound in the same position would be ignored, because the range from `int`
upwards reaches `Any`, and `Sequence[Any]` accepts every sequence.
`Sequence[Super[int]]` is therefore refused rather than silently read as
`Sequence[Any]`. A contravariant position is the mirror image. This follows
Kotlin, whose use-site bounds behave the same way on a parameter that
already declares its variance:

| Position                                                     | `Between[Never, U]`        | `Super[L]`                 | `Between[L, U]`            |
|--------------------------------------------------------------|----------------------------|----------------------------|----------------------------|
| invariant: `List`, `Dict`, `MutableSequence`, unflagged `T`  | a type at or below `U`     | a type at or above `L`     | a type from `L` up to `U`  |
| covariant: `Sequence`, `frozenset`, `Mapping` values, `T_co` | the same as plain `U`      | refused                    | refused, write `U`         |
| contravariant: `T_contra`, the value sent to a `Generator`   | refused                    | the same as plain `L`      | refused, write `L`         |

The error names the plain spelling that says what the bound would have
meant:

```python
@dispatch
def first(xs: Sequence[Super[int]]) -> int: ...

# TypeError: 'xs' of first: Super[int] puts a lower bound on argument 1 of
# Sequence, whose type parameter is covariant, so Sequence[Super[int]] would
# accept every Sequence. Write Sequence without an argument to accept every
# Sequence, or Sequence[int] to accept Sequence[int] and the
# parametrisations below it.
```

A bound has to be the whole type argument. Written as a member of a union,
as in `List[Union[Super[int], str]]`, or as the bound of a `TypeVar` that
fills the position, it is refused, and the error suggests the union of two
parametrisations instead, `Union[List[Super[int]], List[str]]`. A bound is
also refused as an element of a `Tuple` and in the signature of a
`Callable`, since each of those forms already fixes the variance of every
position it has, and in a generic whose variance cannot be read, such as
one with a `ParamSpec` parameter. A bound nested inside another generic
there is legal: `Callable[[List[Super[int]]], None]` bounds the invariant
argument of `List`.

Each bound names its own range, and it does not reach through the position
around it. `List[List[Super[int]]]` has the single argument
`List[Super[int]]` at its outer, invariant position, so it serves a list
declared to hold `List[Super[int]]` and not one declared to hold
`List[int]`. A range of lists of lists is written with a bound at the outer
position as well: `List[Between[Never, List[Super[int]]]]` serves a list
declared to hold `List[int]`, `List[object]` or `List[Super[int]]`. Java's
wildcards and Julia's type bounds read nested bounds the same way.

Two ranges that overlap, without either lying inside the other, make their
overloads ambiguous for a value that declares a type in both.
`List[Between[Never, numbers.Integral]]` and `List[Super[int]]` both serve a
list declared to hold `int` or `numbers.Integral`, so registering the second
of them warns about the ambiguity, and a `priority` on either settles the
calls they share.

A bound on a value parameter and a bound in a type argument ask different
questions. `Super[list]` on a value parameter asks about the value's own
class, so it accepts a `list` or a plain `object()` and refuses an
`IntList`, whose class lies below `list`. `List[Super[int]]` asks about the
type argument that the list declares, so it accepts an `IntList`, which
declares `int`.

### Type checkers and bounded type arguments

A type checker reads `Super[C]` as `Union[C, Any]` and `Between[L, U]` as
`Union[L, U, Any]`. In an invariant type argument that reading suits a lower
bound. With `class Dog(Animal)` and `class Puppy(Dog)`, mypy and pyright
both accept a `list[Dog]`, a `list[Animal]` or a `list[object]` passed to a
`List[Super[Dog]]` parameter, and mypy rejects a `list[Puppy]`, as dispatch
does. An upper bound is different, because mypy reads every member of the
union as a lower bound: it rejects a `list[Puppy]` passed to a
`List[Between[Never, Animal]]` parameter, although dispatch accepts it.

The spelling that both checkers understand for an upper bound is a bounded
`TypeVar`. A parameter annotated `List[A]`, with
`A = TypeVar("A", bound=Animal)`, accepts a list declared to hold `Animal`
or a type below it, and dispatch reads it exactly as
`List[Between[Never, Animal]]`:

```pycon
>>> Integer = tx.TypeVar("Integer", bound=numbers.Integral)
>>> @dispatch
... def narrowest(xs: tx.List[Integer]) -> str:
...     return "integral or narrower"
>>> narrowest(IntList())
'integral or narrower'
>>> try:
...     narrowest(ObjectList())
... except NoMethodError:
...     print("no match")
no match
```

`Super` and `Between` then serve for lower and two-sided bounds. mypy cannot
check the upper end of a two-sided bound in an invariant type argument, so
it rejects a valid call whose argument lies below the upper end, while
pyright accepts every call to such a parameter. Where mypy's error gets in
the way, the method can be written with its parameter annotated
`List[Any]`, which every checker accepts, and registered with the bounded
hint instead. The method of `feed` below accepts a list declared to hold
`Dog`, while a checker sees only its `List[Any]` annotation:

```pycon
>>> from bagof.dispatchers import Function
>>> class Animal: pass
>>> class Dog(Animal): pass
>>> class Puppy(Dog): pass
>>> class Dogs(tx.List[Dog]): pass
>>> feed = Function("feed")
>>> @feed.register((tx.List[Between[Puppy, Animal]],))
... def _(pets: tx.List[tx.Any]) -> str:
...     return "fed"
>>> feed(Dogs())
'fed'
```

Alternatively, a call that mypy rejects can carry a
`# type: ignore[arg-type]` comment.
