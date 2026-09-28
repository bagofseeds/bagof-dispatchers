# Gotchas

Most of dispatch reads exactly the way the type hints suggest it should.
This page collects the handful of places where the reasonable-looking guess
is wrong, together with the rule that actually applies and a short example
of it in action. Each section links to the guide page that covers its topic
in full.

## An invariant container does not order itself

`List[int]` and `List[bool]` describe two different, incomparable kinds of
list, because the type variable behind a mutable container such as `list`
is invariant. A plain list declares no type argument of its own, so it
matches both overloads equally, and a call passing one is ambiguous rather
than resolved in favor of the narrower-looking hint:

```pycon
>>> import typing_extensions as tx
>>> from bagof.dispatchers import Function, AmbiguousMethodError
>>> handle = Function("handle")
>>> @handle.register
... def _ints(xs: tx.List[int]) -> str:
...     return "ints"
>>> @handle.register
... def _bools(xs: tx.List[bool]) -> str:
...     return "bools"
>>> try:
...     handle([True, False])
... except AmbiguousMethodError:
...     print("ambiguous")
ambiguous
```

A covariant type variable, such as the one behind `Sequence`, would order
these two the same way it orders `bool` and `int` themselves. See
[Variance](variance.md) for the full account, including how a value that
does declare its type argument, such as an instance of a class written
against a parametrised base, is matched against it directly.

## `float` means exactly `float`

Dispatch matches the runtime type of a value, so an overload annotated
`float` accepts only actual `float` instances. Python's numeric tower,
where an `int` is treated as a kind of `float` for the purposes of static
type checking, is not a rule dispatch follows:

```pycon
>>> from bagof.dispatchers import dispatch, NoMethodError
>>> @dispatch
... def halve(x: float) -> float:
...     return x / 2
>>> try:
...     halve(3)
... except NoMethodError:
...     print("no overload for int")
no overload for int
```

Annotate the parameter `numbers.Real` instead when any real number should
do, including an `int`, a `bool`, or a `fractions.Fraction`:

```pycon
>>> from numbers import Real
>>> @dispatch
... def halve(x: Real) -> float:
...     return x / 2
>>> halve(3)
1.5
>>> halve(2.0)
1.0
```

[Dispatching on numbers](numbers.md) covers `numbers.Integral` and
`numbers.Complex` too.

## A protocol is read from declarations, not from what is set right now

A [`runtime_checkable`][typing.runtime_checkable] protocol's data member
counts as present the way a static type checker would read it, which is not
always the way `isinstance` would. An annotation declares the member even
before anything sets the corresponding attribute, and a read-only member,
such as a property with no setter, satisfies a protocol member that could in
principle be assigned to:

```pycon
>>> import typing_extensions as tx
>>> from bagof.dispatchers import dispatch
>>> @tx.runtime_checkable
... class Named(tx.Protocol):
...     name: str
>>> @dispatch
... def greet(x: Named) -> str:
...     return "hello, " + x.name
>>> @dispatch
... def greet(x: object) -> str:
...     return "hello, stranger"
>>> class Guest:
...     pass
>>> guest = Guest()
>>> greet(guest)
'hello, stranger'
>>> guest.name = "Ada"
>>> greet(guest)
'hello, Ada'
>>> class ReadOnlyPerson:
...     @property
...     def name(self):
...         return "read-only"
>>> greet(ReadOnlyPerson())
'hello, read-only'
```

A [`ClassVar`][typing.ClassVar] declaration reads as a separate kind of
member, looked up on the class rather than on the instance, and it is never
satisfied by a plain class attribute of the same name:

```pycon
>>> from typing import ClassVar
>>> @tx.runtime_checkable
... class Kinded(tx.Protocol):
...     kind: ClassVar[str]
>>> @dispatch
... def describe(x: Kinded) -> str:
...     return "a " + x.kind
>>> @dispatch
... def describe(x: object) -> str:
...     return "something"
>>> class Cat:
...     kind: ClassVar[str] = "cat"
>>> describe(Cat())
'a cat'
>>> class Dog:
...     kind = "dog"
>>> describe(Dog())
'something'
```

Both rules follow the [typing
specification](https://typing.python.org/en/latest/spec/protocol.html) and
match how mypy and pyright read the same protocol. [Protocols](protocols.md)
covers what happens next: a class that declares every member of a protocol
it satisfies becomes the more specific overload, while a class that
declares none of them is left ambiguous against it.

## An ambiguous call is always an error

A call that matches two equally specific overloads never picks one and
stays quiet about it. It raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError], and there
are two ways to prevent that from happening in the first place.
[`Exact`][bagof.dispatchers.Exact] excludes an overload's subclasses when a
handler written for the base type must not also catch a more specific one,
such as `bool` being a subclass of `int`:

```pycon
>>> from bagof.dispatchers import dispatch, Exact
>>> @dispatch
... def label(x: int) -> str:
...     return "an integer"
>>> @dispatch
... def label(x: Exact[bool]) -> str:
...     return "a boolean"
>>> label(5)
'an integer'
>>> label(True)
'a boolean'
```

`priority` breaks a tie between two overloads that are, and are meant to
stay, equally specific at the type level:

```pycon
>>> import warnings
>>> from bagof.dispatchers import Function
>>> combine = Function("combine")
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")
...     @combine.register((float, object), priority=1)
...     def _(a, b): return "left"
...     @combine.register((object, float))
...     def _(a, b): return "right"
>>> combine(1.0, 2.0)
'left'
```

See [When two overloads are equally specific](ambiguous-overloads.md) and
[Exact types](exact-types.md) for more on each.

## A declared parametrisation drives dispatch, not a container's contents

A class written against a parametrised generic base, such as
`class GL(list[T])` on Python 3.9 and later, dispatches by the type
argument it was built with. That argument is read once, from how the
instance was constructed, and never by looking at the values the instance
holds. `GL[int]` and `GL[bool]` both count as declaring `List[int]` and
`List[bool]` respectively, so instances built from them settle the
ambiguity that a plain list leaves standing. Calling `handle`, the same
function registered above for `List[int]` and `List[bool]`, with each one
now picks the overload its declared argument calls for:

```python
T = tx.TypeVar("T")

class GL(list[T]):
    pass

handle(GL[int]([1, 2]))     # 'ints'
handle(GL[bool]([True]))    # 'bools'
```

A plain `[1, 2]` still matches every parametrisation equally, whatever
values it holds, because a plain `list` never declares a type argument of
its own. [Variance](variance.md) covers declared parametrisations in
detail, including what happens when a class declares two conflicting ones
through separate bases.
