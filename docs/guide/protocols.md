# Protocols

A [`runtime_checkable`][typing.runtime_checkable] protocol, introduced by
[PEP 544](https://peps.python.org/pep-0544/), describes a shape a value can
have, rather than a class it must inherit from. Dispatch matches a value
against a protocol by checking whether the value has that shape, and a
protocol can require a data member as well as methods:

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
```

A data member counts as present when the value sets it on itself, or when
the value's class declares it through an annotation, a class attribute, a
property, or a slot. A method counts as present when the value's class
defines it. Dispatch looks up these members without running any of the
value's own code: it never calls a property, and it never falls through to
`__getattr__`. A class that lists the protocol among its bases always
matches it, the same as `isinstance` would find.

## A class is more specific than a protocol it declares

When a class declares every member a protocol requires, an overload written
for the class wins over one written for the protocol. A class declares a
member the same way a type checker reads it, through an annotation
(`name: str`), a class attribute, a property, a slot, or a dataclass field:

```pycon
>>> class User:
...     name: str
...     def __init__(self, name: str) -> None:
...         self.name = name
>>> @dispatch
... def greet(x: User) -> str:
...     return "welcome back, " + x.name
>>> greet(User("Grace"))
'welcome back, Grace'
>>> greet(guest)
'hello, Ada'
```

An annotation declares a member whether or not `__init__` actually sets the
corresponding attribute, and whether the annotation is written on the class
itself or inherited from one of its bases. An instance of `User` therefore
matches `Named` even before `name` is set on it, which is where dispatch
differs from `isinstance`: `isinstance` looks for the attribute to actually
be present, while dispatch reads the declaration.

A read-only member also counts as declared: a property without a setter, or
an annotation such as `name: Final = "x"`. A type checker would reject
either of these as satisfying `Named`, since `Named.name` can be assigned
to, but dispatch only ever reads the member, so it accepts both.

A sub-protocol is more specific than the protocols it extends.

## Class variables

When a protocol declares a member as a [`ClassVar`][typing.ClassVar],
dispatch reads that member off the class rather than off the instance. Only
a `ClassVar` annotation declares a member this way, whether or not the
annotation carries a value. A plain class attribute does not declare a class
variable; it declares an instance variable's default, and dispatch reads it
accordingly. An attribute set on the instance does not count as a class
variable either:

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

The relationship holds in the other direction too. A `ClassVar` declaration
never declares an ordinary instance member such as `name`, even when it
carries a value, and neither does a dataclass `InitVar`. Both rules match
how the [typing
specification](https://typing.python.org/en/latest/spec/protocol.html)
and static type checkers read class variables and instance variables.

## When a call matches both

A class that declares none of a protocol's members is neither more specific
nor less specific than the protocol itself. When an instance of such a class
is given the member some other way, it satisfies both the protocol and the
class, and a call matching both overloads is ambiguous:

```pycon
>>> from bagof.dispatchers import AmbiguousMethodError
>>> @dispatch
... def badge(x: Named) -> str:
...     return "named"
>>> @dispatch
... def badge(x: Guest) -> str:
...     return "guest"
>>> try:
...     badge(guest)
... except AmbiguousMethodError:
...     print("ambiguous")
ambiguous
```

Registering the two overloads produces no warning, and
[`ambiguities()`][bagof.dispatchers.Function.ambiguities] does not list them
as a pair. The clash depends on what a particular instance happens to have,
and only shows up at the call. To resolve it, annotate the member on the
class (`name: str`) so its overload becomes the more specific one, or give
one overload a higher `priority`.

!!! note
    A protocol without `@runtime_checkable` cannot be checked against a
    value at all, so an overload written for one never matches.
