---
icon: fontawesome/solid/plug
---

# Protocols

A `runtime_checkable` protocol matches a value by what it **has**, not by
what it inherits. A protocol can ask for a data member as well as methods:

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

A data member counts when the value has it, set on the instance or defined
by its class, or when its class declares it with an annotation. A method
counts when the value's class defines it. Members are looked up without
running any of the value's code: a property is not called, and
`__getattr__` is not asked. A class that lists the protocol among its bases
always matches, as with `isinstance`.

## A class is more specific than a protocol it declares

An overload on a class wins over one on a protocol when the class
**declares** every member, as a type checker reads it: an annotation
(`name: str`), a class attribute, a property, a method, or a dataclass
field.

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

An annotation counts whether or not `__init__` sets the attribute, and
whether it is written on the class or on one of its bases. So an instance
of `User` matches `Named` even before `name` is set. Python's own
`isinstance` differs here: it looks for the attribute itself.

A sub-protocol is more specific than the protocols it extends.

## Class variables

A member the protocol declares as a `ClassVar` is read off the class. A
class attribute declares it, and so does a `ClassVar` annotation. An
attribute set on the instance does not:

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
...     def __init__(self) -> None:
...         self.kind = "dog"
>>> describe(Dog())
'something'
```

The other way round, a `ClassVar` annotation with no value does not declare
an ordinary member such as `name`, and neither does a dataclass `InitVar`.

## When a call matches both

A class that declares none of the protocol's members is neither more nor
less specific than the protocol. An instance that is given the member
matches both overloads, and the call is ambiguous:

```pycon
>>> from bagof.dispatchers import Function, AmbiguousMethodError
>>> badge = Function("badge")
>>> @badge.register
... def _named(x: Named) -> str:
...     return "named"
>>> @badge.register
... def _guest(x: Guest) -> str:
...     return "guest"
>>> try:
...     badge(guest)
... except AmbiguousMethodError:
...     print("ambiguous")
ambiguous
```

Registering the two gives no warning, and `ambiguities()` does not list
them: the clash only shows at the call. Annotate the member on the class
(`name: str`) to make its overload the more specific one, or give one
overload a higher `priority`.

!!! note
    A protocol without `@runtime_checkable` cannot be checked against a
    value, so an overload on one never matches.
