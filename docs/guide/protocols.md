---
icon: fontawesome/solid/plug
---

# Protocols

A `runtime_checkable` protocol matches a value by what it **has**, not by
what it inherits. A protocol can ask for a data member as well as methods:

```pycon
>>> import dataclasses
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
by its class. A method counts when the value's class defines it. Members are
looked up without running any of the value's code: a property is not called,
and `__getattr__` is not asked. A class that lists the protocol among its
bases always matches, as with `isinstance`.

## A class is more specific than a protocol it declares

An overload on a class wins over one on a protocol, when the class
**declares** every member — defines it itself (a class attribute, a
property, a method) or lists it as a dataclass field that `__init__` sets:

```pycon
>>> @dataclasses.dataclass
... class User:
...     name: str
>>> @dispatch
... def greet(x: User) -> str:
...     return "welcome back, " + x.name
>>> greet(User("Grace"))
'welcome back, Grace'
```

A bare annotation on a plain class (`name: str`, with no value) does not
declare the member: an instance may never set it. Neither does a dataclass
field written `field(init=False)` without a default, which `__init__` leaves
unset. A sub-protocol is more specific than the protocols it extends.

## When a call matches both

A class that annotates a member but does not declare it is neither more nor
less specific than the protocol. An instance that sets the member matches
both overloads, and the call is ambiguous. Registering the two gives no
warning, and `ambiguities()` does not list them — the clash only shows at the
call:

```pycon
>>> import warnings
>>> from bagof.dispatchers import Function, AmbiguousMethodError
>>> class Member:
...     name: str
...     def __init__(self, name: str) -> None:
...         self.name = name
>>> badge = Function("badge")
>>> with warnings.catch_warnings():
...     warnings.simplefilter("error")   # nothing is raised here
...     @badge.register
...     def _named(x: Named) -> str:
...         return "named"
...     @badge.register
...     def _member(x: Member) -> str:
...         return "member"
>>> badge.ambiguities()
[]
>>> try:
...     badge(Member("Lin"))
... except AmbiguousMethodError:
...     print("ambiguous")
ambiguous
```

Either give the class overload a higher `priority`:

```pycon
>>> badge = Function("badge")
>>> @badge.register
... def _named(x: Named) -> str:
...     return "named"
>>> @badge.register(priority=1)
... def _member(x: Member) -> str:
...     return "member"
>>> badge(Member("Lin"))
'member'
```

or declare the member on the class — a class default, or a dataclass field
as `User` does above — so the class is more specific:

```pycon
>>> class Member:
...     name: str = ""
...     def __init__(self, name: str) -> None:
...         self.name = name
>>> badge = Function("badge")
>>> @badge.register
... def _named(x: Named) -> str:
...     return "named"
>>> @badge.register
... def _member(x: Member) -> str:
...     return "member"
>>> badge(Member("Lin"))
'member'
```

!!! note
    A protocol without `@runtime_checkable` cannot be checked against a
    value, so an overload on one never matches.
