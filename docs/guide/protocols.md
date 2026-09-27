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
property, a method) or lists it as a dataclass field:

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
declare the member: an instance may never set it. Such a class matches the
protocol instance by instance, and is not more specific than it. A
sub-protocol is more specific than the protocols it extends.

!!! note
    A protocol without `@runtime_checkable` cannot be checked against a
    value, so an overload on one never matches.
