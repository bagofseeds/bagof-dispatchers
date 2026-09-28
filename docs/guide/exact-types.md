# Exact types

By default, an overload registered for `int` also accepts `bool`, because
`bool` is a subclass of `int`. `Exact[C]` narrows a hint against that
default: it matches a value only when the value's type is exactly `C`, not
any of its subclasses:

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

This is the reverse of the usual multiple-dispatch situation, where an
overload written for a base class is meant to also serve its subclasses.
`Exact` is for the times a handler written for a base type must not run on
a more specific one instead. To a static type checker, `Exact[int]` is
indistinguishable from plain `int`.

## Exact inside `Type` and `Hint`

`Exact` also narrows a position nested inside another hint. Placing it
inside a `Type[...]` bracket dispatches on the class object itself,
matching one exact class rather than that class and its subclasses:

```pycon
>>> from typing import Type
>>> from bagof.dispatchers import dispatch, Exact
>>> @dispatch
... def which(cls: Type[int]) -> str:
...     return "int or a subclass"
>>> @dispatch
... def which(cls: Type[Exact[int]]) -> str:
...     return "exactly int"
>>> which(int)
'exactly int'
>>> which(bool)
'int or a subclass'
```

The same composition works with [`Hint`](hints-as-values.md), where
`Hint[Exact[int]]` matches the hint `int` alone rather than every hint
below it.

Writing `Exact` around the whole form, as `Exact[Type[int]]`, means
exactly the same thing; it is normalised to the inner spelling
`Type[Exact[int]]`, which is the form to prefer when writing a
signature.
