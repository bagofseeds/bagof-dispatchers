# Dispatching on hints

Some functions take a type hint as an ordinary argument rather than a value
of that type. A registry that maps types to handlers, a schema builder, and
a serialiser all pass hints around at runtime and branch on what each hint
is. `Hint[X]` lets such a function dispatch on the hint it receives, the
same way an ordinary overload dispatches on a value.

An argument matches `Hint[X]` when the argument is itself a type hint and
is a sub-hint of `X`. So `Hint[int]` matches the hints `int` and `bool`,
and `Hint[Union]` matches any union:

```pycon
>>> from typing import Union
>>> from bagof.dispatchers import dispatch, Hint
>>> @dispatch
... def describe(h: Hint[int]) -> str:
...     return "an int-like hint"
>>> @dispatch
... def describe(h: Hint[Union]) -> str:
...     return "a union"
>>> describe(bool)
'an int-like hint'
>>> describe(Union[int, str])
'a union'
```

Here `describe` is called with the hints `bool` and `Union[int, str]`
themselves, not with instances of them. The bare `Union` written as the
argument to `Hint` stands for "any union at all", so every parametrised
union reaches the second overload.

## Ordering hints by specificity

`Hint` forms are ordered against each other by their arguments, so the
usual most-specific-wins selection applies. Combining `Hint` with
[`Exact`](exact-types.md) narrows a match from "any sub-hint of `X`" to
"the hint `X` itself", which gives a strictly more specific overload:

```pycon
>>> from bagof.dispatchers import Exact
>>> @dispatch
... def kind(h: Hint[Union]) -> str:
...     return "some union"
>>> @dispatch
... def kind(h: Hint[Exact[Union]]) -> str:
...     return "the bare Union form"
>>> kind(Union[int, str])
'some union'
>>> kind(Union)
'the bare Union form'
```

A parametrised union such as `Union[int, str]` is a union but is not the
bare `Union` form, so it takes the general overload. The bare `Union`
itself matches `Hint[Exact[Union]]`, and because that hint is more specific
than `Hint[Union]`, it wins. The match on `Exact` is by the exact hint, not
by mere equivalence, so a free `TypeVar`, which accepts the same values as
`Any` without being the hint `Any`, does not match `Hint[Exact[Any]]`.

## Checking whether a value is a hint

A function that dispatches on hints often needs to know first whether it
was handed a hint at all. `ishint` answers that. It accepts every kind of
type hint, from a plain class through the whole typing vocabulary, and
rejects an ordinary value that was never meant to be a hint:

```pycon
>>> from bagof.dispatchers.core import ishint
>>> ishint(int)
True
>>> ishint(Union[int, str])
True
>>> ishint(42)
False
```
