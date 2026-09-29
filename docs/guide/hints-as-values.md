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

## Hints above a bound

`Hint[Super[X]]` turns the match around. Instead of the hints below `X`,
it accepts the hint `X` together with every hint that `X` is a sub-hint
of, so `Hint[Super[bool]]` matches `bool`, `int`, `object` and `Any`, but
not `str`. `SuperHint[X]` is a shorter spelling of the same hint. The
exact hint is more specific than both of the other forms, so a
`Hint[Exact[bool]]` overload takes `bool` itself while the lower bound
takes everything above it:

```pycon
>>> from typing import Any
>>> from bagof.dispatchers import Super
>>> @dispatch
... def widen(h: Hint[Super[bool]]) -> str:
...     return "bool or a wider hint"
>>> @dispatch
... def widen(h: Hint[Exact[bool]]) -> str:
...     return "exactly bool"
>>> @dispatch
... def widen(h: Hint[Any]) -> str:
...     return "some other hint"
>>> widen(int)
'bool or a wider hint'
>>> widen(Any)
'bool or a wider hint'
>>> widen(bool)
'exactly bool'
>>> widen(str)
'some other hint'
```

A lower bound does not replace the catch-all. `Hint[Super[bool]]` still
leaves out every hint that is not above `bool`, so the fallback for hint
dispatch remains `Hint[Any]`, described next. The page on
[exact types](exact-types.md) sets out how `Super` behaves inside `Type`
as well.

## Writing a catch-all

Use `Hint[Any]`, not `object`, as the fallback overload for hint dispatch.
A `Hint` form is deliberately incomparable with an ordinary class, so an
`object` overload is neither more nor less specific than a `Hint[X]` one; a
call matching both then has no most specific method and dispatch reports an
ambiguity. `Hint[Any]` sits above every other `Hint` form, so a more
specific `Hint[X]` always wins over it and the fallback is only reached when
nothing else applies.

```pycon
>>> from typing import Any
>>> @dispatch
... def classify(h: Hint[int]) -> str:
...     return "int-like"
>>> @dispatch
... def classify(h: Hint[Any]) -> str:
...     return "some other hint"
>>> classify(bool)
'int-like'
>>> classify(str)
'some other hint'
```

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
