# bagof-dispatchers

`bagof.dispatchers` provides multiple dispatch for Python. Register several
implementations of a function, and each call runs the implementation whose
declared parameter types most specifically match the runtime types of the
arguments, the way Julia's multiple dispatch works.

```pycon
>>> from bagof.dispatchers import dispatch
>>> @dispatch
... def area(width: int, height: int) -> int:
...     return width * height
>>> @dispatch
... def area(radius: float) -> float:
...     return 3 * radius * radius
>>> area(3, 4)
12
>>> area(2.0)
12.0
```

The second `def` named `area` does not replace the first. It adds an
overload, and the name `area` stays bound to a single dispatched function
that chooses between the two on every call. [Getting started][getting-started]
walks through registering overloads in more detail.

## What sets it apart

- Dispatch reads the full vocabulary of type hints, not only plain classes.
  Unions, literals, generics, `TypedDict`, and type variables ordered by
  their declared variance all narrow which overload applies. See
  [Variance][variance].
- `Exact[C]` matches a value only when its type is exactly `C`, for the cases
  where a subclass should not quietly take over another overload's place;
  `bool` is otherwise also an `int`. See [Exact types][exact-types].
- `Hint[X]` dispatches on a type hint passed as a value, matching the hints
  that are sub-hints of `X`, so a function handed a hint can branch on what
  the hint is. See [Dispatching on hints][hints-as-values].
- `Super[C]` accepts a value whose class is `C` or a class above it, and
  inside `Type[...]` or `Hint[...]` it accepts the class or hint `C` and
  everything above it, so `Type[Super[Dog]]` accepts the class `Dog` and
  each class it derives from. `Between[L, U]` bounds from both sides at
  once, so `Between[Dog, Animal]` accepts a value whose class lies
  between `Dog` and `Animal`, both included, and
  `Type[Between[Dog, Animal]]` accepts those classes themselves. As the
  type argument of an invariant generic, a bound names a range of
  arguments, so `List[Super[int]]` accepts a list declared to hold `int`,
  `numbers.Integral` or `object`. See [Exact types][exact-types] and
  [Variance][variance].
- A call that matches no overload, or that matches two equally specific
  overloads, is always a clear error rather than a silent guess.
  `NoMethodError` is a `TypeError` carrying the closest candidates, and
  `AmbiguousMethodError` names the overloads that tie. A function's
  `ambiguities()` finds such ties before any call is made, so a test suite
  can assert that there are none. See
  [When two overloads are equally specific][ambiguous] and
  [When nothing matches][no-match].
- Dispatch is aware of argument names as well as positions, so a call binds
  by position or by keyword, either way, against the same overloads. See
  [Registering overloads][registering].
- Registries are explicit. `@dispatch` keeps functions separate per module;
  build a [`Dispatcher`][registries] when several modules should extend one
  shared function instead.
- The library supports Python 3.8 through the current release, and depends
  on nothing beyond [`typing_extensions`][typing_extensions].

## Install

```sh
pip install bagof-dispatchers
```

## Status

The project is at an early stage: the API is still settling, and some of it
may still move. Issues and ideas are welcome at
[bagofseeds/bagof-dispatchers][issues].
