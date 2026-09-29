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
- `Super[C]`, written inside `Type[...]` or `Hint[...]`, is the mirror image
  of `Exact`: it accepts `C` together with everything above it, so
  `Type[Super[Dog]]` accepts the class `Dog` and each class it derives from.
  `Between[L, U]`, written in the same place, bounds the argument from both
  sides, so `Type[Between[Dog, Animal]]` accepts the classes from `Dog` up
  to `Animal`. See [Exact types][exact-types].
- A call that matches no overload, or that matches two equally specific
  overloads, is always a clear error rather than a silent guess.
  `NoMethodError` is a `TypeError` carrying the closest candidates, and
  `AmbiguousMethodError` names the overloads that tie. See
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
pip install git+https://github.com/bagofseeds/bagof-dispatchers.git
```

## Status

The project is at an early stage: the API is still settling, and some of it
may still move. Issues and ideas are welcome at
[bagofseeds/bagof-dispatchers][issues].

`bagof.dispatchers` is the low-level root of the `bagof` family. It owns the
hint subtype relation and the introspection helpers the other bags build on;
see [Part of bagof][part-of-bagof] and the [project overview][bagof]. For how
it compares with `plum`, `multipledispatch`, `functools.singledispatch`, and
Julia's own multiple dispatch, see the [comparison page][comparison].

[getting-started]: https://bagofseeds.github.io/bagof-dispatchers/guide/getting-started/
[registering]: https://bagofseeds.github.io/bagof-dispatchers/guide/registering-overloads/
[ambiguous]: https://bagofseeds.github.io/bagof-dispatchers/guide/ambiguous-overloads/
[no-match]: https://bagofseeds.github.io/bagof-dispatchers/guide/no-match/
[exact-types]: https://bagofseeds.github.io/bagof-dispatchers/guide/exact-types/
[hints-as-values]: https://bagofseeds.github.io/bagof-dispatchers/guide/hints-as-values/
[variance]: https://bagofseeds.github.io/bagof-dispatchers/guide/variance/
[registries]: https://bagofseeds.github.io/bagof-dispatchers/guide/registries/
[part-of-bagof]: https://bagofseeds.github.io/bagof-dispatchers/guide/part-of-bagof/
[comparison]: https://bagofseeds.github.io/bagof-dispatchers/comparison/
[typing_extensions]: https://typing-extensions.readthedocs.io/
[bagof]: https://bagofseeds.github.io/bagof/
[issues]: https://github.com/bagofseeds/bagof-dispatchers/issues
