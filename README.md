# bagof-dispatchers

**Multiple dispatch driven by type hints.**

`bagof.dispatchers` lets you register several implementations of a function
and picks the right one from the **runtime types of the arguments**, choosing
the most specific matching signature the way Julia's multiple dispatch does.

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

A second `def` of the same name **adds an overload** rather than replacing it;
the name stays bound to a dispatched function you keep calling. See
[Getting started][getting-started].

## What sets it apart

- **Dispatch understands the full hint vocabulary**, not just plain classes:
  unions, literals, generics, `TypedDict`, and `TypeVar` — ordered by its
  declared variance — all narrow which overload matches. See
  [Variance][variance].
- **`Exact[C]` matches a type without its subclasses**, for the cases where
  a subclass shouldn't quietly take another overload's place — `bool` is
  otherwise also an `int`. See [Exact types][exact-types].
- **Ambiguity and no-match are both clear errors, never a silent guess.**
  `AmbiguousMethodError` names the overloads that tie; `NoMethodError` is a
  `TypeError` carrying the closest ones. See
  [When two overloads are equally specific][ambiguous] and
  [When nothing matches][no-match].
- **Dispatch is name-aware**: a call binds by position or by keyword, either
  way, against the same overloads. See [Registering overloads][registering].
- **Registries are explicit.** `@dispatch` keeps functions separate per
  module; build a [`Dispatcher`][registries] when several modules should
  extend one shared function instead.
- Python 3.8 through current, with only
  [`typing_extensions`][typing_extensions] as a dependency.

## Install

```sh
pip install git+https://github.com/bagofseeds/bagof-dispatchers.git
```

## Status

Early. The API is settling, and things may still move. Issues and ideas are
welcome at [bagofseeds/bagof-dispatchers][issues].

`bagof.dispatchers` is the low-level root of the `bagof` family: it owns the
hint subtype relation and introspection helpers the other bags build on. See
[Part of bagof][part-of-bagof] and the [project overview][bagof].

[getting-started]: https://bagofseeds.github.io/bagof-dispatchers/guide/getting-started/
[registering]: https://bagofseeds.github.io/bagof-dispatchers/guide/registering-overloads/
[ambiguous]: https://bagofseeds.github.io/bagof-dispatchers/guide/ambiguous-overloads/
[no-match]: https://bagofseeds.github.io/bagof-dispatchers/guide/no-match/
[exact-types]: https://bagofseeds.github.io/bagof-dispatchers/guide/exact-types/
[variance]: https://bagofseeds.github.io/bagof-dispatchers/guide/variance/
[registries]: https://bagofseeds.github.io/bagof-dispatchers/guide/registries/
[part-of-bagof]: https://bagofseeds.github.io/bagof-dispatchers/guide/part-of-bagof/
[typing_extensions]: https://typing-extensions.readthedocs.io/
[bagof]: https://bagofseeds.github.io/bagof/
[issues]: https://github.com/bagofseeds/bagof-dispatchers/issues
