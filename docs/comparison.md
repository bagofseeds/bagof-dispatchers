---
icon: fontawesome/solid/scale-balanced
---

# Comparison with other dispatch systems

`bagof.dispatchers` is one of several ways to get multiple dispatch in
Python, and Python is one of several languages where multiple dispatch is
available at all. This page compares its behaviour with three other Python
libraries, [`plum`](https://github.com/beartype/plum),
[`multipledispatch`](https://github.com/mrocklin/multipledispatch), and the
standard library's [`functools.singledispatch`][functools.singledispatch],
and with Julia, whose built-in multiple dispatch is the design this package
takes as its reference point. The claims about `bagof.dispatchers` below
were checked by reading its source and running the examples. The claims
about the other systems reflect their current, published behaviour, checked
the same way wherever that was possible, and stated narrowly where it was
not.

## At a glance

|  | `functools.singledispatch` | `multipledispatch` | `plum` | Julia | bagof.dispatchers |
| --- | --- | --- | --- | --- | --- |
| **Dispatch** | | | | | |
| Arguments dispatched on | first only | all positional | all | all | **all** |
| Reads the wider hint vocabulary (`Union`, generics, `Literal`, `TypedDict`) | `Union` only | no | most, not `TypedDict` | n/a, own type system | **yes** |
| Registration style | decorator, reads the `def` | decorator, types given as arguments | decorator, reads the `def` | `function f(x::Int)` | **decorator, reads the `def`, or explicit hints** |
| | | | | | |
| **Ambiguity** | | | | | |
| An ambiguous call | n/a; rare ABC conflicts raise `RuntimeError` | silent pick | error | error | **error** |
| Deliberate tie-break keyword | n/a | none | `precedence` | — | **`priority`** |
| | | | | | |
| **Generics and values** | | | | | |
| Reads a container's contents at the call | n/a | n/a | yes | no | **no** |
| Parametric types invariant | n/a | n/a | no, covariant by content | yes | **yes** |
| An `Exact`-style "exclude subclasses" overload | no | no | no | no | **yes** |
| | | | | | |
| **Around the edges** | | | | | |
| Function namespacing across modules | n/a, no shared registry | shared global by default | shared global by default | n/a, module system | **per module by default; shared via `Dispatcher`** |
| Runtime dependency | none | none | `beartype`, `rich` | n/a | **none beyond `typing_extensions`** |

## Julia

Julia dispatches every function call on the runtime types of all of its
arguments, not only the first, and chooses the most specific method whose
declared parameter types accept them. An unannotated parameter accepts any
value, the same as a parameter typed `Any`. When a call has more than one
applicable method and neither is more specific than the other, Julia does
not guess. It raises a `MethodError` reporting the ambiguity rather than
picking one of the tied methods, and `bagof.dispatchers` follows this model
closely enough that the resemblance is intentional. Dispatch reads every
argument, the most specific method wins, and a genuine tie raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError] instead of
choosing arbitrarily.

Julia's parametric types are also invariant by default. `Vector{Int}` is not
a subtype of `Vector{Real}`, even though `Int` is a subtype of `Real`. Two
methods written for different type arguments of the same parametric type are
therefore themselves incomparable, unless the type parameter is explicitly
declared covariant. This is the same rule `bagof.dispatchers` applies to an
invariant type variable, described in [Variance](guide/variance.md), and the
reason `List[int]` and `List[bool]` do not order one way or the other here
either.

The two systems differ where the underlying platforms differ. Julia's
dispatch is a language feature backed by a compiler that specializes and
compiles a distinct method body for each concrete combination of argument
types it actually sees. `bagof.dispatchers` is an ordinary library instead. It resolves the
applicable method by comparing type hints at call time, and caches the
result per call shape and argument type. It runs on the standard CPython
interpreter, though, and generates no specialized code of its own.

## `functools.singledispatch`

`functools.singledispatch`, added to the standard library by
[PEP 443](https://peps.python.org/pep-0443/), dispatches on the type of a
function's first argument only, leaving every other parameter untyped from
the dispatcher's point of view. Because only one argument
matters, there is no notion of two methods being incomparable across
different positions, and so no ambiguity error to raise. The implementation
for the most specific registered class in the argument's method resolution
order is chosen instead, following the same rule `isinstance` would.

```pycon
>>> from functools import singledispatch
>>> @singledispatch
... def area(shape):
...     raise TypeError("unsupported shape")
>>> @area.register
... def _(shape: int) -> int:
...     return shape * shape
>>> area(4)
16
```

Registering against a plain class, or a `Union` of classes on a recent
enough Python, works well. Registering against a parametrised generic such
as `typing.List[int]` does not, since `singledispatch` requires each
registered annotation to be an actual class. `bagof.dispatchers` reads the
full vocabulary of type hints at every argument position instead. A
signature can narrow on a union, a generic, or a `TypedDict`, at more than
one parameter at once, none of which `singledispatch` was designed to
express.

## `multipledispatch`

`multipledispatch` is closer in spirit: it dispatches on the runtime types
of every positional argument, using its own `@dispatch(int, int)` style of
registration. Where it diverges from both Julia and `bagof.dispatchers` is
in what happens when two registered signatures are equally specific for the
same call. Registering the second of such a pair produces an
`AmbiguityWarning` at definition time, but the warning is only advisory. A
later call that actually lands on the tied signatures does not raise, and
`multipledispatch` picks one of them without telling the caller which.
There is no keyword to break such a tie deliberately, so the only recourse
is to add a third, strictly more specific signature that covers the
ambiguous case. `bagof.dispatchers` raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError] at the call
itself, and a `priority` keyword lets an overload win a tie on purpose; see
[When two overloads are equally specific](guide/ambiguous-overloads.md).

`multipledispatch` also defaults to a single namespace shared by the whole
process. `@dispatch` with no further argument keys a function by its bare
name alone. Two unrelated modules that happen to define a same-named
function under that decorator therefore silently extend one shared
function, unless the caller passes an explicit `namespace=`. `@dispatch` in
`bagof.dispatchers` defaults the other way, keeping every module's
functions of the same name separate. Sharing a function across modules is
instead an explicit choice, made by building a
[`Dispatcher`][bagof.dispatchers.Dispatcher]; see
[Your own registry](guide/registries.md).

## `plum`

Of the three libraries, `plum` reads the closest to the same vocabulary of
hints that `bagof.dispatchers` does. Registered signatures may use `Union`,
`Literal`, and typing generics, and calls that tie between equally specific
signatures raise an `AmbiguousLookupError` rather than resolving silently,
much like `bagof.dispatchers`' own
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError]. `plum`
also has its own tie-breaking keyword, `precedence`, which plays the same
role as `bagof.dispatchers`' `priority`.

`plum` achieves this vocabulary by building on
[`beartype`](https://github.com/beartype/beartype), a runtime type-checking
library, and the two projects take different views of what a container's
declared type argument means. Registering `List[int]` and `List[str]`
overloads in `plum`, and calling with a mixed list such as `[1, "a"]`,
raises a lookup error. `plum` inspects the list's own elements at the call,
and finds that neither annotation matches every one of them.
`bagof.dispatchers` never inspects a container's contents. Two overloads
written for `List[int]` and `List[bool]` are equally specific for a plain
list regardless of what it holds, because an invariant type variable makes
the two hints incomparable. `bagof.dispatchers` instead reads the
parametrisation a value declared when it was constructed, the way an
instance of `class IntBox(Box[int])` does, or one built directly as
`Box[int]()`; see [Variance](guide/variance.md) for the full account.

The two libraries also part ways over an arbitrary user-defined `Generic`
class with a variance-flagged type variable, rather than one of `plum`'s own
`@parametric` types or a generic `beartype` resolves on its own. Asked to
compare two parametrisations of such a class, `plum` currently warns that it
could not resolve or classify the hint, and falls back to a best-effort
answer. `bagof.dispatchers` reads the declared variance of the type
variable directly, and orders the two parametrisations accordingly with no
fallback involved.

Neither `plum` nor the other two libraries has an equivalent of
[`Exact`][bagof.dispatchers.Exact]. All three inherit Python's ordinary
subclass relationship, so a value of a subclass matches whichever overload
its base class would. None of them offers a way to write an overload that
excludes the subclasses of the class it names; see
[Exact types](guide/exact-types.md) for why that distinction sometimes
matters.

## In summary

The three Python libraries agree with `bagof.dispatchers` on the basic
shape of multiple dispatch. Each collects several implementations under one
name and chooses the most specific one by comparing the runtime types of
the arguments. `bagof.dispatchers` treats an ambiguous call as always an
error, never a silent choice, and reads a value's declared parametrisation
for a generic rather than inspecting a container's contents. It also offers
[`Exact`][bagof.dispatchers.Exact], for the cases where a subclass should
not inherit a base class's overload, and it depends on nothing beyond
[`typing_extensions`](https://typing-extensions.readthedocs.io/), so it
carries no runtime type-checking library of its own.
