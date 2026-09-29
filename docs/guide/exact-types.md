# Exact types

By default, an overload registered for `int` also accepts `bool`, because
`bool` is a subclass of `int`. `Exact[C]` narrows a hint against that
default: it matches a value only when the value's type is exactly `C`, not
any of its subclasses:

```pycon
>>> from bagof.dispatchers import dispatch, Exact
>>> @dispatch
... def label(x: Exact[int]) -> str:
...     return "exactly an int"
>>> @dispatch
... def label(x: object) -> str:
...     return "something else"
>>> label(5)
'exactly an int'
>>> label(True)
'something else'
```

`True` is an `int`, since `bool` is a subclass of `int`, so a plain `int`
overload would take it. `Exact[int]` refuses it, and the call falls through
to the `object` overload.

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

## Bounds: `Super` and `Between`

A plain annotation places an upper bound on a value's class: an overload
for `Animal` accepts an instance of `Animal` or of any class derived from
it. `Super[C]` places a lower bound instead, and accepts a value whose
class is `C` or a class that `C` derives from, up to `object`.
`Between[L, U]` places both bounds at once, and accepts a value whose class
lies between `L` and `U`, both included. Together with `Exact`, these let
an overload say which classes it serves, and leave the others to other
overloads:

```pycon
>>> from bagof.dispatchers import Between, Exact, Super, dispatch
>>> class Animal: pass
>>> class Dog(Animal): pass
>>> class Puppy(Dog): pass
>>> @dispatch
... def greet(x: Super[Dog]) -> str:
...     return "Dog or one of its bases"
>>> @dispatch
... def greet(x: Exact[Dog]) -> str:
...     return "exactly a Dog"
>>> @dispatch
... def greet(x: Puppy) -> str:
...     return "a Puppy"
>>> greet(Animal())
'Dog or one of its bases'
>>> greet(object())
'Dog or one of its bases'
>>> greet(Dog())
'exactly a Dog'
>>> greet(Puppy())
'a Puppy'
```

Like `Exact`, a bound on a value is a deliberate departure from the usual
rule that code written for a class also serves its subclasses. `Exact[C]`
refuses the instances of every subclass of `C`, and `Super[C]` refuses the
instances of every class below `C`. That suits an overload meant for the
general case that must not capture the more specialised classes, which
have, or will be given, overloads of their own. Here the `Super[Dog]`
overload handles an `Animal` or a plain `object`, and a `Puppy` never
reaches it, even when no `Puppy` overload has been registered.

Each of these spellings compares the value's class against a range of
classes. A plain `C` is the range `Between[Never, C]`, from the bottom up to
`C`, and `Super[C]` is the range `Between[C, object]`. `Exact[C]` is
narrower than `Between[C, C]`: `Between[C, C]` accepts a value whose class
is equivalent to `C`, which `register` or `__subclasshook__` can arrange
for a class other than `C`, while `Exact[C]` accepts only a value whose
class is `C` itself. One range is more specific than another when it lies
inside it, so an overload for `Between[Dog, Animal]` takes `Animal` from
the `Super[Dog]` overload, while `object`, which lies above `Animal`, stays
where it was:

```pycon
>>> @dispatch
... def greet(x: Between[Dog, Animal]) -> str:
...     return "between Dog and Animal"
>>> greet(Animal())
'between Dog and Animal'
>>> greet(object())
'Dog or one of its bases'
>>> greet(Dog())
'exactly a Dog'
```

Because a bound on a value is compared against the value's class, each
bound has to be a hint that a class can be compared against, such as a
class, an abstract base class, a union of classes, a protocol with methods
only, `Never` or `Any`. A hint that is matched against the value itself,
such as a `Literal`, a `TypedDict`, a protocol with data members, or a
parametrised generic such as `List[int]`, is refused as the bound of a
value when the overload is registered.

### Bounds inside `Type` and `Hint`

The same bounds apply to a class passed as a value. A `Type[C]` parameter
accepts the class `C` and every class derived from it, and
`Type[Super[C]]` accepts the class `C` and every class that `C` derives
from. It suits a function that receives a class and relies on every
instance of `C` being an instance of that class as well, such as a
function that checks whether a container declared to hold instances of the
class it was given can also hold a `C`. `Type[Between[Dog, Animal]]`
accepts the classes `Dog` and `Animal` and those between them, and refuses
`Puppy` and `object`. Inside `Hint[...]`, where the value passed is a type
hint, the ranges hold hints instead. Inside either form, any hint can be a
bound, because the value passed is itself compared as a class or a hint.
The spellings available at each level are these:

| Where                      | Spelling              | Accepts                                             |
|----------------------------|-----------------------|-----------------------------------------------------|
| On a value parameter       | `C`                   | an instance of `C` or of a subclass                 |
| On a value parameter       | `Exact[C]`            | an instance whose type is exactly `C`               |
| On a value parameter       | `Super[C]`            | an instance whose type is `C` or a class above it   |
| On a value parameter       | `Between[L, U]`       | an instance whose type lies from `L` up to `U`      |
| On a `Type[...]` parameter | `Type[C]`             | the class `C` or a subclass of it                   |
| On a `Type[...]` parameter | `Type[Exact[C]]`      | the class `C` alone                                 |
| On a `Type[...]` parameter | `Type[Super[C]]`      | the class `C` or a class it derives from            |
| On a `Type[...]` parameter | `Type[Between[L, U]]` | a class from `L` up to `U`, both ends included      |
| On a `Hint[...]` parameter | `Hint[X]`             | the hint `X` or a sub-hint of it                    |
| On a `Hint[...]` parameter | `Hint[Exact[X]]`      | the hint `X` alone                                  |
| On a `Hint[...]` parameter | `Hint[Super[X]]`      | the hint `X` or a hint above it                     |
| On a `Hint[...]` parameter | `Hint[Between[L, U]]` | a hint from `L` up to `U`, both ends included       |

The `Type` forms combine the same way the value forms do.
`Type[Exact[Dog]]` is more specific than both `Type[Dog]` and
`Type[Super[Dog]]`, since the single class `Dog` belongs to each of them,
so an exact overload wins wherever it applies:

```pycon
>>> from typing import Type
>>> @dispatch
... def relate(cls: Type[Super[Dog]]) -> str:
...     return "Dog or one of its bases"
>>> @dispatch
... def relate(cls: Type[Exact[Dog]]) -> str:
...     return "exactly Dog"
>>> @dispatch
... def relate(cls: Type[Puppy]) -> str:
...     return "Puppy or a subclass"
>>> relate(Animal)
'Dog or one of its bases'
>>> relate(object)
'Dog or one of its bases'
>>> relate(Dog)
'exactly Dog'
>>> relate(Puppy)
'Puppy or a subclass'
```

`SuperType[C]` is a shorter spelling of `Type[Super[C]]`, and
`SuperHint[X]` is a shorter spelling of `Hint[Super[X]]`. Each always
needs its bound, since an unbounded lower bound would accept every class or
every hint, which plain `type` and `Hint` already express. The alias
expands to the `Type` form it stands for:

```pycon
>>> from bagof.dispatchers import SuperType
>>> SuperType[Dog] == Type[Super[Dog]]
True
```

### Overlapping bounds

A plain `Animal` and a `Super[Dog]` are not ordered against each other,
because each accepts a value the other refuses: a `Puppy` is an `Animal`
but its class is not above `Dog`, and a plain `object()` has a class above
`Dog` but is not an `Animal`. They do share the values whose class lies in
between, a `Dog` and an `Animal`, so a call with either of those matches
both overloads and neither is more specific. Registering the second of the
two therefore warns about the ambiguity, and the same holds for
`Type[Animal]` and `Type[Super[Dog]]` with the classes `Dog` and `Animal`.

There are two ways to settle the tie. An overload for `Exact[Dog]` settles
the call with a `Dog`, since it is more specific than both, but a call with
an `Animal` still ties. When such a call is made, the ambiguity error
suggests an `Exact[...]` overload for it:

```python
@dispatch
def feed(x: Animal) -> str: ...

@dispatch  # warns: ambiguous with feed(x: Animal)
def feed(x: Super[Dog]) -> str: ...

@dispatch  # feed(Dog()) now picks this overload; feed(Animal()) still ties
def feed(x: Exact[Dog]) -> str: ...
```

The alternative is to give one of the two overloads a higher `priority`
when it is registered, in place of the second overload above. The lower
bound then wins every value the two share, so nothing warns and nothing
ties:

```python
@dispatch
def feed(x: Animal) -> str: ...

@feed.register(priority=1)
def feed_ancestor(x: Super[Dog]) -> str: ...
```

A range and a plain class can overlap in the same way. `Dog` and
`Between[Dog, Animal]` are not ordered against each other, since a `Puppy`
belongs only to the first and an `Animal` only to the second, yet both
accept a `Dog`. Registering overloads for both therefore warns that a call
with a `Dog` is ambiguous, and the same two remedies apply.

### Writing a bound

The lower bound of a range must be a sub-hint of its upper bound. A range
with nothing in it is refused as soon as it is written, and when the
bounds are merely the wrong way round, the message suggests the reversed
spelling:

```pycon
>>> Between[Animal, Dog]
Traceback (most recent call last):
    ...
TypeError: Between[Animal, Dog] is empty: Animal is not a sub-hint of Dog, so no hint lies between them. Did you mean Between[Dog, Animal]?
```

`Between[Any, C]` is refused as well, because `Any` is the top of the
order, and no class or hint below `C` also lies above `Any`. A range with
no lower bound is written `Between[Never, C]`, which means the same as
plain `C`. A lower bound cannot be a quoted forward reference, because
nothing would ever resolve it; quoting the whole annotation, as in
`'Type[Between[Dog, Animal]]'`, works as usual.

Written around the whole form, `Super[Type[Dog]]` means the same as
`Type[Super[Dog]]` and is normalised to it. `Between` has no such outer
spelling, so `Between[Type[Dog], Type[Animal]]` is refused on a value with
a message naming `Type[Between[Dog, Animal]]`. `Exact`, `Super` and
`Between` cannot be nested inside one another, even through a union or a
`TypeVar`, since each of them already describes the whole hint at its
position. A bound can be a member of a union on a value parameter, as in
`Optional[Super[Dog]]`, or the bound of a `TypeVar` used there, but inside
`Type[...]` or `Hint[...]` it has to be the whole argument. A bound is also
refused as a constraint of a `TypeVar`, as an element of a `Tuple`, in the
signature of a `Callable`, and as a type argument of another generic, such
as `List[Super[int]]`. Each of these is refused when the overload is
registered, with an error naming the parameter and the spelling to use
instead.

### Type checkers

A static type checker reads `Super[C]` as `Union[C, Any]` and
`Between[L, U]` as `Union[L, U, Any]`, so it never rejects a call that
dispatch itself would accept, whether on a value or inside `Type[...]`.
Inside the function body, though, a checker treats a parameter annotated
`Super[Dog]` as a `Dog`, and checks attribute access against `Dog`,
although the value may be an `Animal` or a plain `object`. When that
difference matters, treat such a value as an `object` in the body. A
parameter annotated `Between[Dog, Animal]` reads as `Dog`, `Animal` or
`Any`, whose shared attributes are those of `Animal`, which matches what
the runtime passes.
