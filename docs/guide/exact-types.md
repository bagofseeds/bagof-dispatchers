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

## Lower bounds: `Super` inside `Type` and `Hint`

A `Type[C]` parameter accepts the class `C` and every class derived from
it, which is an upper bound on the class passed in. `Super[C]` places a
lower bound instead: `Type[Super[C]]` accepts the class `C` and every class
that `C` derives from, up to `object`. It suits a function that receives
a class and relies on every instance of `C` being an instance of that class
as well, such as a function that checks whether a container declared to
hold instances of the class it was given can also hold a `C`.

`Super` belongs inside `Type[...]` or `Hint[...]`, where the value passed
is itself a class or a hint. On an ordinary value parameter it has nothing
to bound, because a value has one concrete class, and code written for the
instances of a class must also accept the instances of its subclasses. The
spellings available at each level are therefore these:

| Where                      | Spelling         | Accepts                                  |
|----------------------------|------------------|------------------------------------------|
| On a value parameter       | `C`              | an instance of `C` or of a subclass      |
| On a value parameter       | `Exact[C]`       | an instance whose type is exactly `C`    |
| On a `Type[...]` parameter | `Type[C]`        | the class `C` or a subclass of it        |
| On a `Type[...]` parameter | `Type[Exact[C]]` | the class `C` alone                      |
| On a `Type[...]` parameter | `Type[Super[C]]` | the class `C` or a class it derives from |
| On a `Hint[...]` parameter | `Hint[X]`        | the hint `X` or a sub-hint of it         |
| On a `Hint[...]` parameter | `Hint[Exact[X]]` | the hint `X` alone                       |
| On a `Hint[...]` parameter | `Hint[Super[X]]` | the hint `X` or a hint above it          |

The three `Type` forms combine the way the table suggests.
`Type[Exact[Dog]]` is more specific than both `Type[Dog]` and
`Type[Super[Dog]]`, since the single class `Dog` belongs to each of them,
so an exact overload wins wherever it applies:

```pycon
>>> from typing import Type
>>> from bagof.dispatchers import dispatch, Exact, Super
>>> class Animal: pass
>>> class Dog(Animal): pass
>>> class Puppy(Dog): pass
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

A plain `Type[Animal]` and a `Type[Super[Dog]]` are not ordered against
each other, because each accepts a class the other refuses: `Puppy` is
below `Animal` but not above `Dog`, and `object` is above `Dog` but not
below `Animal`. They do share the classes in between, `Dog` and `Animal`,
so a call with either of those matches both overloads and neither is more
specific. Registering the second of the two therefore warns about the
ambiguity.

There are two ways to settle the tie. An overload for `Type[Exact[Dog]]`
settles the call with `Dog` itself, since it is more specific than both,
but a call with `Animal` still ties:

```python
@dispatch
def feed(cls: Type[Animal]) -> str: ...

@dispatch  # warns: ambiguous with feed(cls: Type[Animal])
def feed(cls: Type[Super[Dog]]) -> str: ...

@dispatch  # feed(Dog) now picks this overload; feed(Animal) still ties
def feed(cls: Type[Exact[Dog]]) -> str: ...
```

The alternative is to give one of the two overloads a higher `priority`
when it is registered, in place of the second overload above. The lower
bound then wins every class the two share, so nothing warns and nothing
ties:

```python
@dispatch
def feed(cls: Type[Animal]) -> str: ...

@feed.register(priority=1)
def feed_ancestor(cls: Type[Super[Dog]]) -> str: ...
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

Written around the whole form, `Super[Type[Dog]]` means the same as
`Type[Super[Dog]]` and is normalised to it. `Super` and `Exact` cannot be
combined, since an exact class leaves nothing above it to bound. `Super`
on an ordinary value parameter, or anywhere other than directly inside
`Type[...]` or `Hint[...]`, is refused when the overload is registered,
with an error naming the parameter and the spelling to use instead. A
static type checker reads `Super[C]` as `Union[C, Any]`, so it accepts
every class passed to a `Type[Super[C]]` parameter and never rejects a
call that dispatch itself would accept.
