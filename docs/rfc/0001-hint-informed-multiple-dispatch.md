---
icon: fontawesome/solid/file-lines
---

# RFC 0001: Hint-informed multiple dispatch for `bagof.dispatchers`

* **Status:** Implemented.
* **Tracking issue:** bagofseeds/bagof-dispatchers#1.

This document is the design record for the dispatch engine: the subtype
relation between type hints, the rule that decides which registered method a
call selects, and the corner cases the hint vocabulary forces a position on.
Citations name a source without quoting it verbatim: \[[PEP 483]\] and
\[[PEP 484]\] for the typing specification, \[[Julia]\] for Julia's own
multiple-dispatch semantics, and \[[Wiki]\] for background terminology.

---

## 0. Overview

`bagof.dispatchers` is the lowest-level package in the bagof family. It owns
both halves of the design: the hint-level subtype relation (`issubhint`,
`ishintstance`, and the introspection helpers they need) and the
signature-level rule that chooses among registered methods, covering arity,
TypeVar consistency, `Exact`, MRO refinement, and ambiguity detection. The
top-level `bagof.dispatchers` namespace exposes only the dispatch machinery:
`dispatch`, `Dispatcher`, `Function`, `Method`, `Signature`, `Parameter`,
`Exact`, and the error types. The relation and its introspection helpers
live under `bagof.dispatchers.core`, the namespace the rest of the family
builds on. The package's only dependency is `typing_extensions`.

Dispatch follows [Julia]'s symmetric model rather than Python's or [CLOS]'s
asymmetric one. Among the methods applicable to a call, the winner is the
one whose argument types are a subtype of every other applicable method's,
under a partial order, never a summed distance and never a left-to-right
precedence. When two applicable methods are incomparable under that order,
an explicit `priority=` breaks the tie. Failing that, the arguments' own C3
MRO does, the way [`functools.singledispatch`] resolves a diamond. If the tie
still stands, the call raises `AmbiguousMethodError`, naming the candidates
in [Julia]'s own format.

Plain type hints cannot express "this exact class and no subclass," so the
package adds one annotation for it: `Exact[C]`, spelled
`Annotated[C, EXACT]` so that a type checker still sees plain `C`. The
subtype relation itself understands `Exact`, so anything built on the
relation, including converters and validators, gets exactness for free
(§4).

One engine serves two query modes. Calling `f(*values)` dispatches on the
runtime types of the arguments, through `ishintstance`; calling
`f.resolve(*hints)` dispatches on hints directly, through `issubhint`. The
single-key form of the second mode is `resolve_hint`, the direct successor to
the sibling packages' `get_from_registry` (§8.1).

Dispatch is name-aware: a call is bound to each candidate method the way
Python itself binds arguments to parameters, following
`inspect.Signature.bind` semantics, where positionals bind by position,
keywords bind by name, and keyword-only parameters are included. Specificity
then compares the hints that the *same argument* landed in across methods,
rather than comparing parameter names or positions directly. For a call with no keywords, against methods with no
`*args`, this reduces exactly to positional dispatch, but it also lets two
methods with differently named or differently ordered parameters both be
considered for one call. This is the model `bagof.magic._polymorph` already
uses for dispatching on field names, generalised (§8.2). It departs from
[Julia] and [`plum`], both of which dispatch on positional arguments only.

The package supports Python 3.8 through the newest and future versions.
Every typing construct is reached through `import typing_extensions as tx`;
an unrecognised or future hint degrades to `Any`-like behaviour and warns
once, rather than crashing a program that has not yet been updated for a new
Python release (§11).

Repeated TypeVars, as in `def f(x: T, y: T)`, are supported for
applicability: the call is applicable only when its argument types have a
single, consistent solution for `T`. They are also supported for a narrow
specificity tie-break, where two otherwise equally specific methods that
differ only in how their repeated TypeVars group the arguments are settled
in favour of the one that ties more arguments to one consistent type (§3).
[Julia]'s full diagonal dispatch, where a repeated TypeVar can additionally
discriminate on *which* type filled it, is out of scope.

---

## 1. Multiple-dispatch theory

*Single dispatch* is the model behind Python's ordinary methods and
[`functools.singledispatch`]: it chooses an implementation from the dynamic
type of one argument. *Multiple dispatch* chooses from the dynamic types of
several arguments at once [Wiki]\]. The dispatched name is a *generic
function*; each registered implementation is a *method*; the methods
*applicable* to a call are those whose parameter types accept the call's
argument types; and the method actually chosen is the *most specific*
applicable one. Castagna, Ghelli and Longo formalised this in 1995 as
overloaded functions with late binding, governed by a partial order over
signatures.

A type can be understood as the set of values it describes [PEP 483]\]: `t1` is
a subtype of `t2` when every value of `t1` is a value of `t2`, equivalently
when every function accepting a `t2` also accepts a `t1`. For instance,
`bool` is a subtype of `int`, which is a subtype of `object`. Hints other
than plain classes fit the same order: `Literal[1]` is below `int`, `int` is
below `Optional[int]`, and `Union[int, str]` is below `object`. The order is
only partial: `int` and `str`, for instance, are incomparable.

A method's parameter list is compared against a call's argument types as a
tuple, position by position, and `Tuple` types are covariant \[[PEP 483]\]:
`(bool, str)` is a subtype of `(int, str)`. This is why dispatch on
arguments is covariant: a method's signature is a tuple type, and the
method that wins is the one whose argument-tuple type is the smallest
supertype of the call's own \[[Julia]\].

For a call `f(1, "a")`, whose argument types form the tuple `(int, str)`,
every method whose parameter types accept that tuple is applicable.
`(int, str)`, `(int, Any)`, `(Any, Any)`, and `(numbers.Real, str)` all
qualify, and `(int, str)` is chosen because it is a subtype of every other
applicable signature \[[Julia]\]. Definition order plays no part in the choice.

The order need not have a unique smallest applicable element, and when it
does not, the call is *ambiguous*. Given `g(x: float, y)` and
`g(x, y: float)`, a call `g(2.0, 3.0)` finds both applicable, and neither is
a subtype of the other, so the applicable set has two maximal elements and
no minimum. This is a property of the two registrations, not of the call,
and [Julia] raises `MethodError` rather than choosing between them, suggesting
that the caller define the method that would resolve the intersection
\[[Julia]\].

Dispatchers differ in how they handle this situation \[[Wiki]\]. [CLOS] resolves it
by argument precedence, comparing arguments left to right, and never reports
an ambiguity. [Julia], [Dylan], and [Cecil] instead treat every argument
symmetrically and raise. Among Python's own dispatch libraries,
[`multipledispatch`] warns and then picks a winner by topological order,
[`plum`] raises `AmbiguousLookupError`, and [`functools.singledispatch`] raises a plain
`RuntimeError`, but only between two of an argument's virtual base classes.
`bagof.dispatchers` is symmetric, in [Julia]'s sense, with one addition: a
genuine tie is first offered to the arguments' own MRO before it is reported
as ambiguous (§2.2).

An earlier design for this engine summed a numeric "distance" between each
argument's type and each parameter's declared type, and chose the method
with the smallest total. That model was rejected, for reasons independent of
the choices described later in this document:

- it invents a winner where the partial order has none, favouring whichever
  candidate happens to sit shallower in `__mro__`, an arbitrary tie-break
  with none of [CLOS]'s predictability;
- it is not monotone: `numbers.Integral` does not appear in `int.__mro__`,
  since `int` reaches it through ABC registration rather than inheritance,
  so the naive distance falls back to a large sentinel value for that pair.
  A signature that is strictly more specific at one position can therefore
  still lose to one that is less specific there but shallower in the class
  hierarchy;
- it has no answer for hints that are not classes at all, such as `Union`,
  `Literal`, `List[int]`, or a `TypeVar`;
- every dispatcher actually in use, including [Julia]'s `morespecific`,
  [`plum`]'s `Signature.__le__`, [`multipledispatch`]'s `supercedes`/`ambiguous`,
  and [`functools.singledispatch`]'s `_find_impl`, compares signatures with an order,
  never a metric.

---

## 2. The specificity model

### 2.1 The subtype relation: `issubhint`

`issubhint(hint, superhint)` is the primitive every other part of the engine
is built on: it answers whether every value described by `hint` is also
described by `superhint`. The table records that answer case by case; a bold
entry marks a result worth double-checking against intuition.

| Query | Result | Why |
|---|---|---|
| `bool ≤ int`; diamond `D ≤ B`, `D ≤ C` | True | nominal subtyping; a diamond yields two incomparable applicable methods |
| `Any ≤ object` / `object ≤ Any` | False / True | `object` is strictly more specific than `Any`; both may be registered, and `object` wins |
| `T ≤ Any` / `Any ≤ T` (`T` unbound) | True / True | an unbound TypeVar behaves as `Any` |
| `TB ≤ int` / `int ≤ TB` (`TB` bound by `int`) | True / True | a bound TypeVar is equivalent to its bound; it cannot express "exactly this bound" |
| `int ≤ TC`, `TC ≤ int`, `TC ≤ Union[int,str]`, `Union[int,str] ≤ TC` (`TC` constrained to `(int, str)`) | True / False / True / True | a constrained TypeVar is equivalent to the union of its constraints |
| `list ≤ List`, `List ≤ list`, `List[int] ≤ list`, `list ≤ List[int]` | True / True / True / False | `list` and `List` are one equivalence class in what is otherwise a preorder; `List[int]` sits strictly below `list ≡ List`, since a bare `list` promises nothing about its contents |
| `List[bool] ≤ List[int]`, `Dict[str,int] ≤ Dict[str,object]` | **False** | `list` and `dict` are invariant in their type arguments ([PEP 484]): one argument being a subtype of the other does not make one parametrisation a sub-hint of the other (§2.3) |
| `Sequence[bool] ≤ Sequence[int]`, `Mapping[str,bool] ≤ Mapping[str,int]` | True | `Sequence` is covariant; `Mapping` is invariant in its key and covariant in its value (§2.3) |
| `Snk[int] ≤ Snk[bool]` (`Snk`'s TypeVar declared `contravariant=True`); `Box[bool] ≤ Box[int]` (`Box`'s TypeVar carries no variance flag) | True / **False** | a user generic reads its own declared TypeVar: `contravariant` reverses the order, and no flag means invariant (§2.3) |
| `IntBox ≤ Box[int]`, `Sub[bool] ≤ Box[bool]`, `Flip[int,str] ≤ Pair[str,int]`, `IntBox ≤ Box[str]` (`class IntBox(Box[int])`; `class Sub(Box[T])`; `class Flip(Pair[B, A], Generic[A, B])`) | **True** / True / True / False | a sub-hint whose origin differs from the super-hint's is re-expressed through the bases its class was written with, then compared slot by slot (§2.3) |
| `Box[bool] ≤ Box[TB]`, `Box[TB] ≤ Box[int]`, `Box[int] ≤ Box[TC]`, `Snk[bool] ≤ Snk[TB]` (`TB` bound by `int`; `TC` constrained to `(int, str)`) | **True** / **False** / True / False | a TypeVar filling an invariant slot is solved against its bound or constraints on the super side, but stands for its whole family on the sub side, so nothing concrete sits below it; a contravariant slot keeps the plain bound reading (§2.3) |
| `Child ≤ List[int]`, `Child ≤ List[float]`, `Child ≤ List[object]`, `Child ≤ Sequence[object]` (`class Child(List[int])`) | **True** / False / False / True | `Child` is `List[int]` by nominal inheritance; invariance governs comparing that with another `list` parametrisation, while covariance still lets it widen through `Sequence` (§2.3) |
| `GL[bool] ≤ GL[int]`, `GL[int] ≤ List[int]`, `GL[int] ≤ list[str]`, `Sub ≤ list[int]` (`class GL(list[T])`, `class Sub(GL[int])`, Python 3.9+, with no `Generic` in `GL`'s MRO) | **False** / **True** / False / **True** | `GL`'s parameters are the TypeVars its [PEP 585] bases mention, read at their declared variance, and compared through those bases like any other generic (§2.3) |
| `Tuple[int] < Tuple[int, ...] < tuple`, `Tuple[int,...] ≤ Tuple[Any,...]` | True (a chain) | `Tuple` is covariant |
| `Tuple[int,str] ≤ Tuple[int,*Ts]`, `Tuple[int] ≤ Tuple[int,*Ts]`, `Tuple[int,*Ts] ≤ Tuple[*Ts]`, `Tuple[int,*Ts,str] ≤ Tuple[int,*Ts]` | True | `*Ts` is an open run of zero or more `Any`: a fixed prefix captures whatever it leaves over, and a longer fixed prefix or suffix is stricter (§3) |
| `Tuple[int,*Ts]` vs `Tuple[*Ts,int]`; `Tuple[int,*Ts]` vs `Tuple[int,...]` | incomparable | a prefix run and a suffix run, or a `*Ts` run and a `...` run, order neither way |
| `Tuple[*Ts] ≡ Tuple[Any,...] ≡ Tuple[*Us]`, `Tuple[int,...] ≤ Tuple[*Ts]`, `Tuple[*Ts] ≤ tuple` | True | a lone `*Ts` run is the widest tuple shape short of bare `tuple` |
| `Literal[True] < bool`, `Literal[1] < Literal[1,2]` | True | literals sit at the bottom of the order |
| `int ≤ Optional[int]`, `None ≤ Optional[int]`, `Optional[int] ≤ Union[int,str,None]` | True | ordinary union rules |
| `type[bool] < type[int] < type` | True | `type[C]` is covariant |
| `List[int] ≤ Sequence[int]`, `List[int] ≤ Iterable` | True | ABC registration is honoured |
| `Annotated[int,'x'] ≡ int` | True (both directions) | metadata is invisible to the relation except `Exact`, which is unwrapped and handled before delegating to the inner type |
| `Callable\[[int],str] ≤ Callable\[[bool],str]`, and the reverse | True, False | parameters are compared contravariantly (a narrower parameter list makes a more permissive callable) and the return type covariantly |
| `Callable\[[int],R] < Callable[Concatenate[int,P],R] < Callable[P,R] ≡ Callable[...,R]` | True (a chain) | `...` and a bare `ParamSpec` sit at the top of the parameter-list order — they accept a callable of any signature — with a `Concatenate` prefix strictly between the top and a fully fixed list; ordering them this way keeps the relation transitive (§11.1) |
| `int ≤ P` (a `Protocol` that is not `runtime_checkable`), and any other query against `P` | False | Python's `issubclass` would raise here; the relation instead answers `False`, so an overload registered on `P` is reachable but never fires, and the call raises `NoMethodError` rather than the program crashing |
| `list ≤ RP` (a runtime protocol) | True | protocols dispatch structurally |
| `Named ≤ HasName`, `Record ≤ HasName`, `Ann ≤ HasName`, `CV ≤ HasName`, `Sub ≤ HasName`, `Other ≤ HasName` (`HasName` a runtime protocol declaring the data member `name: str`; `Named` sets `name = …` on the class; `Record` is a dataclass with a `name` field; `Ann` only annotates it; `CV` declares it `ClassVar[str] = …`; `Sub(HasName, Protocol)`; `Other` an unrelated protocol with the same member) | **True** / **True** / **True** / False / **True** / False | Python's own `issubclass` refuses to answer here; the relation instead asks whether the class declares the member as the *kind* the protocol declares it — an instance variable or a class variable — the way mypy and pyright both read it, and the value level counts a declared member on every instance, so the order stays sound (§2.3) |
| `dict ≤ TD`, `TD ≤ dict`, `TD ≤ Mapping` | False / True / True | a `TypedDict` orders correctly at the hint level |
| `int ≤ Union` (bare) | False | a bare `Union`, `Literal`, or `Type`, with no arguments, means only "is one of these" in the abstract, and never actually applies to a value |
| `Annotated[int,'x'] ≤ Annotated` (bare), `int ≤ Annotated` (bare) | True / **False** | *(0.2.0)* a bare `Annotated` super-hint is now structural: only an `Annotated` form is below it, where before it was opaque and accepted everything |
| `type[Exact[C]\] ≤ type[C]`, `type[C] ≤ type[Exact[C]\]`, `type[Any]` accepts every class | True / **False** / — | `Exact` inside a `type[...]` position is the same identity leaf it is on its own, and `type[...]` now reads its argument through the full relation, so `type[Any]`, `type[Union[…]\]` and `type[T]` all behave (§4) |
| `Hint[bool] ≤ Hint[int]`, `int ≤ Hint[int]`, `Hint[int] ≤ object`, `Hint[Exact[int]\] ≤ Hint[int]` | True / **False** / **False** / True | *(0.2.0)* `Hint[X]` describes type hints, not values: only another `Hint` form is ordered against it, covariantly by its argument, so no ordinary hint sits below it. Above it sit `Any`, a free `TypeVar`, a union that has a `Hint` member, and a wider `Hint` form (up to `Hint ≡ Hint[Any]`), but no ordinary class such as `object`; `Exact` inside narrows to the exact hint (§4) |
| `Type[Super[Animal]\] ≤ Type[Super[Dog]\]`, `Type[Super[Dog]\] ≤ Type[Super[Animal]\]` (`class Dog(Animal)`) | True / **False** | *(0.3.0)* `Super[C]` inside `Type` accepts `C` and every class above it; the classes above `Animal` are among those above `Dog`, so lower bounds are ordered contravariantly by their bound (§4.2) |
| `Type[Exact[Dog]\] ≤ Type[Super[Dog]\]`, `Type[Exact[Dog]\] ≤ Type[Dog]` | True / True | *(0.3.0)* the single class `Dog` belongs to both, so the exact form sits below the plain form and the lower bound alike (§4.2) |
| `Type[Dog] ≤ Type[Super[Dog]\]`, `Type[Super[Dog]\] ≤ Type[Dog]` | **False** / **False** | *(0.3.0)* incomparable: each accepts a class the other refuses (a subclass of `Dog`, and `object`), although both accept `Dog`; registering both warns (§5) |
| `Type[Super[Dog]\] ≤ Type[object]`, `Hint[Super[int]\] ≤ Hint` | True / True | *(0.3.0)* every class is below `object` and every hint below `Any`, so a lower bound sits below the top of its form |
| `Type[Between[Dog, Animal]\] ≤ Type[Animal]`, `Type[Between[Dog, Animal]\] ≤ Type[Dog]` | True / **False** | *(0.3.0)* `Between[L, U]` inside `Type` accepts the classes from `L` up to `U`; the classes from `Dog` up to `Animal` all lie below `Animal`, but `Animal` itself is not below `Dog` (§4.2) |
| `Type[Exact[Dog]\] ≤ Type[Between[Dog, Animal]\]` | True | *(0.3.0)* the single class `Dog` lies between `Dog` and `Animal` (§4.2) |
| `Between[Animal, Dog]` | `TypeError` | *(0.3.0)* an empty interval is refused when it is written, and the message suggests `Between[Dog, Animal]` (§4.2) |
| `Dog() in Super[Dog]`, `Animal() in Super[Dog]`, `Puppy() in Super[Dog]` | True / True / **False** | *(0.3.0)* on a value, a bound constrains the value's class the way `Type[...]` of the same bound constrains a class passed in, so `v in Super[C]` when `C ≤ type(v)` (§4.2) |
| `Between[Dog, Animal] ≤ Animal`, `Dog ≤ Super[Dog]`, `Exact[Dog] ≤ Super[Dog]` | True / **False** / True | *(0.3.0)* on a value, each form is an interval of classes read against `type(v)` and ordered by inclusion; a plain `Dog` is the interval from `Never` up to `Dog`, and its value `Puppy()` keeps it out of `Super[Dog]` (§4.2) |
| `issubhint(Super[Literal[1]\], int)`, `ishintstance(1, Between[Never, List[int]\])` | `TypeError` | *(0.3.0)* on a value, each bound must be a hint that a class can be compared against, and a `Literal` or a parametrised generic is matched against the value itself (§4.2) |
| `List[int] ≤ List[Super[int]\]`, `List[Integral] ≤ List[Super[int]\]`, `List[bool] ≤ List[Super[int]\]`, `List[Super[Integral]\] ≤ List[Super[int]\]` | True / True / **False** / True | *(0.3.0)* the whole argument of an invariant slot may be a bound, which names a range of arguments; one parametrisation is below another when its range lies inside the other's (§4.3) |
| `Sequence[Super[int]\]`, `Sequence[Between[Never, int]\] ≡ Sequence[int]` | `TypeError` / True | *(0.3.0)* a covariant slot already accepts every argument below the one written, so a lower bound there would be ignored and is refused, while an upper bound adds nothing and reads as the plain argument (§4.3) |
| `Dict[str, Between[Never, int]\] ≤ Mapping[str, int]`, `Dict[str, Super[int]\] ≤ Mapping[str, int]` | True / **False** | *(0.3.0)* a range that reaches a covariant slot through the bases is read through its upper end (§4.3) |
| `W[Any] ≤ Snk[int]`, `W[Any] ≤ Snk[Never]` (`class W(Snk[T])`, `Snk` contravariant) | **False** / True | *(0.3.0)* `Any` at `W`'s invariant slot stands for every argument, and it reaches `Snk`'s contravariant slot as that whole range rather than as the point `Any` (§2.3) |
| `issubhint(1, int)`, `issubhint(1, 1)`, `issubhint(int, 1)` | `TypeError` | *(0.2.0)* a non-hint on either side is a caller error, reported for the left argument first, the way `issubclass` rejects a non-class; a non-hint no longer reads as `Any` |

The value-level check, `ishintstance`, does not look at the values held
inside a container. `[1] in List[str]` is True, because the items of the
list are never examined and a plain list carries no type argument to check
them against. A `TypedDict` is the one exception. Its value-level check
reads the mapping's own shape, comparing the keys the mapping provides
against the keys the `TypedDict` declares, so a mapping that supplies the
declared keys is `in TD` even though a plain `dict` never orders below a
`TypedDict` at the hint level (§2.3).

When a value's own class *does* declare a parametrisation, that declaration
is read and used (§2.3). `Box[int]()` records `Box[int]` and is not `in
Box[str]`. `GL[int]()`, for `class GL(list[T])`, records `GL[int]` and is
not `in list[str]`, and an instance of `class Child(List[int])` is not `in
List[float]`. `Box()`, a base class with a free `T`
(`class C(List[T])`), and a value that only declares `Any`, all stay
shallow. `print in Callable\[[int],str]` is True: a callable's own signature
is never inspected, and a `ParamSpec` is never solved from a value.
`True in Literal[1]` is False (\[[PEP 586]\]); `1 in T` is True; `'x' in TB` is
False (for `TB` bound by `int`). `v in HasName` is True exactly when `v` has
a `name`, set on the instance or declared by its class, and False
otherwise, so two instances of the same class can genuinely differ.

A `Hint[X]` is checked at the value level against the hint passed as a
value: `int in Hint[int]` and `bool in Hint[int]` are True, `str in
Hint[int]` is False, and `1 in Hint[int]` is False because `1` is not a
hint at all. `Union[int, str] in Hint[Union]` is True, and `Union in
Hint[Exact[Union]\]` is True while `Union[int, str] in Hint[Exact[Union]\]`
is False. Because a `Hint` matches on the value rather than on its Python
type, the call cache keys such an argument on the hint itself.

*(0.3.0)* A lower bound turns the value-level match around. A class `v` is
`in Type[Super[C]\]` when `C ≤ v`, so `Animal` and `object` are `in
Type[Super[Dog]\]` while `Puppy`, `int`, and the instance `Dog()` are not.
A hint `v` is `in Hint[Super[X]\]` when `X ≤ v`, so `numbers.Integral`,
`object`, `Any`, a free `TypeVar`, and `Union[int, str]` are
`in Hint[Super[int]\]`, while `bool`, `Literal[1]`, `str`, and the value `1`
are not. An interval asks for both bounds at once. A class `v` is `in
Type[Between[L, U]\]` when `L ≤ v` and `v ≤ U`, so `Dog` and `Animal` are
`in Type[Between[Dog, Animal]\]` while `Puppy` and `object` are not, and a
hint is `in Hint[Between[L, U]\]` under the same condition, so `int`,
`numbers.Integral`, and `numbers.Real` are `in Hint[Between[int, Real]\]`
while `bool`, `object`, and `Any` are not.

*(0.3.0)* On a value parameter, the same bounds are read against the
value's own class. A value `v` is `in Super[C]` when `C ≤ type(v)`, and
`in Between[L, U]` when `L ≤ type(v)` and `type(v) ≤ U`, so `Dog()` and
`Animal()` are `in Super[Dog]` while `Puppy()` and `1` are not, and `Dog()`
and `Animal()` are `in Between[Dog, Animal]` while `Puppy()` and
`object()` are not. This is the rule `Type[...]` applies, with `type(v)` in
place of the class passed, and it is the rule a plain class already
follows, since `v in C` when `type(v) ≤ C`.

### 2.2 Binding and selection (name-aware, normative)

Throughout, `⊑` denotes `issubhint`, already extended to understand `Exact`
(§4). Dispatch binds a call to each candidate method *before* comparing
them, and specificity compares the hints of the slots each argument landed
in, rather than comparing parameter names or positions directly. Comparing
name to name would be wrong: `def f(a, b)` and `def f(x, y)`, called as
`f(1, 2)`, clearly compete for the same two arguments, despite using
different parameter names.

A **signature** `S` is a triple `(params, varargs, varkw)`. `params` is an
ordered mapping from parameter name to `Parameter(hint, kind, default)`,
where `kind` is one of `POSITIONAL_ONLY`, `POSITIONAL_OR_KEYWORD`, or
`KEYWORD_ONLY`; `varargs` is the optional `*args` hint, written `h_*`;
`varkw` is the optional `**kwargs` hint, written `h_**`. Every parameter in
`params` is dispatched: an unannotated parameter is treated as `Any`, so it
still participates, just trivially. The return annotation plays no part in
dispatch.

A **call** `C` is `(v_0, …, v_{n-1}; {k_j: w_j})`: `n` positional values and
a mapping of keyword values. Its **shape** `σ(C)` is `(n, sorted keyword
names)`, and its **arguments** `Args(C)` are the positions `0..n-1` together
with the keyword names `k_j`.

**Binding**, `bind(S, C)`, restates `inspect.Signature.bind`, precomputed
once per method for speed. Positional values fill the positional slots in
order, with any surplus going to `*args` or, if there is none, failing the
bind. Each keyword fills the same-named keyword-able parameter if it is not
already filled, otherwise `**kwargs`, otherwise the call fails to bind. A
required parameter left unfilled fails the bind as well, while a parameter
left unfilled but carrying a default is **default-filled**. When binding
succeeds, `hint_S(a)` denotes the hint of the slot argument `a` landed in; it
is `Any` for an argument absorbed by an unannotated catch-all.

A method is **applicable to a call's values** when `bind(S, C)` succeeds and,
for every argument `a` in `Args(C)`, `ishintstance(value_a, hint_S(a))`
holds. An extra positional is checked against `h_*`, and an extra keyword
against `h_**`; every bound argument must also satisfy the TypeVar
consistency rule of §3. Default-filled parameters are not arguments, so
their hints are neither checked nor compared for specificity
(`Function(dispatch_defaults=True)` is the documented exception, §6). A
method is **applicable to a call's hints**, for the `resolve(*hints,
**named_hints)` query mode, by the same rule with `issub(q_a, hint_S(a))` in
place of the value check.

Specificity is defined per call, not once at registration: for two methods
applicable to the same call `C`, `A ⊑_C B` when, for every argument `a` in
`Args(C)`, `hint_A(a) ⊑ hint_B(a)`, together with the §3 consistency rule.
Which slot an argument lands in depends on the call's shape, so the order
itself is defined per shape and computed and cached per shape (§6) rather
than once at registration. For every fixed shape, though, it remains a
genuine partial order, so real ambiguities still surface rather than being
masked.

For a call with no keywords, against methods that declare no `*args`,
`hint_S(i)` is exactly the `i`-th hint of the classic positional tuple, so
applicability, `⊑_C`, the maximal set, and every tie-break below coincide
exactly with a plain positional dispatcher. This reduction is checked
directly: the test suite keeps a positional reference implementation and
asserts that the two agree over generated positional calls.

A keyword that names no declared parameter binds to `**kwargs`, or makes the
method inapplicable if there is none. When `**kwargs` is annotated, such a
keyword is checked against its hint and enters specificity through
`hint_S(a)`; unannotated, it is checked against `Any`.

Every keyword a call spills into one `**kwargs: T` is grouped into a single
block for the §3 grouping tie-break, so a method with `**kwargs: T` is more
specific than one with an untyped `**kwargs`. Applicability solves `T`
jointly over that whole group, together with every other slot the same `T`
reaches, exactly as `*args: T` does over the positionals it absorbs. So
`f(a=1, b=True)` matches with `T = int`, while `f(a=1, b="x")` has no
consistent solution and does not match. `Unpack[TD]` written on `**kwargs`
is treated as unannotated (§11).

Selecting one method among those applicable to a call `C` proceeds in four
steps:

1. Compute the maximal set, `Max`: the applicable methods with no strictly
   more specific applicable rival under `⊑_C`. If exactly one method
   remains, it is chosen.
2. Among the methods in `Max`, keep only those with the highest explicit
   `priority` (default `0`).
3. Among those, apply **MRO refinement**, argument by argument. Method `A`
   dominates method `B` when, for every argument `a`, either
   `hint_A(a) ≡ hint_B(a)`, or both hints are classes appearing in
   `type(value_a).__mro__`, with `A`'s class no further from the query than
   `B`'s and strictly nearer for at least one argument. Hints are unwrapped
   for this comparison, and `Exact[C]` reads as `C`, so an `Exact[C]`
   refines only at the argument where it agrees with `C`. This resolves a
   diamond `class D(B, C)` to `B`,
   the same way [`functools.singledispatch`] resolves it. Protocols and ABCs
   that do not appear in the MRO, unions, literals, and parametrised
   generics give no refinement at all. As a result, a diamond over two
   parametrisations of one generic (`class Both(Ints, Strs)`, where `Ints`
   is `Box[int]` and `Strs` is `Box[str]`) stays ambiguous between overloads
   on `Box[int]` and `Box[str]` (§2.3). A refinement can never override a
   strict specificity win at another argument: if the method MRO would
   favour disagrees with `⊑_C` at some other position, the pair stays
   ambiguous rather than being decided by MRO alone.
4. Among those, apply **tightness**, which restates the classic arity rule:
   fewer arguments absorbed by a catch-all (`*args` or `**kwargs`) wins,
   then fewer default-filled parameters, then no `**kwargs` at all, then no
   `*args` at all.

If more than one method survives every step, one further tie-break applies
before the call is declared ambiguous: the repeated-TypeVar grouping
refinement of §3, which only ever settles a tie step 4 left standing, and
never overturns a strict win from an earlier step. `resolve_hint` (§8.1), the
single-key lookup this design also provides, applies the same step-3 MRO
refinement to break a tie between equally specific class keys, using the
query's own MRO in place of an argument's runtime type. So `{Enum, str}`
resolves `class Color(str, Enum)` to `str`, and `{C, B}` resolves the
diamond `D(B, C)` to `B`.

After every step, exactly one method surviving means that method is chosen.
More than one means `AmbiguousMethodError`, and no method both bindable and
applicable means `NoMethodError`. The latter's message distinguishes three
causes: no method accepts a given keyword, with a `difflib`-based "did you
mean" suggestion drawn from every method's parameter names; a required
argument is missing for every candidate; or the argument types matched
nothing.

Priority is checked before MRO because an explicit choice should always
beat an implicit one. Both refinements are necessarily partial, since each
only ever compares one argument's hint against the same argument's
counterpart elsewhere, so a genuine conflict across two different arguments
is left ambiguous by either of them, and neither can override a strict
specificity win.

A handful of cases are worth spelling out, because they show what
"name-aware" means in practice:

- Two methods with the same parameter names in the same order behave exactly
  like plain positional dispatch, for every spelling of a call such as
  `area(c, 2.0)`.
- Two methods with different parameter names, such as `f(a, b)` against
  `f(x, y)`, compete on the same two arguments for a positional call,
  `f(1, 2)`, but a keyword call binds only to the parameter names that
  actually exist on each side.
- A keyword reachable only through one method's `**kwargs` still loses to a
  method that declares that name as a real parameter, when both are
  otherwise applicable.
- Two methods with the same names in a different order are equivalent for a
  positional call, and therefore genuine duplicate registrations to be
  avoided, but they are only ambiguous, not duplicate, for a keyword call,
  unless a `priority` separates them.
- A positional-only parameter, `def p(x, /)`, makes `p(x=1)` fail to bind at
  all, raising `NoMethodError`.

### 2.3 Where variance enters

[PEP 483] defines variance for a generic type `G`: given `t2 ⊑ t1`, `G` is
*covariant* if `G[t2] ⊑ G[t1]`, *contravariant* if `G[t1] ⊑ G[t2]`, and
*invariant* if neither holds. Variance belongs to the generic's parameter
*position*, as the type's author declares it, never to the particular
argument filling that position at a call site. The relation reads each
position's declared variance and applies it slot by slot, comparing a
sub-hint's argument `A` against a super-hint's argument `B` at the same
position. A covariant position requires `A ⊑ B`, a contravariant one
requires `B ⊑ A`, and an invariant one requires `A ≡ B`. The one exception
is `Any`, or a free TypeVar, on the super side, which remains a top that
even an invariant slot may widen to, following gradual typing's consistency
rule rather than strict equivalence. Nesting composes by recursion, so the
signs of nested positions multiply.

#### Where a position's variance comes from

A user-defined generic reads its own declared TypeVar live, off
`__parameters__`. A TypeVar marked `covariant=True` gives a covariant
position, one marked `contravariant=True` gives a contravariant one, and one
marked as neither is invariant, following [PEP 484]. This is the variance
the TypeVar *declares*, not necessarily the variance of the position it
happens to fill. An unflagged `T` is invariant wherever it is used, so the
equivalent class written against `List[T]` gets no relief just because two
of its parametrisations happen to be related. A `T_co` written into one of
`list`'s own invariant slots is a mistake a type checker reports on the
class itself, yet the relation still reads it at its word, as covariant,
matching what a checker goes on to do with it. Such a class is ill-typed,
and the order over it is only a preorder rather than a genuine partial order
among well-typed classes: it is possible for `Cov[bool] ⊑ Cov[int] ⊑
list[int]` to hold while `Cov[bool]` is not a sub-hint of `list[int]`.

A [PEP 695] type parameter declared `infer_variance` asks a type checker to
work out its variance from where it is used inside the class body. The
result is covariant if it only appears in read or return positions,
contravariant if it only appears in write or parameter positions, and
invariant if it appears in both. That inference is a piece of static
analysis a type
checker performs; nothing about it is computed or stored at runtime. Such a
parameter exposes only `__infer_variance__ = True` at runtime, with
`__covariant__` and `__contravariant__` both `False`, and the inferred
result itself is never materialized anywhere (checked directly against
Python 3.13 and 3.14). The library could replicate the inference by walking
the class body the way a checker does, but that would mean duplicating a
type checker for a benefit too small to justify it. Reading such a
parameter as invariant instead is the sound, conservative choice: invariant
is the most restrictive variance in the order, so treating an
`infer_variance` parameter this way can only make dispatch stricter, never
produce an unsound match. Someone who needs the exact covariant or
contravariant behaviour can declare an explicit `TypeVar(..., covariant=True)`
or `contravariant=True`, whose variance the library reads directly.

A class whose only generic bases are [PEP 585] aliases, such as
`class GL(list[T])` or `class GD(dict[str, T])` (available from Python 3.9
onward), lists no `__parameters__` of its own. Its parameters are instead
the TypeVars those bases mention, taken in order of first appearance, the
way `Generic` itself would collect them, and each is read the same way as
above. `GL[bool]` is therefore not a sub-hint of `GL[int]`, for exactly the
reason a plain `class GL(List[T])` would not be: the unflagged `T` it
inherits from `list` is invariant, regardless of which spelling of the base
introduced it.

```python
class GL(list[T]):  # a PEP 585 base, with no `Generic[T]` in sight
    ...

issubhint(GL[bool], GL[int])
# False -- `T` is unflagged, so `list`'s invariance still applies
```

A standard-library generic's variance is looked up in a table vendored from
CPython's own `typing` module, the reference implementation of the typing
specification. `list`, `set`, `dict`, and `MutableSequence` are invariant.
`Sequence`, `frozenset`, `Collection`, `Iterable`, and `Type[C]` are
covariant, `Mapping` is invariant in its key and covariant in its value, and
`Generator` and `Coroutine` are covariant in what they yield and return and
contravariant in what is sent to them. A CI test on Python 3.8 regenerates
this table from the live `typing` module and asserts that it still agrees,
so the table stays anchored to the specification rather than drifting from
it. `Tuple` and `Callable` are not in this table; each keeps its own
dedicated comparison, tuple shape for `Tuple` and contravariant parameters
with a covariant return for `Callable`.

As a consequence, some same-origin comparisons are narrower than a naive
subtype check would suggest. Because `list` is invariant, `List[bool]` is
not a sub-hint of `List[int]`, where reading it by argument subtyping alone
would say it was. Only same-origin pairs where one argument is a genuine
subtype of the other, inside an invariant container, are affected this way.
A covariant container such as `Sequence` or `frozenset` keeps the order a
naive reading would expect, while a contravariant one reverses it. Two
`List[X]` overloads whose arguments are subtype-related become incomparable
rather than ordered, and are therefore ambiguous at a call unless a
`priority` separates them. Registering both is legitimate, not a mistake to
warn about: a plain `list` value declares no type argument at all, so both
overloads genuinely apply to it, while a value whose class declares its own
parametrisation, described below, is still dispatched precisely. No
registration-time warning fires for this pair; the ambiguity, when it
exists, only shows up at the call.

Differing origins are handled by expressing the sub-hint through the
super-hint's origin. When a sub-hint's origin differs from a super-hint's,
as when a user subclass is compared against one of its generic bases, it is
re-expressed as a parametrisation of the super-hint's origin, through the
bases the sub-hint's own class was *written* with. This means its
`__orig_bases__`, read off the class's own namespace; a class that declares
none of its own is followed through its plain `__bases__` instead.

Every base the class was written with is followed this way, and the
sub-hint counts as below the super-hint if *any* of the parametrisations it
reaches that way is. For example, a diamond `class D(A, B)`, where `A` is
written as `Box[int]` and `B` as `Box[str]`, is genuinely both a `Box[int]`
and a `Box[str]`, the same way `class Two(List[T], Container[U])` makes
`Two[int, str]` both a `Container[int]` and a `Container[str]`. A type
checker would reject a
class shaped like this. Accepting it here is what keeps the relation
transitive through every base a class names, rather than silently picking
the first one, and the consequence is that such a value is genuinely
ambiguous between overloads on two of the parametrisations it reaches.

`Both()`, for the `D(A, B)` example above, is ambiguous between a
`Box[int]` overload and a `Box[str]` overload. `Two[int, str]()` is
likewise ambiguous between a `Container[int]` overload and a
`Container[str]` overload. Picking only the first listed base would have
resolved either case silently and, from the caller's point of view,
arbitrarily. Registering a third overload on the more specific `Ints` type
alone does not settle `Both()` either, since `Ints` is below `Box[int]` but
not below `Box[str]`, and MRO refinement gives no index to a parametrised
generic; a `priority` is needed to choose.

Filling in each base uses `typing`'s own subscription machinery: `Box[T][bool]`
is `Box[bool]`, and this pairs each base's own type variables with the
sub-hint's arguments by identity rather than by position, so
`class Flip(Pair[B, A], Generic[A, B])` correctly makes `Flip[int, str]` a
`Pair[str, int]`. A base with no free variable of its own, such as
`Box[int]`, is used exactly as written, which is what makes
`class IntBox(Box[int])` a `Box[int]`. A parametrised standard-library base,
such as `List[int]`, is read positionally against a standard-library origin
it subclasses that takes the same number of arguments, exactly as two
standard-library origins have always been compared.

*(0.3.0)* An argument that stands for more than one type keeps doing so
when it reaches a base. `Any` or an unconstrained TypeVar at an invariant
slot of the sub-hint stands, on the sub side, for every argument it admits
(see "`Any` and gradual typing" below), so `W[Any]` for `class W(Snk[T])`
stands for every `W[Y]`, and therefore for every `Snk[Y]`. When such an
argument lands as the whole argument of the base's slot, it is handed over
as the range it stands for, `Between[Never, B]` for a TypeVar bounded by `B`
and `Between[Never, Any]` otherwise, and read at the base's variance the
way §4.3 reads a bound: through its upper end at a covariant slot, which is
what the plain argument already gave, and through its lower end at a
contravariant one, where the plain argument `Any` would have been the
bottom of the order instead. A constrained TypeVar at an invariant slot
stands for each of its constraints in turn, so the sub-hint is below when
each of those parametrisations is. Handing over the point `Any` instead
would put `W[Any]` below `Snk[int]` while `W[bool]`, one of the
parametrisations it stands for, is not, and the order would stop being
transitive. An open argument nested inside another hint, as in
`class N(Sequence[List[T]])`, is handed over as written.

Sometimes nothing can be mapped this way:

- a runtime-only standard-library subclass such as `Counter`, which records
  no parametrised base at all;
- a generic class written bare, the way a bare `Box` would be;
- a `ParamSpec` or `TypeVarTuple` generic;
- a base whose substitution itself raises;
- an arity mismatch, such as comparing `Dict[K, V]` against `Iterable`.

In each of these cases, the arguments fall back to being compared
positionally, the way they always were before this mechanism existed.
`Tuple`, `Callable`, and `Type` keep their own dedicated comparisons
throughout.

Argument positions in a call are themselves covariant, independently of the
variance inside any one hint: a parameter *consumes* the value passed to it,
so a call is applicable when `type(v) ⊑ P`, and a smaller `P` is more
specific. This is exactly `Tuple`'s covariance applied to the whole argument
tuple, since `Julia`'s method signatures are themselves tuple types, and it is
a separate axis from the variance of a position *inside* one hint, discussed
above.

A value's declared parametrisation, not its contents, governs how deeply it
is checked. Dispatch never inspects a container's contents: `type([True])`
is plain `list`, so `[1]` still satisfies `ishintstance` against
`List[int]`, regardless of the container's declared variance, because a
plain `list` promises nothing about what it holds. But when a value's own
class *does* declare a parametrisation, the value check uses it. This is
read, in order, from two places. The first is the instance's
`__orig_class__`, which `typing` sets when a generic alias like `Box[int]`
is called, and which the runtime alias type sets the same way for a class
such as `class GL(list[T])` when `GL[int]()` is called. This is read only
off an instance of a `Generic` subclass, or of a class whose MRO carries a
[PEP 585] base, and it is compared against any parametrised class hint,
whether a user generic or a standard-library one (so `Row[int]()`, for
`class Row(Sequence[T])`, is a `Sequence[int]` and not a `Sequence[str]`),
and it is used whenever it re-expresses as a parametrisation of the hint's
own origin. The second, failing that, is the class's own written bases,
used when `type(v)` itself re-expresses as a parametrisation of the hint's
origin, the way `class Child(List[int])` does. Either path decides the
check by `issubhint(declared, G[args])`.

A value's declared parametrisation is used to re-express a hint only when
*every* base reaching a given origin is fully declared, not merely one of
them. `Two[Any, str]()`, for `class Two(List[T], Container[U])`, reaches
`Container[Any]` through its `List` base and `Container[str]` through its
own; because these disagree, the check stays shallow rather than picking one
of them. Reading it by `Container[str]` alone, for instance, would reject
the value as a `Container[bytes]`. Yet the same value would still pass,
shallowly through `List[Any]`, as a `Collection[bytes]`, which sits below
`Container[bytes]` in the order — putting the value in a sub-hint without
putting it in that sub-hint's own super-hint.

This is why the hint-level relation and the value-level check use different
quantifiers over the same set of bases. The hint-level relation, above,
accepts a sub-hint through *any one* base that reaches the super-hint's
origin, because a hint carries no ambiguity about which parametrisation is
meant. A value's *declared* parametrisation, by contrast, is trusted only
when *every* base reaching a given origin agrees, because trusting just one
while another disagrees could accept a value that a stricter reading
through the other base would reject.

Short of a declared parametrisation, the check stays shallow in a handful of
cases:

- a plain built-in instance, including one built from a [PEP 585] alias
  such as `list[int]([1])`, which is simply a plain list;
- an instance built from a bare, unparametrised class;
- a `__slots__` class with no `__dict__` to record a parametrisation in;
- a frozen dataclass, where `typing` itself swallows the
  `FrozenInstanceError` it would otherwise raise while trying to record one;
- `self`, seen from inside its own `__init__`, since the record is only
  written after `__init__` returns;
- a declared parametrisation whose arguments include a free TypeVar, `Any`,
  or an unresolved name (`class C(List[T])`, `Box[Any]()`, `Box["int"]()`),
  since none of these say what the value actually holds.

A `ParamSpec` or `TypeVarTuple` generic, whose arguments do not pair
one-to-one with parameters, also stays shallow.

Where a value does carry a declared parametrisation, the comparison follows
the specification's variance rules exactly, which is stricter than the
shallow check in an invariant position. `Box[int]()` no longer matches
`Box[object]`, `Box[Union[int, str]\]`, or `Box[Optional[int]\]`, and an
instance of `class Strs(List[str])` no longer matches `List[object]`,
though it still matches `List[Any]`, plain `list`, and the covariant
`Sequence[object]`.

A TypeVar filling the hint's invariant slot is solved the same way as at
the hint level, below. `Box[int]()` still matches `Box[T]` when `T` is
unbound or bound by `object` or by `numbers.Real`, and it matches a TypeVar
constrained to `(int, str)`, but not one bound by `float`, since there is
no numeric-tower promotion here. `Tuple`, `Callable`, `Type[C]`, and
`TypedDict` each keep their own value-level checks, untouched by this
mechanism.

#### `Any` and gradual typing

The typing specification distinguishes
*subtype of* from *consistent with* [PEP 483]: `Any` is consistent with
everything, but is neither a subtype nor a supertype of anything else, while
`object` remains the nominal top of the ordinary subtype order. A dispatcher
still has to order `(object,)` against `(Any,)` for two overloads to make
sense, and the relation answers that `object` is strictly below `Any`. So `Any`,
and an unannotated parameter, form the widest possible catch-all, and an
`object`-typed method beats it, matching `Julia`'s own reading. Inside an
invariant slot, the same consistency reading keeps a free TypeVar or `Any`
above every parametrisation `G[X]`, so a generic fallback overload stays
comparable to more specific ones rather than becoming incomparable with all
of them.

A TypeVar filling an invariant slot is solved against its bound or
constraints, not read as exactly its bound. On the super side, it stands
for *some* type within its bound or constraints, exactly as a type checker
would solve it. On the sub side, by contrast, it stands for the whole
family of types that could fill it, which no single concrete type can be
said to contain. A free TypeVar, or `Any`, remains the top of the order as
above. For a TypeVar
bounded by `B`: `G[A] ⊑ G[TB]` when `A ⊑ B`, since the TypeVar is read by its
own bound. So `Box[bool] ⊑ Box[TB]` when `TB` is bounded by `int`, and a
`Box[TB]` fallback sits above every one of its concrete specialisations; two
bounded TypeVars order by comparing their bounds. For a TypeVar constrained
to `(C1, …, Cn)`: an argument `A` must be equivalent to one of the `Ci`, and
a constrained TypeVar on the sub side must have each of its own constraints
match one of the super side's. A TypeVar on the sub side is never below a
*concrete* super-side type: `Box[TB]` is not below `Box[int]`, while
`Box[int]` is strictly below `Box[TB]`. Each of these rules reduces to `⊑`
or `≡` against the super side's bound or constraints, so the resulting order
remains a preorder.

A covariant slot already reads a TypeVar by its bound, which is exactly what
"solving" it gives; nothing changes there. A **contravariant** slot,
however, keeps the plain bound reading rather than solving, because solving
it would mean asking whether the two sides *overlap*, that is, whether some
type exists below both bounds, and overlap is not transitive. In a diamond
`class D(B, C)`, both `B` and `C` overlap with `D` without overlapping each
other, and deciding overlap in general would require knowing the whole,
open class hierarchy, which the relation cannot do. So `Snk[bool]()` does
not match `Snk[T]` bounded by `int`, even though a type checker would
accept it. This is a documented limitation, accepted because it keeps the
relation itself computable and transitive, rather than trying to answer a
question that is not decidable over an open hierarchy.

*(0.3.0)* **Bounds on a position.** A `Super[C]` or a `Between[L, U]` written
as the whole argument of an invariant slot names a range of arguments, and
`G[A] ⊑ G[B]` holds when the range of `A` lies inside the range of `B`
(§4.3). A plain argument is a range of one point, and a TypeVar bounded by
`B` is the range `Between[Never, B]`, which is exactly the super-side rule
above restated; the sub-side rule is the same range read from the other
side. At a covariant or contravariant slot the declared variance already
widens a plain argument, so a range there means its upper or its lower end.
A bound that says only that is accepted and read as the plain argument, and
one whose other end would be ignored is refused, which is how Kotlin treats
a use-site projection on a parameter that declares its variance.

A generic that mixes variance across its positions can leave two of its
parametrisations incomparable. `Generator[Y, S, R]` is covariant in what
it yields and returns and contravariant in what is sent to it. `Any` is the
top of a covariant position but the *bottom* of a contravariant one, so
`Generator[int, None, None]` and `Generator[int, Any, Any]` order neither
way. Two overloads built on them are therefore ambiguous, which is sound,
since an antisymmetric preorder is allowed to leave some pairs
incomparable.

A generic-class TypeVar from `bagof.hints.typevars` carries its variance
the same way. `hints.typevars.co.INT` declares a covariant position,
`contra.INT` a contravariant one, and `inv.INT` (as well as `infer.INT`) an
invariant one, when such a variable fills a user generic's declared
parameter. [PEP 484] and the type checkers forbid a variance-flagged TypeVar
from being used as a plain function *parameter*; when one is anyway, it is
read by its bound, and `Exact[int]` remains the tool for asking for
exactness (§4).

The value-level check is deliberately structural where the hint-level
relation is nominal. `ishintstance(v, H)` is not defined as
`issubhint(type(v), H)`: for a runtime `Protocol`, `Hashable`, or
`Callable`, the value check asks whether the value *itself* has the
required capability, while the hint-level relation asks a nominal question
about the *type*. So `ishintstance(object(), Hashable)` is true, because a
plain `object` instance has `__hash__`, even though
`issubhint(object, Hashable)` is false. A `Mapping` subclass that sets
`__hash__ = None` on itself is exactly why the nominal answer is the
conservative, safe one for the type-level relation to give. The two query
modes are deliberately different for this reason: `f(value)` asks a
structural question and `f.resolve(hint)` asks a nominal one, and collapsing
them into one answer would make value dispatch miss a value that
structurally satisfies a protocol. This sits in the same family of
gradual-typing corner as the `Callable[...]` wildcard, and the affected
comparisons, `Hashable` reached only through `object` and the
`Callable[...]` wildcard, are deliberately excluded from the property-tested
preorder-law corpus described below, rather than special-cased into it.

#### Protocols with data members

Python's own `issubclass` refuses to answer for a runtime-checkable
protocol that declares a data member, such as `name: str`, because whether
an instance has that attribute is a property of the instance rather than
the class. The relation instead reads such a protocol member by member. A
class is below the protocol when it declares every member the protocol
does. A value is in the protocol when it actually holds every member. These
are two related but distinct questions, worked out below, whose answers can
come apart for exactly the classes that never declare what they set.

```pycon
>>> import typing_extensions as tx
>>> from bagof.dispatchers.core import issubhint, ishintstance
>>> @tx.runtime_checkable
... class HasName(tx.Protocol):
...     name: str
...
>>> class Named:
...     name: str  # declares the member the protocol asks for
...     def __init__(self, name):
...         self.name = name
...
>>> issubhint(Named, HasName)
True
>>> ishintstance(Named("Ada"), HasName)
True

```

#### Reading members at the value level

A value `v` is in a protocol `P` when `type(v)` lists `P` among its bases,
exactly as `isinstance` already counts it, or when every one of `P`'s
members is satisfied. Every **method** `P` declares must be defined by
`type(v)`, and not simply set to `None`. Every member `P` declares as a
**class variable**, through a `ClassVar` annotation, must be declared as a
class variable somewhere in `type(v)`'s own MRO, with or without a value,
read off the class alone the way a type checker reads a class attribute.
Every other, ordinary **data member** must either be present in the
instance's own `__dict__`, or declared as an instance variable anywhere in
the class's MRO. Such a declaration is a plain annotation (`name: str`, so
long as it is not itself marked `ClassVar`, `InitVar`, or `KW_ONLY`), or a
class attribute, a property, or a slot that the class defining it does not
mark `ClassVar` or `InitVar`.

Members are found statically. A property is never called, and
`__getattr__` is never consulted, matching how `isinstance` itself reads a
protocol from Python 3.12 onward. As a result, the relation answers the
same way on every supported Python and with either typing module, where
`isinstance` itself would not be consistent: before 3.12, `typing`'s own
`isinstance` calls `hasattr`, which does call properties. Because methods
are read off the class, only the data members actually depend on the
particular instance, so a method assigned to one instance alone is not
counted, even though a native `isinstance` from 3.12 onward would count it.

This deliberately follows the type checkers' own reading rather than
`isinstance`'s in three further ways:

- an annotated member counts as present even on an instance that never
  actually set it;
- a member set on the instance that the class instead declares `ClassVar`,
  or a plain class attribute with no annotation at all, does not count as
  an instance member;
- a class attribute declared `ClassVar` never counts toward an instance
  member either.

#### Reading members at the hint level

`C ⊑ P` when `C` itself lists `P` among its bases, including through a
sub-protocol, or when `C` is not itself a protocol and *declares* every one
of `P`'s members. This is the same way a type checker reads structural
subtyping: each method `C` defines (again, not as `None`), and each data
member declared, anywhere in `C`'s MRO, as the kind `P` declares it. An
**instance variable** is declared by a plain annotation, including a
dataclass field whether or not it is `init=False`, or by a class attribute,
a property, or a slot that the class defining it does not mark `ClassVar`.
A value given in the class body for such an attribute is simply the
instance variable's default. A **class variable** is declared only by a
`ClassVar` annotation, with or without a value.

The two kinds are mutually exclusive, matching how the [typing
specification](https://typing.python.org/en/latest/spec/protocol.html)
treats a protocol's own members: a member is an instance variable unless it
is explicitly declared `ClassVar`. This also matches what both mypy and
pyright do. Each rejects a plain class attribute standing in for a
`ClassVar` member, and each rejects a plain annotation standing in for an
instance-variable member declared without one, with the message "expected
class variable, got instance variable" or its mirror. The relation enforces
the same exclusivity.

When an annotation is still text, as it is under `from __future__ import
annotations`, it is parsed rather than executed. Its leading dotted name is
resolved purely by dictionary lookup, first in the class's own namespace,
then its defining module's, then the builtins, and, for a dotted name, in
the intermediate module's own namespace too. The `ClassVar` marker is
recognised by identity rather than by name. An import alias such as `from
typing import ClassVar as CV` still reads as `ClassVar`, and a user class
the reader happens to have named `InitVar` reads as itself rather than
being mistaken for `dataclasses.InitVar`. A name that cannot be
resolved this way falls back to being read by its last dotted component.
`InitVar`, `KW_ONLY`, and a `TypedDict`'s own keys declare no attribute at
all.

A protocol that does not itself list `P` among its bases is never below it
merely for sharing the same member names. A protocol that only declares
data members is below a method-only protocol `Q` when it lists `Q`, or when
its own methods happen to cover `Q`'s members. It is never below `Q`
through a data member alone — a distinction Python's own `issubclass` does
not draw, since it would accept a bare annotation that an instance is
nonetheless free to hold without.

#### Keeping the two levels sound

The order must never place `C` below `P` while some instance of `C` fails
to actually satisfy `P`. A bare annotation with no value, such as `name:
str` alone, promises nothing at runtime, since a class that only annotates
`name` and never sets it in `__init__` produces instances with no such
attribute at all.

The relation does not, however, refuse the annotation at the hint level.
Refusing it, which the type checkers themselves do not do, would leave a
class that both annotates a member and sets it in `__init__` incomparable
with the protocol, making an overload on each ambiguous for every one of
its instances. Instead, the value level counts a merely-annotated member as
present on *every* instance of the class, whether or not that particular
instance ever set it. The two levels read one shared per-class record of
what the whole MRO declares, gathered once across every class in the MRO
rather than decided class by class. A subclass therefore automatically
declares everything its bases do, and `v ∈ C ⊑ P` implies `v ∈ P` by
construction, transitively through the whole class hierarchy; the test
suite sweeps this property directly. A `ClassVar` member is read off the
class at both levels, for the same reason.

#### Documented divergences from type checkers

A few cases are left as documented residue rather than guarded against,
because guarding against them is either impossible or not worth the cost:

- a subclass that sets an inherited method to `None`;
- an attribute or annotation added to a class after that class was first
  dispatched on, since the per-class record is memoised at first use;
- a subclass that redeclares an inherited instance variable as a `ClassVar`
  still sits below the protocol, because the union taken over the whole MRO
  keeps the base class's own declaration. In `class SubCV(Ann):
  name: ClassVar[str]`, where `Ann` annotates `name: str` as an instance
  variable, `SubCV ⊑ Ann ⊑ HasName` is preserved even though both type
  checkers reject the redeclaration itself;
- a **read-only** member, such as a property with no setter or `name:
  Final = "x"`, counts toward a protocol's instance-variable requirement
  even though both type checkers reject it with "expected settable
  variable, got read-only attribute". Dispatch only ever reads a member and
  never assigns to it, so the value genuinely has what dispatch needs, and
  the divergence from the type checkers is deliberate rather than an
  oversight;
- an annotated member that a particular instance never set simply reaches
  the chosen overload as a value lacking that attribute, exactly as it
  would reach a plain function a type checker had already accepted it for.

#### The one gap the order cannot close

A class that declares none of a protocol's members at all — no annotation,
attribute, property, slot, method, or dataclass field — is incomparable
with that protocol at the hint level. Yet a particular instance of it can
still acquire the member at runtime, and so belong to the protocol at the
value level:

```pycon
>>> class Quiet:
...     def __init__(self, name):
...         self.name = name  # set on the instance, never declared
...
>>> issubhint(Quiet, HasName)
False
>>> ishintstance(Quiet("Bo"), HasName)
True

```

An overload on `Quiet` and an overload on `HasName` are then ambiguous for
an instance like `Quiet("Bo")`, and `AmbiguousMethodError` is the correct
answer, since neither overload is more specific than the other. No warning
fires at registration time, and `ambiguities()` reports nothing, because
both overloads are genuinely incomparable at the hint level, which is all
registration-time analysis can see. The remedy is the caller's: annotate
the member on the class, or give one of the two overloads a higher
`priority`. The guide's page on protocols shows this exact case running,
together with the more common one where the class wins outright.

Registration itself treats such a protocol like any other hint, so an
overload on it is reachable the normal way. A protocol that is not
`runtime_checkable` is unaffected by any of this: it simply answers `False`
at both levels, so an overload registered on it can never match a call.

---

## 3. TypeVars in signatures

| Kind | Applicability at a position | Hint used for specificity |
|---|---|---|
| unbound `T` | any value (`T` behaves as `Any`) | `Any` |
| bound by `B` | an instance of `B` | `B` |
| constrained to `(C1, C2)` | an instance of one `Ci` (`bool` solves as `int`) | `Union[C1, C2]` |
| carrying a `default=` (\[[PEP 696]\]) | as above — the default is a static-checker fallback and plays no role at runtime | as above |

The origin of a TypeVar does not matter to dispatch: a legacy
`tx.TypeVar(...)`, a variance-flagged `bagof.hints.typevars.*` variable, and
a [PEP 695] `def f[T]` variable are all plain `TypeVar` instances and are
dispatched on identically. `Signature.from_callable` reads a function's
hints alone; `__type_params__` is only consulted to name a variable in an
error message.

[PEP 696] defaults play no part in dispatch, for any of the three kinds above.
The relation reads a TypeVar's upper bound through a private helper that
tries its bound, then the union of its constraints, then falls back to
`Any`. It reads `__default__` through `getattr(tv, "__default__",
tx.NoDefault)` rather than the `.has_default()` method a native Python 3.12
[PEP 695] TypeVar lacks. Reading it this way also fixes a bug the summed-distance
predecessor had: `TypeVar("TB", bound=float, default=int)` used to read as
bound to `int`, its default, rather than `float`, its actual bound, because
the older lookup helper preferred the default over the bound.
`unwrap`'s own public contract still prefers default, then constraints,
then bound, in that order, because a factory that needs to *build* a value
legitimately wants the default; only the relation's own reading of a
TypeVar's upper bound ignores it.

#### Repeated TypeVars

A signature such as `def same(x: T, y: T)` requires consistency over its
**bound arguments**: the arguments whose landed hint is that same `T`,
meaning every declared parameter with that hint together with any surplus
positional absorbed by an `*args: T`. This holds regardless of whether the
call spells them positionally or by keyword — `same(1, 2)`, `same(x=1,
y=2)`, and `same(y=2, x=1)` are all checked the same way. A call is
applicable only when those argument types have a **greatest element** under
`⊑`. `(int, bool)` solves `T = int`, since `bool ⊑ int`, while `(int, str)`
has no such solution, because joining them to `object` would collapse `(T,
T)` into `(object, object)`, and the relation refuses to invent a join this
way. A constrained `T` instead requires that every bound argument resolve
to the *same* one of its constraints. A default-filled parameter annotated
`T` contributes nothing, since it is not an argument at all.

A `**kwargs: T` group solves `T` for applicability jointly, across the
keywords it captures together with every other slot carrying `T`. This is
the same greatest-element solve that `*args: T` performs over its
positionals, and those same keywords also form one group for the
specificity tie-break described below. This sits deliberately between two
other models. `Julia`'s strict diagonal dispatch lets a repeated TypeVar
additionally discriminate on which concrete type filled it. Mypy's static
join, on the other hand, would collapse `(T, T)` to the join of the two
argument types rather than requiring them to already share one.

A constrained TypeVar whose constraint is itself a runtime protocol with
data members is solved from the argument's *class*, through
`issubhint(type(v), Ci)`, at every position it fills. As a result, a value
that only belongs to the protocol through what its particular instance
holds does not select that constraint. Declaring the member on the class,
or using a plain `bound=`, which is checked value by value instead, is the
way to reach it. A generic protocol
with data members, such as `HasItem[int]` declaring `item: T`, checks only
that the members are present. Since a structural value declares no type
arguments of its own, what it actually holds under `item` is never compared
against `int`, the same way a plain list's items are never compared against
its declared element type.

#### Specificity with TypeVars

Position by position, a TypeVar is replaced
by the hint from the table above, so `(int, int)` is strictly below
`(T, T)`, which is equivalent to `(Any, Any)`. One further tie-break applies
inside that equivalence. When two methods are equivalent position by
position, the one whose repeated TypeVars group *strictly more* arguments
into one consistent type is the more specific of the two. This recovers the
outcome `Julia`'s own diagonal dispatch would give for the "same type" case.
This tie-break is the very last selection step, reached only when two
methods are already equally specific after priority, MRO refinement, and
tightness (§2.2) have all failed to separate them. It compares the two
methods' TypeVar groupings over the bound arguments, meaning which argument
positions share one TypeVar's identity, and one grouping wins only when it
strictly refines the other: it ties every pair the other ties, and at least
one pair more besides. Neither grouping refines the other when they are
equal, such as `(T, U)` against `(U, T)`, or when each ties a pair the
other does not, such as `(T, T, U)` against `(T, U, U)`; in either case the
two stay incomparable, and the call remains ambiguous. Because it only ever
applies once every earlier step has failed to decide, this tie-break can
never overturn a strict specificity win, and can never make two genuinely
independent signatures comparable; it only ever resolves a tie that would
otherwise be ambiguous. A signature with any repeated group beats one with
none at all, including a fully unannotated `(Any, Any)`; this is the least
surprising reading of "more constraint is more specific" for the cases this
section would otherwise leave silent. A group of exactly one argument constrains nothing on its own, so this
refinement only ever applies when a TypeVar is bound at two or more
positions of the actual call. `*args: T` against a plain `*args` is
refined, and therefore resolved, for a call of two or more arguments, but
the two tie for a call of zero or one argument, and are therefore ambiguous
there. `**kwargs: T` against a plain `**kwargs` behaves the same way over
the keywords a call spills into the catch-all: refined for two or more
captured keywords, tied for zero or one.

Hint-level `resolve` queries containing a TypeVar use `issubhint` unchanged;
there is no separate rule for them.

Only a **top-level** TypeVar, meaning one that names an argument's own
landed hint directly, is solved jointly across positions. A TypeVar nested
*inside* a hint, such as the `T` in `Box[T]` or `List[T]`, is solved
locally instead, once per position, inside `issubhint` itself, and never
jointly across positions. `(Box[T], Box[T])` still accepts `Box[int](),
Box[str]()` as two independent arguments, the same way `(List[T], List[T])`
accepts two lists that happen to declare different element types.

#### Variadic parameter kinds

`*args: *Ts` and `*args: P.args` are both read
as an `Any` tail for a single element. A bare `Ts` or `P`, a bare
`Concatenate[...]`, or a top-level `Unpack[Ts]`, are each invalid as a
*parameter* annotation and raise `TypeError` at registration (a bare `Ts`
written on `*args` produces an error pointing the caller at
`*args: Unpack[Ts]` instead). Two open runs inside one tuple or parameter
list, such as `Tuple[*Ts, *Us]`, are refused at registration too, enforcing
[PEP 646]'s single-unpack rule, which `typing` itself does not enforce at
runtime.

A `ParamSpec` is solved at the **hint level**. Named at several top-level
`Callable` slots, it must capture a consistent parameter list at each one —
the direct analogue of a repeated TypeVar, where consistency again means
the captured lists have a greatest element under the parameter-list order.

A `TypeVarTuple` is solved at the hint level the same way. Named at several
top-level `Tuple` slots, it must capture a consistent *run* at each: the
covariant, tuple-shaped analogue of the same idea, where consistency means
the captured runs have a greatest element under the tuple-shape order.
`(int,)` and `(bool,)` solve to `(int,)`. `(int,)` and `(str,)`, or two runs
of different length, are inconsistent, and a closed run alongside an open
one solves to the open one. A `*args: *Ts` absorbs its own positionals into
one run of that same `Ts`, solved jointly with every `Tuple[..., *Ts]` slot
elsewhere in the signature. So `(t: Tuple[*Ts], *args: *Ts)` applies to
`resolve(Tuple[int, str], int, str)` but not to `resolve(Tuple[int])`,
since the empty `*args` run there disagrees with `(int,)`. A
`Callable\[[int, *Ts], R]` parameter list rides the same open-tail machinery
`ParamSpec` uses, with its own `*Ts` tail solved by that machinery, keyed
separately from any tuple run of the same `Ts` elsewhere. Only a top-level,
landed `Tuple` or `Callable` joins a group this way. A `*Ts` nested inside
another hint does not, nor does a `*Ts` in the middle of a `Callable`
parameter list followed by a fixed suffix — `Callable\[[int, *Ts, str], R]`
degrades to an open `Concatenate[int, P]` shape instead. `*Ts` never enters the
repeated-TypeVar specificity tie-break above. It is not a `TypeVar` at all,
deliberately the opposite of how `*args: T` behaves, so `*args: *Ts` against
a plain `*args` stays ambiguous regardless of how many positional arguments
the call carries.

Value-level solving of `ParamSpec` and `TypeVarTuple` is out of scope for
now (§10): a callable or tuple value is matched shallowly, without ever
inspecting its actual signature or element shape.

---

## 4. "Exact type" vs. "subtype": `Exact[C]`

Ordinary type hints have no way to say "this class, and none of its
subclasses." `Exact[C]` fills that gap while still presenting as plain
`int` to a type checker. It is spelled `bagof.dispatchers.Exact[int]` and
implemented as `tx.Annotated[int, EXACT]` behind a private sentinel, the
same pattern `bagof.magic`'s `Frozen[int]` uses. It is understood directly
inside `issubhint` and `ishintstance`, rather than through a wrapper around
them, so anything built on the relation, including converters and
validators, gets exactness for free.

None of the tools already in the hint vocabulary can do this job. A bound
TypeVar is equivalent to its bound, not restricted to it. A TypeVar with a
single constraint is not legal, and one with two or more constraints still
accepts subclasses of each. Variance flags are rejected on a TypeVar used as
a function parameter. And `type[C]` constrains an argument that is itself a
class object, not the exactness of an ordinary instance. (`Julia` gets
exactness without any of this machinery, because its concrete types are
final by construction; Python's classes are open, so dispatch needs an
explicit marker.)

`Exact[C]` behaves as a **leaf subtype** of `C`, a genuine bottom sitting
just under `C` in the order, which is what keeps `⊑` a proper preorder
(reflexive and transitive) once `Exact` is added to it:

- At the **value level**, a call is applicable through an `Exact[C]`
  parameter only when `type(v) is C` exactly. Applying `Exact` to anything
  other than a class (aside from `NoneType`) raises `TypeError` at
  registration.
- At the **hint level**, `issub(Exact[C], C)` is true, since a value that is
  exactly `C` is certainly a `C`, but `issub(C, Exact[C])` is false, and so
  is `issub(D, Exact[C])` for any subclass `D` of `C`: nothing ordinary sits
  below `Exact[C]` at all. The only ordinary hints that do sit below it are
  literals every one of whose values happens to have type exactly `C`:
  `issub(Literal[v], Exact[C])` holds exactly when `type(v) is C`, so
  `Literal[1]` is below `Exact[int]`, but `Literal[True]`, a `bool`, is
  not. `issub(Exact[C1], Exact[C2])` holds exactly when `C1` and `C2` are the
  same class.
- In the wider **order**, `Exact[C]` sits strictly below `C`, and
  `issub(Exact[C], P)`, for any `P` that is not itself a union or a TypeVar
  containing `Exact`, reduces to `issub(C, P)`. A `P` that does contain
  `Exact`, such as a union member or a TypeVar's bound, is checked member by
  member instead of reducing this way, since reducing first would lose the
  exactness. `issub(Exact[C], Union[Exact[C], …])` and
  `issub(Exact[C], TypeVar(bound=Exact[C]))` are both true for exactly this
  reason. Two `Exact` hints over different classes are incomparable. MRO
  refinement (§2.2) treats `Exact[C]` as `C`, and it is keyed by class for
  the cache, so it caches exactly as a plain class hint would.
- For **resolution**, the hint-level `resolve()` may additionally reach an
  `Exact[C]` entry for a query equivalent to `C`. This is a lookup convenience
  layered on top of the relation, not a change to `⊑` itself, which still
  keeps a plain query `q` from being a sub-hint of `Exact[C]`.

```python
from bagof.dispatchers import dispatch, Exact

@dispatch
def describe(x: Exact[int]) -> str:  # exactly int, so not bool
    return "an integer"

@dispatch
def describe(x: object) -> str:      # everything else, True included
    return "something else"
```

`Exact` answers the reverse of the usual dispatch need: an `int` overload
that should not fire for `True`, or a base class whose subclasses each need
their own dedicated method. It lives in `bagof.dispatchers.core._exact`, and
is a candidate for promotion to `bagof-hints` if a package that does not
otherwise depend on `bagof.dispatchers` ever needs it.

### 4.1 Composing `Exact` inside `Type` and `Hint`

*(0.2.0)* `Exact` composes with `Type[…]` and with `Hint[…]` (§6) under one
rule: placing `Exact` **inside** the bracket flips that position's match
from subtype to identity, while `Exact` around the **whole** form means the
same thing and is normalised to the inner spelling. `Exact[Type[int]\]`
normalises to `Type[Exact[int]\]`, `Exact[Hint[int]\]` to `Hint[Exact[int]\]`,
and a bare `Exact[Type]` to `Exact[type]`.

The leaf rules then compose automatically, because both `type[…]` and
`Hint[…]` compare their argument through `issubhint`. So `type[Exact[C]\]` is
a leaf below `type[C]` exactly as `Exact[C]` is a leaf below `C`:
`issub(type[Exact[C]\], type[C])` is true, `issub(type[C], type[Exact[C]\])`
is false, and `issub(type[Exact[C1]\], type[Exact[C2]\])` holds only when
`C1` and `C2` are the same class. `Hint[Exact[X]\]` behaves the same way
within the `Hint` order. At the value level, `type[Exact[C]\]` matches a
class `v` only when `v is C`, and `Hint[Exact[X]\]` matches a hint only when
it is structurally `X` itself, judged by identity of spelling rather than
by equivalence, so a free `TypeVar` — equivalent to `Any` but not the hint
`Any` — does not match `Hint[Exact[Any]\]`.

Reading `type[…]`'s argument through the full relation also fixes the forms
that were degenerate before: `type[Any]` now accepts every class,
`type[Union[…]\]` accepts a class in the union, and `type[T]` for a free
TypeVar accepts every class.

### 4.2 Bounds: `Super[C]` and `Between[L, U]`

*(0.3.0)* A hint on a value parameter constrains the value's own class
exactly as `Type[hint]` constrains a class passed in. A plain `C` on a
value accepts `v` when `type(v) ≤ C`, which is `Type[C]` asked of
`type(v)`, and `Exact[C]` on a value is `Type[Exact[C]\]` asked of
`type(v)`. `Super[C]` and `Between[L, U]` complete that family, both on a
value and inside `Type` or `Hint`.

An ordinary hint is an upper bound: `Animal` accepts an instance of
`Animal` or of a class below it, and `Type[Animal]` accepts `Animal` and
the classes below it. `Super[C]` supplies the missing lower bound.
`Super[Dog]` accepts a value whose class is `Dog` or a class that `Dog`
derives from, `Type[Super[Dog]\]` accepts the class `Dog` and every class
that `Dog` derives from, and `Hint[Super[int]\]` accepts the hint `int`
and every hint that `int` is a sub-hint of. It is spelled
`bagof.dispatchers.Super[C]` and implemented, like `Exact`, as
`tx.Annotated[C, SUPER]` behind a private sentinel. `SuperType[C]` and
`SuperHint[X]` are aliases that expand to exactly `Type[Super[C]\]` and
`Hint[Super[X]\]`.

`Between[L, U]` supplies both bounds at once. `Between[Dog, Animal]`
accepts a value whose class lies from `Dog` up to `Animal`, both included,
`Type[Between[Dog, Animal]\]` accepts those classes themselves, and
`Hint[Between[int, Real]\]` accepts the hints from `int` up to
`numbers.Real`. It is spelled `bagof.dispatchers.Between[L, U]` and
implemented as `tx.Annotated[U, LOWER(L)]`, the upper bound annotated with
a private marker that carries the lower one. There are no `BetweenType` or
`BetweenHint` aliases, since `Type[Between[Dog, Animal]\]` already reads
naturally.

The relation reads each of these forms as a closed interval of the
classes, or hints, it accepts. A plain `X` is the interval from the bottom
`Never` up to `X`, `Super[C]` is the interval from `C` up to the top, which
is `object` on a value and inside `Type`, and `Any` inside `Hint`, and
`Between[L, U]` is the interval from `L` up to `U`. `Type[X]` is therefore
the same as `Type[Between[Never, X]\]`, and `Type[Super[C]\]` the same as
`Type[Between[C, object]\]`; on a value, `X` is the same as
`Between[Never, X]` and `Super[C]` the same as `Between[C, object]`.
`Exact[C]` is the point below `Between[C, C]`: it names the one hint `C`
structurally, while `Between[C, C]` is the interval of every hint
equivalent to `C`. On a value that difference is between a class
identical to `C` and a class that `register` or `__subclasshook__` has
made mutually a subclass of `C`. One form is below another when its
interval lies inside the other's, meaning that its lower bound is higher
and its upper bound is lower. The rules that follow from that reading are
these:

| Query | Result | Why |
|---|---|---|
| `Type[Super[D]\] ≤ Type[Super[C]\]` | iff `C ≤ D` | a higher lower bound leaves fewer classes above it |
| `Type[Exact[X]\] ≤ Type[Super[C]\]` | iff `C ≤ X` | the one class `X` is in the interval when it is above `C` |
| `Type[X] ≤ Type[Super[C]\]` | False | a subclass of `X` can always be defined that is not above `C` |
| `Type[Super[C]\] ≤ Type[X]` | iff `object ≤ X` | the interval reaches `object`, so only a top contains it |
| `Type[Super[C]\] ≤ Type[Exact[X]\]` | False | conservative: an interval above `C` is never taken to be a single class |
| `Type[Between[L1, U1]\] ≤ Type[Between[L2, U2]\]` | iff `L2 ≤ L1` and `U1 ≤ U2` | the first interval lies inside the second; every row above is a special case of this one |

`Hint` follows the same table with `Any` in place of `object`. On a value
parameter the same table holds with `type(v)` in place of the class passed
and `object` as the top, so `Between[Dog, Animal] < Animal`, `Exact[Dog]`
is below every interval that contains `Dog`, and `Dog` and `Super[Dog]`
are incomparable. A union below a bound distributes over its members, a
`TypeVar` is read as its upper bound, and a `Literal` is below a bound
when each of its values is accepted, so `Literal[1] ≤ Super[int]` while
`Literal[True]` is not. Every rule is an inclusion of intervals, so the
relation stays a preorder, and it stays sound: when `A ≤ B`, every class,
hint, or value in `A` is also in `B`. Values are matched against the same
interval (§2.1), which keeps the two levels in agreement.

On a value, each bound must be a hint that a class can be compared
against: for every class `K`, `K ≤ E` must hold exactly when an instance
of `K` is `in E`. A class, an ABC, `object`, `type`, `NoneType`, a
`NewType`, a bare alias such as `List`, a protocol with methods only,
`Never`, `Any`, a union of such hints, and a `TypeVar` bounded by one all
qualify. A `Literal`, a `TypedDict`, a protocol with data members, a
parametrised generic, `Tuple[...]`, `Callable[[...], R]`, and the `Type`
and `Hint` forms do not, because each of them reads the value itself or
names no class, so `Between[Never, E]` would accept nothing while `E`
accepts values. Such a bound on a value is refused wherever it is read,
naming the offending bound; `Between[Type[A], Type[B]\]` is refused with a
message naming `Type[Between[A, B]\]`. Inside `Type` or `Hint`, where the
value passed is itself a class or a hint, any hint can be a bound, so the
same `Between[Literal[1], int]` that is refused on a value is legal inside
`Hint[...]`.

An interval must not be empty. `Between[L, U]` requires `L ≤ U`, and an
empty one is refused when it is written, with a message that suggests
`Between[U, L]` when that spelling is not empty. `Between[Any, U]` is
refused with a message pointing to `Between[Never, U]`, the interval with
no lower bound, which means the same as plain `U`. An upper bound written
as a forward reference is checked once it resolves, when the method is
registered or when its signature is first used, and the refusal then
names the parameter. A lower bound cannot itself be a quoted forward
reference, because it is carried as `Annotated` metadata, which
`get_type_hints` never resolves; quoting the whole annotation works as
usual. The degenerate intervals are legal: `Between[C, C]` holds the hints
equivalent to `C`, and `Between[Never, Never]`, `Between[Never, Any]`, and
`Between[Any, Any]` mean what their bounds say.

A bound stands in four kinds of position. On a value parameter it may be
the whole hint, a member of the parameter's union, as in
`Optional[Super[Dog]\]`, or the bound of a `TypeVar` used there. Inside
`Type` or `Hint` it must be the whole argument: a union member or a
`TypeVar` bound inside the argument is refused, because the argument is
compared as a class or a hint, which would read the bound as if it stood
on a value. As the type argument of a generic it must likewise be the
whole argument of a slot, where it names a range of arguments, and the
variance of the slot decides whether that range means anything (§4.3).
Everywhere else it is refused: as a constraint of a `TypeVar`, which is
solved to the one constraint an argument's class is below and so would
never apply, and as an element of a `Tuple` or in the signature of a
`Callable`, whose variance the form already fixes. These positions nest,
so a bound inside a generic inside a `Callable` parameter, as in
`Callable[[List[Super[int]\]], None]`, stands at a slot of `List` and is
legal. Registration refuses each misplaced bound with an error naming the
parameter and the spelling to write instead, and the relation refuses the
same hints with the same messages. An unsubscripted `Super`, `SuperType`,
`SuperHint`, or `Between` is refused too, since an unbounded form names nothing. `Super[Type[C]\]`
is normalised to `Type[Super[C]\]` and `Super[Super[C]\]` collapses to
`Super[C]`. `Exact`, `Super`, and `Between` cannot be nested inside one
another in any order, even through a union or a `TypeVar`, because an
exact type leaves nothing above it to bound and each form already
describes the whole hint at its position.

The degenerate bounds need no special case. On a value, `Super[Never]`
accepts exactly what `object` accepts, `Super[object]` accepts only an
instance of `object` itself, and `Super[Any]` accepts no value, because no
class is above `Any`. In the same way `Type[Super[Never]\]` accepts every
class, like `type`, `Type[Super[object]\]` accepts only `object`, and
`Type[Super[Any]\]` accepts no class at all, while `Hint[Super[Any]\]`
accepts the tops: `Any`, a free `TypeVar`, and an opaque form.

To a type checker, `Super[C]` is `Union[C, Any]`. `Exact[C]` can present as
plain `C`, but `Type[Super[Dog]\]` read as `Type[Dog]` would reject a call
with `Animal` that dispatch accepts. `Union[C, Any]` is the sound generic
spelling: it uses its type variable, which a checker requires of a generic
alias, and `Type[Union[C, Any]\]` accepts every class object, so a checker
never rejects a call that the runtime accepts. The aliases are declared
with a \[[PEP 613]\] `TypeAlias` annotation, which both mypy and pyright
read. `Between[L, U]` is `Union[L, U, Any]` for the same reason. Checked
with mypy 1.19.1 and pyright 1.1.408, neither rejects a call with `Dog`,
`Animal`, `object`, or `Puppy` on a `Type[Between[Dog, Animal]\]`
parameter, nor one with `Animal`, `object`, `Dog`, `Puppy`, or `int` on a
`Type[Super[Dog]\]` parameter, nor any call on a value parameter
annotated `Super[Dog]`, `Between[Dog, Animal]`, or
`Optional[Super[Dog]\]`: both over-accept, and neither rejects a call that
dispatch accepts. Inside the function body the same reading has a cost: a
checker reads `x: Super[Dog]` as `Union[Dog, Any]`, and so checks
attribute access against `Dog` although the runtime may pass an
`Animal()`. `Between[Dog, Animal]` reads as `Union[Dog, Animal, Any]`,
whose shared attributes are those of `Animal`, which is right. When the
difference matters, a `Super[C]` value is best treated as `object` in the
body.

Two things are deliberately left out. There is no option to flip the
direction in which `Type` orders its argument; a lower bound is written
where it is meant. And a contravariant `TypeVar` is read exactly as before
(§2.3): `Super` bounds a hint, and is not a spelling of a generic's
contravariant parameter.

Bounding a value's class from below has little precedent. Julia can write
`f(x::T) where {T>:Dog}`, but its concrete types are final, so
`typeof(x) >: Dog` collapses to `typeof(x) == Dog` for a concrete `Dog`,
and to nothing for an abstract one. Java's `? super L` and Kotlin's `in`
exist only on the type arguments of a generic, never on a value, and C++
and Rust have no equivalent. Python's classes are open, which is what
gives an interval of a value's class real content. `Exact`, `Super`, and
`Between` on a value are one family of exact-class-interval tests, with
`Exact` the point and the other two the half-open and closed intervals,
and each is a deliberate opt-out of substitutability: `Exact[C]` excludes
the instances of subclasses of `C`, and `Super[C]` and `Between[L, U]`
exclude the instances of classes below `L`. The type arguments of a
generic are where bounds do have a long history, which §4.3 compares.

### 4.3 Bounds as type arguments

*(0.3.0)* An argument at an invariant slot of a generic means exactly that
type: `List[int]` accepts a list declared to hold `int`, and neither one
declared to hold `bool` nor one declared to hold `object`. A bound written
as the whole argument of such a slot widens it into a range of arguments.
`List[Super[int]\]` stands for every `List[Y]` with `int ≤ Y`, so it
accepts a list declared to hold `int`, `numbers.Integral`, or `object`, and
refuses one declared to hold `bool` or `str`.
`Box[Between[Never, Integral]\]` stands for every `Box[Y]` with
`Y ≤ Integral`. A value that declares nothing, such as a plain `[1]`,
matches every parametrisation, as it always has (§2.1), and a value that
declares a bounded parametrisation, such as `Box[Super[int]\]()`, is read
as the range it names.

Each bound is an anonymous range of its own. Two bounds in one signature
are independent of each other, and a bound nested inside an argument does
not widen the slot around it: `List[List[Super[int]\]\]` has the single
argument `List[Super[int]\]` at its outer, invariant slot, so it accepts a
list declared to hold `List[Super[int]\]` and not one declared to hold
`List[int]`. The range of lists whose items are below `List[Super[int]\]`
is written `List[Between[Never, List[Super[int]\]\]\]`. Every end of a
bound at a slot may be any hint, because a slot compares hint with hint,
so the `Between[Literal[1], int]` that is refused on a value (§4.2) is
legal in `List[...]`.

The order is inclusion of ranges, as in §4.2, with `Any` as the top. At an
invariant slot a plain `X` is the single point from `X` to `X`,
`Super[C]` runs from `C` to `Any`, `Between[L, U]` from `L` to `U`, `Any`
and a free TypeVar from `Never` to `Any`, and a TypeVar bounded by `B` from
`Never` to `B`, which is the reading §2.3 already gives it. A constrained
TypeVar stands for one of its constraints, each a point. `Exact[C]` is an
ordinary point there, because the exact hint `C` is a narrower argument
than `C`: `List[Exact[int]\]` is below `List[Between[Never, int]\]` and
not below `List[int]` or `List[Super[int]\]`. The rules that follow are
these, with `TB` bounded by `int`:

| Query | Result | Why |
|---|---|---|
| `List[int] ≤ List[Super[int]\]`, `List[object] ≤ List[Super[int]\]` | True / True | the one argument lies in the range above `int` |
| `List[bool] ≤ List[Super[int]\]`, `List[Any] ≤ List[Super[int]\]` | **False** / **False** | `bool` is below `int`, and `Any` on the sub side stands for every argument |
| `List[Super[Integral]\] ≤ List[Super[int]\]` | True | a higher lower end leaves a smaller range |
| `List[Super[int]\] ≤ List[int]`, `List[Super[int]\] ≤ List[Any]` | **False** / True | a range is below a point only when it is that point, and below `Any` always |
| `List[TB] ≡ List[Between[Never, int]\]`, `List[Between[int, int]\] ≡ List[int]`, `List[Between[Never, Any]\] ≡ List[Any]` | True | the same range, written three ways each |
| `List[Between[int, Real]\] ≤ List[Super[int]\]`, `List[Between[Never, int]\] ≤ List[Super[int]\]` | True / **False** | the first range lies above `int`, the second reaches below it |

A covariant or contravariant slot already widens its argument, so a range
there collapses to one of its ends. At a covariant slot every `G[Y]` with
`Y ≤ U` is below `G[U]`, so a range means its upper end, and at a
contravariant slot every `G[Y]` with `Y ≥ L` is below `G[L]`, so a range
means its lower end. A bound whose other end is trivial says exactly that
and is accepted as redundant, while a bound whose other end would be
silently dropped is refused as conflicting, with a message naming the plain
spelling:

| Slot | `Between[Never, U]` | `Super[L]`, `Between[L, Any]` | `Between[L, U]`, both ends non-trivial |
|---|---|---|---|
| invariant: `list`, both `dict` slots, `set`, `MutableSequence`, a `Mapping` key, an unflagged `T`, a [PEP 585] subclass | arguments at or below `U` | arguments at or above `L` | arguments from `L` up to `U` |
| covariant: `Sequence`, `frozenset`, `Iterable`, a `Mapping` value, a `T_co` | redundant: `G[U]` | refused: would accept every `G` | refused: write `G[U]` |
| contravariant: a `T_contra`, the send slot of `Generator` | refused: would accept every `G` | redundant: `G[L]` | refused: write `G[L]` |

A bound that a user generic's invariant slot accepts can reach a variant
slot of a base, as `Row[Super[int]\]` does for `class Row(Sequence[T])`. It
was written where it can be read, so it is not refused there; it stands for
every `Row[Y]` in its range, and so reaches `Sequence` through that range's
upper end, which makes `Row[Super[int]\]` below `Sequence[Any]` but not
below `Sequence[int]`, and `Dict[str, Between[Never, int]\]` below
`Mapping[str, int]`. A class that writes a bound directly into a variant
base, such as `class Row(Sequence[Super[int]\])`, is refused whenever it is
compared against a parametrisation of that base, with the message the hint
`Sequence[Super[int]\]` would get, prefixed with the class. A bound that
reaches a base only as part of a union, or at a slot whose variance cannot
be read, is refused as well.

A bound must be the whole argument of its slot. A union member or a TypeVar
bound inside a slot, as in `List[Union[Super[int], str]\]`, is refused with
a message suggesting `Union[List[Super[int]\], List[str]\]`, and a bound is
refused as an argument of a generic whose variance cannot be read, such as
one with a `ParamSpec` or a `TypeVarTuple` parameter. `Tuple` elements and
`Callable` parameters and returns keep refusing a bound (§4.2), since the
form already fixes their variance and a bound could at best repeat it.
Registration, the relation, and `ishintstance` refuse the same hints with
the same messages, so `ishintstance([], Sequence[Super[int]\])` raises
although a plain list declares nothing.

Specificity stays the sub-hint order, with no separate heuristic and no
option to change it; `priority` settles what the order leaves tied. Two
ranges that overlap without either lying inside the other, such as
`List[Between[Never, Integral]\]` and `List[Super[int]\]`, which share the
parametrisations `List[int]` and `List[Integral]`, make their methods
ambiguous, and registering both warns (§5).

To a type checker the aliases of §4.2 still apply, so `List[Super[int]\]`
reads as `List[Union[int, Any]\]`. Checked with mypy 1.19.1 and pyright
1.1.408, with `class Dog(Animal)` and `class Puppy(Dog)`: pyright reports
no error for any call to a parameter annotated `List[Super[Dog]\]`,
`List[Between[Never, Animal]\]`, `List[Between[Puppy, Animal]\]`, or
`Dict[str, Super[Dog]\]`, so it never rejects a call that dispatch
accepts, and it over-accepts. mypy is exact for a lower bound, accepting
`list[Dog]`, `list[Animal]`, `list[object]`, and `list[Any]` for
`List[Super[Dog]\]` and rejecting `list[Puppy]` and `list[str]`, and it is
exact for `Dict[str, Super[Dog]\]`. For an upper bound it is not: mypy
accepts `list[X]` against `list[Union[A, B, Any]\]` only when `A ≤ X` and
`B ≤ X`, reading every member as a lower bound, so it rejects `list[Puppy]`
and `list[Dog]` for `List[Between[Never, Animal]\]` and for
`List[Between[Puppy, Animal]\]`, calls that dispatch accepts. No generic
alias spelling avoids this, and a bounded TypeVar is the spelling of an
upper bound that both checkers check natively: `List[A]` with
`A = TypeVar("A", bound=Animal)` is checked exactly by both, and dispatch
reads it as `List[Between[Never, Animal]\]`. `Super` and `Between` serve
for lower and two-sided bounds, and mypy cannot check the upper end of a
two-sided bound at an invariant slot. Where mypy's rejection gets in the
way, the function can be annotated `List[Any]` and registered with
`Function.register((List[Between[Puppy, Animal]\],))`, or the call can
carry `# type: ignore[arg-type]`.

Bounds on type arguments are well established elsewhere. Julia bounds a
type parameter from above and below, as in `Vector{>:Int}` or
`T where Int<:T<:Real`, and its parameters are all invariant, so every slot
takes a bound; its method specificity, though, comes from a separate
`morespecific` heuristic with known non-transitive cases, where here it is
the sub-hint order, whose preorder argument is the whole story. Java's
wildcards `? super L` and `? extends U` exist because all its generics are
invariant, and never combine both ends. Kotlin's use-site projections are
the closest match, because Kotlin, like Python, also has declaration-site
variance: an `in` projection on an `out` parameter is an error, as a
conflicting bound is refused here, and an `out` projection on an `out`
parameter is a redundant-projection warning, where here a redundant bound
is accepted silently, so that code that builds hints generically does not
trip over it. Nested bounds behave as in Java and Julia, where
`List<List<? super Integer>>` and `Vector{Vector{>:Int}}` do not accept a
list of lists of integers either.

---

## 5. Ambiguity and no-match errors

Every dispatch error lives in `_errors.py`. `DispatchError` is the common
base, and it subclasses `TypeError`, since calling a dispatched function
with types nothing supports is, after all, a type error, and existing
`except TypeError:` handlers keep working unchanged. `NoMethodError` and
`AmbiguousMethodError` both subclass it, and both carry `.function`, `.call`
(the argument classes for a value call, or the queried hints for
`resolve`), and `.candidates`. Problems detected at registration time,
rather than at a call, raise a plain `TypeError` or `ValueError` instead.

An ambiguity is reported in `Julia`'s own format, printing classes rather than
values, so the message never has to render an arbitrary `repr`:

```
AmbiguousMethodError: area(int, int) is ambiguous.
Candidates:
  area(x: int, y: Any) @ shapes.py:12
  area(x: Any, y: int) @ shapes.py:15
Possible fix, define
  area(x: int, y: int)
```

The suggested fix is the signature built from the call's own classes
(substituting `Exact[...]` wherever a candidate already used it), always
applicable to the call that triggered the error, and always strictly more
specific than every one of the listed candidates. This is a simpler,
always-correct stand-in for `Julia`'s own type-intersection suggestion, which
can itself be more specific than necessary.

A failed match is reported the same way, marking the offending arguments
with `!`:

```
NoMethodError: no method matching area(str).
`area` has 3 methods, but none is defined for this combination of argument types.
Closest candidates:
  area(x: !int, y: int = 0) @ shapes.py:12
  area(x: !float) @ shapes.py:20
```

"Closest" ranks candidates by how many positions are applicable and how
well the call's arity matches, showing at most eight. When a function has
no methods registered at all, the message instead reads "`area` has no
methods yet: the module that defines them has not been imported." Both
errors are
raised from `Function.dispatch` or `Function.resolve`, never from inside the
`__call__` frame itself, so a traceback does not point into the dispatch
machinery. `Function.ambiguities()`, the analogue of `Julia`'s
`detect_ambiguities`, returns every pair of registered, incomparable
methods that could both apply to some call, so a test suite can assert that
a registry has none.

Registration only ever *warns* about a guaranteed ambiguity, never raises,
so the order in which modules happen to be imported can never break a
program that was working before. A pair separated by an explicit, differing
`priority` is resolved deterministically at every call, since the higher
priority always wins, so it is neither warned about at registration nor
listed by `ambiguities()`. A same-origin parametrised pair, such as `List[int]` and `List[str]`
registered side by side, is likewise not warned about or listed.
Registering both is entirely legitimate, since a plain `list` value applies
to both, and the incomparability between them only becomes visible at an
actual call. A value that declares its own parametrisation — an instance of
`Child(List[int])`, or one built as `Box[int]()` or `GL[int]()` — picks the
right one outright, and only a value that declares nothing at all is
genuinely ambiguous between them.

*(0.3.0)* A guaranteed ambiguity is recognised by checking that, at every
argument, the two methods' hints share a value, since a call built from one
shared value per argument then matches both, and neither method is more
specific. Two hints ordered against each other always share the narrower
one's values. A lower bound adds pairs that share values without being
ordered: `Type[Animal]` and `Type[Super[Dog]\]` are incomparable, yet both
accept `Dog` and `Animal`, and `Type[Dog]` and
`Type[Between[Dog, Animal]\]` both accept `Dog`. On a value the same
holds for `Animal` and `Super[Dog]`, which share a `Dog()` and an
`Animal()`, and for `Dog` and `Between[Dog, Animal]`, which share a
`Dog()`. Such a pair is recognised by reading each hint, or each `Type` or
`Hint` argument, as an interval (§4.2) and asking whether some candidate
lies in both intervals. The candidates are the ends of the two intervals
and, on a value or inside `Type`, every class on the MRO of a lower end
that is a class, because a class shared by both intervals lies above both
lower ends. That is how `Type[Between[A, C]\]` and `Type[Between[B, D]\]` are
found to share `X`, for a class `X(C, D)` with subclasses `A` and `B`,
although no end of either interval lies in the other. There a candidate
must also be a class, since the intervals hold classes. The
test is sufficient, so a reported overlap always has a real call behind it,
and for nominal class hierarchies, in which no class is made a subclass
through `register` or `__subclasshook__`, it is also complete when every
bound is a class. It is only ever consulted for a pair that involves a lower
bound, which leaves every other warning as it was. Registering
`Type[Animal]` and `Type[Super[Dog]\]` side by side therefore warns, and so
does registering `Animal` and `Super[Dog]`. A method for
`Type[Exact[Dog]\]` settles the call with `Dog`, but not the one with
`Animal`, while a `priority` on either method settles both; on a value,
`Exact[Dog]` settles the call with a `Dog()` in the same way.

*(0.3.0)* A bound in a type argument (§4.3) adds pairs of the same kind.
`List[Between[Never, Integral]\]` and `List[Super[int]\]` are unordered,
yet a list declared to hold `int` belongs to both. Such a pair is
recognised by looking for a parametrisation below both hints: one is built
on the more derived of the two origins, with each argument replaced in turn
by an end of the range it names or, for a shared origin, by the argument
the other hint gives there, and each candidate is checked against both
hints by the relation itself. The test is therefore sufficient, so a
reported overlap always has a real value behind it, and it is only
consulted when a bound stands among the arguments, which keeps
`List[int]` and `List[str]` silent as before.

The MRO tie-break (§2.2) does not apply to a bound. A bound names a range
of classes rather than one position in the value's MRO, just as
`Type[C]` does, so `mro_index` gives no refinement for it, and a call with
a `Dog()` stays ambiguous between `Animal` and `Super[Dog]` exactly as a
call with `Dog` is between `Type[Animal]` and `Type[Super[Dog]\]`. When such
a call raises, the suggested fix spells the argument `Exact[Dog]`, because
the exact class is below every competitor that accepts the value, a bound
included, while the plain class `Dog` is not below `Super[Dog]`.

---

## 6. Public API

```python
from bagof.dispatchers import dispatch

@dispatch
def area(shape: Circle) -> float:
    return 3.14159 * shape.r ** 2

@dispatch
def area(shape: Rect) -> float:
    return shape.w * shape.h

area(Circle(1))
# 3.14159
```

Redefining `area` a second time in the same module adds a method to the
existing generic function rather than rebinding the name to a fresh one.

#### Registries and `Function` identity

A registry is a `Dispatcher`
instance, and what identifies one `Function`, a named group of methods,
inside a given registry depends on which registry it is. The module-level
`bagof.dispatchers.dispatch` identifies a `Function` by
`(defining module, __qualname__)`, so the same name defined in two different
modules produces two entirely independent functions, sharing no methods. A
`Dispatcher()` constructed directly instead identifies a `Function` by
`__qualname__` alone, independent of which module registers into it, so
every module that registers `area` into the *same* constructed instance
extends *one* shared `Function`. This is how a shared, cross-module generic
function is built.

A `Function` is reached through its dispatcher's `functions` namespace,
either `d.functions["area"]` by item or `d.functions.area` by attribute.
Both of these get the existing `Function` or create it on first access, so
`area = d.functions.area` written in one module composes correctly with
registrations made from another. `@dispatch` also returns the `Function` it
just extended. A method is added either with `@dispatch` decorating
`def area(...)` directly (which takes the name from the `def`), or with
`@area.register` or `@area.dispatch` decorating any callable, including one
defined as `def _(...)`, whose own name is then ignored. An explicit
signature can be laid over the wrapped callable: positional hints are given
as a tuple, named hints as a dict, and any keyword arguments passed
alongside them are registration options such as `priority`, never hints.
For example, `area.register((int,), {"scale": float}, priority=0)(callable)`.
Passing a class as the implementation, as in `area.register(SomeClass)`,
dispatches on that class's `__init__` or `__new__`. The signature-only
form with no wrapped function at all, whether `from_hints` or
`from_mapping` given a tuple key, is positional-only by construction. To
declare a `/`, a `*`, an `*args`, or a `**kwargs` without writing a
function, pass a `Signature` object directly as the `from_mapping` key
instead.

`d.functions` is a **protocol-only** namespace: it supports the mapping
protocol (`d.functions["area"]`, `d.functions.area`, iterating over it by
name, `len(d.functions)`, and `name in d.functions`), and it defines no
named methods of its own. So *every* function name, including `register`,
`items`, or `map`, is safe to register there, while the `Dispatcher` itself
keeps its ordinary methods (`d.register(...)`, `d.clear_cache()`) with no
risk of collision. Attribute access ignores any name starting with an
underscore, so that a tool or REPL probing the object for dunder or private
attributes never accidentally creates an empty `Function`.

Which key this namespace uses follows the identity rule above. On a
constructed `Dispatcher()` it is the bare name, so `d.functions.area` is
unambiguous. On the module-level `dispatch`, the key also carries the
module, so there the `Function` is taken from the decorator's own return
value, or from a qualified lookup, rather than a bare-name attribute.

```python
# registry.py -- a shared, cross-module generic function
from bagof.dispatchers import Dispatcher
dispatch = Dispatcher()
area = dispatch.functions.area       # the (initially empty) Function

# shapes.py
from registry import dispatch
@dispatch
def area(s: Circle) -> float: ...    # extends registry's area

# app.py
from registry import area
area(Circle(1))                      # sees shapes.py's method
```

#### Surface

The package exposes two namespaces with disjoint object sets.
`bagof.dispatchers` (its `__all__`, and the intended public API) holds
`dispatch`, `Dispatcher`, `Function`, `Method`, `Signature`, `Parameter`,
`Exact`, `Hint`, `Super`, `SuperType`, `SuperHint`, `Between`,
`DispatchError`, `NoMethodError`, and `AmbiguousMethodError`. `bagof.dispatchers.core` (also its own `__all__`)
holds the relation and introspection helpers that the rest of the family
reuses: `issubhint`, `ishintstance`, `ishint`, `resolve_hint`,
`safe_get_origin`, `safe_get_args`, `get_origin_uw`, `get_args_uw`,
`unwrap`, `normalise_hint`, `is_typeddict`, `typeddict_required_keys`,
`safe_issubclass`, `safe_isinstance`, `issubclassable`, `issubscriptable`,
`get_concrete_type`, `type2hint`, `eq_safenan`, `Unset`, `UNSET`,
`NoneType`, `UnionType`, and `UNION_TYPES`. `Exact` and `Hint` are
documented once each, under the top-level API, since they are the objects
both namespaces need. `issubhint` and `ishintstance` understand them
directly, but they belong conceptually to the dispatch surface. `ishint`,
which reports whether an object is a type hint at all, is a `.core` helper
only, since it is a building block rather than part of the dispatch API.
*(0.2.0)* `Hint` and `ishint` are new in this release. *(0.3.0)* `Super`,
`SuperType`, `SuperHint`, and `Between` are new in this release; like
`Exact` and `Hint`, they are documented once, under the top-level API, and
understood directly by the relation in `.core`.

Key objects:

- **`Dispatcher()`** is a registry, identifying its `Function`s by the rule
  above. Reach them through `d.functions` (`d.functions.area`,
  `d.functions["area"]`, iterable by name).
- **`Function`** supports `__call__(*args, **kwargs)`;
  `dispatch(*args, **kwargs) -> Method`, which binds and selects without
  calling; `resolve(*hints, **named_hints, default=UNSET,
  ambiguity="raise") -> Method`; `register(...)`; `from_mapping(mapping)`;
  `methods`; `ambiguities()`; `clear_cache()`; and `__get__`, which binds
  `self` or `cls` as argument zero (unannotated, so it is `Any` by default).
  It carries `functools.update_wrapper` metadata and is deliberately not a
  `dict` subclass. `Function(dispatch_defaults=True)` treats a
  default-filled dispatched parameter as if the caller had written its
  default value out explicitly, the same rule `bagof.magic._polymorph`
  already uses (§8.2), and is off by default.
- *(0.3.0)* **`Function`** also reports what selection sees for a call
  without acting on it. `candidates(*args, **kwargs) -> Tuple[Method, ...]`
  returns every method that accepts the call, most specific first, and
  `bestcandidates(*args, **kwargs) -> Tuple[Method, ...]` returns the
  methods that survive every tie-break of §2.2 and §3. That is one method
  when `dispatch` would succeed, the tied methods when `dispatch` would
  raise `AmbiguousMethodError` (whose `candidates` attribute holds the same
  tuple), and no method at all when nothing applies. `itercandidates` and
  `iterbestcandidates` return the same sequences as iterators, while
  `resolve_candidates(*hints, **named_hints)` and
  `resolve_bestcandidates(*hints, **named_hints)` are the hint-level
  counterparts, reaching exactly the methods `resolve` reaches. None of the
  six raises `NoMethodError` or `AmbiguousMethodError`. The order of
  `candidates` is built in layers: the best candidates come first, then
  each layer holds the methods that nothing still unplaced is strictly
  more specific than, ordered by descending priority and then by
  registration order. A method therefore always follows every method that
  is strictly more specific than it, but the relative position of two
  incomparable methods is not a claim about specificity. The same
  question is answered by [CLOS]'s `compute-applicable-methods`, which
  returns the applicable methods sorted by precedence, by [Julia]'s
  `methods(f, types)`, and by [`multipledispatch`]'s
  `Dispatcher.dispatch_iter`, which yields the matching implementations
  in order of specificity.
- **`Method`** exposes `signature`, `function`, `priority`, `__call__`, and
  `__repr__`.
- **`Parameter(name, hint, kind, default)`** is frozen; `kind` mirrors
  `inspect.Parameter.kind`, and `required` is `default is Parameter.empty`.
- **`Signature`** exposes an ordered `parameters: Mapping[str, Parameter]`,
  plus `.varargs`, `.varkw`, and `.dispatched_names`. `from_callable(fn)`
  reads hints through `tx.get_type_hints(fn, include_extras=True)` and
  names, kinds, and defaults through `inspect.signature` (an unannotated
  parameter reads as `Any`; keyword-only parameters are dispatched by name
  like any other). `from_hints(*hints, **named_hints)` builds one for
  explicit registration, as in `@dispatch((int,), {"scale": float})`.
  `bind(args, kwargs) -> Optional[Binding]` and `le(other, shape)`, which
  gives specificity for one shape, round out the interface.
- **`bagof.dispatchers.core.resolve_hint(hint, mapping, *, default=UNSET,
  ambiguity="raise")`** is the hint-level functional API, the direct
  successor to `get_from_registry` (§8.1). It lives under `.core` because
  that is what `bagof-core-magic` itself reuses.

The naive dispatcher's `pre_check`, `post_check`, `__apply__`, and
`__error__` hooks are deliberately dropped: ordinary decorators cover the
same need, `dispatch()` and `resolve()` are the only dispatch-specific steps
worth exposing, and `resolve(default=...)` covers what `__error__` used to
be for. Subclassing `Function` or `Dispatcher` is supported, in
anticipation of `bagof.magic`'s own eventual field-name dispatch needs.

Forward references are handled the way `bagof.magic._resolve._Deferred`
already does: a raw annotation that raises `NameError` or `TypeError` when
first read is kept as-is and retried on the function's first dispatch; if
it is still unresolvable then, the error names the function and the
parameter. On Python 3.14 and its lazy annotations (\[[PEP 649]\]/\[[PEP 749]\]), the
resolution order is `tx.get_type_hints(include_extras=True)` first, then
`annotationlib.get_annotations(fn, format=Format.FORWARDREF)`, and finally
the raw `__annotations__` with deferral.

#### Caching and thread safety

Because the specificity order is defined per
call shape (§2.2), the cache has two levels.

The first is a **shape plan**, one per distinct `σ(C)`. For each method, it
precomputes that method's binding outcome for the shape, meaning which slot
each argument lands in, or that the method cannot bind at all, together
with the pairwise `⊑_σ` comparison over every pair of methods for that
shape. Alongside these, it records, for each argument position, whether it
is *value-dependent*, *declaration-dependent*, or *member-dependent*.

- A position is *value-dependent* when any method's hint there is a
  `Literal` or a `type[...]`, a `Hint[...]`, or a `Union` or `TypeVar`
  whose members or upper bound include one, or a concrete `TypedDict`. Its
  value-level check reads the argument value itself rather than its type
  alone; for a `Hint[...]`, that value is the hint passed in, so `int` and
  `str`, both of type `type`, key the cache separately.
- A position is *declaration-dependent* when any method's hint there is a
  parametrised class generic, user-defined or standard-library, directly
  or through a `Union`, `TypeVar`, or `Annotated` (but never through
  `Tuple`, `Callable`, `Type[C]`, or a `TypedDict`). Its value check reads
  the `__orig_class__` recorded on an instance of a `Generic` subclass, or
  of a class written against a [PEP 585] alias (§2.3).
- A position is *member-dependent* when any method's hint there is a
  runtime protocol with data members, directly or through a `Union`,
  `TypeVar`, or `Annotated`. Its value check reads those members off the
  instance, recorded as the sorted union of every such protocol's data
  members landing at that position.

The shape-plan cache itself is a bounded LRU, and every plan in it is
discarded whenever a method is registered.

The second level is a **call cache**, under each shape plan, keyed by
`tuple(type(v_i) for positional v_i) + tuple((k, type(w_k)) for k in sorted
keyword names)`, with two adjustments. A value-dependent argument keys by
`(type, value)` rather than `type` alone, and a declaration-dependent one
keys by `(type, __orig_class__)`, comparing the recorded parametrisation by
identity rather than equality. `typing` itself caches `Box[int]`, so every
`Box[int]()` shares one cache entry, and comparing by identity avoids
merging entries that `==` would wrongly treat as equal (`Literal[1] ==
Literal[True]` on Python 3.8, for instance). A [PEP 585] record such as
`GL[int]` is a fresh object at every subscription, so it is instead keyed
by its origin, its own arguments, each keyed the same recursive way, and
whether it is unpacked; every `GL[int]()` still lands in the same cache
entry this way.

A position that is both value-dependent and declaration-dependent is safe
to key this way even though the value's own `==` need not agree with the
record — a dataclass generic, for instance, compares only its fields. The
declared-parametrisation record itself is only ever read off an instance of
a `Generic` subclass or of a class whose MRO carries a [PEP 585] base. It
is gated by one memoised per-class check that both the value check and the
cache key consult, so the two stay in lockstep and no other value is ever
probed for a record it cannot have. Because `typing`'s own subscription
cache is a bounded LRU, a fresh `Box[int]` object created after eviction
simply misses the call cache once; it can never produce a wrong method. A
value the per-class gate refuses, such as a plain `list` at a `List[int]`
argument, is never probed and keys simply as `(type, None)`.

These extra key computations have a measured cost. A cached call at a
gated-but-record-less position (a plain list at a `List[int]` argument)
costs roughly 0.1 to 0.25 microseconds more than an ordinary type-keyed
call. This is the cost of asking the gate and of building, hashing, and
comparing the richer key. A value that does carry a record costs more again: roughly
0.5 to 0.6 microseconds for a `Box[int]()`, and roughly 0.7 microseconds for
a `GL[int]()`, whose record is keyed by its own parts. At a
member-dependent position, the key instead carries a tuple of booleans, one
per recorded data member, saying whether the value has each one. This is
read by the same function the value check itself uses, so the key always
covers exactly what the check reads, and never the value's own identity:
every instance of a class that declares the same members shares one entry.
A member the class declares as an instance variable, whether by annotation
or by attribute, property, or slot, reads `True` for every instance of that
class. This comes from a per-class memo that also records how to look up
the class's own instances, so the read costs one set lookup rather than a
walk of the MRO. A `ClassVar` member, being read off the class itself, is
not part of the key at all. Because this read happens on every call, a
cached call at a
member-dependent position costs roughly one microsecond more than an
ordinary type-keyed one. A position with several of these dependences at
once carries every relevant part of the key together: the value, the
record, and the member booleans, since none of them can stand in for
another. A value's own `==` reflects neither its declared record nor which
of its attributes happen to be set. An unhashable value at a
value-dependent argument makes the call uncacheable. Positional and keyword
spellings of what looks like "the same" call are different shapes, and
therefore different cache keys, because they genuinely can bind differently. This is not an incidental
cost, but a direct consequence of binding being name-aware.

The cache is invalidated on every `register` call, when the methods tuple
is rebuilt and published in a single assignment following the same pattern
`bagof.magic._polymorph._Registry` uses, and whenever
`abc.get_cache_token()` changes, the same signal [`functools.singledispatch`]
watches. Registration takes a `threading.Lock`; reads are lock-free, which
matters on the free-threaded Python 3.13 build.

*(0.2.0)* Beneath the two dispatch-level caches sits a third, in
`core._relation`: `issubhint` memoises its own result by `(hint,
superhint)`. Selection asks the same sub-hint question repeatedly, and the
same pair recurs across a shape plan's pairwise comparisons, so the memo
saves the recursive relation from recomputing it. Because internal
recursion goes back through the public `issubhint`, a nested comparison is
cached at every level, not only at the top. The cache is a bounded,
insertion-ordered working set (`RELATION_CACHE_SIZE`, read at store time),
evicting oldest-first exactly as the call cache does; reads are lock-free,
and a lock guards only eviction, insertion, and the clear. It is emptied
whenever `abc.get_cache_token()` changes, so a late `ABC.register` never
leaves a stale answer; a result computed while such a change lands is
dropped rather than stored, checked once more under the lock. Only ABC
registration is watched, though, exactly as the standard library's own ABC
caches watch it: mutating a class in a way the token does not track, such
as adding a data member to a `Protocol` or editing a `TypedDict` after it
has been compared, is not invalidated, and callers who do that must clear
the cache themselves. An unhashable pair is simply answered without
caching. On an interpreter whose `typing.Literal` equality merges distinct
literals (Python 3.8 and 3.9.0), the cache is keyed by object identity so
that two hints that compare equal but must dispatch differently are never
collapsed onto one entry. `ishintstance` is not cached, since it reads a
value whose identity is not a stable key. The relation cache only ever
changes how fast an answer is reached, never the answer itself.

Every rendered error names the full signature involved, such as
`area(shape: !Circle, scale: float = 1.0) @ shapes.py:12`, with `/` and `*`
markers where relevant, and `*args: H` or `**kwargs: H` spelled out. The
`!` marks the argument that actually failed. A binding failure, as opposed
to a type mismatch, is rendered in words after the signature instead, such
as `-- no parameter 'scake'`, `-- missing 'y'`, or `-- 'x' is
positional-only`.

---

## 7. Module layout

The top-level package is dispatch-only. `__init__.py` re-exports exactly
`dispatch`, `Dispatcher`, `Function`, `Method`, `Signature`, `Parameter`,
`Exact`, `Hint`, `Super`, `SuperType`, `SuperHint`, `Between`, and the
three error types. `_lattice.py` holds `equivalent()`, `overlaps()` (§5),
TypeVar solving, and the value-dependence classifier described in §6. This
is dispatch-internal code that builds on `core._relation`, while the actual
value check (`ishintstance`) is always called directly from `core`, never
wrapped. `_signature.py` holds `Signature`, `Parameter`, `Binding`, and the
precomputed per-method binder; `_method.py` holds `Method`, including its
deferred hint resolution; `_function.py` holds `Function` itself: the
methods tuple, the per-shape order, the two-level cache and its `abc`
token, `dispatch`/`resolve`/`__call__`/`__get__`, `ambiguities()`,
`register`, and `from_mapping`. `_dispatcher.py` holds `Dispatcher`, the
module-level `dispatch`, and the `functions` namespace. `_errors.py` holds
the three error types and their message builders. `_constants.py` holds the
package's `__dispatch_*__` attribute names.

`bagof.dispatchers.core` is a separate, dependency-light subpackage whose
own `__init__.py` is a pure facade re-exporting the relation and
introspection helpers the rest of the family reuses. `core/_compat.py`
holds version-pinned names (`NoneType`, `UnionType`, `UNION_TYPES`), the
special-form detection described in §11, `TypedDict` markers, and the
`spellings(name)` helper that reconciles `typing` against
`typing_extensions`. `core/_sentinels.py` holds `Unset` and `UNSET`.
`core/_exact.py` holds `Exact`, its `EXACT` sentinel, and the helpers that
read it; these are also re-exported at the top level. *(0.3.0)*
`core/_super.py` holds `Super`, `SuperType`, `SuperHint`, their `SUPER`
sentinel, and the messages that refuse the unbounded forms; the three
public names are re-exported at the top level. *(0.3.0)*
`core/_bounds.py` holds `Between` and its `LOWER` marker, the interval
reading of a bound on a value or of a `Type` or `Hint` argument (§4.2),
the readers of a bound written as a type argument and of its slot's
variance (§4.3), and the position, endpoint, and variance refusals shared
by every bound written where it cannot stand, including the empty-interval
message; `Between` is re-exported at the top level. `core/_introspect.py` holds
the general-purpose introspection helpers (`safe_get_origin`/
`safe_get_args`, `get_origin_uw`/`get_args_uw`, `unwrap`, `normalise_hint`,
alias and `NewType` resolution, `issubclassable`, `issubscriptable`, the
`TypedDict` helpers, `safe_issubclass`, `safe_isinstance`,
`get_concrete_type`, `type2hint`, `eq_safenan`, and `mro_index`, which both
value dispatch and `resolve_hint` share). `core/_relation.py` holds
`issubhint` and `ishintstance` and every branch they dispatch to:
`Exact`-aware and `Callable`-variance-aware, with an opaque fall-through for
any unrecognised form (§11), and the private `_typevar_upper` helper.
`core/_registry.py` holds `resolve_hint` itself (§8.1).

Test modules mirror this layout one-to-one (`test_introspect.py`,
`test_relation_callable.py`, `test_relation_modern.py`, `test_exact.py`,
`test_lattice.py`, `test_signature.py`, `test_dispatch_values.py`,
`test_dispatch_hints.py`, `test_registry.py`, `test_typevars.py`,
`test_cache.py`, and the usual `test_docstrings.py`, `test_import.py`,
`test_module_surface.py`).

The package's only runtime dependency is `typing_extensions>=4.13`; an
optional test extra adds `numpy` solely to exercise `eq_safenan` against
numpy's own scalar types. Every private module is `_`-prefixed, following
the family's usual convention; `core/` is the one public subpackage, and
its own `__init__` is a facade with no logic of its own. The split into two
namespaces is deliberate for griffe's sake as well as the reader's: the two
namespaces export disjoint object sets, with dispatch objects at the top
level and relation and introspection helpers under `.core`. As a result,
nothing is ever documented under two different dotted paths, and `Exact`,
the one object both sides need, is documented exactly once, under the
top-level API. Every
name under `.core` traces back to original bagof code; none of it derives
from CPython's `dataclasses`, so no PSF licence notice travels with it (the
`dataclasses`-derived code in `bagof-magic`'s own builder is unrelated, and
stays there).

---

## 8. `resolve_hint` and the relationship to `_polymorph`

### 8.1 `resolve_hint`'s parity with the old summed-distance lookup

`resolve_hint` (§6) replaced an older lookup, `get_from_registry`, that
summed a numeric distance the same way the rejected dispatch model of §1
did. Moving to the relation-based lookup settled two behavioural corners
where the two disagreed, and confirmed a third already agreed:

1. The MRO tie-break for equally specific class keys (§2.2, step 3)
   carries over unchanged. When several keys in the mapping are equally
   specific and the query itself is a class, the tie is broken by the
   query's own MRO, using the same `mro_index` helper the value-dispatch
   path uses. `{Enum, str}` resolves `class Color(str, Enum)` to `str`, and
   the diamond `class D(B, C)` resolves `{C, B}` to `B`, in both cases
   independent of how the mapping happened to be built. A tie the MRO
   cannot break, such as a non-class query or class keys that sit equally
   far from the query in its MRO, still falls through to the caller's
   `ambiguity` setting.
2. A bare `TypedDict` marker is a sub-hint of `dict`. Every `TypedDict`
   value is, after all, a `dict`, while a plain `dict` is not a
   `TypedDict`. The bare marker is therefore the unique most-specific key
   over `dict` for a `TypedDict`-subclass query, and this resolves the same
   way regardless of the mapping's insertion order. (This is a purely
   nominal, hint-level fact, distinct from any question about what a
   `TypedDict` accepts at the value level.)
3. A bare `Literal` stays genuinely ambiguous against a concrete class. A
   bare `Literal`, meaning "any literal value," is incomparable with a
   class, since not every literal value is, say, an `int`, and MRO
   refinement gives no help here either; `{Literal, int}` queried against
   `Literal[1]` is, correctly, left to the caller's `ambiguity` setting.

Every consumer that looked entries up by hint, meaning the converters,
validators, and factories packages, calls `resolve_hint` directly today, with
`ambiguity="warn"` where a registry might still hold an unresolved tie and
`"raise"` once it is known to be clean; their registration dictionaries are
unchanged.

Beyond `resolve_hint` itself, building the relation on `issubhint` and
`ishintstance` changed a handful of other behaviours relative to the
package's summed-distance predecessor, each already described in its own
section above:

- a non-`runtime_checkable` `Protocol` answers `False` rather than raising
  (§2.1);
- a constrained TypeVar behaves as the union of its constraints (§2.1, §3);
- `Callable`'s parameters are contravariant and its return type covariant
  (§2.1, §2.3);
- `Exact` (§4) is understood by the relation directly.

`eq_safenan`, which the cache and the relation both rely on for comparing
`NaN`-bearing values, no longer imports `numpy` at all, since
`numbers.Real` already covers a numpy scalar through its own ABC
registration.

### 8.2 Name-aware dispatch and `bagof.magic._polymorph`

`bagof.magic._polymorph` already dispatches on **field name**, for a
`Magic` class's polymorphic construction. It computes once, per class,
where each constrained field arrives — a position for a positional field,
its public name for a keyword-able one. It then binds one call to those
names and checks each field's value against a `Literal`-shaped
specification. Ties are broken by `(priority, number of specs, precision,
MRO depth)`, a lexicographic stand-in for "priority, then specificity over
names, then MRO depth." The way a call is bound to those names and each
candidate judged applicable is this package's own model, applied to the
parameter list of one generated `__init__`. Selection is where the two
differ. `bagof.magic._polymorph` breaks ties with the fixed lexicographic
tuple above, whereas this package ranks the applicable methods by the
partial order of §3, taking the single most specific one and reporting
ambiguity when two are incomparable.

A field's `on={...}` clause corresponds to a `Method` whose `Signature` is
the owning class's `__init__` signature, with the constrained names' hints
replaced by the field's own value specifications and every other name read
as `Any`. Reading a default-filled dispatched parameter as if its default
had been written out is the behaviour a `Function` opts into with
`dispatch_defaults=True`, so the binding matches, and
`AmbiguousPolymorphError` and `NoPolymorphError` correspond to the dispatch
errors of §5. No adapter from positions to names is needed to make the
correspondence work, because both models are name-aware from the start.

---

## 9. Corner cases

Each case below is covered by a dedicated test.

#### Hints and the lattice

- `object` is strictly more specific than `Any`, and an unannotated
  parameter is read as `Any`.
- `List[int]` is strictly below `list`, which is equivalent to `List`
  itself.
- Two overloads on `List[int]` and `List[str]` at the same position are
  ambiguous at the call (resolved by `priority`), but registering both
  raises no warning, since both are genuinely legitimate registrations
  (§2.3).
- `Optional` and `Union` are ordered by `issubhint`; a `None` argument is
  read as `NoneType`.
- A bare `Union`, `Literal`, or `Type`, used with no arguments, is never
  applicable to a value, and warns.
- `Literal` is value-dependent; `True` is not `in Literal[1]`; `NaN`
  comparisons go through `eq_safenan`; an unhashable value at a
  value-dependent argument is left uncached.
- The `Tuple[X, ...]` / `Tuple[X, Y]` / `tuple` chain orders as expected;
  items are never inspected; `Tuple[()]` is valid.
- `Callable` parametrisations order by contravariant parameters and
  covariant return; the full chain `Callable\[[int], R] <
  Callable[Concatenate[int, P], R] < Callable[P, R] ≡ Callable[..., R]`
  holds (§11.1); a repeated `ParamSpec` is solved by greatest element at
  the hint level, and shallowly at the value level.
- `type[X]` is value-dependent; `type[bool] < type[int] < type`.
  *(0.2.0)* `type[X]` reads its argument through the full relation, so
  `type[Any]` accepts every class, `type[Union[…]\]` accepts a class in the
  union, and `type[Exact[C]\]` is an identity leaf below `type[C]`.
- `Annotated` metadata other than `Exact` is invisible
  (`Annotated[X, ...] ≡ X`); applying `Exact` to a non-class raises
  `TypeError`. *(0.2.0)* A bare, unsubscripted `Annotated` super-hint is
  structural instead of opaque: `Annotated[X, …] ≤ Annotated` holds and
  `int ≤ Annotated` does not.
- *(0.2.0)* `Hint[X]` dispatches on a hint passed as a value: a `Hint`
  form is ordered only against another `Hint` form, covariantly by its
  argument, so no ordinary hint sits below it, and above it sit only the
  tops (`Any`, a free `TypeVar`, a union with a `Hint` member) and a wider
  `Hint` form up to `Hint ≡ Hint[Any]` -- never an ordinary class such as
  `object`. `Hint[Exact[X]\]` matches the exact hint `X` structurally, not
  merely something equivalent to it, so a free `TypeVar` does not match
  `Hint[Exact[Any]\]`. `Hint[X]` is value-dependent, and the outer form
  `Exact[Hint[X]\]` normalises to `Hint[Exact[X]\]`.
- *(0.3.0)* `Super[C]` is a lower bound (§4.2). On a value parameter it
  accepts a value whose class is `C` or above it, and inside `Type` or
  `Hint` the class or hint `C` and everything above it. On a value it may
  also be a member of the parameter's union or the bound of a `TypeVar`
  used there, and each bound must be a hint a class can be compared
  against; inside `Type` or `Hint`, and as the type argument of a
  generic (§4.3), it must be the whole argument, and any hint can be its
  bound. `Super[Never]` on a value is equivalent to
  `object`, `Super[object]` accepts only an instance of `object` itself,
  and `Super[Any]` accepts no value. In the same way `Type[Super[Any]\]`
  accepts no class, `Type[Super[object]\]` accepts only `object`, and
  `Type[Super[Never]\]` accepts every class; `Hint[Super[Any]\]` accepts
  only the tops. `Super[Super[C]\]` collapses to `Super[C]`, and
  `Super[Type[C]\]` normalises to `Type[Super[C]\]`. `Super[Exact[C]\]`
  and `Exact[Super[C]\]` are refused when they are written. An
  unsubscripted `Super`, `SuperType`, or `SuperHint`, and a bound in a
  position where it cannot stand, are refused at registration with the
  parameter named, and by the relation with the same message.
- *(0.3.0)* `Between[L, U]` stands in the same positions as `Super[C]`
  (§4.2). `Between[C, C]` is legal and holds the hints equivalent to `C`,
  which is wider than `Exact[C]`, so `Exact[C] ≤ Between[C, C]` and
  `Type[Exact[C]\] ≤ Type[Between[C, C]\]` but not the reverse; on a
  value, `Between[C, C]` accepts a value whose class is equivalent to `C`,
  not only one whose class is `C` itself. The degenerate
  `Between[Never, Never]`, `Between[Never, Any]`, and `Between[Any, Any]`
  are legal, while `Between[Any, U]` for any other `U` is refused as
  empty, with a message pointing to `Between[Never, U]`. `Exact`, `Super`,
  and `Between` cannot be nested inside one another in any order, even
  through a union or a `TypeVar`. There is no outer form:
  `Between[Type[A], Type[B]\]` is not rewritten, and on a value it is
  refused with a message naming `Type[Between[A, B]\]`.
- *(0.3.0)* A bound may be the whole type argument of an invariant slot,
  where it names a range of arguments (§4.3): `List[Super[int]\]` accepts a
  list declared to hold `int` or a type above it. At a covariant slot,
  `Sequence[Between[Never, C]\]` is the same as `Sequence[C]` and
  `Sequence[Super[C]\]` is refused; at a contravariant slot the reverse
  holds. A bound in a union or as a `TypeVar` bound inside a slot, a bound
  in a generic whose variance cannot be read, and a bound in a `Tuple` or
  `Callable` are refused, while one nested inside a generic there is
  legal. A bound does not reach through the slot around it, so
  `List[List[Super[C]\]\]` is a single point at its outer slot. A value
  declaring `Box[Super[int]\]` is read as that range, and
  `ishintstance([], Sequence[Super[int]\])` raises, since the hint is
  refused.
- *(0.3.0)* `Any` or an unconstrained TypeVar at an invariant slot reaches
  a base's slot as the range it stands for, and a constrained TypeVar as
  each of its constraints, so `W[Any] ≤ Snk[int]` is False for
  `class W(Snk[T])` with `Snk` contravariant (§2.3).
- *(0.3.0)* `Type[C]` and `Type[Super[D]\]` with `D ≤ C` are incomparable
  yet share the classes between `D` and `C`, so registering both warns, and
  a call with one of those classes is ambiguous unless `priority` or an
  `Exact` method settles it (§5). The same holds on a value for `C` and
  `Super[D]`, and for the values whose class lies between `D` and `C`.
- *(0.2.0)* `issubhint` and `ishintstance` reject a non-hint the way
  `issubclass` and `isinstance` do: a non-hint on either side of
  `issubhint`, or as the hint argument of `ishintstance`, raises
  `TypeError` rather than being read as `Any`; `issubhint` reports the
  left argument first when both are at fault. `ishint` reports whether an
  object is a hint at all.
- A `TypedDict` orders correctly at the hint level; its value-level check
  inspects the mapping's own shape.
- A runtime `Protocol` is structural; two protocols a value structurally
  satisfies are ambiguous unless comparable; a `Protocol` that is not
  `runtime_checkable` registers but answers `False` at both levels, so it
  is reachable yet never actually matches, and the call raises
  `NoMethodError` rather than crashing.
- A runtime protocol with data members is read member by member, with the
  value-level check keyed by which members the value actually has (§2.3).
- A class that annotates a member sits below the protocol and wins over it,
  whether the call is positional or keyword, and an instance of it that
  never set the member is still in the protocol at the value level too.
- A class that declares nothing, whose instance later acquires the member
  anyway, stays genuinely ambiguous against the protocol.
- An inherited annotation still counts, and a subclass that redeclares an
  inherited member as `ClassVar` still sits below the protocol.
- The two member kinds, instance variable and class variable, are mutually
  exclusive, exactly as the type checkers read them.
- A text annotation's marker is looked up by name rather than evaluated, so
  an import alias, a module-qualified name, or a user class named
  `InitVar` are all read correctly.
- A dataclass field with `init=False` counts as a member; an `InitVar` and
  a `TypedDict`'s own keys do not.
- A read-only member, such as a property with no setter or a `Final`
  attribute, counts toward a protocol's requirement — a deliberate
  divergence from what a type checker would accept.
- ABC registration is honoured through `issubclass`; a late `register()`
  call invalidates the cache through the `abc` cache token.
- A diamond `class D(B, C)` resolves to `B`; a class compared against a
  satisfied ABC that is not in its MRO is ambiguous rather than resolved.
- The preorder's laws (reflexivity, transitivity, and antisymmetry up to
  equivalence) are checked by property-based tests over a broad corpus of
  hints.

#### Modern typing forms (§11)

- `type X = …` resolves to its value; two aliases of one underlying value
  replace each other with a warning; recursive alias resolution stops at
  its own origin.
- `L[int]`, for `type L[T] = list[T]`, resolves to `List[int]`.
- `NewType` resolves to its declared supertype.
- The `Tuple[int, *Ts]` family orders as described in §3: a `*Ts` run
  captures whatever is left over, a longer fixed prefix or suffix is
  stricter, `Tuple[int, *Ts]` against `Tuple[*Ts, int]` and against
  `Tuple[int, ...]` are both incomparable, and `Tuple[*Ts] ≡
  Tuple[Any, ...]`; a repeated `*Ts` at several top-level `Tuple` slots is
  solved jointly by greatest element at the hint level (value-level
  solving remains out of scope); two open runs in one parameter list raise
  `TypeError` at registration; `Tuple[int, *Tuple[str, int]\]` flattens.
- The `Callable[P, R]` family orders as described in §2.1 and §11.1, with
  `...`/bare `P` at the top and a `Concatenate` prefix strictly between; a
  repeated `P` is solved jointly by greatest element at the hint level;
  `Callable\[[int, *Ts], R]` rides the open-tail machinery.
- `*args: P.args` and `*args: *Ts` are both read as an `Any` tail; a bare
  `Ts`/`P`, or a top-level `Unpack[Ts]`, used as a parameter annotation
  raises `TypeError` at registration; `Unpack[TD]` on `**kwargs` is
  ignored.
- A `Never` parameter makes its method never applicable.
- `Required`, `NotRequired`, `ReadOnly`, `Final`, and `ClassVar` are all
  transparent, read as their inner type.
- An unrecognised or future special form is opaque, behaves like `Any`, and
  emits one `UnknownHintWarning`; it never crashes.
- `TypeVar(bound=float, default=int)` reads as bound to `float`; an `int`
  argument does not match it, since there is no numeric-tower promotion at
  runtime.
- A native [PEP 695] `TypeVar` with no `__default__` is read through
  `getattr(..., NoDefault)` rather than `.has_default()`.
- `list[int]`, on Python 3.9 and 3.10, is not mistaken for an ordinary
  class.
- Both spellings of `Unpack`, native and from `typing_extensions`, are
  recognised on Python 3.11.

#### Calls and parameters

- Default values give a method an arity range, and among otherwise-tied
  methods the one with the shorter fixed arity wins on tightness.
- A method with `*args: H` loses on tightness to one with a matching fixed
  arity.
- A call with no arguments matches only zero-arity methods.
- A method defined inside a class is bound through `__get__`, and `self` or
  `cls` becomes argument zero, read as `Any` unless explicitly annotated.
- An unhashable argument leaves the call uncached.
- A lying `__class__` is a documented limitation: `type(v)` is what the
  cache key and MRO refinement read, while `isinstance` semantics (which a
  lying `__class__` can fool) govern applicability.
- No error message ever calls `repr` on an argument's value.

#### Name-aware binding

- The same argument bound positionally in one call and by keyword in
  another still selects the same method.
- A keyword that only some methods accept simply makes the others
  unbindable, not ambiguous with them; a keyword no method accepts raises
  `NoMethodError`, naming it with a "did you mean" suggestion.
- A positional-only parameter (`/`) can only ever be filled positionally; a
  keyword-only one (`*`) is dispatched by name and never filled
  positionally.
- A default-filled parameter is excluded from both applicability and
  specificity, unless the function was built with `dispatch_defaults=True`.
- Extra keywords bind to `**kwargs: H`, checked against `H` when annotated
  and against `Any` otherwise; `Unpack[TD]` there is treated as
  unannotated.
- Two methods with the same parameter names in a different order are not
  duplicate registrations, since a positional call still tells them apart,
  though a keyword call between them is ambiguous without an explicit
  `priority`.
- A TypeVar shared between `*args` and a named parameter requires
  consistency over every bound argument together (§3).
- Passing the same value under two spellings at once, as in `f(1, x=1)`,
  fails to bind at all, matching ordinary Python call semantics.
- `f(1, 2)` and `f(1, y=2)` are different shapes, and therefore different
  cache keys.
- The reduce-to-positional guarantee (§2.2) is checked directly, as a
  property test against a dedicated positional reference implementation.

#### Registration and lifecycle

- A new registration clears the cache, extends the methods tuple, and
  publishes the result atomically.
- Concurrent registration and calls are never torn, thanks to the
  registration lock and the atomic publish.
- The same name registered in two different modules against the
  module-level `dispatch` produces two distinct `Function`s; re-registering
  a method after a module reload replaces it and emits a `RuntimeWarning`.
- A malformed registration raises `TypeError`.
- Two methods tied on equal, explicit priority are still ambiguous:
  priority only ever breaks a tie when it differs, and never overrides a
  strict specificity win.
- A pair separated by differing priority is resolved deterministically and
  is therefore neither warned about nor listed by `ambiguities()`.
- `ambiguities()` lists every pair the registry has actually warned about.
- A same-origin parametrised pair, whether invariant like `List[int]` and
  `List[bool]`, or otherwise incomparable like `List[int]` and
  `List[str]`, is incomparable and hence ambiguous at a call that
  supplies no declared parametrisation, but registering both is
  legitimate, so neither warns nor is listed (§2.3).
- The re-exported names under `.core` retain their identity, and
  `resolve_hint`'s behaviour matches its predecessor's wherever the two
  were meant to agree (§8.1).

---

## 10. Scope

Some capabilities are explicitly out of scope for the design, either
permanently or pending further work.

A handful of capabilities are not planned at all, because they conflict with
the symmetric design (§1):

- `invoke`/next-method fall-through to a less specific overload;
- dispatch on a function's return type;
- dispatch on a keyword-only parameter that is not itself part of the call's
  binding;
- static overload synthesis, the kind a type checker performs at analysis
  time;
- asymmetric, left-to-right argument precedence in the style of CLOS.

Other work is deferred, and is tracked separately:

- Value-level solving of `ParamSpec` and `TypeVarTuple`: a callable or
  tuple *value* is matched shallowly, without inspecting its actual
  signature or element shape, so `print` matches every `Callable[...]`
  slot and `(1, "a")` matches every `Tuple[...]` slot; only the hint level
  solves either of them (§3).
- The repeated-TypeVar grouping tie-break (§3) is not extended to a
  repeated `ParamSpec` or `TypeVarTuple`: two incomparable open prefixes,
  or two incomparable runs, stay ambiguous, and `*args: *Ts` never beats a
  plain `*args`.
- The following are all out of scope for `ParamSpec`: `P.args` and
  `P.kwargs` as first-class components, which for now remain an `Any`
  tail; `ParamSpec(bound=...)`; [PEP 696] defaults on a `ParamSpec`; and the
  consistency of one `ParamSpec` across *nested* positions within a single
  hint, since only a top-level, landed `Callable` joins a group.
- The following are all out of scope for `TypeVarTuple`: more than one
  unpack in a single parameter list; `Unpack[Ts]` written on `**kwargs`;
  [PEP 696] defaults on a `TypeVarTuple`; `TypeVarTuple` bounds, which do
  not exist in the [typing
  specification](https://typing.python.org/en/latest/spec/generics.html);
  and a `*Ts` in the middle of a `Callable` parameter list followed by a
  fixed suffix, which degrades to an open shape instead.

---

## 11. Modern typing constructs and version support

The design commits, as a first-class goal, to supporting Python 3.8 through
the newest and future versions. Every construct is reached exclusively
through `import typing_extensions as tx`, and the relation treats every
spelling of a construct uniformly and tolerates whatever it does not yet
recognise. Two facts about the summed-distance baseline this improves on
shaped the approach.

First, nothing in that baseline actually crashed, but a great deal of it
was silently dead. For every modern construct tested, its
`issubhint`/`ishintstance` answered `False` in both directions, so a method
annotated with a `type` alias, a `NewType`, `Never`, `TypeIs`, or
`Required[...]`, for instance, registered successfully and simply never
fired. For a dispatcher, that is worse than an outright crash, since
nothing signals the mistake. §11.2 turns "silently dead" into "understood,
or explicitly and visibly degraded."

Second, `typing.X` and the corresponding `typing_extensions.X` are not the
same object, and which one a given construct actually produces drifts by
Python version, measured to differ across 3.10 through 3.13.
`tx.get_origin(tx.Unpack[Ts])`, for example, returns
`typing_extensions.Unpack` on 3.11 but `typing.Unpack` on 3.13, and a
native `type X = int` alias is not an instance of `tx.TypeAliasType`. Every
identity check the relation performs on a special form therefore has to
accept both spellings, generalising the `_TYPEDDICT_MARKERS` approach into
the `_compat.spellings(name)` helper.

### 11.1 Per-construct handling

| Construct | Native since | `typing_extensions` backport | Handling and scope |
|---|---|---|---|
| [PEP 695] `def f[T]` / `class C[T]` | 3.12 | — (syntax only) | `__type_params__` holds ordinary `TypeVar`s; dispatch treats them identically to legacy ones. A default is read through `getattr(tv, "__default__", tx.NoDefault)`, since a native 3.12 TypeVar has no `has_default()`. **Supported.** |
| [PEP 695] `type X = …` (`TypeAliasType`) | 3.12 | 4.6+ | `resolve_alias` detects one by duck type (`__value__` plus `__type_params__`) or either spelling, returns `__value__`, and substitutes arguments for a subscripted `G[int]` through typing's own `__getitem__`; resolution is recursive, with a cycle guard, and runs at the top of `issubhint`/`ishintstance` and from `normalise_hint` (not from `unwrap`). **Supported.** |
| [PEP 613] `TypeAlias` | 3.10 | 4.x | The bound value is an ordinary hint; the bare marker itself falls under the unknown-form rule. **Supported (trivially).** |
| [PEP 696] defaults | 3.13 | 4.4+ | Only the bound or constraints are read, through `_typevar_upper`; the default is ignored. **Supported.** |
| [PEP 646] `TypeVarTuple`/`Unpack`/`*Ts` | 3.11 | 4.1+ | `Unpack[Ts]` inside `Tuple[...]` is an open run of zero or more `Any`, split into prefix and suffix by the tuple-shape classifier; a fixed prefix or suffix captures the rest, a longer one is stricter, and `Tuple[*Ts] ≡ Tuple[Any, ...]`; a repeated `*Ts` at top-level `Tuple` slots (and in `Callable\[[int, *Ts], R]`, through the open-tail path) is solved jointly by greatest element at the hint level; `*args: *Ts` reads as an `Any` tail sharing that same run; a bare `Ts` or a top-level `Unpack[Ts]` used as a parameter, and two open runs in one list, both raise `TypeError` at registration. Value-level solving is deferred (§10); a `*Ts` in the middle of a list followed by a fixed suffix degrades to an open shape; `Unpack[Ts]` on `**kwargs`, and [PEP 696] defaults on a `TypeVarTuple`, are both out of scope. **Supported at the hint level.** |
| [PEP 612] `ParamSpec`/`Concatenate` | 3.10 | 4.x | `Callable\[[int], R] < Callable[Concatenate[int, P], R] < Callable[P, R] ≡ Callable[..., R]`, with `...`/bare `P` at the top of the parameter-list order (§2.1); `Concatenate` is a contravariant prefix, and a longer prefix is stricter; a repeated `P` is solved jointly by greatest element at the hint level; `*args: P.args` reads as an `Any` tail; a bare `P` or `Concatenate` used as a parameter raises `TypeError` at registration. Value-level solving is deferred (§10). **Supported at the hint level.** |
| `NewType` | 3.5 native / 3.10 class-based | typing_extensions class on 3.8/3.9 | `resolve_newtype` follows `__supertype__` recursively, inside `normalise_hint`. **Supported.** |
| `Never`/`NoReturn` | 3.11/3.6 | 4.1+ | The bottom type; a `Never` parameter makes its method never applicable — an explicit way to forbid a combination. **Supported.** |
| `TypeGuard`/`TypeIs` | 3.10/3.13 | 4.x/4.10+ | Treated as `bool`. **Supported.** |
| `LiteralString` | 3.11 | 4.1+ | Treated as `str`. **Supported.** |
| `Self` | 3.11 | 4.0+ | A method's `Self` resolves to the owner class through `__get__`; on a free function, it falls under the unknown-form rule. **Supported, or degraded, as appropriate.** |
| `Required`/`NotRequired`/`ReadOnly` | 3.11/3.11/3.13 | 4.0+/4.9+ | Transparent qualifiers, unwrapped to their inner type. **Supported.** |
| `Final`/`ClassVar` | 3.8 | — | Transparent qualifiers, unwrapped to their inner type. **Supported.** |
| `Annotated`/`Doc` ([PEP 727]) | 3.9 / typing_extensions | 4.x/4.9+ | Transparent except for `EXACT` (§4); `Annotated` is a class through 3.12 but not on 3.13, which is pinned for. **Supported.** |
| [PEP 604] `X \| Y` | 3.10 | — | Recognised through `types.UnionType`, tracked in `UNION_TYPES`; on 3.14, `types.UnionType is typing.Union` is pinned for. **Supported.** |
| [PEP 585] `list[int]` | 3.9 | — | `isinstance(list[int], type)` is true on 3.9 and 3.10, so it is guarded with `get_origin(x) is None` before anything treats it as a plain class; equivalent to `List[int]`. A class written against one, such as `class GL(list[T])`, takes the TypeVars its [PEP 585] bases mention, each at its declared variance, and its instances' declared parametrisations are read the same way a `Generic` instance's are (§2.3). **Supported.** |
| User `Generic[T]` | 3.8 | — | The origin is checked by `isinstance`; arguments are compared by each position's declared variance, read live off `__parameters__` (`covariant`/`contravariant`/unflagged→invariant, `infer_variance`→invariant); a subclass is compared through the bases it was written with, and a value by the parametrisation it declares (§2.3). **Supported.** |
| Unknown or future form | — | typing_extensions checked first | The opaque rule of §11.2. **Degraded, not crashed.** |

A note on the numeric tower: `issubhint(int, TypeVar(bound=float))` is
`False`. [PEP 484]'s `float`/`complex` promotion is a convention type
checkers apply statically; dispatch instead follows runtime `isinstance`
semantics, which do not promote. Callers who want `int` and `float` to
dispatch together write `numbers.Real` or `Union[int, float]` explicitly.

### 11.2 Forward-compatibility strategy

An unrecognised hint is treated as opaque, behaving like `Any`. It is
accepted by every other hint as a superhint (`issubhint(x, unknown)` and
`ishintstance(v, unknown)` both answer `True`), while being a subhint only
of itself and of `Any`. The first time a given unrecognised form is
encountered, it emits one `UnknownHintWarning` naming the form, so that a
method built on it remains *reachable* rather than silently dead.

Inside the relation, forms are handled in a fixed order. `normalise_hint`
first resolves `None`, `NewType`, `TypeAliasType`, and any transparent
qualifier. Known non-class forms are then matched by identity, accepting
either spelling, and an origin that is itself a class takes the ordinary
structural path after that. Anything still left over falls to the opaque
rule above.
There is no bare `return False` fall-through anywhere for a form the
relation simply does not recognise.

The set of known special forms is kept as an explicit tuple for the common
case, backed by a structural fallback in `_compat`. A value counts as a
special form if it is a `type` whose `__module__` is `typing` or
`typing_extensions` and it is not `Generic`, `Protocol`, or a `TypedDict`
marker. Combined with the `try`/`except TypeError → False` guard already
inside `safe_issubclass`, no future construct, one that does not exist yet
on any currently supported Python, can raise its way out of the relation.

### 11.3 Version-by-version notes

- On Python 3.8, every modern construct is reached through
  `typing_extensions`. Native `list[int]` and `X | Y` do not exist, so a
  stringified annotation using either raises `TypeError` inside
  `get_type_hints`, caught and deferred like any other forward reference.
  `NewType` is a `typing_extensions` class rather than a function.
- On Python 3.9, [PEP 585] arrives, and `isinstance(list[int], type)` is
  true, so `issubclassable` and `safe_issubclass` guard against it with
  `get_origin(x) is None`.
- On Python 3.10, `X | Y`, `ParamSpec`, `Concatenate`, `TypeAlias`, and
  `TypeGuard` all become native; `NewType` becomes a class; both spellings
  of everything are still required.
- On Python 3.11, `Any` becomes a class; `Never`, `Self`, `LiteralString`,
  `TypeVarTuple`, `Unpack`, and `Required` all become native, and the
  [PEP 585] `GenericAlias` trap from 3.9/3.10 is fixed. `typing_extensions`
  still supplies its own `Unpack`, `TypeVarTuple`, and `TypeVar`, distinct
  from the native ones. `tx.Unpack is not typing.Unpack`, and the star
  syntax `Tuple[int, *Ts]` produces `typing.Unpack[Ts]` while
  `tx.Unpack[Ts]` produces the `typing_extensions` one; the two are not
  even equal, so `spellings("Unpack")`, comparing by identity across both,
  is mandatory rather than a convenience. `tx.TypeVarTuple is not
  typing.TypeVarTuple` either, so a `TypeVarTuple` can only be recognised
  by `isinstance`, never by class identity. Re-registering a method that
  switches between `*Ts` and `Unpack[Ts]` reads as a brand-new method,
  since structural equality compares the origin by identity. This is an
  acceptable consequence, not a bug.
- On Python 3.12, [PEP 695] syntax and a native `TypeAliasType` arrive
  (duck-typed rather than assumed); native [PEP 695] TypeVars still lack
  `__default__`; `Annotated` remains a class.
- On Python 3.13, [PEP 696] defaults, `TypeIs`, and `ReadOnly` all become
  native; `Annotated` is no longer a class; the free-threaded build exists,
  which is why registration takes an explicit lock.
- From Python 3.14 onward, lazy annotations (\[[PEP 649]\]/\[[PEP 749]\]) are read through
  `annotationlib` with `Format.FORWARDREF`; `Union` becomes a class, with
  `types.UnionType is typing.Union`. Anything newer than this is covered by
  the opaque rule, the structural special-form fallback, and `spellings()`,
  producing a warning rather than a crash.

[PEP 483]: https://peps.python.org/pep-0483/
[PEP 484]: https://peps.python.org/pep-0484/
[PEP 585]: https://peps.python.org/pep-0585/
[PEP 586]: https://peps.python.org/pep-0586/
[PEP 604]: https://peps.python.org/pep-0604/
[PEP 612]: https://peps.python.org/pep-0612/
[PEP 613]: https://peps.python.org/pep-0613/
[PEP 646]: https://peps.python.org/pep-0646/
[PEP 649]: https://peps.python.org/pep-0649/
[PEP 695]: https://peps.python.org/pep-0695/
[PEP 696]: https://peps.python.org/pep-0696/
[PEP 727]: https://peps.python.org/pep-0727/
[PEP 749]: https://peps.python.org/pep-0749/
[Julia]: https://docs.julialang.org/en/v1/manual/methods/
[Wiki]: https://en.wikipedia.org/wiki/Multiple_dispatch
[CLOS]: https://lispcookbook.github.io/cl-cookbook/clos.html
[Dylan]: https://opendylan.org/
[Cecil]: https://projectsweb.cs.washington.edu/research/projects/cecil/www/cecil.html
[`plum`]: https://beartype.github.io/plum/intro.html
[`multipledispatch`]: https://multiple-dispatch.readthedocs.io/en/latest/
[`functools.singledispatch`]: https://docs.python.org/3/library/functools.html#functools.singledispatch
