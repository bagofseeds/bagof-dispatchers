---
icon: fontawesome/solid/file-lines
---

# RFC 0001 — Hint-informed multiple dispatch for `bagof.dispatchers`

- **Status:** Design proposal (for owner review). No implementation code written yet.
- **Tracking issue:** bagofseeds/bagof-dispatchers#1
- **Scope tag of the design work:** `[Fable Scope: Plan Only]`

> **Evidence & citation note.** Every claim about the *existing* relation
> (`issubhint` / `ishintstance` / `get_from_registry`) was verified by running
> the live code in `bagof-core-magic` under CPython 3.11, and the modern-typing
> behaviour in §11 by probing `typing_extensions` 4.15 under CPython 3.10–3.13
> (3.8 / 3.9 / 3.14 were not installed; statements about them are marked
> *verify*). The environment's network policy blocks `peps.python.org`,
> `en.wikipedia.org`, `docs.julialang.org` and `typing.python.org`, so
> citations marked **[PEP 483]**, **[Wiki]**, **[Julia]**, **[spec]** are from
> working knowledge (section names given) and **must be verified against source**
> before any quoted error text is copied into docstrings.

---

## 0. Executive summary (the decisions, up front)

1. **`bagof.dispatchers` is the lowest-level package in the family.** It owns
   **both** the hint-level subtype relation (`issubhint`, `ishintstance`, and
   every introspection helper they need) **and** the signature-level order
   (tuples, arity, TypeVar consistency, `Exact`, MRO refinement, ambiguity).
   The top-level `bagof.dispatchers` API is **dispatch-only** (`dispatch`,
   `Dispatcher`, `Function`, `Method`, `Signature`, `Parameter`, `Exact`, the
   errors); the relation and introspection helpers live under the
   **`bagof.dispatchers.core`** namespace, which is what `bagof-core-magic`
   re-exports. Its only dependency is `typing_extensions`; it depends on neither
   `bagof-core-magic` nor `bagof-hints`. `bagof-core-magic` **depends on
   `bagof-dispatchers`** and re-exports the relocated names unchanged, keeping
   only the magic object model. End-state dependency graph:

   ```
   typing_extensions
    └─ bagof-dispatchers      # introspection + relation + Exact + dispatch + resolve_hint
        └─ bagof-core-magic   # MagicHint/MagicError/MultipleCauses/get_default/REAL_TYPES; re-exports the rest
            └─ converters, validators, factories
                └─ magic
   bagof-hints                # independent, unchanged
   ```

   No cycle is possible: dispatchers imports nothing from `bagof.*`.

2. **Julia (symmetric) semantics for the order, Python semantics for the
   tie-break.** Most-specific by the subtype partial order on argument tuples;
   when two applicable methods are incomparable, refine by explicit `priority=`,
   then by the arguments' own C3 MRO (as `functools.singledispatch` does); if
   still tied, **raise `AmbiguousMethodError` listing the candidates in Julia's
   format**. Never sum type-distances; never fall back to left-to-right argument
   precedence (CLOS-style asymmetric dispatch **[Wiki]**).

3. **Add one annotation, `Exact[C]`**, spelled `Annotated[C, EXACT]` so type
   checkers still see `C`. TypeVars cannot express exactness (§4). The relation
   itself (`issubhint`/`ishintstance`) is made `Exact`-aware, so validators and
   converters get exactness for free.

4. **Two query modes, one engine.** `f(*values)` (value-level, via
   `ishintstance`) and `f.resolve(*hints)` (hint-level, via `issubhint`). The
   second is what `get_from_registry` is, so the siblings' single-key lookups
   become `resolve_hint(hint, mapping, …)`.

5. **Dispatch is name-aware.** A call is bound to each method's parameters the
   way Python binds it (`inspect.Signature.bind` semantics: positionals by
   position, keywords by name, keyword-only included), and selection compares the
   hints each *argument* landed in — reducing provably to positional dispatch for
   positional calls. This is the model `bagof.magic._polymorph` already uses for
   field-name dispatch, and it departs from Julia/plum, which dispatch on
   positionals only.

6. **Support Python 3.8 through the latest and future versions** (§11). Every
   construct is reached through `import typing_extensions as tx`; the relation is
   uniform across spellings and **forward-tolerant** — an unrecognised or future
   hint degrades to `Any`-like behaviour and warns, never crashes.

7. **Repeated TypeVars (`(T, T)`)** are supported for applicability and for a
   narrow specificity tie-break (implemented in Phase 7: a signature whose
   repeated TypeVars constrain more arguments to one consistent type is more
   specific); Julia's full diagonal semantics are deferred. Flagged for review.

---

## 1. Multiple-dispatch theory primer

**Terminology [Wiki].** *Single dispatch* (Python methods,
`functools.singledispatch`) chooses an implementation from the dynamic type of
one argument. *Multiple dispatch* chooses from the dynamic types of **several**
arguments. A *generic function* is the name; each implementation is a *method*;
the methods *applicable* to a call are those whose parameter types accept the
call's argument types; the one chosen is the *most specific*. Formalised by
Castagna, Ghelli & Longo (1995) as overloaded functions with late binding, the
choice governed by a *partial order* on signatures.

**Types as sets [PEP 483].** `t1` is a subtype of `t2` when every value of `t1`
is a value of `t2` and every function accepting `t2` accepts `t1`:
`bool ⊂ int ⊂ object`. With hints as elements: `Literal[1] ≤ int`,
`int ≤ Optional[int]`, `Union[int, str] ≤ object`. It is a *partial* order —
`int` and `str` are incomparable.

**Signatures are tuples, compared position-wise.** `Tuple` is covariant
**[PEP 483]**, so `(bool, str) ≤ (int, str)`. This is why *argument* dispatch is
covariant — a method signature is a tuple type and the winner is the one whose
argument-tuple type is the smallest supertype of the call's **[Julia]**.

**Applicable & most specific [Julia].** For `f(1, "a")`, call type `(int, str)`;
`(int, str)`, `(int, Any)`, `(Any, Any)`, `(numbers.Real, str)` are all
applicable; `(int, str)` wins because it is ≤ every other. Definition order
never matters.

**Why "most specific" can be undefined → ambiguity [Julia].** `g(x: float, y)`
and `g(x, y: float)` called `g(2.0, 3.0)`: both applicable, neither ≤ the other.
The applicable set has two *maximal* elements and no minimum — an **ambiguity**,
a property of the *registrations*, not of the call. Julia raises a `MethodError`
rather than picking arbitrarily, and suggests defining the intersection method.

**Symmetric vs asymmetric resolution [Wiki].** CLOS resolves by argument
precedence (left-to-right) and never reports ambiguity; Julia/Dylan/Cecil treat
arguments symmetrically and error. `multipledispatch` warns then picks by a
topological order; `plum` raises `AmbiguousLookupError`; `singledispatch` raises
`RuntimeError("Ambiguous dispatch")` only between two virtual ABC bases. **This
package is symmetric with a Python MRO refinement** (§2.2).

**Why summed numeric distance is the wrong model** (the naive `type_distance`):
- It invents a winner where the theory says there is none, favouring whichever
  side sits shallower in `__mro__` — arbitrariness without CLOS's predictability.
- It is not monotone: `int → numbers.Integral` is not in `int.__mro__` (ABC
  registration), so the naive code returns a large sentinel distance, and a
  strictly-more-specific position can lose to a shallower less-specific one.
- It cannot compare non-classes (`Union`, `Literal`, `List[int]`, TypeVars).
- Every real dispatcher (Julia `morespecific`, `plum` `Signature.__le__`,
  `multipledispatch` `supercedes`/`ambiguous`, `singledispatch` `_find_impl`)
  uses an *order*, not a metric.

---

## 2. The specificity model

### 2.1 The base relation: `issubhint`, as it behaves

`issubhint(hint, superhint)` is the primitive. The table records the current
measured behaviour; where a row is marked **(post-fix)** the Phase 1 rewrite
changes it, and the change is called out.

| Query | Result | Consequence / note |
|---|---|---|
| `bool ≤ int`, diamond `D ≤ B`, `D ≤ C` | True | nominal subtyping; a diamond yields two incomparable applicable methods |
| `Any ≤ object` / `object ≤ Any` | False / True | `object` is strictly more specific than `Any`; both may be registered, `object` wins |
| `T ≤ Any` / `Any ≤ T` (unbound) | True / True | unbound TypeVar ≡ `Any` |
| `TB ≤ int` / `int ≤ TB` (`bound=int`) | True / True | a bound TypeVar ≡ its bound → cannot mean "exactly" |
| `int ≤ TC`, `TC ≤ int`, `TC ≤ Union[int,str]`, `Union[int,str] ≤ TC` | T/F/T/**F→T (post-fix)** | constrained TypeVar becomes `≡` the union of its constraints |
| `list ≤ List`, `List ≤ list`, `List[int] ≤ list`, `list ≤ List[int]` | T/T/T/F | a **preorder** with equivalence classes; `List[int] < list ≡ List` |
| `List[bool] ≤ List[int]`, `Dict[str,int] ≤ Dict[str,object]` | **False** | `list`/`dict` are invariant (PEP 484): a subtype argument is not a sub-hint (§2.3; #50) |
| `Sequence[bool] ≤ Sequence[int]`, `Mapping[str,bool] ≤ Mapping[str,int]` | True | `Sequence` covariant; `Mapping` key-invariant, value-covariant (spec table, §2.3; #50) |
| `Snk[int] ≤ Snk[bool]` (user `contravariant=True`), `Box[bool] ≤ Box[int]` (user unflagged) | True / **False** | user generics read their declared `TypeVar`: contravariant reverses, unflagged is invariant (§2.3; #50) |
| `IntBox ≤ Box[int]`, `Sub[bool] ≤ Box[bool]`, `Flip[int,str] ≤ Pair[str,int]`, `IntBox ≤ Box[str]` | **True** / T / T / F | differing origins: the sub-hint is re-expressed through the bases its class was written with — `class IntBox(Box[int])`, `class Sub(Box[T])`, `class Flip(Pair[B, A], Generic[A, B])` — then compared slot by slot (§2.3; V5 of #50) |
| `Box[bool] ≤ Box[TB]`, `Box[TB] ≤ Box[int]`, `Box[int] ≤ Box[TC]` (`TB` bound=int, `TC(int, str)`), `Snk[bool] ≤ Snk[TB]` | **True** / **False** / True / False | a `TypeVar` in an invariant slot is solved on the super side and a family on the sub side (V3 read it as exactly its bound: F / T / F); a contravariant slot keeps the bound reading (§2.3; V5 of #50) |
| `Child ≤ List[int]`, `Child ≤ List[float]`, `Child ≤ List[object]`, `Child ≤ Sequence[object]` (`class Child(List[int])`) | **True** / F / F / T | `Child` is `List[int]` nominally; invariance governs comparing that with another `list` parametrisation, covariance lets it widen through `Sequence` (V5 of #50) |
| `Tuple[int] < Tuple[int, ...] < tuple`, `Tuple[int,...] ≤ Tuple[Any,...]` | True chain | covariant `Tuple` |
| `Tuple[int,str] ≤ Tuple[int,*Ts]`, `Tuple[int] ≤ Tuple[int,*Ts]`, `Tuple[int,*Ts] ≤ Tuple[*Ts]`, `Tuple[int,*Ts,str] ≤ Tuple[int,*Ts]` | True | `*Ts` is an open run of 0+ `Any`: a fixed prefix captures the rest, a longer fixed prefix/suffix is stricter (Phase-8(b), #28) |
| `Tuple[int,*Ts]` vs `Tuple[*Ts,int]`, `Tuple[int,*Ts]` vs `Tuple[int,...]` | incomparable | a prefix run and a suffix run, or a `*Ts` run and a `...` run, do not order either way |
| `Tuple[*Ts] ≡ Tuple[Any,...] ≡ Tuple[*Us]`, `Tuple[int,...] ≤ Tuple[*Ts]`, `Tuple[*Ts] ≤ tuple` | True | a lone `*Ts` run is the widest tuple, below only bare `tuple` |
| `Literal[True] < bool`, `Literal[1] < Literal[1,2]` | True | literals are the bottom |
| `int ≤ Optional[int]`, `None ≤ Optional[int]`, `Optional[int] ≤ Union[int,str,None]` | True | union rules |
| `type[bool] < type[int] < type` | True | `Type[C]` covariant |
| `List[int] ≤ Sequence[int]`, `List[int] ≤ Iterable` | True | ABC registration honoured |
| `Annotated[int,'x'] ≡ int` | T/T | metadata invisible → `Exact` handled before delegating |
| `Callable[[int],str]` vs `Callable[[bool],str]` | F/F → **contravariant (post-fix)** | params compared contravariantly, return covariantly |
| `Callable[[int],R] < Callable[Concatenate[int,P],R] < Callable[P,R] ≡ Callable[...,R]` | True | `...` / bare `P` top the parameter lists, a `Concatenate` prefix sits between (row-flip, §11.1; issue #32) — so `Callable[...,R] ≤ Callable[[int],R]` and `Callable[Concatenate[int,P],R] ≤ Callable[[int],R]` are **False** (were True), which restores transitivity |
| `int ≤ P` (non-`runtime_checkable` Protocol) | raises → **False (post-fix)** | guarded; an overload on one registers, but answers False at both levels, so it never matches and the call raises `NoMethodError` |
| `list ≤ RP` (runtime protocol) | True | protocols dispatch structurally |
| `Named ≤ HasName`, `Record ≤ HasName`, `Ann ≤ HasName`, `Sub ≤ HasName`, `Other ≤ HasName` (`HasName` a runtime protocol with the data member `name: str`; `Named` sets `name = …` on the class, `Record` is a dataclass with a `name` field, `Ann` only annotates it, `Sub(HasName, Protocol)`, `Other` an unrelated protocol with the same member) | **True** / **True** / False / **True** / False | Python refuses `issubclass` here; the relation asks whether every instance has the member, which only the class's own definitions — or a dataclass field — promise (§2.3; #56) |
| `dict ≤ TD`, `TD ≤ dict`, `TD ≤ Mapping` | F/T/T | TypedDict orders correctly at hint level |
| `int ≤ Union` (bare) | False | bare `Union`/`Literal`/`Type` mean "is one of these"; dead for value dispatch |

Value level (`ishintstance`): `{'a':1} in TD` → **False**; `[1] in List[str]`
→ **True** (items never inspected, and a plain list declares no arguments);
a value's **declared** parametrisation is read when it has one (V5 of #50):
`Box[int]()` records `Box[int]` and is **not** in `Box[str]`, and an instance of
`class Child(List[int])` is **not** in `List[float]` — while `Box()`, a base
with a free `T` (`class C(List[T])`), and a declared `Any` stay shallow;
`print in Callable[[int],str]` → True
(a callable's own signature is never inspected, and a `ParamSpec` is not
solved from values — the value level is unchanged by the row-flip, #33);
`True in Literal[1]` → False (PEP 586); `1 in T` True, `'x' in TB` False;
`v in HasName` → True when `v` has `name`, set on the instance or defined by
its class, and False otherwise — two instances of one class can differ (#56).

### 2.2 Definitions and selection (name-aware, normative)

`⊑` = `issubhint` (which now handles `Exact` internally). Dispatch is
**name-aware**: a call is bound to each method before selection, and specificity
compares the hints of the slots the *same argument* landed in — comparing
"name to name" directly would be wrong for `def f(a, b)` vs `def f(x, y)` called
`f(1, 2)`, which clearly compete on the same two arguments.

- **Signature** `S = (params, varargs, varkw)`: `params` is an ordered map
  `name → Parameter(hint, kind, default)` with `kind ∈ {POSITIONAL_ONLY,
  POSITIONAL_OR_KEYWORD, KEYWORD_ONLY}`; `varargs` the optional `*args` hint
  `h_*`; `varkw` the optional `**kwargs` hint `h_**`. Every parameter in `params`
  is **dispatched** (unannotated → `Any`, so it participates trivially). Return
  annotation ignored.
- **Call** `C = (v_0 … v_{n-1}; {k_j: w_j})`; its **shape** `σ(C) = (n, sorted
  keyword names)`; its **arguments** `Args(C) = {0..n-1} ∪ {k_j}`.
- **Binding** `bind(S, C)` (an `inspect.Signature.bind` restatement, precomputed
  per method for speed): positionals fill positional slots in order, surplus →
  `*args` or fail; each keyword fills the same-named keyword-able parameter if
  unfilled, else `**kwargs`, else fail; a required parameter left unfilled fails;
  unfilled parameters with defaults are **default-filled**. On success,
  `hint_S(a)` is the hint of the slot each argument `a` landed in (`Any` for an
  unannotated catch-all).
- **Applicable to values**: `bind(S, C)` succeeds, `∀a ∈ Args(C):
  ishintstance(value_a, hint_S(a))` (an extra positional is checked against `h_*`,
  an extra keyword against `h_**`), and §3 TypeVar consistency over the bound
  arguments. **Default-filled parameters are not arguments** — their hints are
  not checked and they do not enter specificity (owner's rule; the
  `dispatch_defaults` exception is in §6).
- **Applicable to hints** (`resolve(*hints, **named_hints)`): the same with
  `issub(q_a, hint_S(a))`.
- **Specificity, per call**: for two methods applicable to `C`, `A ⊑_C B` iff
  `∀a ∈ Args(C): hint_A(a) ⊑ hint_B(a)` plus §3 consistency. The order is defined
  **per shape** (which slot each argument hits depends on the shape), so it is
  computed and cached per shape (§6), not once at registration — but it remains a
  partial order for every shape, so genuine ambiguities still surface.
- **Reduce-to-positional guarantee (a theorem, to be tested).** For a call with
  no keywords and methods without `*args`, `hint_S(i)` is the i-th hint of the
  old positional tuple, so applicability, `⊑_C`, `Max` and every tie-break below
  coincide *exactly* with a positional engine. Keep a positional reference
  implementation in `tests/` and assert equality on generated positional calls.
- **Extra keywords / `**kwargs: T`**: a keyword naming no declared parameter
  binds to `**kwargs` (or makes the method inapplicable); it is checked against
  `h_**` when annotated and enters specificity through `hint_S(a)` (= `h_**` or
  `Any`). A `**kwargs: T` groups every keyword it captures into one block for
  the §3 grouping tie-break (**Implemented**, Phase 8d), so a `**kwargs: T`
  method is more specific than one with an untyped `**kwargs`; applicability
  solves `T` *jointly* over those keywords together with every other slot
  carrying `T`, exactly as `*args: T` does over the positionals it absorbs.
  The greatest-element rule applies, so `f(a=1, b=True)` matches with `T = int`
  while `f(a=1, b="x")` has no consistent `T` and does not.
  `Unpack[TD]` on `**kwargs` is treated as unannotated (§11).

**Selection.** `Max` = applicable methods with no strictly more specific one
under `⊑_C`. Then, in order, drop members strictly dominated under:
  1. **explicit priority** (higher wins; default 0);
  2. **MRO refinement** (per argument): A dominates B iff for every `a`,
     `hint_A(a) ≡ hint_B(a)`, or both hints (unwrapped, with `Exact[C]` read as
     `C` per §4 — an `Exact[C]` refines only at index 0, where it agrees with
     `C`) are classes in `type(value_a).__mro__` with
     `index(hint_A(a)) ≤ index(hint_B(a))`,
     strictly `<` for some `a` (resolves the diamond `D(B, C)` to `B`, as
     `singledispatch` does; protocols/ABCs not in the MRO, unions, literals and
     parametrised generics give no refinement);
  3. **tightness** (restates the old arity rule): fewer arguments absorbed by
     catch-alls (`*args`/`**kwargs`), then fewer default-filled parameters, then
     no `**kwargs`, then no `*args`.

  The single-key lookup `resolve_hint` (§8.1) applies the **same step-3 MRO
  refinement** to a tie between equally specific *class* keys, reading the
  class *query* in place of an argument's runtime type: `{Enum, str}` resolves
  `class Color(str, Enum)` to `str`, and `{C, B}` resolves the diamond
  `D(B, C)` to `B`. A tie MRO cannot break falls to its `ambiguity` handling.

  `|Max| = 1` → done; `|Max| > 1` → **`AmbiguousMethodError`**; nothing
  bindable-and-applicable → **`NoMethodError`**, whose message distinguishes "no
  method accepts keyword `scake`" (with a `difflib` did-you-mean over all
  methods' parameter names), "missing argument `y` for every candidate", and
  "argument types matched nothing". Priority precedes MRO because explicit beats
  implicit; the refinements are partial, so cross-argument conflicts stay
  ambiguous and none overrides a strict specificity win (tested).

**Worked cases (each a test):** same names/order → identical to positional for
all four spellings of `area(c, 2.0)`; different names (`f(a,b)` vs `f(x,y)`) →
positional call competes per position, keyword call binds only where names exist;
name reachable only via `**kwargs` → the declared-parameter method wins when
applicable; same names/different order → equivalent for keyword calls (ambiguous
without priority) but not positional, so *not* duplicates at registration;
positional-only (`p(x, /)`) → `p(x=1)` unbindable → `NoMethodError`.

### 2.3 Where variance enters (spec-defined; #50)

PEP 483: for `t2 ≤ t1`, `G` is *covariant* if `G[t2] ≤ G[t1]`, *contravariant*
if `G[t1] ≤ G[t2]`, *invariant* if neither. Variance is a property of the
generic's **parameter position**, defined by the spec — never by the argument.
The relation reads it per position and applies it slot by slot (`A` sub-side
arg, `B` super-side arg): covariant `A ⊑ B`, contravariant `B ⊑ A`, invariant
`A ≡ B` (with `Any`/a free `T` on the super side a top an invariant slot may
widen to, via gradual consistency). Nesting composes by recursion, so the signs
multiply.

- **Where each position's variance comes from.** A user generic reads its
  declared `TypeVar` live off `__parameters__`: `covariant=True` → covariant,
  `contravariant=True` → contravariant, **neither → invariant** (PEP 484). A
  PEP 695 `infer_variance` variable is unknowable at runtime and read as
  invariant. A stdlib generic is looked up in a table vendored from CPython's
  `typing` (the spec's reference implementation), keyed by runtime origin: `list`
  / `set` / `dict` / `MutableSequence` / … invariant, `Sequence` / `frozenset` /
  `Collection` / `Iterable` / `Type[C]` / … covariant, `Mapping` key-invariant
  value-covariant, `Generator` / `Coroutine` yield-cov send-contra return-cov. A
  3.8 CI test regenerates the table from the live `typing` and asserts equality,
  so it stays spec-sourced. `Tuple` and `Callable` are not in the table: they
  keep their own dedicated paths (tuple shape; contravariant params, covariant
  return).
- **The consequence (the reversal).** `List[bool] ⊑ List[int]` is now **False**
  (`list` invariant), where it was True. Only same-origin pairs whose one
  argument is a *subtype* of the other in an *invariant* container change; a
  covariant container (`Sequence`, `frozenset`) keeps its ordering, and a
  contravariant one reverses. Two `List[X]` overloads with subtype-related
  arguments become incomparable → ambiguous (set a priority), where before the
  narrower one won. Registering both is **legitimate** — a plain list declares
  no type arguments, so both genuinely apply to it, while an instance of `class
  Child(List[int])` is dispatched precisely (below) — so **no registration
  warning is emitted**; the ambiguity surfaces at the call, where a `priority`
  resolves it.
- **Differing origins: base-parameter substitution (V5).** A sub-hint whose
  origin differs from the super-hint's is re-expressed as a parametrisation of
  the super-hint's origin before its slots are compared, through the bases each
  class was *written* with (`__orig_bases__`, read off the class's own
  namespace; a class with none of its own is followed through `__bases__`),
  breadth-first and cycle-guarded, so the nearest base wins and, between bases
  at the same depth, the first listed (a diamond `class D(A, B)` over
  `A(Box[int])` and `B(Box[str])` is a `Box[int]`). Each base is filled in with the sub-hint's
  own arguments by typing's own subscription (`Box[T][bool]` is `Box[bool]`),
  pairing arguments to variables by identity, not position — so `class
  Flip(Pair[B, A], Generic[A, B])` makes `Flip[int, str]` a `Pair[str, int]`. A
  base with no free variable (`Box[int]`) is used as written, which is what
  makes `class IntBox(Box[int])` a `Box[int]`; a parametrised stdlib base
  (`List[int]`) is read positionally against a stdlib origin it subclasses that
  takes as many arguments (`Sequence`), exactly as two stdlib origins always
  were. When nothing maps — a runtime-only stdlib subclass (`Counter`, which
  records no parametrised base), a generic class written bare (`Sub`, like a
  bare `Box`), a `ParamSpec`/`TypeVarTuple` generic, a base whose substitution
  raises, or an arity mismatch (`Dict[K, V]` against `Iterable`) — the
  arguments are compared positionally, as before. `Tuple`, `Callable` and
  `Type` keep their dedicated paths.
- **Argument positions of a call are covariant.** A parameter *consumes* the
  argument; applicability is `type(v) ⊑ P`; "more specific" is "smaller P". This
  is `Tuple` covariance on the argument tuple — Julia's signatures *are* tuple
  types. This is a separate axis from the per-position variance *inside* a hint.
- **Value applicability reads only what a value declares.** Dispatch never
  reads a container's contents: `type([True])` is `list`, so
  `ishintstance([1], List[int])` stays True — a plain list matches every
  `List[…]`, whatever the container's variance. But when the value itself
  *declares* its parametrisation, `ishintstance(v, G[args])` uses it (V5),
  looking in order at (1) the instance's `__orig_class__` — set by typing when
  `Box[int]()` is called — read only off an instance of a `Generic` subclass,
  against any parametrised class hint (a user generic or a stdlib one, so
  `Row[int]()` for `class Row(Sequence[T])` is a `Sequence[int]` and not a
  `Sequence[str]`), and used when it re-expresses as a parametrisation of `G`;
  then (2) the class's written bases, when `type(v)` re-expresses as a
  parametrisation of `G` (`class Child(List[int])`). Either decides by
  `issubhint(declared, G[args])`. Otherwise the check stays shallow: a builtin
  instance (and `list[int]([1])`, which is a plain list), an instance built
  from the bare class, a `__slots__` class with no `__dict__` (nowhere to
  record it) or a frozen dataclass (typing swallows the `FrozenInstanceError`),
  `self` seen from inside `__init__` (the record is written only after
  `__init__` returns), and a declaration whose arguments hold a free `TypeVar`,
  `Any` or an unresolved name (`class C(List[T])`, `Box[Any]()`,
  `Box["int"]()`), since none says what the value holds. A `ParamSpec` /
  `TypeVarTuple` generic, whose arguments do not pair one per parameter, stays
  shallow too. Where a value does declare, the comparison is the spec's, so
  an invariant position tightens what used to match shallowly: `Box[int]()` no
  longer matches `Box[object]`, `Box[Union[int, str]]` or `Box[Optional[int]]`,
  and a `class Strs(List[str])` instance no longer matches `List[object]` —
  it still matches `List[Any]`, `list` and `Sequence[object]`. A `TypeVar` in
  the hint's invariant slot is solved (next bullets), so `Box[int]()` matches
  `Box[T ≤ object]`, `Box[TC(int, str)]` and `Box[T ≤ numbers.Real]`, though
  not `Box[T ≤ float]` (no numeric tower). `Tuple`, `Callable`, `Type[C]` and a
  `TypedDict` keep their own value checks.
- **`Any` / gradual typing [PEP 483, spec].** The spec separates *subtype of*
  from *consistent with*: `Any` is consistent with everything but is neither its
  subtype nor supertype; `object` is the nominal top. A dispatcher must still
  order `(object,)` vs `(Any,)`; the relation answers `object < Any`, so
  `Any`/unannotated is the widest catch-all and an `object` method beats it —
  Julia's reading. Inside an invariant slot the same consistency reading keeps a
  free `T`/`Any` above every `G[X]`, so a generic-fallback overload stays
  comparable.
- **A `TypeVar` in an invariant slot is solved (V5; owner decision).** On the
  super side it stands for *some* type within its bound or constraints, as a
  type checker solves it; on the sub side it stands for the whole family,
  which no single type contains. A free `T`/`Any` is the top, as above. A
  bounded `T ≤ B`: `G[A] ⊑ G[T]` iff `A ⊑ B` (a `TypeVar` `A` read by its own
  bound), so `Box[bool] ⊑ Box[TB ≤ int]` and a `Box[TB]` fallback sits above its
  specialisations; two bounded ones order by their bounds. A constrained
  `TC(C1, …, Cn)`: `A` must be equivalent to one `Ci` (a constrained `A`: each of
  its constraints to one). A `TypeVar` on the sub side against a concrete super
  side is never below it: `Box[TB] ⋢ Box[int]`, so `Box[int]` is strictly below
  `Box[TB]` (V3 read `TB` as exactly its bound: `Box[bool] ⋢ Box[TB]` and
  `Box[TB] ≡ Box[int]`). Each rule reduces to `⊑` or `≡` against the super
  side's bound or constraints, so the order stays a preorder (the law corpus
  holds `Box[TB]`, `Box[TC]`, `Src[TB]`, `Snk[TB]`). A **covariant** slot already
  read a `TypeVar` by its bound, which is what solving it gives. A
  **contravariant** slot keeps the bound reading: solving there asks whether
  the two sides *overlap* (`Snk[bool]` against `Snk[T ≤ int]` needs some `T`
  below both), which is not transitive — in a diamond `D(B, C)`, `B` and `C`
  both overlap `D` but not each other — and cannot be decided over an open
  class hierarchy. So `Snk[bool]()` does not match `Snk[T ≤ int]`, though a
  checker would accept it (a documented limitation).
- **A mixed-sign generic can leave parameterisations incomparable.** When one
  generic mixes signs across its positions — `Generator[Y_co, S_contra, R_co]`
  (yield covariant, send contravariant, return covariant) — `Any` is the top of
  a covariant slot but the *bottom* of a contravariant one, so `Generator[int,
  None, None]` and `Generator[int, Any, Any]` order neither way. Two such
  overloads are ambiguous; this is sound and law-preserving (an antisymmetric
  preorder permits incomparable elements).
- **The co/contra/inv/infer TypeVars of `bagof-hints`** describe *generic-class*
  variance, and are now **honoured** when such a variable declares a user
  generic's position: `hints.typevars.co.INT` gives a covariant position,
  `contra.INT` a contravariant one, `inv.INT` (and `infer`) an invariant one.
  PEP 484 / mypy forbid a variance-flagged `TypeVar` as a function *parameter*;
  used there (as an argument), it is read as its bound, and `Exact[int]` remains
  the way to ask for exactness.
- **Structural vs nominal: `ishintstance(v, H)` is deliberately not
  `issubhint(type(v), H)`.** The value-level check asks whether the value
  itself satisfies `H` — for a runtime `Protocol`, `Hashable`, or `Callable`
  that is answered *structurally*, from the value's own capabilities — while the
  hint-level relation asks a *nominal* question about the type. So
  `ishintstance(object(), Hashable)` is True (an `object` has `__hash__`) even
  though `issubhint(object, Hashable)` is False; a `Mapping` subclass that sets
  `__hash__ = None` is why the nominal answer is the safe one for the relation.
  The two are the two query modes on purpose — `f(value)` is structural,
  `f.resolve(hint)` is nominal — and collapsing them would make value dispatch
  miss a structurally-satisfied protocol. This is a gradual-typing corner (the
  same family as `Callable[...]`'s `...` wildcard): the affected rows —
  `Hashable` reached via `object`, and the `Callable[...]` wildcard — are kept
  out of the preorder-law corpus rather than special-cased.
- **Protocols with data members (#56).** Python refuses `issubclass()`
  against a runtime protocol that declares a data member (`name: str`),
  because whether a value has one is a property of the instance. The
  relation reads such a protocol member by member, split by where each is
  read:
  - *value level* — `v ∈ P` when `type(v)` lists `P` among its bases (as
    `isinstance` counts it), or when every **method** of `P` is defined by
    `type(v)` (not as `None`) and every **data member** is present on `v`:
    in its instance `__dict__` or anywhere in its class's MRO. Members are
    found statically, as `inspect.getattr_static` finds them and as
    `isinstance` does from 3.12 on — a property is not called and
    `__getattr__` is not asked — so the answer is the same on every
    supported Python and both `typing` / `typing_extensions` spellings,
    where `isinstance` itself is not (`typing` before 3.12 calls `hasattr`).
    Methods are read off the class, as a method-only protocol is decided,
    so only the data members depend on the instance; a method assigned on
    the instance alone is not counted, where 3.12+ `isinstance` would.
  - *hint level* — `C ⊑ P` when `C` lists `P` among its bases (a
    sub-protocol included), or when `C` is not a protocol and **declares**
    every member: each method defined by the class (not as `None`), each
    data member defined by the class itself (a class attribute, a property,
    a slot) or listed as a dataclass field that the generated `__init__`
    sets (`init=True`; a `field(init=False)` counts only through a plain
    default, which is a class attribute). Another protocol that does not
    list `P` is not below it, even with the same members. A data protocol is
    below a method-only protocol `Q` when it lists `Q`, or when its methods
    cover `Q`'s members — never through a data member, which Python's
    `issubclass` would accept from an annotation but an instance may hold
    alone.
  - *soundness* — the order must never put `C` below `P` while an instance
    of `C` fails `v ∈ P`. A **bare annotation** on a plain class
    (`name: str` with no value) promises nothing at runtime, so it does not
    declare the member: `Ann ⊑ HasName` is False, and an `Ann()` that never
    set `name` is not in `HasName`. A **dataclass field** its generated
    `__init__` sets does declare it: a dataclass promises such a field on
    every instance. A `field(init=False)` without a default is left for the
    class to set, so it declares nothing. The residue — a hand-written
    `__init__` that skips a field, a `del`, a call dispatched on `self` from
    inside `__init__` before the field is set, a subclass that redeclares an
    inherited field `init=False`, or one that sets an inherited method to
    `None` — breaks the class's own promise, and is documented rather than
    guarded.
  - *an overlap the order does not see* — a class that annotates a member
    and sets it in `__init__` (`Member`) is incomparable with the protocol
    at the hint level, yet every such instance is in both. An overload on
    each is therefore ambiguous for that instance, and `AmbiguousMethodError`
    is the answer: neither is more specific. Registration gives no warning
    and `ambiguities()` lists nothing, since both look for hint-level
    overlap. The remedies are the user's: give the class overload a higher
    `priority=`, or declare the member on the class (a class default, or a
    dataclass field) so that `Member ⊑ HasName`. The guide
    (`guide/protocols.md`) shows both, runnably. Whether dispatch should
    resolve this itself is an open owner decision; the behaviour is
    deliberately left as is.
  - *registration* — an overload on such a protocol is reachable like any
    other; a protocol that is not `runtime_checkable` is unchanged (it
    answers False, so an overload on it never matches).

---

## 3. TypeVars in signatures

| Kind | Applicability at a position | Position hint for specificity |
|---|---|---|
| unbound `T` | any value (`T ≡ Any`) | `Any` |
| `bound=B` | instance of `B` | `B` |
| constraints `(C1, C2)` | instance of some `Ci` (`bool` solves as `int`) | `Union[C1, C2]` |
| with `default=` (PEP 696) | as above — the default is a static-checker fallback, ignored | as above |

- **Origin does not matter.** Legacy `tx.TypeVar(...)`, `bagof.hints.typevars.*`,
  and PEP 695 `def f[T]` variables are all `TypeVar` instances and dispatch
  identically. `Signature.from_callable` reads hints only; `__type_params__` is
  consulted only to *name* variables in error messages.
- **Defaults (PEP 696) are irrelevant to dispatch**, across all three kinds.
  The relation reads the upper bound via a private `_typevar_upper(tv)` =
  bound / `Union[constraints]` / `Any`, using `getattr(tv, "__default__",
  tx.NoDefault)` (never `.has_default()`, which a native 3.12 PEP 695 TypeVar
  lacks). This fixes a measured bug: `TypeVar("TB", bound=float, default=int)`
  currently gives `TB ⊑ float` = False because `unwrap` prefers the default.
  The public `unwrap`'s default→constraints→bound contract stays (factories
  legitimately want the default to *build*).

**Repeated TypeVar `(T, T)` — v1 rule.** Consistency is over the **bound
arguments**: collect the arguments whose landed-slot hint is `T` (declared
parameters, and surplus positionals when `h_*` is `T`) — so `def same(x: T,
y: T)` is checked whether the call is `same(1, 2)`, `same(x=1, y=2)` or
`same(y=2, x=1)`. Applicability requires a *consistent solution*: those argument
classes must have a **greatest element** under `⊑` (`(int, bool)` solves
`T = int`; `(int, str)` is not applicable — a join to `object` would collapse
`(T, T)` to `(bound, bound)`). Constrained `T`: all solve to the *same*
constraint. A default-filled parameter annotated `T` contributes nothing (it is
not an argument). A `**kwargs: T` solves `T` for *applicability* jointly, over
its captured keywords together with every other slot carrying `T` — the same
greatest-element solve `*args: T` gets — and those keywords also form one group
for the specificity tie-break below (Phase 8d). This is the Pythonic middle
between Julia's strict diagonal and mypy's join.

A constrained `TypeVar` whose constraint is a runtime protocol with data
members (#56) is solved from the argument's *class*, `issubhint(type(v), Ci)`,
so a value that is in the protocol only by what its instance holds does not
select that constraint, at one position or several: declare the member on
the class, or use a `bound=`, which is checked value by value. A generic
protocol with data members (`HasItem[int]`, `item: T`) checks the members'
presence only; a structural value declares no arguments, so what it holds
under `item` is not compared with `int`, as a plain list's items are not.

**Specificity with TypeVars.** Position-wise a TypeVar is replaced by its table
entry, so `(int, int) < (T, T) ≡ (Any, Any)`. One tie-break inside `≡`: when
`A ≡ B` position-wise, the signature whose repeated TypeVars group *strictly
more* arguments into one consistent type is more specific (recovers Julia's
`same_type` outcome). **Implemented** (Phase 7): the tie-break is the last
selection step, reached only when two methods are already equally specific by
priority, MRO and tightness. It compares the two methods' TypeVar groupings
over the bound arguments — argument positions grouped by TypeVar identity — and
one grouping wins only when it is a strict refinement of the other (it ties
every pair the other ties, and at least one pair more), all landed hints being
equivalent. When neither grouping refines the other — equal groupings
(`(T, U)` vs `(U, T)`), or each tying a pair the other does not (`(T, T, U)` vs
`(T, U, U)`) — the pair stays incomparable and hence ambiguous. It therefore
only ever breaks a tie that was ambiguous before, never overturns a strict
specificity win, and never makes two genuinely independent signatures
comparable. A signature with a repeated group beats one with none, including a
fully unannotated `(Any, Any)` — the least-surprising reading of "more
constraint = more specific" where §3 was otherwise silent. A group of one
constrains nothing, so refinement needs a `TypeVar` bound at **two or more**
positions of the call: `*args: T` vs `*args` refines (and so resolves) for a
call of 2+ arguments, but ties into a single-element group — and is therefore
ambiguous — for 0 or 1 argument. `**kwargs: T` vs `**kwargs` behaves the same
way over the keywords a call spills into the catch-all (Phase 8d): it refines
for 2+ captured keywords and ties for 0 or 1.

**Hint-level `resolve` with TypeVars in the query** uses `issubhint` unchanged.

**Variadic kinds.** `*args: *Ts` / `*args: P.args` → an `Any` tail for a lone
element; a bare `Ts`/`P` (or a `Concatenate[...]`), or a top-level `Unpack[Ts]`,
as a *parameter* annotation is an invalid hint → registration `TypeError`
(a bare `Ts` on `*args` too, pointing at `*args: Unpack[Ts]`). Two open runs in
one tuple / parameter list — `Tuple[*Ts, *Us]` — are refused at registration
(PEP 646's single-unpack rule, which `typing` does not enforce at runtime).
**`ParamSpec` is solved at the hint level**: a `ParamSpec` named at several
top-level `Callable` slots must capture a consistent parameter list at each (the
analogue of a repeated `TypeVar`), the group consistent iff the captured lists
have a *greatest element* under the parameter-list order.

**`TypeVarTuple` is solved at the hint level too** (Phase-8(b), #28): a `*Ts`
named at several top-level `Tuple` slots must capture a consistent *run* at each
— the covariant tuple analogue — the group consistent iff the captured runs have
a greatest element under the tuple-shape order (`(int,)` & `(bool,)` → `(int,)`;
`(int,)` & `(str,)`, or runs of different arity, are inconsistent; a closed run
and an open run solve to the open one). A `*args: *Ts` absorbs its positionals
into one run of the same `Ts`, solved jointly with every `Tuple[..., *Ts]` slot,
so `(t: Tuple[*Ts], *args: *Ts)` applies to `resolve(Tuple[int, str], int, str)`
but not to `resolve(Tuple[int])` (the empty `*args` run disagrees with `(int,)`).
A `Callable[[int, *Ts], R]` list rides the `ParamSpec` open-tail path, its `*Ts`
tail solved by the same parameter-list machinery, keyed separately from any
tuple run of the same `Ts`. Only a top-level landed `Tuple`/`Callable` joins a
group; a `*Ts` nested in another hint, or a `*Ts` in the middle of a `Callable`
list with a fixed suffix (`Callable[[int, *Ts, str], R]`, which degrades to an
open `Concatenate[int, P]`-shape), is not jointly solved. `*Ts` never enters the
§3 repeated-grouping specificity tie-break — it is not a `TypeVar`, the
deliberate opposite of `*args: T`, so `*args: *Ts` vs `*args` stays ambiguous for
any positional count. **Value-level `P`/`Ts` solving is deferred (#33/#35)** — a
callable or tuple value is matched shallowly, its shape never inspected.

---

## 4. "Exact type" vs "subtype": `Exact[C]`

**Add `Exact[C]`**, spelled `bagof.dispatchers.Exact[int]`, implemented as
`tx.Annotated[int, EXACT]` with a private sentinel (the `bagof.magic`
`Frozen[int]` pattern), so type checkers see plain `int`. It is understood
*inside* `issubhint`/`ishintstance` (not a wrapper), so converters and
validators can use it too.

Why TypeVars are **not** enough (each verified): a bound TypeVar is *equivalent*
to its bound; a single-constraint TypeVar is illegal and two constraints still
accept subclasses; variance flags are rejected in signatures; `type[C]`
constrains class-object arguments, not instance exactness. (Julia gets exactness
free because concrete types are final; Python needs a marker.)

Runtime semantics (`Exact[C]` is a **leaf subtype** of `C` — this keeps `⊑`
a proper preorder, reflexive and transitive, with `Exact` present):
- values: applicable iff `type(v) is C`; `Exact` of a non-class (other than
  `NoneType`) → registration `TypeError`.
- hint-level: `issub(Exact[C], C)` is `True` (an exactly-`C` value is a `C`),
  but `issub(C, Exact[C])` is **`False`**, and `issub(D, Exact[C])` is `False`
  for a subclass `D` of `C` — nothing ordinary sits below `Exact[C]`. The only
  ordinary hints below it are literals whose every value has type exactly `C`:
  `issub(Literal[v], Exact[C])` iff `type(v) is C` (so `Literal[1] ⊑ Exact[int]`
  but `Literal[True]`, a `bool`, does not). `issub(Exact[C1], Exact[C2])` iff
  `C1 ≡ C2`.
- order: `Exact[C] < C`; `issub(Exact[C], P)` iff `issub(C, P)` — for a `P`
  that is not itself a Union/TypeVar containing `Exact` — those distribute
  first (`issub(Exact[C], Union[Exact[C], …])` and
  `issub(Exact[C], TypeVar(bound=Exact[C]))` are `True`, matched member by
  member rather than reduced to `issub(C, P)`, which would lose the
  exactness); `Exact[C]`/`Exact[D]` incomparable for `C ≢ D`. MRO refinement
  treats it as `C`; class-keyed, so it caches normally.
- resolution: hint-level `resolve()` (Phase 4) may *additionally* select an
  `Exact[C]` entry for a query `q ≡ C` — a lookup convenience layered on top of
  the relation, not a change to `⊑` itself (which keeps `q ⋢ Exact[C]`).

```python
from bagof.dispatchers import dispatch, Exact

@dispatch
def describe(x: int) -> str:          # int and any subclass, bool included
    return "an integer"

@dispatch
def describe(x: Exact[bool]) -> str:  # exactly bool
    return "a boolean"
```

`Exact` is for the reverse of the usual need: an `Exact[int]` that must *not*
fire for `True`, or a base class whose subclasses each need their own method.
Lives in `bagof.dispatchers._exact`; promote to `bagof-hints` only if a package
not depending on dispatchers needs it.

---

## 5. Ambiguity & no-match errors

`_errors.py`: `DispatchError(TypeError)` base (a call with unsupported types is a
`TypeError` in Python; `except TypeError:` keeps working); `NoMethodError`;
`AmbiguousMethodError`. Attributes: `.function`, `.call` (argument classes, or
hints for `resolve`), `.candidates`. Registration problems raise plain
`TypeError`/`ValueError`.

Ambiguity message (Julia's format, classes not values so no giant `repr`):

```
AmbiguousMethodError: area(int, int) is ambiguous.
Candidates:
  area(x: int, y: Any) @ shapes.py:12
  area(x: Any, y: int) @ shapes.py:15
Possible fix, define
  area(x: int, y: int)
```

The "possible fix" is the signature of the call's own classes (with `Exact[...]`
where a candidate used it): always applicable, always strictly more specific
than every candidate — a simpler, always-correct version of Julia's
type-intersection suggestion.

No-match message (Julia's, `!` marks the offending positions):

```
NoMethodError: no method matching area(str).
`area` has 3 methods, but none is defined for this combination of argument types.
Closest candidates:
  area(x: !int, y: int = 0) @ shapes.py:12
  area(x: !float) @ shapes.py:20
```

"Closest" = sorted by (positions applicable, arity match), top 8; no methods at
all → "`area` has no methods yet: the module that defines them has not been
imported". Raised in `Function.dispatch`/`resolve`, not in the `__call__` frame.
`Function.ambiguities()` (Julia's `detect_ambiguities`) returns incomparable
registered pairs that could both apply — a test helper CI can assert empty.
Registration only *warns* for guaranteed-ambiguous shapes, never raises, so
import order cannot break a program. A pair split by a differing `priority` is
resolved deterministically at the call — the higher priority wins — so it is
neither warned nor listed. A same-origin parametrised pair (`List[int]` vs
`List[str]`) is *not* warned or listed either: registering both is legitimate
and the incomparability only shows at the call.

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

>>> area(Circle(1))
3.14159
```

Redefining `area` in the same module *adds a method* rather than rebinding the
name.

**Registries & `Function` identity.** A registry is a `Dispatcher` instance, and
what identifies a `Function` (a named group of methods) *within* it depends on
which registry it is:

- **`bagof.dispatchers.dispatch`** (the module-level default) identifies a
  `Function` by **`(defining module, __qualname__)`** — so the same name in two
  modules is two **independent** functions; `moduleA`'s `area` and `moduleB`'s
  `area` never share methods or override each other.
- **A `Dispatcher()` you construct** identifies a `Function` by **`__qualname__`
  only**, module-independent — so every module that registers `area` into that
  *same shared instance* extends **one** `Function`. This is how you build a
  shared, cross-module generic function.

Get a `Function` to hand around through the dispatcher's **`functions`
namespace**: `d.functions["area"]` (item) or the sugar `d.functions.area`
(attribute); both **get-or-create** the `Function`, so `area = d.functions.area`
in one module and registrations in another compose. `@dispatch` also *returns*
the `Function`. Add methods with `@dispatch` on `def area(...)` (name from the
def) or `@area.register` / `@area.dispatch` on any function (name ignored — the
`def _` form). Explicit signatures overlay hints onto the wrapped function:
**positional hints are a tuple, named hints a dict, and keyword arguments are
registration options (`priority`), never hints** — e.g.
`area.register((int,), {"scale": float}, priority=0)(callable)`. A class passed
as an implementation (`area.register(SomeClass)`) dispatches on its
`__init__`/`__new__`. The def-less, hints-only form (`from_hints`,
`from_mapping` with a tuple key) is **positional-only**; to spell `/`, `*`,
`*args` or `**kwargs` without a function, pass a `Signature` object as the
`from_mapping` key (the escape hatch).

`d.functions` is a **protocol-only namespace**: it exposes the mapping protocol
(`d.functions["area"]`, `d.functions.area`, `for name in d.functions`,
`len(d.functions)`, `name in d.functions`) and **no named methods**, so *every*
function name — `register`, `items`, `map`, … — is safe there while the
`Dispatcher` itself keeps ordinary methods (`d.register(...)`, `d.clear_cache()`)
with no collision. (Attribute access ignores `_`-prefixed names so tool/REPL
probes never mint empty functions.) The namespace's key is the identity rule
above: on a `Dispatcher()` you construct it is the bare name, so `d.functions.area`
is unambiguous; on the module-level `dispatch` the key is `(module, __qualname__)`,
so there you take the `Function` from the decorator's return value (or a qualified
lookup) rather than by bare-name attribute.

```python
# a shared, cross-module generic function
# registry.py
from bagof.dispatchers import Dispatcher
dispatch = Dispatcher()
area = dispatch.functions.area       # the (initially empty) Function

# shapes.py
from registry import dispatch
@dispatch
def area(s: Circle) -> float: ...    # extends registry's `area`

# app.py
from registry import area
area(Circle(1))                      # sees shapes.py's method
```

Surface — **two namespaces, disjoint object sets**:

**`bagof.dispatchers`** (`__all__`, the clean public API) — `dispatch`,
`Dispatcher`, `Function`, `Method`, `Signature`, `Parameter`, `Exact`,
`DispatchError`, `NoMethodError`, `AmbiguousMethodError`.

**`bagof.dispatchers.core`** (`__all__`, the relation/introspection helpers
`bagof-core-magic` reuses) — `issubhint`, `ishintstance`, `resolve_hint`,
`safe_get_origin`, `safe_get_args`, `get_origin_uw`, `get_args_uw`, `unwrap`,
`normalise_hint`, `is_typeddict`, `typeddict_required_keys`, `safe_issubclass`,
`safe_isinstance`, `issubclassable`, `issubscriptable`, `get_concrete_type`,
`type2hint`, `eq_safenan`, `Unset`, `UNSET`, `NoneType`, `UnionType`,
`UNION_TYPES`. `Exact` is documented under the top-level API (its canonical home)
and understood by `issubhint`/`ishintstance` in `.core`.

Key objects:
- `Dispatcher()` — a registry. Identity rule as above (module-level `dispatch`:
  `(module, __qualname__)`; constructed instance: `__qualname__`). Functions are
  reached through the `d.functions` namespace (`d.functions.area` /
  `d.functions["area"]`, iterable by name), a **protocol-only** object carrying no
  named methods, so no function name collides with a `Dispatcher` method.
- `Function` — `__call__(*args, **kwargs)`, `dispatch(*args, **kwargs) -> Method`
  (bind-then-select without calling), `resolve(*hints, **named_hints,
  default=UNSET, ambiguity="raise") -> Method`, `register(...)`,
  `from_mapping(mapping)`, `methods`, `ambiguities()`, `clear_cache()`,
  `__get__` (binds `self`/`cls` as argument 0; unannotated → `Any`),
  `functools.update_wrapper` metadata. **Not** a `dict` subclass.
  `Function(dispatch_defaults=True)` treats default-filled dispatched parameters
  as arguments carrying their default value — the `_polymorph` semantics ("a
  default is as good as a value the caller wrote out"); off by default.
- `Method` — `signature`, `function`, `priority`, `__call__`, `__repr__`.
- `Parameter(name, hint, kind, default)` — frozen; `kind` mirrors
  `inspect.Parameter.kind`; `required = default is Parameter.empty`.
- `Signature` — `parameters: Mapping[str, Parameter]` (ordered), `.varargs`,
  `.varkw`, `.dispatched_names`; `from_callable(fn)` via `tx.get_type_hints(fn,
  include_extras=True)` for hints + `inspect.signature` for names/kinds/defaults
  (unannotated → `Any`; keyword-only params **are** dispatched, by name);
  `from_hints(*hints, **named_hints)` for explicit registration
  (`@dispatch((int,), {"scale": float})`); `bind(args, kwargs) -> Optional[Binding]`;
  `le(other, shape)` = specificity for a shape.
- `bagof.dispatchers.core.resolve_hint(hint, mapping, *, default=UNSET,
  ambiguity="raise")` — the hint-level functional API, `get_from_registry`'s
  successor (lives in `.core`, since it is the helper `bagof-core-magic` reuses).

Hooks — the naive `pre_check`/`post_check`/`__apply__`/`__error__` are **dropped**
(ordinary decorators; `dispatch()`/`resolve()` expose the only dispatch-specific
step; `resolve(default=)` is what `__error__` was for). Subclassing
`Function`/`Dispatcher` is supported for `bagof.magic`'s eventual field-name
needs.

Forward references: keep raw annotations on `NameError`/`TypeError`, retry on
first dispatch (`bagof.magic._resolve._Deferred` precedent); still unresolvable →
`NameError` naming the function/parameter. On 3.14 (PEP 649/749) prefer
`tx.get_type_hints(include_extras=True)`, then `annotationlib.get_annotations(fn,
format=Format.FORWARDREF)`, then raw `__annotations__` + deferral.

Caching & thread-safety — **two levels**, because the order is per shape:
1. A **shape plan** per `σ(C)`: for each method, its precomputed binding outcome
   for that shape (slot assignment or "cannot bind") and the pairwise `⊑_σ`
   matrix over the shape's arguments, plus which arguments are value-dependent
   (any method's hint there is `Literal`/`type[...]`, or a `Union`/`TypeVar`
   whose members or upper bound include one — a concrete TypedDict is now in
   this set too, since its value-level shape check reads the mapping's keys and
   value types, not the argument's type alone), and, short of that, which are
   *declaration-dependent* (any method's hint there is a parametrised class
   generic, user or stdlib — `Box[int]`, `Sequence[int]` — directly or through
   a `Union`/`TypeVar`/`Annotated`, but not `Tuple`/`Callable`/`Type[C]`/a
   `TypedDict`: its value check reads a `Generic` instance's `__orig_class__`,
   V5), and which are *member-dependent* (any method's hint there is a
   runtime protocol with data members, directly or through a
   `Union`/`TypeVar`/`Annotated`: its value check reads those members off the
   instance, #56 — recorded as the sorted union of the data members of every
   such protocol landing there). Bounded LRU over
   shapes; rebuilt on `register`.
2. Under each plan, a **call cache** keyed by `tuple(type(v_i)) + tuple((k,
   type(w_k)) for k in sorted keywords)`, with `(type, value)` at value-dependent
   arguments and `(type, __orig_class__)` at declaration-dependent ones — the
   recorded parametrisation, compared by identity (typing caches `Box[int]`, so
   every `Box[int]()` shares one entry, and identity never merges records `==`
   would, e.g. `Literal[1] == Literal[True]` on 3.8), never the instance. A
   position that is both keys on the value *and* the record — the value's own
   `==` need not see the record (a dataclass generic compares its fields). The
   record is read only off an instance of a `Generic` subclass, as the value
   check does, so no other value is probed. Typing's subscription cache is a
   bounded LRU, so after churn a fresh `Box[int]` object is a new entry: a
   missed hit, never a wrong method. A value that is not a `Generic` instance —
   a plain list at a `List[int]` argument — is not probed and keys as
   `(type, None)`; a cached call at such an argument costs roughly 0.1–0.2 µs
   more than at a type-keyed one (building, hashing and comparing the richer
   key). At a
   member-dependent argument the part carries a tuple of booleans, one per
   recorded data member, saying whether the value has it — read by the one
   function the value check uses, so the key always covers what the check
   reads, and never the value or its identity: every instance of a class
   holding the same members shares one entry. That read happens on every
   call, so a cached call at a member-dependent argument costs roughly 1 µs
   more than at a type-keyed one. A position with several dependences
   carries every part — the
   value, the record, the members — since none stands in for another (a
   value's `==` sees neither the record nor which attributes are set). An
   unhashable value at a value-dependent argument → uncached.
   Positional and keyword spellings of "the same" call are different shapes and
   therefore different keys (they can bind differently — required, not
   incidental).

Invalidate on every `register` (methods tuple rebuilt and published in one
assignment — the `_polymorph._Registry` pattern) and when `abc.get_cache_token()`
changes (as `singledispatch` does). `threading.Lock` on registration; lock-free
reads (matters on the free-threaded 3.13 build).

Errors render **named** signatures: `area(shape: !Circle, scale: float = 1.0) @
shapes.py:12`, with `/` and `*` markers, `*args: H`/`**kwargs: H`; the `!` sits on
the failing argument; a binding failure is rendered in words after the signature
(`— no parameter 'scake'`, `— missing 'y'`, `— 'x' is positional-only`).

---

## 7. Module layout

```
src/bagof/dispatchers/
  __init__.py        # CLEAN public API — re-exports ONLY: dispatch, Dispatcher, Function, Method,
                     #   Signature, Parameter, Exact, DispatchError, NoMethodError, AmbiguousMethodError
  _lattice.py        # equivalent(), TypeVar solving, value-dependence classifier —
                     #   mro_index() moved to core/_introspect.py so resolve_hint shares it (#19).
                     #   dispatch-internal, builds on core._relation. The value check stays in the
                     #   relation: the v1 engine calls core.ishintstance directly (no wrapper). The
                     #   Phase-8 TypedDict-shape value check is introduced then under an explicit,
                     #   accurate name.
  _signature.py      # Signature, Parameter, Binding, the precomputed per-method binder
  _method.py         # Method (+ deferred hint resolution)
  _function.py       # Function: methods tuple, per-shape order, two-level cache + abc token,
                     #   dispatch/resolve/__call__/__get__, ambiguities, register, from_mapping
  _dispatcher.py     # Dispatcher, the module-level `dispatch`, the `functions` namespace
  _errors.py         # DispatchError, NoMethodError, AmbiguousMethodError, message builders
  _constants.py      # `__dispatch_*__` attribute names
  core/
    __init__.py      # facade — re-exports the relation/introspection helpers reused by bagof.core.magic
    _compat.py       # NoneType, UnionType, UNION_TYPES, special-form pinning + structural fallback,
                     #   TypedDict markers, spellings(name), UnknownHintWarning
    _sentinels.py    # Unset, UNSET
    _exact.py        # Exact, EXACT sentinel, is_exact, exact_target  (re-exported at the top level too)
    _introspect.py   # safe_get_origin/args, get_origin_uw/args_uw, unwrap, normalise_hint (resolves
                     #   aliases/NewType/qualifiers), resolve_alias, resolve_newtype, issubclassable,
                     #   issubscriptable, is_typeddict, typeddict_required_keys, safe_issubclass,
                     #   safe_isinstance, get_concrete_type, type2hint, _typing_spelling, eq_safenan,
                     #   mro_index (shared by _lattice value dispatch and _registry resolve_hint, #19)
    _relation.py     # issubhint (+ branches), ishintstance (+ helpers) — Exact/Callable-variance-aware,
                     #   opaque fall-through for unknown forms, _typevar_upper
    _registry.py     # resolve_hint(): get_from_registry's successor over Function.from_mapping
tests/
  # moved verbatim from bagof-core-magic: test_issubhint*.py, test_subhint_semantics.py,
  #   test_typeddict_spellings.py, and the ishintstance/introspection halves of test_introspection.py
  test_introspect.py, test_relation_callable.py, test_relation_modern.py, test_exact.py, test_lattice.py,
  test_signature.py, test_dispatch_values.py, test_dispatch_hints.py, test_registry.py,
  test_typevars.py, test_cache.py, test_docstrings.py, test_import.py, test_module_surface.py
docs/index.md (dispatch), docs/core.md (the relation/introspection API, moved from core-magic's api.md)
```

`pyproject.toml`: **`dependencies = ["typing_extensions>=4.13"]`** — nothing
else, in Phase 0 and forever (optional test extra may add `numpy` only to
exercise `eq_safenan`). Private modules stay `_`-prefixed per house rule; the one
public subpackage is `core/`, whose `__init__` is a facade (no code) re-exporting
the helpers. The split is clean for griffe because the two namespaces export
**disjoint** object sets — dispatch objects at the top, relation/introspection
helpers under `.core` — so nothing is documented under two dotted paths; `Exact`
(the one object both use) is documented once, under the top-level API. Two doc
pages: `docs/index.md` (dispatch) and `docs/core.md` (the relation/introspection
API, moved from core-magic's `api.md`).

### 7a. Per-function relocation table

Provenance: all relocated code is original bagof code (no `NOTICE`, nothing
CPython-derived in core-magic — checked). The `dataclasses`-derived code is in
`bagof-magic`'s builder and does **not** move; no PSF notice travels.

| Function (core-magic `__init__.py` lines) | Verdict | Justification |
|---|---|---|
| `NoneType`, `UnionType`, `UNION_TYPES`, `_SPECIAL_FORMS`, `_is_special_form` (47–94) | PORT → `core/_compat.py` (+ structural special-form fallback, see §11.2) | version pinning already right for 3.8–3.14 |
| `Unset`, `UNSET` (97–122) | PORT → `core/_sentinels.py` | needed for `resolve(default=)` |
| `safe_get_origin`, `safe_get_args`, `get_origin_uw`, `get_args_uw` (1389–1437) | PORT | small, version-safe, tested |
| `unwrap`, `_unwrap_typevar` (1336–1386) | PORT | cycle guard + default→constraints→bound order are subtle and correct |
| `normalise_hint` (929–950) | REIMPLEMENT-IMPROVED | same signature; now also resolves `TypeAliasType`/`NewType` and strips transparent qualifiers |
| `is_typeddict` family, `typeddict_required_keys` (760–862) | PORT | both-spellings + 3.8 `__required_keys__` fallback already solved |
| `safe_isinstance` (899–926) | PORT | correct |
| `safe_issubclass` (865–896) | REIMPLEMENT-IMPROVED | `try/except TypeError → False` (non-runtime Protocols); `get_origin(x) is None` guard for the 3.9/3.10 `GenericAlias`-is-a-`type` trap |
| `issubclassable`, `issubscriptable` (737–757, 1478–1489) | REIMPLEMENT-IMPROVED | same `GenericAlias` trap guard |
| `type2hint`, `_TYPE2HINT_NAMES` (1492–1579) | PORT | explicit name table reused for pretty-printing |
| `_typing_spelling` (1582–1633) | PORT (private) | exact-key fast path in `resolve_hint` |
| `get_concrete_type` family (484–549) | PORT → `core/_introspect.py` | pure hint→class introspection; `MagicHint.fallback` uses it via re-export |
| `eq_safenan`, `_NaN` (1440–1475) | REIMPLEMENT-IMPROVED | drop numpy import at the root (`numbers.Real` covers numpy scalars via ABC); `REAL_TYPES` stays in core-magic |
| `ishintstance` family (953–1036) | REIMPLEMENT-IMPROVED | verbatim semantics + `Exact` awareness + Protocol guard; no TypedDict change |
| `issubhint` + all branches (1039–1333) | REIMPLEMENT-IMPROVED | keep structure + documented behaviour; add `Exact`, `Callable` contravariance, constrained-TypeVar/union symmetry, Protocol guard, `_typevar_upper`, opaque fall-through, bounded memo on hashable pairs |
| `get_from_registry` family (583–735) | REPLACE → `resolve_hint` in `core/_registry.py` | summed MRO distance is the rejected model; core-magic keeps a 2-line shim |
| `resolve_alias`, `resolve_newtype` | NEW → `core/_introspect.py` | PEP 695 `TypeAliasType` / `NewType` resolution (§11) |
| `get_default` (552–580) | STAYS in core-magic | about default *values*, not the relation |
| `MagicHint`, `MagicError`, `MultipleCauses`, `REAL_TYPES` (64–481) | STAYS in core-magic | the object model is what core-magic is after the pivot |

---

## 8. Reconciliation & migration

### 8.1 `bagof-core-magic` depends on `bagof-dispatchers` and re-exports

`bagof-core-magic/pyproject.toml` gains `"bagof-dispatchers"`. Its
`__init__.py` keeps its `__all__` **byte-for-byte** (so `test_module_surface.py`
passes) and becomes:

```python
from bagof.dispatchers.core import (       # relocated names: the same objects
    UNSET, Unset, NoneType, UnionType, UNION_TYPES, eq_safenan,
    get_concrete_type, get_origin_uw, get_args_uw, safe_get_origin,
    safe_get_args, safe_isinstance, safe_issubclass, ishintstance,
    issubhint, issubclassable, issubscriptable, is_typeddict,
    typeddict_required_keys, type2hint, unwrap, normalise_hint,
    resolve_hint as _resolve_hint,
)


def get_from_registry(hint: tx.Any, registry: dict) -> tx.Any:
    """(existing docstring, plus: new code should call bagof.dispatchers.core.resolve_hint)"""
    return _resolve_hint(hint, registry, default=None, ambiguity="warn")
```

followed by the unchanged `MagicHint`, `MagicError`, `MultipleCauses`,
`get_default`, `REAL_TYPES`. Every current `from bagof.core.magic import …`
keeps working. `ambiguity="warn"` picks by the refined order then registration
order and emits a `RuntimeWarning` — the one deliberate non-silence, so the
family learns where an ambiguous registry was silently resolved by insertion
order before.

Tests after the move: relation tests leave with the code; the registry tests
**stay** (`test_introspection.py:66–306, 484–554`) as the shim's contract, plus
one identity test per re-export (`bagof.core.magic.issubhint is
bagof.dispatchers.issubhint`). Explicitly-asserted parity deltas (all
improvements): an `Any`-keyed entry is reachable; a `Union` key matches its
members; a `List[int]` key matches a `List[bool]` query; unhashable keys work.
Any other difference is a bug in the port. No deprecation warnings this round.

### 8.2 Dependents — recommended end state: import from dispatchers directly

Measured import surfaces:
- **converters**: `MagicError, MagicHint, MultipleCauses, UNSET, get_args_uw,
  get_from_registry, get_origin_uw, ishintstance, issubhint, issubscriptable,
  safe_get_args, safe_get_origin, safe_isinstance, safe_issubclass,
  typeddict_required_keys, unwrap`
- **factories**: `MagicError, MagicHint, MultipleCauses, UNSET,
  get_from_registry, safe_get_args, safe_get_origin, safe_isinstance,
  safe_issubclass, typeddict_required_keys, unwrap`
- **validators**: `MagicError, MagicHint, MultipleCauses, UNSET,
  get_from_registry, ishintstance, safe_get_args, safe_get_origin,
  safe_isinstance, safe_issubclass, typeddict_required_keys, unwrap`
- **magic**: `UnionType`

**Settled `resolve_hint` parity decisions (#19).** Two corners where the
relation-based lookup first diverged from the summed-distance
`get_from_registry` are reconciled so the shim preserves consumer behaviour:

1. **MRO tie-break for equally specific class keys.** When several accepting
   keys are equally specific and the *query* is a class, the tie is broken by
   the query's MRO — the same RFC §2.2 step-3 refinement the value-dispatch
   path uses (`mro_index`, now shared from `core`). So `{Enum, str}` resolves
   `class Color(str, Enum)` to `str`, and the diamond `D(B, C)` resolves
   `{C, B}` to `B`, both order-independently, matching the old lookup. A tie
   MRO cannot break — a non-class query, or class keys equidistant in the MRO —
   still falls to `ambiguity` (warn + registration order, or raise).
2. **Bare `TypedDict` ⊑ `dict`.** The bare `TypedDict` marker ("any
   TypedDict") is now a sub-hint of `dict` at the relation level — every
   TypedDict value is a `dict` — while `dict` is *not* a sub-hint of the
   marker. `TypedDict` is therefore the unique most-specific key over `dict`
   for a TypedDict-subclass query, resolved order-independently (this is the
   nominal hint relation only, distinct from #30/#42's value-level extras).
3. **Bare `Literal` vs a concrete class stays ambiguous.** A bare `Literal`
   ("any literal value") is genuinely incomparable with a class (not every
   literal is an `int`) and MRO does not apply, so `{Literal, int}` vs
   `Literal[1]` remains an `ambiguity`-handled tie by design.

Keep `MagicHint`/`MagicError`/`MultipleCauses` from `bagof.core.magic`; import
everything else from `bagof.dispatchers.core` directly. The `get_from_registry(hint,
registry) or fallback` call sites (`converters/base.py:308`,
`validators/base.py:299`, `factories/base.py:257`) become `resolve_hint(hint,
registry, default=fallback, ambiguity="warn")`, then `"raise"` once each
registry is clean. Their registration dicts stay the registration API.

### 8.3 Improvements made at the source (formerly "extensions requested of core-magic")

- non-`runtime_checkable` Protocol answers `False` instead of raising;
- constrained TypeVar `≡` union of its constraints;
- `Callable` params contravariant, return covariant;
- `get_from_registry`'s summed distance → specificity order, with the same
  MRO refinement for equally specific class keys that value dispatch uses
  (§2.2, §8.1 #19);
- `Exact` understood by `issubhint`/`ishintstance`;
- `eq_safenan` numpy-free at the root.

### 8.4 `_polymorph` reconciliation (why name-aware dispatch subsumes it)

`bagof.magic._polymorph` already dispatches by **field name**:
`discriminants` computes once per class where each constrained field arrives (a
position for positional fields, its public name for keyword-able ones), `_read`
binds one call to those names (never a keyword-only field from `args` nor a
positional-only one from `kwargs`), `matches` is applicability with per-name value
specs (a `Literal`-shaped hint per field), and `rank = (priority, len(specs),
Σprecision, depth)` is a lexicographic stand-in for (priority, specificity over
names, MRO depth). That is exactly this model's `bind` + name-aware specificity,
restricted to the generated `__init__`'s parameter list. Migration sketch (a later
phase): each `on={...}` becomes a `Method` whose `Signature` is the owner's
`__init__` signature with the constrained names' hints replaced by the spec hints
and every other name `Any`; `select` becomes `Function(dispatch_defaults=True)
.dispatch(*args, **kwargs)`; `AmbiguousPolymorphError`/`NoPolymorphError` become
subclasses of the dispatch errors. The model needs no positional-to-name adapter.

---

## 9. Corner-case checklist (each is a test)

**Hints & lattice:** `Any` vs `object` (`object < Any`; unannotated = `Any`) ·
`List[int] < list ≡ List`; `List[int]`+`List[str]` at one position → ambiguous
at the call (set a priority), no registration warning — registering both is
legitimate · `Optional`/`Union` ordered by `issubhint`; `None` arg is
`NoneType` · bare `Union`/`Literal`/`Type` never applicable to a value (warn) ·
`Literal` value-dependent; `True`∉`Literal[1]`; NaN via `eq_safenan`; unhashable
→ uncached · `Tuple[X,...]`/`Tuple[X,Y]`/`tuple` chain; items not inspected;
`Tuple[()]` ok · `Callable` parametrisations ordered (post-fix contravariance);
`Callable[[int],R] < Callable[Concatenate[int,P],R] < Callable[P,R] ≡
Callable[...,R]` (`...`/bare `P` top the lists; row-flip §11.1, #32); a repeated
`ParamSpec` solved by greatest element at the hint level, value level shallow
(#33) · `type[X]` value-dependent; `type[bool] <
type[int] < type` · `Annotated` (non-`Exact`) `≡ X`; `Exact` of a non-class →
`TypeError` · TypedDict hint-level fine; value-level shape-checks the mapping ·
Protocols: runtime structural, two satisfied → ambiguous unless comparable,
non-runtime → registers, answers False at both levels, so it never matches
and the call raises `NoMethodError` · runtime protocol with data members
→ read member by member, value-level on the instance, keyed by which members
the value has (#56) · ABCs via `issubclass`; late `register()`
→ cache-token invalidation · diamond `D(B,C)` → `B`; B vs satisfied-ABC →
ambiguous · preorder laws property-tested.

**Modern forms (§11):** `type X = …` alias → its value; two aliases of one value
→ duplicate replace + warning; recursive alias → stops at origin · `L[int]` for
`type L[T] = list[T]` → `List[int]` · `NewType` → supertype · `Tuple[int, *Ts]`
chain (a `*Ts` run captures the rest; longer fixed prefix/suffix stricter;
`Tuple[int,*Ts]` vs `Tuple[*Ts,int]` and vs `Tuple[int,...]` incomparable;
`Tuple[*Ts] ≡ Tuple[Any,...]`); a repeated `*Ts` at top-level `Tuple` slots
solved jointly by greatest element (hint level, #35 for values), `*args: *Ts`
sharing that run; two open runs in one list → registration `TypeError`;
`Tuple[int,*Tuple[str,int]]` flattens · `Callable[P,R]` chain, `...`/`P` the
top; `Concatenate[int,P]` contravariant prefix, longer prefix more specific; a
repeated `P` solved by greatest element (hint level, #33 for values);
`Callable[[int,*Ts],R]` rides the open-tail path · `*args: P.args`/`*args: *Ts`
→ `Any` tail (a bare `Ts`/top-level `Unpack[Ts]` param → registration
`TypeError`); `**kwargs` ignored · `Never` param →
never applicable · `Required`/
`ReadOnly`/`Final`/`ClassVar` → as inner · unknown/future special form → opaque
≈ `Any` + one `UnknownHintWarning`, never raises · `TypeVar(bound=float,
default=int)` → bound wins, `int` arg does not match (no numeric-tower
promotion) · PEP 695 TypeVar without `__default__` → `getattr(..., NoDefault)` ·
`list[int]` on 3.9/3.10 not mistaken for a class · both-spelling `Unpack` on
3.11 recognised.

**Calls & parameters:** defaults → arity range, shorter fixed arity wins on
tightness · `*args: H` tail, fixed arity beats tail · zero-arg call → zero-arity
methods · methods in classes via `__get__`; `self`/`cls` = argument 0, `Any`
unless annotated · unhashable args uncached · lying `__class__` documented
(`type(v)` for cache/MRO, `isinstance` for applicability) · errors never `repr`
values.

**Name-aware binding:** same parameter positional in one call and keyword in
another → same method chosen · a keyword accepted by some methods only → the
others are unbindable, not ambiguous; a keyword accepted by none → `NoMethodError`
naming it with a did-you-mean · positional-only (`/`) → bind positionally only ·
keyword-only (`*`) → dispatched by name, never filled positionally · default-filled
names → excluded from applicability/specificity unless `dispatch_defaults=True` ·
`**kwargs: H` → extra keywords checked against `H` (unannotated → `Any`),
`Unpack[TD]` treated as unannotated (v1) · same names/different order → not
duplicates at registration; keyword calls ambiguous without priority · a shared
TypeVar on `*args` and a named param → consistency over all bound arguments · the
same value passed twice under two spellings (`f(1, x=1)`) → unbindable (Python
semantics) · `f(1, 2)` and `f(1, y=2)` are different shapes/keys · reduce-to-
positional theorem → property test against the reference positional engine.

**Registration & lifecycle:** new registration → cache cleared, order extended,
atomic publish · concurrent registration/call → never torn · same name in two
modules → distinct `Function`s; reload → replacement with `RuntimeWarning` ·
bad registrations → `TypeError` · equal-`priority` ties still ambiguous, never
override a strict specificity win · a differing-`priority` clash is resolved
deterministically, so it is neither warned nor listed (V4) · `ambiguities()`
lists warned pairs · a same-origin parametrised pair (`List[int]`/`List[str]`,
invariant `List[int]`/`List[bool]`) is incomparable and ambiguous at the call,
but registering both is legitimate so it is not warned or listed ·
core-magic re-export identity; `get_from_registry` parity.

---

## 10. Phasing

Gate per phase: `ruff check src tests`, `codespell`, suite green on **3.8 and
3.13** (plus 3.10/3.11/3.12 in the §11 matrix where feasible). Release order:
dispatchers must be published (or git-referenced in `tests/requirements.txt`, as
the siblings already do) before the core-magic shim PR merges.

- **Phase 0 — Rename the template.** `src/bagof/things` → `src/bagof/dispatchers`;
  `pyproject.toml` name/URLs/`versioningit.write.file`; `dependencies =
  ["typing_extensions>=4.13"]`; `zensical.toml`, `README.md`,
  `tests/test_import.py`. _(Trivial config/rename — suitable for the triage
  layer.)_
- **Phase 1 — Relocate/reimprove the hint relation into `core/` + core-magic
  shim.** Three commits: (1) pure move of `core/_compat`, `core/_sentinels`,
  `core/_introspect`, `core/_relation` (+ the `core/__init__` facade) with
  core-magic's relation test files moved verbatim and green (reviewable as a diff
  against core-magic); (2) the §8.3 source improvements + `core/_exact` + all §11.1
  modern-typing handling (`resolve_alias`, `resolve_newtype`, transparent
  qualifiers, `Never`/`LiteralString`/`TypeGuard`/`TypeIs`, `_typevar_upper`,
  `Unpack`/`ParamSpec`/`Concatenate` degradation, the `GenericAlias` trap,
  `spellings()`, structural `is_special_form`, opaque fall-through with
  `UnknownHintWarning`), each with new tests; (3) `eq_safenan` without numpy.
  Then the core-magic shim PR (§8.1), gated on the sibling suites passing.
  **[Fable Scope: Review Only] — MANDATORY.** This is the shared root under the
  whole family; the review checklist includes: both-spelling identity
  everywhere, no `return False` fall-through for unknown forms, alias cycle
  guard, no `.has_default()` calls, and commit-1-is-a-pure-move.
- **Phase 2 — `_lattice.py`.** `equivalent`, `mro_index`, TypeVar solving,
  value-dependence classifier (the value check stays in the relation —
  `core.ishintstance` — with no wrapper); preorder-law property tests over ~40
  hints incl. `Exact` and `Callable` pairs. Review recommended (lighter).
- **Phase 3 — `_signature.py` + `_method.py`.** `Parameter`, the ordered
  name→`Parameter` map, the precomputed per-method binder and `Binding`,
  `from_callable`/`from_hints(*hints, **named)`, `NewType`, deferred strings, the
  3.14 `annotationlib` path, `*args: *Ts`/`P.args` tails; typevar table +
  consistency; `Method` repr/source. **Differential-test the binder against
  `inspect.Signature.bind`** over generated signatures/shapes (success/failure and
  slot assignment must agree exactly, incl. `/`, `*`, defaults, `*args`,
  `**kwargs`, duplicate-value cases).
- **Phase 4 — `_function.py` + `_errors.py` + `core/_registry.py`.** Name-aware
  value dispatch (§2.2): shape plans, per-shape specificity, the two-level cache,
  abc token, `dispatch_defaults`, `register` with replacement + ambiguity warnings,
  both errors with named-signature rendering (`!` marker + binding-failure words),
  `Function.from_mapping`, `resolve_hint` with exact-key fast path, parity suite
  ported from core-magic. **[Fable Scope: Review Only] — MANDATORY** (the order is
  now per shape, binding adds a second inapplicability source, and the cache has a
  second level). Checklist: (a) reduce-to-positional theorem holds vs the
  reference engine; (b) binder ≡ `inspect.Signature.bind`; (c) `⊑_σ` is
  antisymmetric-up-to-`≡` and transitive per shape; (d) selection is independent
  of registration order for every shape; (e) shape plans dropped on `register` and
  `abc` token change; (f) no cross-shape key collision (`(1,2)` vs `(1,y=2)`);
  (g) default-filled parameters never influence selection unless
  `dispatch_defaults=True`; (h) `**kwargs: H` checked, missing annotation compares
  as `Any`; (i) error rendering names the failing argument and never `repr`s
  values; (j) `ambiguities()`'s heuristic shapes documented as such.
- **Phase 5 — `_dispatcher.py`, both `__init__.py` files, docs.** The clean
  top-level `__init__` + the `core/__init__` facade, the `functions` namespace,
  name grouping, `update_wrapper`; two doc pages (`docs/index.md` dispatch,
  `docs/core.md` the relation API); 3.8-safe `pycon`; `test_docstrings.py`,
  `test_module_surface.py` asserting the two namespaces stay disjoint and complete.
- **Phase 6 — Dependent import migration.** One PR each for
  converters/validators/factories (§8.2), `ambiguity="warn"` → `"raise"`; magic
  swaps `UnionType`.
- **Phase 7 — TypeVar specificity tie-break + `(T,T)` polish.** *Implemented.*
  The §3 repeated-group rule is the last selection step (after priority, MRO
  and tightness, before declaring ambiguity): a method whose repeated TypeVars
  group strictly more arguments into one consistent type wins an otherwise
  tied pair. Groupings are compared by TypeVar identity through `_lattice`;
  `_pair_ambiguous`/`ambiguities()` no longer flag a pair the tie-break
  separates. Genuinely independent groupings (`(T,U)`/`(U,T)`) and partial
  refinements stay ambiguous. Covered by `tests/test_function.py` (the
  repeated-TypeVar cases) and the repeated-TypeVar sweep in
  `tests/test_reference_engine.py`. **Review recommended** (least-precedented
  rule).
- **Phase 8 (v2).** TypedDict shape matching, introduced in `_lattice.py`
  under an explicit, accurate name, and flipping TypedDict to value-dependent
  (+ the separate owner decision on `ishintstance`/validators); joint TypeVar
  solving
  through `**kwargs: T` (**landed**, Phase 8d: the captured keywords group for
  the §3 grouping tie-break, so `**kwargs: T` beats an untyped `**kwargs`, and
  applicability solves `T` jointly over those keywords with every other slot
  carrying `T`, mirroring `*args: T`, covered by `tests/test_function.py`);
  `ParamSpec`/`Concatenate` solving (**landed**, Phase 8c: the `Callable`
  parameter-list order flipped so `...`/`P` top the lists — restoring
  transitivity, #32 — and a repeated `ParamSpec` is solved jointly by greatest
  element at the hint level, covered by `tests/test_paramspec_dispatch.py`,
  `tests/test_relation_callable.py` and `tests/test_lattice.py`);
  `TypeVarTuple`/`Unpack`/`*Ts` solving & ordering (**landed**, Phase 8b, #28:
  a `*Ts` is an open run of 0+ `Any` in a tuple, ordered through a tuple-shape
  classifier/matcher in `_issubargs`; a repeated `*Ts` at top-level `Tuple`
  slots — and a `Callable[[int,*Ts],R]` tail via the `ParamSpec` open-tail path
  — is solved jointly by greatest element at the hint level, with `*args: *Ts`
  sharing that run; covered by `tests/test_relation_tuple_variadic.py`,
  `tests/test_typevartuple_dispatch.py` and `tests/test_lattice.py`);
  `Callable` deep element check;
  `DeprecationWarning` `__getattr__` in core-magic; the `_polymorph` →
  `Function` migration (§8.4).

Deferred to a later phase (tracked in **#33** for `ParamSpec`, **#35** for
`TypeVarTuple`), out of scope here:

- **Value-level `ParamSpec`/`TypeVarTuple` solving (#33/#35).** A callable or
  tuple *value* is matched shallowly — its signature / element shape is never
  inspected — so `P`/`Ts` are not solved from values (`print` matches every
  `Callable[...]` slot, `(1, "a")` matches every `Tuple[...]` slot). Only the
  hint level (`resolve`, and hint-level applicability) solves them.
- The Phase-7 repeated-grouping tie-break is **not** extended to a repeated
  `ParamSpec` or `TypeVarTuple`: two incomparable open prefixes / runs stay
  ambiguous, and `*args: *Ts` never beats `*args`.
- `P.args` / `P.kwargs` as first-class components (they remain an `Any` tail),
  `ParamSpec(bound=)` and PEP 696 `ParamSpec` defaults, and the consistency of
  the same `P` at *nested* positions of one hint (only a top-level landed
  `Callable` joins the group) are all out of scope.
- More than one unpack per list, `Unpack[Ts]` on `**kwargs`, PEP 696
  `TypeVarTuple` defaults, `Ts` bounds (which do not exist), and `*Ts` in the
  middle of a `Callable` list with a fixed suffix (which degrades to an open
  shape) are all out of scope for `TypeVarTuple` (#35).

Non-goals (stated in the README): `invoke`/`next_method` fall-through,
return-type dispatch, dispatch on keyword-only parameters, static overload
synthesis, asymmetric (left-to-right) precedence.

---

## 11. Modern typing constructs & version support

**Design principle (first-class): support 3.8 → latest and future Pythons.**
Every construct is reached through `import typing_extensions as tx`; the relation
is uniform across spellings and forward-tolerant. Two measured facts shape this:

1. **Nothing crashes today, but a lot is silently dead.** For every modern form
   probed, the current `issubhint`/`ishintstance` return `False` both ways — so a
   method annotated with a `type` alias, `NewType`, `Never`, `TypeIs`,
   `Required[...]`, etc. registers and *never fires*. For dispatch that is worse
   than a crash; §11.2 turns "dead" into "understood or explicitly degraded".
2. **`typing.X is not tx.X` drifts by version** (measured: different sets on
   3.10/3.11/3.12/3.13; e.g. `tx.get_origin(tx.Unpack[Ts])` is
   `typing_extensions.Unpack` on 3.11 but `typing.Unpack` on 3.13; a native
   `type X = int` is not an instance of `tx.TypeAliasType`). **Every identity
   check on a special form must accept both spellings** — the `_TYPEDDICT_MARKERS`
   trick generalised via a `_compat.spellings(name)` helper.

### 11.1 Per-construct handling (native / typing_extensions backport / handling / v1 scope)

| Construct | Native | typing_extensions | Handling · v1 scope |
|---|---|---|---|
| **PEP 695 `def f[T]` / `class C[T]`** | 3.12 | — (syntax) | `__type_params__` holds ordinary `TypeVar`s; dispatch identical to legacy. Use `getattr(tv, "__default__", tx.NoDefault)` (native 3.12 TypeVars lack `has_default`). **Support** |
| **PEP 695 `type X = …` (`TypeAliasType`)** | 3.12 | 4.6+ | `resolve_alias(hint)` in `_introspect`: detect by duck type (`__value__` + `__type_params__`) or either spelling; return `__value__`; substitute args for a subscripted `G[int]` via typing's own `__getitem__`; recursive with cycle guard; called at top of `issubhint`/`ishintstance` and from `normalise_hint` (not `unwrap`). **Support** |
| **PEP 613 `TypeAlias`** | 3.10 | 4.x | the bound value is an ordinary hint; the bare marker falls under the unknown-form rule. **Support (trivial)** |
| **PEP 696 defaults** | 3.13 | 4.4+ | read bound/constraints only via `_typevar_upper`; default ignored. **Support** |
| **PEP 646 `TypeVarTuple`/`Unpack`/`*Ts`** | 3.11 | 4.1+ | `Unpack[Ts]` in `Tuple[...]` is an open run of 0+ `Any` (prefix/suffix split through the tuple-shape classifier in `_issubargs`); a fixed prefix/suffix captures the rest, a longer one is stricter, `Tuple[*Ts] ≡ Tuple[Any,...]`; a repeated `*Ts` at top-level `Tuple` slots (and `Callable[[int,*Ts],R]` via the open-tail path) solved jointly by greatest element at the hint level; `*args: *Ts` → `Any` tail sharing that run; bare `Ts`/top-level `Unpack[Ts]` param, `*args: Ts` without unpack, and two open runs in one list → registration `TypeError`. Value-level `Ts` solving deferred (#35); `*Ts` mid-list with a suffix degrades to an open shape; `Unpack[Ts]` on `**kwargs` and PEP 696 `TypeVarTuple` defaults are out. **Support (hint-level)** |
| **PEP 612 `ParamSpec`/`Concatenate`** | 3.10 | 4.x | `Callable[[int],R] < Callable[Concatenate[int,P],R] < Callable[P,R] ≡ Callable[...,R]` (`...`/`P` top the lists — row-flip, #32; restores transitivity); `Concatenate` a contravariant prefix, longer prefix more specific; a repeated `P` solved jointly by greatest element at the hint level; `*args: P.args` → `Any` tail; bare `P`/`Concatenate` as a param → registration `TypeError`. Value-level `P` solving deferred (#33). **Support (hint-level)** |
| **`NewType`** | 3.5/3.10 | typing_extensions class 3.8/3.9 | `resolve_newtype` → `__supertype__`, recursive, in `normalise_hint`. **Support** |
| **`Never`/`NoReturn`** | 3.11/3.6 | 4.1+ | bottom type; a `Never` param makes a method never applicable (explicit "forbid this combination"). **Support** |
| **`TypeGuard`/`TypeIs`** | 3.10/3.13 | 4.x/4.10+ | treat as `bool`. **Support** |
| **`LiteralString`** | 3.11 | 4.1+ | treat as `str`. **Support** |
| **`Self`** | 3.11 | 4.0+ | method → owner class via `__get__`; free function → unknown-form rule. **Support/Degrade** |
| **`Required`/`NotRequired`/`ReadOnly`** | 3.11/3.11/3.13 | 4.0+/4.9+ | transparent qualifiers → unwrap to inner. **Support** |
| **`Final`/`ClassVar`** | 3.8 | — | transparent qualifiers → inner. **Support** |
| **`Annotated`/`Doc` (PEP 727)** | 3.9/typing_extensions | 4.x/4.9+ | transparent except `EXACT`; `Annotated` is a class ≤3.12, not 3.13 (pinned). **Support** |
| **PEP 604 `X \| Y`** | 3.10 | — | `types.UnionType` in `UNION_TYPES`; 3.14 `types.UnionType is typing.Union` (pinned; verify). **Support** |
| **PEP 585 `list[int]`** | 3.9 | — | `isinstance(list[int], type)` is True on 3.9/3.10 → guard with `get_origin(x) is None` before treating as a class; `≡ List[int]`. **Support** |
| **User `Generic[T]`** | 3.8 | — | origin `isinstance`; args compared by the position's declared variance, read live off `__parameters__` (`covariant`/`contravariant`/unflagged→invariant, `infer_variance`→invariant); a subclass is compared through its written bases, and a value by the parametrisation it declares (`Box[int]()`, `class IntBox(Box[int])`), else shallow (§2.3; #50). **Support** |
| **Unknown / future form** | — | typing_extensions first | opaque rule (§11.2). **Degrade** |

Numeric-tower note (docs): `issubhint(int, T_bound_float)` is False — the spec's
float/complex promotion is a static convention; dispatch follows runtime
`isinstance`. Users write `numbers.Real` or `Union[int, float]`.

### 11.2 Forward-compatibility strategy

**Rule:** an unrecognised hint is opaque and behaves like `Any` — accepted by
everything as a superhint (`issubhint(x, unknown)` → True, `ishintstance(v,
unknown)` → True), a subhint only of itself and `Any` — and its first sighting
emits one `UnknownHintWarning(RuntimeWarning)` naming the form. This keeps a
method *reachable* rather than silently dead. Handling order inside the relation:
(1) `normalise_hint` (None, `NewType`, `TypeAliasType`, transparent qualifiers);
(2) known non-class forms by both-spelling identity; (3) an origin that is a
class → structural path; (4) anything else → opaque rule. There is no
`return False` fall-through for "unknown".

**`_SPECIAL_FORMS` audit.** Keep the explicit tuple as the fast path and add a
structural fallback in `_compat`: `x` is a special form if it is a `type` whose
`__module__` is `typing`/`typing_extensions` and it is not `Generic`,
`Protocol`, or a TypedDict marker. Together with `try/except TypeError → False`
in `safe_issubclass`, no future form can raise out of the relation.

### 11.3 Support-matrix conclusion (one line per version)

- **3.8** — everything modern from typing_extensions; no `list[int]`/`X | Y` at
  runtime, so stringified annotations of those raise `TypeError` in
  `get_type_hints` (catch & defer); `NewType` is a typing_extensions class (verify).
- **3.9** — PEP 585 arrives and `isinstance(list[int], type)` is True → guard
  `issubclassable`/`safe_issubclass` with `get_origin(x) is None`.
- **3.10** — `X | Y`, `ParamSpec`/`Concatenate`/`TypeAlias`/`TypeGuard` native;
  `NewType` becomes a class; both spellings required.
- **3.11** — `Any` becomes a class (pinned); `Never`/`Self`/`LiteralString`/
  `TypeVarTuple`/`Unpack`/`Required` native; `GenericAlias` trap fixed;
  typing_extensions still owns `Unpack`/`TypeVarTuple`/`TypeVar` — both spellings.
  Measured: `tx.Unpack is not typing.Unpack`, and the star syntax
  `Tuple[int, *Ts]` yields `typing.Unpack[Ts]` while `tx.Unpack[Ts]` yields the
  `typing_extensions` one (not `==`), so `_UNPACK_FORMS = spellings("Unpack")`
  (identity over both) is mandatory; `tx.TypeVarTuple is not typing.TypeVarTuple`
  too, so a `TypeVarTuple` is recognised only by `isinstance`, never by class
  identity. Re-registration switching `*Ts` ↔ `Unpack[Ts]` reads as a new method
  (structural-eq compares the origin by `is`), which is acceptable.
- **3.12** — PEP 695 syntax + native `TypeAliasType` (duck-type it); native PEP
  695 TypeVars lack `__default__` (`getattr(..., NoDefault)`); `Annotated` still
  a class.
- **3.13** — PEP 696 defaults native; `TypeIs`/`ReadOnly` native; `Annotated` no
  longer a class; free-threaded build exists (the registration lock matters).
- **3.14+** — PEP 649/749 lazy annotations (`annotationlib`,
  `Format.FORWARDREF`); `Union` becomes a class, `types.UnionType is
  typing.Union` (pinned; verify); anything newer is covered by the opaque rule,
  the structural special-form fallback and `spellings()` — a warning, not a
  crash.

---

## Appendix — critical source files

- `bagof-core-magic/src/bagof/core/magic/__init__.py` — source of every
  relocated function (lines in §7a); shrinks to the object model + re-exports.
- `bagof-core-magic/tests/test_issubhint*.py`, `test_subhint_semantics.py`,
  `test_typeddict_spellings.py`, `test_introspection.py` — tests that move with
  the code (registry subset stays as the shim's contract).
- `bagof-magic/src/bagof/magic/_polymorph.py` — precedent for priority,
  replacement and atomic registries.
- `bagof-converters/src/bagof/converters/base.py` (236–308),
  `bagof-validators/src/bagof/validators/base.py` (~299),
  `bagof-factories/src/bagof/factories/base.py` (~257) — Phase 6 call sites.

## Appendix — open owner decisions

1. **Verbatim citation verification** — allowlist `peps.python.org`,
   `typing.python.org`, `docs.julialang.org`, `en.wikipedia.org` in the
   environment's Network settings to let a follow-up pass confirm the marked
   citations and Julia error text.
2. **v2 `ishintstance` TypedDict shape check** — changing what a TypedDict hint
   accepts at the value level affects `validators/base.py:108` directly; that is
   a validators decision, kept out of the relocation.
