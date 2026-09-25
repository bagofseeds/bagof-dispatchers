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
   Its only dependency is `typing_extensions`; it depends on neither
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

5. **Support Python 3.8 through the latest and future versions** (§11). Every
   construct is reached through `import typing_extensions as tx`; the relation is
   uniform across spellings and **forward-tolerant** — an unrecognised or future
   hint degrades to `Any`-like behaviour and warns, never crashes.

6. **Repeated TypeVars (`(T, T)`)** are supported for applicability in v1 with a
   narrow specificity tie-break; Julia's full diagonal semantics are deferred.
   Flagged for review.

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
| `List[bool] ≤ List[int]`, `Dict[str,int] ≤ Dict[str,object]` | True | diverges from PEP 483 invariance of mutable containers — deliberate for value dispatch (§2.3) |
| `Tuple[int] < Tuple[int, ...] < tuple`, `Tuple[int,...] ≤ Tuple[Any,...]` | True chain | covariant `Tuple` |
| `Literal[True] < bool`, `Literal[1] < Literal[1,2]` | True | literals are the bottom |
| `int ≤ Optional[int]`, `None ≤ Optional[int]`, `Optional[int] ≤ Union[int,str,None]` | True | union rules |
| `type[bool] < type[int] < type` | True | `Type[C]` covariant |
| `List[int] ≤ Sequence[int]`, `List[int] ≤ Iterable` | True | ABC registration honoured |
| `Annotated[int,'x'] ≡ int` | T/T | metadata invisible → `Exact` handled before delegating |
| `Callable[[int],str]` vs `Callable[[bool],str]` | F/F → **contravariant (post-fix)** | params compared contravariantly, return covariantly |
| `int ≤ P` (non-`runtime_checkable` Protocol) | raises → **False (post-fix)** | guarded; registration still refuses with a friendly message |
| `list ≤ RP` (runtime protocol) | True | protocols dispatch structurally |
| `dict ≤ TD`, `TD ≤ dict`, `TD ≤ Mapping` | F/T/T | TypedDict orders correctly at hint level |
| `int ≤ Union` (bare) | False | bare `Union`/`Literal`/`Type` mean "is one of these"; dead for value dispatch |

Value level (`ishintstance`): `{'a':1} in TD` → **False**; `[1] in List[str]`
→ **True** (items never inspected); `print in Callable[[int],str]` → True;
`True in Literal[1]` → False (PEP 586); `1 in T` True, `'x' in TB` False.

### 2.2 Definitions (normative)

`⊑` = `issubhint` (which now handles `Exact` internally).

- **Signature** `S = (h_1, …, h_n; tail)`; `tail` from `*args: h_*`; arity range
  `[min_arity, max_arity]` (defaults lower `min_arity`; a tail makes
  `max_arity = ∞`). Keyword-only parameters are **not** part of the signature
  (Julia rule; `plum` agrees).
- **Position hint** `S[i]` = `h_i` for `i < n`, else `h_*`.
- **Applicable to values** `(v_1..v_m)`: `min_arity ≤ m ≤ max_arity`,
  `∀i: is_instance(v_i, S[i])`, plus §3 TypeVar consistency.
- **Applicable to hints** `(q_1..q_m)`: same, with `issub(q_i, S[i])`.
- **Specificity** `A ⊑ B` at arity `m`: `∀i<m: A[i] ⊑ B[i]` plus consistency.
  `A < B` iff `A ⊑ B ∧ ¬(B ⊑ A)`; `A ≡ B` iff both.
- **Selection.** `Max` = applicable methods with no strictly more specific
  applicable method. Then, in order:
  1. `|Max| = 1` → done.
  2. Drop members strictly dominated under **explicit priority** (higher wins;
     default 0).
  3. Drop members strictly dominated under **MRO refinement**: A dominates B iff
     at every position `A[i] ≡ B[i]`, or both hints (unwrapped, non-`Exact`) are
     classes in `type(v_i).__mro__` with `index(A[i]) ≤ index(B[i])`, strictly
     `<` somewhere. Resolves `D(B, C)` to `B`, as `singledispatch` /
     `get_from_registry` do. Protocols/ABCs not in the MRO, unions, literals and
     parametrised generics give no refinement.
  4. Drop members dominated under **arity tightness**: smaller `max_arity`, then
     larger `min_arity` (fixed arity beats a `*args`/default that merely tolerates
     the count; Julia orders fixed arity before `Vararg`).
  5. `|Max| > 1` → **`AmbiguousMethodError`**; `|Max| = 0` → **`NoMethodError`**.

  Priority precedes MRO because explicit beats implicit. Steps 2–4 are partial
  refinements, so a cross-position conflict remains ambiguous; and none ever
  overrides a strict specificity win (tested).

### 2.3 Where variance enters (reconciled with PEP 483 and `bagof.hints.typevars`)

PEP 483: for `t2 ≤ t1`, `G` is *covariant* if `G[t2] ≤ G[t1]`, *contravariant*
if `G[t1] ≤ G[t2]`, *invariant* if neither. `Tuple`/`FrozenSet`/`Union`/`Type`
covariant; mutable containers invariant; `Callable` contravariant in params,
covariant in return.

- **Argument positions of a call are covariant.** A parameter *consumes* the
  argument; applicability is `type(v) ⊑ P`; "more specific" is "smaller P". This
  is `Tuple` covariance on the argument tuple — Julia's signatures *are* tuple
  types.
- **Inside a hint**, the relation is covariant everywhere, including
  `List[bool] ⊑ List[int]`, which PEP 483 calls unsound for a *reference*. For
  dispatch *on a value* the question is "can this value be described by this
  hint" — values carry no type arguments (`type([True])` is `list`), so
  covariance is the only reading that yields any order between parametrised
  hints. Julia can be invariant only because its parametric types are concrete
  at runtime. **Documented caveat + registration warning:** `List[int]` and
  `List[str]` are both applicable to any list and incomparable.
- **Real contravariance** (`Callable` params) is handled in Phase 1 (post-fix).
- **`Any` / gradual typing [PEP 483, spec].** The spec separates *subtype of*
  from *consistent with*: `Any` is consistent with everything but is neither its
  subtype nor supertype; `object` is the nominal top. A dispatcher must still
  order `(object,)` vs `(Any,)`; the relation answers `object < Any`, so
  `Any`/unannotated is the widest catch-all and an `object` method beats it —
  Julia's reading. Documented.
- **The co/contra/inv/infer TypeVars of `bagof-hints`** describe *generic-class*
  variance, and PEP 484 / mypy forbid a variance-flagged TypeVar as a function
  parameter. The dispatcher reads `__bound__`/`__constraints__` only and
  **ignores the variance flags**: `hints.typevars.co.INT` and `inv.INT` dispatch
  identically. Docs say so and point at `Exact[int]` for exactness.

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

**Repeated TypeVar `(T, T)` — v1 rule.** Applicability requires a *consistent
solution*: the argument classes at `T`'s positions must have a **greatest
element** under `⊑` (`(int, bool)` solves `T = int`; `(int, str)` is not
applicable — a join to `object` would collapse `(T, T)` to `(bound, bound)`).
Constrained `T`: all positions solve to the *same* constraint. This is the
Pythonic middle between Julia's strict diagonal and mypy's join.

**Specificity with TypeVars.** Position-wise a TypeVar is replaced by its table
entry, so `(int, int) < (T, T) ≡ (Any, Any)`. One tie-break inside `≡`: when
`A ≡ B` position-wise, the signature with more repeated-TypeVar position groups
is strictly more specific (recovers Julia's `same_type` outcome). **Flag:
review** — least-precedented rule.

**Hint-level `resolve` with TypeVars in the query** uses `issubhint` unchanged.

**Variadic kinds.** `*args: *Ts` / `*args: P.args` → an `Any` tail; a bare
`Ts`/`P` as a *parameter* annotation is an invalid hint → registration
`TypeError`. Full `TypeVarTuple`/`ParamSpec` solving is deferred (§11, v2).

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

Runtime semantics:
- values: applicable iff `type(v) is C`; `Exact` of a non-class (other than
  `NoneType`) → registration `TypeError`.
- hint-level: `issub(q, Exact[C])` iff `q ≡ C`, so `Exact[int]` matches query
  `int` (and `Annotated[int, …]`, `inv.INT`) but not `bool`.
- order: `Exact[C] < C`; `issub(Exact[C], P)` iff `issub(C, P)`;
  `Exact[C]`/`Exact[D]` incomparable for `C ≢ D`. MRO refinement treats it as
  `C`; class-keyed, so it caches normally.

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
import order cannot break a program.

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

Redefining `area` in the same module/class *adds a method*: `dispatch` groups by
`(module, __qualname__)` (plum's design; reads like Julia's "define another
method"). Explicit signatures: `@dispatch(int, str)` and
`f.register(float, str)(callable)`.

Surface (`__all__`), two documented groups:

**Dispatch** — `dispatch`, `Dispatcher`, `Function`, `Method`, `Signature`,
`Exact`, `resolve_hint`, `DispatchError`, `NoMethodError`,
`AmbiguousMethodError`.

**Hints** — `issubhint`, `ishintstance`, `safe_get_origin`, `safe_get_args`,
`get_origin_uw`, `get_args_uw`, `unwrap`, `normalise_hint`, `is_typeddict`,
`typeddict_required_keys`, `safe_issubclass`, `safe_isinstance`,
`issubclassable`, `issubscriptable`, `get_concrete_type`, `type2hint`,
`eq_safenan`, `Unset`, `UNSET`, `NoneType`, `UnionType`, `UNION_TYPES`.

Key objects:
- `Dispatcher()` — namespace of `Function`s; `dispatch = Dispatcher()` is the
  module default. Own instances isolate a library's names.
- `Function` — `__call__`, `dispatch(*args) -> Method`, `resolve(*hints,
  default=UNSET, ambiguity="raise") -> Method`, `register(...)`,
  `from_mapping(mapping)`, `methods`, `ambiguities()`, `clear_cache()`,
  `__get__` (binds `self`; unannotated `self` → `Any`), `functools.update_wrapper`
  metadata. **Not** a `dict` subclass.
- `Method` — `signature`, `function`, `priority`, `__call__`, `__repr__`.
- `Signature` — `from_callable(fn)` via `tx.get_type_hints(fn,
  include_extras=True)` (unannotated → `Any`; `*args: H` → tail; defaults →
  `min_arity`; kw-only ignored), `from_hints`, `__le__`/`__lt__` = specificity.
- `resolve_hint(hint, mapping, *, default=UNSET, ambiguity="raise")` — the
  hint-level functional API, `get_from_registry`'s successor.

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

Caching & thread-safety: per-`Function` dict keyed by `tuple(type(a) for a in
args)`; value-dependent positions (`Literal`, `type[...]`, TypedDict in v2)
contribute `(type(a), a)` when hashable, else uncached. Invalidate on every
`register` (methods tuple rebuilt and published in one assignment — the
`_polymorph._Registry` pattern) and when `abc.get_cache_token()` changes (as
`singledispatch` does). Pairwise specificity precomputed at registration.
`threading.Lock` on registration; lock-free reads (matters on the free-threaded
3.13 build).

---

## 7. Module layout

```
src/bagof/dispatchers/
  __init__.py       # re-exports + __all__ only
  _compat.py        # NoneType, UnionType, UNION_TYPES, special-form pinning + structural fallback,
                    #   TypedDict markers, spellings(name), UnknownHintWarning
  _sentinels.py     # Unset, UNSET
  _introspect.py    # safe_get_origin/args, get_origin_uw/args_uw, unwrap, normalise_hint (now resolves
                    #   aliases/NewType/qualifiers), resolve_alias, resolve_newtype, issubclassable,
                    #   issubscriptable, is_typeddict, typeddict_required_keys, safe_issubclass,
                    #   safe_isinstance, get_concrete_type, type2hint, _typing_spelling, eq_safenan
  _exact.py         # Exact, EXACT sentinel, is_exact, exact_target
  _relation.py      # issubhint (+ branches) and ishintstance (+ helpers) — Exact-aware, Callable-variance-aware,
                    #   opaque fall-through for unknown forms, _typevar_upper
  _lattice.py       # equivalent(), is_instance() (Exact + v2 TypedDict shape), mro_index(), TypeVar solving,
                    #   value-dependence classifier — calls issubhint directly
  _signature.py     # Signature
  _method.py        # Method (+ deferred hint resolution)
  _function.py      # Function: methods tuple, pairwise order, cache + abc token, dispatch/resolve/__call__/
                    #   __get__, ambiguities, register, from_mapping
  _registry.py      # resolve_hint(): get_from_registry's successor over Function.from_mapping
  _dispatcher.py    # Dispatcher, `dispatch`
  _errors.py        # DispatchError, NoMethodError, AmbiguousMethodError, message builders
  _constants.py     # `__dispatch_*__` attribute names
tests/
  # moved verbatim from bagof-core-magic: test_issubhint*.py, test_subhint_semantics.py,
  #   test_typeddict_spellings.py, and the ishintstance/introspection halves of test_introspection.py
  test_introspect.py, test_relation_callable.py, test_relation_modern.py, test_exact.py, test_lattice.py,
  test_signature.py, test_dispatch_values.py, test_dispatch_hints.py, test_registry.py,
  test_typevars.py, test_cache.py, test_docstrings.py, test_import.py, test_module_surface.py
docs/index.md (dispatch), docs/hints.md (the relation/introspection API, moved from core-magic's api.md)
```

`pyproject.toml`: **`dependencies = ["typing_extensions>=4.13"]`** — nothing
else, in Phase 0 and forever (optional test extra may add `numpy` only to
exercise `eq_safenan`). Flat private modules per house rule; a
`bagof.dispatchers.hints` subpackage was rejected because griffe would document
the same objects under two dotted paths.

### 7a. Per-function relocation table

Provenance: all relocated code is original bagof code (no `NOTICE`, nothing
CPython-derived in core-magic — checked). The `dataclasses`-derived code is in
`bagof-magic`'s builder and does **not** move; no PSF notice travels.

| Function (core-magic `__init__.py` lines) | Verdict | Justification |
|---|---|---|
| `NoneType`, `UnionType`, `UNION_TYPES`, `_SPECIAL_FORMS`, `_is_special_form` (47–94) | PORT → `_compat.py` (+ structural special-form fallback, see §11.2) | version pinning already right for 3.8–3.14 |
| `Unset`, `UNSET` (97–122) | PORT → `_sentinels.py` | needed for `resolve(default=)` |
| `safe_get_origin`, `safe_get_args`, `get_origin_uw`, `get_args_uw` (1389–1437) | PORT | small, version-safe, tested |
| `unwrap`, `_unwrap_typevar` (1336–1386) | PORT | cycle guard + default→constraints→bound order are subtle and correct |
| `normalise_hint` (929–950) | REIMPLEMENT-IMPROVED | same signature; now also resolves `TypeAliasType`/`NewType` and strips transparent qualifiers |
| `is_typeddict` family, `typeddict_required_keys` (760–862) | PORT | both-spellings + 3.8 `__required_keys__` fallback already solved |
| `safe_isinstance` (899–926) | PORT | correct |
| `safe_issubclass` (865–896) | REIMPLEMENT-IMPROVED | `try/except TypeError → False` (non-runtime Protocols); `get_origin(x) is None` guard for the 3.9/3.10 `GenericAlias`-is-a-`type` trap |
| `issubclassable`, `issubscriptable` (737–757, 1478–1489) | REIMPLEMENT-IMPROVED | same `GenericAlias` trap guard |
| `type2hint`, `_TYPE2HINT_NAMES` (1492–1579) | PORT | explicit name table reused for pretty-printing |
| `_typing_spelling` (1582–1633) | PORT (private) | exact-key fast path in `resolve_hint` |
| `get_concrete_type` family (484–549) | PORT → `_introspect.py` | pure hint→class introspection; `MagicHint.fallback` uses it via re-export |
| `eq_safenan`, `_NaN` (1440–1475) | REIMPLEMENT-IMPROVED | drop numpy import at the root (`numbers.Real` covers numpy scalars via ABC); `REAL_TYPES` stays in core-magic |
| `ishintstance` family (953–1036) | REIMPLEMENT-IMPROVED | verbatim semantics + `Exact` awareness + Protocol guard; no TypedDict change |
| `issubhint` + all branches (1039–1333) | REIMPLEMENT-IMPROVED | keep structure + documented behaviour; add `Exact`, `Callable` contravariance, constrained-TypeVar/union symmetry, Protocol guard, `_typevar_upper`, opaque fall-through, bounded memo on hashable pairs |
| `get_from_registry` family (583–735) | REPLACE → `resolve_hint` in `_registry.py` | summed MRO distance is the rejected model; core-magic keeps a 2-line shim |
| `resolve_alias`, `resolve_newtype` | NEW → `_introspect.py` | PEP 695 `TypeAliasType` / `NewType` resolution (§11) |
| `get_default` (552–580) | STAYS in core-magic | about default *values*, not the relation |
| `MagicHint`, `MagicError`, `MultipleCauses`, `REAL_TYPES` (64–481) | STAYS in core-magic | the object model is what core-magic is after the pivot |

---

## 8. Reconciliation & migration

### 8.1 `bagof-core-magic` depends on `bagof-dispatchers` and re-exports

`bagof-core-magic/pyproject.toml` gains `"bagof-dispatchers"`. Its
`__init__.py` keeps its `__all__` **byte-for-byte** (so `test_module_surface.py`
passes) and becomes:

```python
from bagof.dispatchers import (            # relocated names: the same objects
    UNSET, Unset, NoneType, UnionType, UNION_TYPES, eq_safenan,
    get_concrete_type, get_origin_uw, get_args_uw, safe_get_origin,
    safe_get_args, safe_isinstance, safe_issubclass, ishintstance,
    issubhint, issubclassable, issubscriptable, is_typeddict,
    typeddict_required_keys, type2hint, unwrap, normalise_hint,
)
from bagof.dispatchers import resolve_hint as _resolve_hint


def get_from_registry(hint: tx.Any, registry: dict) -> tx.Any:
    """(existing docstring, plus: new code should call bagof.dispatchers.resolve_hint)"""
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

Keep `MagicHint`/`MagicError`/`MultipleCauses` from `bagof.core.magic`; import
everything else from `bagof.dispatchers` directly. The `get_from_registry(hint,
registry) or fallback` call sites (`converters/base.py:308`,
`validators/base.py:299`, `factories/base.py:257`) become `resolve_hint(hint,
registry, default=fallback, ambiguity="warn")`, then `"raise"` once each
registry is clean. Their registration dicts stay the registration API.

### 8.3 Improvements made at the source (formerly "extensions requested of core-magic")

- non-`runtime_checkable` Protocol answers `False` instead of raising;
- constrained TypeVar `≡` union of its constraints;
- `Callable` params contravariant, return covariant;
- `get_from_registry`'s summed distance → specificity order + MRO refinement;
- `Exact` understood by `issubhint`/`ishintstance`;
- `eq_safenan` numpy-free at the root.

---

## 9. Corner-case checklist (each is a test)

**Hints & lattice:** `Any` vs `object` (`object < Any`; unannotated = `Any`) ·
`List[int] < list ≡ List`; `List[int]`+`List[str]` at one position →
`RuntimeWarning` · `Optional`/`Union` ordered by `issubhint`; `None` arg is
`NoneType` · bare `Union`/`Literal`/`Type` never applicable to a value (warn) ·
`Literal` value-dependent; `True`∉`Literal[1]`; NaN via `eq_safenan`; unhashable
→ uncached · `Tuple[X,...]`/`Tuple[X,Y]`/`tuple` chain; items not inspected;
`Tuple[()]` ok · `Callable` parametrisations ordered (post-fix contravariance);
`Callable[P,R] ≡ Callable[...,R]` · `type[X]` value-dependent; `type[bool] <
type[int] < type` · `Annotated` (non-`Exact`) `≡ X`; `Exact` of a non-class →
`TypeError` · TypedDict hint-level fine; value-level v1 warns, v2 shape-checks ·
Protocols: runtime structural, two satisfied → ambiguous unless comparable,
non-runtime → registration `TypeError` · ABCs via `issubclass`; late `register()`
→ cache-token invalidation · diamond `D(B,C)` → `B`; B vs satisfied-ABC →
ambiguous · preorder laws property-tested.

**Modern forms (§11):** `type X = …` alias → its value; two aliases of one value
→ duplicate replace + warning; recursive alias → stops at origin · `L[int]` for
`type L[T] = list[T]` → `List[int]` · `NewType` → supertype · `Tuple[int, *Ts]`
chain; `Ts` twice → applicability only · `Callable[P,R]` chain;
`Concatenate[int,P]` contravariant prefix · `*args: P.args`/`*args: *Ts` → `Any`
tail; `**kwargs` ignored · `Never` param → never applicable · `Required`/
`ReadOnly`/`Final`/`ClassVar` → as inner · unknown/future special form → opaque
≈ `Any` + one `UnknownHintWarning`, never raises · `TypeVar(bound=float,
default=int)` → bound wins, `int` arg does not match (no numeric-tower
promotion) · PEP 695 TypeVar without `__default__` → `getattr(..., NoDefault)` ·
`list[int]` on 3.9/3.10 not mistaken for a class · both-spelling `Unpack` on
3.11 recognised.

**Calls & parameters:** keyword args forwarded, not dispatched (v1 positional) ·
defaults → arity range, shorter fixed arity wins on tightness · `*args: H` tail,
fixed arity beats tail, `**kwargs` ignored · zero-arg call → zero-arity methods ·
methods in classes via `__get__`; `self` = `Any` unless annotated · unhashable
args uncached · lying `__class__` documented (`type(v)` for cache/MRO,
`isinstance` for applicability) · errors never `repr` values.

**Registration & lifecycle:** new registration → cache cleared, order extended,
atomic publish · concurrent registration/call → never torn · same name in two
modules → distinct `Function`s; reload → replacement with `RuntimeWarning` ·
bad registrations → `TypeError` · `priority` ties still ambiguous, never
overrides a strict specificity win · `ambiguities()` lists warned pairs ·
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
- **Phase 1 — Relocate/reimprove the hint relation + core-magic shim.** Three
  commits: (1) pure move of `_compat`/`_sentinels`/`_introspect`/`_relation` with
  core-magic's relation test files moved verbatim and green (reviewable as a diff
  against core-magic); (2) the §8.3 source improvements + `_exact` + all §11.1
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
- **Phase 2 — `_lattice.py`.** `equivalent`, `is_instance`, `mro_index`, TypeVar
  solving, value-dependence classifier; preorder-law property tests over ~40
  hints incl. `Exact` and `Callable` pairs. Review recommended (lighter).
- **Phase 3 — `_signature.py` + `_method.py`.** `from_callable`/`from_hints`,
  arity/tail/kw-only, `NewType`, deferred strings, the 3.14 `annotationlib` path,
  `*args: *Ts`/`P.args` tails; typevar table + consistency; `Method` repr/source.
- **Phase 4 — `_function.py` + `_errors.py` + `_registry.py`.** Value dispatch
  (§2.2), cache + abc token, `register` with replacement + ambiguity warnings,
  both errors with exact text (incl. `!` marker), `Function.from_mapping`,
  `resolve_hint` with exact-key fast path, parity suite ported from core-magic.
  **Review recommended** (maximal set, MRO refinement, refinement ordering).
- **Phase 5 — `_dispatcher.py`, `__init__.py`, docs.** Name grouping,
  `update_wrapper`; two doc pages (dispatch, hints); 3.8-safe `pycon`;
  `test_docstrings.py`, `test_module_surface.py` covering both name groups.
- **Phase 6 — Dependent import migration.** One PR each for
  converters/validators/factories (§8.2), `ambiguity="warn"` → `"raise"`; magic
  swaps `UnionType`.
- **Phase 7 — TypeVar specificity tie-break + `(T,T)` polish.** §3 repeated-group
  rule, constrained same-constraint, solved-`T` messages. **Review recommended**
  (least-precedented rule).
- **Phase 8 (v2).** TypedDict shape matching in `_lattice.is_instance` (+ the
  separate owner decision on `ishintstance`/validators); keyword-to-position
  binding; full `TypeVarTuple`/`ParamSpec` solving & ordering; `Callable` deep
  element check; `DeprecationWarning` `__getattr__` in core-magic.

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

### 11.1 Per-construct handling (native / te-backport / handling / v1 scope)

| Construct | Native | te | Handling · v1 scope |
|---|---|---|---|
| **PEP 695 `def f[T]` / `class C[T]`** | 3.12 | — (syntax) | `__type_params__` holds ordinary `TypeVar`s; dispatch identical to legacy. Use `getattr(tv, "__default__", tx.NoDefault)` (native 3.12 TypeVars lack `has_default`). **Support** |
| **PEP 695 `type X = …` (`TypeAliasType`)** | 3.12 | 4.6+ | `resolve_alias(hint)` in `_introspect`: detect by duck type (`__value__` + `__type_params__`) or either spelling; return `__value__`; substitute args for a subscripted `G[int]` via typing's own `__getitem__`; recursive with cycle guard; called at top of `issubhint`/`ishintstance` and from `normalise_hint` (not `unwrap`). **Support** |
| **PEP 613 `TypeAlias`** | 3.10 | 4.x | the bound value is an ordinary hint; the bare marker falls under the unknown-form rule. **Support (trivial)** |
| **PEP 696 defaults** | 3.13 | 4.4+ | read bound/constraints only via `_typevar_upper`; default ignored. **Support** |
| **PEP 646 `TypeVarTuple`/`Unpack`/`*Ts`** | 3.11 | 4.1+ | degrade: `*args: *Ts` → `Any` tail; `Unpack[Ts]` in `Tuple[...]` → "zero+ `Any`" slot (prefix/suffix split in `_issubargs`); repeated `Ts` not solved. **Degrade v1**, full ordering deferred |
| **PEP 612 `ParamSpec`/`Concatenate`** | 3.10 | 4.x | degrade: `Callable[P,R] ≡ Callable[...,R]`; `Concatenate[int,P]` contravariant prefix; `*args: P.args` → `Any` tail; bare `P` as a param → registration `TypeError`. **Degrade v1** |
| **`NewType`** | 3.5/3.10 | te class 3.8/3.9 | `resolve_newtype` → `__supertype__`, recursive, in `normalise_hint`. **Support** |
| **`Never`/`NoReturn`** | 3.11/3.6 | 4.1+ | bottom type; a `Never` param makes a method never applicable (explicit "forbid this combination"). **Support** |
| **`TypeGuard`/`TypeIs`** | 3.10/3.13 | 4.x/4.10+ | treat as `bool`. **Support** |
| **`LiteralString`** | 3.11 | 4.1+ | treat as `str`. **Support** |
| **`Self`** | 3.11 | 4.0+ | method → owner class via `__get__`; free function → unknown-form rule. **Support/Degrade** |
| **`Required`/`NotRequired`/`ReadOnly`** | 3.11/3.11/3.13 | 4.0+/4.9+ | transparent qualifiers → unwrap to inner. **Support** |
| **`Final`/`ClassVar`** | 3.8 | — | transparent qualifiers → inner. **Support** |
| **`Annotated`/`Doc` (PEP 727)** | 3.9/te | 4.x/4.9+ | transparent except `EXACT`; `Annotated` is a class ≤3.12, not 3.13 (pinned). **Support** |
| **PEP 604 `X \| Y`** | 3.10 | — | `types.UnionType` in `UNION_TYPES`; 3.14 `types.UnionType is typing.Union` (pinned; verify). **Support** |
| **PEP 585 `list[int]`** | 3.9 | — | `isinstance(list[int], type)` is True on 3.9/3.10 → guard with `get_origin(x) is None` before treating as a class; `≡ List[int]`. **Support** |
| **User `Generic[T]`** | 3.8 | — | origin `isinstance`, args covariant regardless of declared variance (documented value-dispatch divergence). **Support** |
| **Unknown / future form** | — | te first | opaque rule (§11.2). **Degrade** |

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

- **3.8** — everything modern from te; no `list[int]`/`X | Y` at runtime, so
  stringified annotations of those raise `TypeError` in `get_type_hints` (catch
  & defer); `NewType` is a te class (verify).
- **3.9** — PEP 585 arrives and `isinstance(list[int], type)` is True → guard
  `issubclassable`/`safe_issubclass` with `get_origin(x) is None`.
- **3.10** — `X | Y`, `ParamSpec`/`Concatenate`/`TypeAlias`/`TypeGuard` native;
  `NewType` becomes a class; both spellings required.
- **3.11** — `Any` becomes a class (pinned); `Never`/`Self`/`LiteralString`/
  `TypeVarTuple`/`Unpack`/`Required` native; `GenericAlias` trap fixed; te still
  owns `Unpack`/`TypeVarTuple`/`TypeVar` — both spellings.
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
