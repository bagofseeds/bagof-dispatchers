"""The dispatch-internal lattice layer.

The signature order and method selection (later phases) never call the
[`core`][bagof.dispatchers.core] relation directly. They go through the small,
pure helpers here, which sit one step above
[`issubhint`][bagof.dispatchers.core.issubhint] /
[`ishintstance`][bagof.dispatchers.core.ishintstance] and answer the four
questions selection asks of a hint:

* are two hints interchangeable
  ([`equivalent`][bagof.dispatchers._lattice.equivalent])?
* does a value satisfy a hint
  ([`is_instance`][bagof.dispatchers._lattice.is_instance])?
* where does a hint's class sit in a value's MRO, for the tie-break
  ([`mro_index`][bagof.dispatchers._lattice.mro_index])?
* do the arguments bound to one repeated `TypeVar` agree
  ([`solve_typevar`][bagof.dispatchers._lattice.solve_typevar] /
  [`typevar_consistent`][bagof.dispatchers._lattice.typevar_consistent])?

and one the cache asks:

* does a hint's applicability depend on the value, not just its type
  ([`is_value_dependent`][bagof.dispatchers._lattice.is_value_dependent])?

Keeping the engine on this seam means one place decides what "satisfies" and
"same as" mean -- so a later change (the v2 `TypedDict` shape check) lands here
and every phase that builds on it follows.
"""

# dependencies
import typing_extensions as tx

# local
from .core import (
    UNSET,
    get_args_uw,
    get_origin_uw,
    is_typeddict,
    ishintstance,
    issubhint,
    normalise_hint,
    unwrap,
)
from .core._compat import spellings
from .core._exact import exact_target, is_exact

# Every spelling of `Literal`: on 3.8-3.10 `typing_extensions` ships its own,
# a distinct object from `typing`'s, so a single-object `is` check misses one.
_LITERAL_FORMS = spellings("Literal")


def _is_literal(origin: tx.Any) -> bool:
    """Whether `origin` is the `Literal` form, in any spelling."""
    return any(origin is form for form in _LITERAL_FORMS)


# --- equivalence -------------------------------------------------------


def equivalent(a: tx.Any, b: tx.Any) -> bool:
    """Whether two hints accept exactly the same values.

    `a` and `b` are equivalent (`a ≡ b`) when each is a sub-hint of the
    other -- they sit in the same equivalence class of the sub-hint
    preorder, so selection may treat them as interchangeable. This is how
    `#!python list` and `#!python List`, or a bound `TypeVar` and its
    bound, come out equal.

    !!! example
        ```pycon
        >>> from typing import List
        >>> equivalent(list, List)
        True
        >>> equivalent(bool, int)
        False
        ```
    """
    a, b = normalise_hint(a), normalise_hint(b)
    return issubhint(a, b) and issubhint(b, a)


# --- value satisfaction ------------------------------------------------


def is_instance(value: tx.Any, hint: tx.Any) -> bool:
    """Whether `value` satisfies `hint` -- the engine's one value check.

    Every value-level applicability test in the dispatcher goes through
    this function rather than calling
    [`ishintstance`][bagof.dispatchers.core.ishintstance] directly, so the
    meaning of "this value is described by this hint" is decided in one
    place. For v1 it *is*
    [`ishintstance`][bagof.dispatchers.core.ishintstance], which already
    understands [`Exact`][bagof.dispatchers.Exact], `#!python Literal`,
    `#!python Union`, `#!python type[...]`, `#!python TypeVar` and the rest.

    !!! example
        ```pycon
        >>> from bagof.dispatchers import Exact
        >>> is_instance(1, int)
        True
        >>> is_instance(True, Exact[int])   # exactly int, not a subclass
        False
        ```
    """
    # v2 hook (RFC 0001 §9/§10, Phase 8): a `TypedDict` hint should here
    # check the *shape* of a mapping value -- that its keys and their value
    # types match the declaration -- instead of `ishintstance`'s current
    # answer (a plain `dict` is not an instance of a `TypedDict`). That is a
    # deliberate later change, kept out of v1 because it also affects
    # `bagof.validators`; when it lands it lands *here*, so every phase that
    # calls `is_instance` picks it up with no further change.
    return ishintstance(value, hint)


# --- MRO refinement ----------------------------------------------------


def mro_index(hint: tx.Any, value_type: type) -> tx.Optional[int]:
    """Where `hint`'s class sits in `value_type`'s MRO, or `None`.

    This drives the MRO tie-break (RFC 0001 §2.2): when two methods are
    otherwise incomparable, the one whose hint names a *more derived* base
    of the argument's actual type wins -- the diamond `#!python D(B, C)`
    resolves to `B`, exactly as
    [`functools.singledispatch`][functools.singledispatch] does.

    A refinement is only defined when the hint names a single ordinary
    class that is a nominal base of `value_type`:

    * an [`Exact`][bagof.dispatchers.Exact]`[C]` hint counts as `C`;
    * a bare class returns its index in
      `#!python value_type.__mro__`;
    * a class that is *not* in the MRO -- a `#!python Protocol` or ABC
      satisfied structurally or by registration rather than inheritance --
      gives no refinement (`#!python None`);
    * a `#!python Union`, `#!python Literal`, `#!python type[...]` or any
      other parametrised generic gives no refinement either.

    Returns
    -------
    int or None
        The index of the hint's class in `#!python value_type.__mro__`
        (`0` is `value_type` itself), or `#!python None` when no
        refinement applies.

    !!! example
        ```pycon
        >>> class B: pass
        >>> class C: pass
        >>> class D(B, C): pass
        >>> mro_index(B, D) < mro_index(C, D)   # D resolves to B
        True
        ```
    """
    hint = normalise_hint(hint)
    if is_exact(hint):
        cls = normalise_hint(exact_target(hint))
    else:
        cls = unwrap(hint, tx.Annotated)
    # Only a bare class names a position in the MRO. A parametrised generic,
    # union, literal or `type[...]` carries a typing origin; a `TypeVar` or
    # any non-class is not a class at all -- none of them refine.
    if not (isinstance(cls, type) and tx.get_origin(cls) is None):
        return None
    mro = getattr(value_type, "__mro__", ())
    for index, base in enumerate(mro):
        if base is cls:
            return index
    return None


# --- repeated TypeVar solving ------------------------------------------


def solve_typevar(
    classes: tx.Iterable[tx.Any], typevar: tx.Any
) -> tx.Any:
    """Solve one `TypeVar` against the classes bound to its positions.

    A signature may name the same `TypeVar` at several positions
    (`#!python def same(x: T, y: T)`). For a call to apply, the argument
    classes that landed in those positions must agree on a single solution
    (RFC 0001 §3):

    * an **unbound or bound** `TypeVar` follows the *greatest-element*
      rule -- one of the argument classes must be a super-hint of every
      other, and that class is the solution. `#!python (int, bool)` solves
      to `#!python int`; `#!python (int, str)` has no greatest element and
      is unsolvable.
    * a **constrained** `TypeVar` requires every argument class to solve to
      the *same* constraint (a subclass solves as its constraint:
      `#!python bool` solves `#!python TypeVar("T", int, str)` as
      `#!python int`). `#!python (int, str)` picks two different
      constraints and is unsolvable.

    The result is the type the variable stands for -- a class for the
    greatest-element rule, a constraint for the constrained rule -- which
    the caller uses both to decide applicability and, later, as the
    position's hint for specificity. When no positions carry the variable
    the classes are empty and it stands for its full upper bound.

    Parameters
    ----------
    classes
        The argument classes bound to this variable's positions, in any
        order.
    typevar
        The `TypeVar` (its `#!python __bound__` / `#!python __constraints__`
        are read; a PEP 696 default is ignored, as elsewhere in dispatch).

    Returns
    -------
    Any
        The solved type, or [`UNSET`][bagof.dispatchers.core.UNSET] when the
        classes do not agree.

    !!! example
        ```pycon
        >>> from typing import TypeVar
        >>> T = TypeVar("T")
        >>> solve_typevar((int, bool), T)
        <class 'int'>
        >>> typevar_consistent((int, str), T)   # no greatest element
        False
        ```
    """
    classes = tuple(classes)
    constraints = getattr(typevar, "__constraints__", ())
    if constraints:
        return _solve_constrained(classes, constraints)
    return _solve_greatest(classes, typevar)


def _solve_greatest(
    classes: tx.Tuple[tx.Any, ...], typevar: tx.Any
) -> tx.Any:
    """The greatest-element solution for an unbound/bound `TypeVar`."""
    if not classes:
        # No position constrains the variable: it stands for its full upper
        # bound (its bound, or `Any` when it is unbound).
        bound = getattr(typevar, "__bound__", None)
        return bound if bound is not None else tx.Any
    for candidate in classes:
        if all(issubhint(other, candidate) for other in classes):
            return candidate
    return UNSET


def _solve_constrained(
    classes: tx.Tuple[tx.Any, ...],
    constraints: tx.Tuple[tx.Any, ...],
) -> tx.Any:
    """The same-constraint solution for a constrained `TypeVar`."""
    if not classes:
        # No position constrains the variable: it stands for the union of
        # its constraints, the hint a constrained variable is equivalent to.
        return tx.Union[constraints]
    solution = UNSET
    for cls in classes:
        chosen = UNSET
        for constraint in constraints:
            if issubhint(cls, constraint):
                chosen = constraint
                break
        if chosen is UNSET:
            # This class matches no constraint at all.
            return UNSET
        if solution is UNSET:
            solution = chosen
        elif solution is not chosen:
            # Two classes picked different constraints.
            return UNSET
    return solution


def typevar_consistent(
    classes: tx.Iterable[tx.Any], typevar: tx.Any
) -> bool:
    """Whether the classes bound to one `TypeVar` agree on a solution.

    The boolean face of
    [`solve_typevar`][bagof.dispatchers._lattice.solve_typevar]: `True`
    when a consistent solution exists, `False` when it does not. Use it
    where only applicability matters and the solved type is not needed.

    !!! example
        ```pycon
        >>> from typing import TypeVar
        >>> T = TypeVar("T")
        >>> typevar_consistent((int, bool), T)
        True
        >>> typevar_consistent((int, str), T)
        False
        ```
    """
    return solve_typevar(classes, typevar) is not UNSET


# --- value dependence --------------------------------------------------


def is_value_dependent(hint: tx.Any) -> bool:
    """Whether a hint's applicability depends on the value, not its type.

    Most hints are decided by an argument's *type* alone: `#!python int`
    accepts a value iff `#!python type(value)` is `#!python int` or a
    subclass. A few are decided by the *value* itself, so the dispatch
    cache must key on the value there, not only on its type (RFC 0001 §6):

    * `#!python Literal[...]` -- `#!python 1` matches `#!python Literal[1]`
      but `#!python 2` does not, though both are `#!python int`;
    * `#!python type[C]` / `#!python Type[C]` -- one class object matches
      and another does not, though both have type `#!python type`;
    * a `#!python TypedDict` -- once the v2 shape check lands, two dicts of
      the same type match differently by their contents.

    An [`Exact`][bagof.dispatchers.Exact]`[C]` hint is *not*
    value-dependent: it checks `#!python type(value) is C`, which the type
    alone answers.

    !!! example
        ```pycon
        >>> from typing import List, Literal
        >>> is_value_dependent(Literal[1])
        True
        >>> is_value_dependent(List[int])
        False
        ```
    """
    hint = normalise_hint(hint)
    if is_exact(hint):
        # `Exact[C]` is `type(value) is C` -- decided by the type alone.
        return False
    hint = unwrap(hint, tx.Annotated)
    origin = get_origin_uw(hint)
    if _is_literal(origin):
        return True
    if origin is type and get_args_uw(hint):
        # `type[C]`; a bare `type` (no arguments) is decided by the type of
        # the value alone -- whether it is a class -- so it is not here.
        return True
    if is_typeddict(hint):
        return True
    return False
