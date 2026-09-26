"""The dispatch-internal lattice layer.

The signature order and method selection (later phases) never call the
[`core`][bagof.dispatchers.core] relation directly. They go through the small,
pure helpers here, which sit one step above
[`issubhint`][bagof.dispatchers.core.issubhint] /
[`ishintstance`][bagof.dispatchers.core.ishintstance] and answer the four
questions selection asks of a hint:

* are two hints interchangeable
  ([`equivalent`][bagof.dispatchers._lattice.equivalent])?
* where does a hint's class sit in a value's MRO, for the tie-break
  ([`mro_index`][bagof.dispatchers._lattice.mro_index])?
* do the arguments bound to one repeated `TypeVar` agree
  ([`solve_typevar`][bagof.dispatchers._lattice.solve_typevar] /
  [`typevar_consistent`][bagof.dispatchers._lattice.typevar_consistent])?

and one the cache asks:

* does a hint's applicability depend on the value, not just its type
  ([`is_value_dependent`][bagof.dispatchers._lattice.is_value_dependent])?

The value check itself stays in the relation: the engine calls
[`ishintstance`][bagof.dispatchers.core.ishintstance] directly. Its
`TypedDict`-shape value check (RFC 0001 §9, Phase 8) lives there under the
name `_ishintstance_typeddict`, and
[`is_value_dependent`][bagof.dispatchers._lattice.is_value_dependent] marks a
concrete `TypedDict` value-dependent to match -- so the call cache keys on the
mapping's shape at a `TypedDict`-typed argument.
"""

# stdlib
from collections import abc

# dependencies
import typing_extensions as tx

# local
from .core import (
    UNSET,
    get_args_uw,
    get_origin_uw,
    issubhint,
    normalise_hint,
    unwrap,
)
from .core._compat import UNION_TYPES, is_typeddict_marker
from .core._exact import exact_target, is_exact
from .core._introspect import _looks_like_class, is_typeddict
from .core._relation import (
    _callable_param_shape,
    _is_literal,
    _is_subscripted_tuple,
    _issubparams,
    _issubtupleshape,
    _match_params,
    _match_tuple,
    _ParamShape,
    _tuple_shape,
    _TupleShape,
    _typevar_upper,
)

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
    * a bare, unparametrised alias counts as its origin class, in whichever
      spelling it was written -- `#!python List` and `#!python list` name
      the same position, as do `#!python Sequence` and
      `#!python collections.abc.Sequence`;
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
    # Only a bare class names a position in the MRO. A `TypeVar` or any
    # non-class does not refine at all.
    if not _looks_like_class(cls):
        # A bare typing alias with no arguments is equivalent to its origin
        # class, whichever spelling it was written in -- `typing.Sequence`
        # and `collections.abc.Sequence` name the same MRO position. A
        # *parametrised* generic (`List[int]`, `type[C]`) carries arguments
        # that constrain more than the class does, so it names no position.
        if get_args_uw(cls):
            return None
        cls = get_origin_uw(cls)
        if not _looks_like_class(cls):
            # A union, literal or bare `Callable` has no plain-class origin.
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
    # mypy's rule: the solution is the first constraint every class is a
    # sub-hint of, so the classes all pick the same one. Testing the whole
    # group against each constraint in turn -- rather than each class against
    # the constraints -- makes the answer independent of the order the
    # constraints and classes are given in.
    for constraint in constraints:
        if all(issubhint(cls, constraint) for cls in classes):
            return constraint
    return UNSET


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


# --- repeated ParamSpec solving ----------------------------------------


def paramspec_captures(
    query: tx.Any, hint: tx.Any
) -> tx.Iterator[tx.Tuple[tx.Any, _ParamShape]]:
    """Yield `(ParamSpec, captured tail)` for a top-level `Callable` slot.

    When a landed slot `hint` is `#!python Callable[..., R]` whose parameter
    list ends in a [`ParamSpec`][typing.ParamSpec] `P`, and the `query` that
    reached it is a `#!python Callable[...]` whose list matches, `P` captures
    the tail of the query's list beyond the slot's committed prefix. A
    signature that names one `ParamSpec` at several slots must capture a
    consistent tail at each (RFC 0001 §3, the `ParamSpec` analogue of repeated
    [`TypeVar`][typing.TypeVar] solving).

    Only a **top-level** `Callable` is read: a `ParamSpec` nested inside
    another hint (`#!python Optional[Callable[P, int]]`) is not jointly solved.
    The captured tail is worked out against the *query's* parameters, so a
    `ParamSpec` solved from `#!python Callable[Concatenate[int, P], R]` sees
    the query's arguments contravariantly, exactly as applicability does.
    """
    query = unwrap(normalise_hint(query), tx.Annotated)
    hint = unwrap(normalise_hint(hint), tx.Annotated)
    if get_origin_uw(query) is not abc.Callable:
        return
    if get_origin_uw(hint) is not abc.Callable:
        return
    query_args = get_args_uw(query)
    hint_args = get_args_uw(hint)
    if not query_args or not hint_args:
        return
    hint_shape = _callable_param_shape(hint, hint_args[0])
    if not isinstance(hint_shape.tail, (tx.ParamSpec, tx.TypeVarTuple)):
        # Only a slot whose list ends in a `ParamSpec` or an unpacked
        # `TypeVarTuple` (`Callable[[int, *Ts], R]`) captures anything; a `...`
        # tail or a fixed list joins no group. Either tail is captured as a
        # `_ParamShape` and solved by `solve_paramspec` -- so a `Ts` shared
        # with a `Tuple` capture is solved per kind, keyed by `id(Ts)` in its
        # own group.
        return
    query_shape = _callable_param_shape(query, query_args[0])
    captured = _match_params(query_shape, hint_shape)
    if captured is None:
        # The lists do not match; applicability has already rejected the call,
        # so there is nothing to capture.
        return
    yield hint_shape.tail, captured


def solve_paramspec(tails: tx.Iterable[_ParamShape]) -> tx.Any:
    """Solve one `ParamSpec` against the tails captured at its slots.

    The `ParamSpec` analogue of
    [`solve_typevar`][bagof.dispatchers._lattice.solve_typevar]: the captured
    tails must have a **greatest element** under the parameter-list order (one
    tail that every other is a sub-list of), and that tail is the solution.
    `#!python ([int])` and `#!python ([bool])` solve to `#!python ([bool])`;
    `#!python ([int])` and `#!python ([str])` have no greatest element and are
    unsolvable; a closed tail and an open tail solve to the open one.

    Returns the solved tail shape, or [`UNSET`][bagof.dispatchers.core.UNSET]
    when the tails do not agree. No slots means the variable is unconstrained,
    which is the widest (open) list.
    """
    tails = tuple(tails)
    if not tails:
        return _ParamShape((), Ellipsis)
    for candidate in tails:
        if all(_issubparams(other, candidate) for other in tails):
            return candidate
    return UNSET


def paramspec_consistent(tails: tx.Iterable[_ParamShape]) -> bool:
    """Whether the tails captured at one `ParamSpec`'s slots agree (§3).

    The boolean face of
    [`solve_paramspec`][bagof.dispatchers._lattice.solve_paramspec]: `True`
    when the captured tails have a greatest element, `False` when they do not.
    """
    return solve_paramspec(tails) is not UNSET


# --- repeated TypeVarTuple solving -------------------------------------


def typevartuple_captures(
    query: tx.Any, hint: tx.Any
) -> tx.Iterator[tx.Tuple[tx.Any, _TupleShape]]:
    """Yield `(TypeVarTuple, captured run)` for a top-level `Tuple` slot.

    When a landed slot `hint` is a `#!python Tuple[...]` whose elements hold an
    unpacked [`TypeVarTuple`][typing.TypeVarTuple] `Ts` (`#!python Tuple[int,
    *Ts]`), and the `query` that reached it is a `#!python Tuple[...]` whose
    shape matches, `Ts` captures the run of the query's elements beyond the
    slot's fixed prefix and suffix. A signature that names one `TypeVarTuple`
    at several slots must capture a consistent run at each -- the covariant
    tuple analogue of the [`ParamSpec`][typing.ParamSpec] rule
    ([`paramspec_captures`][bagof.dispatchers._lattice.paramspec_captures]) and
    of repeated [`TypeVar`][typing.TypeVar] solving (RFC 0001 §3).

    Only a **top-level** `Tuple` is read: a `TypeVarTuple` nested inside
    another hint (`#!python Optional[Tuple[int, *Ts]]`) is not jointly solved.
    """
    query = unwrap(normalise_hint(query), tx.Annotated)
    hint = unwrap(normalise_hint(hint), tx.Annotated)
    if get_origin_uw(query) is not tuple:
        return
    if get_origin_uw(hint) is not tuple:
        return
    hint_args = get_args_uw(hint)
    if not hint_args or not _is_subscripted_tuple(query):
        # A bare `Tuple`/`tuple` query defines no run to capture; the
        # empty-tuple type `Tuple[()]` (empty arguments on 3.11+) does, and
        # captures the empty run.
        return
    query_args = get_args_uw(query)
    hint_shape = _tuple_shape(hint_args)
    if hint_shape.var is None:
        # Only a slot with an unpacked `TypeVarTuple` run captures anything; a
        # closed tuple or a `Tuple[X, ...]` joins no group.
        return
    captured = _match_tuple(_tuple_shape(query_args), hint_shape)
    if captured is None:
        # The shapes do not match; applicability has already rejected the call,
        # so there is nothing to capture.
        return
    yield hint_shape.var, captured


def solve_typevartuple(shapes: tx.Iterable[_TupleShape]) -> tx.Any:
    """Solve one `TypeVarTuple` against the runs captured at its slots.

    The `TypeVarTuple` analogue of
    [`solve_typevar`][bagof.dispatchers._lattice.solve_typevar]: the captured
    runs must have a **greatest element** under the tuple-shape order (one run
    every other is a sub-run of), and that run is the solution. `#!python
    ((int,))` and `#!python ((bool,))` solve to `#!python ((int,))`
    (covariant: `bool` is under `int`); `#!python ((int,))` and `#!python
    ((str,))` have no greatest element and are unsolvable, as do runs of
    different arity; a closed run and an open run solve to the open one.

    Returns the solved run shape, or [`UNSET`][bagof.dispatchers.core.UNSET]
    when the runs do not agree. No slots means the variable is unconstrained,
    which stands for a run of zero-or-more `#!python Any` elements.
    """
    shapes = tuple(shapes)
    if not shapes:
        return _TupleShape((), tx.Any, (), None)
    for candidate in shapes:
        if all(_issubtupleshape(other, candidate) for other in shapes):
            return candidate
    return UNSET


def typevartuple_consistent(shapes: tx.Iterable[_TupleShape]) -> bool:
    """Whether the runs captured at one `TypeVarTuple`'s slots agree (§3).

    The boolean face of
    [`solve_typevartuple`][bagof.dispatchers._lattice.solve_typevartuple]:
    `True` when the captured runs have a greatest element, `False` when they do
    not.
    """
    return solve_typevartuple(shapes) is not UNSET


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
    * a `#!python Union` or a `#!python TypeVar` whose members or upper bound
      include one of the above -- `#!python Optional[Literal["a"]]` keys on
      the value, and so does a `#!python TypeVar` bounded by a
      `#!python Literal`.

    A concrete `#!python TypedDict` is also value-dependent: it dispatches on
    the *shape* of a mapping -- its keys and their value types -- so two
    dicts of the same type can match different methods.

    An [`Exact`][bagof.dispatchers.Exact]`[C]` hint is *not* value-dependent,
    though it might look it: it checks `#!python type(value) is C`, which the
    type alone answers. Nor is the bare `#!python TypedDict` marker, which
    names no fields and so is decided by the value's type alone.

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
    if origin in UNION_TYPES and get_args_uw(hint):
        # A union is value-dependent iff any member is: `Optional[Literal[1]]`
        # and `Union[Type[int], Type[str]]` both key on the value. Nested
        # container arguments (`List[Literal[1]]`, `Tuple[Literal[1]]`) are
        # not descended into: `ishintstance` never inspects a container's
        # items, so the value there does not change the answer.
        return any(is_value_dependent(arg) for arg in get_args_uw(hint))
    if isinstance(hint, tx.TypeVar):
        # A `TypeVar` stands for its upper bound, so it is value-dependent
        # exactly when that bound is: `TypeVar(bound=Literal[1, 2])` and a
        # constrained `TypeVar` over literals both key on the value.
        return is_value_dependent(_typevar_upper(hint))
    if is_typeddict(origin) and not is_typeddict_marker(origin):
        # A concrete `TypedDict` dispatches on the *shape* of the value -- its
        # keys and their value types -- not on the argument's type alone (a
        # plain `dict` at a `TypedDict`-typed argument matches or not by what
        # it holds). So the call cache must carry the value there; an
        # unhashable `dict` value falls through to "uncached" via the shared
        # unhashable-at-value-dependent path. The bare `TypedDict` marker is
        # excluded: it names no fields, so its value-level check is type-only.
        # The origin, not the hint, is read: a parametrised generic
        # `TypedDict` (`GTD[int]`) is a typing alias, not a `TypedDict` class,
        # so it is recognised only through its origin.
        return True
    return False
