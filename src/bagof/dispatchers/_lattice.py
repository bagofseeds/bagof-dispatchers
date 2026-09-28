"""The dispatch-internal lattice layer.

Signature ordering and method selection never call the
[`issubhint`][] and [`ishintstance`][bagof.dispatchers.core.ishintstance]
relation directly. Instead, they go through the small, pure helpers
defined here, which sit one step above that relation and answer the
four questions that selection asks of a hint: whether two hints are
interchangeable ([`equivalent`][]); where a hint's class sits in a
value's MRO, for the tie-break
([`mro_index`][bagof.dispatchers.core.mro_index], which lives in `core`
itself so that [`resolve_hint`][bagof.dispatchers.core.resolve_hint]
can share the same logic); and whether the arguments bound to one
repeated `TypeVar` agree with each other ([`solve_typevar`][] and
[`typevar_consistent`][]).

Three further questions belong to the dispatch cache rather than to
selection: whether a hint's applicability depends on the value itself
and not merely on its type ([`is_value_dependent`][]); short of that,
whether it depends on the parametrisation a value declares
([`is_declaration_dependent`][]); and whether it depends on which of a
protocol's data members the value happens to have
([`instance_members`][]).

The value check itself stays in the relation, and the dispatch engine
calls [`ishintstance`][bagof.dispatchers.core.ishintstance] directly
for it. Its `TypedDict`-shape value check, described in RFC 0001 §9
Phase 8, lives there under the name `_ishintstance_typeddict`, and
[`is_value_dependent`][] marks a concrete `TypedDict` as value-dependent
to match, so that the call cache keys on the mapping's shape wherever a
`TypedDict`-typed argument appears.
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
from .core._exact import is_exact
from .core._introspect import _reads_declared_arguments, is_typeddict
from .core._relation import (
    _callable_param_shape,
    _data_protocol_members,
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
    """Report whether two hints accept exactly the same values.

    `a` and `b` are equivalent, written `a ≡ b`, when each is a
    sub-hint of the other. They then sit in the same equivalence class
    of the sub-hint preorder, so selection may treat them as
    interchangeable. This is how `#!python list` and `#!python List`,
    or a bound `TypeVar` and its bound, come out equal.

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


# `mro_index` now lives in `core` (`bagof.dispatchers.core.mro_index`) so that
# `resolve_hint` can share the same MRO tie-break the value-dispatch path uses
# (RFC 0001 §2.2 / §8.1). The engine imports it from `core` directly.


# --- repeated TypeVar solving ------------------------------------------


def solve_typevar(
    classes: tx.Iterable[tx.Any], typevar: tx.Any
) -> tx.Any:
    """Solve one `TypeVar` against the classes bound to its positions.

    A signature may name the same `TypeVar` at several positions, as in
    `#!python def same(x: T, y: T)`. For a call to apply, the argument
    classes landing in those positions must agree on a single solution,
    following RFC 0001 §3. An unbound or bound `TypeVar` follows the
    greatest-element rule: one of the argument classes must be a
    super-hint of every other, and that class is the solution, so
    `#!python (int, bool)` solves to `#!python int` while
    `#!python (int, str)` has no greatest element and is unsolvable. A
    constrained `TypeVar` instead requires every argument class to
    solve to the same constraint, where a subclass solves as its
    constraint (`#!python bool` solves `#!python TypeVar("T", int,
    str)` as `#!python int`), so `#!python (int, str)` picks two
    different constraints and is unsolvable.

    The result is the type the variable stands for: a class under the
    greatest-element rule, or a constraint under the constrained rule.
    The caller uses it both to decide whether the call applies and,
    later, as the position's hint when comparing specificity. When no
    position carries the variable, `classes` is empty and the variable
    stands for its full upper bound.

    !!! example
        ```pycon
        >>> from typing import TypeVar
        >>> T = TypeVar("T")
        >>> solve_typevar((int, bool), T)
        <class 'int'>
        >>> typevar_consistent((int, str), T)   # no greatest element
        False
        ```

    Parameters
    ----------
    classes
        The argument classes bound to this variable's positions, in any
        order.
    typevar
        The `TypeVar` whose `#!python __bound__` and
        `#!python __constraints__` are read. A PEP 696 default is
        ignored, as it is everywhere else in dispatch.

    Returns
    -------
    Any
        The solved type, or [`UNSET`][]
        when the classes do not agree.
    """
    classes = tuple(classes)
    constraints = getattr(typevar, "__constraints__", ())
    if constraints:
        return _solve_constrained(classes, constraints)
    return _solve_greatest(classes, typevar)


def _solve_greatest(
    classes: tx.Tuple[tx.Any, ...], typevar: tx.Any
) -> tx.Any:
    """Return the greatest-element solution for an unbound or bound
    `TypeVar`.
    """
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
    """Return the same-constraint solution for a constrained `TypeVar`."""
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
    """Report whether the classes bound to one `TypeVar` agree on a solution.

    This is the boolean face of [`solve_typevar`][]: it is `#!python True`
    when a consistent solution exists and `#!python False` when it does
    not. Use it where only applicability matters and the solved type
    itself is not needed.

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

    Suppose a landed slot `hint` is `#!python Callable[..., R]` whose
    parameter list ends in a [`ParamSpec`][typing.ParamSpec] `P`, and
    the `query` that reached it is a `#!python Callable[...]` whose
    list matches. Then `P` captures the tail of the query's list beyond
    the slot's committed prefix. A signature that names one `ParamSpec`
    at several slots must capture a consistent tail at each of them;
    this is the `ParamSpec` analogue, described in RFC 0001 §3, of
    repeated [`TypeVar`][typing.TypeVar] solving.

    Only a top-level `Callable` is read here: a `ParamSpec` nested
    inside another hint, as in `#!python Optional[Callable[P, int]]`,
    is not jointly solved. The captured tail is worked out against the
    query's parameters, so a `ParamSpec` solved from
    `#!python Callable[Concatenate[int, P], R]` sees the query's
    arguments contravariantly, exactly as applicability itself does.
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

    This is the `ParamSpec` analogue of [`solve_typevar`][]. The
    captured tails must have a greatest element under the
    parameter-list order, meaning one tail that every other is a
    sub-list of, and that tail is the solution. `#!python ([int])` and
    `#!python ([bool])` solve to `#!python ([bool])`; `#!python ([int])`
    and `#!python ([str])` have no greatest element and are unsolvable;
    a closed tail and an open tail solve to the open one.

    The result is the solved tail shape, or [`UNSET`][] when the tails
    do not agree. No slots at all means the variable is unconstrained,
    which is the widest, open list.
    """
    tails = tuple(tails)
    if not tails:
        return _ParamShape((), Ellipsis)
    for candidate in tails:
        if all(_issubparams(other, candidate) for other in tails):
            return candidate
    return UNSET


def paramspec_consistent(tails: tx.Iterable[_ParamShape]) -> bool:
    """Report whether the tails captured at one `ParamSpec`'s slots agree.

    This is the boolean face of [`solve_paramspec`][], described in RFC
    0001 §3: `#!python True` when the captured tails have a greatest
    element, `#!python False` when they do not.
    """
    return solve_paramspec(tails) is not UNSET


# --- repeated TypeVarTuple solving -------------------------------------


def typevartuple_captures(
    query: tx.Any, hint: tx.Any
) -> tx.Iterator[tx.Tuple[tx.Any, _TupleShape]]:
    """Yield `(TypeVarTuple, captured run)` for a top-level `Tuple` slot.

    Suppose a landed slot `hint` is a `#!python Tuple[...]` whose
    elements hold an unpacked [`TypeVarTuple`][typing.TypeVarTuple]
    `Ts`, as in `#!python Tuple[int, *Ts]`, and the `query` that reached
    it is a `#!python Tuple[...]` whose shape matches. Then `Ts`
    captures the run of the query's elements beyond the slot's fixed
    prefix and suffix. A signature that names one `TypeVarTuple` at
    several slots must capture a consistent run at each of them: the
    covariant tuple analogue of the [`ParamSpec`][typing.ParamSpec] rule
    that [`paramspec_captures`][] implements, and of repeated
    [`TypeVar`][typing.TypeVar] solving, both described in RFC 0001 §3.

    Only a top-level `Tuple` is read here: a `TypeVarTuple` nested
    inside another hint, as in `#!python Optional[Tuple[int, *Ts]]`, is
    not jointly solved.
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

    This is the `TypeVarTuple` analogue of [`solve_typevar`][]. The
    captured runs must have a greatest element under the tuple-shape
    order, meaning one run that every other is a sub-run of, and that
    run is the solution. `#!python ((int,))` and `#!python ((bool,))`
    solve to `#!python ((int,))`, since the relation is covariant and
    `bool` is under `int`. `#!python ((int,))` and `#!python ((str,))`
    have no greatest element and are unsolvable, as are runs of
    different arity; a closed run and an open run solve to the open
    one.

    The result is the solved run shape, or [`UNSET`][] when the runs do
    not agree. No slots at all means the variable is unconstrained,
    which stands for a run of zero or more `#!python Any` elements.
    """
    shapes = tuple(shapes)
    if not shapes:
        return _TupleShape((), tx.Any, (), None)
    for candidate in shapes:
        if all(_issubtupleshape(other, candidate) for other in shapes):
            return candidate
    return UNSET


def typevartuple_consistent(shapes: tx.Iterable[_TupleShape]) -> bool:
    """Report whether the runs captured at one `TypeVarTuple`'s slots agree.

    This is the boolean face of [`solve_typevartuple`][], described in
    RFC 0001 §3: `#!python True` when the captured runs have a greatest
    element, `#!python False` when they do not.
    """
    return solve_typevartuple(shapes) is not UNSET


# --- value dependence --------------------------------------------------


def is_value_dependent(hint: tx.Any) -> bool:
    """Report whether a hint's applicability depends on the value itself.

    Most hints are decided by an argument's type alone: `#!python int`
    accepts a value exactly when `#!python type(value)` is
    `#!python int` or a subclass of it. A few hints are decided by the
    value itself, so the dispatch cache must key on the value in those
    cases rather than only on its type, following RFC 0001 §6. A
    `#!python Literal[...]` is one such case: `#!python 1` matches
    `#!python Literal[1]` but `#!python 2` does not, even though both
    are `#!python int`. `#!python type[C]` and `#!python Type[C]` are
    another: one class object matches and another does not, even though
    both have type `#!python type`. A `#!python Union` or a `#!python
    TypeVar` whose members or upper bound include one of these is
    value-dependent too, so `#!python Optional[Literal["a"]]` keys on
    the value, and so does a `#!python TypeVar` bounded by a
    `#!python Literal`.

    A concrete `#!python TypedDict` is also value-dependent: it
    dispatches on the shape of a mapping, meaning its keys and their
    value types, so two dicts of the same type can match different
    methods.

    A parametrised user generic, such as `#!python Box[int]`, is not
    value-dependent, even though two instances of one class can match
    it differently. What decides the match is the parametrisation each
    instance was built from, `#!python Box[int]()` versus
    `#!python Box[str]()`, not the instance itself, so this case falls
    under the narrower key that [`is_declaration_dependent`][]
    describes. The same is true of a standard-library generic such as
    `#!python Sequence[int]`, which an instance of a generic subclass of
    `#!python Sequence` declares the same way.

    A [`runtime_checkable`][typing.runtime_checkable] protocol with
    data members, such as one declaring `#!python name: str`, is not
    value-dependent either, even though two instances of one class can
    match it differently, one having set `#!python name` in
    `__init__` and one not. What decides the match is only which of
    its data members the value has, so this case falls under the
    narrower key that [`instance_members`][] describes: the value's
    type together with those members' presence, never the value
    itself.

    An [`Exact`][bagof.dispatchers.Exact]`[C]` hint is not
    value-dependent, though it might look as if it were: it checks
    `#!python type(value) is C`, a question the type alone answers. Nor
    is the bare `#!python TypedDict` marker, which names no fields and
    so is decided by the value's type alone.

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


def is_declaration_dependent(hint: tx.Any) -> bool:
    """Report whether a hint's applicability depends on a declaration.

    This is the narrower of the two dependences the call cache keys on,
    sitting between "the type" and "the value", as described in RFC
    0001 §6. A parametrised generic, such as `#!python Box[int]` or
    `#!python Sequence[int]`, is matched against the parametrisation an
    instance was built from: `#!python Box[int]()` records
    `#!python Box[int]` on itself, and so does `#!python GL[int]()` for
    a class written against a builtin generic, such as
    `#!python class GL(list[T])` on Python 3.9 and later. Two instances
    of one class can therefore match different methods, so the cache
    keys such an argument on its type together with that record, rather
    than on the instance, and every `#!python Box[int]()` shares one
    cache entry. A value of any other class, such as a plain
    `#!python list`, is never asked, and keys as its type paired with
    `#!python None`.

    `#!python Type[C]`, a `TypedDict`, `#!python Tuple`, and
    `#!python Callable` keep their own checks and are not declaration-
    dependent; the first two are value-dependent instead. A
    `#!python Union` or `#!python TypeVar` is declaration-dependent when
    a member or its upper bound is, matching the rule
    [`is_value_dependent`][] follows.

    !!! example
        ```pycon
        >>> from typing import Generic, List, Tuple, TypeVar
        >>> T = TypeVar("T")
        >>> class Box(Generic[T]): pass
        >>> is_declaration_dependent(Box[int])
        True
        >>> is_declaration_dependent(List[int])
        True
        >>> is_declaration_dependent(Tuple[int])
        False
        ```
    """
    hint = normalise_hint(hint)
    if is_exact(hint):
        # `Exact[C]` is `type(value) is C` -- decided by the type alone.
        return False
    hint = unwrap(hint, tx.Annotated)
    args = get_args_uw(hint)
    origin = get_origin_uw(hint)
    if origin in UNION_TYPES and args:
        return any(is_declaration_dependent(arg) for arg in args)
    if isinstance(hint, tx.TypeVar):
        return is_declaration_dependent(_typevar_upper(hint))
    # Exactly the hints whose value check reads `__orig_class__` (see
    # `_declared_parametrisation`).
    return bool(args) and _reads_declared_arguments(origin)


def instance_members(hint: tx.Any) -> tx.Tuple[str, ...]:
    """Return the attributes a hint's value check reads off the value itself.

    This is the third dependence the call cache keys on, beside the
    value and the declared parametrisation, as described in RFC 0001
    §6. A [`runtime_checkable`][typing.runtime_checkable] protocol with
    data members, such as one declaring `#!python name: str`, is
    matched by what the value holds, so two instances of one class can
    match it differently: one that set `#!python name` and one that did
    not. Its methods and its `#!python ClassVar` members are read off
    the value's class, so only its other data members depend on the
    instance, and the cache keys such an argument on the value's type
    together with which of these members the value has. Every instance
    of one class holding the same members then shares one cache entry.
    A member the class declares with a plain annotation is present on
    every instance, so it keys them all alike.

    The return value is those data members' names, sorted, or
    `#!python ()` for a hint that reads nothing off the instance: a
    protocol whose members are all methods or class variables, one that
    is not runtime-checkable, or any other hint. A `#!python Union`
    gathers the members of all its arguments, and a `#!python TypeVar`
    reads its upper bound, matching the rule
    [`is_value_dependent`][] follows.

    Two limits apply to what is read off the instance. A constrained
    `TypeVar` whose constraint is such a protocol is solved from the
    argument's class, `#!python issubhint(type(v), c)`, so a value that
    belongs to the protocol only by what its instance holds does not
    select that constraint. And a generic protocol, such as one
    declaring `#!python item: T` and checked as
    `#!python HasItem[int]`, only checks that the members are present,
    not what they hold, since a structural value declares no type
    arguments to compare against.

    !!! example
        ```pycon
        >>> import typing_extensions as tx
        >>> @tx.runtime_checkable
        ... class Named(tx.Protocol):
        ...     name: str
        >>> instance_members(Named)
        ('name',)
        >>> instance_members(tx.Optional[Named])
        ('name',)
        >>> instance_members(int)
        ()
        ```
    """
    hint = normalise_hint(hint)
    if is_exact(hint):
        # `Exact[C]` is `type(value) is C` -- decided by the type alone.
        return ()
    hint = unwrap(hint, tx.Annotated)
    args = get_args_uw(hint)
    origin = get_origin_uw(hint)
    if origin in UNION_TYPES and args:
        names = set()  # type: tx.Set[str]
        for arg in args:
            names.update(instance_members(arg))
        return tuple(sorted(names))
    if isinstance(hint, tx.TypeVar):
        return instance_members(_typevar_upper(hint))
    # Exactly the hints whose value check reads members off the value (see
    # `_ishintstance_protocol`).
    members = _data_protocol_members(origin)
    return members.data if members is not None else ()
