"""Questions method selection and the call cache ask about hints.

Selecting between candidate methods and caching the outcome of a call
both need more than a raw yes-or-no subtype check, so neither talks to
[`issubhint`][] or
[`ishintstance`][bagof.dispatchers.core.ishintstance] directly. Instead
they go through the small, pure functions collected here, sitting one
layer above the subtype relation. For selection there are three
questions: whether two hints accept exactly the same values and so can
be treated as interchangeable ([`equivalent`][]); where a value's class
falls in its own MRO relative to another hint, used as a tie-break
([`mro_index`][bagof.dispatchers.core.mro_index], defined in `core`
itself so that [`resolve_hint`][bagof.dispatchers.core.resolve_hint] can
reuse it); and whether the several positions where a signature repeats
one `TypeVar` agree on what it should stand for
([`solve_typevar`][] and [`typevar_consistent`][]).

A further three questions belong to the call cache rather than to
selection, since they decide what a cache key needs to capture about an
argument beyond its plain type: whether a hint's applicability turns on
the value itself, as a `Literal` does ([`is_value_dependent`][]);
failing that, whether it turns on a parametrisation the value declared
at construction, as a generic instance does
([`is_declaration_dependent`][]); and whether it turns on which of a
structural protocol's data attributes the value happens to carry
([`instance_members`][]).

Actually checking a value against a hint remains the subtype relation's
job, and the dispatch engine calls
[`ishintstance`][bagof.dispatchers.core.ishintstance] for that directly.
Its handling of `TypedDict` shape, described in RFC 0001 §9 Phase 8,
lives there under the name `_ishintstance_typeddict`; the corresponding
entry in this module, [`is_value_dependent`][], marks a concrete
`TypedDict` as value-dependent so that the cache keys on a mapping's
shape wherever a `TypedDict`-typed parameter is involved.
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

    Two hints `a` and `b` are equivalent when each is a sub-hint of the
    other, meaning every value one accepts is also accepted by the
    other. Equivalent hints occupy the same position in the sub-hint
    ordering, so selection can treat them as the same hint wherever the
    distinction between them would not matter. `#!python list` and
    `#!python List` are equivalent this way, and so are a bound
    `TypeVar` and the type it is bound to.

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
    """Solve one `TypeVar` against the classes found at its positions.

    A signature can name the same `TypeVar` more than once, as in
    `#!python def same(x: T, y: T)`. For a call to such a signature to
    apply, the classes of the arguments landing at those positions have
    to agree on what `T` stands for, following RFC 0001 §3, and this
    function works that agreement out. An unbound or merely bound
    `TypeVar` is solved by the greatest-element rule. One of the argument
    classes must be a super-hint of every other one given, and that
    class is the solution, so `#!python (int, bool)` solves to
    `#!python int` while `#!python (int, str)` has no greatest element
    and cannot be solved. A constrained `TypeVar` is solved differently,
    by requiring every argument class to resolve to the same constraint,
    where a subclass resolves to whichever constraint it is under
    (`#!python bool` resolves to `#!python int` under
    `#!python TypeVar("T", int, str)`); `#!python (int, str)` then
    resolves to two different constraints and is unsolvable there too.

    The value returned is the type the variable stands for once solved:
    a class under the greatest-element rule, or a constraint under the
    constrained rule. A caller uses that value first to decide whether
    the call applies at all, and later as the type this position
    contributes when specificity is compared. When the variable does not
    appear at any position, `classes` comes in empty and the variable is
    taken to stand for its full upper bound.

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
        The argument classes found at this variable's positions, in any
        order.
    typevar
        The `TypeVar` being solved. Its `#!python __bound__` and
        `#!python __constraints__` are read; a PEP 696 default is
        ignored here as it is throughout dispatch.

    Returns
    -------
    Any
        The solved type, or [`UNSET`][] when the given classes do not
        agree on one.
    """
    classes = tuple(classes)
    constraints = getattr(typevar, "__constraints__", ())
    if constraints:
        return _solve_constrained(classes, constraints)
    return _solve_greatest(classes, typevar)


def _solve_greatest(
    classes: tx.Tuple[tx.Any, ...], typevar: tx.Any
) -> tx.Any:
    """Find the greatest-element solution for an unbound or bound `TypeVar`."""
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
    """Find the same-constraint solution for a constrained `TypeVar`."""
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
    """Report whether the classes found at one `TypeVar`'s positions agree.

    This answers the same question as [`solve_typevar`][], reduced to a
    boolean: `#!python True` when the classes have a consistent solution
    and `#!python False` when they do not. It suits a caller that only
    needs to know whether the call applies, without needing the solved
    type itself.

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

    Suppose a parameter's declared `hint` is a
    `#!python Callable[..., R]` whose parameter list ends in a
    [`ParamSpec`][typing.ParamSpec] `P`, and the argument's `query` type
    is itself a matching `#!python Callable[...]`. Then the part of the
    query's parameter list left over once the slot's fixed prefix is
    accounted for belongs to `P`; this is what "captured" means here. A
    signature that names the same
    `ParamSpec` at more than one parameter needs the tail captured at
    each of them to agree, which is the `ParamSpec` counterpart,
    described in RFC 0001 §3, of solving a repeated
    [`TypeVar`][typing.TypeVar].

    Only a `Callable` appearing directly as the parameter's hint is
    considered. A `ParamSpec` buried inside another hint, as in
    `#!python Optional[Callable[P, int]]`, is not solved jointly across
    its occurrences. The captured tail is read off the query's own
    parameter list, so a `ParamSpec` solved out of
    `#!python Callable[Concatenate[int, P], R]` sees the query's
    parameters the same contravariant way that applicability itself
    does.
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
    """Solve one `ParamSpec` against the tails captured at its positions.

    This plays the same role for a `ParamSpec` that [`solve_typevar`][]
    plays for a `TypeVar`. The captured tails must have a greatest
    element under the parameter-list ordering, that is, one tail that
    every other tail is a sub-list of, and that tail becomes the
    solution. `#!python ([int])` and `#!python ([bool])` solve to
    `#!python ([bool])`, while `#!python ([int])` and `#!python ([str])`
    have no greatest element and cannot be solved; between a closed tail
    and an open one, the open tail is always the solution.

    The value returned is the solved tail shape, or [`UNSET`][] when the
    given tails disagree. With no captured tails at all, the variable is
    unconstrained and solves to the widest possible tail, an open list.
    """
    tails = tuple(tails)
    if not tails:
        return _ParamShape((), Ellipsis)
    for candidate in tails:
        if all(_issubparams(other, candidate) for other in tails):
            return candidate
    return UNSET


def paramspec_consistent(tails: tx.Iterable[_ParamShape]) -> bool:
    """Report whether the tails captured at one `ParamSpec`'s positions agree.

    This is the boolean form of [`solve_paramspec`][], described in RFC
    0001 §3: `#!python True` when the captured tails have a greatest
    element and can be solved, `#!python False` when they cannot.
    """
    return solve_paramspec(tails) is not UNSET


# --- repeated TypeVarTuple solving -------------------------------------


def typevartuple_captures(
    query: tx.Any, hint: tx.Any
) -> tx.Iterator[tx.Tuple[tx.Any, _TupleShape]]:
    """Yield `(TypeVarTuple, captured run)` for a top-level `Tuple` slot.

    Suppose a parameter's declared `hint` is a `#!python Tuple[...]`
    whose elements contain an unpacked
    [`TypeVarTuple`][typing.TypeVarTuple] `Ts`, as in
    `#!python Tuple[int, *Ts]`, and the argument's `query` type is a
    `#!python Tuple[...]` of matching shape. Then the run of the query's
    elements left over, once the fixed prefix and suffix are accounted
    for, belongs to `Ts`. A signature naming the same
    `TypeVarTuple` at more than one parameter needs the run captured at
    each of them to agree, which is the covariant, tuple-shaped
    counterpart of the [`ParamSpec`][typing.ParamSpec] rule that
    [`paramspec_captures`][] implements, and of solving a repeated
    [`TypeVar`][typing.TypeVar]; all three are described in RFC 0001 §3.

    Only a `Tuple` appearing directly as the parameter's hint is
    considered. A `TypeVarTuple` buried inside another hint, as in
    `#!python Optional[Tuple[int, *Ts]]`, is not solved jointly across
    its occurrences.
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
    """Solve one `TypeVarTuple` against the runs captured at its positions.

    This plays the same role for a `TypeVarTuple` that
    [`solve_typevar`][] plays for a `TypeVar`. The captured runs must
    have a greatest element under the tuple-shape ordering, that is, one
    run that every other run is a sub-run of, and that run becomes the
    solution. Because element position is covariant here, `#!python
    ((int,))` and `#!python ((bool,))` solve to `#!python ((int,))`.
    `#!python ((int,))` and `#!python ((str,))` have no greatest element
    and cannot be solved, and neither can runs of different length;
    between a closed run and an open one, the open run is always the
    solution.

    The value returned is the solved run shape, or [`UNSET`][] when the
    given runs disagree. With no captured runs at all, the variable is
    unconstrained and stands for a run of zero or more `#!python Any`
    elements.
    """
    shapes = tuple(shapes)
    if not shapes:
        return _TupleShape((), tx.Any, (), None)
    for candidate in shapes:
        if all(_issubtupleshape(other, candidate) for other in shapes):
            return candidate
    return UNSET


def typevartuple_consistent(shapes: tx.Iterable[_TupleShape]) -> bool:
    """Report whether the runs captured at one `TypeVarTuple`'s positions
    agree.

    This is the boolean form of [`solve_typevartuple`][], described in
    RFC 0001 §3: `#!python True` when the captured runs have a greatest
    element and can be solved, `#!python False` when they cannot.
    """
    return solve_typevartuple(shapes) is not UNSET


# --- value dependence --------------------------------------------------


def is_value_dependent(hint: tx.Any) -> bool:
    """Report whether a hint's applicability depends on the value itself.

    Whether an argument matches most hints can be decided from its type
    alone: `#!python int` accepts a value exactly when
    `#!python type(value)` is `#!python int` or one of its subclasses,
    with no need to look at the value further. A handful of hints do
    need to look at the value, and the dispatch cache has to key on the
    value itself rather than only on its type for those, following RFC
    0001 §6. `#!python Literal[...]` is the clearest example:
    `#!python 1` matches `#!python Literal[1]` while `#!python 2` does
    not, even though both share the type `#!python int`.
    `#!python type[C]` and `#!python Type[C]` are another example, since
    one class object matches and another does not despite both having
    type `#!python type`. A `#!python Union` or `#!python TypeVar` is
    value-dependent whenever one of its members, or its upper bound,
    is, so `#!python Optional[Literal["a"]]` is value-dependent and so
    is a `#!python TypeVar` bounded by a `#!python Literal`.

    A concrete `#!python TypedDict` is value-dependent as well, because
    it dispatches on the shape of a mapping, its keys and their value
    types, rather than on the type of the mapping object itself; two
    dicts of the same Python type can therefore match different methods.

    A parametrised user-defined generic such as `#!python Box[int]` is
    not treated as value-dependent, even though two instances of the
    same class can match differently depending on how each was
    constructed. What decides the match there is the parametrisation an
    instance was built with, `#!python Box[int]()` as opposed to
    `#!python Box[str]()`, and not the instance's other contents, so this
    case is covered by the narrower dependence
    [`is_declaration_dependent`][] describes instead. The same holds for
    a standard-library generic such as `#!python Sequence[int]`, which a
    subclass instance declares its parametrisation for in the same way.

    A [`runtime_checkable`][typing.runtime_checkable] protocol with data
    members, such as one declaring `#!python name: str`, is likewise not
    value-dependent, even though two instances of one class can match it
    differently when one has set `#!python name` in `__init__` and the
    other has not. What decides the match there is only which data
    members are present, so this case falls under the narrower
    dependence [`instance_members`][] describes: the value's type
    together with which members it carries, never the value's contents
    directly.

    An [`Exact`][bagof.dispatchers.Exact]`[C]` hint looks as though it
    might depend on the value, but it does not: it checks
    `#!python type(value) is C`, a question the type alone answers. The
    bare `#!python TypedDict` marker is not value-dependent either, since
    it names no fields and so is decided by the value's type alone.

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

    This is the narrower of the two dependences the call cache tracks,
    sitting between plain type-based matching and full value-based
    matching, as described in RFC 0001 §6. A parametrised generic such as
    `#!python Box[int]` or `#!python Sequence[int]` is matched against
    the parametrisation an instance was built with. Constructing
    `#!python Box[int]()` records `#!python Box[int]` on the instance,
    and the same happens for `#!python GL[int]()` where `GL` is written
    against a builtin generic, as in `#!python class GL(list[T])` on
    Python 3.9 and later. Because two instances of the same class can
    therefore match different methods, the cache keys such an argument on
    its type together with that recorded parametrisation rather than on
    the instance itself, so every `#!python Box[int]()` shares a single
    cache entry. A value of any other class, such as a plain
    `#!python list`, never has this recorded parametrisation looked up,
    and keys simply as its type paired with `#!python None`.

    `#!python Type[C]`, a `TypedDict`, `#!python Tuple`, and
    `#!python Callable` each have their own matching logic and are not
    declaration-dependent; the first two are value-dependent instead. A
    `#!python Union` or `#!python TypeVar` is declaration-dependent
    whenever one of its members, or its upper bound, is, mirroring the
    rule [`is_value_dependent`][] follows.

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
    """List the attributes a hint's value check reads directly off a value.

    This is the third dependence the call cache tracks, alongside the
    value itself and the declared parametrisation, as described in RFC
    0001 §6. A [`runtime_checkable`][typing.runtime_checkable] protocol
    with data members, such as one declaring `#!python name: str`, is
    matched structurally: two instances of the same class can match it
    differently, one having set `#!python name` and the other not.
    Because the protocol's methods and its `#!python ClassVar` members
    are already determined by the value's class, only its remaining data
    members depend on the particular instance, and the cache keys such an
    argument on the value's type together with which of those members it
    happens to carry. Every instance of one class carrying the same
    members then shares a single cache entry, and a member the class
    itself declares with a plain annotation is present on every instance
    alike, so it never distinguishes them.

    The names returned are those data members, sorted, or `#!python ()`
    for a hint that reads nothing off the instance at all, whether
    because its members are all methods or class variables, because it
    is not runtime-checkable, or because it is some other kind of hint
    entirely. A `#!python Union` combines the members of every one of its
    arguments, and a `#!python TypeVar` defers to its upper bound,
    mirroring the rule [`is_value_dependent`][] follows.

    Two further limits apply. A constrained `TypeVar` whose constraint is
    such a protocol is solved from the argument's class instead, using
    `#!python issubhint(type(v), c)`, so a value that only belongs to the
    protocol because of what its instance holds does not select that
    constraint. A generic protocol, such as one declaring
    `#!python item: T` and checked as `#!python HasItem[int]`, only
    checks that the members are present and not what they hold, since a
    structurally matched value carries no declared type arguments to
    compare against.

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
