"""Questions method selection and the call cache ask about hints.

Selecting between candidate methods and caching the outcome of a call
both need more than a raw yes-or-no subtype check, so neither talks to
[`issubhint`][] or
[`ishintstance`][bagof.dispatchers.core.ishintstance] directly. Instead
they go through the small, pure functions collected here, sitting one
layer above the subtype relation. For selection there are four
questions: whether two hints accept exactly the same values and so can
be treated as interchangeable ([`equivalent`][]); whether two hints
that are not ordered against each other still share a value, which
makes a pair of methods written with them ambiguous
([`overlaps`][]); where a value's class
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
import itertools
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
from .core._bounds import bounds_of, is_bound, slot_bounds
from .core._compat import UNION_TYPES, is_typeddict_marker, spellings
from .core._exact import is_exact
from .core._hint import hint_arg, is_hint_form
from .core._introspect import (
    _reads_declared_arguments,
    is_typeddict,
    safe_issubclass,
)
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
    _with_arguments,
)

# Every spelling of `Any`.
_ANY_FORMS = spellings("Any")

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


def overlaps(a: tx.Any, b: tx.Any) -> bool:
    """Report whether two unordered hints, one of them bounded from
    below, share a value.

    Two hints that are ordered against each other always share the
    values of the narrower one, which is what the registration-time
    ambiguity check relies on. A lower bound breaks that shortcut: the
    hints `Animal` and `#!python Super[Dog]` are not ordered either way,
    yet a value of class `Dog` belongs to both, so a call with it cannot
    choose between methods written with them. `Dog` and
    `#!python Between[Dog, Animal]` share such a value in the same way,
    and so do `#!python Type[Animal]` and `#!python Type[Super[Dog]]`,
    which share the class `Dog`. This function recognises such a pair.
    Either both hints stand on a value and at least one of them is a
    [`Super`][bagof.dispatchers.Super] or a
    [`Between`][bagof.dispatchers.Between] bound, or both are `Type`
    forms or both `Hint` forms and at least one argument is such a bound.
    Neither may be [`Exact`][bagof.dispatchers.Exact], whose pairs the
    order already decides. Each hint or argument is then read as an
    interval by [`bounds_of`][], and the two share a value when some
    candidate lies in both intervals. The candidates are the ends of the
    two intervals and, on a value or inside `Type`, every class on the
    MRO of a lower end that is a class, since a class shared by both
    intervals lies above both lower ends. There a candidate must also be
    a class, because the interval holds classes. The test is therefore a
    sufficient condition: when it reports an overlap, a value in both
    really exists, which is the direction a warning needs. When the
    bounds involved are classes of a nominal hierarchy, in which no class
    is made a subclass through `register` or `__subclasshook__`, it also
    finds every overlap there is. A `#!python Literal` names no class to
    try, so `#!python Literal[1]` and `#!python Super[int]` are reported
    as not overlapping, although the value `1` belongs to both.

    Two parametrised generics, one of them with a bound among its type
    arguments, overlap when some parametrisation lies below both. Such a
    parametrisation is looked for on the more derived of the two
    origins. Each of its type arguments is replaced by an end of the
    range that the argument names or, when the two origins are the same,
    by the argument that the other hint gives at that position. Each
    candidate is then checked against both hints by the relation itself,
    so `#!python List[Between[Never, numbers.Integral]]` and
    `#!python List[Super[int]]` are reported as overlapping through
    `#!python List[int]`. Hints with no bound among their arguments are
    never reported, which keeps `#!python List[int]` and
    `#!python List[str]` apart.

    A parametrised union on either side overlaps the other hint when one
    of its members does, which covers a parameter written as
    `#!python Optional[Type[Super[Dog]]]`. A `TypeVar` is read as its
    upper bound, the way the relation reads it everywhere else.

    !!! example
        ```pycon
        >>> from typing import Type
        >>> from bagof.dispatchers import Super
        >>> class Animal: pass
        >>> class Dog(Animal): pass
        >>> overlaps(Animal, Super[Dog])
        True
        >>> overlaps(Type[Animal], Type[Super[Dog]])
        True
        >>> overlaps(Type[int], Type[Super[Dog]])
        False
        ```
    """
    a, b = _read(a), _read(b)
    if isinstance(a, tx.TypeVar):
        return overlaps(_typevar_upper(a), b)
    if isinstance(b, tx.TypeVar):
        return overlaps(a, _typevar_upper(b))
    if _is_union(a):
        return any(overlaps(member, b) for member in get_args_uw(a))
    if _is_union(b):
        return any(overlaps(a, member) for member in get_args_uw(b))
    if is_bound(a) or is_bound(b):
        # Two hints on a value, at least one of them a bound: each is read as
        # an interval of classes that a value's class must lie in.
        return _share_a_value(a, b, object)
    if get_origin_uw(a) is type and get_origin_uw(b) is type:
        a, b = _type_arg(a), _type_arg(b)
        return _share_a_value(a, b, object) or _share_an_argument(a, b)
    if is_hint_form(a) and is_hint_form(b):
        a, b = hint_arg(a), hint_arg(b)
        return _share_a_value(a, b, tx.Any) or _share_an_argument(a, b)
    return _share_an_argument(a, b)


def _read(hint: tx.Any) -> tx.Any:
    """Normalise a hint for [`overlaps`][], keeping the markers it carries.

    A plain `Annotated` wrapper is removed, while an `Exact`, `Super` or
    `Between` hint, which is built from `Annotated`, is kept whole.
    """
    hint = normalise_hint(hint)
    if is_bound(hint) or is_exact(hint):
        return hint
    return unwrap(hint, tx.Annotated)


def _is_union(hint: tx.Any) -> bool:
    """Report whether `hint` is a parametrised union, and not a bound or an
    `Exact` built on one.
    """
    if is_bound(hint) or is_exact(hint):
        return False
    return get_origin_uw(hint) in UNION_TYPES and bool(get_args_uw(hint))


def _share_a_value(a: tx.Any, b: tx.Any, top: tx.Any) -> bool:
    """Report whether two intervals, one of them a bound, share a member.

    `a` and `b` are read as intervals by [`bounds_of`][], with `top` as
    the upper end of a `Super[C]`. `top` is `object` when the members are
    classes, whether the classes of values or the classes passed to
    `Type`, and `Any` when the members are hints.
    """
    if not (is_bound(a) or is_bound(b)):
        return False
    if is_exact(a) or is_exact(b):
        return False
    first, second = bounds_of(a, top), bounds_of(b, top)
    candidates = list(first + second)
    if top is object:
        # A class shared by both intervals lies above both lower ends, so
        # in a nominal hierarchy it is on the MRO of each lower end that is
        # a class. Each candidate is still checked against both intervals.
        for lower in (first[0], second[0]):
            if isinstance(lower, type):
                candidates.extend(lower.__mro__)
    for end in candidates:
        if top is object and not isinstance(end, type):
            continue
        if _within(end, first) and _within(end, second):
            return True
    return False


# How many parametrisations `_share_an_argument` tries before it gives up.
_MAX_WITNESSES = 64


def _share_an_argument(a: tx.Any, b: tx.Any) -> bool:
    """Report whether two parametrised generics, at least one of them with
    a bound among its type arguments, share a parametrisation.

    Two parametrisations of one generic that are not ordered can still
    have a parametrisation below both, which a value can declare: a
    `List[int]` is below both
    `#!python List[Between[Never, numbers.Integral]]` and
    `#!python List[Super[int]]`. The same holds when the origin of one
    hint derives from the origin of the other, as `list` derives from
    `MutableSequence`. The parametrisations tried are built on the more
    derived origin, with each type argument replaced by one of the ends
    of the range it names, or by an argument the other hint gives at the
    same position when the origins are the same. Each one is then
    checked against both hints by the relation itself, so an overlap
    reported here is always real. Hints with no bound among their
    arguments are never reported, which keeps `List[int]` and
    `List[str]` apart, as the relation's own order does.
    """
    a, b = _read(a), _read(b)
    if any(is_bound(each) or is_exact(each) for each in (a, b)):
        # A bound or an `Exact` hint names a range of classes or hints, not a
        # parametrisation, and has been read as such already.
        return False
    if not (_holds_argument_bound(a) or _holds_argument_bound(b)):
        return False
    origin_a, origin_b = get_origin_uw(a), get_origin_uw(b)
    if origin_a is origin_b:
        choices = [
            _arguments_to_try(x) + _arguments_to_try(y)
            for x, y in zip(get_args_uw(a), get_args_uw(b))
        ]
        base = a
    elif safe_issubclass(origin_a, origin_b):
        choices = [_arguments_to_try(x) for x in get_args_uw(a)]
        base = a
    elif safe_issubclass(origin_b, origin_a):
        choices = [_arguments_to_try(x) for x in get_args_uw(b)]
        base = b
    else:
        return False
    if not choices:
        # A class that fills in its bases' arguments itself, such as
        # `class IntList(List[int])`, has no arguments of its own to vary.
        return False
    for args in itertools.islice(itertools.product(*choices), _MAX_WITNESSES):
        witness = _with_arguments(base, args)
        if issubhint(witness, a) and issubhint(witness, b):
            return True
    return False


def _holds_argument_bound(hint: tx.Any) -> bool:
    """Report whether a bound stands among the type arguments of the
    parametrised generic `hint`, at any depth.
    """
    if is_bound(hint) or is_exact(hint):
        return False
    origin = get_origin_uw(hint)
    if (
        not isinstance(origin, type)
        or origin is tuple
        or origin is abc.Callable
    ):
        # `Tuple` and `Callable` hold no bound among their own arguments, and
        # their arguments describe a shape rather than one argument per slot.
        return False
    return any(
        is_bound(arg) or _holds_argument_bound(arg)
        for arg in get_args_uw(hint)
    )


def _arguments_to_try(arg: tx.Any) -> tx.List[tx.Any]:
    """List the type arguments to try in place of `arg` when looking for a
    parametrisation that two hints share.

    A bound, `Any` or a `TypeVar` gives the ends of the range it names, a
    constrained `TypeVar` its constraints, a parametrised generic with a
    bound among its own arguments the parametrisations built the same
    way, and any other argument itself.
    """
    arg = normalise_hint(arg)
    variable = unwrap(arg, tx.Annotated)
    constraints = getattr(variable, "__constraints__", ())
    if isinstance(variable, tx.TypeVar) and constraints:
        return list(constraints)
    if is_bound(arg) or _is_open(variable):
        return list(slot_bounds(arg))
    if _holds_argument_bound(arg):
        choices = [_arguments_to_try(each) for each in get_args_uw(arg)]
        return [
            _with_arguments(arg, args)
            for args in itertools.islice(itertools.product(*choices), 8)
        ]
    return [arg]


def _is_open(arg: tx.Any) -> bool:
    """Report whether a type argument is `Any` or a `TypeVar`."""
    return isinstance(arg, tx.TypeVar) or any(
        arg is form for form in _ANY_FORMS
    )


def _type_arg(hint: tx.Any) -> tx.Any:
    """Return the argument of a `Type` form, or `Any` for a bare one."""
    args = get_args_uw(hint)
    return args[0] if args else tx.Any


def _within(end: tx.Any, bounds: tx.Tuple[tx.Any, tx.Any]) -> bool:
    """Report whether `end` lies between a `(lower, upper)` pair of hints."""
    lower, upper = bounds
    return issubhint(lower, end) and issubhint(end, upper)


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

    A [`Hint`][bagof.dispatchers.Hint]`[X]` is value-dependent for the
    same reason: it dispatches on the hint passed as a value rather than
    on that value's Python type, so `#!python int` and `#!python str`,
    both of type `#!python type`, must key the cache separately.

    An [`Exact`][bagof.dispatchers.Exact],
    [`Super`][bagof.dispatchers.Super] or
    [`Between`][bagof.dispatchers.Between] hint on a value is not
    value-dependent, because each of them compares only the value's
    class against a range of classes, so a cache keyed on the type is
    exact for it.

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
    if is_hint_form(origin):
        # A `Hint[X]` matches on the hint passed as a value, not on that
        # value's type, so two hints of the same Python type can match
        # differently. The cache must therefore key on the value itself.
        return True
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
    and keys simply as its type paired with `#!python None`. A generic
    with a bound as a type argument, such as
    `#!python List[Super[int]]`, is declaration-dependent like any other
    parametrised generic, since the bound is compared against the
    recorded argument.

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
        names: tx.Set[str] = set()
        for arg in args:
            names.update(instance_members(arg))
        return tuple(sorted(names))
    if isinstance(hint, tx.TypeVar):
        return instance_members(_typevar_upper(hint))
    # Exactly the hints whose value check reads members off the value (see
    # `_ishintstance_protocol`).
    members = _data_protocol_members(origin)
    return members.data if members is not None else ()
