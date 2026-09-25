"""The hint-level subtype relation: `issubhint` and `ishintstance`."""

# stdlib
import warnings
from collections import abc

# dependencies
import typing_extensions as tx

# local
from ._compat import (
    UNION_TYPES,
    UnknownHintWarning,
    is_plausible_hint,
    is_typeddict_marker,
    spellings,
)
from ._exact import exact_target, is_exact
from ._introspect import (
    eq_safenan,
    get_args_uw,
    get_origin_uw,
    normalise_hint,
    safe_get_args,
    safe_get_origin,
    safe_issubclass,
    unwrap,
)

# --- known non-class forms ---------------------------------------------

# Bottom types: a value is never one, and only a bottom is a sub-hint of a
# bottom.
_NEVER_FORMS = spellings("Never") + spellings("NoReturn")

# Forms that stand in, for dispatch, for an ordinary class.
_LITERALSTRING_FORMS = spellings("LiteralString")
_TYPEGUARD_FORMS = spellings("TypeGuard") + spellings("TypeIs")

# `Concatenate` shapes a `Callable` parameter list can carry: a fixed,
# contravariant prefix followed by an open `ParamSpec` tail (RFC 11.1).
_CONCATENATE_FORMS = spellings("Concatenate")

# Every spelling of the forms the relation reads by identity. On 3.8-3.10
# `typing_extensions` ships its own `Any` / `Literal`, distinct objects from
# `typing`'s, so a single-object `is` check silently misses the other.
_ANY_FORMS = spellings("Any")
_LITERAL_FORMS = spellings("Literal")


def _is_any(hint: tx.Any) -> bool:
    """Whether `hint` is `Any`, in any spelling."""
    return any(hint is form for form in _ANY_FORMS)


def _is_literal(origin: tx.Any) -> bool:
    """Whether `origin` is the `Literal` form, in any spelling."""
    return any(origin is form for form in _LITERAL_FORMS)


def _is_never(hint: tx.Any) -> bool:
    """Whether `hint` is a bottom type (`Never`/`NoReturn`)."""
    return any(hint is form for form in _NEVER_FORMS)


def _known_form(hint: tx.Any) -> tx.Any:
    """Map a known non-class form to the class it dispatches as.

    `LiteralString` dispatches as [`str`][], and `TypeGuard[...]` /
    `TypeIs[...]` as [`bool`][]. Every other hint is returned unchanged.
    """
    if any(hint is form for form in _LITERALSTRING_FORMS):
        return str
    if any(hint is form for form in _TYPEGUARD_FORMS):
        return bool
    origin = safe_get_origin(hint)
    if any(origin is form for form in _TYPEGUARD_FORMS):
        return bool
    return hint


def _typevar_upper(tv: tx.Any) -> tx.Any:
    """The upper bound of a `TypeVar`, for dispatch.

    Its bound, the union of its constraints, or [`Any`][typing.Any] -- and
    **not** its PEP 696 default, which is a static-checker fallback that
    dispatch ignores (so `#!python TypeVar("T", bound=float, default=int)`
    dispatches as `float`).
    """
    constraints = getattr(tv, "__constraints__", ())
    if constraints:
        return tx.Union[constraints]
    bound = getattr(tv, "__bound__", None)
    if bound is not None:
        return bound
    return tx.Any


def _equivalent(a: tx.Any, b: tx.Any) -> bool:
    """Whether two hints accept exactly the same values."""
    return issubhint(a, b) and issubhint(b, a)


_WARNED_UNKNOWN = set()  # type: set


def _warn_key(hint: tx.Any) -> tx.Any:
    """A stable, hashable key identifying an unknown *form*.

    A future form and every hint built from it (`Unpack[Ts]`, `Unpack[Us]`,
    ...) share one origin, so the origin -- or the form itself when it has
    none -- keys the warning, deduping per form rather than per `repr`.
    """
    origin = safe_get_origin(hint)
    try:
        hash(origin)
    except TypeError:  # pragma: no cover  -- an unhashable origin
        origin = None
    return origin if origin is not None else id(type(hint))


def _warn_unknown(hint: tx.Any) -> None:
    """Warn once per form that `hint` is unrecognised and treated as `Any`."""
    key = _warn_key(hint)
    if key in _WARNED_UNKNOWN:
        return
    _WARNED_UNKNOWN.add(key)
    try:
        shown = repr(hint)
    except Exception:  # pragma: no cover  -- a hint whose repr raises
        shown = object.__repr__(hint)
    warnings.warn(
        f"Type hint {shown} is not recognised; treating it as `Any` for "
        "dispatch.",
        UnknownHintWarning,
        stacklevel=3,
    )


def _not_a_hint_message(obj: tx.Any) -> str:
    """The error text for an object that is not a usable type hint."""
    if isinstance(obj, str):
        # A bare string is a forward reference, which cannot be resolved
        # without the namespace it was written in -- not available here.
        return (
            f"Cannot use the string {obj!r} as a type hint: a forward "
            "reference needs the namespace it was written in to be "
            "resolved, which is not available here. Pass the type itself, "
            "not its name."
        )
    kind = type(obj).__name__
    return (
        f"Expected a type hint, but got {obj!r} (of type {kind}), which is "
        "not a type or a typing construct."
    )


def ishintstance(obj: tx.Any, hint: tx.Any) -> bool:
    """
    Like isinstance, but the second argument can be a type hint.

    * If `hint` is [`type`][] or [`Type[...]`][tx.Type], checks
      that `obj` is a type and that it is valid subclass of the hint argument.
    * If `hint` is a [`Literal`][tx.Literal], checks that `obj` is one of
      its values. The value must match in type as well: `#!python True` is
      not a valid `#!python Literal[1]`, even though `#!python True == 1`.
    * If `hint` is a [`Union`][tx.Union], checks `obj` against each of
      its members.
    * Otherwise, returns `#!python  issubhint(type(obj), hint)`.

    !!! warning
        A container's **item types are not checked**: a value carries its
        type, and a type carries no arguments, so `#!python [1, 2]` is a
        valid `#!python List[str]` as far as this function is concerned.
        (Python itself refuses `#!python isinstance(x, list[int])` for the
        same reason.) Checking the items means iterating them, which is
        the caller's decision to make - `bagof.validators` does it.

    !!! example
        ```pycon
        >>> ishintstance(1, int)
        True
        >>> ishintstance(1, Literal[1, 2])
        True
        >>> ishintstance(bool, Type[int])
        True
        ```
    """
    hint = normalise_hint(hint)
    # `Exact[C]` first, before the `Annotated` metadata is unwrapped: the
    # value's type must be exactly `C`.
    if is_exact(hint):
        return type(obj) is normalise_hint(exact_target(hint))
    hint = _known_form(hint)
    if _is_never(hint):
        # A bottom type describes no value.
        return False
    if _is_any(hint):
        return True
    # Resolve typevars to their bound/constraints (never their default), so
    # a typevar behaves exactly like the hint it stands for.
    hint = unwrap(hint, tx.Annotated)
    while isinstance(hint, tx.TypeVar):
        upper = normalise_hint(_typevar_upper(hint))
        # Exactness can be reached through a bound (`TypeVar(bound=Exact[C])`),
        # so re-check before the `Annotated` wrapper is stripped.
        if is_exact(upper):
            return type(obj) is normalise_hint(exact_target(upper))
        hint = unwrap(upper, tx.Annotated)
    hint = _known_form(hint)
    if _is_any(hint):
        return True
    origin_uw = get_origin_uw(hint)
    if origin_uw is type:
        return _ishintstance_type(obj, hint)
    if _is_literal(origin_uw):
        return _ishintstance_literal(obj, hint)
    if origin_uw in UNION_TYPES:
        args = get_args_uw(hint)
        if args:
            return any(ishintstance(obj, arg) for arg in args)
    if is_typeddict_marker(origin_uw):
        # The bare `TypedDict` marker: a value is one iff its type is a
        # `TypedDict` (`safe_issubclass` reads it structurally). Without
        # this it would fall to the opaque rule and accept everything.
        return safe_issubclass(type(obj), origin_uw)
    if isinstance(origin_uw, type):
        # Only the origin can be checked here: a value carries its type,
        # and a type carries no arguments - `type([1])` is `list`, never
        # `List[int]`. Checking the arguments means looking at the items,
        # which is the caller's business, not an instance check's.
        return safe_issubclass(type(obj), origin_uw)
    return issubhint(type(obj), hint)


def _ishintstance_literal(obj: tx.Any, hint: tx.Any) -> bool:
    """Check that a value is one of a `Literal`'s values."""
    # Both the type and the value must match. Python compares `True == 1`
    # and `1 == 1.0` as equal, but PEP 586 makes literal matching
    # type-aware, so `Literal[1]` must reject `True` and `1.0`.
    # `eq_safenan` keeps a NaN literal comparable with itself.
    return any(
        type(arg) is type(obj) and eq_safenan(arg) == eq_safenan(obj)
        for arg in get_args_uw(hint)
    )


def _ishintstance_type(obj: tx.Any, hint: tx.Any) -> bool:
    """Like isinstance, but the second argument can be a type hint."""
    # Unwrap the hint, do *not* take its origin: the origin of `type[T]`
    # is the bare `type`, whose `get_args` is always empty, which would
    # make every `type[T]` behave like an unparametrised `type`.
    hint_uw = unwrap(hint)
    if safe_get_origin(hint_uw) is not type:
        # Invalid superhint -> error
        raise TypeError(f"Hint {hint} is not a type[]")
    args_uw = tx.get_args(hint_uw)
    if not args_uw:
        # hint is `type` (or `tx.Type`), so any type is valid
        return isinstance(obj, type)
    # hint is `type[T]` (or `tx.Type[T]`), so check obj is a subclass of T
    return isinstance(obj, type) and safe_issubclass(obj, args_uw[0])


def issubhint(hint: tx.Any, superhint: tx.Any) -> bool:
    """
    Check that a hint is a sub-hint for another hint.

    A hint is a valid subhint if all values that are valid for the hint
    are also valid for the superhint.

    Arguments are compared covariantly, so `#!python List[bool]` is a
    subhint of `#!python List[int]`. A hint with no arguments is *not* a
    subhint of one that has them - a bare `#!python list` may hold
    anything, so it cannot stand in for a `#!python List[int]`.

    !!! note
        An **unparametrised** `#!python Union` or `#!python Literal` asks
        a different question: *is this hint one of those?* So
        `#!python issubhint(int, Union)` is `#!python False` (an
        `#!python int` is not a union) even though
        `#!python issubhint(int, Union[int, str])` is `#!python True`.
        This makes them usable as a `#!python BOUND`, and it is why the
        relation is not transitive through a bare `#!python Union`.

    !!! example
        ```pycon
        >>> from typing import List, Union
        >>> issubhint(bool, int)
        True
        >>> issubhint(Union[int, str], Union[int, str, bytes])
        True
        >>> issubhint(List[bool], List[int])
        True
        >>> issubhint(list, List[int])  # a bare list may hold anything
        False
        >>> issubhint(int, str)
        False
        ```
    """
    hint, superhint = normalise_hint(hint), normalise_hint(superhint)

    # `Exact` first, before any `Annotated` metadata is unwrapped. `Exact[C]`
    # is a *leaf* subtype of `C`: an exactly-`C` value is a `C`, so
    # `Exact[C] <= C`, but neither `C` nor any subclass of `C` is exactly-`C`,
    # so nothing ordinary is `<= Exact[C]`. Keeping it a proper leaf is what
    # makes `<=` a preorder (reflexive and transitive) with `Exact` present.
    if is_exact(hint):
        target = normalise_hint(exact_target(hint))
        if is_exact(superhint):
            # `Exact[D] <= Exact[C]` iff `D` and `C` are the same type.
            supertarget = normalise_hint(exact_target(superhint))
            return _equivalent(target, supertarget)
        # `Exact[D] <= P` iff `D <= P` (an exactly-`D` value is a `D`).
        return issubhint(target, superhint)
    if is_exact(superhint):
        target = normalise_hint(exact_target(superhint))
        # The only ordinary hints below `Exact[C]` are `Literal`s whose every
        # value has type exactly `C`: `Literal[1] <= Exact[int]`, but
        # `Literal[True]` (a `bool`) does not.
        if _is_literal(get_origin_uw(hint)):
            args = get_args_uw(hint)
            return bool(args) and all(type(arg) is target for arg in args)
        return False

    hint, superhint = _known_form(hint), _known_form(superhint)

    # Bottom types: only a bottom is a sub-hint of a bottom; a bottom is a
    # sub-hint of everything.
    if _is_never(superhint):
        return _is_never(hint)
    if _is_never(hint):
        return True

    # shortcircuits
    if _is_any(superhint):
        return True

    if hint is superhint:
        return True

    # Unwrap superhint origin
    origin_uw = get_origin_uw(superhint)

    if _is_any(origin_uw):
        return True

    if isinstance(origin_uw, tx.TypeVar):
        return _issubtypevar(hint, superhint)

    if isinstance(hint, tx.TypeVar):
        # Read the typevar's bound/constraints (never its default) so it is
        # checked against the superhint exactly as the hint it stands for.
        # The superhint-is-a-typevar case is already handled above.

        # For constraints, each constraint must be a subhint of the
        # superhint
        constraints = getattr(hint, "__constraints__", ())
        if constraints:
            if origin_uw in UNION_TYPES:
                # Against a union, ask about the union the typevar
                # stands for: each constraint on its own is not a union,
                # but their combination is.
                return issubhint(_typevar_upper(hint), superhint)
            return all(
                issubhint(constraint, superhint)
                for constraint in constraints
            )

        # For bounds, the bound must be a subhint of the superhint
        return issubhint(_typevar_upper(hint), superhint)

    if not _is_literal(origin_uw) and _is_literal(get_origin_uw(hint)):
        # The hint is a Literal and the superhint is not, so the class,
        # union and NoneType branches below cannot see the literal's
        # values - they only ever compare origins. A Literal is a subhint
        # here iff every one of its values is valid for the superhint, the
        # same question ishintstance already answers for a single value.
        # A bare, unparametrised Literal has no values and stays False.
        args = get_args_uw(hint)
        return bool(args) and all(
            ishintstance(arg, superhint) for arg in args
        )

    if origin_uw in UNION_TYPES:
        return _issubunion(hint, superhint)

    if _is_literal(origin_uw):
        return _issubliteral(hint, superhint)

    if origin_uw is type(None):
        return _issubnone(hint, superhint)

    if origin_uw is type:
        return _issubtype(hint, superhint)

    if origin_uw is abc.Callable:
        return _issubcallable(hint, superhint)

    if is_typeddict_marker(origin_uw):
        # The bare `TypedDict` marker as a super-hint: `safe_issubclass`
        # reads it structurally. Without this it would fall to the opaque
        # rule below and accept everything.
        return _issubclasshint(hint, superhint, origin_uw)

    if isinstance(origin_uw, type):
        return _issubclasshint(hint, superhint, origin_uw)

    # A recognised typing construct with no branch of its own -- or a future
    # form -- is opaque: treated as `Any` so a method annotated with it stays
    # reachable, and reported once. An object that is plainly *not* a hint (a
    # value, a plain function) is a caller error, so it raises instead.
    if is_plausible_hint(superhint):
        _warn_unknown(superhint)
        return True
    raise TypeError(_not_a_hint_message(superhint))


def _issubclasshint(hint: tx.Any, superhint: tx.Any, origin: type) -> bool:
    """Check that a hint is a sub-hint for a hint whose origin is a class."""
    # Compare origins, not the hints themselves: a parametrised alias
    # (`List[int]`) and an `Annotated` wrapper are not instances of
    # `type`, so handing either to `safe_issubclass` directly would
    # answer False for every one of them.
    hint_uw = unwrap(hint)
    if not safe_issubclass(get_origin_uw(hint_uw), origin):
        return False

    superargs = safe_get_args(unwrap(superhint))
    if not superargs:
        # An unparametrised superhint constrains nothing further.
        return True

    args = safe_get_args(hint_uw)
    if not args:
        # `list` cannot stand in for `List[int]`: it may hold anything.
        return False

    return _issubargs(args, superargs)


def _issubargs(
    args: tx.Tuple[tx.Any, ...], superargs: tx.Tuple[tx.Any, ...]
) -> bool:
    """Check a hint's arguments against a superhint's, covariantly."""
    # A trailing ellipsis (`Tuple[int, ...]`, `Callable[..., int]`) means
    # "any number of these", so it does not line up positionally.
    if Ellipsis in superargs or Ellipsis in args:
        if Ellipsis not in superargs:
            return False
        if Ellipsis not in args:
            # Every argument must satisfy the repeated one.
            head = tuple(a for a in superargs if a is not Ellipsis)
            return len(head) == 1 and all(
                issubhint(arg, head[0]) for arg in args
            )
        args = tuple(a for a in args if a is not Ellipsis)
        superargs = tuple(a for a in superargs if a is not Ellipsis)

    if len(args) != len(superargs):
        return False

    return all(
        issubhint(arg, superarg) for arg, superarg in zip(args, superargs)
    )


def _issubnone(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for NoneType."""
    none_uw = get_origin_uw(superhint)
    if none_uw is not type(None):
        raise TypeError(f"nonehint {superhint} is not a NoneType")
    origin_uw = get_origin_uw(hint)
    return origin_uw is type(None)


def _issubliteral(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a Literal."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if not _is_literal(safe_get_origin(superhint_uw)):
        # Superhint is not a literal -> error
        raise TypeError(f"Super-hint {superhint} is not a Literal")
    if not _is_literal(safe_get_origin(hint_uw)):
        # Hint is not a Literal, cannot be a subhint
        return False
    # !! We use tx.get_origin instead of safe_get_origin
    # !! to differentiate tx.Literal (origin is None)
    # !! from tx.Literal[()] (origin is tx.Literal)
    if not tx.get_origin(superhint_uw):
        # All literals are subhints of `tx.Literal`
        return True
    if not tx.get_origin(hint_uw):
        # tx.Literal is not a subhint of tx.Literal[...]
        return False
    # Check that all args of hint are in superhint
    args = safe_get_args(hint_uw)
    superargs = safe_get_args(superhint_uw)
    return all(arg in superargs for arg in args)


def _issubtypevar(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a TypeVar."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if not isinstance(superhint_uw, tx.TypeVar):
        # Invalid superhint -> error
        raise TypeError(f"Super-hint {superhint} is not a TypeVar")
    if hint_uw is superhint_uw:
        # Exact match
        return True
    if getattr(superhint_uw, "__constraints__", ()):
        # If constraints, check that hint is a subhint of one of them
        constraints = superhint_uw.__constraints__
        for constraint in constraints:
            if issubhint(hint, constraint):
                return True
        # A constrained typevar is equivalent to the union of its
        # constraints, so a union hint is a subhint when every member is a
        # subhint of some constraint (`Union[C1, C2] <= TypeVar(_, C1, C2)`,
        # the symmetric partner of `TypeVar(...) <= Union[...]`).
        if get_origin_uw(hint) in UNION_TYPES:
            members = get_args_uw(hint)
            if members:
                return all(
                    any(issubhint(member, constraint)
                        for constraint in constraints)
                    for member in members
                )
        # Else, if hint is a TypeVar, check that all its constraints are
        # subhints of one of the superhint's constraints
        if isinstance(hint_uw, tx.TypeVar):
            subconstraints = getattr(hint_uw, "__constraints__", ())
            if not subconstraints:
                return False
            return all(
                any(
                    issubhint(subconstraint, constraint)
                    for constraint in constraints
                )
                for subconstraint in subconstraints
            )
        # Otherwise, constraints do not match
        return False
    elif getattr(superhint_uw, "__bound__", None) is not None:
        # If bound, check that hint is a subhint of the bound
        bound = superhint_uw.__bound__
        if issubhint(hint, bound):
            return True
        # Else, if hint is a TypeVar, check that its bound is a subhint of
        # the superhint's bound
        if isinstance(hint_uw, tx.TypeVar):
            subbound = getattr(hint_uw, "__bound__", None)
            if subbound is None:
                return False
            return issubhint(subbound, bound)
        # Otherwise, bound does not match
        return False
    else:
        # Unconstrained TypeVar -> any hint is a subhint
        return True


def _issubunion(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a Union."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) not in UNION_TYPES:
        # Invalid superhint -> error
        raise TypeError(f"union {superhint} is not a Union type")
    # !! We use tx.get_origin instead of safe_get_origin
    # !! to differentiate tx.Union (origin is None)
    # !! from tx.Union[...] (origin is tx.Union)
    if not tx.get_origin(superhint_uw):
        # Every union - and only a union - is a subhint of the bare
        # `tx.Union`, the same rule `_issubliteral` applies for a bare
        # `Literal`. A *parametrised* union is different: a plain type
        # is a subhint of one that contains it, which the member logic
        # below works out.
        return safe_get_origin(hint_uw) in UNION_TYPES
    # Collect the hint's member hints. A hint is a subhint of the union if
    # each of its members is a subhint of one of the union's members:
    #   * a parametrised union contributes its arguments;
    #   * the bare `tx.Union` is not a subhint of a parametrised union;
    #   * any other hint (e.g. `int`) is a single member, so that a plain
    #     type is a subhint of a union that contains it.
    if safe_get_origin(hint_uw) in UNION_TYPES:
        if not tx.get_origin(hint_uw):
            return False
        members = safe_get_args(hint_uw)
    else:
        members = (hint_uw,)
    superargs = safe_get_args(superhint_uw)
    return all(
        any(issubhint(member, superarg) for superarg in superargs)
        for member in members
    )


def _issubtype(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a type[...] hint."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) is not type:
        # Invalid superhint -> error
        raise TypeError(f"superhint {superhint} is not a type")
    if safe_get_origin(hint_uw) is not type:
        # Hint is not a type, cannot be a subhint
        return False
    if not tx.get_args(superhint_uw):
        # All types are subhints of `tx.Type`
        return True
    if not tx.get_args(hint_uw):
        # tx.Type is not a subhint of tx.Type[...]
        return False
    # Check that the hint's arg is a subclass of the superhint's arg
    args = safe_get_args(hint_uw)
    superargs = safe_get_args(superhint_uw)
    return safe_issubclass(args[0], superargs[0])


def _issubcallable(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a `Callable[...]`.

    Parameters are compared contravariantly and the return type covariantly.
    A bare `ParamSpec` parameter list is a wildcard `#!python ...` on either
    side; a `#!python Concatenate[X, P]` list is a contravariant fixed prefix
    followed by an open tail (RFC 11.1). A callable *class* -- a function
    type, `#!python type`, `#!python Type[C]`, or a class with `__call__` --
    has no parameter list, so it stands in only for an unparametrised
    `Callable`.
    """
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    superargs = safe_get_args(superhint_uw)
    hint_origin = get_origin_uw(hint_uw)
    if hint_origin is not abc.Callable:
        # A callable *class* has no parameter list: it can only stand in for a
        # bare `Callable`, exactly as `list` cannot stand in for `List[int]`.
        if not safe_issubclass(hint_origin, abc.Callable):
            return False
        return not superargs
    if not superargs:
        # A bare `Callable` constrains nothing further.
        return True
    args = safe_get_args(hint_uw)
    if not args:
        # A bare `callable` may accept anything: it cannot stand in for a
        # parametrised `Callable[...]`.
        return False
    # `(params, return)`: `params` is a list, `...`, or a ParamSpec form.
    sub_params, sub_ret = args[0], args[-1]
    sup_params, sup_ret = superargs[0], superargs[-1]
    # Return type is covariant.
    if not issubhint(sub_ret, sup_ret):
        return False
    return _issubcallable_params(
        hint_uw, superhint_uw, sub_params, sup_params
    )


def _callable_params(
    alias: tx.Any, params: tx.Any
) -> tx.Tuple[str, tx.Any]:
    """Classify a `Callable` parameter list (RFC 11.1).

    Returns one of `#!python ("any", None)` for a wildcard (`#!python ...`, a
    bare `ParamSpec`, or an unknown shape), `#!python ("prefix", [X, ...])`
    for a `Concatenate` fixed prefix followed by an open tail, or
    `#!python ("list", [...])` for a fixed parameter list. The `alias` is the
    whole `Callable[...]` hint, needed to reach `__parameters__` where an
    older Python has erased the `ParamSpec` out of the arguments.
    """
    if params is Ellipsis or isinstance(
        params, (tx.ParamSpec, tx.ParamSpecArgs, tx.ParamSpecKwargs)
    ):
        return "any", None
    if any(safe_get_origin(params) is form for form in _CONCATENATE_FORMS):
        # `Concatenate[X1, ..., Xn, P]`: the fixed prefix is everything but
        # the trailing `ParamSpec`.
        return "prefix", list(tx.get_args(params)[:-1])
    seq = list(params) if isinstance(params, (list, tuple)) else None
    if seq is None:
        # An unknown shape degrades to a wildcard rather than raising.
        return "any", None
    parameters = getattr(alias, "__parameters__", ())
    has_ps = any(isinstance(p, tx.ParamSpec) for p in parameters)
    if seq and isinstance(seq[-1], tx.ParamSpec):
        # Below 3.10 `Concatenate[X, P]` is flattened to `[X, ..., P]`.
        return "prefix", seq[:-1]
    if not seq and has_ps:
        # Below 3.10 a bare `P` is erased to `[]`, surviving only in
        # `__parameters__` -- so an empty list plus a `ParamSpec` there is a
        # wildcard, not a genuine zero-argument list.
        return "any", None
    return "list", seq


def _issubcallable_params(
    sub_alias: tx.Any, sup_alias: tx.Any, sub: tx.Any, sup: tx.Any
) -> bool:
    """Compare two `Callable` parameter lists, contravariantly (RFC 11.1)."""
    sub_kind, sub_val = _callable_params(sub_alias, sub)
    sup_kind, sup_val = _callable_params(sup_alias, sup)
    # A wildcard on either side accepts any parameter list.
    if sub_kind == "any" or sup_kind == "any":
        return True
    sub_open = sub_kind == "prefix"
    sup_open = sup_kind == "prefix"
    # A closed (fixed-arity) sub cannot stand in for an open super, which may
    # be called with arbitrarily many arguments the sub does not accept.
    if sup_open and not sub_open:
        return False
    # A fixed list must be at least as long as an open sub's committed prefix.
    if sub_open and not sup_open and len(sup_val) < len(sub_val):
        return False
    # Two fixed lists must have equal arity.
    if not sub_open and not sup_open and len(sub_val) != len(sup_val):
        return False
    # Two open prefixes: the sub's prefix cannot be the longer, or it would
    # demand arguments the super need not supply.
    if sub_open and sup_open and len(sub_val) > len(sup_val):
        return False
    # Compare the overlapping fixed positions contravariantly: each
    # super-parameter must be a sub-hint of the matching sub-parameter (a
    # function taking `int` can stand in for one taking `bool`).
    overlap = min(len(sub_val), len(sup_val))
    return all(issubhint(sup_val[i], sub_val[i]) for i in range(overlap))
