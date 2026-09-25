"""The hint-level subtype relation: `issubhint` and `ishintstance`."""

# stdlib
import warnings
from collections import abc

# dependencies
import typing_extensions as tx

# local
from ._compat import UNION_TYPES, UnknownHintWarning, spellings
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

# `ParamSpec` shapes a `Callable` parameter list can carry that this
# version handles only by degrading -- comparable to itself, nothing else.
_CONCATENATE_FORMS = spellings("Concatenate")


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


def _warn_unknown(hint: tx.Any) -> None:
    """Warn once that `hint` is unrecognised and treated as `Any`."""
    try:
        key = repr(hint)
    except Exception:  # pragma: no cover  -- a hint whose repr raises
        key = object.__repr__(hint)
    if key in _WARNED_UNKNOWN:
        return
    _WARNED_UNKNOWN.add(key)
    warnings.warn(
        f"Type hint {key} is not recognised; treating it as `Any` for "
        "dispatch.",
        UnknownHintWarning,
        stacklevel=3,
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
    if hint is tx.Any:
        return True
    # Resolve typevars to their bound/constraints (never their default), so
    # a typevar behaves exactly like the hint it stands for.
    hint = unwrap(hint, tx.Annotated)
    while isinstance(hint, tx.TypeVar):
        hint = unwrap(_typevar_upper(hint), tx.Annotated)
    hint = _known_form(hint)
    if hint is tx.Any:
        return True
    origin_uw = get_origin_uw(hint)
    if origin_uw is type:
        return _ishintstance_type(obj, hint)
    if origin_uw is tx.Literal:
        return _ishintstance_literal(obj, hint)
    if origin_uw in UNION_TYPES:
        args = get_args_uw(hint)
        if args:
            return any(ishintstance(obj, arg) for arg in args)
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

    # `Exact` first, before any `Annotated` metadata is unwrapped.
    if is_exact(superhint):
        target = normalise_hint(exact_target(superhint))
        if is_exact(hint):
            # `Exact[D] <= Exact[C]` iff `D` and `C` are the same type.
            return _equivalent(normalise_hint(exact_target(hint)), target)
        # `q <= Exact[C]` iff `q` is equivalent to `C`.
        return _equivalent(hint, target)
    if is_exact(hint):
        # `Exact[D] <= P` iff `D <= P` (`Exact[D]` is strictly below `D`).
        return issubhint(normalise_hint(exact_target(hint)), superhint)

    hint, superhint = _known_form(hint), _known_form(superhint)

    # Bottom types: only a bottom is a sub-hint of a bottom; a bottom is a
    # sub-hint of everything.
    if _is_never(superhint):
        return _is_never(hint)
    if _is_never(hint):
        return True

    # shortcircuits
    if superhint is tx.Any:
        return True

    if hint is superhint:
        return True

    # Unwrap superhint origin
    origin_uw = get_origin_uw(superhint)

    if origin_uw is tx.Any:
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

    if origin_uw is not tx.Literal and get_origin_uw(hint) is tx.Literal:
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

    if origin_uw is tx.Literal:
        return _issubliteral(hint, superhint)

    if origin_uw is type(None):
        return _issubnone(hint, superhint)

    if origin_uw is type:
        return _issubtype(hint, superhint)

    if origin_uw is abc.Callable:
        return _issubcallable(hint, superhint)

    if isinstance(origin_uw, type):
        return _issubclasshint(hint, superhint, origin_uw)

    # An unrecognised or future super-hint is opaque: treat it as `Any`, so
    # a method annotated with it stays reachable, and say so once.
    _warn_unknown(superhint)
    return True


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
    if safe_get_origin(superhint_uw) is not tx.Literal:
        # Superhint is not a literal -> error
        raise TypeError(f"Super-hint {superhint} is not a Literal")
    if safe_get_origin(hint_uw) is not tx.Literal:
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


def _is_paramspec_like(params: tx.Any) -> bool:
    """Whether a `Callable` parameter list is a `ParamSpec`/`Concatenate`."""
    if isinstance(params, tx.ParamSpec):
        return True
    if isinstance(params, (tx.ParamSpecArgs, tx.ParamSpecKwargs)):
        return True
    return any(safe_get_origin(params) is form for form in _CONCATENATE_FORMS)


def _issubcallable(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a `Callable[...]`.

    Parameters are compared contravariantly and the return type
    covariantly. `#!python Callable[..., R]` accepts any parameter list. A
    `ParamSpec`/`Concatenate` parameter list is compared only for equality
    -- it degrades rather than raising.
    """
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if get_origin_uw(hint_uw) is not abc.Callable:
        # Only a callable hint can stand in for a `Callable`.
        return False
    superargs = safe_get_args(superhint_uw)
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
    return _issubcallable_params(sub_params, sup_params)


def _issubcallable_params(sub: tx.Any, sup: tx.Any) -> bool:
    """Compare two `Callable` parameter lists, contravariantly."""
    # `Callable[..., R]` on either side accepts any parameter list.
    if sub is Ellipsis or sup is Ellipsis:
        return True
    # A ParamSpec / Concatenate list degrades to equality: comparable to an
    # identical one, and to nothing else, rather than raising.
    if _is_paramspec_like(sub) or _is_paramspec_like(sup):
        return sub == sup
    if not (isinstance(sub, (list, tuple)) and isinstance(sup, (list, tuple))):
        return sub == sup
    if len(sub) != len(sup):
        return False
    # Contravariant: each super-parameter must be a sub-hint of the matching
    # sub-parameter (a function taking `int` can stand in for one taking
    # `bool`, since it also accepts every `bool`).
    return all(issubhint(sp, bp) for bp, sp in zip(sub, sup))
