"""The hint-level subtype relation: `issubhint` and `ishintstance`."""

# stdlib
import warnings
from collections import abc

# dependencies
import typing_extensions as tx

# local
from ._compat import (
    _UNPACK_FORMS,
    UNION_TYPES,
    UnknownHintWarning,
    is_plausible_hint,
    is_special_form,
    is_typeddict_marker,
    spellings,
)
from ._exact import exact_target, is_exact
from ._introspect import (
    eq_safenan,
    get_args_uw,
    get_origin_uw,
    is_typeddict,
    normalise_hint,
    safe_get_args,
    safe_get_origin,
    safe_issubclass,
    typeddict_field_hints,
    typeddict_required_keys,
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


def _is_unpack(hint: tx.Any) -> bool:
    """Whether `hint` is an `Unpack[...]`, in any spelling.

    On 3.11 `typing.Unpack is not typing_extensions.Unpack`, and the star
    syntax `Tuple[int, *Ts]` and `tx.Unpack[Ts]` produce the two different
    spellings -- so the origin is tested against both.
    """
    return any(safe_get_origin(hint) is form for form in _UNPACK_FORMS)


def _is_unpacked_typevartuple(hint: tx.Any) -> bool:
    """Whether `hint` is `Unpack[Ts]` for a `TypeVarTuple` `Ts`."""
    if not _is_unpack(hint):
        return False
    args = tx.get_args(hint)
    return bool(args) and isinstance(args[0], tx.TypeVarTuple)


def _known_form(hint: tx.Any) -> tx.Any:
    """Map a known non-class form to the class it dispatches as.

    `LiteralString` dispatches as [`str`][], and `TypeGuard[...]` /
    `TypeIs[...]` as [`bool`][]. An unpacked `TypeVarTuple` (`*Ts`) is an open
    run of `Any` elements, so on its own -- an `*args: *Ts` tail read as a
    single slot -- it dispatches as [`Any`][typing.Any]. Every other hint is
    returned unchanged.
    """
    if any(hint is form for form in _LITERALSTRING_FORMS):
        return str
    if any(hint is form for form in _TYPEGUARD_FORMS):
        return bool
    origin = safe_get_origin(hint)
    if any(origin is form for form in _TYPEGUARD_FORMS):
        return bool
    if _is_unpacked_typevartuple(hint):
        return tx.Any
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
    except TypeError:
        # An unhashable origin cannot key the set, so fall back to the form's
        # own type.
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
    except Exception:
        # A hint whose own `repr` raises still has to be named in the warning.
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
    * If `hint` is a [`TypedDict`][tx.TypedDict], checks the *shape* of
      `obj`: it must be a [`dict`][] that holds every required key, and each
      declared key it holds must carry a value of that field's type. Extra
      keys are allowed, and a nested `TypedDict` or container field is read
      recursively.
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
        # The marker names no fields, so there is no shape to check; a plain
        # `dict` is not one of it.
        return safe_issubclass(type(obj), origin_uw)
    if is_typeddict(origin_uw):
        # A concrete `TypedDict` describes the *shape* of a mapping, so a
        # value is one when it has the declared keys with the declared value
        # types -- not when its type is nominally the `TypedDict` (a `dict`
        # literal never is). The bare marker was handled just above, so only
        # a `TypedDict` with fields reaches here.
        return _ishintstance_typeddict(obj, origin_uw)
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


def _ishintstance_typeddict(obj: tx.Any, td: tx.Any) -> bool:
    """Check that a value has the shape a `TypedDict` describes.

    A value matches when it is a [`dict`][], holds every **required** key, and
    every declared key it *does* hold carries a value that satisfies that
    field's hint (checked through `ishintstance`, so a nested `TypedDict` or a
    container field is read the same way as any other value). `Required` /
    `NotRequired` and the class's `total=` are honoured through
    `typeddict_required_keys`.

    Only a `dict` is accepted, not any [`Mapping`][collections.abc.Mapping].
    This keeps the value level in step with the hint level, where
    `TypedDict <= dict`: since every `TypedDict`-shaped value must also be a
    valid `dict`, a non-`dict` mapping that matched the shape but is not a
    `dict` would break `v in S and S <= T => v in T`.

    **Extra keys are allowed**: a `dict` with keys beyond the declared ones
    still matches, so long as the declared keys check out. The reason is the
    nominal subtyping the hint level already encodes: `Sub <= Base` holds for
    a `TypedDict` `Sub` that inherits from `Base`, so every `Sub`-shaped value
    must also satisfy `Base`. Were extra keys rejected, a `Sub` value carrying
    `Sub`'s own extra keys would fail `Base`, breaking
    `v in Sub and Sub <= Base => v in Base`. This matches `pydantic`'s
    `TypeAdapter` (a `TypedDict` names a minimum shape, not a closed one);
    `typeguard` is stricter, and the permissive reading is chosen here to keep
    the value level sound against the hint-level ordering.

    Two *unrelated* `TypedDict`s that happen to share a satisfiable shape are
    not ordered by this check, so a value matching both dispatches to neither
    on its own: selection raises `AmbiguousMethodError`, the same outcome two
    equally-matched `Protocol`s give (RFC 0001 §5).
    """
    if not isinstance(obj, dict):
        return False
    for key in typeddict_required_keys(td):
        if key not in obj:
            return False
    for key, field_hint in typeddict_field_hints(td).items():
        if key not in obj:
            continue
        if isinstance(field_hint, (str, tx.ForwardRef)):
            # A forward reference that could not be resolved -- a bare string
            # on some versions, a `ForwardRef` on others. Its value type
            # cannot be read here, so the present value is accepted rather
            # than raised on; but the field is reported, the same way an
            # unrecognised hint is reported elsewhere, so a shape that silently
            # skips a check is not mistaken for one that passed it.
            _warn_unknown(field_hint)
            continue
        if not ishintstance(obj[key], field_hint):
            return False
    return True


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
        # A super-hint that *contains* `Exact` -- a union with an `Exact`
        # member, or a typevar bounded/constrained by one -- must distribute
        # first, so the exactness is matched member by member rather than lost
        # by reducing to `issubhint(target, superhint)`.
        sup_origin = get_origin_uw(superhint)
        if sup_origin in UNION_TYPES and get_args_uw(superhint):
            return any(
                issubhint(hint, arg) for arg in get_args_uw(superhint)
            )
        if isinstance(sup_origin, tx.TypeVar):
            return _issubtypevar(hint, superhint)
        # `Exact[D] <= P` iff `D <= P` (an exactly-`D` value is a `D`).
        return issubhint(target, superhint)
    if is_exact(superhint):
        target = normalise_hint(exact_target(superhint))
        # A parametrised union sub-hint distributes member by member, so a
        # union that is equivalent to a `Literal` (e.g.
        # `Union[Literal[1], Literal[2]]` == `Literal[1, 2]`) is ordered
        # against `Exact[C]` the same way that `Literal` is -- keeping the
        # relation transitive.
        if get_origin_uw(hint) in UNION_TYPES and get_args_uw(hint):
            return all(
                issubhint(member, superhint) for member in get_args_uw(hint)
            )
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

    # The dual of the Literal sub-hint rule above: the super-hint is now known
    # not to be a union, so a *parametrised* union sub-hint is a subhint iff
    # every one of its members is (`Union[bool, int] <= int`,
    # `Union[int, str] <= object`). A bare, unparametrised `Union` has no
    # members and falls through to the branches below.
    if get_origin_uw(hint) in UNION_TYPES and get_args_uw(hint):
        return all(
            issubhint(member, superhint) for member in get_args_uw(hint)
        )

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

    if is_special_form(origin_uw):
        # A recognised special form with no branch of its own and nothing left
        # to check -- a bare, unsubscripted `Annotated` (its origin is itself,
        # not a class), or a future class-shaped construct -- is opaque:
        # permissive like `Any`, so a hint or registry key written with it
        # stays reachable (RFC 0001 §2.1) rather than being mistaken for a
        # subclassable class below. A subscripted form has already resolved to
        # its inner origin above, so only a bare one reaches here.
        return True

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
    """Check a hint's arguments against a superhint's, covariantly.

    The arguments are read as tuple *shapes* (a fixed prefix, an optional open
    run, a fixed suffix) so a trailing ellipsis (`Tuple[int, ...]`) and an
    unpacked `TypeVarTuple` (`Tuple[int, *Ts]`) are ordered the same way. A
    plain, fully fixed argument list (`List[int]`, `Dict[str, int]`) is a
    closed shape, so it is compared element by element as before.
    """
    return _issubtupleshape(_tuple_shape(args), _tuple_shape(superargs))


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
    A `#!python ...` or a bare `ParamSpec` parameter list is the **top** of
    parameter lists -- the widest, describing every callable -- so a fixed
    list is a sub-hint of it but not the other way round; a
    `#!python Concatenate[X, P]` list is a contravariant fixed prefix followed
    by an open tail, sitting between the fixed lists and the top (RFC 11.1). A
    callable *class* -- a function type, `#!python type`, `#!python Type[C]`,
    or a class with `__call__` -- has no parameter list, so it stands in only
    for an unparametrised `Callable`.
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
    return _issubparams(
        _callable_param_shape(hint_uw, sub_params),
        _callable_param_shape(superhint_uw, sup_params),
    )


class _ParamShape(tx.NamedTuple):
    """A `Callable` parameter list as a shape (RFC 11.1).

    A fixed, contravariant `prefix` followed by a `tail` saying how the list
    ends:

    * `#!python None` -- a **closed** (fixed-arity) list, e.g. `[int, str]`;
    * `#!python Ellipsis` or a [`ParamSpec`][typing.ParamSpec] -- an **open**
      list, one that may be called with arbitrarily many further arguments.

    An open list is the *top* of parameter lists, the way `#!python Tuple[X,
    ...]` tops the fixed-length tuples: a fixed list is a sub-hint of it, but
    it is not a sub-hint of any fixed list.
    """

    prefix: tx.Tuple[tx.Any, ...]
    tail: tx.Any


# A sentinel `_match_params` returns when a closed super matches: there is no
# tail to capture, but the result must be non-`None` to signal the match.
_CLOSED_MATCH = _ParamShape((), None)


def _callable_param_shape(alias: tx.Any, params: tx.Any) -> _ParamShape:
    """Classify a `Callable` parameter list into a `_ParamShape` (RFC 11.1).

    `alias` is the whole `Callable[...]` hint, needed to reach
    `#!python __parameters__` where an older Python has erased a `ParamSpec`
    out of the arguments; `params` is its parameter-list argument.
    """
    # `Ellipsis` and the `ParamSpec` forms name the open top. `ParamSpec` is
    # tested before `list`: on 3.8/3.9 a `ParamSpec` *is* a `list` subclass,
    # so the list branch would otherwise claim it.
    if params is Ellipsis:
        return _ParamShape((), Ellipsis)
    if isinstance(params, tx.ParamSpec):
        return _ParamShape((), params)
    if isinstance(params, (tx.ParamSpecArgs, tx.ParamSpecKwargs)):
        return _ParamShape((), Ellipsis)
    if any(safe_get_origin(params) is form for form in _CONCATENATE_FORMS):
        # `Concatenate[X1, ..., Xn, tail]`: the fixed prefix is everything but
        # the trailing element, which is a `ParamSpec` (or `...` on 3.10+).
        args = tx.get_args(params)
        return _ParamShape(tuple(args[:-1]), args[-1])
    seq = list(params) if isinstance(params, (list, tuple)) else None
    if seq is None:
        # An unknown shape degrades to the widest (open) list rather than
        # raising -- the opaque rule for a form the relation does not model.
        return _ParamShape((), Ellipsis)
    if seq and isinstance(seq[-1], tx.ParamSpec):
        # Below 3.10 `Concatenate[X, P]` is flattened to `[X, ..., P]`, and a
        # bare `P` is the one-element list `[P]` (typing_extensions >= 4.13).
        return _ParamShape(tuple(seq[:-1]), seq[-1])
    if seq and seq[-1] is Ellipsis:
        return _ParamShape(tuple(seq[:-1]), Ellipsis)
    if not seq:
        for parameter in getattr(alias, "__parameters__", ()):
            if isinstance(parameter, tx.ParamSpec):
                # Below 3.10 a bare `P` is erased to `[]`, surviving only in
                # `__parameters__` -- an open list, not a zero-argument one.
                return _ParamShape((), parameter)
    return _ParamShape(tuple(seq), None)


def _match_params(
    sub: _ParamShape, sup: _ParamShape
) -> tx.Optional[_ParamShape]:
    """Match one parameter-list shape against another (RFC 11.1).

    `sub` describes the callables the left `Callable` accepts, `sup` those the
    right does; the match holds when every callable `sub` describes `sup`
    describes too, with the committed prefixes compared **contravariantly**.

    * A **closed** `sup` accepts exactly its own arity: `sub` must be closed
      too, with the same prefix length.
    * An **open** `sup` accepts its committed prefix and anything after: `sub`
      must supply at least that prefix. The variable at `sup`'s tail then
      captures the rest of `sub`'s list -- the leftover prefix plus `sub`'s own
      tail.

    Returns that captured tail shape when `sup` is open, a sentinel closed
    shape when `sup` is closed and matches, or `#!python None` on no match.
    """
    sub_closed = sub.tail is None
    sup_closed = sup.tail is None
    k = len(sub.prefix)
    m = len(sup.prefix)
    if sup_closed:
        if not sub_closed or k != m:
            return None
    elif k < m:
        return None
    # Compare the committed prefix contravariantly: each super-parameter must
    # be a sub-hint of the matching sub-parameter (a function taking `int` can
    # stand in for one taking `bool`).
    for i in range(m):
        if not issubhint(sup.prefix[i], sub.prefix[i]):
            return None
    if sup_closed:
        return _CLOSED_MATCH
    return _ParamShape(tuple(sub.prefix[m:]), sub.tail)


def _issubparams(sub: _ParamShape, sup: _ParamShape) -> bool:
    """Whether one parameter-list shape stands in for another (RFC 11.1)."""
    return _match_params(sub, sup) is not None


class _TupleShape(tx.NamedTuple):
    """A tuple's arguments as a shape (RFC 0001 §11.1, PEP 646).

    A tuple is a fixed, front-aligned `prefix`, an optional open middle run,
    and a fixed, back-aligned `suffix`. The middle run is:

    * `#!python None` in `rep` -- a **closed** (fixed-arity) tuple, e.g.
      `Tuple[int, str]`;
    * otherwise `rep` is the element upper bound of the run -- an **open**
      tuple, `Tuple[int, ...]` (bound `int`) or `Tuple[int, *Ts]` (bound
      `Any`).

    `var` is the `TypeVarTuple` an `*Ts` run stands for, or `#!python None`.
    All fixed positions are covariant; an open run tops the tuples of its
    shape, as `Tuple[int, ...]` tops the fixed-length tuples beginning `int`.
    """

    prefix: tx.Tuple[tx.Any, ...]
    rep: tx.Any
    suffix: tx.Tuple[tx.Any, ...]
    var: tx.Any


# The sentinel `_match_tuple` returns when a closed super matches: nothing is
# captured, but the result must be non-`None` to signal the match.
_CLOSED_TUPLE_MATCH = _TupleShape((), None, (), None)


def _tuple_shape(args: tx.Tuple[tx.Any, ...]) -> _TupleShape:
    """Classify a tuple's arguments into a `_TupleShape`.

    `Tuple[()]` on 3.8-3.10 reports its arguments as the phantom `#!python
    ((),)`; it is normalised here to no elements (the empty-tuple bug #36 is
    tracked separately and untouched). A second open run in one tuple is
    refused (PEP 646 allows a single unpack; `typing` does not enforce it at
    runtime, so the relation does).
    """
    if args == ((),):
        # The 3.8-3.10 `Tuple[()]` phantom: a single empty-tuple element that
        # means "no elements", not a one-element tuple whose element is `()`.
        args = ()
    prefix = []  # type: tx.List[tx.Any]
    suffix = []  # type: tx.List[tx.Any]
    rep = None  # type: tx.Any
    var = None  # type: tx.Any
    open_seen = False
    index = 0
    while index < len(args):
        element = args[index]
        if element is Ellipsis:
            # `Tuple[X, ...]`: the preceding single element is the open run's
            # upper bound.
            if open_seen or not prefix:
                raise TypeError(
                    "a tuple may hold at most one open run of elements "
                    "(`...` or `*Ts`)"
                )
            rep = prefix.pop()
            open_seen = True
            index += 1
            continue
        if _is_unpack(element):
            if open_seen:
                raise TypeError(
                    "a tuple may hold at most one open run of elements "
                    "(`...` or `*Ts`)"
                )
            inner = tx.get_args(element)[0] if tx.get_args(element) else None
            if isinstance(inner, tx.TypeVarTuple):
                rep = tx.Any
                var = inner
                open_seen = True
            elif safe_get_origin(inner) is tuple:
                # `Tuple[int, *Tuple[str, int]]` / `Tuple[*Tuple[str, ...]]`:
                # splice the unpacked tuple's own shape in.
                spliced = _tuple_shape(tx.get_args(inner))
                if spliced.rep is None:
                    # A fixed unpacked tuple flattens into the prefix.
                    prefix.extend(spliced.prefix)
                else:
                    prefix.extend(spliced.prefix)
                    rep = spliced.rep
                    var = spliced.var
                    suffix.extend(spliced.suffix)
                    open_seen = True
            else:
                # An unpacked something the relation does not model: treat its
                # run as `Any`, and report the form once.
                _warn_unknown(element)
                rep = tx.Any
                open_seen = True
            index += 1
            continue
        (suffix if open_seen else prefix).append(element)
        index += 1
    return _TupleShape(tuple(prefix), rep, tuple(suffix), var)


def _match_tuple(
    sub: _TupleShape, sup: _TupleShape
) -> tx.Optional[_TupleShape]:
    """Match one tuple shape against another, covariantly (the tuple twin of
    `_match_params`).

    Returns the shape a `*Ts` in `sup` would capture from `sub` (the run `sub`
    supplies beyond `sup`'s fixed prefix and suffix), a sentinel closed shape
    when `sup` is closed and matches, or `#!python None` on no match. Every
    fixed position is compared covariantly (`issubhint(sub_i, sup_i)`).
    """
    sub_open = sub.rep is not None
    sup_open = sup.rep is not None
    if not sup_open:
        # A closed super accepts exactly its own arity.
        if sub_open or len(sub.prefix) != len(sup.prefix):
            return None
        for element, superel in zip(sub.prefix, sup.prefix):
            if not issubhint(element, superel):
                return None
        return _CLOSED_TUPLE_MATCH
    p2, r2, q2 = sup.prefix, sup.rep, sup.suffix
    if not sub_open:
        # A closed sub against an open super: it must supply the whole fixed
        # prefix and suffix, and its middle elements must satisfy the run.
        elements = sub.prefix
        if len(elements) < len(p2) + len(q2):
            return None
        for i in range(len(p2)):
            if not issubhint(elements[i], p2[i]):
                return None
        for j in range(1, len(q2) + 1):
            if not issubhint(elements[-j], q2[-j]):
                return None
        middle = elements[len(p2): len(elements) - len(q2)]
        for element in middle:
            if not issubhint(element, r2):
                return None
        return _TupleShape(tuple(middle), None, (), None)
    # Both open: the sub's fixed prefix/suffix must cover the super's, its
    # extra fixed elements must satisfy the super's run, and `r1 <= r2` holds.
    p1, r1, q1 = sub.prefix, sub.rep, sub.suffix
    if len(p1) < len(p2) or len(q1) < len(q2):
        return None
    for i in range(len(p2)):
        if not issubhint(p1[i], p2[i]):
            return None
    for extra in p1[len(p2):]:
        if not issubhint(extra, r2):
            return None
    if not issubhint(r1, r2):
        return None
    for j in range(1, len(q2) + 1):
        if not issubhint(q1[-j], q2[-j]):
            return None
    for extra in q1[: len(q1) - len(q2)]:
        if not issubhint(extra, r2):
            return None
    return _TupleShape(
        tuple(p1[len(p2):]), r1, tuple(q1[: len(q1) - len(q2)]), None
    )


def _issubtupleshape(sub: _TupleShape, sup: _TupleShape) -> bool:
    """Whether one tuple shape stands in for another."""
    return _match_tuple(sub, sup) is not None
