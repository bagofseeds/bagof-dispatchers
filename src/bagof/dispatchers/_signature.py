"""Signatures, parameters and call binding, all aware of argument names.

Everything dispatch compares two methods by ultimately comes down to a
[`Signature`][]. It records each of a callable's parameters by name,
together with its type hint, the way it may be passed (positionally, by
keyword, or both), and whether it has a default, and it records the
`#!python *args` and `#!python **kwargs` hints separately from the named
parameters. Given a call, a signature works out which parameter each
argument lands in exactly the way Python itself would
([`bind`][Signature.bind]), or that the call does not fit the signature
at all. From a binding it can then say whether it accepts a given set
of argument values ([`applies_to_values`][Signature.applies_to_values])
or argument hints ([`applies_to_hints`][Signature.applies_to_hints]),
and it can compare its own specificity against another signature's for
a particular call shape ([`le`][Signature.le]).

The binding logic mirrors [`inspect.Signature.bind`][] precisely, but
the plan it follows is worked out once, when the signature is first
built, rather than recomputed on every call.
"""

# stdlib
import functools
import inspect
import itertools

# dependencies
import typing_extensions as tx

# local
from ._lattice import (
    equivalent,
    paramspec_captures,
    paramspec_consistent,
    typevar_consistent,
    typevartuple_captures,
    typevartuple_consistent,
)
from .core import (
    get_args_uw,
    get_origin_uw,
    is_typeddict,
    ishint,
    ishintstance,
    issubhint,
    normalise_hint,
    safe_get_origin,
    unwrap,
)
from .core._bounds import (
    _Lower,
    between_bounds,
    constraint_bound_message,
    empty_interval_message,
    endpoint_bound_message,
    find_bound,
    is_between,
    is_bound,
    is_unbounded_form,
    member_bound_message,
    misplaced_bound_message,
    unbounded_form_message,
    value_bound_message,
    written_ends,
)
from .core._compat import _UNPACK_FORMS, UNION_TYPES, spellings
from .core._exact import exact_target, is_exact
from .core._hint import Hint, is_hint_form
from .core._introspect import _typing_spelling
from .core._relation import (
    _is_subscripted_tuple,
    _is_unpacked_typevartuple,
    _malformed_typeddict_reason,
    _TupleShape,
    arguments_bound_message,
)
from .core._super import is_super, super_target

__all__ = ["Parameter", "Signature", "Binding"]

# The parameter kinds, mirrored from `inspect` so a user never has to import
# both. `empty` marks a parameter with no default.
_empty = inspect.Parameter.empty
_POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
_VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
_KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD

# A placeholder for a value that is never looked at -- used to bind a bare call
# *shape* (so many positionals, these keyword names) with no real arguments.
_PLACEHOLDER = object()


class _CatchAll:
    """A stand-in for `*args` or `**kwargs` inside the binding plan.

    It offers exactly what the binding loop reads off any ordinary
    parameter: a `kind`, a `name` that can never match an actual keyword,
    and a `default` of "none", so the loop can walk it alongside real
    parameters without a special case.
    """

    __slots__ = ("kind",)
    name = None
    default = _empty

    def __init__(self, kind: tx.Any) -> None:
        self.kind = kind


_VARARGS = _CatchAll(_VAR_POSITIONAL)
_VARKW = _CatchAll(_VAR_KEYWORD)

# The fallback signature for a callable Python cannot introspect (a builtin
# type such as `int`, a C function): a bare `(*args, **kwargs)` catch-all, so
# it binds any call and dispatches on `Any`.
_ANY_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter("args", _VAR_POSITIONAL),
        inspect.Parameter("kwargs", _VAR_KEYWORD),
    ]
)

# Every spelling of `Literal`, to recognise it by origin without evaluating
# its members: on 3.8-3.10 `typing_extensions` ships its own, distinct from
# `typing`'s, so a single-object check misses the other.
_LITERAL_FORMS = spellings("Literal")
_PARAMSPEC_ARGKW = tuple(
    form
    for name in ("ParamSpecArgs", "ParamSpecKwargs")
    for form in (getattr(tx, name, None),)
    if isinstance(form, type)
)
# Every spelling of `Concatenate`, to recognise a `Concatenate[...]` used
# (wrongly) as a plain parameter annotation. On 3.10 `typing.Concatenate is
# not tx.Concatenate`, so both spellings must be checked.
_CONCATENATE_FORMS = spellings("Concatenate")


class Parameter:
    """One parameter of a [`Signature`][].

    A `Parameter` is immutable once constructed and holds its `name`,
    its type `hint`, its `kind` (how it may be passed to a call), and its
    `default` value. A parameter that has no default value is required.

    !!! example
        ```pycon
        >>> p = Parameter("x", int, Parameter.POSITIONAL_OR_KEYWORD)
        >>> p.name, p.required
        ('x', True)
        ```

    Parameters
    ----------
    name
        The parameter's name.
    hint
        The type hint dispatch reads for this parameter. Leaving a
        parameter unannotated gives it [`Any`][typing.Any], so it
        accepts any argument.
    kind
        One of `Parameter.POSITIONAL_ONLY`,
        `Parameter.POSITIONAL_OR_KEYWORD` or `Parameter.KEYWORD_ONLY`,
        mirroring [`inspect.Parameter`][]. The `#!python *args` and
        `#!python **kwargs` hints are not represented this way; they live
        directly on the signature instead.
    default
        The parameter's default value, or `Parameter.empty` when it is
        required.

    Attributes
    ----------
    required : bool
        Whether the parameter has no default value.
    """

    __slots__ = ("name", "hint", "kind", "default")

    empty = _empty
    POSITIONAL_ONLY = _POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD = _POSITIONAL_OR_KEYWORD
    KEYWORD_ONLY = _KEYWORD_ONLY
    VAR_POSITIONAL = _VAR_POSITIONAL
    VAR_KEYWORD = _VAR_KEYWORD

    def __init__(
        self,
        name: str,
        hint: tx.Any,
        kind: tx.Any,
        default: tx.Any = _empty,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "hint", hint)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "default", default)

    def __setattr__(self, name: str, value: tx.Any) -> None:
        raise AttributeError("a Parameter is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a Parameter is immutable")

    @property
    def required(self) -> bool:
        """Report whether the parameter has no default value."""
        return self.default is _empty

    def __eq__(self, other: tx.Any) -> bool:
        if not isinstance(other, Parameter):
            return NotImplemented
        return (
            self.name == other.name
            and self.kind == other.kind
            and self.required == other.required
            and _hint_eq(self.hint, other.hint)
        )

    def __hash__(self) -> int:
        return hash((self.name, self.kind, self.required))

    def __repr__(self) -> str:
        text = f"{self.name}: {_render_hint(self.hint)}"
        if not self.required:
            text += f" = {self.default!r}"
        return f"Parameter({text})"


class Binding:
    """Where every argument of a successfully bound call landed.

    [`bind`][Signature.bind] returns a `Binding` on success and
    [`None`][] on failure, so a `Binding` only ever describes a call that
    did fit the signature it was bound against.

    Attributes
    ----------
    slots : Mapping
        Each argument's key (an integer for a positional argument, or
        the keyword's name for a keyword one) mapped to where that
        argument landed: a parameter name, or `Parameter.VAR_POSITIONAL`
        or `Parameter.VAR_KEYWORD` for one absorbed by `#!python *args`
        or `#!python **kwargs`.
    extra_positional : tuple
        The indices of the positional arguments absorbed by
        `#!python *args`.
    extra_keywords : Mapping
        The keyword arguments absorbed by `#!python **kwargs`, keyed by
        name.
    defaulted : frozenset
        The names of the parameters left to their own defaults. These
        were never arguments of the call itself, so dispatch never
        checks their hints.
    """

    __slots__ = ("slots", "extra_positional", "extra_keywords", "defaulted")

    def __init__(
        self,
        slots: tx.Dict[tx.Any, tx.Any],
        extra_positional: tx.Tuple[int, ...],
        extra_keywords: tx.Dict[str, tx.Any],
        defaulted: tx.FrozenSet[str],
    ) -> None:
        self.slots = slots
        self.extra_positional = extra_positional
        self.extra_keywords = extra_keywords
        self.defaulted = defaulted

    def __eq__(self, other: tx.Any) -> bool:
        if not isinstance(other, Binding):
            return NotImplemented
        return (
            self.slots == other.slots
            and self.extra_positional == other.extra_positional
            and self.extra_keywords == other.extra_keywords
            and self.defaulted == other.defaulted
        )

    def __repr__(self) -> str:
        return (
            f"Binding(slots={self.slots!r}, "
            f"extra_positional={self.extra_positional!r}, "
            f"extra_keywords={self.extra_keywords!r}, "
            f"defaulted={self.defaulted!r})"
        )


class Signature:
    """The set of parameters, and their hints, dispatch binds a call against.

    A `Signature` holds its `parameters` in declaration order, together
    with the `#!python *args` hint ([`varargs`][]) and the
    `#!python **kwargs` hint ([`varkw`][]) when the callable it describes
    accepts either. Build one from an existing callable with
    [`from_callable`][], or, with no callable at all, straight from a
    list of hints with [`from_hints`][].

    Every declared parameter takes part in dispatch, whether or not it
    carries a hint of its own. An unannotated parameter is treated as
    [`Any`][typing.Any], so it still has to be filled for a call to bind
    but never narrows down which method gets chosen.

    !!! example
        ```pycon
        >>> def area(shape, scale=1.0): ...
        >>> sig = Signature.from_callable(area)
        >>> list(sig.parameters)
        ['shape', 'scale']
        >>> sig.dispatched_names
        ('shape', 'scale')
        ```
    """

    __slots__ = (
        "_parameters",
        "_varargs",
        "_varkw",
        "_pre",
        "_kwonly",
        "_canonical",
        "_deferred",
        "_fn",
        "_raw",
        "_varargs_name",
        "_varkw_name",
    )

    def __init__(
        self,
        parameters: tx.Mapping[str, Parameter],
        varargs: tx.Any = None,
        varkw: tx.Any = None,
    ) -> None:
        """Build a signature from its parameters and its catch-all hints.

        Parameters
        ----------
        parameters
            The parameters, keyed by name, in the order they should be
            considered.
        varargs
            The `#!python *args` element hint, or [`None`][] when the
            callable takes no `#!python *args` at all. An unannotated
            `#!python *args` is represented as [`Any`][typing.Any], not
            as `#!python None`.
        varkw
            The `#!python **kwargs` value hint, or [`None`][] when the
            callable takes no `#!python **kwargs`.
        """
        self._parameters = dict(parameters)
        self._varargs = varargs
        self._varkw = varkw
        self._deferred = False
        self._fn = None
        self._raw = None
        self._varargs_name = None
        self._varkw_name = None
        self._build_plan()

    # -- construction ---------------------------------------------------

    @classmethod
    def from_callable(cls, fn: tx.Callable[..., tx.Any]) -> "Signature":
        """Read a signature off an existing callable.

        Parameter names, kinds, and defaults come from
        [`inspect.signature`][], while the hints themselves come from
        [`typing_extensions.get_type_hints`][], read with
        [`Annotated`][typing.Annotated] metadata preserved so that a
        construct such as `#!python Exact[...]` survives intact. A
        `#!python *args: H` parameter becomes the signature's `varargs`,
        a `#!python **kwargs: H` parameter becomes its `varkw`, and the
        return annotation is ignored entirely. Any parameter with no
        annotation of its own dispatches on [`Any`][typing.Any].

        A hint can be a forward reference that cannot yet be resolved,
        for instance a name defined further down the same module or one
        imported only under `#!python TYPE_CHECKING`. When that happens,
        the signature holds onto the raw annotation and resolves it the
        first time the signature is actually used for dispatch. A name
        still undefined at that point raises [`NameError`][].
        """
        try:
            isig = inspect.signature(fn)
        except (ValueError, TypeError):
            # A callable with no introspectable signature -- a builtin type
            # such as `int`, or a C function -- still registers as an
            # implementation: it takes a catch-all `(*args, **kwargs)`, so it
            # binds any call and dispatches on `Any` (the widest fallback), and
            # a more specific method still wins over it.
            isig = _ANY_SIGNATURE
        source = _hint_source(fn)
        try:
            hints = _resolve_hints(source)
            deferred = False
            raw = None
        except NameError:
            # A genuine forward reference -- a name defined further down the
            # module or only under `TYPE_CHECKING`. Keep the raw annotations
            # and resolve them the first time the signature is used.
            hints = None
            deferred = True
            raw = _raw_annotations(source)
        except TypeError:
            # `get_type_hints` also raises `TypeError` for reasons that are
            # not a forward reference: a `functools.partial`, a callable
            # instance, or a stringised modern spelling (`"list[int]"`) that
            # the running Python cannot evaluate. Only the last is a real
            # deferral (RFC 0001 §11.3); the others have annotations that are
            # already objects, so resolve those directly instead of pretending
            # every parameter is `Any`.
            raw = _raw_annotations(source)
            if _has_forward(raw):
                hints = None
                deferred = True
            else:
                hints = {
                    name: normalise_hint(value)
                    for name, value in raw.items()
                }
                deferred = False
                raw = None
        return cls._from_inspect(isig, hints, fn, deferred, raw)

    @classmethod
    def from_hints(
        cls, *hints: tx.Any, **named_hints: tx.Any
    ) -> "Signature":
        """Build a signature straight from hints, with no callable behind it.

        This is the primitive that def-less registration forms build on,
        such as
        [`Function.from_mapping`][bagof.dispatchers.Function.from_mapping].
        Each positional hint given becomes a positional-only parameter,
        and each keyword hint becomes a parameter of that name that a
        call may fill either positionally or by keyword. Every parameter
        produced this way is required, with no default of its own.

        !!! example
            ```pycon
            >>> sig = Signature.from_hints(int, scale=float)
            >>> sig.dispatched_names
            ('scale',)
            ```
        """
        params: tx.Dict[str, Parameter] = {}
        for index, hint in enumerate(hints):
            name = f"_{index}"
            normalised = normalise_hint(hint)
            subject = f"positional hint {index}"
            _reject_misplaced_bound(name, normalised, subject=subject)
            _reject_malformed_typeddict(name, normalised, subject=subject)
            params[name] = Parameter(name, normalised, _POSITIONAL_ONLY)
        for name, hint in named_hints.items():
            normalised = normalise_hint(hint)
            _reject_misplaced_bound(name, normalised)
            _reject_malformed_typeddict(name, normalised)
            params[name] = Parameter(
                name, normalised, _POSITIONAL_OR_KEYWORD
            )
        return cls(params)

    @classmethod
    def _from_inspect(
        cls,
        isig: inspect.Signature,
        hints: tx.Optional[tx.Dict[str, tx.Any]],
        fn: tx.Callable[..., tx.Any],
        deferred: bool,
        raw: tx.Optional[tx.Dict[str, tx.Any]],
    ) -> "Signature":
        """Assemble a signature from an [`inspect.Signature`][] and its
        hints.
        """
        params: tx.Dict[str, Parameter] = {}
        varargs = None
        varkw = None
        varargs_name = None
        varkw_name = None
        for name, param in isig.parameters.items():
            hint = _hint_for(name, hints, raw, deferred)
            if param.kind is _VAR_POSITIONAL:
                varargs_name = name
                varargs = _catch_all_or_any(hint)
                _reject_variadic_param(name, varargs, fn, catch_all=True)
                _reject_misplaced_bound(name, hint, fn)
                _reject_malformed_typeddict(name, hint, fn)
            elif param.kind is _VAR_KEYWORD:
                varkw_name = name
                varkw = _catch_all_or_any(hint)
                _reject_misplaced_bound(name, hint, fn)
                _reject_malformed_typeddict(name, hint, fn)
            else:
                _reject_variadic_param(name, hint, fn)
                _reject_misplaced_bound(name, hint, fn)
                _reject_malformed_typeddict(name, hint, fn)
                params[name] = Parameter(
                    name, hint, param.kind, param.default
                )
        self = cls(params, varargs, varkw)
        self._deferred = deferred
        self._fn = fn
        self._raw = raw
        self._varargs_name = varargs_name
        self._varkw_name = varkw_name
        return self

    def _build_plan(self) -> None:
        """Precompute the order [`bind`][Signature.bind] walks the
        parameters in.

        `_pre` holds, in order, every parameter an argument could fill by
        position. `_kwonly` holds the keyword-only parameters. `_canonical`
        holds the complete order the binder actually walks, with the
        `#!python *args` and `#!python **kwargs` stand-ins inserted at
        the same points Python itself would put them.
        """
        values = list(self._parameters.values())
        self._pre = tuple(
            p for p in values if p.kind is not _KEYWORD_ONLY
        )
        self._kwonly = tuple(
            p for p in values if p.kind is _KEYWORD_ONLY
        )
        canonical = list(self._pre)
        if self._varargs is not None:
            canonical.append(_VARARGS)
        canonical.extend(self._kwonly)
        if self._varkw is not None:
            canonical.append(_VARKW)
        self._canonical = tuple(canonical)

    def _settle(self) -> None:
        """Resolve any deferred forward-reference hints, once.

        This runs ahead of every use of the signature for dispatch. When
        a name it needs is still undefined by the time it runs, it
        raises [`NameError`][], naming the callable it was resolving
        hints for.
        """
        if not self._deferred:
            return
        try:
            hints = _resolve_hints(_hint_source(self._fn))
        except (NameError, TypeError) as error:
            name = getattr(self._fn, "__qualname__", repr(self._fn))
            raise NameError(
                f"cannot resolve the type hints of {name}: {error}"
            ) from error
        new_params: tx.Dict[str, Parameter] = {}
        for name, param in self._parameters.items():
            hint = normalise_hint(hints.get(name, tx.Any))
            # A forward reference that resolved to a `ParamSpec`/`Concatenate`
            # is refused here, the same as one written outright.
            _reject_variadic_param(name, hint, self._fn)
            # A forward reference that resolved to a malformed `TypedDict`, to
            # a bound in a position where it cannot stand, or to an empty
            # interval, is refused here too, now that the name has become
            # readable.
            _reject_misplaced_bound(name, hint, self._fn)
            _reject_malformed_typeddict(name, hint, self._fn)
            new_params[name] = Parameter(
                name, hint, param.kind, param.default
            )
        self._parameters = new_params
        if self._varargs_name is not None:
            resolved = normalise_hint(hints.get(self._varargs_name, tx.Any))
            self._varargs = _catch_all_or_any(resolved)
            _reject_variadic_param(
                self._varargs_name, self._varargs, self._fn, catch_all=True
            )
            _reject_misplaced_bound(self._varargs_name, resolved, self._fn)
            _reject_malformed_typeddict(
                self._varargs_name, resolved, self._fn
            )
        if self._varkw_name is not None:
            resolved = normalise_hint(hints.get(self._varkw_name, tx.Any))
            self._varkw = _catch_all_or_any(resolved)
            _reject_misplaced_bound(self._varkw_name, resolved, self._fn)
            _reject_malformed_typeddict(self._varkw_name, resolved, self._fn)
        # Build the plan before clearing the deferred flag: a reader on a
        # free-threaded build (3.13t) must never see `_deferred` false while
        # the plan still reflects the unresolved hints, so the flag is
        # published last.
        self._build_plan()
        self._deferred = False

    # -- public data ----------------------------------------------------

    @property
    def parameters(self) -> tx.Mapping[str, Parameter]:
        """A read-only, name-keyed view of the parameters, in order."""
        return _ReadonlyMap(self._parameters)

    @property
    def varargs(self) -> tx.Any:
        """The `#!python *args` element hint, or [`None`][] if there is none.

        An unannotated `#!python *args` is represented as
        [`Any`][typing.Any]; `#!python None` means the callable does not
        accept `#!python *args` at all.
        """
        return self._varargs

    @property
    def varkw(self) -> tx.Any:
        """The `#!python **kwargs` value hint, or [`None`][] if there is none.

        An unannotated `#!python **kwargs` is represented as
        [`Any`][typing.Any]; `#!python None` means the callable does not
        accept `#!python **kwargs` at all.
        """
        return self._varkw

    @property
    def dispatched_names(self) -> tx.Tuple[str, ...]:
        """The names an argument may be given by keyword and still dispatch.

        This covers the positional-or-keyword and keyword-only
        parameters, in order. A positional-only parameter can only be
        dispatched on by its position, so it never appears here.
        """
        return tuple(
            name
            for name, param in self._parameters.items()
            if param.kind is not _POSITIONAL_ONLY
        )

    @staticmethod
    def shape(
        args: tx.Sequence[tx.Any], kwargs: tx.Mapping[str, tx.Any]
    ) -> tx.Tuple[int, tx.Tuple[str, ...]]:
        """Reduce a call to its shape: how many positionals, which keywords.

        Whether one signature is more specific than another can only be
        decided relative to a particular shape, because which parameter
        an argument lands in depends on how the call was actually
        written. Two calls that pass "the same" arguments differently,
        such as `#!python f(1, 2)` and `#!python f(1, y=2)`, count as
        different shapes here.
        """
        return (len(args), tuple(sorted(kwargs)))

    # -- binding --------------------------------------------------------

    def bind(
        self,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
    ) -> tx.Optional[Binding]:
        """Bind a call to the parameters, exactly the way Python itself would.

        The [`Binding`][] returned on success says where every argument
        landed. [`None`][] comes back instead when the call simply does
        not fit, whether because it has too many positionals for a
        signature with no `#!python *args`, an unexpected keyword for one
        with no `#!python **kwargs`, the same parameter filled twice, or
        a required parameter left with nothing to fill it.
        This follows [`inspect.Signature.bind`][] precisely, working from
        the plan computed once when the signature was built.

        !!! example
            ```pycon
            >>> sig = Signature.from_hints(int, y=int)
            >>> sig.bind((1,), {"y": 2}) is None
            False
            >>> sig.bind((1, 2, 3), {}) is None   # too many positionals
            True
            ```
        """
        slots: tx.Dict[tx.Any, tx.Any] = {}
        extra_positional: tx.List[int] = []
        remaining = dict(kwargs)
        defaulted: tx.Set[str] = set()

        parameters = iter(self._canonical)
        arg_vals = enumerate(args)
        parameters_ex: tx.Tuple[tx.Any, ...] = ()

        while True:
            try:
                index, _ = next(arg_vals)
            except StopIteration:
                # Positionals exhausted; look at the next parameter to
                # decide whether the keyword phase can carry on.
                try:
                    param = next(parameters)
                except StopIteration:
                    break
                if param.kind is _VAR_POSITIONAL:
                    break
                if param.name in remaining:
                    if param.kind is _POSITIONAL_ONLY:
                        return None
                    parameters_ex = (param,)
                    break
                if param.kind is _VAR_KEYWORD or param.default is not _empty:
                    parameters_ex = (param,)
                    break
                # A required parameter with nothing to fill it.
                return None
            else:
                # A positional argument to place.
                try:
                    param = next(parameters)
                except StopIteration:
                    return None  # too many positional arguments
                if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                    return None  # too many positional arguments
                if param.kind is _VAR_POSITIONAL:
                    slots[index] = _VAR_POSITIONAL
                    extra_positional.append(index)
                    for extra_index, _ in arg_vals:
                        slots[extra_index] = _VAR_POSITIONAL
                        extra_positional.append(extra_index)
                    break
                if (
                    param.name in remaining
                    and param.kind is not _POSITIONAL_ONLY
                ):
                    return None  # multiple values for the same parameter
                slots[index] = param.name

        # The keyword phase: every parameter not filled positionally.
        varkw_present = False
        for param in itertools.chain(parameters_ex, parameters):
            if param.kind is _VAR_KEYWORD:
                varkw_present = True
                continue
            if param.kind is _VAR_POSITIONAL:
                continue
            if param.name in remaining:
                remaining.pop(param.name)
                if param.kind is _POSITIONAL_ONLY:
                    return None
                slots[param.name] = param.name
            elif param.default is _empty:
                return None  # missing a required argument
            else:
                defaulted.add(param.name)

        if remaining:
            if not varkw_present:
                return None  # unexpected keyword argument
            for name in remaining:
                slots[name] = _VAR_KEYWORD

        return Binding(
            slots,
            tuple(extra_positional),
            dict(remaining),
            frozenset(defaulted),
        )

    # -- applicability --------------------------------------------------

    def applies_to_values(
        self,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
    ) -> bool:
        """Report whether the signature accepts a call made with these values.

        Three things must all hold for this to succeed. The call has to
        bind in the first place; every bound value then has to be an
        instance of whichever hint it landed in, with an unannotated
        catch-all accepting anything at all; and any
        [`TypeVar`][typing.TypeVar] repeated across more than one
        parameter has to be solvable consistently across every value
        that reached it. A parameter left to its own default was never
        an argument of the call, so its hint plays no part in this
        check.

        !!! example
            ```pycon
            >>> sig = Signature.from_hints(int, y=str)
            >>> sig.applies_to_values((1,), {"y": "a"})
            True
            >>> sig.applies_to_values((1,), {"y": 2})
            False
            ```
        """
        self._settle()
        binding = self.bind(args, kwargs)
        if binding is None:
            return False
        groups: tx.Dict[int, tx.Tuple[tx.Any, tx.List[tx.Any]]] = {}
        for key, hint in self._iter_arguments(binding):
            value = args[key] if isinstance(key, int) else kwargs[key]
            # A `Callable` value is matched shallowly -- its own signature is
            # never inspected -- so a `ParamSpec` in the slot is not solved
            # from values here, only from hints in `applies_to_hints`. A
            # `*Ts` / `Tuple[..., *Ts]` slot is shallow the same way: a value
            # binds a `Tuple[int, *Ts]` by `isinstance(v, tuple)` alone, and a
            # `TypeVarTuple` is solved only at the hint level.
            if not ishintstance(value, hint):
                return False
            if _is_plain_typevar(hint):
                groups.setdefault(id(hint), (hint, []))[1].append(type(value))
        for hint, classes in groups.values():
            if not typevar_consistent(classes, hint):
                return False
        return True

    def applies_to_hints(
        self,
        hints: tx.Sequence[tx.Any],
        named_hints: tx.Mapping[str, tx.Any],
    ) -> bool:
        """Report whether the signature accepts a call described by hints.

        This mirrors [`applies_to_values`][] at the level of hints:
        the call still has to bind, and every query hint then has to be
        a sub-hint of whatever hint it landed in, with the same
        consistency required of a `TypeVar` repeated across positions.
        As a lookup convenience, described in RFC 0001 §4 and shared
        with [`resolve_hint`][bagof.dispatchers.core.resolve_hint], a
        slot typed as [`Exact`][bagof.dispatchers.Exact]`[C]` also
        accepts a query equivalent to plain `#!python C`, even though
        `#!python C` by itself is not actually a sub-hint of
        `#!python Exact[C]`.
        """
        self._settle()
        binding = self.bind(hints, named_hints)
        if binding is None:
            return False
        groups: tx.Dict[int, tx.Tuple[tx.Any, tx.List[tx.Any]]] = {}
        pgroups: tx.Dict[int, tx.List[tx.Any]] = {}
        tgroups: tx.Dict[int, tx.List[_TupleShape]] = {}
        for key, hint in self._iter_arguments(binding):
            query = hints[key] if isinstance(key, int) else named_hints[key]
            if not _hint_query_accepts(query, hint):
                return False
            if _is_plain_typevar(hint):
                groups.setdefault(id(hint), (hint, []))[1].append(query)
            # A `ParamSpec` named at several `Callable` slots must capture the
            # same parameter list at each, just as a repeated `TypeVar` must
            # agree on a class.
            for pspec, tail in paramspec_captures(query, hint):
                pgroups.setdefault(id(pspec), []).append(tail)
            # A `TypeVarTuple` named at several `Tuple` slots must capture the
            # same run at each -- the covariant tuple analogue.
            for tvt, run in typevartuple_captures(query, hint):
                tgroups.setdefault(id(tvt), []).append(run)
        # A `*args: *Ts` absorbs zero or more positionals into one run of the
        # same `Ts`, solved jointly with every `Tuple[..., *Ts]` slot. The run
        # is captured even when empty, so a `Ts` bound to `(int, str)` at a
        # `Tuple[int, str]` slot is inconsistent with the empty `*args` run.
        if _is_unpacked_typevartuple(self._varargs):
            tvt = tx.get_args(self._varargs)[0]
            run = _TupleShape(
                tuple(hints[i] for i in binding.extra_positional),
                None,
                (),
                None,
            )
            tgroups.setdefault(id(tvt), []).append(run)
        for hint, classes in groups.values():
            if not typevar_consistent(classes, hint):
                return False
        for tails in pgroups.values():
            if not paramspec_consistent(tails):
                return False
        for runs in tgroups.values():
            if not typevartuple_consistent(runs):
                return False
        return True

    def _iter_arguments(
        self, binding: Binding
    ) -> tx.Iterator[tx.Tuple[tx.Any, tx.Any]]:
        """Yield `(key, landed hint)` for each argument of a binding.

        Every keyword captured by a `#!python **kwargs: T` parameter
        lands on that one variable, so all of them join the same
        repeated-`TypeVar` solve that `#!python *args: T` and the
        explicitly named slots take part in.
        """
        for key, landed in binding.slots.items():
            if landed is _VAR_POSITIONAL:
                yield key, self._catch_all_hint(self._varargs)
            elif landed is _VAR_KEYWORD:
                yield key, self._catch_all_hint(self._varkw)
            else:
                yield key, self._parameters[landed].hint

    @staticmethod
    def _catch_all_hint(hint: tx.Any) -> tx.Any:
        """Treat an unannotated catch-all hint as `Any`, accepting anything."""
        return tx.Any if hint is None else hint

    # -- specificity ----------------------------------------------------

    def le(self, other: "Signature", shape: tx.Any) -> bool:
        """Report whether this signature is at least as specific as `other`.

        Both signatures are bound to the given call `shape`, and then,
        at every argument position, this signature's landed hint has to
        be a sub-hint of `other`'s landed hint there, written `A ⊑ B`.
        A signature unable to bind the shape at all is not comparable to
        the other one, and the answer comes back [`False`][] in that
        case.

        The repeated-`TypeVar` grouping refinement from RFC 0001 §3,
        where a method whose repeated `TypeVar`s tie strictly more
        arguments together than another's counts as more specific, is a
        separate selection step living on
        [`Function`][bagof.dispatchers.Function] rather than here. This
        method deliberately stays the plain, per-argument sub-hint
        comparison and leaves that refinement out.

        !!! example
            ```pycon
            >>> a = Signature.from_hints(int, int)
            >>> b = Signature.from_hints(int, object)
            >>> shape = Signature.shape((1, 2), {})
            >>> a.le(b, shape)
            True
            >>> b.le(a, shape)
            False
            ```
        """
        self._settle()
        other._settle()
        mine = self._bind_shape(shape)
        theirs = other._bind_shape(shape)
        if mine is None or theirs is None:
            return False
        my_hints = self._hints_by_key(mine)
        their_hints = other._hints_by_key(theirs)
        for key, hint in my_hints.items():
            if not issubhint(hint, their_hints[key]):
                return False
        return True

    def _bind_shape(self, shape: tx.Any) -> tx.Optional[Binding]:
        """Bind a bare call shape, filling every position with a
        placeholder.
        """
        count, names = shape
        args = (_PLACEHOLDER,) * count
        kwargs = {name: _PLACEHOLDER for name in names}
        return self.bind(args, kwargs)

    def _hints_by_key(self, binding: Binding) -> tx.Dict[tx.Any, tx.Any]:
        """Map each bound argument's key to the parameter hint it landed in."""
        result: tx.Dict[tx.Any, tx.Any] = {}
        for key, landed in binding.slots.items():
            if landed is _VAR_POSITIONAL:
                result[key] = self._catch_all_hint(self._varargs)
            elif landed is _VAR_KEYWORD:
                result[key] = self._catch_all_hint(self._varkw)
            else:
                result[key] = self._parameters[landed].hint
        return result

    def _full_shape(self) -> tx.Tuple[int, tx.Tuple[str, ...]]:
        """Find the shape of a call that would fill every parameter here.

        Positional-only and positional-or-keyword parameters are counted
        as filled by position, and keyword-only parameters by name. This
        is the shape implicitly used whenever two signatures are compared
        with `#!python <=` or `#!python <`.
        """
        return (
            len(self._pre),
            tuple(sorted(p.name for p in self._kwonly)),
        )

    def __le__(self, other: tx.Any) -> bool:
        if not isinstance(other, Signature):
            return NotImplemented
        return self.le(other, self._full_shape())

    def __lt__(self, other: tx.Any) -> bool:
        if not isinstance(other, Signature):
            return NotImplemented
        shape = self._full_shape()
        return self.le(other, shape) and not other.le(self, shape)

    # -- equality -------------------------------------------------------

    def __eq__(self, other: tx.Any) -> bool:
        if not isinstance(other, Signature):
            return NotImplemented
        # Resolve any deferred forward references first, so two signatures for
        # the same callable compare equal once one of them has been used. A
        # name that is still undefined is left deferred rather than raising:
        # equality answers a question about the signatures as written, and a
        # still-unresolved hint is then compared by its forward-reference name
        # (never fed to the sub-hint relation, which has no namespace for it).
        self._settle_quietly()
        other._settle_quietly()
        if list(self._parameters) != list(other._parameters):
            return False
        for mine, theirs in zip(
            self._parameters.values(), other._parameters.values()
        ):
            if mine != theirs:
                return False
        return self._catch_all_eq(
            self._varargs, other._varargs
        ) and self._catch_all_eq(self._varkw, other._varkw)

    def same_as(self, other: "Signature") -> bool:
        """Report whether two signatures were written the same way,
        structurally.

        `same_as` is stricter than [`==`][Signature.__eq__], which holds
        whenever two signatures accept and order calls identically
        regardless of how they were spelled. Under `==`,
        `#!python (x: T, y: T)` equals `#!python (x: T, y: U)`, since
        both reduce to `#!python (Any, Any)`, and
        `#!python Exact[int]` equals a `#!python TypeVar` bound to
        `#!python int`. `same_as` instead asks whether the two
        signatures were spelled identically: the same parameter names,
        kinds, and required-ness, with hints that match structurally,
        meaning [`TypeVar`][typing.TypeVar]s compared by identity,
        generic aliases compared by origin and arguments, and forward
        references compared by name.

        This is the comparison a registry uses to decide whether a new
        method replaces an existing one. Only a method registered with
        exactly the same spelling, the shape produced by a module reload
        or an accidentally doubled decorator, replaces what is already
        there, while two methods that merely happen to be equivalent
        despite being written differently are both kept.
        """
        self._settle_quietly()
        other._settle_quietly()
        if list(self._parameters) != list(other._parameters):
            return False
        for mine, theirs in zip(
            self._parameters.values(), other._parameters.values()
        ):
            if (
                mine.name != theirs.name
                or mine.kind != theirs.kind
                or mine.required != theirs.required
                or not _structural_hint_eq(mine.hint, theirs.hint)
            ):
                return False
        return self._catch_all_same(
            self._varargs, other._varargs
        ) and self._catch_all_same(self._varkw, other._varkw)

    @staticmethod
    def _catch_all_same(a: tx.Any, b: tx.Any) -> bool:
        """Report whether two `*args`/`**kwargs` hints were written the same
        way.
        """
        if (a is None) != (b is None):
            return False
        if a is None:
            return True
        return _structural_hint_eq(a, b)

    def _settle_quietly(self) -> None:
        """Settle deferred hints for an equality check, tolerating a bad name.

        [`_settle`][], which a dispatch use calls directly, raises when
        a name it needs never became defined. This calls it too but
        swallows that failure instead, leaving the signature deferred, so
        that comparing two signatures for equality never itself raises.
        """
        try:
            self._settle()
        except NameError:
            pass

    @staticmethod
    def _catch_all_eq(a: tx.Any, b: tx.Any) -> bool:
        """Report whether two `*args`/`**kwargs` hints match, `None`
        included.
        """
        if (a is None) != (b is None):
            return False
        if a is None:
            return True
        return _hint_eq(a, b)

    def __hash__(self) -> int:
        return hash(
            (
                tuple(
                    (p.name, p.kind, p.required)
                    for p in self._parameters.values()
                ),
                self._varargs is not None,
                self._varkw is not None,
            )
        )

    def __repr__(self) -> str:
        return f"Signature({_render_parameters(self)})"


class _ReadonlyMap(tx.Mapping):
    """A thin read-only view over an already-ordered mapping."""

    __slots__ = ("_data",)

    def __init__(self, data: tx.Dict[str, tx.Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> tx.Any:
        return self._data[key]

    def __iter__(self) -> tx.Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(dict(self._data))


# --- hint reading ------------------------------------------------------


def _forward_name(hint: tx.Any) -> tx.Optional[str]:
    """Find the forward-reference name a still-unresolved hint carries, if any.

    A hint kept as a raw string, a stringised annotation, or a
    [`ForwardRef`][typing.ForwardRef] object carries nothing but a name.
    Two such hints are therefore compared by that name alone; a hint
    that has already resolved carries no such name at all.
    """
    if isinstance(hint, str):
        return hint
    return getattr(hint, "__forward_arg__", None)


def _has_forward_ref(hint: tx.Any) -> bool:
    """Report whether a hint holds a forward reference anywhere inside it.

    A hint kept as a raw string, as a [`ForwardRef`][typing.ForwardRef],
    or as a generic carrying one at any depth, as in
    `#!python List["Later"]` or `#!python Optional["Node"]`, still counts
    as unresolved. Such a hint has no namespace behind it for the
    sub-hint relation to read, so it has to be compared structurally
    instead of through that relation.

    Every bare string this function reaches really is a forward
    reference, because the two sources of a merely decorative string are
    both guarded before that point: a [`Literal`][typing.Literal]
    returns early without ever descending into its members, and
    [`Annotated`][typing.Annotated] recurses only into its wrapped type,
    never into its metadata. So any other string found at any depth is a
    genuine reference, whether it is a
    [`ForwardRef`][typing.ForwardRef] or a bare string left unwrapped by
    a PEP 585 builtin generic such as `#!python list["Node"]` or by a
    `#!python Union` member.

    A [`Callable`][typing.Callable]'s parameter list arrives from
    [`get_args`][typing.get_args] as a plain `#!python list` rather than
    as a direct argument, so `#!python Callable[["X"], int]` becomes
    `#!python ([ForwardRef("X")], int)`, and such a list is descended
    into element by element to find a reference inside it.
    """
    if isinstance(hint, str):
        return True
    if getattr(hint, "__forward_arg__", None) is not None:
        return True
    if any(safe_get_origin(hint) is form for form in _LITERAL_FORMS):
        # A `Literal`'s arguments are values, never types or references.
        return False
    metadata = getattr(hint, "__metadata__", None)
    if metadata is not None:
        # `Annotated[T, ...]`: only the wrapped type `T` can carry a reference;
        # the metadata is arbitrary values, so it is not descended into.
        return _has_forward_ref(hint.__origin__)
    for arg in tx.get_args(hint):
        if isinstance(arg, list):
            # A `Callable` parameter list is a plain list of parameter hints.
            if any(_has_forward_ref(element) for element in arg):
                return True
        elif _has_forward_ref(arg):
            return True
    return False


def _hint_eq(a: tx.Any, b: tx.Any) -> bool:
    """Report whether two parameter hints are equivalent, forward
    references included.

    An unresolved name is never handed to the sub-hint relation, since
    the relation has no namespace to resolve it against and would
    otherwise treat it as [`Any`][typing.Any] by default. When either
    side is a top-level forward reference, the two hints are compared by
    name instead, so a raw string and a
    [`ForwardRef`][typing.ForwardRef] naming the same thing are found
    equal. When the forward reference sits nested inside a generic, as
    in `#!python List["Later"]` or `#!python Optional["Node"]`, the
    comparison is structural: a generic alias is compared by its origin
    and its arguments, and each nested `ForwardRef` by name, so two
    genuinely different spellings are not collapsed together. A
    reference's `__forward_module__` plays no part in this, so the same
    name recorded against two different modules still compares equal,
    matching the top-level case. Two hints that have both fully resolved
    are compared with [`equivalent`][] instead.
    """
    a_name = _forward_name(a)
    b_name = _forward_name(b)
    if a_name is not None or b_name is not None:
        return a_name == b_name
    if _has_forward_ref(a) or _has_forward_ref(b):
        # A nested forward reference has no namespace, so the two are compared
        # structurally rather than through the sub-hint relation. Canonicalise
        # each side's generic spelling first (`list["Zed"]` -> `List["Zed"]`)
        # so two spellings of the same type match, then compare structurally so
        # a bare-string reference equals a `ForwardRef` of the same name. A
        # plain `a == b` is spelling-sensitive on both counts.
        return _structural_hint_eq(_typing_spelling(a), _typing_spelling(b))
    return equivalent(a, b)


def _hint_query_accepts(query: tx.Any, hint: tx.Any) -> bool:
    """Report whether a hint-level query is accepted by a landed slot `hint`.

    In the ordinary case this is simply
    `#!python issubhint(query, hint)`. A slot typed as
    [`Exact`][bagof.dispatchers.Exact]`[C]` is additionally reachable by
    a query equivalent to `#!python C`, the RFC 0001 §4 lookup
    convenience kept consistent with
    [`resolve_hint`][bagof.dispatchers.core.resolve_hint]. This only
    widens which queries count as applicable; it leaves the sub-hint
    relation and the specificity order untouched, and both of those
    still compare `#!python Exact[C]` as the distinct leaf hint that it
    is.
    """
    if issubhint(query, hint):
        return True
    if is_exact(hint):
        inner = normalise_hint(exact_target(hint))
        return issubhint(query, inner) and issubhint(inner, query)
    return False


def _structural_hint_eq(a: tx.Any, b: tx.Any) -> bool:
    """Report whether two hints are the same as written, not merely equivalent.

    This is the stricter counterpart to [`_hint_eq`][]: it asks whether
    two hints share the same structure, never whether they happen to
    accept the same values. `#!python T` and `#!python U`, two distinct
    [`TypeVar`][typing.TypeVar]s, are therefore not equal here even
    though each is equivalent to [`Any`][typing.Any] on its own, and a
    bound `#!python TypeVar` is not equal to its own bound either. This
    distinction is what separates a genuine re-registration, with an
    identical spelling produced by a module reload or a doubled
    decorator, from two different methods that merely happen to be
    equivalent under the sub-hint relation.

    A forward reference is compared by its name, whether written as a
    raw string or as a [`ForwardRef`][typing.ForwardRef]. A
    `#!python TypeVar` is compared by identity. A generic alias is
    compared by its origin and its arguments, each of those compared
    the same way recursively, so `#!python List[int]` equals
    `#!python List[int]` but not `#!python List[str]`, and any
    `#!python Annotated` or `#!python Exact` metadata is compared as
    well. Everything else falls back to ordinary equality.
    """
    a, b = normalise_hint(a), normalise_hint(b)
    a_name, b_name = _forward_name(a), _forward_name(b)
    if a_name is not None or b_name is not None:
        return a_name == b_name
    if isinstance(a, tx.TypeVar) or isinstance(b, tx.TypeVar):
        # A TypeVar is the same only as itself: two variables with identical
        # bounds are still distinct positions in a signature.
        return a is b
    if isinstance(a, list) or isinstance(b, list):
        # A `Callable`'s parameter list arrives from `get_args` as a plain
        # `list` of parameter hints (`Callable[[int], str]` -> `([int], str)`).
        # Compare it element by element, canonicalising each element's spelling
        # so `Callable[[list[int]], int]` and `Callable[[List[int]], int]` are
        # the same as written. A `...`, `ParamSpec` or `Concatenate` parameter
        # list is not a `list`, so a list never matches one of those.
        if not (isinstance(a, list) and isinstance(b, list)):
            return False
        if len(a) != len(b):
            return False
        return all(
            _structural_hint_eq(x, y) for x, y in zip(a, b)
        )
    a_generic = tx.get_origin(a) is not None
    b_generic = tx.get_origin(b) is not None
    if a_generic != b_generic:
        return False
    if not a_generic:
        # A plain, non-generic argument: a class, `Any`, `None`, an
        # `Annotated` metadata object, or a `Literal` member value. The type
        # check keeps `Literal` members apart where `==` alone would not
        # (`1 == True` and `1 == 1.0` are both true, but the literals differ).
        if type(a) is not type(b):
            return False
        if isinstance(a, _Lower):
            # The lower bound of a `Between` is a hint, compared as one, so
            # `Between[List[int], object]` and `Between[list[int], object]`
            # are the same spelling.
            return _structural_hint_eq(a.lower, b.lower)
        try:
            return bool(a == b)
        except Exception:  # pragma: no cover  # noqa: BLE001
            # Defensive: a metadata object or literal member whose `==` raises.
            return a is b
    if safe_get_origin(a) is not safe_get_origin(b):
        return False
    if safe_get_origin(a) is tuple and (
        _is_subscripted_tuple(a) != _is_subscripted_tuple(b)
    ):
        # `Tuple[()]` (the empty-tuple type) and a bare `Tuple` both report
        # no arguments on 3.11+, so the arity check below cannot tell them
        # apart. Different signatures -- their subscripted-ness must agree.
        return False
    args_a = tx.get_args(a)
    args_b = tx.get_args(b)
    if len(args_a) != len(args_b):
        return False
    return all(
        _structural_hint_eq(x, y) for x, y in zip(args_a, args_b)
    )


def _is_plain_typevar(hint: tx.Any) -> bool:
    """Report whether `hint` is a plain `TypeVar`, excluding `TypeVarTuple`.

    Below Python 3.11, `typing_extensions` makes both
    `#!python isinstance(Unpack[Ts], tx.TypeVar)` and
    `#!python isinstance(Ts, tx.TypeVar)` come back `#!python True`, so a
    bare `#!python isinstance(hint, tx.TypeVar)` check would fold a
    `#!python *args: *Ts` tail into the repeated-`TypeVar` solve and
    wrongly reject a call whose types genuinely differ there. A
    `TypeVarTuple` has its own separate solving process, so it is
    excluded from this check.
    """
    return (
        isinstance(hint, tx.TypeVar)
        and not isinstance(hint, tx.TypeVarTuple)
        and not _is_unpacked_typevartuple(hint)
    )


def _catch_all_or_any(hint: tx.Any) -> tx.Any:
    """Normalise a `*args`/`**kwargs` hint, treating an unsupported form
    as `Any`.

    `*args: P.args`, `**kwargs: P.kwargs`, and
    `**kwargs: Unpack[TypedDict]` are all treated as an unannotated
    catch-all in this version of dispatch, following RFC 0001 §2.2, §3,
    and §11.1: since the tail accepts anything regardless, the hint
    simply becomes [`Any`][typing.Any].

    An unpacked `TypeVarTuple` written as `*args: *Ts` is the one
    exception, kept exactly as written. On its own it would behave just
    like [`Any`][typing.Any] for a single element. But the same `Ts` can
    also appear at a `#!python Tuple[..., *Ts]` slot elsewhere, where the
    two are solved jointly under RFC 0001 §3, so the tail has to stay
    identifiable rather than being collapsed away. Every other hint
    passes through unchanged.
    """
    if _is_unpacked_typevartuple(hint):
        return hint
    if _PARAMSPEC_ARGKW and isinstance(hint, _PARAMSPEC_ARGKW):
        return tx.Any
    origin = safe_get_origin(hint)
    if any(origin is form for form in _UNPACK_FORMS):
        return tx.Any
    return hint


def _reject_variadic_param(
    name: str, hint: tx.Any, fn: tx.Any, catch_all: bool = False
) -> None:
    """Refuse a variadic-only hint written where an ordinary value hint
    belongs.

    A [`ParamSpec`][typing.ParamSpec], a `#!python Concatenate[...]`, or
    a [`TypeVarTuple`][typing.TypeVarTuple] describes a
    `#!python Callable`'s parameter list or a `#!python Tuple`'s run of
    elements, never a value on its own. Written as a plain parameter's
    annotation, it describes nothing at all, so registration refuses it
    with a message naming the parameter.

    The `#!python *args: P.args`, `#!python **kwargs: P.kwargs`, and
    `#!python *args: Unpack[Ts]` forms are already turned into a
    catch-all before they would reach this check as a declared
    parameter, so they pass through here without complaint. Only a bare
    `#!python TypeVarTuple` written directly on `#!python *args` is
    refused, through the `catch_all` branch, with a message pointing at
    the `#!python *args: Unpack[Ts]` spelling it should have used
    instead.

    PEP 646 allows a single unpacked `#!python TypeVarTuple` per tuple or
    parameter list; a second open run in the same one is refused here,
    since `typing` itself does not reject that case at runtime.
    """
    if catch_all:
        # A `*args` slot: `*args: Unpack[Ts]` (a run) and `*args: P.args` (an
        # `Any` tail) are already accepted; a bare `TypeVarTuple` is not a
        # valid annotation on its own.
        if isinstance(hint, tx.TypeVarTuple):
            raise TypeError(
                f"*{name} of {fn}: a bare TypeVarTuple is not a valid "
                f"annotation; write *{name}: Unpack[Ts]"
            )
        return
    bad = isinstance(hint, tx.ParamSpec) or (
        bool(_PARAMSPEC_ARGKW) and isinstance(hint, _PARAMSPEC_ARGKW)
    )
    if not bad:
        origin = safe_get_origin(hint)
        bad = any(origin is form for form in _CONCATENATE_FORMS)
    if bad:
        raise TypeError(
            f"{name!r} of {fn}: a ParamSpec/Concatenate is only valid inside "
            "Callable[...] (or as *args: P.args / **kwargs: P.kwargs)"
        )
    if isinstance(hint, tx.TypeVarTuple) or _is_unpacked_typevartuple(hint):
        raise TypeError(
            f"{name!r} of {fn}: a TypeVarTuple is only valid unpacked inside "
            "Tuple[...] or Callable[[...], ...], or as *args: Unpack[Ts]"
        )
    if _has_two_open_runs(hint):
        raise TypeError(
            f"{name!r} of {fn}: a Tuple or parameter list may hold at most "
            "one unpacked TypeVarTuple (Unpack[Ts])"
        )


# Where `_misplaced_bound_in` meets a hint: the hint of a value parameter, the
# argument of `Type` or `Hint`, a union member or a `TypeVar` bound inside
# such an argument, a hint a bound is written with, or a type argument of a
# generic.
_CONCATENATE_FORMS = spellings("Concatenate")


def _misplaced_bound_in(
    hint: tx.Any, where: str = "value"
) -> tx.Optional[str]:
    """Explain what is wrong with a bound that `hint` carries, if anything.

    A `Super[C]` or a `Between[L, U]` can stand on a value parameter,
    either as its whole hint, as a member of its union, or as the bound
    of a `TypeVar` used there, and each bound it is written with must
    then be a hint that a class can be compared against. It can also
    stand as the whole argument of a `Type` or a `Hint` form, and as the
    whole type argument of a generic, where any hint can be a bound. At
    a slot of a generic, the variance of the slot decides whether the
    bound can be read there, exactly as the relation decides it when it
    compares the hint. Everywhere else a bound is refused, which covers
    a union member or a `TypeVar` bound inside the argument of `Type`,
    of `Hint` or of a generic's slot, a bound nested inside another
    bound, a constraint of a `TypeVar`, an element of a `Tuple`, and the
    signature of a `Callable`. An unsubscripted `Super`, `SuperType`,
    `SuperHint` or `Between` is refused wherever it appears.

    `where` says which of those positions `hint` stands in, and the walk
    carries it down into the hints `hint` is made of. The result is the
    message to raise for the first misplaced bound, or `#!python None`
    when there is nothing to report. Each `Between` is also re-checked for
    emptiness, which a bound written as a forward reference could only be
    checked for once it resolved. A forward-reference string is skipped,
    since it can only be checked once it resolves.
    """
    hint = normalise_hint(hint)
    if is_unbounded_form(hint):
        return unbounded_form_message(hint)
    if is_bound(hint):
        return _misplaced_in_bound(hint, where)
    member = "member" if where == "argument" else where
    if is_exact(hint):
        # `Exact` cannot hold a bound, any more than a bound can hold one.
        return _first_misplaced((exact_target(hint),), "endpoint")
    if isinstance(hint, tx.TypeVar):
        bound = getattr(hint, "__bound__", None)
        found = _first_misplaced(() if bound is None else (bound,), member)
        if found is not None:
            return found
        # A `ParamSpec` passes for a `TypeVar` on Python 3.8, with neither
        # a bound nor constraints.
        constraints = getattr(hint, "__constraints__", ())
        for constraint in constraints:
            inner = find_bound(constraint)
            if inner is not None:
                return misplaced_bound_message(
                    inner, constraint_bound_message
                )
        return _first_misplaced(constraints, member)
    args = get_args_uw(hint)
    origin = get_origin_uw(hint)
    if origin in UNION_TYPES:
        return _first_misplaced(args, member)
    if origin is type or is_hint_form(origin):
        return _first_misplaced(args, "argument")
    if any(origin is form for form in _CONCATENATE_FORMS):
        # A `Concatenate` prefix belongs to the `Callable` it is written in,
        # which has already read it.
        return _first_misplaced(args, "slot")
    if not args:
        return None
    # The arguments of any other generic are read by the relation's own
    # rule, and then each of them is walked for what it holds.
    found = arguments_bound_message(origin, args)
    if found is not None:
        return found
    return _first_misplaced(args, "slot")


def _misplaced_in_bound(bound: tx.Any, where: str) -> tx.Optional[str]:
    """Explain what is wrong with the bound `bound`, met in `where`."""
    if where == "member":
        return member_bound_message(bound)
    if where == "endpoint":
        return endpoint_bound_message(bound)
    if where == "value":
        found = value_bound_message(bound)
        if found is not None:
            return found
    found = _first_misplaced(written_ends(bound), "endpoint")
    if found is None and is_between(bound):
        found = empty_interval_message(*between_bounds(bound))
    return found


def _first_misplaced(
    args: tx.Sequence[tx.Any], where: str
) -> tx.Optional[str]:
    """Return the first message [`_misplaced_bound_in`][] gives for `args`.

    A `Callable`'s parameter list arrives as a plain list of hints, and
    each of its items is walked in the same position. A forward-reference
    string, and anything else that is not a hint, such as a `Literal`
    value, is skipped.
    """
    for arg in args:
        for item in arg if isinstance(arg, list) else (arg,):
            if isinstance(item, str) or not ishint(item):
                continue
            found = _misplaced_bound_in(item, where)
            if found is not None:
                return found
    return None


def _reject_misplaced_bound(
    name: str,
    hint: tx.Any,
    fn: tx.Any = None,
    subject: tx.Optional[str] = None,
) -> None:
    """Refuse a parameter whose hint writes a bound where it cannot stand.

    `Super[C]` and `Between[L, U]` describe a range of classes, read
    against the class of a value on a value parameter and against the
    class or hint passed inside `Type` or `Hint`. [`_misplaced_bound_in`][]
    lists the positions where a bound cannot be read that way, and
    registration refuses a bound in any of them with a message naming the
    parameter and the spelling to write instead. An interval that turns
    out to be empty once its forward references resolve is refused the
    same way. The relation refuses the same hints with the same messages,
    which covers callers that never register a method. `subject`
    overrides how the parameter is named, as for
    [`_reject_malformed_typeddict`][].
    """
    message = _misplaced_bound_in(hint)
    if message is not None:
        named = subject if subject is not None else repr(name)
        where = ""
        if fn is not None:
            where = f" of {getattr(fn, '__name__', fn)}"
        raise TypeError(f"{named}{where}: {message}")


def _top_level_typeddicts(hint: tx.Any) -> tx.List[tx.Any]:
    """Collect the `TypedDict`s a hint carries at its own top level.

    The result is the hint itself, once its
    [`Annotated`][typing.Annotated] wrapper is stripped, when the hint
    is a `TypedDict`, or each such member found in a top-level
    [`Union`][typing.Union] or [`Optional`][typing.Optional]. A
    parametrised generic `TypedDict` such as `#!python Movie[int]` is
    read through its origin, so a malformed generic is still caught even
    when it is used parametrised. A `TypedDict` buried inside a
    container, as in `#!python List[Movie]`, is left out, since a value
    at that slot binds against the container's own shape, not the
    `TypedDict`'s.
    """
    hint = unwrap(normalise_hint(hint))
    if safe_get_origin(hint) in UNION_TYPES:
        found: tx.List[tx.Any] = []
        for member in tx.get_args(hint):
            found.extend(_top_level_typeddicts(member))
        return found
    # A generic `TypedDict` used parametrised is a `_GenericAlias`, not a
    # class, so read its origin (`Movie[int]` -> `Movie`) before the class
    # check -- as the value-level TypedDict check resolves the origin too.
    origin = safe_get_origin(hint) or hint
    return [origin] if is_typeddict(origin) else []


def _reject_malformed_typeddict(
    name: str,
    hint: tx.Any,
    fn: tx.Any = None,
    subject: tx.Optional[str] = None,
) -> None:
    """Refuse a parameter whose hint is a malformed `TypedDict` (PEP 728).

    A `TypedDict` whose nominal hint ordering disagrees with its
    value-level shape check would make a method registered against it
    mis-dispatch, breaking the guarantee that `v in Sub` together with
    `Sub <= Base` implies `v in Base`. Registration refuses such a hint
    with a message naming both the parameter and the specific
    violation. A well-formed `TypedDict`, and any hint that is not a
    `TypedDict` at all, is accepted without complaint.

    `subject` overrides how the slot is named in that message, for a
    synthetic parameter with no name meant for a user to see, so the
    message reads `positional hint 0` rather than the internal name
    `'_0'`; otherwise the parameter's own name is shown.
    """
    for typeddict in _top_level_typeddicts(hint):
        reason = _malformed_typeddict_reason(typeddict)
        if reason is not None:
            named = subject if subject is not None else repr(name)
            where = ""
            if fn is not None:
                where = f" of {getattr(fn, '__name__', fn)}"
            raise TypeError(f"{named}{where}: {reason}")


def _has_two_open_runs(hint: tx.Any) -> bool:
    """Report whether any tuple or parameter list inside `hint` holds two
    or more unpacked runs.

    PEP 646 allows only a single unpacked
    [`TypeVarTuple`][typing.TypeVarTuple] per list, but `typing` does not
    enforce that at runtime on its own, so this walks `hint`'s arguments
    recursively to catch a violation such as
    `#!python Tuple[*Ts, *Us]` wherever it occurs, including nested
    inside a `#!python Callable` parameter list.
    """
    args = tx.get_args(hint)
    if not args:
        return False
    if sum(1 for arg in args if _is_unpacked_typevartuple(arg)) >= 2:
        return True
    for arg in args:
        if isinstance(arg, list):
            # A `Callable` parameter list is a plain list of its parameters.
            if sum(1 for e in arg if _is_unpacked_typevartuple(e)) >= 2:
                return True
            if any(_has_two_open_runs(e) for e in arg):
                return True
        elif _has_two_open_runs(arg):
            return True
    return False


def _hint_source(fn: tx.Callable[..., tx.Any]) -> tx.Any:
    """Find the object whose annotations actually describe `fn`'s parameters.

    [`get_type_hints`][typing_extensions.get_type_hints] reads
    annotations off a function, a method, or a module, but a class's
    parameters live on its constructor and a callable instance's live on
    its `#!python __call__`, neither of which is the object itself. Each
    of these is unwrapped here to whichever member's parameter names
    match what [`inspect.signature`][] reports for the original
    callable. A class unwraps to its `#!python __init__` (with
    `#!python self` dropped), or to its `#!python __new__` when
    `#!python __init__` is only the one inherited from
    `#!python object`, matching the constructor that
    `inspect.signature(cls)` itself reads; a callable instance unwraps to
    its type's `#!python __call__`; and a [`functools.partial`][]
    unwraps to the callable it wraps.
    """
    if isinstance(fn, functools.partial):
        return _hint_source(fn.func)
    if isinstance(fn, type):
        return _constructor_of(fn)
    if inspect.isfunction(fn) or inspect.ismethod(fn):
        return fn
    # A callable instance: its parameter annotations live on the class's
    # `__call__`, which is what `get_type_hints` can read.
    call = getattr(type(fn), "__call__", None)  # noqa: B004
    return call if call is not None else fn


def _constructor_of(cls: type) -> tx.Any:
    """Find the member whose annotations describe a class's constructor.

    This mirrors how [`inspect.signature`][] itself picks a class's
    signature: the class's own `#!python __init__` when it defines one,
    otherwise its own `#!python __new__` when it defines that, and
    otherwise the class itself, since a class built entirely by
    `#!python object`'s own construction takes no parameters worth
    dispatching on.
    """
    init = getattr(cls, "__init__", None)
    if init is not None and init is not object.__init__:
        return init
    new = getattr(cls, "__new__", None)
    if new is not None and new is not object.__new__:
        return new
    return cls


def _has_forward(raw: tx.Optional[tx.Dict[str, tx.Any]]) -> bool:
    """Report whether any raw annotation still holds a forward reference.

    A nested forward reference counts here too, so a stringised modern
    spelling such as `#!python Optional["list[int]"]`, written on an
    older Python that cannot evaluate it directly, defers as a whole
    rather than being read only partway.
    """
    return any(
        _has_forward_ref(value)
        for value in (raw or {}).values()
    )


def _resolve_hints(fn: tx.Callable[..., tx.Any]) -> tx.Dict[str, tx.Any]:
    """Resolve a callable's hints, preserving any `Annotated` metadata."""
    return dict(tx.get_type_hints(fn, include_extras=True))


def _raw_annotations(
    fn: tx.Callable[..., tx.Any],
) -> tx.Dict[str, tx.Any]:
    """Read a callable's raw, unresolved annotations, for deferral and
    rendering.

    On Python 3.14, annotations are evaluated lazily, so reading them the
    ordinary way can itself raise; the [`annotationlib`][]
    forward-reference format is tried as a fallback there, since it reads
    each annotation without evaluating the names inside it.
    """
    try:
        return dict(getattr(fn, "__annotations__", {}) or {})
    except Exception:  # noqa: BLE001 -- 3.14 lazy annotations may raise here
        pass
    try:
        import annotationlib  # type: ignore[import-not-found]

        return dict(
            annotationlib.get_annotations(
                fn, format=annotationlib.Format.FORWARDREF
            )
        )
    except Exception:  # noqa: BLE001 -- no annotations we can read
        return {}


def _hint_for(
    name: str,
    hints: tx.Optional[tx.Dict[str, tx.Any]],
    raw: tx.Optional[tx.Dict[str, tx.Any]],
    deferred: bool,
) -> tx.Any:
    """Find the hint for one parameter, resolved now or kept raw for later."""
    if deferred:
        return (raw or {}).get(name, tx.Any)
    return normalise_hint((hints or {}).get(name, tx.Any))


# --- rendering ---------------------------------------------------------


def _render_hint(hint: tx.Any) -> str:
    """Render a short, readable spelling of a hint for a signature's `repr`."""
    if isinstance(hint, str):
        return hint
    forward = getattr(hint, "__forward_arg__", None)
    if forward is not None:
        return forward
    if hint is tx.Any:
        return "Any"
    # An `Exact[C]` reads back as `Exact[C]`, not its `Annotated` spelling.
    if is_exact(hint):
        return f"Exact[{_render_hint(exact_target(hint))}]"
    if is_super(hint):
        return f"Super[{_render_hint(super_target(hint))}]"
    if is_between(hint):
        lower, upper = between_bounds(hint)
        return f"Between[{_render_hint(lower)}, {_render_hint(upper)}]"
    # `Type[Exact[C]]` and `Hint[Exact[C]]` render their argument recursively,
    # so a nested `Exact` reads as `Exact[C]`, not `Annotated[C, EXACT]`.
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)
    if origin is type and args:
        return f"Type[{_render_hint(args[0])}]"
    if origin is Hint and args:
        return f"Hint[{_render_hint(args[0])}]"
    if origin in UNION_TYPES and args:
        return _render_union(hint, args)
    if isinstance(hint, type):
        return hint.__name__
    text = str(hint)
    text = text.replace("typing_extensions.", "").replace("typing.", "")
    if args and "[" in text and _holds_a_bound(args):
        # A generic with a bound among its arguments, such as
        # `List[Super[int]]`: the name it was written with, then each
        # argument rendered the same way.
        return f"{text[:text.index('[')]}[{_render_arguments(args)}]"
    return text


def _holds_a_bound(args: tx.Sequence[tx.Any]) -> bool:
    """Report whether a `Super` or a `Between` appears anywhere among
    `args`, the arguments of a generic, however deeply nested.
    """
    for arg in args:
        if isinstance(arg, tx.TypeVar):
            # The `TypeVar` family first: on 3.8 the `ParamSpec` backport is a
            # `list` holding itself.
            continue
        if isinstance(arg, (list, tuple)):
            # A parameter list, of a `Callable` or of a `ParamSpec` generic.
            if _holds_a_bound(arg):
                return True
        elif is_bound(arg) or _holds_a_bound(get_args_uw(arg)):
            return True
    return False


def _render_arguments(args: tx.Sequence[tx.Any]) -> str:
    """Render the arguments of a generic for [`_render_hint`][].

    A parameter list, of a `Callable` or of a `ParamSpec` generic, is
    rendered in its brackets, and a `...` as written.
    """
    shown = []
    for arg in args:
        if isinstance(arg, (list, tuple)) and not isinstance(arg, tx.TypeVar):
            shown.append(f"[{_render_arguments(arg)}]")
        elif arg is Ellipsis:
            shown.append("...")
        else:
            shown.append(_render_hint(arg))
    return ", ".join(shown)


def _render_union(hint: tx.Any, args: tx.Tuple[tx.Any, ...]) -> str:
    """Render a union with each member rendered by `_render_hint`.

    The spelling follows the one the running interpreter gives the union,
    whether `Optional[X]`, `Union[X, Y]` or `X | Y`, so that only the
    members change: a `Super[C]`, `Between[L, U]` or `Exact[C]` among
    them reads back as such rather than as its `Annotated` spelling.
    """
    text = str(hint).replace("typing_extensions.", "").replace("typing.", "")
    members = [_render_hint(arg) for arg in args]
    # Python 3.14 prints every union as `X | Y`, so the two spellings below
    # are only produced, and only exercised by the tests, on earlier versions.
    if text.startswith("Optional["):  # pragma: no cover  -- Python < 3.14
        (member,) = [
            m for arg, m in zip(args, members) if arg is not type(None)
        ]
        return f"Optional[{member}]"
    if text.startswith("Union["):  # pragma: no cover  -- Python < 3.14
        return f"Union[{', '.join(members)}]"
    return " | ".join(members)


def _render_parameters(
    sig: "Signature",
    highlight: tx.Optional[tx.Collection[tx.Any]] = None,
) -> str:
    """Render a signature's parameters with `/`, `*`, `*args`, `**kwargs`.

    Every marker lands exactly where Python itself would place it: a `/`
    after the positional-only group, a bare `*` (or
    `#!python *args: H`) before the keyword-only group, and
    `#!python **kwargs: H` last of all.

    When `highlight` is given, every slot it names has its hint marked
    with a leading `#!python !`, as in `#!python x: !int`, pointing out
    the argument responsible for a dispatch error. A slot is named
    either by its parameter's own name or, for the catch-alls, by
    `Parameter.VAR_POSITIONAL` or `Parameter.VAR_KEYWORD`.
    """
    marked = frozenset(highlight) if highlight else frozenset()
    out: tx.List[str] = []
    positional_only = [
        p for p in sig._parameters.values() if p.kind is _POSITIONAL_ONLY
    ]
    positional_or_keyword = [
        p
        for p in sig._parameters.values()
        if p.kind is _POSITIONAL_OR_KEYWORD
    ]
    out.extend(
        _render_parameter(p, p.name in marked) for p in positional_only
    )
    if positional_only:
        out.append("/")
    out.extend(
        _render_parameter(p, p.name in marked)
        for p in positional_or_keyword
    )
    if sig._varargs is not None:
        out.append(
            _render_varargs(
                sig._varargs,
                sig._varargs_name or "args",
                _VAR_POSITIONAL in marked,
            )
        )
    elif sig._kwonly:
        out.append("*")
    out.extend(
        _render_parameter(p, p.name in marked) for p in sig._kwonly
    )
    if sig._varkw is not None:
        out.append(
            _render_varkw(
                sig._varkw,
                sig._varkw_name or "kwargs",
                _VAR_KEYWORD in marked,
            )
        )
    return ", ".join(out)


def _render_parameter(param: Parameter, mark: bool = False) -> str:
    bang = "!" if mark else ""
    text = f"{param.name}: {bang}{_render_hint(param.hint)}"
    if not param.required:
        text += f" = {param.default!r}"
    return text


def _render_varargs(
    hint: tx.Any, name: str = "args", mark: bool = False
) -> str:
    bang = "!" if mark else ""
    if hint is tx.Any:
        return f"*{name}{': !Any' if mark else ''}"
    return f"*{name}: {bang}{_render_hint(hint)}"


def _render_varkw(
    hint: tx.Any, name: str = "kwargs", mark: bool = False
) -> str:
    bang = "!" if mark else ""
    if hint is tx.Any:
        return f"**{name}{': !Any' if mark else ''}"
    return f"**{name}: {bang}{_render_hint(hint)}"
