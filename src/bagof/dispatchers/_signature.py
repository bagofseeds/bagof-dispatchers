"""Name-aware signatures, parameters and call binding.

A [`Signature`][bagof.dispatchers._signature.Signature] is what dispatch
compares. It records each parameter by **name**, its type hint, how it may be
passed (positional, keyword, …) and whether it has a default, together with the
`#!python *args` and `#!python **kwargs` hints. From that it can

* **bind** a call the way Python does
  ([`bind`][bagof.dispatchers._signature.Signature.bind]) -- working out which
  parameter each argument lands in, or that the call does not fit at all;
* say whether it **applies** to a set of argument *values*
  ([`applies_to_values`][bagof.dispatchers._signature.Signature.applies_to_values])
  or *hints*
  ([`applies_to_hints`][bagof.dispatchers._signature.Signature.applies_to_hints]);
* order two signatures by **specificity** for a given call shape
  ([`le`][bagof.dispatchers._signature.Signature.le]).

Binding follows [`inspect.Signature.bind`][] exactly, but the per-signature
plan is worked out once, when the signature is built, so a call binds without
rebuilding it each time.
"""

# stdlib
import functools
import inspect
import itertools

# dependencies
import typing_extensions as tx

# local
from ._lattice import equivalent, typevar_consistent
from .core import (
    ishintstance,
    issubhint,
    normalise_hint,
    safe_get_origin,
)
from .core._compat import spellings
from .core._exact import exact_target, is_exact

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
    """A stand-in for `*args` / `**kwargs` in the binding plan.

    It carries only what the binding loop reads off a parameter -- its kind,
    a name that never matches a keyword, and "no default" -- so the loop can
    treat it like any other entry.
    """

    __slots__ = ("kind",)
    name = None
    default = _empty

    def __init__(self, kind: tx.Any) -> None:
        self.kind = kind


_VARARGS = _CatchAll(_VAR_POSITIONAL)
_VARKW = _CatchAll(_VAR_KEYWORD)

# The `*args` / `**kwargs` variadic forms that v1 reads as an unannotated
# catch-all (RFC 0001 §2.2, §3, §11.1): an unpacked `TypeVarTuple` (`*Ts`), a
# `ParamSpec`'s `.args` / `.kwargs`, and an `Unpack[TypedDict]`. Each spelling
# only exists on newer Pythons or through `typing_extensions`, so every lookup
# is guarded.
_UNPACK_FORMS = spellings("Unpack")
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


class Parameter:
    """One parameter of a [`Signature`][bagof.dispatchers.Signature].

    A parameter is immutable and carries four things: its `name`, its type
    `hint`, its `kind` (how it may be passed) and its `default`. A parameter
    with no default is **required**.

    Parameters
    ----------
    name
        The parameter's name.
    hint
        The type hint dispatch reads. An unannotated parameter is given
        [`Any`][typing.Any], so it accepts anything.
    kind
        One of `Parameter.POSITIONAL_ONLY`,
        `Parameter.POSITIONAL_OR_KEYWORD` or `Parameter.KEYWORD_ONLY`,
        mirroring [`inspect.Parameter`][]. The `#!python *args` and
        `#!python **kwargs` hints live on the signature, not here.
    default
        The default value, or `Parameter.empty` when the parameter is
        required.

    Attributes
    ----------
    required : bool
        Whether the parameter has no default.

    !!! example
        ```pycon
        >>> p = Parameter("x", int, Parameter.POSITIONAL_OR_KEYWORD)
        >>> p.name, p.required
        ('x', True)
        ```
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
        """Whether the parameter has no default."""
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
    """The result of binding a call to a [`Signature`][].

    A binding records where every argument of a call landed. It is what
    [`bind`][bagof.dispatchers.Signature.bind] returns on success (and
    [`None`][] is returned on failure).

    Attributes
    ----------
    slots : Mapping
        Each argument's key -- an integer for a positional, the name for a
        keyword -- mapped to where it landed: a parameter name, or
        `Parameter.VAR_POSITIONAL` / `Parameter.VAR_KEYWORD` for an argument
        absorbed by `#!python *args` / `#!python **kwargs`.
    extra_positional : tuple
        The indices of positional arguments absorbed by `#!python *args`.
    extra_keywords : Mapping
        The keyword arguments absorbed by `#!python **kwargs`, by name.
    defaulted : frozenset
        The names of parameters left to their defaults. These are **not**
        arguments of the call, so dispatch ignores them.
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
    """A name-aware signature dispatch can bind and order.

    A signature holds its `parameters` in order, plus the `#!python *args`
    hint ([`varargs`][bagof.dispatchers.Signature.varargs]) and the
    `#!python **kwargs` hint
    ([`varkw`][bagof.dispatchers.Signature.varkw]) when the callable takes
    them. Build one from a callable with
    [`from_callable`][bagof.dispatchers.Signature.from_callable], or from
    hints alone with
    [`from_hints`][bagof.dispatchers.Signature.from_hints].

    Every declared parameter is dispatched -- an unannotated one on
    [`Any`][typing.Any], so it takes part but never narrows the choice.

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
        """Build a signature from parameters and the catch-all hints.

        Parameters
        ----------
        parameters
            The parameters, in order, keyed by name.
        varargs
            The `#!python *args` element hint, or [`None`][] when the
            callable takes no `#!python *args`. An unannotated `#!python
            *args` is [`Any`][typing.Any], not `#!python None`.
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
        """Read a signature off a callable.

        The parameter names, kinds and defaults come from
        [`inspect.signature`][], the hints from
        [`typing_extensions.get_type_hints`][] (keeping
        [`Annotated`][typing.Annotated] metadata, so `#!python Exact[...]`
        survives). A `#!python *args: H` becomes the signature's `varargs`
        and a `#!python **kwargs: H` its `varkw`; the return annotation is
        ignored. Any parameter without an annotation is dispatched on
        [`Any`][typing.Any].

        When a hint is a forward reference that cannot be resolved yet -- a
        name defined further down the module, or one only imported under
        `#!python TYPE_CHECKING` -- the signature keeps the raw annotations
        and resolves them the first time it is actually used for dispatch. If
        the name is still undefined then, a [`NameError`][] is raised.
        """
        isig = inspect.signature(fn)
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
        """Build a signature from hints alone, with no callable.

        This is the explicit form behind `#!python @dispatch(int,
        scale=float)`. Each positional hint becomes a positional-only
        parameter; each keyword hint a parameter of that name that may be
        passed either way. Every parameter is required.

        !!! example
            ```pycon
            >>> sig = Signature.from_hints(int, scale=float)
            >>> sig.dispatched_names
            ('scale',)
            ```
        """
        params = {}  # type: tx.Dict[str, Parameter]
        for index, hint in enumerate(hints):
            name = f"_{index}"
            params[name] = Parameter(
                name, normalise_hint(hint), _POSITIONAL_ONLY
            )
        for name, hint in named_hints.items():
            params[name] = Parameter(
                name, normalise_hint(hint), _POSITIONAL_OR_KEYWORD
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
        """Assemble a signature from an [`inspect.Signature`][] and hints."""
        params = {}  # type: tx.Dict[str, Parameter]
        varargs = None
        varkw = None
        varargs_name = None
        varkw_name = None
        for name, param in isig.parameters.items():
            hint = _hint_for(name, hints, raw, deferred)
            if param.kind is _VAR_POSITIONAL:
                varargs_name = name
                varargs = _catch_all_or_any(hint)
            elif param.kind is _VAR_KEYWORD:
                varkw_name = name
                varkw = _catch_all_or_any(hint)
            else:
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
        """Precompute the binding plan from the parameters.

        `_pre` are the parameters an argument may fill by position, in order;
        `_kwonly` the keyword-only ones; `_canonical` the full order the
        binder walks, with the `#!python *args` / `#!python **kwargs`
        stand-ins slotted where Python puts them.
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
        """Resolve deferred forward-reference hints, once.

        Called before any dispatch use. On failure the names are still
        undefined, which is a [`NameError`][] naming the callable.
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
        new_params = {}  # type: tx.Dict[str, Parameter]
        for name, param in self._parameters.items():
            hint = normalise_hint(hints.get(name, tx.Any))
            new_params[name] = Parameter(
                name, hint, param.kind, param.default
            )
        self._parameters = new_params
        if self._varargs_name is not None:
            self._varargs = _catch_all_or_any(
                normalise_hint(hints.get(self._varargs_name, tx.Any))
            )
        if self._varkw_name is not None:
            self._varkw = _catch_all_or_any(
                normalise_hint(hints.get(self._varkw_name, tx.Any))
            )
        # Build the plan before clearing the deferred flag: a reader on a
        # free-threaded build (3.13t) must never see `_deferred` false while
        # the plan still reflects the unresolved hints, so the flag is
        # published last.
        self._build_plan()
        self._deferred = False

    # -- public data ----------------------------------------------------

    @property
    def parameters(self) -> tx.Mapping[str, Parameter]:
        """The parameters, in order, keyed by name (read-only)."""
        return _ReadonlyMap(self._parameters)

    @property
    def varargs(self) -> tx.Any:
        """The `#!python *args` element hint, or [`None`][] if there is none.

        An unannotated `#!python *args` reads as [`Any`][typing.Any];
        `#!python None` means the callable takes no `#!python *args`.
        """
        return self._varargs

    @property
    def varkw(self) -> tx.Any:
        """The `#!python **kwargs` value hint, or [`None`][] if there is none.

        An unannotated `#!python **kwargs` reads as [`Any`][typing.Any];
        `#!python None` means the callable takes no `#!python **kwargs`.
        """
        return self._varkw

    @property
    def dispatched_names(self) -> tx.Tuple[str, ...]:
        """The names an argument may be dispatched on by keyword.

        These are the positional-or-keyword and keyword-only parameters, in
        order. A positional-only parameter is dispatched by position, so it
        is not named here.
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
        """The call shape: the number of positionals and the keyword names.

        The order of a signature over another is decided **per shape**,
        because which parameter each argument lands in depends on how the
        call is spelled. Two spellings of "the same" call -- `#!python
        f(1, 2)` and `#!python f(1, y=2)` -- are different shapes.
        """
        return (len(args), tuple(sorted(kwargs)))

    # -- binding --------------------------------------------------------

    def bind(
        self,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
    ) -> tx.Optional[Binding]:
        """Bind a call to the parameters, the way Python would.

        Returns a [`Binding`][bagof.dispatchers._signature.Binding] saying
        where each argument landed, or [`None`][] when the call does not fit
        -- too many positionals with no `#!python *args`, an unexpected
        keyword with no `#!python **kwargs`, a value given twice, or a
        required parameter left unfilled. This mirrors
        [`inspect.Signature.bind`][] exactly, from a plan worked out when the
        signature was built.

        !!! example
            ```pycon
            >>> sig = Signature.from_hints(int, y=int)
            >>> sig.bind((1,), {"y": 2}) is None
            False
            >>> sig.bind((1, 2, 3), {}) is None   # too many positionals
            True
            ```
        """
        slots = {}  # type: tx.Dict[tx.Any, tx.Any]
        extra_positional = []  # type: tx.List[int]
        remaining = dict(kwargs)
        defaulted = set()  # type: tx.Set[str]

        parameters = iter(self._canonical)
        arg_vals = enumerate(args)
        parameters_ex = ()  # type: tx.Tuple[tx.Any, ...]

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
        """Whether the signature accepts a call of these argument *values*.

        The call must bind, every bound value must be an instance of the
        hint it landed in (an unannotated catch-all accepting anything), and
        any repeated [`TypeVar`][typing.TypeVar] must be consistent across
        the values that reached it. Default-filled parameters are not
        arguments, so their hints are not checked.

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
        groups = {}  # type: tx.Dict[int, tx.Tuple[tx.Any, tx.List[tx.Any]]]
        for key, hint, groupable in self._iter_arguments(binding):
            value = args[key] if isinstance(key, int) else kwargs[key]
            if not ishintstance(value, hint):
                return False
            if groupable and isinstance(hint, tx.TypeVar):
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
        """Whether the signature accepts a call described by *hints*.

        The hint-level twin of
        [`applies_to_values`][bagof.dispatchers.Signature.applies_to_values]:
        the call must bind, and each query hint must be a sub-hint of the
        hint it landed in, with the same repeated-`TypeVar` consistency.
        """
        self._settle()
        binding = self.bind(hints, named_hints)
        if binding is None:
            return False
        groups = {}  # type: tx.Dict[int, tx.Tuple[tx.Any, tx.List[tx.Any]]]
        for key, hint, groupable in self._iter_arguments(binding):
            query = hints[key] if isinstance(key, int) else named_hints[key]
            if not issubhint(query, hint):
                return False
            if groupable and isinstance(hint, tx.TypeVar):
                groups.setdefault(id(hint), (hint, []))[1].append(query)
        for hint, classes in groups.values():
            if not typevar_consistent(classes, hint):
                return False
        return True

    def _iter_arguments(
        self, binding: Binding
    ) -> tx.Iterator[tx.Tuple[tx.Any, tx.Any, bool]]:
        """Yield `(key, landed hint, groupable)` for each bound argument.

        `groupable` is false for an argument absorbed by `#!python **kwargs`,
        which does not take part in repeated-`TypeVar` solving in v1.
        """
        for key, landed in binding.slots.items():
            if landed is _VAR_POSITIONAL:
                yield key, self._catch_all_hint(self._varargs), True
            elif landed is _VAR_KEYWORD:
                yield key, self._catch_all_hint(self._varkw), False
            else:
                yield key, self._parameters[landed].hint, True

    @staticmethod
    def _catch_all_hint(hint: tx.Any) -> tx.Any:
        """An unannotated catch-all accepts anything, so read it as `Any`."""
        return tx.Any if hint is None else hint

    # -- specificity ----------------------------------------------------

    def le(self, other: "Signature", shape: tx.Any) -> bool:
        """Whether this signature is at least as specific as *other*.

        For the given call `shape`, both signatures are bound and, for every
        argument, this signature's landed hint must be a sub-hint of
        *other*'s (`A ⊑ B`). A signature that cannot bind the shape is not
        comparable, so the answer is [`False`][].

        The repeated-`TypeVar` group-count tie-break (RFC 0001 §3) is Phase 7
        and is not applied here.

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
        """Bind a bare call shape, with placeholder arguments."""
        count, names = shape
        args = (_PLACEHOLDER,) * count
        kwargs = {name: _PLACEHOLDER for name in names}
        return self.bind(args, kwargs)

    def _hints_by_key(self, binding: Binding) -> tx.Dict[tx.Any, tx.Any]:
        """Map each bound argument's key to the hint it landed in."""
        result = {}  # type: tx.Dict[tx.Any, tx.Any]
        for key, landed in binding.slots.items():
            if landed is _VAR_POSITIONAL:
                result[key] = self._catch_all_hint(self._varargs)
            elif landed is _VAR_KEYWORD:
                result[key] = self._catch_all_hint(self._varkw)
            else:
                result[key] = self._parameters[landed].hint
        return result

    def _full_shape(self) -> tx.Tuple[int, tx.Tuple[str, ...]]:
        """The shape of a call that fills every parameter of this signature.

        Positional and positional-or-keyword parameters are passed by
        position, keyword-only ones by name. This is the shape implied when
        two signatures are compared with `#!python <=` / `#!python <`.
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

    def _settle_quietly(self) -> None:
        """Settle deferred hints for equality, swallowing an unresolved name.

        Unlike [`_settle`][bagof.dispatchers._signature.Signature._settle],
        which a dispatch use calls and which raises on a name that never
        became defined, this leaves an unresolvable signature deferred so
        that equality stays total.
        """
        try:
            self._settle()
        except NameError:
            pass

    @staticmethod
    def _catch_all_eq(a: tx.Any, b: tx.Any) -> bool:
        """Whether two `*args`/`**kwargs` hints match, `None` included."""
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
    """A tiny read-only view over an ordered mapping."""

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
    """The forward-reference name of a still-unresolved hint, or `None`.

    A hint kept as a raw string (a stringised annotation) or a
    [`ForwardRef`][typing.ForwardRef] carries only a name. Two such hints are
    compared by that name; a resolved hint has none.
    """
    if isinstance(hint, str):
        return hint
    return getattr(hint, "__forward_arg__", None)


def _has_forward_ref(hint: tx.Any, top_level: bool = True) -> bool:
    """Whether a hint holds a forward reference anywhere, however nested.

    A hint kept as a raw string, a [`ForwardRef`][typing.ForwardRef], or a
    generic that carries one at any depth (`#!python List["Later"]`,
    `#!python Optional["Node"]`) is still unresolved. Such a hint has no
    namespace behind it, so the sub-hint relation cannot read it and it is
    compared structurally instead.

    A bare string is a forward reference only as the *whole* hint. When
    recursing into a hint's arguments, [`typing`][] has already wrapped a
    genuine nested forward reference in a [`ForwardRef`][typing.ForwardRef];
    the only bare strings it leaves inside a hint are
    [`Literal`][typing.Literal] members and [`Annotated`][typing.Annotated]
    metadata, which are values rather than references. So a nested bare string
    is not read as a forward reference -- only a `ForwardRef` is -- and a
    `Literal`'s members and an `Annotated`'s metadata are never descended
    into.
    """
    if isinstance(hint, str):
        # A bare string is a reference only as the whole hint; nested, it is a
        # `Literal` member or `Annotated` metadata reached below, never a ref.
        return top_level
    if getattr(hint, "__forward_arg__", None) is not None:
        return True
    if any(safe_get_origin(hint) is form for form in _LITERAL_FORMS):
        # A `Literal`'s arguments are values, never types or references.
        return False
    metadata = getattr(hint, "__metadata__", None)
    if metadata is not None:
        # `Annotated[T, ...]`: only the wrapped type `T` can carry a reference;
        # the metadata is arbitrary values, so it is not descended into.
        return _has_forward_ref(hint.__origin__, top_level=False)
    return any(
        _has_forward_ref(arg, top_level=False) for arg in tx.get_args(hint)
    )


def _hint_eq(a: tx.Any, b: tx.Any) -> bool:
    """Whether two parameter hints are equivalent, forward references included.

    An unresolved name is never handed to the sub-hint relation, which has no
    namespace to resolve it and would treat the name as [`Any`][typing.Any].
    When either side is a top-level forward reference the two are compared by
    name, so a raw string and a [`ForwardRef`][typing.ForwardRef] naming the
    same thing match. When the forward reference is nested inside a generic
    (`#!python List["Later"]`, `#!python Optional["Node"]`) the two are
    compared structurally instead -- a generic alias compares by its origin and
    its arguments, and each nested `ForwardRef` by name, so two genuinely
    different spellings do not collapse to equal. Two fully resolved hints are
    compared with [`equivalent`][bagof.dispatchers._lattice.equivalent].
    """
    a_name = _forward_name(a)
    b_name = _forward_name(b)
    if a_name is not None or b_name is not None:
        return a_name == b_name
    if _has_forward_ref(a) or _has_forward_ref(b):
        return a == b
    return equivalent(a, b)


def _catch_all_or_any(hint: tx.Any) -> tx.Any:
    """A `*args`/`**kwargs` hint, with a v1-unsupported variadic form as `Any`.

    `*args: *Ts` (an unpacked `TypeVarTuple`), `*args: P.args`,
    `**kwargs: P.kwargs` and `**kwargs: Unpack[TypedDict]` all read as an
    unannotated catch-all in v1 (RFC 0001 §2.2, §3, §11.1): the tail takes
    anything, so the hint is [`Any`][typing.Any]. Every other hint is left
    unchanged.
    """
    if _PARAMSPEC_ARGKW and isinstance(hint, _PARAMSPEC_ARGKW):
        return tx.Any
    origin = safe_get_origin(hint)
    if any(origin is form for form in _UNPACK_FORMS):
        return tx.Any
    return hint


def _hint_source(fn: tx.Callable[..., tx.Any]) -> tx.Any:
    """The object whose annotations describe `fn`'s parameters.

    [`get_type_hints`][typing_extensions.get_type_hints] reads annotations off
    a function, method, class or module, but not off a
    [`functools.partial`][] or a callable instance. Those are unwrapped to the
    underlying function or the class's `#!python __call__`, whose parameter
    names still match the names
    [`inspect.signature`][] reports for the original callable.
    """
    if isinstance(fn, functools.partial):
        return _hint_source(fn.func)
    if (
        inspect.isfunction(fn)
        or inspect.ismethod(fn)
        or isinstance(fn, type)
    ):
        return fn
    # A callable instance: its parameter annotations live on the class's
    # `__call__`, which is what `get_type_hints` can read.
    call = getattr(type(fn), "__call__", None)  # noqa: B004
    return call if call is not None else fn


def _has_forward(raw: tx.Optional[tx.Dict[str, tx.Any]]) -> bool:
    """Whether any raw annotation still holds a forward reference.

    A nested forward reference counts too -- a stringised modern spelling like
    `#!python Optional["list[int]"]` on an older Python defers as a whole.
    """
    return any(
        _has_forward_ref(value)
        for value in (raw or {}).values()
    )


def _resolve_hints(fn: tx.Callable[..., tx.Any]) -> tx.Dict[str, tx.Any]:
    """Read a callable's hints, keeping `Annotated` metadata."""
    return dict(tx.get_type_hints(fn, include_extras=True))


def _raw_annotations(
    fn: tx.Callable[..., tx.Any],
) -> tx.Dict[str, tx.Any]:
    """The raw, unresolved annotations, for deferral and rendering.

    On Python 3.14 the annotations are lazy, so reading them the ordinary way
    may itself raise; the [`annotationlib`][] forward-ref format reads them
    without evaluating the names.
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
    """The hint for one parameter, resolved now or kept raw for later."""
    if deferred:
        return (raw or {}).get(name, tx.Any)
    return normalise_hint((hints or {}).get(name, tx.Any))


# --- rendering ---------------------------------------------------------


def _render_hint(hint: tx.Any) -> str:
    """A short, readable spelling of a hint for a signature's `repr`."""
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
    if isinstance(hint, type):
        return hint.__name__
    text = str(hint)
    return text.replace("typing_extensions.", "").replace("typing.", "")


def _render_parameters(sig: "Signature") -> str:
    """Render a signature's parameters with `/`, `*`, `*args`, `**kwargs`.

    The markers land where Python puts them: a `/` after the positional-only
    group, a bare `*` (or `#!python *args: H`) before the keyword-only group,
    and `#!python **kwargs: H` last.
    """
    out = []  # type: tx.List[str]
    positional_only = [
        p for p in sig._parameters.values() if p.kind is _POSITIONAL_ONLY
    ]
    positional_or_keyword = [
        p
        for p in sig._parameters.values()
        if p.kind is _POSITIONAL_OR_KEYWORD
    ]
    out.extend(_render_parameter(p) for p in positional_only)
    if positional_only:
        out.append("/")
    out.extend(_render_parameter(p) for p in positional_or_keyword)
    if sig._varargs is not None:
        out.append(_render_varargs(sig._varargs))
    elif sig._kwonly:
        out.append("*")
    out.extend(_render_parameter(p) for p in sig._kwonly)
    if sig._varkw is not None:
        out.append(_render_varkw(sig._varkw))
    return ", ".join(out)


def _render_parameter(param: Parameter) -> str:
    text = f"{param.name}: {_render_hint(param.hint)}"
    if not param.required:
        text += f" = {param.default!r}"
    return text


def _render_varargs(hint: tx.Any) -> str:
    if hint is tx.Any:
        return "*args"
    return f"*args: {_render_hint(hint)}"


def _render_varkw(hint: tx.Any) -> str:
    if hint is tx.Any:
        return "**kwargs"
    return f"**kwargs: {_render_hint(hint)}"
