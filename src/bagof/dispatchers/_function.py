"""The multiple-dispatch callable: [`Function`][].

A [`Function`][bagof.dispatchers._function.Function] is a named group of
[`Method`][bagof.dispatchers.Method]s. Calling it binds the call to each
method, keeps those that apply, and runs the most specific one; when two are
equally specific it raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError], and when none
applies [`NoMethodError`][bagof.dispatchers.NoMethodError].

Selection is name-aware (RFC 0001 §2.2): a call is bound the way Python binds
it, and specificity compares the hints of the slots the *same* argument landed
in. The most specific method is the one whose landed hints are a sub-hint of
every competitor's, position by position; ties are broken by explicit
`priority`, then the argument's own MRO, then how tightly the signature fits.

Registration is thread-safe and lock-free to read: each `register` builds a new
method tuple and publishes it in one assignment, so a concurrent call never
sees a half-updated group. Results are cached per call shape and per concrete
argument-type key, and the cache is dropped when the methods change or when an
[`abc.register`][abc.ABCMeta.register] elsewhere could change what
`#!python isinstance` answers.
"""

# stdlib
import abc
import functools
import inspect
import threading
import warnings

# dependencies
import typing_extensions as tx

# local
from . import _errors
from ._errors import AmbiguousMethodError, NoMethodError
from ._lattice import equivalent, is_value_dependent, mro_index
from ._method import Method
from ._signature import Parameter, Signature, _render_hint
from .core import UNSET, ishintstance, issubhint
from .core._exact import exact_target, is_exact

__all__ = ["Function"]

# A stand-in for "nothing here" that is distinct from every real cached value
# and from the caches' own emptiness.
_MISS = object()

# A cache-token value no `abc.get_cache_token()` ever returns, so a cache
# stamped with it is always seen as stale and rebuilt on first use.
_NO_TOKEN = object()

# A placeholder positional/keyword value used to bind a bare call *shape*.
_PLACEHOLDER = object()


class _Cache:
    """The two-level dispatch cache, stamped with what it is valid for.

    The cache is thrown away and rebuilt whole when the methods change or the
    ABC cache token moves, so an entry is never read against a state it was
    not computed for. Its two dicts are only ever mutated while holding the
    owning function's lock, and each individual read on the hot path is a
    single dict lookup -- atomic on every build, free-threaded included.
    """

    __slots__ = ("methods", "token", "shape_plans", "call_cache")

    def __init__(self, methods: tx.Tuple[Method, ...], token: tx.Any) -> None:
        self.methods = methods
        self.token = token
        self.shape_plans = {}  # type: tx.Dict[tx.Any, _Plan]
        self.call_cache = {}  # type: tx.Dict[tx.Any, Method]


class _Plan:
    """What a call *shape* fixes, independent of the argument values.

    Binding depends only on the shape (how many positionals, which keyword
    names), so it is worked out once per shape: which methods can bind it,
    where each argument lands in each, the pairwise specificity order, and
    which arguments a hint reads by value rather than by type.
    """

    __slots__ = ("bindable", "le_matrix", "value_dependent")

    def __init__(
        self,
        bindable: tx.Sequence[
            tx.Tuple[Method, tx.Any, tx.Dict[tx.Any, tx.Any]]
        ],
        le_matrix: tx.Dict[tx.Tuple[int, int], bool],
        value_dependent: tx.FrozenSet[tx.Any],
    ) -> None:
        # Each entry: (method, binding for the shape, {arg key: landed hint}).
        self.bindable = tuple(bindable)
        self.le_matrix = le_matrix
        self.value_dependent = value_dependent


class Function:
    """A named group of methods dispatched by argument type.

    Build one, register methods on it, then call it: the call runs the most
    specific method whose parameter types accept the arguments.

    Parameters
    ----------
    name
        The function's name, used in error messages. When omitted it is taken
        from the first method registered.

    Attributes
    ----------
    methods : tuple
        The registered methods, in registration order (read-only snapshot).

    !!! example
        ```pycon
        >>> f = Function("area")
        >>> _ = f.register(lambda shape: 3.14)     # a fallback for anything
        >>> def rect(w: int, h: int) -> int: return w * h
        >>> _ = f.register(rect)
        >>> f(3, 4)
        12
        ```
    """

    def __init__(self, name: tx.Optional[str] = None) -> None:
        self._name = name
        self._lock = threading.RLock()
        self._methods = ()  # type: tx.Tuple[Method, ...]
        self._cache = _Cache((), _NO_TOKEN)

    # -- public data ----------------------------------------------------

    @property
    def name(self) -> str:
        """The function's name."""
        return self._name if self._name is not None else "<function>"

    @property
    def methods(self) -> tx.Tuple[Method, ...]:
        """The registered methods, in registration order."""
        return self._methods

    # -- registration ---------------------------------------------------

    def register(
        self, *hints: tx.Any, priority: int = 0, **named_hints: tx.Any
    ) -> tx.Any:
        """Register a method, or return a decorator that does.

        Three spellings:

        * `#!python f.register(fn)` registers `fn`, reading its signature
          from its own annotations.
        * `#!python f.register(int, scale=float)` returns a decorator that
          registers the function it wraps, **overlaying** the given hints onto
          the function's own parameters -- `#!python int` onto the first
          parameter, `#!python scale=float` onto the parameter named `scale`
          -- keeping the function's parameter names, kinds and defaults.
        * `#!python f.register(priority=1)` (or either form above with
          `#!python priority=`) sets the tie-break priority.

        Registering a function whose signature matches one already registered
        replaces it, with a [`RuntimeWarning`][] -- the case a module reload
        or a doubled decorator produces.

        Returns
        -------
        Function or Callable
            The function itself when a method was registered directly (so
            `#!python f.register(fn)` and `#!python @f.register` both leave the
            name bound to the function), or the decorator otherwise.
        """
        if (
            len(hints) == 1
            and not named_hints
            and _is_implementation(hints[0])
        ):
            return self._add(Method(hints[0], priority=priority))

        def decorator(fn: tx.Callable[..., tx.Any]) -> "Function":
            signature = _overlay(fn, hints, named_hints)
            self._add(Method(fn, signature, priority=priority))
            return self

        return decorator

    @classmethod
    def from_mapping(
        cls,
        mapping: tx.Mapping[tx.Any, tx.Callable[..., tx.Any]],
        *,
        name: tx.Optional[str] = None,
    ) -> "Function":
        """Build a function from a mapping of hint-spec to callable.

        Each key describes a signature and each value is the callable to run
        for it. A key that is a tuple gives one positional hint per element; a
        single-hint key gives one positional hint; a
        [`Signature`][bagof.dispatchers.Signature] key is used as-is.

        !!! example
            ```pycon
            >>> f = Function.from_mapping({(int,): abs, (str,): len}, name="f")
            >>> f(-3), f("abcd")
            (3, 4)
            ```
        """
        function = cls(name=name)
        for key, impl in mapping.items():
            if isinstance(key, Signature):
                signature = key  # type: Signature
            elif isinstance(key, tuple):
                signature = Signature.from_hints(*key)
            else:
                signature = Signature.from_hints(key)
            function._add(Method(impl, signature))
        return function

    def _add(self, method: Method) -> "Function":
        """Add `method`, replacing an identical one, and drop the cache."""
        with self._lock:
            methods = _replace_or_append(self._methods, method)
            self._warn_new_ambiguities(methods, method)
            # Publish the methods tuple first, then invalidate the cache: a
            # reader that sees the new methods but the old cache finds the
            # cache stale (its methods are not the published tuple) and
            # rebuilds it, so the publish order is safe on a free-threaded
            # build either way round.
            self._methods = methods
            self._cache = _Cache(methods, _NO_TOKEN)
            if self._name is None:
                self._adopt_metadata(method.function)
        return method.function

    def _adopt_metadata(self, fn: tx.Callable[..., tx.Any]) -> None:
        """Take name and metadata from the first registered function."""
        self._name = getattr(fn, "__name__", None)
        try:
            functools.update_wrapper(self, fn, updated=())
        except (AttributeError, TypeError):  # pragma: no cover
            # Defensive: `update_wrapper` copies each attribute best-effort and
            # does not raise for the callables `register` accepts, but a truly
            # exotic one should not break registration -- the name is enough.
            pass

    def _warn_new_ambiguities(
        self, methods: tx.Tuple[Method, ...], method: Method
    ) -> None:
        """Warn when `method` is guaranteed to be ambiguous with another.

        Only guaranteed ambiguities are warned (RFC 0001 §5): a pair that
        binds the same shape, is incomparable, and whose landed hints are
        comparable at every argument -- so some call matches both. A pair that
        only clashes for a value neither is written for (a diamond subclass
        yet to exist) is left for the call to surface.
        """
        for other in methods:
            if other is method:
                continue
            shapes = _shapes_for_pair(method, other)
            if any(_pair_ambiguous(method, other, shape) for shape in shapes):
                warnings.warn(
                    f"{method.describe()} is ambiguous with "
                    f"{other.describe()}: a call matching both has no most "
                    f"specific method. Give one a higher priority, or make "
                    f"one more specific.",
                    RuntimeWarning,
                    stacklevel=4,
                )
                return

    # -- calling --------------------------------------------------------

    def __call__(self, *args: tx.Any, **kwargs: tx.Any) -> tx.Any:
        """Dispatch on the argument values and run the chosen method."""
        return self.dispatch(*args, **kwargs).function(*args, **kwargs)

    def dispatch(self, *args: tx.Any, **kwargs: tx.Any) -> Method:
        """Choose the method the argument *values* select, without calling it.

        Raises [`NoMethodError`][bagof.dispatchers.NoMethodError] when nothing
        applies, or
        [`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError] when
        two methods are equally specific.
        """
        token = abc.get_cache_token()
        cache = self._cache
        if cache.token != token or cache.methods is not self._methods:
            cache = self._refresh(token)
        shape = Signature.shape(args, kwargs)
        plan = cache.shape_plans.get(shape)
        if plan is not None:
            key = _call_key(args, kwargs, plan)
            if key is not None:
                try:
                    hit = cache.call_cache.get(key, _MISS)
                except TypeError:
                    hit = _MISS
                if hit is not _MISS:
                    return hit
        # A miss (or an as-yet-unplanned shape): resolve under the lock, where
        # the cache cannot be swapped out underneath the write.
        with self._lock:
            cache = self._ensure(token)
            plan = cache.shape_plans.get(shape)
            if plan is None:
                plan = self._build_plan(shape, cache)
            method = self._resolve_values(args, kwargs, plan)
            key = _call_key(args, kwargs, plan)
            if key is not None:
                try:
                    cache.call_cache[key] = method
                except TypeError:
                    # An unhashable value at a value-dependent argument -- this
                    # call cannot be a cache key, so it is simply not cached.
                    pass
            return method

    def resolve(
        self,
        *hints: tx.Any,
        default: tx.Any = UNSET,
        ambiguity: str = "raise",
        **named_hints: tx.Any,
    ) -> tx.Any:
        """Choose the method a call described by *hints* would select.

        The hint-level twin of
        [`dispatch`][bagof.dispatchers.Function.dispatch]: each argument is
        given as a type hint rather than a value, and selection uses the
        sub-hint relation. Returns the chosen
        [`Method`][bagof.dispatchers.Method].

        Parameters
        ----------
        default
            Returned when no method applies, instead of raising
            [`NoMethodError`][bagof.dispatchers.NoMethodError].
        ambiguity
            What to do when two methods are equally specific: `#!python
            "raise"` (the default) raises
            [`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError];
            `#!python "warn"` takes the first registered and warns; `#!python
            "ignore"` takes it silently.
        """
        token = abc.get_cache_token()
        with self._lock:
            cache = self._ensure(token)
            shape = Signature.shape(hints, named_hints)
            plan = cache.shape_plans.get(shape)
            if plan is None:
                plan = self._build_plan(shape, cache)
        applicable = [
            index
            for index, (method, _, _) in enumerate(plan.bindable)
            if method.signature.applies_to_hints(hints, named_hints)
        ]
        if not applicable:
            if default is not UNSET:
                return default
            raise self._no_method(hints, named_hints, values=False)
        maximal = _maximal(applicable, plan)
        winner = self._break_ties(
            maximal, plan, hints, named_hints, values=False
        )
        if winner is None:
            if ambiguity == "raise":
                raise self._ambiguous(
                    maximal, plan, hints, named_hints, values=False
                )
            winner = min(maximal)
            if ambiguity == "warn":
                warnings.warn(
                    f"{self._call_desc(hints, named_hints, values=False)} is "
                    f"ambiguous; taking the first method registered. Give one "
                    f"a higher priority to choose.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return plan.bindable[winner][0]

    def __get__(
        self, instance: tx.Any, owner: tx.Optional[type] = None
    ) -> tx.Any:
        """Bind the function as a method: `self` becomes argument 0.

        Accessed on an instance, it returns a bound view that prepends the
        instance to every call, so a `Function` used as a class attribute
        dispatches on `#!python self` (unannotated, so on
        [`Any`][typing.Any]) and the rest. Accessed on the class, it returns
        the function itself.
        """
        if instance is None:
            return self
        return _BoundFunction(self, instance)

    # -- diagnostics ----------------------------------------------------

    def ambiguities(self) -> tx.List[tx.Tuple[Method, Method]]:
        """The pairs of methods that could dispatch ambiguously.

        Each pair binds a common call shape, is incomparable, and has
        comparable hints at every argument of that shape -- so some call
        matches both with no most specific method. This is a heuristic over
        each method's own fully-applied shape (RFC 0001 §5): it finds the
        ambiguities a call written straightforwardly would hit, not every
        ambiguity reachable through `#!python *args` spreading or an
        unforeseen subclass.

        !!! example
            ```pycon
            >>> f = Function("g")
            >>> _ = f.register(float, object)
            >>> _ = f.register(object, float)
            >>> [(a.name, b.name) for a, b in f.ambiguities()]
            [('g', 'g')]
            ```
        """
        methods = self._methods
        pairs = []  # type: tx.List[tx.Tuple[Method, Method]]
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                first, second = methods[i], methods[j]
                shapes = _shapes_for_pair(first, second)
                if any(
                    _pair_ambiguous(first, second, shape) for shape in shapes
                ):
                    pairs.append((first, second))
        return pairs

    # -- cache management -----------------------------------------------

    def _ensure(self, token: tx.Any) -> _Cache:
        """Return a cache valid for `token` and the current methods.

        The caller holds the lock. A stale cache -- built for other methods or
        an older ABC token -- is replaced with a fresh empty one.
        """
        cache = self._cache
        if cache.token != token or cache.methods is not self._methods:
            cache = _Cache(self._methods, token)
            self._cache = cache
        return cache

    def _refresh(self, token: tx.Any) -> _Cache:
        """Take the lock and return a cache valid for `token`."""
        with self._lock:
            return self._ensure(token)

    def _build_plan(self, shape: tx.Any, cache: _Cache) -> _Plan:
        """Work out and store the plan for `shape` (caller holds the lock)."""
        existing = cache.shape_plans.get(shape)
        if existing is not None:
            return existing
        count, names = shape
        placeholder_args = (_PLACEHOLDER,) * count
        placeholder_kwargs = {name: _PLACEHOLDER for name in names}
        bindable = []  # type: tx.List[tx.Tuple[Method, tx.Any, tx.Dict]]
        for method in cache.methods:
            method.signature._settle()
            binding = method.signature.bind(
                placeholder_args, placeholder_kwargs
            )
            if binding is None:
                continue
            landed = _landed_hints(method.signature, binding)
            bindable.append((method, binding, landed))
        le_matrix = {}  # type: tx.Dict[tx.Tuple[int, int], bool]
        for i, (first, _, _) in enumerate(bindable):
            for j, (second, _, _) in enumerate(bindable):
                le_matrix[(i, j)] = first.signature.le(
                    second.signature, shape
                )
        value_dependent = set()  # type: tx.Set[tx.Any]
        for _, _, landed in bindable:
            for key, hint in landed.items():
                if is_value_dependent(hint):
                    value_dependent.add(key)
        plan = _Plan(bindable, le_matrix, frozenset(value_dependent))
        cache.shape_plans[shape] = plan
        return plan

    # -- resolution -----------------------------------------------------

    def _resolve_values(
        self,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
        plan: _Plan,
    ) -> Method:
        """Pick the most specific applicable method for a value call."""
        applicable = [
            index
            for index, (method, _, _) in enumerate(plan.bindable)
            if method.signature.applies_to_values(args, kwargs)
        ]
        if not applicable:
            raise self._no_method(args, kwargs, values=True)
        maximal = _maximal(applicable, plan)
        winner = self._break_ties(maximal, plan, args, kwargs, values=True)
        if winner is None:
            raise self._ambiguous(maximal, plan, args, kwargs, values=True)
        return plan.bindable[winner][0]

    def _break_ties(
        self,
        candidates: tx.List[int],
        plan: _Plan,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
        values: bool,
    ) -> tx.Optional[int]:
        """Reduce a set of equally specific methods to one, or `None`.

        Applied in order (RFC 0001 §2.2): explicit `priority` (higher wins),
        then -- for a value call -- the argument's own MRO (a hint naming a
        more derived base wins), then tightness (a signature absorbing fewer
        arguments into catch-alls, with fewer defaults, wins).
        """
        if len(candidates) == 1:
            return candidates[0]
        best_priority = max(
            plan.bindable[index][0].priority for index in candidates
        )
        candidates = [
            index
            for index in candidates
            if plan.bindable[index][0].priority == best_priority
        ]
        if len(candidates) == 1:
            return candidates[0]
        if values:
            candidates = [
                index
                for index in candidates
                if not any(
                    other != index
                    and self._mro_dominates(
                        plan, other, index, args, kwargs
                    )
                    for other in candidates
                )
            ]
            if len(candidates) == 1:
                return candidates[0]
        tightness = {
            index: _tightness(
                plan.bindable[index][1], plan.bindable[index][0].signature
            )
            for index in candidates
        }
        best = min(tightness.values())
        candidates = [
            index for index in candidates if tightness[index] == best
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _mro_dominates(
        self,
        plan: _Plan,
        winner: int,
        loser: int,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
    ) -> bool:
        """Whether method `winner`'s hints refine `loser`'s by argument MRO."""
        won = plan.bindable[winner][2]
        lost = plan.bindable[loser][2]
        if set(won) != set(lost):  # pragma: no cover
            # A defensive check: for one call both methods bind the same
            # argument keys, so this is never reached.
            return False
        strict = False
        for key in won:
            here, there = won[key], lost[key]
            value = args[key] if isinstance(key, int) else kwargs[key]
            value_type = type(value)
            a = mro_index(here, value_type)
            b = mro_index(there, value_type)
            if a is None or b is None:
                if not equivalent(here, there):
                    return False
                continue
            if a > b:
                return False
            if a < b:
                strict = True
        return strict

    # -- error building -------------------------------------------------

    def _no_method(
        self,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
        values: bool,
    ) -> NoMethodError:
        """A [`NoMethodError`][] for a call nothing applies to."""
        methods = self._methods
        call_desc = self._call_desc(args, kwargs, values)
        if not methods:
            return NoMethodError(
                _errors.render_no_method(self.name, call_desc, 0, ()),
                function=self.name,
                call=call_desc,
                candidates=(),
            )
        note = self._unknown_keyword_note(kwargs, methods)
        analyses = [
            _analyse_candidate(method, args, kwargs, values)
            for method in methods
        ]
        analyses.sort(key=lambda entry: (-entry[0], entry[1]))
        closest = [
            method.describe(highlight)
            + ("" if reason is None else f" -- {reason}")
            for _, _, method, highlight, reason in analyses[:8]
        ]
        return NoMethodError(
            _errors.render_no_method(
                self.name, call_desc, len(methods), closest, note
            ),
            function=self.name,
            call=call_desc,
            candidates=tuple(methods),
        )

    def _unknown_keyword_note(
        self,
        kwargs: tx.Mapping[str, tx.Any],
        methods: tx.Tuple[Method, ...],
    ) -> tx.Optional[str]:
        """A did-you-mean line when a keyword names no method's parameter."""
        names = set()  # type: tx.Set[str]
        takes_varkw = False
        for method in methods:
            names.update(method.signature.dispatched_names)
            if method.signature.varkw is not None:
                takes_varkw = True
        if takes_varkw:
            return None
        for keyword in kwargs:
            if keyword not in names:
                suggestion = _errors.did_you_mean(keyword, names)
                extra = (
                    f" Did you mean {suggestion!r}?" if suggestion else ""
                )
                return f"No method accepts a keyword {keyword!r}.{extra}"
        return None

    def _ambiguous(
        self,
        maximal: tx.List[int],
        plan: _Plan,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
        values: bool,
    ) -> AmbiguousMethodError:
        """An [`AmbiguousMethodError`][] for equally specific methods."""
        call_desc = self._call_desc(args, kwargs, values)
        candidates = [plan.bindable[index][0] for index in maximal]
        lines = [method.describe() for method in candidates]
        fix = self._possible_fix(maximal, plan, args, kwargs, values)
        return AmbiguousMethodError(
            _errors.render_ambiguous(call_desc, lines, fix),
            function=self.name,
            call=call_desc,
            candidates=tuple(candidates),
        )

    def _possible_fix(
        self,
        maximal: tx.List[int],
        plan: _Plan,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
        values: bool,
    ) -> str:
        """The signature of the call's own types -- always the tighter method.

        An argument a competitor matched with
        [`Exact`][bagof.dispatchers.Exact] is spelled `#!python Exact[...]`, so
        the suggested method wins over that competitor too.
        """
        prototype = plan.bindable[maximal[0]][1]
        parts = []  # type: tx.List[str]
        for index, value in enumerate(args):
            name = prototype.slots.get(index)
            label = name if isinstance(name, str) else f"a{index}"
            parts.append(
                f"{label}: "
                f"{self._fix_hint(index, plan, maximal, value, values)}"
            )
        for keyword in sorted(kwargs):
            value = kwargs[keyword]
            parts.append(
                f"{keyword}: "
                f"{self._fix_hint(keyword, plan, maximal, value, values)}"
            )
        return f"{self.name}({', '.join(parts)})"

    def _fix_hint(
        self,
        key: tx.Any,
        plan: _Plan,
        maximal: tx.List[int],
        value: tx.Any,
        values: bool,
    ) -> str:
        """The hint for one argument in a "possible fix" signature."""
        if values:
            value_type = type(value)
            base = value_type.__name__
            for index in maximal:
                landed = plan.bindable[index][2].get(key)
                if (
                    landed is not None
                    and is_exact(landed)
                    and equivalent(exact_target(landed), value_type)
                ):
                    return f"Exact[{base}]"
            return base
        return _render_hint(value)

    def _call_desc(
        self,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
        values: bool,
    ) -> str:
        """The call rendered by argument type (or hint), never by value."""
        parts = [
            (type(arg).__name__ if values else _render_hint(arg))
            for arg in args
        ]
        for keyword in sorted(kwargs):
            argument = kwargs[keyword]
            shown = (
                type(argument).__name__ if values else _render_hint(argument)
            )
            parts.append(f"{keyword}={shown}")
        return f"{self.name}({', '.join(parts)})"


# --- bound method view -------------------------------------------------


class _BoundFunction:
    """A [`Function`][] with a leading argument fixed, for use as a method."""

    __slots__ = ("_function", "_instance")

    def __init__(self, function: Function, instance: tx.Any) -> None:
        self._function = function
        self._instance = instance

    def __call__(self, *args: tx.Any, **kwargs: tx.Any) -> tx.Any:
        return self._function(self._instance, *args, **kwargs)

    def dispatch(self, *args: tx.Any, **kwargs: tx.Any) -> Method:
        """Choose the method for `self` and these arguments, uncalled."""
        return self._function.dispatch(self._instance, *args, **kwargs)

    def resolve(self, *hints: tx.Any, **kwargs: tx.Any) -> tx.Any:
        """Resolve at the hint level with `self`'s type as argument 0."""
        return self._function.resolve(
            type(self._instance), *hints, **kwargs
        )

    def __repr__(self) -> str:
        return (
            f"<bound {self._function.name} of {self._instance!r}>"
        )


# --- selection helpers -------------------------------------------------


def _maximal(applicable: tx.List[int], plan: _Plan) -> tx.List[int]:
    """The most specific applicable methods -- those with none below them.

    A method is kept when no other applicable method is *strictly* more
    specific than it under the per-shape order. A single survivor is the
    winner; more than one is a genuine ambiguity for the tie-breaks to settle.
    """
    matrix = plan.le_matrix
    return [
        index
        for index in applicable
        if not any(
            other != index
            and matrix[(other, index)]
            and not matrix[(index, other)]
            for other in applicable
        )
    ]


def _tightness(
    binding: tx.Any, signature: Signature
) -> tx.Tuple[int, int, int, int]:
    """How tightly a method fits, smaller being tighter (RFC 0001 §2.2).

    Ordered by: arguments absorbed by `#!python *args` / `#!python **kwargs`,
    then parameters left to defaults, then whether the signature has
    `#!python **kwargs` at all, then whether it has `#!python *args`.
    """
    absorbed = len(binding.extra_positional) + len(binding.extra_keywords)
    return (
        absorbed,
        len(binding.defaulted),
        1 if signature.varkw is not None else 0,
        1 if signature.varargs is not None else 0,
    )


def _shapes_for_pair(
    first: Method, second: Method
) -> tx.FrozenSet[tx.Any]:
    """The fully-applied shapes to test a pair of methods against.

    A method whose hints are still forward references cannot be shaped yet, so
    it contributes no shape -- the pair's ambiguity is left for first dispatch
    to surface rather than forcing the hints to resolve at registration.
    """
    shapes = set()  # type: tx.Set[tx.Any]
    for method in (first, second):
        try:
            shapes.add(_full_shape(method.signature))
        except NameError:
            pass
    return frozenset(shapes)


def _pair_ambiguous(first: Method, second: Method, shape: tx.Any) -> bool:
    """Whether two methods are guaranteed ambiguous for `shape`.

    Both must bind the shape, be incomparable, land their arguments on the
    same keys, and have comparable hints at every argument -- so the tuple
    that is most specific at each position matches both. A method with a hint
    still unresolved cannot be compared, so the pair is treated as not (yet)
    ambiguous.
    """
    try:
        return _pair_ambiguous_resolved(first, second, shape)
    except NameError:
        return False


def _pair_ambiguous_resolved(
    first: Method, second: Method, shape: tx.Any
) -> bool:
    """The body of [`_pair_ambiguous`][], assuming hints resolve."""
    first_binding = _bind_shape(first.signature, shape)
    second_binding = _bind_shape(second.signature, shape)
    if first_binding is None or second_binding is None:
        return False
    if first.signature.le(second.signature, shape) or second.signature.le(
        first.signature, shape
    ):
        return False
    first_landed = _landed_hints(first.signature, first_binding)
    second_landed = _landed_hints(second.signature, second_binding)
    if set(first_landed) != set(second_landed):  # pragma: no cover
        # Both methods bind the same shape, so they land the same argument
        # keys; this guards an invariant rather than a reachable case.
        return False
    for key in first_landed:
        here, there = first_landed[key], second_landed[key]
        if not (issubhint(here, there) or issubhint(there, here)):
            return False
    return True


def _bind_shape(signature: Signature, shape: tx.Any) -> tx.Any:
    """Bind a bare shape with placeholder arguments."""
    signature._settle()
    count, names = shape
    return signature.bind(
        (_PLACEHOLDER,) * count, {name: _PLACEHOLDER for name in names}
    )


def _full_shape(signature: Signature) -> tx.Tuple[int, tx.Tuple[str, ...]]:
    """The shape of a call filling every parameter of `signature`."""
    signature._settle()
    positional = 0
    keyword_only = []  # type: tx.List[str]
    for name, parameter in signature.parameters.items():
        if parameter.kind is Parameter.KEYWORD_ONLY:
            keyword_only.append(name)
        else:
            positional += 1
    return (positional, tuple(sorted(keyword_only)))


def _landed_hints(
    signature: Signature, binding: tx.Any
) -> tx.Dict[tx.Any, tx.Any]:
    """Map each bound argument's key to the hint it landed in."""
    result = {}  # type: tx.Dict[tx.Any, tx.Any]
    for key, slot in binding.slots.items():
        # A `*args` / `**kwargs` slot is only ever assigned when the signature
        # has that catch-all, so its hint is set (`Any` when unannotated).
        if slot is Parameter.VAR_POSITIONAL:
            result[key] = signature.varargs
        elif slot is Parameter.VAR_KEYWORD:
            result[key] = signature.varkw
        else:
            result[key] = signature.parameters[slot].hint
    return result


def _call_key(
    args: tx.Sequence[tx.Any],
    kwargs: tx.Mapping[str, tx.Any],
    plan: _Plan,
) -> tx.Optional[tx.Tuple[tx.Any, ...]]:
    """The cache key for a concrete call under a shape's plan.

    The type of each argument keys it, plus the value itself where the shape's
    hints read a value rather than a type (a `#!python Literal`, a
    `#!python type[...]`). A value-dependent argument whose value is unhashable
    cannot be a key, so the whole call is left uncached and the key is `None`.
    """
    value_dependent = plan.value_dependent
    parts = [len(args)]  # type: tx.List[tx.Any]
    for index, value in enumerate(args):
        if index in value_dependent:
            parts.append((type(value), _KeyValue(value)))
        else:
            parts.append(type(value))
    for keyword in sorted(kwargs):
        value = kwargs[keyword]
        if keyword in value_dependent:
            parts.append((keyword, type(value), _KeyValue(value)))
        else:
            parts.append((keyword, type(value)))
    return tuple(parts)


class _KeyValue:
    """A value wrapped so an unhashable one raises on use, not on build.

    The cache key holds a value only at a value-dependent argument. Wrapping it
    keeps [`hash`][hash] and equality delegating to the value, so two calls
    with the same literal share a key, while an unhashable value raises
    [`TypeError`][] from the surrounding `dict` access -- caught to leave the
    call uncached -- rather than from building the key tuple.
    """

    __slots__ = ("value",)

    def __init__(self, value: tx.Any) -> None:
        self.value = value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: tx.Any) -> bool:
        if not isinstance(other, _KeyValue):
            return NotImplemented
        return type(self.value) is type(other.value) and (
            self.value == other.value
        )


# --- candidate analysis for error messages ----------------------------


def _analyse_candidate(
    method: Method,
    args: tx.Sequence[tx.Any],
    kwargs: tx.Mapping[str, tx.Any],
    values: bool,
) -> tx.Tuple[int, int, Method, tx.Set[tx.Any], tx.Optional[str]]:
    """Score a method for a failed call, and say why it did not fit.

    Returns `(matched, arity_gap, method, highlight, reason)`: how many
    arguments matched (more is closer), how far its arity is from the call
    (nearer is closer), the method, the slots to mark with `#!python !`, and a
    binding-failure phrase when it could not bind at all.
    """
    signature = method.signature
    signature._settle()
    binding = signature.bind(args, kwargs)
    arity_gap = abs(len(signature.parameters) - (len(args) + len(kwargs)))
    if binding is None:
        return (-1, arity_gap, method, set(), _why_unbindable(
            signature, args, kwargs
        ))
    landed = _landed_hints(signature, binding)
    highlight = set()  # type: tx.Set[tx.Any]
    matched = 0
    for key, hint in landed.items():
        argument = args[key] if isinstance(key, int) else kwargs[key]
        ok = (
            ishintstance(argument, hint)
            if values
            else issubhint(argument, hint)
        )
        if ok:
            matched += 1
        else:
            highlight.add(binding.slots[key])
    return (matched, arity_gap, method, highlight, None)


def _why_unbindable(
    signature: Signature,
    args: tx.Sequence[tx.Any],
    kwargs: tx.Mapping[str, tx.Any],
) -> str:
    """A short phrase for why a call does not bind to `signature`."""
    parameters = signature.parameters
    positional_slots = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is not Parameter.KEYWORD_ONLY
    ]
    for keyword in kwargs:
        parameter = parameters.get(keyword)
        if parameter is None:
            if signature.varkw is None:
                return f"takes no keyword {keyword!r}"
        elif parameter.kind is Parameter.POSITIONAL_ONLY:
            return f"{keyword!r} is positional-only"
    if len(args) > len(positional_slots) and signature.varargs is None:
        count = len(positional_slots)
        plural = "" if count == 1 else "s"
        return f"takes {count} positional argument{plural}"
    filled = set(positional_slots[: len(args)])
    for name, parameter in parameters.items():
        if parameter.required and name not in filled and name not in kwargs:
            return f"missing {name!r}"
    # Everything is accounted for individually, yet the call still did not
    # bind -- the same argument was given twice (once positionally, once by
    # name).
    return "gives an argument twice"


# --- registration helpers ----------------------------------------------


def _is_implementation(candidate: tx.Any) -> bool:
    """Whether a single `register` argument is a function to register.

    A plain function, method, lambda or [`functools.partial`][] is an
    implementation; a type or typing construct is a *hint* the caller is
    overlaying, so it makes `register` a decorator instead.
    """
    return inspect.isroutine(candidate) or isinstance(
        candidate, functools.partial
    )


def _overlay(
    fn: tx.Callable[..., tx.Any],
    hints: tx.Tuple[tx.Any, ...],
    named_hints: tx.Mapping[str, tx.Any],
) -> Signature:
    """Overlay explicit hints onto a callable's own signature.

    Positional hints replace the first parameters' hints in order; named hints
    replace the hints of the parameters they name. Names, kinds and defaults
    are kept, so a call still binds the way the function's own parameters say.
    """
    base = Signature.from_callable(fn)
    base._settle()
    parameters = list(base.parameters.values())
    overridable = [
        parameter
        for parameter in parameters
        if parameter.kind is not Parameter.KEYWORD_ONLY
    ]
    if len(hints) > len(overridable):
        raise TypeError(
            f"{getattr(fn, '__name__', fn)} takes {len(overridable)} "
            f"positional parameter(s), but {len(hints)} hint(s) were given "
            f"to register it with."
        )
    replacements = {}  # type: tx.Dict[str, tx.Any]
    for parameter, hint in zip(overridable, hints):
        replacements[parameter.name] = hint
    for name, hint in named_hints.items():
        if name not in base.parameters:
            raise TypeError(
                f"{getattr(fn, '__name__', fn)} has no parameter {name!r} to "
                f"register a hint for."
            )
        replacements[name] = hint
    new_parameters = {}  # type: tx.Dict[str, Parameter]
    for parameter in parameters:
        hint = replacements.get(parameter.name, parameter.hint)
        new_parameters[parameter.name] = Parameter(
            parameter.name, hint, parameter.kind, parameter.default
        )
    return Signature(new_parameters, base.varargs, base.varkw)


def _replace_or_append(
    methods: tx.Tuple[Method, ...], method: Method
) -> tx.Tuple[Method, ...]:
    """`methods` with `method` added, replacing one of identical signature.

    A method whose signature matches one already registered replaces it in
    place, with a [`RuntimeWarning`][] -- the shape a module reload or a
    doubled decorator produces. A method with a new signature is appended.
    """
    for index, existing in enumerate(methods):
        if existing.signature == method.signature:
            warnings.warn(
                f"replacing an existing method {existing.describe()} with a "
                f"new one of the same signature.",
                RuntimeWarning,
                stacklevel=4,
            )
            return methods[:index] + (method,) + methods[index + 1 :]
    return methods + (method,)
