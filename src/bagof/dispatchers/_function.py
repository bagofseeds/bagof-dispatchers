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
`priority`, then the argument's own MRO, then how tightly the signature fits,
and finally by repeated `TypeVar`s -- a method whose repeated `TypeVar`s
strictly refine another's grouping (tie every pair it ties, and at least one
more) wins.

Registration is thread-safe and lock-free to read: each `register` builds a new
method tuple and publishes it in one assignment, so a concurrent call never
sees a half-updated group. Results are cached per call shape and per concrete
argument-type key, and the cache is dropped when the methods change or when an
[`abc.register`][abc.ABCMeta.register] elsewhere could change what
`#!python isinstance` answers.
"""

# stdlib
import abc
import collections.abc
import functools
import itertools
import threading
import warnings

# dependencies
import typing_extensions as tx

# local
from . import _errors
from ._errors import AmbiguousMethodError, NoMethodError
from ._lattice import equivalent, is_value_dependent, mro_index
from ._method import Method
from ._signature import (
    Parameter,
    Signature,
    _catch_all_or_any,
    _render_hint,
)
from .core import (
    UNSET,
    ishintstance,
    issubhint,
    normalise_hint,
    safe_get_origin,
)
from .core._compat import is_plausible_hint
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

# The most call keys one shape's plan caches before it starts evicting the
# oldest. A function called with unboundedly many distinct argument-type keys
# (many value-dependent literals, say) then keeps a bounded working set rather
# than growing without limit.
_CALL_CACHE_CAP = 1024


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
        # Re-entrant, not a plain lock: registration can re-enter itself. A
        # deferred forward reference resolved during selection evaluates the
        # annotation, which may import a module that registers another method;
        # and `ishintstance` can reach a user `__instancecheck__` /
        # `__subclasshook__` that dispatches back into this same function.
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

    def register(self, *args: tx.Any, **options: tx.Any) -> tx.Any:
        """Register a method, or return a decorator that does.

        `register` takes **either** an implementation **or** hints -- never
        both in a way that could be confused, since a type is both a callable
        and a valid hint:

        * **Implementation form** -- `#!python f.register(impl)`, where `impl`
          is any callable: a function, a **class** (dispatched on its
          `#!python __init__` / `#!python __new__`), or a callable instance.
          The signature is read from `impl` itself. `#!python f.register(int)`
          registers the `#!python int` **type** as an implementation,
          dispatched on its constructor -- it is *not* read as a hint.
        * **Hint-overlay form** -- the argument is a `#!python tuple` of
          positional hints and/or a `#!python dict` of named hints, and a
          decorator is returned that overlays those hints onto the wrapped
          function's own parameters, keeping its names, kinds and defaults:

            * `#!python @f.register((int, float))` -- positional hints (always
              a tuple, even for one: `#!python (int,)`);
            * `#!python @f.register({"scale": float})` -- named hints;
            * `#!python @f.register((int,), {"scale": float})` -- both;
            * `#!python @f.register()` -- no hints, register by the wrapped
              function's own signature.

        The form is chosen by the first argument's type: a `#!python tuple` or
        `#!python dict` is hints, anything else is the implementation, and no
        argument at all is the hint-overlay decorator with nothing to overlay.

        **Named hints go in the dict, never as keyword arguments.** A keyword
        argument to `register` is a registration *option* -- only `#!python
        priority` is understood -- so `#!python f.register(int, priority=5)`
        registers `#!python int` at priority 5, while `#!python
        f.register(scale=float)` is an error pointing to `#!python
        f.register({"scale": float})`.

        Parameters
        ----------
        priority
            A tie-break applied before the type-based order: a higher priority
            wins between two otherwise equally specific methods. Defaults to
            `0`. Given as a keyword, alongside either form.

        A method registered with the same signature *as written* as one already
        registered replaces it, with a [`RuntimeWarning`][] -- the case a
        module reload or a doubled decorator produces.

        Returns
        -------
        Callable
            The registered callable, in both forms -- so `#!python
            f.register(fn)`, `#!python @f.register` and `#!python
            @f.register((int,))` all leave the name bound to the function
            (the [`functools.singledispatch`][functools.singledispatch]
            convention).
        """
        priority = _registration_priority(options)
        if args and not isinstance(args[0], (tuple, dict)):
            # Implementation form: the first argument is the callable itself.
            impl = args[0]
            if len(args) > 1:
                raise TypeError(
                    "register(impl) takes a single implementation; to overlay "
                    "hints, pass them as a tuple and/or a dict -- "
                    "register((int, str)) or register({'x': int})."
                )
            if not callable(impl):
                raise TypeError(
                    f"register(...) expected a callable to register, or a "
                    f"tuple/dict of hints, but got {impl!r}."
                )
            return self._add(Method(impl, priority=priority))

        hints, named_hints = _split_hint_args(args)

        def decorator(fn: tx.Callable[..., tx.Any]) -> tx.Any:
            signature = _overlay(fn, hints, named_hints)
            return self._add(Method(fn, signature, priority=priority))

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
            first = not self._methods
            methods = _replace_or_append(self._methods, method)
            self._warn_new_ambiguities(methods, method)
            # Publish the methods tuple first, then invalidate the cache: a
            # reader that sees the new methods but the old cache finds the
            # cache stale (its methods are not the published tuple) and
            # rebuilds it, so the publish order is safe on a free-threaded
            # build either way round.
            self._methods = methods
            self._cache = _Cache(methods, _NO_TOKEN)
            if first:
                self._adopt_metadata(method.function)
        return method.function

    def _adopt_metadata(self, fn: tx.Callable[..., tx.Any]) -> None:
        """Take metadata from the first registered function.

        Copies the documentation, module and wrapped callable so the function
        stands in for its implementation to `#!python help` and introspection.
        A function reached by name already has that name; it is kept, so
        registering an anonymous `#!python def _` onto it does not rename it.
        Only a function that arrived without one takes its name from here.
        """
        given = self._name
        try:
            functools.update_wrapper(self, fn, updated=())
        except (AttributeError, TypeError):  # pragma: no cover
            # Defensive: `update_wrapper` copies each attribute best-effort and
            # does not raise for the callables `register` accepts, but a truly
            # exotic one should not break registration -- the name is enough.
            pass
        if given is not None:
            # Keep the name the function was reached by, not the first
            # implementation's, for both the property and introspection.
            self._name = given
            self.__name__ = given
            self.__qualname__ = given
        else:
            self._name = getattr(fn, "__name__", None)

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
            cache = self._refresh()
        shape = Signature.shape(args, kwargs)
        plan = cache.shape_plans.get(shape)
        if plan is not None:
            key = _call_key(args, kwargs, plan)
            try:
                hit = cache.call_cache.get(key, _MISS)
            except TypeError:
                # An unhashable value at a value-dependent argument makes the
                # key unhashable, so this call was never cached; resolve it.
                hit = _MISS
            if hit is not _MISS:
                return hit
        # A miss (or an as-yet-unplanned shape): resolve under the lock, where
        # the cache cannot be swapped out underneath the write.
        with self._lock:
            cache = self._ensure()
            plan = cache.shape_plans.get(shape)
            if plan is None:
                plan = self._build_plan(shape, cache)
            method = self._resolve_values(args, kwargs, plan)
            key = _call_key(args, kwargs, plan)
            _store_call(cache.call_cache, key, method)
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

        As a lookup convenience -- the same one
        [`resolve_hint`][bagof.dispatchers.core.resolve_hint] grants, and
        matching RFC 0001 §4 -- a method whose parameter is
        [`Exact`][bagof.dispatchers.Exact]`[C]` is reachable by a plain-`C`
        query, even though `#!python C` on its own is not a sub-hint of
        `#!python Exact[C]`. This does not change the sub-hint relation or the
        specificity order, only which methods a hint query counts as
        applicable.

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
        with self._lock:
            cache = self._ensure()
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
            >>> f = Function("area")
            >>> @f.register((float, object))
            ... def rank(a, b): return 1
            >>> @f.register((object, float))
            ... def order(a, b): return 2
            >>> [(a.name, b.name) for a, b in f.ambiguities()]
            [('rank', 'order')]
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

    def _ensure(self) -> _Cache:
        """Return a cache valid for the current ABC token and methods.

        The caller holds the lock. The ABC cache token is re-read here, under
        the lock, rather than trusted from a value read before it was taken: a
        token that advanced in between would otherwise stamp the fresh cache
        with a stale value and let the next reader serve it. A stale cache --
        built for other methods or an older token -- is replaced with a fresh
        empty one.
        """
        token = abc.get_cache_token()
        cache = self._cache
        if cache.token != token or cache.methods is not self._methods:
            cache = _Cache(self._methods, token)
            self._cache = cache
        return cache

    def _refresh(self) -> _Cache:
        """Take the lock and return a cache valid for the current token."""
        with self._lock:
            return self._ensure()

    def clear_cache(self) -> None:
        """Drop the dispatch cache, so the next call recomputes selection.

        The registered methods are untouched; only the cached shape plans and
        per-call results are discarded. Rarely needed -- registration and an
        [`abc.register`][abc.ABCMeta.register] elsewhere both invalidate the
        cache on their own -- but available for a value whose
        `#!python isinstance` behaviour has changed in a way the ABC cache
        token does not track.
        """
        with self._lock:
            self._cache = _Cache(self._methods, _NO_TOKEN)

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

        Applied in order (RFC 0001 §2.2, §3): explicit `priority` (higher
        wins), then -- for a value call -- the argument's own MRO (a hint
        naming a more derived base wins), then tightness (a signature absorbing
        fewer arguments into catch-alls, with fewer defaults, wins), and last
        the repeated-`TypeVar` refinement (a method whose repeated `TypeVar`s
        strictly refine another's grouping wins). A tie that
        survives every step leaves more than one candidate and is ambiguous
        (`None`).
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
        # Last, the repeated-TypeVar refinement (RFC 0001 §3): among methods
        # still tied, one whose repeated TypeVars constrain strictly more
        # arguments to a single consistent type is more specific. Drop any
        # candidate another refines this way. Refinement is a strict partial
        # order, so a single most-refined survivor wins and two incomparable
        # ones leave the tie -- and hence the ambiguity -- standing.
        candidates = [
            index
            for index in candidates
            if not any(
                other != index
                and self._group_dominates(plan, other, index)
                for other in candidates
            )
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _group_dominates(
        self, plan: _Plan, winner: int, loser: int
    ) -> bool:
        """Whether `winner` refines `loser` by repeated `TypeVar`s (§3).

        The last selection tie-break, reached only when two methods are
        otherwise equally specific. It compares the two methods' landed hints
        and their repeated-`TypeVar` groupings for the call's shape through
        [`_group_more_specific`][bagof.dispatchers._function._group_more_specific].
        """
        won_method, won_binding, won_landed = plan.bindable[winner]
        lost_method, lost_binding, lost_landed = plan.bindable[loser]
        won_partition = _typevar_partition(
            won_method.signature, won_binding
        )
        lost_partition = _typevar_partition(
            lost_method.signature, lost_binding
        )
        return _group_more_specific(
            won_landed, won_partition, lost_landed, lost_partition
        )

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
            # A refinement may never overrule a strict specificity win the
            # other way: if the loser's hint is strictly more specific at this
            # argument, the two conflict across arguments and must stay
            # ambiguous rather than be decided by MRO. This is what keeps
            # `Exact[int]/object` vs `int/int` ambiguous -- both read as `int`
            # at position 0 by MRO, so without this check the second would win
            # on position 1 alone (RFC 0001 §2.2: the refinements are partial
            # and none overrides a strict specificity win).
            if issubhint(there, here) and not issubhint(here, there):
                return False
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
    a_le = first.signature.le(second.signature, shape)
    b_le = second.signature.le(first.signature, shape)
    # A strict one-way order means one method is unambiguously more specific,
    # so the pair is not ambiguous. If neither holds (incomparable) or both
    # hold (equivalent yet spelled differently, so neither replaced the other
    # at registration), a call can match both with no most specific method --
    # carry on to confirm their hints are comparable at every argument.
    if a_le != b_le:
        return False
    # Neither is strictly more specific. If the signatures differ in tightness
    # -- one absorbs fewer arguments into a `*args` / `**kwargs`, or leans on
    # fewer defaults -- the tighter one always wins that tie-break, so the pair
    # is never actually ambiguous. Only an equal-tightness pair is guaranteed
    # ambiguous (a fixed-arity method beating a `*args` tail is not).
    if _tightness(first_binding, first.signature) != _tightness(
        second_binding, second.signature
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
    # The repeated-TypeVar tie-break (RFC 0001 §3) settles some otherwise-tied
    # pairs: when one method's repeated TypeVars constrain strictly more
    # arguments to a consistent type than the other's, that one is the more
    # specific and the pair is not ambiguous -- so it is neither warned at
    # registration nor listed by `ambiguities`.
    first_partition = _typevar_partition(first.signature, first_binding)
    second_partition = _typevar_partition(second.signature, second_binding)
    if _group_more_specific(
        first_landed, first_partition, second_landed, second_partition
    ) or _group_more_specific(
        second_landed, second_partition, first_landed, first_partition
    ):
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


def _typevar_partition(
    signature: Signature, binding: tx.Any
) -> tx.Dict[tx.Any, tx.Any]:
    """Group each bound argument by the repeated `TypeVar` it landed in.

    Returns a label per argument key, built from the same grouping
    applicability solves over (`_iter_arguments`, so a
    `#!python **kwargs`-absorbed argument does not group in v1). Two keys share
    a label only when they landed in the *same* groupable
    [`TypeVar`][typing.TypeVar]; a non-`TypeVar` argument gets a label unique
    to its key, so it forms a block of its own. The labels are only ever
    compared for equality, which is all the group tie-break needs.
    """
    labels = {}  # type: tx.Dict[tx.Any, tx.Any]
    for key, hint, groupable in signature._iter_arguments(binding):
        if groupable and isinstance(hint, tx.TypeVar):
            # Identity, not the variable itself: two distinct `TypeVar`s that
            # happen to be equal must land in different blocks.
            labels[key] = id(hint)
        else:
            labels[key] = ("solo", key)
    return labels


def _group_more_specific(
    a_landed: tx.Dict[tx.Any, tx.Any],
    a_partition: tx.Dict[tx.Any, tx.Any],
    b_landed: tx.Dict[tx.Any, tx.Any],
    b_partition: tx.Dict[tx.Any, tx.Any],
) -> bool:
    """Whether `a`'s repeated `TypeVar`s make it strictly more specific (§3).

    The repeated-`TypeVar` tie-break of RFC 0001 §3, reached only when two
    methods are already equally specific by every earlier measure. `a` wins
    when two conditions both hold:

    * **nothing else tells them apart** -- at every argument the two land
      *equivalent* hints, so neither is more specific there (an unbound
      `#!python T` and an unannotated `#!python Any` are equivalent, as are a
      bound `#!python TypeVar` and its bound); and
    * **`a` groups strictly more** -- every pair of arguments `b` ties to one
      repeated `TypeVar`, `a` ties too, and `a` ties at least one pair `b`
      leaves independent.

    Grouping more arguments to a single consistent type is the more
    constrained, so the more specific, reading. When neither method groups
    strictly more than the other -- equal groupings, or each grouping a pair
    the other does not -- the answer is [`False`][] both ways and the pair
    stays incomparable, hence ambiguous.
    """
    if set(a_landed) != set(b_landed):  # pragma: no cover
        # Both methods bind the same shape, so they land the same argument
        # keys; this guards an invariant rather than a reachable case.
        return False
    keys = list(a_landed)
    for key in keys:
        # "All else equal": a difference in the landed hint itself is settled
        # by the sub-hint order, not by this tie-break.
        if not equivalent(a_landed[key], b_landed[key]):
            return False
    strict = False
    for first, second in itertools.combinations(keys, 2):
        a_together = a_partition[first] == a_partition[second]
        b_together = b_partition[first] == b_partition[second]
        if b_together and not a_together:
            # `b` ties a pair `a` leaves independent, so `a` does not group a
            # superset of `b` -- `a` cannot be the strict refinement.
            return False
        if a_together and not b_together:
            strict = True
    return strict


def _store_call(
    call_cache: tx.Dict[tx.Any, Method], key: tx.Any, method: Method
) -> None:
    """Cache `method` under `key`, bounding the cache and skipping bad keys.

    The per-plan cache is capped: when it is full and the key is new, the
    oldest entry (dict insertion order) is evicted first, so a function called
    with unboundedly many distinct keys keeps only a bounded working set. A key
    that cannot be hashed -- an unhashable value at a value-dependent argument
    -- is simply not cached.
    """
    try:
        if key not in call_cache and len(call_cache) >= _CALL_CACHE_CAP:
            call_cache.pop(next(iter(call_cache)))
        call_cache[key] = method
    except TypeError:
        # An unhashable value-dependent argument: this call cannot be a key.
        pass


def _call_key(
    args: tx.Sequence[tx.Any],
    kwargs: tx.Mapping[str, tx.Any],
    plan: _Plan,
) -> tx.Tuple[tx.Any, ...]:
    """The cache key for a concrete call under a shape's plan.

    The type of each argument keys it, plus the value itself where the shape's
    hints read a value rather than a type (a `#!python Literal`, a
    `#!python type[...]`). The key tuple is always built; a value-dependent
    argument whose value is unhashable is wrapped so the tuple builds fine and
    the [`TypeError`][] surfaces only when the key is hashed (on a `dict`
    access), where the caller catches it and leaves the call uncached.
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
        if isinstance(self.value, collections.abc.Mapping):
            # A mapping at a value-dependent argument is a `TypedDict`-shape
            # position, where applicability is decided key by key with the
            # value types read type-aware. A mapping's own `==` is value-based
            # (`1 == 1.0 == True`), so keying by it would serve one shape for
            # a differently-typed one -- and a *hashable* mapping (a `dict`
            # subclass that defines `__hash__`, `frozendict`, ...) would slip
            # past the unhashable-value fallback and be cached wrongly. Refuse
            # to hash it, so the call falls through to "uncached" like a plain
            # `dict`. Nothing is lost for the other value-dependent kinds: a
            # mapping never satisfies a `Literal` or a `type[...]`.
            raise TypeError(
                "a mapping value cannot key the dispatch cache"
            )
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


def _registration_priority(options: tx.Dict[str, tx.Any]) -> int:
    """Pull `priority` out of the registration options, rejecting the rest.

    A keyword argument to `register` is a registration option, never a named
    hint -- those go in a dict. Only `priority` is understood; any other
    keyword is a mistake, named with a pointer to the dict form.
    """
    priority = options.pop("priority", 0)
    if options:
        unexpected = sorted(options)[0]
        shown = _render_hint(options[unexpected])
        raise TypeError(
            f"register() got an unexpected keyword {unexpected!r}. Keyword "
            f"arguments to register are options (priority=...), not hints; "
            f"pass named hints as a dict, e.g. "
            f"register({{{unexpected!r}: {shown}}})."
        )
    return priority


def _split_hint_args(
    args: tx.Tuple[tx.Any, ...],
) -> tx.Tuple[tx.Tuple[tx.Any, ...], tx.Dict[str, tx.Any]]:
    """Split the hint-overlay arguments into positional and named hints.

    The arguments are a `#!python tuple` of positional hints, a `#!python dict`
    of named hints, both, or neither. Anything else -- or two of the same kind
    -- is a caller error.
    """
    hints = ()  # type: tx.Tuple[tx.Any, ...]
    named = {}  # type: tx.Dict[str, tx.Any]
    seen_tuple = False
    seen_dict = False
    for arg in args:
        if isinstance(arg, tuple):
            if seen_tuple:
                raise TypeError(
                    "register(...) takes at most one tuple of positional "
                    "hints."
                )
            hints = arg
            seen_tuple = True
        elif isinstance(arg, dict):
            if seen_dict:
                raise TypeError(
                    "register(...) takes at most one dict of named hints."
                )
            named = arg
            seen_dict = True
        else:
            raise TypeError(
                f"register(...) hints must be given as a tuple (positional) "
                f"and/or a dict (named), but got {arg!r}."
            )
    return hints, named


def _overlay(
    fn: tx.Callable[..., tx.Any],
    hints: tx.Tuple[tx.Any, ...],
    named_hints: tx.Mapping[str, tx.Any],
) -> Signature:
    """Overlay explicit hints onto a callable's own signature.

    Positional hints replace the first parameters' hints in order; named hints
    replace the hints of the parameters they name, and may also name the
    `#!python *args` / `#!python **kwargs` catch-all to set its element / value
    hint. Names, kinds and defaults are kept, so a call still binds the way the
    function's own parameters say. Each hint is normalised and checked to be a
    real type hint, and a parameter may not be given a hint twice.
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
    varargs_name = base._varargs_name
    varkw_name = base._varkw_name
    replacements = {}  # type: tx.Dict[str, tx.Any]
    varargs_hint = base.varargs  # the *args element hint, or None
    varkw_hint = base.varkw  # the **kwargs value hint, or None
    for parameter, hint in zip(overridable, hints):
        replacements[parameter.name] = _overlay_hint(fn, parameter.name, hint)
    for name, hint in named_hints.items():
        checked = _overlay_hint(fn, name, hint)
        if name == varargs_name:
            varargs_hint = _catch_all_or_any(checked)
        elif name == varkw_name:
            varkw_hint = _catch_all_or_any(checked)
        elif name in base.parameters:
            if name in replacements:
                raise TypeError(
                    f"{getattr(fn, '__name__', fn)} is given a hint for "
                    f"{name!r} twice -- once by position and once by name. "
                    f"Give it just one."
                )
            replacements[name] = checked
        else:
            raise TypeError(
                f"{getattr(fn, '__name__', fn)} has no parameter {name!r} to "
                f"register a hint for."
            )
    new_parameters = {}  # type: tx.Dict[str, Parameter]
    for parameter in parameters:
        hint = replacements.get(parameter.name, parameter.hint)
        new_parameters[parameter.name] = Parameter(
            parameter.name, hint, parameter.kind, parameter.default
        )
    signature = Signature(new_parameters, varargs_hint, varkw_hint)
    # Keep the catch-alls' written names, so the method still renders as
    # `*items` / `**opts` rather than the generic `*args` / `**kwargs`.
    signature._varargs_name = varargs_name
    signature._varkw_name = varkw_name
    return signature


def _overlay_hint(
    fn: tx.Callable[..., tx.Any], target: str, hint: tx.Any
) -> tx.Any:
    """Normalise a registration hint and check it is a real type hint.

    A value that is not a type or typing construct -- a stray tuple, a number,
    a string -- would otherwise register a method that silently never matches,
    so it is refused at registration with a message naming the parameter. A
    parametrised form (`#!python Annotated[int, ...]`, `#!python Exact[int]`)
    is plausible through its origin even when the whole is not.
    """
    normalised = normalise_hint(hint)
    plausible = is_plausible_hint(normalised) or is_plausible_hint(
        safe_get_origin(normalised)
    )
    if not plausible:
        raise TypeError(
            f"register(...) got {hint!r} as the hint for {target!r} of "
            f"{getattr(fn, '__name__', fn)}, which is not a type or a typing "
            f"construct. Pass a type hint, such as int or List[int]."
        )
    return normalised


def _replace_or_append(
    methods: tx.Tuple[Method, ...], method: Method
) -> tx.Tuple[Method, ...]:
    """`methods` with `method` added, replacing one spelled the same way.

    A method whose signature is written the *same way* as one already
    registered -- same parameter names, kinds, required-ness and structurally
    equal hints -- replaces it in place, with a [`RuntimeWarning`][]; that is
    the shape a module reload or a doubled decorator produces. A method that is
    merely *equivalent* under the sub-hint relation but spelled differently
    (`#!python (x: T, y: T)` vs `#!python (x: T, y: U)`, `#!python Exact[int]`
    vs `#!python int`) is a distinct method and is appended -- selection then
    orders the two, and `_warn_new_ambiguities` flags them if they clash.
    """
    for index, existing in enumerate(methods):
        if existing.signature.same_as(method.signature):
            warnings.warn(
                f"replacing an existing method {existing.describe()} with a "
                f"new one of the same signature.",
                RuntimeWarning,
                stacklevel=4,
            )
            return methods[:index] + (method,) + methods[index + 1 :]
    return methods + (method,)
