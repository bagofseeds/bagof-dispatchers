"""The callable that dispatches a call across several methods: [`Function`][].

A [`Function`][] gathers a group of [`Method`][]s under one name. A call
to it is checked against every method in the group, narrowed down to
whichever ones actually accept that call, and then handed to the single
most specific one among those. A call that no method accepts raises
[`NoMethodError`][], and one that two methods accept equally well, with
neither more specific than the other, raises [`AmbiguousMethodError`][]
instead of an arbitrary pick.

Selection follows Python's own argument-binding rules before it ever
looks at types, matching RFC 0001 §2.2: a call is bound to each method
exactly as Python would bind it, and specificity then compares, position
by position, the hint each method assigns to the same argument. The
method that comes out most specific is the one whose hint at every
position is a sub-hint of every competing method's hint there. Ties
between equally specific methods are broken in a fixed order: an
explicit `priority` first, then the argument's own MRO, then how
tightly each signature fits the call, and finally a check for repeated
`TypeVar`s. In that last check, a method that ties strictly more
argument pairs together under one repeated `TypeVar` than another
method does is taken as the more constrained, and therefore the more
specific, choice.

Registering a method is thread-safe, and looking one up needs no lock
at all. Each call to `register` builds a whole new tuple of methods and
publishes it in a single assignment, so a lookup running at the same
time either sees the group before the change or after it, never midway
through. Results are cached, keyed by the shape of the call and then by
the concrete types (and, where needed, values) of its arguments, and the
cache is thrown away whenever the methods change or an
[`abc.register`][abc.ABCMeta.register] call anywhere in the program could
have changed what `#!python isinstance` reports.
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
from ._lattice import (
    equivalent,
    instance_members,
    is_declaration_dependent,
    is_value_dependent,
    overlaps,
)
from ._method import Method
from ._signature import (
    Parameter,
    Signature,
    _catch_all_or_any,
    _is_plain_typevar,
    _reject_bare_super,
    _reject_malformed_typeddict,
    _reject_variadic_param,
    _render_hint,
)
from .core import (
    UNSET,
    ishintstance,
    issubhint,
    mro_index,
    normalise_hint,
    safe_get_origin,
)
from .core._compat import SameObject as _SameObject
from .core._compat import is_plausible_hint
from .core._exact import exact_target, is_exact
from .core._introspect import _PEP585_ALIAS
from .core._relation import (
    _may_record_parametrisation,
    _present_data_members,
)

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
    """One dispatch cache, tagged with the state it was computed for.

    An instance of `_Cache` is discarded and replaced wholesale, rather
    than patched, whenever the function's methods change or the ABC
    cache token moves, so a lookup never reads a cache entry left over
    from before that change. Both of its dicts are only ever mutated
    while the owning function's lock is held, but reading one on the hot
    path is a single dict lookup, which is atomic regardless of build,
    including a free-threaded one.
    """

    __slots__ = ("methods", "token", "shape_plans", "call_cache")

    def __init__(self, methods: tx.Tuple[Method, ...], token: tx.Any) -> None:
        self.methods = methods
        self.token = token
        self.shape_plans = {}  # type: tx.Dict[tx.Any, _Plan]
        self.call_cache = {}  # type: tx.Dict[tx.Any, Method]


class _Plan:
    """Everything about a call shape that does not depend on argument values.

    Whether a method can bind a call, and how, depends only on the
    call's shape, meaning its number of positional arguments and its set
    of keyword names, so a `_Plan` is computed once per shape and then
    reused for every call sharing it. It records which methods can bind
    the shape, where each argument lands within each of those methods,
    and the specificity order between every pair of them. It also
    records which arguments need more than a bare type to key the cache
    correctly: some because their hint reads the value itself, some
    because it reads a parametrisation the value declared at
    construction, and some because it reads which protocol data members
    the value carries.
    """

    __slots__ = (
        "bindable",
        "le_matrix",
        "value_dependent",
        "declared",
        "members",
        "dependent",
        "value_only",
        "declared_only",
        "members_only",
    )

    def __init__(
        self,
        bindable: tx.Sequence[
            tx.Tuple[Method, tx.Any, tx.Dict[tx.Any, tx.Any]]
        ],
        le_matrix: tx.Dict[tx.Tuple[int, int], bool],
        value_dependent: tx.FrozenSet[tx.Any],
        declared: tx.FrozenSet[tx.Any] = frozenset(),
        members: tx.Optional[tx.Mapping[tx.Any, tx.Tuple[str, ...]]] = None,
    ) -> None:
        # Each entry: (method, binding for the shape, {arg key: landed hint}).
        self.bindable = tuple(bindable)
        self.le_matrix = le_matrix
        self.value_dependent = value_dependent
        # Argument keys keyed on `(type, declared parametrisation)`. A key in
        # both sets carries both: a value's own `==` need not tell two
        # parametrisations apart (a dataclass generic compares its fields).
        self.declared = declared
        # Argument key -> the protocol data members read off the value there,
        # sorted: the argument is keyed on which of them the value has. A key
        # here and in either set above carries every part.
        self.members = dict(members or {})
        # Every argument key whose part is more than the bare type.
        self.dependent = value_dependent | declared | frozenset(self.members)
        # The keys that read exactly one thing beside the type, which
        # `_call_key` builds inline; the rest go through `_dependent_part`.
        # Most dependent arguments read one, and the call cache is on the
        # path of every call.
        members_set = frozenset(self.members)
        self.value_only = value_dependent - declared - members_set
        self.declared_only = declared - value_dependent - members_set
        self.members_only = {
            key: names
            for key, names in self.members.items()
            if key not in value_dependent and key not in declared
        }


class Function:
    """A single named callable backed by several type-selected methods.

    A `Function` starts out empty. Each method registered on it adds one
    more implementation to choose from, and calling the `Function`
    itself picks whichever registered method's parameter types best fit
    the arguments given, then runs it.

    !!! example
        ```pycon
        >>> f = Function("area")
        >>> _ = f.register(lambda shape: 3.14)     # a fallback for anything
        >>> def rect(w: int, h: int) -> int: return w * h
        >>> _ = f.register(rect)
        >>> f(3, 4)
        12
        ```

    Parameters
    ----------
    name
        The function's name, shown in its error messages. When left
        unset, the name is taken instead from the first method
        registered on it.

    Attributes
    ----------
    methods : tuple
        The registered methods, in the order they were registered. This
        tuple is a read-only snapshot, not a live view.
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
        """The function's name, or a placeholder if none was ever given."""
        return self._name if self._name is not None else "<function>"

    @property
    def methods(self) -> tx.Tuple[Method, ...]:
        """A snapshot of the registered methods, in registration order."""
        return self._methods

    # -- registration ---------------------------------------------------

    def register(self, *args: tx.Any, **options: tx.Any) -> tx.Any:
        """Register a method, or return a decorator that does.

        `register` is called in one of two distinct forms, decided by
        its first argument, since a type is both a callable and a valid
        hint and the two forms are never mixed together in one call.

        The implementation form, `#!python f.register(impl)`, takes any
        callable as `impl`: a plain function, a class (dispatched on its
        `#!python __init__` or `#!python __new__`), or an arbitrary
        callable instance. Its signature is read straight off `impl`, so
        `#!python f.register(int)` registers `#!python int` itself as an
        implementation dispatched on its constructor, rather than
        reading `int` as a hint.

        The hint-overlay form instead takes a `#!python tuple` of
        positional hints, a `#!python dict` of named hints, or both, and
        returns a decorator. That decorator overlays the given hints onto
        the wrapped function's own parameters when it is applied,
        keeping that function's names, parameter kinds, and defaults
        unchanged. `#!python @f.register((int, float))` supplies
        positional hints, always as a tuple even when there is only one,
        as in `#!python (int,)`; `#!python @f.register({"scale": float})`
        supplies named hints; `#!python @f.register((int,), {"scale": float})`
        supplies both together; and `#!python @f.register()` supplies
        none at all, registering the wrapped function under its own
        existing signature.

        Which form applies is decided purely by the type of the first
        argument: a `#!python tuple` or `#!python dict` selects the
        hint-overlay form, any other value selects the implementation
        form, and no argument at all gives the hint-overlay decorator
        with nothing to overlay.

        Named hints always belong in the dict, never as keyword
        arguments to `register` itself. A keyword argument there is a
        registration option instead, and the only one currently
        understood is `#!python priority`, so
        `#!python f.register(int, priority=5)` registers
        `#!python int` at priority 5, while
        `#!python f.register(scale=float)` is rejected with an error
        pointing at `#!python f.register({"scale": float})` instead.

        Registering a method whose signature, as written, exactly
        matches one already registered replaces that earlier method and
        issues a [`RuntimeWarning`][]. This is the situation a module
        reload or an accidentally doubled decorator produces.

        Parameters
        ----------
        priority
            A tie-break considered ahead of type-based specificity:
            between two methods a call would otherwise match equally
            well, the one with the higher priority is chosen. Defaults
            to `0`, and is given as a keyword alongside either form.

        Returns
        -------
        Callable
            The registered callable, returned the same way in both
            forms, so that `#!python f.register(fn)`,
            `#!python @f.register`, and `#!python @f.register((int,))`
            all leave the decorated name bound to the dispatched
            function rather than to the plain implementation, following
            the [`functools.singledispatch`][functools.singledispatch]
            convention.
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
        """Build a function whose methods come straight from a mapping.

        Each key in `mapping` describes one method's signature, and the
        value paired with it is the callable that method runs. A tuple
        key supplies one positional hint per element; a single hint used
        as a key by itself supplies one positional hint; and a
        [`Signature`][] given directly as a key is used exactly as it
        is.

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
        """Add `method` to the group, replacing an identically shaped one.

        Adding a method invalidates the dispatch cache, since the set of
        candidates it was built from has changed. The wrapped callable is
        returned, not the `Method` itself, so that a decorator applying
        `register` leaves the original name bound to that callable.
        """
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
        """Adopt the first registered callable's introspection metadata.

        Copying the docstring, module, and wrapped-callable reference
        over lets the `Function` stand in for that first implementation
        under `#!python help` and other introspection. A `Function`
        reached by name already has that name and keeps it, so
        registering an anonymous `#!python def _` onto it afterward does
        not rename it; only a `Function` that had no name to begin with
        takes one from the callable here.
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

        Only a guaranteed ambiguity is warned about here, following RFC
        0001 §5. Such a pair binds the same call shape, is incomparable
        once priority is equal, and lands comparable hints at every one
        of that shape's arguments, so that some call is bound to match
        both with nothing to choose between them. A pair separated by a
        differing `priority` is resolved deterministically whenever it
        is actually called, so no warning is raised for it. A pair that
        would only clash for a value neither method was written with in
        mind, such as a diamond subclass that does not exist yet, is
        left for an actual call to surface instead.
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
        """Select a method by the arguments' runtime types, then run it."""
        return self.dispatch(*args, **kwargs).function(*args, **kwargs)

    def dispatch(self, *args: tx.Any, **kwargs: tx.Any) -> Method:
        """Select the method these arguments call for, without running it.

        [`NoMethodError`][] is raised when no method accepts the call,
        and [`AmbiguousMethodError`][] when two methods accept it equally
        well.
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
        """Select the method that a call described by hints would call for.

        `resolve` mirrors [`dispatch`][] at the level of hints rather
        than values: each argument is given as a type hint standing for
        a whole range of possible values, and selection compares those
        hints through the sub-hint relation instead of checking actual
        arguments. It returns the [`Method`][] selected this way.

        As a lookup convenience, the same one that
        [`resolve_hint`][bagof.dispatchers.core.resolve_hint] grants and
        matching RFC 0001 §4, a method whose parameter is
        [`Exact`][bagof.dispatchers.Exact]`[C]` can also be reached by a
        plain `C` query, even though `#!python C` on its own is not
        actually a sub-hint of `#!python Exact[C]`. This convenience
        changes which methods a hint query is considered to reach; it
        changes neither the sub-hint relation itself nor the specificity
        order between methods.

        Parameters
        ----------
        default
            Returned instead of raising [`NoMethodError`][] when no
            method applies to the given hints.
        ambiguity
            What to do when two methods are equally specific.
            `#!python "raise"`, the default, raises
            [`AmbiguousMethodError`][]. `#!python "warn"` instead takes
            whichever method was registered first and warns about the
            choice. `#!python "ignore"` makes that same choice silently.
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
        """Bind the function as an instance method, `self` filling argument 0.

        Accessing this through an instance returns a bound view that
        inserts that instance ahead of every call's own arguments, so a
        `Function` used as a class attribute dispatches on
        `#!python self` (unannotated, and therefore matched against
        [`Any`][typing.Any]) together with whatever else was passed.
        Accessing it through the class itself, with no instance, returns
        the function unchanged.
        """
        if instance is None:
            return self
        return _BoundFunction(self, instance)

    # -- diagnostics ----------------------------------------------------

    def ambiguities(self) -> tx.List[tx.Tuple[Method, Method]]:
        """List the pairs of methods that could dispatch to an ambiguous call.

        Two methods appear in the returned list when they can bind a
        common call shape, are incomparable once priority is set aside,
        and land comparable hints at every argument of that shape, so
        that some call matches both of them with no most specific one to
        prefer. A pair separated by a differing `priority` is resolved
        deterministically whenever it is actually called, so such a pair
        is left out. This check is a heuristic run over each method's own
        fully applied shape, following RFC 0001 §5: it surfaces the
        ambiguities a straightforwardly written call would hit, not every
        ambiguity reachable only through `#!python *args` spreading or a
        subclass that does not exist yet.

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

        The caller is expected to already hold the lock. The ABC cache
        token is read again here, rather than trusted from a value
        captured before the lock was acquired. A token that advanced in
        the meantime would otherwise get stamped onto a fresh cache as
        though it were still current, and a later reader would then
        serve stale results from it. Any cache built for a different
        set of methods or an older token is discarded and replaced with a
        fresh, empty one.
        """
        token = abc.get_cache_token()
        cache = self._cache
        if cache.token != token or cache.methods is not self._methods:
            cache = _Cache(self._methods, token)
            self._cache = cache
        return cache

    def _refresh(self) -> _Cache:
        """Acquire the lock, then return a cache valid for the current
        token.
        """
        with self._lock:
            return self._ensure()

    def clear_cache(self) -> None:
        """Discard the dispatch cache, forcing the next call to recompute it.

        The registered methods themselves are left untouched; only the
        cached shape plans and per-call results are thrown away. This is
        rarely needed in practice, since both registering a method and an
        [`abc.register`][abc.ABCMeta.register] call anywhere in the
        program already invalidate the cache automatically. It remains
        available for the one case those do not cover: a value whose
        `#!python isinstance` behaviour changed in a way the ABC cache
        token does not reflect.
        """
        with self._lock:
            self._cache = _Cache(self._methods, _NO_TOKEN)

    def _build_plan(self, shape: tx.Any, cache: _Cache) -> _Plan:
        """Compute and store the plan for `shape`; the caller holds the
        lock.
        """
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
        declared = set()  # type: tx.Set[tx.Any]
        members = {}  # type: tx.Dict[tx.Any, tx.Set[str]]
        for _, _, landed in bindable:
            for key, hint in landed.items():
                if is_value_dependent(hint):
                    value_dependent.add(key)
                if is_declaration_dependent(hint):
                    declared.add(key)
                names = instance_members(hint)
                if names:
                    # Every method's protocol at this argument, together: the
                    # key must cover whatever any of their checks reads.
                    members.setdefault(key, set()).update(names)
        plan = _Plan(
            bindable,
            le_matrix,
            frozenset(value_dependent),
            frozenset(declared),
            {key: tuple(sorted(names)) for key, names in members.items()},
        )
        cache.shape_plans[shape] = plan
        return plan

    # -- resolution -----------------------------------------------------

    def _resolve_values(
        self,
        args: tx.Sequence[tx.Any],
        kwargs: tx.Mapping[str, tx.Any],
        plan: _Plan,
    ) -> Method:
        """Find the most specific method that accepts this value call."""
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
        """Narrow a set of equally specific methods to one, or to `None`.

        The tie-breaks below are tried one after another, in the order
        RFC 0001 §2.2 and §3 lay out. Explicit `priority` is tried
        first, the higher value winning outright. For a value call, the
        argument's own MRO is tried next, favoring whichever hint names
        the more derived base class. After that comes tightness, where
        the signature absorbing fewer arguments into catch-alls and
        relying on fewer defaults wins. Last is the repeated-`TypeVar`
        refinement, where a method whose repeated `TypeVar`s tie
        strictly more argument pairs together than another's wins. If
        more than one candidate survives every one of these steps, the
        tie is genuine, and `#!python None` reports it as such.
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
        """Report whether `winner` refines `loser` by repeated `TypeVar`s.

        This is the last of the selection tie-breaks from RFC 0001 §3,
        reached only once two methods are otherwise equally specific. It
        compares the two methods' hints and their repeated-`TypeVar`
        groupings for the call's shape by deferring to
        [`_group_more_specific`][].
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
        """Report whether method `winner` refines `loser` by argument MRO."""
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
        """Build a [`NoMethodError`][] for a call that no method accepts."""
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
        """Build a did-you-mean line for a keyword no method declares."""
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
        """Build an [`AmbiguousMethodError`][] for a tie between methods."""
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
        """Build a signature from the call's own types that would settle
        the tie.

        An argument that one of the competing methods matched through
        [`Exact`][bagof.dispatchers.Exact] is spelled
        `#!python Exact[...]` in the suggestion too, so the suggested
        signature would outrank that competitor as well, not just the
        others.
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
        """Render the hint for one argument of a "possible fix" signature."""
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
        """Render the call by each argument's type or hint, never its value."""
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
    """A [`Function`][] with its leading argument fixed, acting as a
    method.
    """

    __slots__ = ("_function", "_instance")

    def __init__(self, function: Function, instance: tx.Any) -> None:
        self._function = function
        self._instance = instance

    def __call__(self, *args: tx.Any, **kwargs: tx.Any) -> tx.Any:
        return self._function(self._instance, *args, **kwargs)

    def dispatch(self, *args: tx.Any, **kwargs: tx.Any) -> Method:
        """Select the method for `self` and these arguments, without running
        it.
        """
        return self._function.dispatch(self._instance, *args, **kwargs)

    def resolve(self, *hints: tx.Any, **kwargs: tx.Any) -> tx.Any:
        """Resolve at the hint level, with the type of `self` as argument 0."""
        return self._function.resolve(
            type(self._instance), *hints, **kwargs
        )

    def __repr__(self) -> str:
        return (
            f"<bound {self._function.name} of {self._instance!r}>"
        )


# --- selection helpers -------------------------------------------------


def _maximal(applicable: tx.List[int], plan: _Plan) -> tx.List[int]:
    """Keep only the most specific applicable methods, those nothing beats.

    A method survives when no other applicable method is strictly more
    specific than it under this shape's ordering. Exactly one survivor
    means dispatch has its winner; more than one is a genuine tie, left
    for the further tie-breaks to try to settle.
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
    """Measure how tightly a method fits a call, following RFC 0001 §2.2.

    A smaller result means a tighter fit. The measure is a tuple
    compared lexicographically: first the number of arguments absorbed
    into `#!python *args` or `#!python **kwargs`, then the number of
    parameters left to their defaults, then whether the signature has
    `#!python **kwargs` at all, and finally whether it has
    `#!python *args`.
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
    """Find the fully applied shapes worth checking a pair of methods against.

    A method whose hints are still unresolved forward references cannot
    be shaped yet and simply contributes no shape here, leaving any
    ambiguity it might turn out to have for the first real dispatch to
    surface, rather than forcing its hints to resolve during
    registration.
    """
    shapes = set()  # type: tx.Set[tx.Any]
    for method in (first, second):
        try:
            shapes.add(_full_shape(method.signature))
        except NameError:
            pass
    return frozenset(shapes)


def _pair_ambiguous(first: Method, second: Method, shape: tx.Any) -> bool:
    """Report whether two methods are guaranteed to be ambiguous for `shape`.

    Both methods have to bind the shape, be incomparable once priority is
    equal, land their arguments on the same keys, and carry comparable
    hints at every one of those arguments. That way, no combination of
    argument types can make one of them strictly more specific than the
    other at every position. A method whose hint is still unresolved
    cannot be compared at all, so such a pair is simply treated as not,
    or not yet, ambiguous.
    """
    try:
        return _pair_ambiguous_resolved(first, second, shape)
    except NameError:
        return False


def _pair_ambiguous_resolved(
    first: Method, second: Method, shape: tx.Any
) -> bool:
    """Do the work of [`_pair_ambiguous`][], assuming every hint resolves."""
    first_binding = _bind_shape(first.signature, shape)
    second_binding = _bind_shape(second.signature, shape)
    if first_binding is None or second_binding is None:
        return False
    # A differing `priority` breaks the tie deterministically at dispatch (the
    # higher one wins, RFC 0001 §2.2/§5), so the pair is never ambiguous at a
    # call -- do not warn or list it.
    if first.priority != second.priority:
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
        # Some value must fit both hints at every argument. Two ordered hints
        # share the narrower one's values; two hints a lower bound leaves
        # unordered may still share one, which `overlaps` finds.
        if not (
            issubhint(here, there)
            or issubhint(there, here)
            or overlaps(here, there)
        ):
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
    """Bind a bare call shape, filling every position with a placeholder."""
    signature._settle()
    count, names = shape
    return signature.bind(
        (_PLACEHOLDER,) * count, {name: _PLACEHOLDER for name in names}
    )


def _full_shape(signature: Signature) -> tx.Tuple[int, tx.Tuple[str, ...]]:
    """Find the shape of a call that would fill every parameter of
    `signature`.
    """
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
    """Map each bound argument's key to the parameter hint it landed in."""
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
    """Group each bound argument by which repeated `TypeVar` it landed in.

    The result gives one label per argument key. Two keys share a label
    only when both landed in the same [`TypeVar`][typing.TypeVar];
    an argument landing on any other kind of hint gets a label that
    belongs to it alone, forming a block of one. Nothing about these
    labels matters beyond equality, which is all the repeated-`TypeVar`
    tie-break needs from them.

    A `#!python **kwargs: T` slot is grouped here along with everything
    else, since every keyword it captures lands on the same variable.
    The tie-break sees them as one block that must agree on a consistent
    `T`, and a method with `#!python **kwargs: T` is accordingly more
    specific than one with an untyped `#!python **kwargs`. Applicability
    solves `T` jointly across those same captured keywords too, exactly
    as it does for `#!python *args: T`.

    A `#!python *args: *Ts` slot is deliberately excluded from this
    grouping, since its landed hint is an unpacked
    [`TypeVarTuple`][typing.TypeVarTuple] rather than a
    [`TypeVar`][typing.TypeVar]; it stays in a block of its own and never
    wins this particular tie-break, the opposite treatment from
    `#!python *args: T`, per RFC 0001 §3. Its own joint solving happens
    in applicability, not here.
    """
    labels = {}  # type: tx.Dict[tx.Any, tx.Any]
    for key, hint in signature._iter_arguments(binding):
        if _is_plain_typevar(hint):
            # Identity, not the variable itself: two distinct `TypeVar`s that
            # happen to be equal must land in different blocks. The
            # `**kwargs`-absorbed keys land the signature's `**kwargs`
            # variable, so they all share its block. A `*args: *Ts` tail is
            # *not* a plain `TypeVar` (though 3.8 mis-reports it as one), so it
            # stays solo and never wins this tie-break.
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
    """Report whether `a`'s repeated `TypeVar`s make it strictly more specific.

    This implements the repeated-`TypeVar` tie-break from RFC 0001 §3,
    reached only once two methods are already equally specific by every
    earlier measure. `a` is judged to win when two conditions both hold.
    The first is that nothing else already tells the two methods apart.
    Every argument lands an equivalent hint under both, so neither is
    more specific at any position on its own (an unbound `#!python T` is
    equivalent to an unannotated `#!python Any`, and a bound
    `#!python TypeVar` is equivalent to its bound). The second is that
    `a`'s grouping is a strict superset of `b`'s: every pair of
    arguments that `b` ties together under one repeated `TypeVar` is
    also tied together by `a`, and `a` additionally ties at least one
    pair that `b` leaves independent.

    Tying more arguments to a single consistent type is the more
    constrained reading, and therefore the more specific one. When
    neither method's grouping is a strict superset of the other's,
    whether because the two groupings are equal or because each ties a
    pair the other leaves apart, this returns [`False`][] in both
    directions, and the pair remains incomparable, and so ambiguous.
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
    """Cache `method` under `key`, respecting the cache's size cap.

    Each plan's own call cache is capped in size. Once it is full and
    `key` is one it has not seen before, the entry inserted longest ago
    is evicted first, keeping a function called with unboundedly many
    distinct keys down to a bounded working set instead of growing
    forever. A key that turns out not to be hashable, as happens for an
    unhashable value at a value-dependent argument, is simply left
    uncached.
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
    """Build the cache key for a concrete call, under a shape's plan.

    Every argument contributes its type to the key at minimum. An
    argument whose hint reads the value itself, as a `#!python Literal`
    or a `#!python type[...]` does, contributes that value too; one
    whose hint reads a parametrisation the value declared at
    construction, as with a generic such as `#!python Box[int]`,
    contributes that parametrisation; and one whose hint reads which of
    a protocol's data members the value carries contributes that
    instead. Building the key tuple never fails outright. An unhashable
    value at a value-dependent argument is wrapped so the tuple can still
    be built, and the resulting [`TypeError`][] only surfaces once the
    key is actually hashed, during a `dict` access, where the caller
    catches it and leaves that call uncached.
    """
    dependent = plan.dependent
    parts = [len(args)]  # type: tx.List[tx.Any]
    for index, value in enumerate(args):
        if index not in dependent:
            parts.append(type(value))
        elif index in plan.declared_only:
            parts.append((type(value), _declared_key(value)))
        elif index in plan.value_only:
            parts.append((type(value), _KeyValue(value)))
        elif index in plan.members_only:
            names = plan.members_only[index]
            parts.append((type(value), _members_key(value, names)))
        else:
            parts.append(_dependent_part(value, index, plan))
    for keyword in sorted(kwargs):
        value = kwargs[keyword]
        if keyword not in dependent:
            parts.append((keyword, type(value)))
        elif keyword in plan.declared_only:
            parts.append((keyword, type(value), _declared_key(value)))
        elif keyword in plan.value_only:
            parts.append((keyword, type(value), _KeyValue(value)))
        elif keyword in plan.members_only:
            names = plan.members_only[keyword]
            parts.append((keyword, type(value), _members_key(value, names)))
        else:
            parts.append((keyword,) + _dependent_part(value, keyword, plan))
    return tuple(parts)


def _dependent_part(
    value: tx.Any, key: tx.Any, plan: _Plan
) -> tx.Tuple[tx.Any, ...]:
    """Build the key part for an argument whose hints read several things
    at once.

    The part begins with the value's type, then adds one entry for each
    of the things the hints landing at `key` actually read: the value
    itself, the parametrisation it declares, and the protocol data
    members it carries. None of these three can stand in for another,
    since a value's own `==` reflects neither its recorded
    parametrisation, as a dataclass-based generic compares only its
    fields, nor which attributes happen to be set on it.
    [`_call_key`][] itself builds the simpler, single-entry part for an
    argument that reads only one of these things.
    """
    part = (type(value),)  # type: tx.Tuple[tx.Any, ...]
    if key in plan.value_dependent:
        part += (_KeyValue(value),)
    if key in plan.declared:
        part += (_declared_key(value),)
    names = plan.members.get(key)
    if names:
        part += (_members_key(value, names),)
    return part


# Which of the protocol data members `names` the value has, in order: a tuple
# of booleans, read by the very function the value check uses, so the key
# always covers what the check reads -- never the value itself, so every
# instance of one class that holds the same members shares one entry.
_members_key = _present_data_members


def _declared_key(value: tx.Any) -> tx.Any:
    """Find what keys `value` at a declaration-dependent argument, beside
    its type.

    The result is the parametrisation the instance recorded when it was
    constructed, such as `#!python Box[int]` for a
    `#!python Box[int]()` instance, or `#!python None` when nothing was
    recorded. This is never the instance itself, so every instance built
    from the same parametrisation shares one cache entry.

    Only an instance of a `Generic` subclass, or of a class written
    against a PEP 585 alias such as `#!python class GL(list[T])`, is
    even asked for this record. That is gated by the same
    `_may_record_parametrisation` check the value-checking side of
    dispatch applies before reading it, so the two stay consistent with
    each other. Any other value, such as a plain `#!python list` given
    for a `#!python List[int]` argument, a `str` given for a
    `#!python Union[Box[int], str]` one, or a lazy proxy whose
    `__getattr__` performs real work, is never probed at all.
    """
    if not _may_record_parametrisation(type(value)):
        return None
    try:
        recorded = value.__orig_class__
    except Exception:
        # Absent (`AttributeError`), or a `__getattr__` that raises: the value
        # check reads nothing either (`_orig_class`).
        return None
    if type(recorded) is not _PEP585_ALIAS:
        # A typing record (`Box[int]`) is keyed by identity: `_record_key`'s
        # last case, without the call.
        return _SameObject(recorded)
    return _record_key(recorded)


# Whether a PEP 585 alias can be unpacked (`*tuple[int]`, Python 3.11+),
# which gives it the origin and arguments of the packed one. Before that,
# reading `__unpacked__` off an alias is forwarded to its origin and raises.
_UNPACKABLE = hasattr(_PEP585_ALIAS, "__unpacked__")


def _record_key(recorded: tx.Any) -> tx.Any:
    """Turn a recorded parametrisation into the form the cache keys it by.

    A PEP 585 alias such as `#!python GL[int]` is not itself cached by
    Python at subscription time, so each `#!python GL[int]()` records a
    fresh alias object rather than a shared one. This builds a key from
    its parts instead: its origin, each of its arguments (recursively
    keyed the same way), and whether it appears unpacked, as in
    `#!python *tuple[int]`, so that every `#!python GL[int]()` still
    shares a single cache entry despite the differing objects. A plain
    class, one whose metaclass is `#!python type`, already compares and
    hashes by identity and needs no wrapping to serve as its own key.
    Anything else falls back to being keyed by identity, through
    [`_SameObject`][].
    """
    kind = type(recorded)
    if kind is type:
        return recorded
    if kind is _PEP585_ALIAS:
        origin = recorded.__origin__
        args = recorded.__args__
        for arg in args:
            if type(arg) is not type:
                # Only plain classes are their own keys: key them all.
                args = tuple(map(_record_key, args))
                break
        return (
            origin if type(origin) is type else _SameObject(origin),
            args,
            recorded.__unpacked__ if _UNPACKABLE else False,
        )
    return _SameObject(recorded)


class _KeyValue:
    """A value wrapped so that being unhashable fails on use, not on build.

    A cache key carries an actual value only for a value-dependent
    argument, and this wrapper is what such a value passes through.
    [`hash`][hash] and equality both delegate to the wrapped value, so
    two calls sharing the same literal end up sharing a key, while an
    unhashable value raises [`TypeError`][] only from the surrounding
    `dict` access, not from constructing the key tuple itself; the
    caller catches that error there and simply leaves the call uncached.
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
    """Score how close `method` came to accepting a failed call, and say
    why not.

    The tuple returned is `(matched, arity_gap, method, highlight,
    reason)`. `matched` counts the arguments that did fit, with a higher
    count meaning closer; `arity_gap` measures the distance between the
    method's arity and the call's, with a smaller gap meaning closer;
    `method` is the candidate itself; `highlight` names the slots to mark
    with `#!python !` in the rendered error; and `reason` is a short
    phrase explaining a total binding failure, or `None` when binding
    succeeded and only a type mismatch was the problem.
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
    """Explain, in a short phrase, why a call fails to bind to `signature`."""
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
    """Extract `priority` from the registration options, rejecting
    anything else.

    A keyword argument to `register` is always a registration option
    and never a named hint, since named hints belong in the overlay
    dict instead. Only `priority` is a recognised option; any other
    keyword is treated as a mistake and named in the error, along with a
    pointer to the dict form it should have used.
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
    """Split the hint-overlay arguments into positional hints and named hints.

    `args` may hold a `#!python tuple` of positional hints, a
    `#!python dict` of named hints, both together, or neither at all.
    Anything besides a tuple or a dict, or two arguments of the same
    kind, is a mistake on the caller's part and raises.
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

    Positional hints replace the hints of the first parameters, taken in
    order, while named hints replace the hint of whichever parameter
    each one names, and may also target the `#!python *args` or
    `#!python **kwargs` catch-all to set its element or value hint.
    Every parameter's name, kind, and default is kept exactly as it was,
    so a call still binds precisely the way the underlying function's
    own signature says it should. Each hint given this way is
    normalised and checked to be a genuine type hint, and no parameter
    may be given a hint through both forms at once.
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
        # A hint targeting `*args` / `**kwargs` may be a `P.args` / `P.kwargs`
        # form, which degrades to an `Any` tail rather than being refused.
        allow_variadic = name in (varargs_name, varkw_name)
        checked = _overlay_hint(fn, name, hint, allow_variadic)
        if name == varargs_name:
            # A bare TypeVarTuple on `*args` describes nothing on its own;
            # refuse it here as the decorator path does, pointing at
            # `*args: Unpack[Ts]`.
            _reject_variadic_param(name, checked, fn, catch_all=True)
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
    fn: tx.Callable[..., tx.Any],
    target: str,
    hint: tx.Any,
    allow_variadic: bool = False,
) -> tx.Any:
    """Normalise a registration hint, checking that it is a genuine type hint.

    A value that is not a type or a typing construct at all, such as a
    stray tuple, a number, or a plain string, would otherwise register a
    method that silently never matches anything. It is refused right
    here instead, with a message that names the parameter it was meant
    for. A
    parametrised form such as `#!python Annotated[int, ...]` or
    `#!python Exact[int]` still counts as plausible through its origin
    even when the whole hint itself would not.

    A `#!python ParamSpec` or `#!python Concatenate[...]` given for an
    ordinary parameter is refused as well. `allow_variadic` lifts that
    refusal only when the target is `#!python *args` or
    `#!python **kwargs`, where a `P.args` or `P.kwargs` form is instead
    allowed to degrade to a plain `#!python Any` tail.
    """
    normalised = normalise_hint(hint)
    if not allow_variadic:
        _reject_variadic_param(target, normalised, fn)
    _reject_bare_super(target, normalised, fn)
    _reject_malformed_typeddict(target, normalised, fn)
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
    """Add `method` to `methods`, replacing one written the same way, if any.

    A method whose signature is written exactly like one already
    registered, with the same parameter names, kinds, required-ness, and
    structurally equal hints, replaces that earlier method in place and
    raises a [`RuntimeWarning`][], which is the shape produced by a
    module reload or an accidentally doubled decorator. A method that is
    only equivalent under the sub-hint relation but written differently,
    such as `#!python (x: T, y: T)` against `#!python (x: T, y: U)`, or
    `#!python Exact[int]` against plain `#!python int`, counts as a
    distinct method and is appended instead; selection then has to order
    the two at dispatch time, and
    [`_warn_new_ambiguities`][Function._warn_new_ambiguities] flags them
    if they turn out to clash.
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
