"""Registries of named functions: [`Dispatcher`][] and [`dispatch`][].

A [`Dispatcher`][bagof.dispatchers.Dispatcher] holds a collection of named
[`Function`][bagof.dispatchers.Function]s and hands them out through a small
`functions` namespace. The module-level
[`dispatch`][bagof.dispatchers.dispatch] is the ready-made registry most code
uses: decorate a `#!python def` with it and the most specific overload runs on
each call.

The two registries differ only in **what identifies a function**:

* [`dispatch`][bagof.dispatchers.dispatch] keys a function by its **module and
  qualified name**, so the same name in two modules is two independent
  functions -- a `#!python @dispatch def area` in one module never merges with
  one in another.
* A [`Dispatcher`][bagof.dispatchers.Dispatcher] you build yourself keys a
  function by its **qualified name alone**, so every module registering that
  name into the same instance extends one shared function -- the way to build a
  cross-module generic function.
"""

# stdlib
import sys
import threading

# dependencies
import typing_extensions as tx

# local
from ._function import Function

__all__ = ["Dispatcher", "dispatch"]


def _qualname(fn: tx.Any) -> str:
    """The qualified name a callable is keyed by, best effort."""
    name = getattr(fn, "__qualname__", None)
    if name is None:
        name = getattr(fn, "__name__", None)
    if name is None:
        # A callable instance (an object with `__call__`) has no name of its
        # own; fall back to its type's, so it still keys stably.
        name = type(fn).__qualname__
    return name


def _module_of(fn: tx.Any) -> tx.Optional[str]:
    """The module a callable was defined in, or `None` when it has none."""
    return getattr(fn, "__module__", None)


def _is_impl(value: tx.Any) -> bool:
    """Whether `value` is an implementation rather than an overlay of hints.

    A `#!python tuple` of positional hints or a `#!python dict` of named hints
    is the overlay form; any other callable -- a function, a class, a callable
    instance -- is the implementation itself.
    """
    return callable(value) and not isinstance(value, (tuple, dict))


def _caller_module() -> tx.Optional[str]:
    """The `#!python __name__` of the module two frames up, or `None`.

    Called from a `functions`-namespace dunder, it names the module the access
    was written in: the user's frame sits two levels above (this helper, then
    the dunder). It is used only by the module-level
    [`dispatch`][bagof.dispatchers.dispatch], whose functions are keyed per
    module; a [`Dispatcher`][bagof.dispatchers.Dispatcher] ignores it.
    """
    try:
        frame = sys._getframe(2)
    except ValueError:  # pragma: no cover
        # Too few frames on the stack (only reachable from deep C entry
        # points); there is no caller module to name.
        return None
    return frame.f_globals.get("__name__")


class _Functions:
    """A protocol-only view onto a dispatcher's functions.

    It exposes only the mapping protocol -- `#!python view["area"]`,
    `#!python view.area`, `#!python "area" in view`,
    `#!python for name in view`, `#!python len(view)` -- and **no named
    methods**, so every function name (`#!python register`, `#!python items`,
    `#!python map`, ...) is safe to reach through it without colliding with a
    method of its own.

    Both item and attribute access **get-or-create**: naming a function that
    does not exist yet makes an empty one, so a registry and its callers can
    reach the same function by name in any order. Attribute access ignores
    names beginning with `#!python _`, so a REPL or tool probing for dunders
    never mints an empty function; item access does not, so a function may
    still be named with a leading underscore through `#!python view["_x"]`.
    """

    # A name-mangled slot, so even the reference back to the dispatcher is not
    # reachable as an ordinary underscore attribute -- `view._owner` misses and
    # is refused like any other private probe.
    __slots__ = ("__owner",)

    def __init__(self, owner: "Dispatcher") -> None:
        self.__owner = owner

    def __getattr__(self, name: str) -> Function:
        if name.startswith("_"):
            # Never mint a function for a dunder or private probe.
            raise AttributeError(name)
        return self.__owner._function_by_name(name, _caller_module())

    def __getitem__(self, name: str) -> Function:
        return self.__owner._function_by_name(name, _caller_module())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.__owner._has_function_name(name, _caller_module())

    def __iter__(self) -> tx.Iterator[str]:
        return iter(self.__owner._function_names(_caller_module()))

    def __len__(self) -> int:
        return len(list(self.__owner._function_names(_caller_module())))

    def __repr__(self) -> str:
        names = sorted(self.__owner._function_names(_caller_module()))
        return "functions({})".format(", ".join(repr(n) for n in names))


class Dispatcher:
    """A registry of named [`Function`][bagof.dispatchers.Function]s.

    Register overloads by decorating a `#!python def` with the dispatcher; each
    `#!python def` of the same name adds an overload rather than replacing the
    name, and the dispatcher returns the
    [`Function`][bagof.dispatchers.Function] so the name stays bound to it.

    A `Dispatcher` you build keys each function by its **qualified name
    alone**, so registering the same name from several modules into one
    instance builds a single shared generic function.

    !!! example
        ```pycon
        >>> from bagof.dispatchers import Dispatcher
        >>> shapes = Dispatcher()
        >>> @shapes
        ... def area(w: int, h: int) -> int:
        ...     return w * h
        >>> @shapes
        ... def area(r: float) -> float:
        ...     return 3.14159 * r * r
        >>> area(3, 4)
        12
        >>> area(2.0)
        12.56636
        ```

    Reach a function to hand around through the `functions` namespace:

    !!! example
        ```pycon
        >>> shapes.functions["area"] is area
        True
        >>> "area" in shapes.functions
        True
        ```
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._functions = {}  # type: tx.Dict[tx.Any, Function]
        self._view = _Functions(self)

    @property
    def functions(self) -> _Functions:
        """The get-or-create namespace of this dispatcher's functions.

        A protocol-only view: `#!python d.functions.area` and
        `#!python d.functions["area"]` both return the function named
        `#!python "area"`, creating it empty if it does not exist yet, and the
        view carries no named methods of its own so no function name can
        collide with one.
        """
        return self._view

    def __call__(self, *args: tx.Any, **options: tx.Any) -> tx.Any:
        """Register an overload, or return a decorator that does.

        Used bare on a `#!python def` (`#!python @d`), it registers that
        function -- read as the implementation, dispatched on its own
        parameters -- and returns the [`Function`][bagof.dispatchers.Function]
        it belongs to. Used with an overlay of hints
        (`#!python @d((int,), {"scale": float})`) or options
        (`#!python @d(priority=5)`), it returns a decorator that registers the
        function it wraps with those hints laid over its parameters.

        The overlay follows
        [`Function.register`][bagof.dispatchers.Function.register]:
        **positional hints are a tuple, named hints a dict, keyword arguments
        are options** (only `#!python priority`), never hints.
        """
        return self.register(*args, **options)

    def register(self, *args: tx.Any, **options: tx.Any) -> tx.Any:
        """Register an overload, or return a decorator that does.

        The same as calling the dispatcher directly (`#!python @d`); see
        [`__call__`][bagof.dispatchers.Dispatcher.__call__]. It returns the
        [`Function`][bagof.dispatchers.Function] the overload joined, so the
        decorated name binds to the dispatched function.
        """
        if args and _is_impl(args[0]):
            impl = args[0]
            function = self._function_for(impl)
            # Let `Function.register` do the validation and adopt the name;
            # its return value (the callable) is dropped -- the name binds to
            # the function, the dispatcher's convention.
            function.register(*args, **options)
            return function

        # Overlay form: the hints/options arrive now, the callable when the
        # returned decorator is applied.
        def decorator(fn: tx.Callable[..., tx.Any]) -> Function:
            function = self._function_for(fn)
            function.register(*args, **options)(fn)
            return function

        return decorator

    def clear_cache(self) -> None:
        """Drop every function's dispatch cache.

        Rarely needed -- caches are invalidated automatically on registration
        and when the ABC registry changes -- but available for a registry
        whose applicable types were altered in a way nothing else observes.
        """
        with self._lock:
            for function in self._functions.values():
                function.clear_cache()

    def __repr__(self) -> str:
        count = len(list(self._function_names(None)))
        return f"{type(self).__name__}({count} function(s))"

    # -- identity -------------------------------------------------------
    #
    # A `Dispatcher` keys a function by qualified name alone; the module-level
    # `dispatch` overrides these to key by (module, qualified name). A method
    # nested in a class carries its class in its qualified name, so two
    # classes' same-named methods stay distinct and each is reached as a class
    # attribute (through `Function.__get__`) rather than by bare name here.

    def _key_for_impl(self, fn: tx.Any) -> tx.Any:
        return _qualname(fn)

    def _key_for_name(self, name: str, module: tx.Optional[str]) -> tx.Any:
        return name

    def _names_for(self, module: tx.Optional[str]) -> tx.Iterator[str]:
        return iter(list(self._functions))

    def _has_name(self, name: str, module: tx.Optional[str]) -> bool:
        return self._key_for_name(name, module) in self._functions

    # -- shared get-or-create -------------------------------------------

    def _function_for(self, fn: tx.Any) -> Function:
        """The function `fn` registers into, created empty if new."""
        return self._get_or_create(self._key_for_impl(fn), None)

    def _function_by_name(
        self, name: str, module: tx.Optional[str]
    ) -> Function:
        """The function named `name`, created empty if new."""
        return self._get_or_create(self._key_for_name(name, module), name)

    def _has_function_name(
        self, name: str, module: tx.Optional[str]
    ) -> bool:
        with self._lock:
            return self._has_name(name, module)

    def _function_names(
        self, module: tx.Optional[str]
    ) -> tx.Iterator[str]:
        with self._lock:
            return self._names_for(module)

    def _get_or_create(
        self, key: tx.Any, name: tx.Optional[str]
    ) -> Function:
        with self._lock:
            function = self._functions.get(key)
            if function is None:
                function = Function(name)
                self._functions[key] = function
            return function


class _ModuleDispatcher(Dispatcher):
    """The module-level [`dispatch`][bagof.dispatchers.dispatch] registry.

    Identical to a [`Dispatcher`][bagof.dispatchers.Dispatcher] except that a
    function is keyed by its **module and qualified name**, so the same name in
    two modules names two independent functions.
    """

    def _key_for_impl(self, fn: tx.Any) -> tx.Any:
        return (_module_of(fn), _qualname(fn))

    def _key_for_name(self, name: str, module: tx.Optional[str]) -> tx.Any:
        # A bare name here is resolved against the caller's module, so
        # `dispatch.functions.area` names this module's `area`. (For the
        # module-level registry, prefer the value the decorator returns; the
        # namespace is unambiguous on a `Dispatcher` you build.)
        return (module, name)

    def _names_for(self, module: tx.Optional[str]) -> tx.Iterator[str]:
        return iter(
            [name for (mod, name) in self._functions if mod == module]
        )


dispatch = _ModuleDispatcher()
"""The ready-made registry for hint-informed multiple dispatch.

Decorate a `#!python def` with `#!python @dispatch` to register an overload;
each `#!python def` of the same name in the same module adds an overload, and
`#!python @dispatch` returns the [`Function`][bagof.dispatchers.Function] so
the name stays bound to the dispatched function. Functions are keyed by module
and qualified name, so the same name in another module is independent.

!!! example
    ```pycon
    >>> from bagof.dispatchers import dispatch
    >>> @dispatch
    ... def describe(x: int) -> str:
    ...     return "an integer"
    >>> @dispatch
    ... def describe(x: str) -> str:
    ...     return "a string"
    >>> describe(7)
    'an integer'
    >>> describe("hi")
    'a string'
    ```

Lay explicit hints over a function's parameters with the overlay form --
positional hints as a tuple, named hints as a dict, `#!python priority` as a
keyword option:

!!! example
    ```pycon
    >>> @dispatch((int,), {"scale": int})
    ... def scaled(value, scale):
    ...     return value * scale
    >>> scaled(3, scale=4)
    12
    ```
"""
