"""Registries of named functions: [`Dispatcher`][] and [`dispatch`][].

A [`Dispatcher`][] holds a collection of named [`Function`][]s and
hands them out through a small `functions` namespace. The module-level
[`dispatch`][] is the ready-made registry most code uses: decorate a
`#!python def` with it, and the most specific overload runs on each
call.

The two registries differ only in what identifies a function.
[`dispatch`][] keys a function by its module together with its
qualified name, so the same name defined in two modules produces two
independent functions; a `#!python @dispatch def area` in one module
never merges with one in another. A [`Dispatcher`][] you build yourself
keys a function by its qualified name alone, so every module that
registers that name into the same instance extends one shared
function, which is how a generic function is built across modules.
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
    """Return the qualified name a callable is keyed by, best effort."""
    name = getattr(fn, "__qualname__", None)
    if name is None:
        name = getattr(fn, "__name__", None)
    if name is None:
        # A callable instance (an object with `__call__`) has no name of its
        # own; fall back to its type's, so it still keys stably.
        name = type(fn).__qualname__
    return name


def _module_of(fn: tx.Any) -> tx.Optional[str]:
    """Return the module a callable was defined in, or `None` if none."""
    return getattr(fn, "__module__", None)


def _is_impl(value: tx.Any) -> bool:
    """Report whether `value` is an implementation rather than a hint overlay.

    A `#!python tuple` of positional hints, or a `#!python dict` of
    named hints, is the overlay form. Any other callable, whether a
    function, a class, or a callable instance, is the implementation
    itself.
    """
    return callable(value) and not isinstance(value, (tuple, dict))


def _caller_module() -> tx.Optional[str]:
    """Return the `#!python __name__` of the module two frames up, if any.

    This is called from a dunder method of the `functions` namespace,
    and it names the module where the access was written: the user's
    frame sits two levels above the call, above this helper and above
    the dunder method that calls it. Only the module-level [`dispatch`][],
    whose functions are keyed per module, uses this; a [`Dispatcher`][]
    ignores it.
    """
    try:
        frame = sys._getframe(2)
    except ValueError:  # pragma: no cover
        # Too few frames on the stack (only reachable from deep C entry
        # points); there is no caller module to name.
        return None
    return frame.f_globals.get("__name__")


def _is_qualified(key: tx.Any) -> bool:
    """Report whether `key` is a `#!python (module, name)` pair of two strings.

    That pair is the qualified spelling accepted alongside a bare name.
    It names the module explicitly instead of taking it from the
    calling frame, which is what looking up a function from a
    cross-module re-export needs.
    """
    return (
        isinstance(key, tuple)
        and len(key) == 2
        and isinstance(key[0], str)
        and isinstance(key[1], str)
    )


class _Functions:
    """A protocol-only view onto a dispatcher's functions.

    It exposes only the mapping protocol, `#!python view["area"]`,
    `#!python view.area`, `#!python "area" in view`,
    `#!python for name in view`, and `#!python len(view)`, and defines
    no named methods of its own, so every function name (`register`,
    `items`, `map`, and so on) can be reached through it without
    colliding with a method.

    Both item access and attribute access get or create: naming a
    function that does not exist yet creates an empty one, so a
    registry and its callers can reach the same function by name in any
    order. Attribute access ignores names beginning with
    `#!python _`, so a REPL or a tool probing for dunder attributes
    never mints an empty function by accident; item access applies no
    such filter, so a function can still be named with a leading
    underscore through `#!python view["_x"]`.

    Item access also accepts a `#!python (module, name)` pair, the
    qualified spelling, which names the module explicitly rather than
    reading it from the calling frame. That is how the module-level
    registry's function is reached from a module other than the one
    that registered it.
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

    def __getitem__(self, key: tx.Any) -> Function:
        if isinstance(key, str):
            return self.__owner._function_by_name(key, _caller_module())
        if _is_qualified(key):
            module, name = key
            return self.__owner._function_by_name(name, module)
        raise TypeError(
            f"a function is named by a string, or by a (module, name) pair "
            f"of strings for a cross-module lookup; got {key!r}."
        )

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return self.__owner._has_function_name(key, _caller_module())
        if _is_qualified(key):
            module, name = key
            return self.__owner._has_function_name(name, module)
        return False

    def __iter__(self) -> tx.Iterator[str]:
        return iter(self.__owner._function_names(_caller_module()))

    def __len__(self) -> int:
        return len(list(self.__owner._function_names(_caller_module())))

    def __repr__(self) -> str:
        names = sorted(self.__owner._function_names(_caller_module()))
        return "functions({})".format(", ".join(repr(n) for n in names))


class Dispatcher:
    """A registry of named [`Function`][]s.

    Register overloads by decorating a `#!python def` with the
    dispatcher. Each `#!python def` of the same name adds an overload
    rather than replacing the name, and the dispatcher returns the
    [`Function`][] the overload joined, so the name stays bound to it.

    A `Dispatcher` you build keys each function by its qualified name
    alone, so registering the same name from several modules into one
    instance builds a single generic function shared across them.

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

        This is a protocol-only view: `#!python d.functions.area` and
        `#!python d.functions["area"]` both return the function named
        `#!python "area"`, creating it empty if it does not already
        exist, and the view defines no named methods of its own, so no
        function name can collide with one.

        Attribute access ignores names beginning with `#!python _`, so
        a REPL or a tool probing for dunder attributes never mints an
        empty function by accident. Item access applies no such filter,
        so `#!python d.functions["_x"]` still names a function whose
        name starts with an underscore. For that reason, avoid
        registering an anonymous overload, such as
        `#!python @dispatch def _` or a `#!python lambda`, on the
        module-level [`dispatch`][]: every anonymous overload keys the
        same `#!python "_"` or `#!python "<lambda>"` name and collapses
        into a single function. Give each overload a real name instead,
        or overlay hints onto a named `#!python def`.
        """
        return self._view

    def __call__(self, *args: tx.Any, **options: tx.Any) -> tx.Any:
        """Register an overload, or return a decorator that does.

        Used bare on a `#!python def`, as `#!python @d`, it registers
        that function directly, taken as the implementation dispatched
        on its own parameters, and returns the [`Function`][] it
        belongs to. Used with an overlay of hints, as
        `#!python @d((int,), {"scale": float})`, or with options, as
        `#!python @d(priority=5)`, it instead returns a decorator that
        registers the function it wraps with those hints laid over its
        parameters.

        The overlay follows the same convention as
        [`Function.register`][]: positional hints are given as a tuple,
        named hints as a dict, and keyword arguments are always
        options, currently only `#!python priority`, never hints.
        """
        return self.register(*args, **options)

    def register(self, *args: tx.Any, **options: tx.Any) -> tx.Any:
        """Register an overload, or return a decorator that does.

        This does the same thing as calling the dispatcher directly,
        `#!python @d`; see [`__call__`][]. It returns the [`Function`][]
        the overload joined, so the decorated name binds to the
        dispatched function rather than to the plain implementation.
        """
        self._reject_bad_first_arg(args)
        if args and _is_impl(args[0]):
            impl = args[0]
            key = self._key_for_impl(impl)
            function, created = self._get_or_create(key, None)
            # Let `Function.register` do the validation and adopt the name;
            # its return value (the callable) is dropped -- the name binds to
            # the function, the dispatcher's convention. A registration that
            # fails leaves no empty function behind -- but only one this call
            # made, never one a namespace access minted earlier.
            try:
                function.register(*args, **options)
            except BaseException:
                if created:
                    self._discard_orphan(key, function)
                raise
            return function

        # Overlay form: the hints/options arrive now, the callable when the
        # returned decorator is applied.
        def decorator(fn: tx.Callable[..., tx.Any]) -> Function:
            key = self._key_for_impl(fn)
            function, created = self._get_or_create(key, None)
            try:
                function.register(*args, **options)(fn)
            except BaseException:
                if created:
                    self._discard_orphan(key, function)
                raise
            return function

        return decorator

    @staticmethod
    def _reject_bad_first_arg(args: tx.Tuple[tx.Any, ...]) -> None:
        """Reject a first argument that is neither hints nor a callable.

        An overload is registered on a callable, or on a tuple or dict
        overlay of hints. Anything else, such as a bare string or a
        number, could only fail later in a more confusing way, so it is
        refused immediately, with a message that points at the correct
        spelling.
        """
        if (
            args
            and not isinstance(args[0], (tuple, dict))
            and not callable(args[0])
        ):
            raise TypeError(
                f"an overload is registered on a callable -- a def, a class "
                f"or a callable object -- optionally with a tuple of "
                f"positional hints and/or a dict of named hints; "
                f"got {args[0]!r}."
            )

    def _discard_orphan(self, key: tx.Any, function: Function) -> None:
        """Drop a function created for a registration that then failed.

        Registration creates the function before validating the
        overload, so a failure would otherwise leave an empty function
        reachable through the namespace. This is called only for a
        function that this particular registration created, and it
        removes the function only if it is still that same one and
        still holds no methods, so a concurrent writer's registration
        is never disturbed.
        """
        with self._lock:
            if (
                self._functions.get(key) is function
                and not function.methods
            ):
                del self._functions[key]

    def clear_cache(self) -> None:
        """Drop every function's dispatch cache.

        This is rarely needed, since caches are invalidated
        automatically on registration and whenever the ABC registry
        changes, but it remains available for a registry whose
        applicable types were altered in a way nothing else observes.
        """
        with self._lock:
            for function in self._functions.values():
                function.clear_cache()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self._functions)} function(s))"

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

    def _function_by_name(
        self, name: str, module: tx.Optional[str]
    ) -> Function:
        """Return the function named `name`, creating it empty if new."""
        function, _created = self._get_or_create(
            self._key_for_name(name, module), name
        )
        return function

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
    ) -> tx.Tuple[Function, bool]:
        """Return the function at `key`, and whether this call created it.

        The second value is `#!python True` only when the key was
        absent and a fresh, empty function was created for it, so a
        caller can tell a function it just created apart from one a
        namespace access minted earlier.
        """
        with self._lock:
            function = self._functions.get(key)
            if function is None:
                function = Function(name)
                self._functions[key] = function
                return function, True
            return function, False


class _ModuleDispatcher(Dispatcher):
    """The module-level [`dispatch`][] registry.

    This behaves exactly like a [`Dispatcher`][] you build yourself,
    except that a function is keyed by its module together with its
    qualified name, so the same name defined in two modules names two
    independent functions.

    Decorate a `#!python def` to register an overload; each
    `#!python def` of the same name in the same module adds another
    overload to it.

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

    Lay explicit hints over a function's parameters with the overlay
    form: positional hints as a tuple, named hints as a dict, and
    `#!python priority` as a keyword option.

    !!! example
        ```pycon
        >>> @dispatch((int,), {"scale": int})
        ... def scaled(value, scale):
        ...     return value * scale
        >>> scaled(3, scale=4)
        12
        ```
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

Decorate a `#!python def` with `#!python @dispatch` to register an
overload. Each `#!python def` of the same name in the same module adds
an overload, and `#!python @dispatch` returns the [`Function`][] it
belongs to, so the name stays bound to the dispatched function rather
than to the plain implementation. Functions are keyed by module
together with qualified name, so the same name defined in another
module names an independent function.

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

Lay explicit hints over a function's parameters with the overlay form:
positional hints as a tuple, named hints as a dict, and
`#!python priority` as a keyword option.

!!! example
    ```pycon
    >>> @dispatch((int,), {"scale": int})
    ... def scaled(value, scale):
    ...     return value * scale
    >>> scaled(3, scale=4)
    12
    ```
"""
