"""Groups of dispatched functions: [`Dispatcher`][] and [`dispatch`][].

A [`Dispatcher`][] gathers several named [`Function`][]s together and
gives access to them through its small `functions` namespace. Most code
never builds one directly and instead uses the module-level
[`dispatch`][] registry: decorating an ordinary `#!python def` with it
turns that definition into an overload, and calling the resulting name
runs whichever registered overload fits the arguments best.

The two kinds of registry differ only in how they identify which
function a given definition belongs to. [`dispatch`][] keys a function
by its defining module together with its qualified name, so two modules
that each define `#!python @dispatch def area` end up with two separate
functions rather than one shared between them. A [`Dispatcher`][] built
directly keys by qualified name alone, with no regard to the module, so
several modules registering under the same name all extend one function
in common. That sharing across modules is how a single dispatched
function is assembled piece by piece from code spread across a package.
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
    """Find the qualified name a callable should be keyed by."""
    name = getattr(fn, "__qualname__", None)
    if name is None:
        name = getattr(fn, "__name__", None)
    if name is None:
        # A callable instance (an object with `__call__`) has no name of its
        # own; fall back to its type's, so it still keys stably.
        name = type(fn).__qualname__
    return name


def _module_of(fn: tx.Any) -> tx.Optional[str]:
    """Return the module a callable was defined in, or `None` if it has none.
    """
    return getattr(fn, "__module__", None)


def _is_impl(value: tx.Any) -> bool:
    """Report whether `value` is an implementation, not a hint overlay.

    Registration accepts two shapes for its first argument: an overlay of
    hints, written as a `#!python tuple` of positional hints or a
    `#!python dict` of named ones, or the implementation itself, which
    can be a plain function, a class, or any other callable object.
    """
    return callable(value) and not isinstance(value, (tuple, dict))


def _caller_module() -> tx.Optional[str]:
    """Return the `#!python __name__` of the module two frames up, if any.

    This is always called from within a dunder method of the `functions`
    namespace, so the code that triggered the lookup sits two frames
    above it, above both this helper and the dunder method calling it.
    Naming that caller's module is only meaningful for the module-level
    [`dispatch`][] registry, whose functions are keyed per module; a
    [`Dispatcher`][] built directly has no use for it.
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

    Naming a function this way, alongside the plain bare-name spelling,
    names its module explicitly rather than inferring it from the
    calling frame. Explicit naming is what a cross-module re-export of a
    function needs, since the frame that performs the lookup there is not
    the frame that registered the function.
    """
    return (
        isinstance(key, tuple)
        and len(key) == 2
        and isinstance(key[0], str)
        and isinstance(key[1], str)
    )


class _Functions:
    """A namespace of a dispatcher's functions, reached by name.

    Only the mapping protocol is exposed: `#!python view["area"]`,
    `#!python view.area`, `#!python "area" in view`,
    `#!python for name in view`, and `#!python len(view)`. No named
    method of its own is defined, which matters because it means a
    function called `register`, `items`, `map`, or anything else could
    still be reached through this view without colliding with a method
    of the same name.

    Both item access and attribute access create a function on demand:
    naming one that does not yet exist brings an empty one into being, so
    a registry and whatever code reaches into it can agree on a function
    by name regardless of which one runs first. Attribute access skips
    names starting with `#!python _`, so that a REPL or a tool probing
    for dunder attributes never creates an empty function as a side
    effect; item access carries no such restriction, so a function whose
    name does begin with an underscore is still reachable through
    `#!python view["_x"]`.

    Item access additionally accepts a `#!python (module, name)` pair,
    which names the module explicitly instead of inferring it from the
    calling frame. That pair is how a module other than the one that
    registered a function reaches it through the module-level registry.
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
    """A registry that groups overloads into named [`Function`][]s.

    Decorating an ordinary `#!python def` with a `Dispatcher` registers
    it as an overload rather than binding the name to that one plain
    function; a second `#!python def` given the same name adds a second
    overload to the same name instead of replacing the first. In both
    cases the decorator hands back the [`Function`][] the overload joined,
    so the decorated name ends up bound to the dispatched function rather
    than to the last individual implementation.

    A `Dispatcher` built this way keys each function purely by its
    qualified name, with no reference to the module it was registered
    from, so several modules that register under the same name all
    contribute overloads to the same shared function.

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
        self._functions: tx.Dict[tx.Any, Function] = {}
        self._view = _Functions(self)

    @property
    def functions(self) -> _Functions:
        """The namespace through which this dispatcher's functions are reached.

        `#!python d.functions.area` and `#!python d.functions["area"]`
        both return the function named `#!python "area"`, creating an
        empty one first if none exists yet under that name. The view
        defines no named method of its own, so no function name can ever
        collide with one.

        Attribute access skips names beginning with `#!python _`, so
        that a REPL or a tool probing for dunder attributes never
        creates an empty function as a side effect. Item access carries
        no such restriction, so `#!python d.functions["_x"]` still names
        a function whose own name starts with an underscore. This is why
        an anonymous overload, such as `#!python @dispatch def _` or a
        bare `#!python lambda`, should be avoided on the module-level
        [`dispatch`][] registry. Every anonymous overload shares the same
        `#!python "_"` or `#!python "<lambda>"` name and so collapses
        into one function regardless of where it was written. Giving each
        overload a real name, or overlaying hints onto a named
        `#!python def`, avoids that collision.
        """
        return self._view

    def __call__(self, *args: tx.Any, **options: tx.Any) -> tx.Any:
        """Register an overload, or return a decorator that does.

        Applied bare to a `#!python def`, as `#!python @d`, it registers
        that function immediately, dispatched on the parameter hints it
        already carries, and returns the [`Function`][] it now belongs
        to. Applied with an overlay of hints instead, as
        `#!python @d((int,), {"scale": float})`, or with options such as
        `#!python @d(priority=5)`, it returns a decorator, which performs
        the registration once it is applied to the function, using
        whichever hints were laid over its parameters.

        The overlay follows the same convention as
        [`Function.register`][]: positional hints go in a tuple, named
        hints in a dict, and any keyword argument is always an option
        (currently only `#!python priority`) rather than a hint.
        """
        return self.register(*args, **options)

    def register(self, *args: tx.Any, **options: tx.Any) -> tx.Any:
        """Register an overload, or return a decorator that does.

        Calling `register` has exactly the same effect as calling the
        dispatcher directly with `#!python @d`; see [`__call__`][] for
        the full behaviour. It hands back the [`Function`][] the overload
        just joined, so a decorated name is bound to the dispatched
        function rather than to the plain implementation underneath it.
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

        Registration only ever makes sense given a callable, or a tuple
        or dict overlay of hints, as its first argument. Anything else,
        such as a plain string or a number, would eventually fail deeper
        inside registration in a way that is harder to trace back to the
        mistake, so it is rejected here instead, with a message that
        points toward the correct spelling.
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

        The function a registration targets is created before its
        overload is validated, so a validation failure would otherwise
        leave a permanently empty function sitting in the namespace. This
        is called only on behalf of a registration that created the
        function itself, and even then it only removes the function when
        it is still the very same object and still has no methods, so a
        concurrent registration that has since added to it is left
        undisturbed.
        """
        with self._lock:
            if (
                self._functions.get(key) is function
                and not function.methods
            ):
                del self._functions[key]

    def clear_cache(self) -> None:
        """Drop every function's dispatch cache.

        Registration already invalidates a function's cache
        automatically, and so does a change to the ABC registry, so this
        method is rarely needed in practice. It stays available for the
        remaining case: a change to what a type is applicable to that
        nothing else in the dispatcher observes on its own.
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
        """Return the function named `name`, creating an empty one if needed.
        """
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
        """Return the function stored at `key`, and whether it was just
        created.

        The second value comes back `#!python True` only when `key` was
        previously absent and a fresh, empty function had to be created
        for it, letting a caller distinguish a function it just brought
        into being from one that already existed, however it got there.
        """
        with self._lock:
            function = self._functions.get(key)
            if function is None:
                function = Function(name)
                self._functions[key] = function
                return function, True
            return function, False


class _ModuleDispatcher(Dispatcher):
    """The class behind the module-level [`dispatch`][] registry.

    Everything about this matches an ordinary [`Dispatcher`][] built
    directly, with one difference: a function is keyed by its defining
    module together with its qualified name, so the same name defined in
    two different modules names two separate functions rather than one
    shared function.

    Decorating a `#!python def` registers an overload; a second
    `#!python def` of the same name in the same module adds a second
    overload to that same function.

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

    The overlay form lays explicit hints over a function's parameters
    instead of reading them from its annotations, with positional hints
    given as a tuple and named hints as a dict, alongside
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
        # A bare name is resolved against the caller's module here, so
        # `dispatch.functions.area` names this module's own `area`. (For the
        # module-level registry, prefer the value the decorator returns; the
        # namespace is unambiguous on a `Dispatcher` built directly.)
        return (module, name)

    def _names_for(self, module: tx.Optional[str]) -> tx.Iterator[str]:
        return iter(
            [name for (mod, name) in self._functions if mod == module]
        )


dispatch = _ModuleDispatcher()
"""The ready-made registry most code uses for multiple dispatch.

Decorating a `#!python def` with `#!python @dispatch` turns it into a
registered overload; a second `#!python def` sharing its name in the
same module becomes a second overload of the same function rather than
replacing the first. Either way, `#!python @dispatch` hands back the
[`Function`][] the overload belongs to, so the decorated name stays
bound to the dispatched function instead of to one individual
implementation. Functions are keyed by their defining module together
with their qualified name, so the same name defined in a different
module names a function of its own.

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

The overlay form lays explicit hints over a function's parameters
instead of reading them from its annotations, with positional hints
given as a tuple and named hints as a dict, alongside
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
