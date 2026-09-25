"""Tests for the registries: `Dispatcher`, `dispatch`, the `functions` view."""

# stdlib
import types
import warnings

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import (
    AmbiguousMethodError,
    Dispatcher,
    Function,
    NoMethodError,
    dispatch,
)
from bagof.dispatchers._dispatcher import _ModuleDispatcher


def _named(
    module: str, qualname: str, ret: tx.Any, hint: type = int
) -> tx.Callable[..., tx.Any]:
    """A one-parameter function pretending to live in `module` as `qualname`.

    Its qualified name is set explicitly, so it keys the way a module-level
    ``def`` does (qualname equal to its bare name) rather than carrying the
    ``<locals>`` a function nested in a test would.
    """

    def impl(x: hint) -> tx.Any:  # type: ignore[valid-type]
        return ret

    impl.__module__ = module
    impl.__qualname__ = qualname
    impl.__name__ = qualname.rsplit(".", 1)[-1]
    impl.__annotations__ = {"x": hint, "return": object}
    return impl


# --- module-level `dispatch`: per-module isolation ---------------------


def test_dispatch_isolates_by_module() -> None:
    """The same qualname in two modules is two independent functions."""
    a = dispatch(_named("synthetic_a", "area", "from-a"))
    b = dispatch(_named("synthetic_b", "area", "from-b"))
    assert a is not b
    assert a(1) == "from-a"
    assert b(1) == "from-b"


def test_dispatch_same_module_extends_one_function() -> None:
    """Two overloads registered from one module join one function."""
    first = dispatch(_named("synthetic_same", "render", "int", hint=int))
    second = dispatch(_named("synthetic_same", "render", "str", hint=str))
    assert first is second
    assert first(3) == "int"
    assert first("x") == "str"


def test_dispatch_returns_the_function() -> None:
    """A bare `@dispatch` returns the `Function`, not the raw callable."""

    @dispatch
    def synthetic_returns(x: int) -> int:
        return x

    assert isinstance(synthetic_returns, Function)
    assert synthetic_returns(5) == 5


# --- constructed Dispatcher: shared across modules ---------------------


def test_constructed_shares_across_modules() -> None:
    """A `Dispatcher` keys by qualname alone, so modules compose.

    Two overloads with the same qualified name but different modules join one
    function -- the shared, cross-module generic function pattern.
    """
    registry = Dispatcher()
    first = registry(_named("mod_a", "combine", "a", hint=int))
    second = registry(_named("mod_b", "combine", "b", hint=str))
    assert first is second  # same qualname -> one function
    assert first(3) == "a"
    assert first("x") == "b"


def test_module_dispatcher_is_a_dispatcher() -> None:
    """The module-level `dispatch` is a `Dispatcher` instance."""
    assert isinstance(dispatch, Dispatcher)
    assert isinstance(dispatch, _ModuleDispatcher)


# --- the `functions` namespace ----------------------------------------


def test_functions_get_or_create_and_identity() -> None:
    """Attribute and item access return the one function of that name."""
    registry = Dispatcher()
    made = registry.functions.area
    assert isinstance(made, Function)
    assert registry.functions["area"] is made
    assert registry.functions.area is made


def test_functions_compose_with_registration() -> None:
    """A function taken by name and one registered later are the same.

    A module-level ``def``'s qualified name equals its bare name, so a
    registration keys the same function the namespace hands out by name.
    """
    registry = Dispatcher()
    handle = registry.functions.area
    registered = registry(_named("shapes", "area", "an area", hint=int))
    assert registered is handle
    assert handle(3) == "an area"


def test_function_named_like_an_attribute_does_not_collide() -> None:
    """A function may be named `register`, `items`, ... without clashing."""
    registry = Dispatcher()
    for name in ("register", "items", "functions", "clear_cache", "dispatch"):
        function = registry.functions[name]
        assert isinstance(function, Function)
        assert registry.functions[name] is function
    # The dispatcher's own methods are untouched by those names.
    assert callable(registry.register)
    assert callable(registry.clear_cache)


def test_functions_membership_iteration_length() -> None:
    """The view supports `in`, iteration and `len`."""
    registry = Dispatcher()
    _ = registry.functions.alpha
    _ = registry.functions.beta
    assert "alpha" in registry.functions
    assert "missing" not in registry.functions
    assert set(registry.functions) == {"alpha", "beta"}
    assert len(registry.functions) == 2


def test_functions_ignores_underscored_attributes() -> None:
    """Attribute access does not mint a function for a private probe."""
    registry = Dispatcher()
    with pytest.raises(AttributeError):
        _ = registry.functions._owner
    with pytest.raises(AttributeError):
        _ = registry.functions.__wrapped__
    assert len(registry.functions) == 0
    # Item access is explicit, so a leading underscore is allowed there.
    assert isinstance(registry.functions["_x"], Function)


def test_module_functions_view_keys_by_caller_module() -> None:
    """`dispatch.functions.name` names the function of the calling module."""
    view_function = dispatch.functions.surface_probe
    assert isinstance(view_function, Function)
    # A module-level def in *this* module keys the same (module, name).
    registered = dispatch(_named(__name__, "surface_probe", 7))
    assert dispatch.functions["surface_probe"] is registered
    assert view_function is registered


# --- the overlay decorator forms --------------------------------------


def test_overlay_positional_and_named_hints() -> None:
    """`@d((int,), {"factor": int})` lays hints over the parameters."""
    registry = Dispatcher()

    @registry((int,), {"factor": int})
    def scale(value: tx.Any, factor: tx.Any) -> tx.Any:
        return value * factor

    assert isinstance(scale, Function)
    assert scale(3, factor=4) == 12


def test_overlay_priority_breaks_a_tie() -> None:
    """`priority` given to the overlay form breaks an otherwise-tie."""
    registry = Dispatcher()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        @registry((float, object))
        def _(a: tx.Any, b: tx.Any) -> str:
            return "left"

        @registry((object, float), priority=5)
        def _(a: tx.Any, b: tx.Any) -> str:  # noqa: F811 -- an overload
            return "right"

    resolved = _  # the decorator returns the shared Function
    assert isinstance(resolved, Function)
    assert resolved(1.0, 2.0) == "right"


def test_overlay_named_hint_as_keyword_is_rejected() -> None:
    """A named hint passed as a keyword points at the dict form."""
    registry = Dispatcher()

    with pytest.raises(TypeError, match="dict"):

        @registry(scale=float)
        def _(value: tx.Any, scale: tx.Any) -> tx.Any:
            return value


def test_bare_class_registers_on_init() -> None:
    """A class registered bare dispatches on its `__init__`."""
    registry = Dispatcher()

    class Boxed:
        def __init__(self, value: int) -> None:
            self.value = value

    function = registry(Boxed)
    assert isinstance(function, Function)
    assert function(5).value == 5


# --- end-to-end via the public API ------------------------------------


def test_end_to_end_dispatch() -> None:
    """A most-specific overload runs; a plain one is the fallback."""
    registry = Dispatcher()

    @registry
    def describe(x: object) -> str:
        return "anything"

    @registry
    def describe(x: int) -> str:  # noqa: F811 -- an overload
        return "an int"

    assert describe(3) == "an int"
    assert describe("s") == "anything"


def test_end_to_end_no_method() -> None:
    """A call nothing accepts raises `NoMethodError` (a `TypeError`)."""
    registry = Dispatcher()

    @registry
    def only_ints(x: int) -> int:
        return x

    with pytest.raises(NoMethodError) as info:
        only_ints("not an int")
    assert isinstance(info.value, TypeError)
    assert info.value.function == "only_ints"


def test_end_to_end_ambiguous() -> None:
    """Two equally specific overloads raise `AmbiguousMethodError`."""
    registry = Dispatcher()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        @registry((float, object))
        def _(a: tx.Any, b: tx.Any) -> str:
            return "left"

        @registry((object, float))
        def _(a: tx.Any, b: tx.Any) -> str:  # noqa: F811 -- an overload
            return "right"

    resolved = _  # the decorator returns the shared Function
    with pytest.raises(AmbiguousMethodError):
        resolved(1.0, 2.0)


def test_clear_cache_runs() -> None:
    """`clear_cache` drops every function's cache without error."""
    registry = Dispatcher()

    @registry
    def go(x: int) -> int:
        return x

    assert go(1) == 1
    registry.clear_cache()
    assert go(2) == 2


def test_dispatcher_repr() -> None:
    """The dispatcher's repr names how many functions it holds."""
    registry = Dispatcher()
    _ = registry.functions.a
    _ = registry.functions.b
    assert "2 function" in repr(registry)


def test_functions_repr_names_the_functions() -> None:
    """The view's repr lists the function names it holds."""
    registry = Dispatcher()
    _ = registry.functions.alpha
    assert repr(registry.functions) == "functions('alpha')"


def test_functions_contains_rejects_non_string() -> None:
    """A non-string is never a member of the functions view."""
    registry = Dispatcher()
    assert 123 not in registry.functions


def test_module_functions_iterate_and_count() -> None:
    """The module-level view iterates and counts the caller module's names."""
    dispatch(_named(__name__, "iter_probe_unique", 1))
    assert "iter_probe_unique" in set(dispatch.functions)
    assert len(dispatch.functions) >= 1


def test_callable_instance_and_name_fallbacks() -> None:
    """A callable with no qualname keys by its name, then its type's."""
    registry = Dispatcher()

    class Callable:
        def __call__(self, x: int) -> int:
            return x

    by_type = registry(Callable())  # no __name__/__qualname__ -> type's name
    assert isinstance(by_type, Function)

    named = Callable()
    named.__name__ = "named_instance"  # a __name__ but still no __qualname__
    by_name = registry(named)
    assert isinstance(by_name, Function)
    assert by_type is not by_name


def test_synthetic_module_registration_is_isolated() -> None:
    """A constructed dispatcher composes where the module-level keeps apart.

    Two fabricated modules registering into one `Dispatcher` compose by
    qualname, while the module-level `dispatch` keeps them apart -- the
    guarantee the two registries make.
    """
    types.ModuleType("fab_x")  # fabricated only to make the intent concrete
    shared = Dispatcher()
    fx = _named("fab_x", "op", "x", hint=int)
    fy = _named("fab_y", "op", "y", hint=str)
    assert shared(fx) is shared(fy)  # constructed: one function, two overloads
    assert dispatch(fx) is not dispatch(fy)  # module-level: two functions


# --- key validation and the qualified (module, name) lookup -----------


def test_functions_getitem_rejects_a_non_name() -> None:
    """Item access takes a name or a (module, name) pair, nothing else."""
    registry = Dispatcher()
    with pytest.raises(TypeError):
        _ = registry.functions[3]
    with pytest.raises(TypeError):
        _ = registry.functions[object()]
    with pytest.raises(TypeError):
        _ = registry.functions[("only-one",)]  # not a 2-tuple
    with pytest.raises(TypeError):
        _ = registry.functions[(1, 2)]  # not two strings
    assert len(registry.functions) == 0  # a rejected key mints nothing


def test_functions_qualified_lookup_reaches_a_cross_module_function() -> None:
    """A (module, name) pair names the module-level function explicitly.

    The qualified spelling bypasses frame resolution, so a function
    registered under one module is reachable from another -- what a
    cross-module re-export needs.
    """
    impl = _named("pkg.impl_qtest", "area", "an area", hint=int)
    registered = dispatch(impl)
    assert dispatch.functions["pkg.impl_qtest", "area"] is registered
    assert ("pkg.impl_qtest", "area") in dispatch.functions
    # A different module is a different key, so it does not reach it.
    assert ("pkg.other_qtest", "area") not in dispatch.functions
    # A bare non-pair is never a member.
    assert 3 not in dispatch.functions


def test_qualified_getitem_get_or_creates() -> None:
    """A qualified pair for an unknown function mints it under that key."""
    made = dispatch.functions["pkg.fresh_qtest", "brand_new"]
    assert isinstance(made, Function)
    assert dispatch.functions["pkg.fresh_qtest", "brand_new"] is made


# --- repr counts every function, across modules -----------------------


def test_module_dispatcher_repr_counts_all_modules() -> None:
    """The module-level repr counts functions in every module, not `None`."""
    registry = _ModuleDispatcher()
    registry(_named("repr_m1", "a", 1))
    registry(_named("repr_m2", "b", 2))
    registry(_named("repr_m3", "c", 3))
    assert "3 function(s)" in repr(registry)


# --- eager overlay-arg validation -------------------------------------


def test_bad_first_arg_string_is_rejected_eagerly() -> None:
    """A bare string is neither hints nor a callable, so it is refused."""
    registry = Dispatcher()
    with pytest.raises(TypeError):
        registry("area")
    assert len(registry.functions) == 0


def test_bad_first_arg_number_is_rejected_eagerly() -> None:
    """A number cannot be an overload nor an overlay, so it is refused."""
    registry = Dispatcher()
    with pytest.raises(TypeError):
        registry(5)
    assert len(registry.functions) == 0


# --- a failed registration leaves no orphan function ------------------


def test_impl_extra_positional_leaves_no_orphan() -> None:
    """A too-long implementation call rolls back the empty function."""
    registry = Dispatcher()
    with pytest.raises(TypeError):
        registry(len, "extra")
    assert "len" not in registry.functions
    assert len(registry.functions) == 0


def test_impl_unknown_option_leaves_no_orphan() -> None:
    """A stray keyword rolls back the empty function."""
    registry = Dispatcher()
    with pytest.raises(TypeError):
        registry(len, scale=float)
    assert "len" not in registry.functions
    assert len(registry.functions) == 0


def test_failed_registration_keeps_a_namespace_first_handle() -> None:
    """A handle minted by namespace access survives a later failed register.

    The rollback drops only a function the failing call itself created; a
    pre-existing empty function reached earlier through the namespace is left
    in place, so the handle a caller is holding stays valid.
    """
    registry = Dispatcher()
    handle = registry.functions.render  # minted empty, before any register

    # `render` keys the same way `d.functions.render` does (qualname equal to
    # the bare name), so the failing call touches that very key.
    render = _named("synthetic_render", "render", ret=1)

    with pytest.raises(TypeError):
        registry(render, scale=float)  # a stray keyword: registration fails

    assert registry.functions.render is handle
    assert "render" in registry.functions


def test_overlay_bad_named_hint_leaves_no_orphan() -> None:
    """A hint for a parameter the function lacks rolls back the function."""
    registry = Dispatcher()
    with pytest.raises(TypeError):

        @registry((int,), {"nope": int})
        def _(value: tx.Any) -> tx.Any:
            return value

    assert len(registry.functions) == 0


# --- metadata adoption keeps an explicit name -------------------------


def test_first_impl_metadata_is_adopted_with_an_explicit_name() -> None:
    """A named function adopts its first impl's doc, module and wrapped.

    A function reached by name (through the namespace) has an explicit name;
    it still takes the rest of its metadata from the first implementation, so
    it stands in for it to ``help`` and introspection.
    """
    reg = Function("render")

    def impl(x: int) -> str:
        """the render doc"""
        return f"int:{x}"

    _ = reg.register(impl)
    assert reg.__name__ == "render"  # the explicit name is kept
    assert reg.__doc__ == "the render doc"
    assert reg.__module__ == impl.__module__
    assert reg.__wrapped__ is impl


def test_anonymous_impl_does_not_rename_a_named_function() -> None:
    """`@items.register def _` keeps the function named `items`."""
    items = Function("items")

    @items.register
    def _(x: int) -> str:
        return "one item"

    assert items.name == "items"
    assert items.__name__ == "items"
    assert items.__qualname__ == "items"


def test_namespace_first_function_adopts_via_the_dispatcher() -> None:
    """A name taken first, then registered, adopts the impl's metadata.

    The README headline pattern: reach an empty function by name, then
    register the real ``def`` -- the function ends up named and documented.
    """
    registry = Dispatcher()
    render = registry.functions.render
    impl = _named("shapes_meta", "render", "rendered", hint=int)
    impl.__doc__ = "render an int"
    result = registry(impl)
    assert result is render
    assert render.__name__ == "render"
    assert render.__doc__ == "render an int"
    assert render.__module__ == "shapes_meta"
    assert render.__wrapped__ is impl


# --- the empty-function hint names the cross-module case --------------


def test_empty_function_hint_mentions_cross_module_reach() -> None:
    """Calling an empty function points at both reasons it can be empty."""
    empty = Function("area")
    with pytest.raises(NoMethodError) as info:
        empty(3)
    message = str(info.value)
    assert "has not been imported" in message
    assert "different module" in message
