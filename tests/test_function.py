"""Tests for the dispatch engine (`_function.py`)."""

# stdlib
import typing
import warnings

# dependencies
import pytest

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._errors import AmbiguousMethodError, NoMethodError
from bagof.dispatchers._function import Function
from bagof.dispatchers._method import Method
from bagof.dispatchers._signature import Signature


def _quiet_register(function: Function, *fns: typing.Any) -> None:
    """Register several functions, ignoring ambiguity warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for fn in fns:
            function.register(fn)


# --- basic dispatch ----------------------------------------------------


def test_dispatch_picks_by_type() -> None:
    """A call runs the method whose parameter type accepts the argument."""
    f = Function("describe")

    def an_int(x: int) -> str:
        return "int"

    def a_str(x: str) -> str:
        return "str"

    f.register(an_int)
    f.register(a_str)
    assert f(3) == "int"
    assert f("a") == "str"


def test_most_specific_wins() -> None:
    """A subclass method beats a base-class method for a subclass value."""
    f = Function("f")

    def base(x: object) -> str:
        return "object"

    def specific(x: int) -> str:
        return "int"

    f.register(base)
    f.register(specific)
    assert f(3) == "int"
    assert f("a") == "object"


def test_definition_order_does_not_matter() -> None:
    """Selection is independent of the order methods were registered."""
    first = Function("f")
    _quiet_register(first, lambda x: "obj", _int_method())
    second = Function("f")
    _quiet_register(second, _int_method(), lambda x: "obj")
    # Both resolve a bool the same way: bool <= int <= object.
    assert first.dispatch(True).name == second.dispatch(True).name


def _int_method() -> typing.Callable[[int], str]:
    def m(x: int) -> str:
        return "int"

    return m


def test_dispatch_returns_method_without_calling() -> None:
    """`dispatch` returns the chosen method and does not run it."""
    ran = []
    f = Function("f")

    def m(x: int) -> None:
        ran.append(x)

    f.register(m)
    method = f.dispatch(3)
    assert isinstance(method, Method)
    assert ran == []
    assert method(3) is None
    assert ran == [3]


def test_call_is_dispatch_plus_call() -> None:
    """Calling the function runs the chosen method on the same arguments."""
    f = Function("add")

    def add(x: int, y: int) -> int:
        return x + y

    f.register(add)
    assert f(2, 3) == 5


# --- errors ------------------------------------------------------------


def test_no_method_raises() -> None:
    """A call nothing accepts raises `NoMethodError`."""
    f = Function("f")
    f.register(lambda x: x, )
    # A registered catch-all lambda takes one positional; two args fit nothing.
    with pytest.raises(NoMethodError):
        f(1, 2)


def test_no_method_is_a_type_error() -> None:
    """`NoMethodError` is a `TypeError`, so old guards still catch it."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    f.register(m)
    with pytest.raises(TypeError):
        f("not an int")


def test_no_methods_yet_message() -> None:
    """A function with no methods says the module was not imported."""
    f = Function("area")
    with pytest.raises(NoMethodError, match="has no methods yet"):
        f(1)


def test_ambiguous_raises() -> None:
    """Two incomparable equally specific methods raise the ambiguity error."""
    f = Function("g")

    def by_first(x: float, y: object) -> int:
        return 1

    def by_second(x: object, y: float) -> int:
        return 2

    _quiet_register(f, by_first, by_second)
    with pytest.raises(AmbiguousMethodError):
        f(2.0, 3.0)


# --- priority tie-break ------------------------------------------------


def test_priority_breaks_a_tie() -> None:
    """`priority` chooses between otherwise equally specific methods."""
    f = Function("g")

    def low(x: float, y: object) -> str:
        return "low"

    def high(x: object, y: float) -> str:
        return "high"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register(low)
        f.register(high, priority=1)
    assert f(2.0, 3.0) == "high"


def test_priority_does_not_override_strict_specificity() -> None:
    """A higher priority never beats a strictly more specific method."""
    f = Function("f")

    def general(x: object) -> str:
        return "object"

    def specific(x: int) -> str:
        return "int"

    f.register(general, priority=5)
    f.register(specific)
    assert f(3) == "int"


# --- MRO refinement (the diamond) --------------------------------------


def test_diamond_resolves_by_mro() -> None:
    """A diamond `D(B, C)` resolves to `B`, as single dispatch does."""

    class B:
        pass

    class C:
        pass

    class D(B, C):
        pass

    f = Function("f")

    def for_b(x: B) -> str:
        return "B"

    def for_c(x: C) -> str:
        return "C"

    _quiet_register(f, for_b, for_c)
    assert f(D()) == "B"


# --- tightness tie-break -----------------------------------------------


def test_fixed_arity_beats_varargs_tail() -> None:
    """A fixed-arity method beats one that absorbs the call in `*args`."""
    f = Function("f")

    def fixed(x: int, y: int) -> str:
        return "fixed"

    def tail(x: int, *rest: int) -> str:
        return "tail"

    f.register(fixed)
    f.register(tail)
    assert f(1, 2) == "fixed"


def test_fewer_defaults_wins() -> None:
    """Between two matches, the one filling fewer defaults is tighter."""
    f = Function("f")

    def exact(x: int, y: int) -> str:
        return "exact"

    def with_default(x: int, y: int = 0, z: int = 0) -> str:
        return "default"

    f.register(exact)
    f.register(with_default)
    assert f(1, 2) == "exact"


# --- name-aware binding ------------------------------------------------


def test_keyword_and_positional_choose_the_same_method() -> None:
    """The same parameter, positional in one call and keyword in another."""
    f = Function("f")

    def m(shape: int, scale: float = 1.0) -> str:
        return "m"

    f.register(m)
    assert f(3, 2.0) == "m"
    assert f(3, scale=2.0) == "m"


def test_keyword_only_dispatched_by_name() -> None:
    """A keyword-only parameter is dispatched, never filled positionally."""
    f = Function("f")

    def m(x: int, *, mode: str) -> str:
        return mode

    f.register(m)
    assert f(1, mode="fast") == "fast"
    with pytest.raises(NoMethodError):
        f(1, "fast")  # mode cannot be filled positionally


def test_positional_only_not_bound_by_keyword() -> None:
    """A positional-only parameter cannot be passed by keyword."""
    f = Function("p")
    method = Method(
        lambda v: v,
        Signature.from_hints(int),  # a single positional-only parameter
    )
    f._add(method)
    assert f(3) == 3
    with pytest.raises(NoMethodError):
        f(_0=3)


def test_different_shapes_are_distinct() -> None:
    """`f(1, 2)` and `f(1, y=2)` are different shapes with their own keys."""
    f = Function("f")

    def m(x: int, y: int) -> str:
        return "m"

    f.register(m)
    assert f(1, 2) == "m"
    assert f(1, y=2) == "m"


# --- kwargs ------------------------------------------------------------


def test_extra_keyword_checked_against_varkw() -> None:
    """An extra keyword is checked against `**kwargs: H`."""
    f = Function("f")

    def m(x: int, **rest: str) -> str:
        return "m"

    f.register(m)
    assert f(1, extra="ok") == "m"
    with pytest.raises(NoMethodError):
        f(1, extra=2)  # 2 is not a str


# --- register overlay (#13) --------------------------------------------


def test_register_overlays_positional_hint() -> None:
    """`register((hint,))` overlays the hint onto the first parameter."""
    f = Function("f")

    @f.register((int,))
    def _(x) -> str:  # noqa: ANN001 -- overlaid by the register hint
        return "int"

    assert f(3) == "int"
    with pytest.raises(NoMethodError):
        f("a")


def test_register_overlay_keeps_names_and_defaults() -> None:
    """Overlaying keeps the function's parameter names, kinds and defaults."""
    f = Function("f")

    @f.register((int,), {"scale": float})
    def _(shape, scale=1.0) -> tuple:  # noqa: ANN001
        return (shape, scale)

    assert f(3) == (3, 1.0)  # the default survives
    assert f(3, scale=2.0) == (3, 2.0)


def test_register_overlay_by_name() -> None:
    """A named hint (a dict) overlays the parameter it names."""
    f = Function("f")

    @f.register({"y": int})
    def _(x, y) -> str:  # noqa: ANN001
        return "m"

    assert f("anything", 3) == "m"
    with pytest.raises(NoMethodError):
        f("anything", "not int")


def test_register_overlay_returns_the_implementation() -> None:
    """The hint-overlay decorator returns the wrapped callable, not `self`."""
    f = Function("f")

    def impl(x) -> str:  # noqa: ANN001
        return "m"

    returned = f.register((int,))(impl)
    assert returned is impl  # singledispatch convention: the callable


def test_register_overlay_no_hints_uses_own_signature() -> None:
    """`register()` with no hints registers by the function's own signature."""
    f = Function("f")

    @f.register()
    def _(x: int) -> str:
        return "int"

    assert f(3) == "int"
    with pytest.raises(NoMethodError):
        f("a")


def test_register_too_many_hints() -> None:
    """More positional hints than parameters is a `TypeError`."""
    f = Function("f")

    def one(x) -> None:  # noqa: ANN001
        ...

    with pytest.raises(TypeError, match="positional parameter"):
        f.register((int, str))(one)


def test_register_unknown_named_hint() -> None:
    """A named hint for a parameter that does not exist is a `TypeError`."""
    f = Function("f")

    def one(x) -> None:  # noqa: ANN001
        ...

    with pytest.raises(TypeError, match="no parameter"):
        f.register({"missing": int})(one)


def test_register_direct_returns_the_function_object() -> None:
    """`register(fn)` registers directly and returns the function."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    returned = f.register(m)
    assert returned is m
    assert f(3) == 3


def test_register_class_dispatches_on_its_constructor() -> None:
    """`register(SomeClass)` registers the class on its constructor signature.

    A type is a callable, so it is an implementation -- dispatched on its
    `__init__`, not read as a hint (#14).
    """
    f = Function("f")

    class Widget:
        def __init__(self, size: int) -> None:
            self.size = size

    def fallback(x: object) -> str:
        return "object"

    returned = f.register(Widget)
    assert returned is Widget
    f.register(fallback)
    # A call whose first argument is an int selects the Widget constructor.
    made = f(3)
    assert isinstance(made, Widget) and made.size == 3
    assert f("a") == "object"


def test_register_int_is_an_implementation_not_a_hint() -> None:
    """`register(int)` registers the `int` type itself as an implementation."""
    f = Function("f")
    returned = f.register(int)
    assert returned is int
    # It dispatches on int's constructor: f("3") builds int("3") == 3.
    assert f("3") == 3
    assert len(f.methods) == 1
    assert f.methods[0].function is int


def test_register_callable_instance() -> None:
    """A callable instance registers, dispatched on its `__call__`."""
    f = Function("f")

    class Doubler:
        def __call__(self, x: int) -> int:
            return x * 2

    doubler = Doubler()
    returned = f.register(doubler)
    assert returned is doubler
    assert f(5) == 10


def test_register_impl_with_extra_positional_is_an_error() -> None:
    """An implementation plus a stray positional is a `TypeError`."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="single implementation"):
        f.register(m, int)


def test_register_non_callable_non_hint_is_an_error() -> None:
    """A first argument that is neither callable nor hints is a `TypeError`."""
    f = Function("f")
    with pytest.raises(TypeError, match="expected a callable"):
        f.register(3)


def test_register_two_tuples_is_an_error() -> None:
    """Two positional-hint tuples is a `TypeError`."""
    f = Function("f")

    def m(x) -> None:  # noqa: ANN001
        ...

    with pytest.raises(TypeError, match="at most one tuple"):
        f.register((int,), (str,))(m)


def test_register_two_dicts_is_an_error() -> None:
    """Two named-hint dicts is a `TypeError`."""
    f = Function("f")

    def m(x) -> None:  # noqa: ANN001
        ...

    with pytest.raises(TypeError, match="at most one dict"):
        f.register({"x": int}, {"x": str})(m)


def test_register_overlay_stray_argument_is_an_error() -> None:
    """A hint-overlay argument that is neither a tuple nor a dict errors."""
    f = Function("f")

    def m(x) -> None:  # noqa: ANN001
        ...

    with pytest.raises(TypeError, match="tuple .*and/or a dict"):
        f.register((int,), 5)(m)


def test_register_direct_with_priority() -> None:
    """`register(fn, priority=5)` registers directly at that priority."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    returned = f.register(m, priority=5)
    assert returned is m
    assert f.methods[0].priority == 5


def test_register_decorator_with_priority() -> None:
    """`register(priority=5)` is the no-hint decorator at a priority."""
    f = Function("f")

    @f.register(priority=5)
    def m(x: int) -> int:
        return x

    assert m.__name__ == "m"  # the wrapped function is returned
    assert f.methods[0].priority == 5


def test_register_overlay_with_priority() -> None:
    """A hint overlay can carry a priority alongside its hints."""
    f = Function("f")

    @f.register((int,), priority=3)
    def m(x) -> int:  # noqa: ANN001
        return x

    assert f.methods[0].priority == 3


def test_register_unknown_keyword_points_to_the_dict_form() -> None:
    """A stray keyword is an option error naming the dict form for hints."""
    f = Function("f")

    def m(scale) -> None:  # noqa: ANN001
        ...

    with pytest.raises(TypeError, match=r"unexpected keyword 'scale'"):
        f.register(scale=float)(m)
    with pytest.raises(TypeError, match=r"register\(\{'scale': float\}\)"):
        f.register(scale=float)(m)


def test_register_overlay_rejects_a_non_hint_element() -> None:
    """A tuple element that is not a real hint is a registration error (S1)."""
    f = Function("f")

    def m(x, y) -> None:  # noqa: ANN001
        ...

    # A 2-tuple is not a hint; neither is a bare number.
    with pytest.raises(TypeError, match="not a type or a typing construct"):
        f.register(((str, int),))(m)
    with pytest.raises(TypeError, match="not a type or a typing construct"):
        f.register((5,))(m)
    with pytest.raises(TypeError, match="not a type or a typing construct"):
        f.register({"x": 5})(m)


def test_register_overlay_accepts_parametrised_hints() -> None:
    """A parametrised hint (`Exact[int]`, `List[int]`) overlays fine (S1)."""
    f = Function("f")

    @f.register((Exact[int],))
    def m(x) -> str:  # noqa: ANN001
        return "exact"

    assert f(1) == "exact"
    with pytest.raises(NoMethodError):
        f(True)  # bool is not exactly int


def test_register_overlay_double_target_is_an_error() -> None:
    """Hinting a parameter both positionally and by name is an error (S3)."""
    f = Function("f")

    def m(x, y) -> None:  # noqa: ANN001
        ...

    with pytest.raises(TypeError, match="twice"):
        f.register((int,), {"x": str})(m)


def test_register_overlay_keeps_varargs_and_varkw_names() -> None:
    """Overlaying keeps the `*args` / `**kwargs` written names (S2)."""
    f = Function("f")

    @f.register((int,))
    def m(a, *items, **opts) -> None:  # noqa: ANN001, ANN002, ANN003
        ...

    body = f.methods[0].describe().split(" @ ")[0]
    assert "*items" in body and "**opts" in body


def test_register_overlay_hints_varargs_by_name() -> None:
    """A dict hint can target `*args`, setting its element hint (gap)."""
    f = Function("f")

    @f.register({"args": int})
    def m(a, *args) -> str:  # noqa: ANN001, ANN002
        return "m"

    assert f.methods[0].signature.varargs is int
    assert f("anything", 1, 2) == "m"
    with pytest.raises(NoMethodError):
        f("anything", "not int")


def test_register_overlay_hints_varkw_by_name() -> None:
    """A dict hint can target `**kwargs`, setting its value hint (gap)."""
    f = Function("f")

    @f.register({"kw": str})
    def m(**kw) -> str:  # noqa: ANN003
        return "m"

    assert f.methods[0].signature.varkw is str
    assert f(a="ok") == "m"
    with pytest.raises(NoMethodError):
        f(a=3)  # 3 is not a str


def test_register_replacement_warns() -> None:
    """Re-registering an identical signature replaces it with a warning."""
    f = Function("f")

    def first(x: int) -> str:
        return "first"

    def second(x: int) -> str:
        return "second"

    f.register(first)
    with pytest.warns(RuntimeWarning, match="replacing"):
        f.register(second)
    assert len(f.methods) == 1
    assert f(3) == "second"


# --- from_mapping ------------------------------------------------------


def test_from_mapping_tuple_keys() -> None:
    """`from_mapping` builds methods from tuple-of-hint keys."""
    f = Function.from_mapping({(int,): abs, (str,): len}, name="f")
    assert f(-3) == 3
    assert f("abcd") == 4


def test_from_mapping_single_hint_key() -> None:
    """A bare (non-tuple) key is one positional hint."""
    f = Function.from_mapping({int: abs}, name="f")
    assert f(-2) == 2


def test_from_mapping_signature_key() -> None:
    """A `Signature` key is used as-is."""
    f = Function.from_mapping(
        {Signature.from_hints(int, int): lambda a, b: a + b}, name="add"
    )
    assert f(2, 3) == 5


# --- __get__ descriptor ------------------------------------------------


def test_descriptor_binds_self() -> None:
    """A function used as a method dispatches with `self` as argument 0."""

    class Owner:
        greet = Function("greet")

    def _greet(self, name: str) -> str:  # noqa: ANN001 -- receiver
        return f"hi {name}"

    Owner.greet.register(_greet)
    assert Owner().greet("Sam") == "hi Sam"


def test_descriptor_on_class_returns_the_function() -> None:
    """Accessed on the class, the descriptor returns the function itself."""

    class Owner:
        m = Function("m")

    assert isinstance(Owner.m, Function)


def test_bound_dispatch_and_resolve() -> None:
    """A bound view exposes `dispatch` and `resolve` with `self` prepended."""

    class Owner:
        m = Function("m")

    def _m(self, x: int) -> int:  # noqa: ANN001 -- receiver
        return x

    Owner.m.register(_m)
    owner = Owner()
    bound = owner.m
    assert bound.dispatch(3).name == "_m"
    assert bound.resolve(int).name == "_m"
    assert "bound m" in repr(bound)


# --- resolve (hint level) ----------------------------------------------


def test_resolve_by_hint() -> None:
    """`resolve` chooses a method from query hints, not values."""
    f = Function("f")

    def an_int(x: int) -> str:
        return "int"

    def an_object(x: object) -> str:
        return "object"

    f.register(an_int)
    f.register(an_object)
    assert f.resolve(bool).name == "an_int"
    assert f.resolve(str).name == "an_object"


def test_resolve_default_when_no_match() -> None:
    """`resolve(default=...)` returns the default rather than raising."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    f.register(m)
    assert f.resolve(str, default=None) is None


def test_resolve_no_match_raises() -> None:
    """Without a default, `resolve` raises `NoMethodError`."""
    f = Function("f")
    f.register(_int_method())
    with pytest.raises(NoMethodError):
        f.resolve(str)


def test_resolve_ambiguity_raise() -> None:
    """An ambiguous hint resolution raises by default."""
    f = Function("g")

    def by_first(x: float, y: object) -> int:
        return 1

    def by_second(x: object, y: float) -> int:
        return 2

    _quiet_register(f, by_first, by_second)
    with pytest.raises(AmbiguousMethodError):
        f.resolve(float, float)


def test_resolve_ambiguity_warn_takes_first() -> None:
    """`ambiguity='warn'` takes the first registered method, with a warning."""
    f = Function("g")

    def by_first(x: float, y: object) -> int:
        return 1

    def by_second(x: object, y: float) -> int:
        return 2

    _quiet_register(f, by_first, by_second)
    with pytest.warns(RuntimeWarning):
        assert f.resolve(float, float, ambiguity="warn").name == "by_first"


def test_resolve_ambiguity_ignore_is_silent() -> None:
    """`ambiguity='ignore'` takes the first silently."""
    f = Function("g")

    def by_first(x: float, y: object) -> int:
        return 1

    def by_second(x: object, y: float) -> int:
        return 2

    _quiet_register(f, by_first, by_second)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f.resolve(float, float, ambiguity="ignore").name == "by_first"


# --- ambiguities() -----------------------------------------------------


def test_ambiguities_lists_incomparable_pairs() -> None:
    """`ambiguities` reports the guaranteed-ambiguous method pairs."""
    f = Function("g")

    def by_first(x: float, y: object) -> int:
        return 1

    def by_second(x: object, y: float) -> int:
        return 2

    _quiet_register(f, by_first, by_second)
    pairs = f.ambiguities()
    assert len(pairs) == 1
    names = {pairs[0][0].name, pairs[0][1].name}
    assert names == {"by_first", "by_second"}


def test_no_ambiguity_between_disjoint_types() -> None:
    """Methods on disjoint types (`int` vs `str`) are not ambiguous."""
    f = Function("f")

    def an_int(x: int) -> str:
        return "int"

    def a_str(x: str) -> str:
        return "str"

    f.register(an_int)
    f.register(a_str)
    assert f.ambiguities() == []


def test_registration_warns_on_guaranteed_ambiguity() -> None:
    """Registering a method guaranteed ambiguous with another warns."""
    f = Function("g")

    def by_first(x: float, y: object) -> int:
        return 1

    def by_second(x: object, y: float) -> int:
        return 2

    f.register(by_first)
    with pytest.warns(RuntimeWarning, match="ambiguous"):
        f.register(by_second)


def test_union_spelling_pair_warns_and_is_listed() -> None:
    """Two differently-spelled but equivalent unions are guaranteed ambiguous.

    `Union[int, str]` and `Union[str, int]` are equal but not written the same
    way, so both are kept; a call matching one matches the other with no most
    specific method, so registering the second warns and `ambiguities()` lists
    the pair (M2).
    """
    f = Function("f")

    def first(x: typing.Union[int, str]) -> int:
        return 1

    def second(x: typing.Union[str, int]) -> int:
        return 2

    f.register(first)
    with pytest.warns(RuntimeWarning, match="ambiguous"):
        f.register(second)
    assert len(f.ambiguities()) == 1
    with pytest.raises(AmbiguousMethodError):
        f(1)


def test_optional_spelling_pair_warns_and_is_listed() -> None:
    """`Optional[int]` and `Union[None, int]` are equivalent, distinct forms.

    Same as the union case: equal yet spelled differently, so guaranteed
    ambiguous -- a warning at registration and a listed pair (M2).
    """
    f = Function("f")

    def first(x: typing.Optional[int]) -> int:
        return 1

    def second(x: typing.Union[None, int]) -> int:
        return 2

    f.register(first)
    with pytest.warns(RuntimeWarning, match="ambiguous"):
        f.register(second)
    assert len(f.ambiguities()) == 1
    with pytest.raises(AmbiguousMethodError):
        f(1)


def test_strictly_ordered_pair_is_not_ambiguous() -> None:
    """A strictly-ordered pair (`int` below `object`) is not ambiguous.

    One method is unambiguously more specific, so registering the second does
    not warn and `ambiguities()` stays empty (M2: only a pair with no strict
    order is reported).
    """
    f = Function("f")

    def narrow(x: int) -> int:
        return 1

    def wide(x: object) -> int:
        return 2

    f.register(narrow)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        f.register(wide)  # strictly ordered -> no ambiguity warning
    assert f.ambiguities() == []


# --- Exact -------------------------------------------------------------


def test_exact_does_not_fire_for_subclass() -> None:
    """`Exact[int]` matches an int but not a bool."""
    f = Function("f")

    def exact_int(x: Exact[int]) -> str:
        return "exact int"

    def any_int(x: int) -> str:
        return "int"

    f.register(exact_int)
    f.register(any_int)
    assert f(3) == "exact int"
    assert f(True) == "int"  # bool is int but not exactly int


# --- name adoption -----------------------------------------------------


def test_name_adopted_from_first_method() -> None:
    """A nameless function takes its name from the first method registered."""
    f = Function()
    assert f.name == "<function>"

    def area(x: int) -> int:
        return x

    f.register(area)
    assert f.name == "area"
    assert f.__doc__ is area.__doc__  # metadata copied


def test_name_adoption_survives_a_partial() -> None:
    """Registering a `partial` first adopts nothing but does not fail."""
    import functools

    def base(tag: str, x: int) -> str:
        return f"{tag}:{x}"

    partial = functools.partial(base, "t")
    f = Function()
    f.register(partial)
    assert f(3) == "t:3"


# --- MRO refinement, non-class positions -------------------------------


def test_mro_refinement_with_any_position() -> None:
    """A non-class (`Any`) position gives no refinement but does not block one.

    The diamond is resolved by the class position; the `Any` position is
    equivalent on both sides, so it neither refines nor prevents the win.
    """

    class B:
        pass

    class C:
        pass

    class D(B, C):
        pass

    f = Function("f")

    def for_b(x: B, y: typing.Any) -> str:
        return "B"

    def for_c(x: C, y: typing.Any) -> str:
        return "C"

    _quiet_register(f, for_b, for_c)
    assert f(D(), 1) == "B"


def test_mro_none_position_blocks_domination() -> None:
    """When a position is non-class and unequal, neither method dominates."""
    f = Function("g")

    def by_x(x: float, y: typing.Any) -> int:
        return 1

    def by_y(x: typing.Any, y: float) -> int:
        return 2

    _quiet_register(f, by_x, by_y)
    with pytest.raises(AmbiguousMethodError):
        f(2.0, 3.0)


# --- keyword-shaped ambiguity + possible fix ---------------------------


def test_keyword_ambiguity_possible_fix() -> None:
    """An ambiguity from a keyword call renders the fix by keyword."""
    f = Function("f")

    def by_a(a: float, b: object) -> int:
        return 1

    def by_b(a: object, b: float) -> int:
        return 2

    _quiet_register(f, by_a, by_b)
    with pytest.raises(AmbiguousMethodError) as info:
        f(a=1.0, b=2.0)
    message = str(info.value)
    assert "a: float" in message
    assert "b: float" in message


def test_ambiguities_with_keyword_only() -> None:
    """`ambiguities` handles methods with keyword-only parameters."""
    f = Function("f")

    def by_x(x: float, *, mode: object) -> int:
        return 1

    def by_mode(x: object, *, mode: float) -> int:
        return 2

    _quiet_register(f, by_x, by_mode)
    assert len(f.ambiguities()) == 1


def _forward_ref_method(x: "DefinitelyNotDefinedYet") -> str:  # noqa: F821
    return "fwd"


def test_registration_tolerates_unresolved_forward_ref() -> None:
    """A method with a still-unresolved hint registers without forcing it.

    The registration-time ambiguity heuristic must not resolve a forward
    reference early, so deferred resolution keeps working.
    """
    f = Function("f")

    def concrete(x: int) -> str:
        return "int"

    f.register(concrete)
    # This one's annotation names something that does not exist yet; it
    # defers. Registration must not force it to resolve.
    f.register(_forward_ref_method)
    assert len(f.methods) == 2
    # The diagnostic tolerates the unresolved hint too, rather than raising.
    assert f.ambiguities() == []


def test_build_plan_returns_existing() -> None:
    """Building a plan for an already-planned shape returns the cached one."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    f.register(m)
    cache = f._refresh()
    shape = (1, ())
    first = f._build_plan(shape, cache)
    second = f._build_plan(shape, cache)
    assert first is second


# --- B1: structural replacement, not equivalence-collapse --------------


def _typevar_pair() -> "typing.Tuple[typing.Any, typing.Any]":
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    return T, U


def test_distinct_typevars_are_both_kept() -> None:
    """`(x: T, y: T)` and `(x: T, y: U)` are different methods, both kept.

    Both are equivalent to `(Any, Any)`, so `Signature.__eq__` (semantic
    equivalence) reports them equal; registration must instead compare the
    signatures *as written*, so the two coexist rather than one silently
    replacing the other. Being equivalent yet distinct, they are also
    guaranteed ambiguous, so registering the second warns about the ambiguity
    (never about a replacement).
    """
    T, U = _typevar_pair()
    f = Function("f")

    def same(x: T, y: T) -> str:
        return "same"

    def free(x: T, y: U) -> str:
        return "free"

    f.register(same)
    with pytest.warns(RuntimeWarning, match="ambiguous") as caught:
        f.register(free)  # warns about ambiguity, not replacement
    assert len(f.methods) == 2  # both kept -- neither replaced the other
    assert not any("replacing" in str(w.message) for w in caught)


def test_bound_typevar_and_plain_are_both_kept() -> None:
    """A `TypeVar(bound=int)` method and an `int` method are both kept.

    A bound TypeVar is *equivalent* to its bound, so `Signature.__eq__` reports
    the two signatures equal; registration must compare them structurally so
    the TypeVar method does not silently replace the plain one (or vice versa).
    Equivalent-yet-distinct, they are guaranteed ambiguous, so registering the
    second warns about the ambiguity, not a replacement.
    """
    TB = typing.TypeVar("TB", bound=int)
    f = Function("f")

    def tv(x: TB) -> str:
        return "tv"

    def plain(x: int) -> str:
        return "plain"

    f.register(tv)
    with pytest.warns(RuntimeWarning, match="ambiguous") as caught:
        f.register(plain)
    assert len(f.methods) == 2
    assert not any("replacing" in str(w.message) for w in caught)


def test_typevar_and_unannotated_are_both_kept() -> None:
    """`(x: T, y: T)` and an unannotated `(x, y)` are distinct spellings.

    Both reduce to `(Any, Any)`, so they are equivalent yet spelled
    differently: both kept, and guaranteed ambiguous (a warning, never a
    replacement).
    """
    T, _ = _typevar_pair()
    f = Function("f")

    def repeated(x: T, y: T) -> str:
        return "repeated"

    def bare(x, y) -> str:  # noqa: ANN001 -- unannotated, so (Any, Any)
        return "bare"

    f.register(repeated)
    with pytest.warns(RuntimeWarning, match="ambiguous") as caught:
        f.register(bare)
    assert len(f.methods) == 2
    assert not any("replacing" in str(w.message) for w in caught)


def test_identical_spelling_replaces_with_warning() -> None:
    """A genuine re-registration (same spelling) still replaces, and warns."""
    f = Function("f")

    def first(x: int, y: int) -> str:
        return "first"

    def second(x: int, y: int) -> str:
        return "second"

    f.register(first)
    with pytest.warns(RuntimeWarning, match="replacing"):
        f.register(second)
    assert len(f.methods) == 1
    assert f(1, 2) == "second"


def test_distinct_typevar_spellings_are_order_independent() -> None:
    """`(T,T),(T,U)` and `(T,U),(T,T)` give the same set of two methods."""
    T, U = _typevar_pair()

    def same(x: T, y: T) -> str:
        return "same"

    def free(x: T, y: U) -> str:
        return "free"

    forwards = Function("f")
    backwards = Function("f")
    with warnings.catch_warnings():
        # The equivalent-yet-distinct pair is ambiguous either way round; the
        # point here is that both are kept regardless of order, so the
        # ambiguity warning is not what is under test.
        warnings.simplefilter("ignore", RuntimeWarning)
        forwards.register(same)
        forwards.register(free)
        backwards.register(free)
        backwards.register(same)
    assert len(forwards.methods) == len(backwards.methods) == 2


# --- B2: cross-argument MRO conflict stays ambiguous -------------------


def test_exact_object_vs_int_int_is_ambiguous() -> None:
    """`(Exact[int], object)` vs `(int, int)` on `f(3, 3)` is ambiguous.

    The first is strictly more specific at argument 0 (`Exact[int] < int`), the
    second at argument 1 (`int < object`): a cross-argument conflict with no
    most specific method. MRO refinement must not break the tie by reading
    `Exact[int]` as `int` at position 0 and letting the second win on 1 alone.
    """
    f = Function("f")

    def exact_first(x: Exact[int], y: object) -> str:
        return "exact_first"

    def int_int(x: int, y: int) -> str:
        return "int_int"

    _quiet_register(f, exact_first, int_int)
    with pytest.raises(AmbiguousMethodError):
        f(3, 3)


def test_literal_object_vs_int_int_is_ambiguous() -> None:
    """The `Literal[1]`-in-place-of-`Exact[int]` shape is ambiguous too."""
    f = Function("f")

    def lit_first(x: typing.Literal[1], y: object) -> str:
        return "lit_first"

    def int_int(x: int, y: int) -> str:
        return "int_int"

    _quiet_register(f, lit_first, int_int)
    with pytest.raises(AmbiguousMethodError):
        f(1, 1)


def test_single_dispatch_diamond_still_refines() -> None:
    """The plain single-argument diamond still resolves by MRO (B2 regression).

    `D(B, C)` with methods on `B` and `C`: `B` comes first in `D`'s MRO, so it
    wins -- the refinement the B2 fix must preserve.
    """

    class B:
        pass

    class C:
        pass

    class D(B, C):
        pass

    f = Function("f")

    def for_b(x: B) -> str:
        return "B"

    def for_c(x: C) -> str:
        return "C"

    _quiet_register(f, for_b, for_c)
    assert f(D()) == "B"


# --- N1/N2: cache token re-read, bounded cache, clear_cache ------------


def test_clear_cache_keeps_methods_and_forces_recompute() -> None:
    """`clear_cache` drops cached plans but leaves the methods untouched."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    f.register(m)
    assert f(1) == 1
    assert f._cache.shape_plans  # a plan was cached
    f.clear_cache()
    assert not f._cache.shape_plans  # dropped
    assert list(f.methods)  # methods survive
    assert f(2) == 2  # still dispatches


def test_call_cache_is_bounded() -> None:
    """The per-shape call cache never grows past its cap."""
    from bagof.dispatchers import _function as engine

    f = Function("f")

    def handle(x: object) -> int:
        return 1

    f.register(handle)
    cap = engine._CALL_CACHE_CAP
    # Each distinct argument *type* is a distinct call key under one shape.
    for index in range(cap + 50):
        made = type(f"T{index}", (), {})
        f(made())
    assert len(f._cache.call_cache) <= cap  # bounded


def test_ensure_rereads_token_under_lock() -> None:
    """`_ensure` stamps the cache with the token read under the lock (N1)."""
    import abc as _abc

    f = Function("f")

    def m(x: int) -> int:
        return x

    f.register(m)
    with f._lock:
        cache = f._ensure()
    assert cache.token == _abc.get_cache_token()


# --- N4: resolve() Exact convenience agrees with resolve_hint ----------


def test_resolve_exact_convenience_matches_plain_query() -> None:
    """`resolve(int)` reaches an `Exact[int]` method (RFC §4 convenience)."""
    f = Function("f")

    def exact(x: Exact[int]) -> str:
        return "exact"

    f.register(exact)
    # A plain-`int` query reaches the Exact[int] method, though `int` is not a
    # sub-hint of `Exact[int]` -- matching resolve_hint's key convenience.
    assert f.resolve(int).name == "exact"


def test_resolve_exact_convenience_agrees_with_resolve_hint() -> None:
    """`resolve` and `resolve_hint` answer the Exact convenience alike."""
    from bagof.dispatchers.core import resolve_hint

    f = Function("f")

    def exact(x: Exact[int]) -> str:
        return "exact"

    f.register(exact)
    via_function = f.resolve(int)
    registry = {Exact[int]: "exact"}
    via_registry = resolve_hint(int, registry)
    assert via_function.name == "exact"
    assert via_registry == "exact"


def test_resolve_exact_still_prefers_exact_over_plain() -> None:
    """With an `Exact[int]` and an `int` method, `resolve(int)` picks Exact."""
    f = Function("f")

    def exact(x: Exact[int]) -> str:
        return "exact"

    def plain(x: int) -> str:
        return "plain"

    _quiet_register(f, exact, plain)
    # Both applicable to an `int` query; Exact[int] is the leaf, so it wins.
    assert f.resolve(int).name == "exact"
