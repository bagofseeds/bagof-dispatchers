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


def test_typeddict_method_chosen_by_shape_else_no_method() -> None:
    """A TypedDict method fires for a matching dict; a non-matching one does
    not (Phase 8 value-level shape check)."""
    import typing_extensions as tx

    class Point(tx.TypedDict):
        x: int
        y: int

    f = Function("f")

    def on_point(p: Point) -> str:
        return "point"

    f.register(on_point)
    assert f({"x": 1, "y": 2}) == "point"
    # A dict missing a required key matches no method's shape.
    with pytest.raises(NoMethodError):
        f({"x": 1})
    # A wrongly-typed value likewise.
    with pytest.raises(NoMethodError):
        f({"x": 1, "y": "two"})


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


def test_resolve_applies_repeated_typevar_tiebreak() -> None:
    """The repeated-`TypeVar` tie-break (§3) applies to `resolve` too.

    It is a selection step, so it settles a hint-level `resolve` the same way
    it settles a value call -- not only value dispatch. With the independent
    `(T, U)` registered *before* the repeated `(T, T)`, a query of two equal
    hints resolves to the strictly more specific `(T, T)`; a query of two
    different hints, which `(T, T)` cannot solve, falls to `(T, U)`.
    """
    T, U = _typevar_pair()
    f = Function("f")

    def free(x: T, y: U) -> str:
        return "free"

    def same(x: T, y: T) -> str:
        return "same"

    f.register(free)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the tie-break -> no ambiguity warning
        f.register(same)
    assert f.resolve(int, int).name == "same"
    assert f.resolve(int, str).name == "free"


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
    replacing the other. They are *not* ambiguous: the repeated-`TypeVar`
    tie-break (RFC 0001 §3) makes `(T, T)` -- which ties both arguments to one
    type -- strictly more specific than the independent `(T, U)`, so
    registering the second warns about nothing.
    """
    T, U = _typevar_pair()
    f = Function("f")

    def same(x: T, y: T) -> str:
        return "same"

    def free(x: T, y: U) -> str:
        return "free"

    f.register(same)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # neither ambiguity nor replacement
        f.register(free)
    assert len(f.methods) == 2  # both kept -- neither replaced the other
    assert f.ambiguities() == []
    # The tie-break resolves the equal-argument call to the more specific `(T,
    # T)`, and leaves the mixed-type call to the only method that applies.
    assert f(1, 2) == "same"
    assert f(1, "a") == "free"


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
    differently: both kept, neither replacing the other. They are not
    ambiguous either -- the repeated-`TypeVar` tie-break (RFC 0001 §3) makes
    `(T, T)`, which ties both arguments to one type, strictly more specific
    than the unconstrained pair, so registering the second warns about
    nothing.
    """
    T, _ = _typevar_pair()
    f = Function("f")

    def repeated(x: T, y: T) -> str:
        return "repeated"

    def bare(x, y) -> str:  # noqa: ANN001 -- unannotated, so (Any, Any)
        return "bare"

    f.register(repeated)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # neither ambiguity nor replacement
        f.register(bare)
    assert len(f.methods) == 2
    assert f.ambiguities() == []
    # The equal-argument call goes to the more specific repeated `TypeVar`;
    # a mixed-type call, which `(T, T)` rejects, falls to the bare method.
    assert f(1, 2) == "repeated"
    assert f(1, "a") == "bare"


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
        # The point here is only that both are kept regardless of order; any
        # warning (there is none now the tie-break separates them) is not what
        # is under test.
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


# --- Phase 7: repeated-TypeVar specificity tie-break (RFC 0001 §3) ------


def test_repeated_typevar_beats_independent() -> None:
    """`(T, T)` is more specific than `(T, U)`: a same-type call picks it.

    The canonical Phase 7 case. `(T, T)` constrains both arguments to one
    type; `(T, U)` leaves them independent. For a call whose arguments share a
    type both apply, and the tie-break (RFC 0001 §3) resolves the otherwise
    ambiguous pair to the more constrained `(T, T)`.
    """
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    f = Function("f")

    @f.register((T, T))
    def same(x, y):  # noqa: ANN001, ANN202
        return "same"

    @f.register((T, U))
    def indep(x, y):  # noqa: ANN001, ANN202
        return "indep"

    assert f(1, 2) == "same"  # same type -> the more specific (T, T)
    assert f(1, "a") == "indep"  # mixed -> only (T, U) applies at all


def test_repeated_typevar_registration_is_silent() -> None:
    """Registering `(T, T)` then `(T, U)` warns about nothing.

    The pair is separated by the tie-break, so it is neither guaranteed
    ambiguous (no `RuntimeWarning`) nor listed by `ambiguities()`.
    """
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    f = Function("f")

    @f.register((T, T))
    def same(x, y):  # noqa: ANN001, ANN202
        return "same"

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        @f.register((T, U))
        def indep(x, y):  # noqa: ANN001, ANN202
            return "indep"

    assert f.ambiguities() == []


def test_mixed_type_call_excludes_repeated_typevar() -> None:
    """`(T, T)` is inapplicable to a mixed call, so it never wins there.

    The tie-break only ever chooses between *applicable* methods; consistency
    removes `(T, T)` before selection when the arguments disagree.
    """
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    f = Function("f")

    @f.register((T, T))
    def same(x, y):  # noqa: ANN001, ANN202
        return "same"

    @f.register((T, U))
    def indep(x, y):  # noqa: ANN001, ANN202
        return "indep"

    assert f.dispatch(1, "a").name == "indep"


def test_bound_repeated_typevar_beats_independent() -> None:
    """A repeated bound `TypeVar` beats an independent one, all else equal.

    Two `TypeVar(bound=int)` variables: `(T, T)` ties the arguments together,
    `(T, U)` does not. Both read as `int` position-wise, so nothing but the
    grouping separates them, and the tie-break picks the grouped one.
    """
    T = typing.TypeVar("T", bound=int)
    U = typing.TypeVar("U", bound=int)
    f = Function("f")

    @f.register((T, T))
    def same(x, y):  # noqa: ANN001, ANN202
        return "same"

    @f.register((T, U))
    def indep(x, y):  # noqa: ANN001, ANN202
        return "indep"

    assert f(1, 2) == "same"
    assert f.ambiguities() == []


def test_repeated_typevar_beats_unannotated() -> None:
    """`(T, T)` is more specific than a fully unannotated `(x, y)`.

    An unannotated pair is `(Any, Any)` -- equivalent to `(T, U)` with two
    distinct variables -- so it carries no grouping and `(T, T)` wins the
    same-type call. (RFC 0001 §3 underspecifies this corner; the least
    surprising reading, taken here, is that any repeated `TypeVar` group beats
    a signature with none, since it constrains strictly more.)
    """
    T = typing.TypeVar("T")
    f = Function("f")

    @f.register((T, T))
    def same(x, y):  # noqa: ANN001, ANN202
        return "same"

    def bare(x, y):  # noqa: ANN001, ANN202 -- unannotated -> (Any, Any)
        return "bare"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f.register(bare)

    assert f(1, 2) == "same"
    assert f.ambiguities() == []


def test_two_groups_beat_three_independent() -> None:
    """`(T, T, U)` (two groups) beats `(T, U, V)` (three independent).

    Grouping two of the three arguments to one type is strictly more
    constraint than grouping none, so the two-group signature wins.
    """
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    V = typing.TypeVar("V")
    f = Function("f")

    @f.register((T, T, U))
    def two(x, y, z):  # noqa: ANN001, ANN202
        return "two"

    @f.register((T, U, V))
    def three(x, y, z):  # noqa: ANN001, ANN202
        return "three"

    assert f(1, 2, 3) == "two"
    assert f.ambiguities() == []


def test_swapped_independent_typevars_stay_ambiguous() -> None:
    """`(T, U)` vs `(U, T)` is a genuine tie the tie-break must not resolve.

    Both partitions are two singletons -- identical grouping -- so neither
    refines the other. The pair stays ambiguous, warns at registration, and is
    listed by `ambiguities()`.
    """
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    f = Function("f")

    @f.register((T, U))
    def tu(x, y):  # noqa: ANN001, ANN202
        return "tu"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register((U, T))
        def ut(x, y):  # noqa: ANN001, ANN202
            return "ut"

    assert len(f.ambiguities()) == 1
    with pytest.raises(AmbiguousMethodError):
        f(1, 2)


def test_partial_refinement_stays_ambiguous() -> None:
    """`(T, T, U)` vs `(T, U, U)` is incomparable: neither grouping refines.

    Each groups a pair the other leaves independent (`{0,1}` versus `{1,2}`),
    so neither is a strict refinement and the pair stays ambiguous.
    """
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    f = Function("f")

    @f.register((T, T, U))
    def left(x, y, z):  # noqa: ANN001, ANN202
        return "left"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register((T, U, U))
        def right(x, y, z):  # noqa: ANN001, ANN202
            return "right"

    assert len(f.ambiguities()) == 1
    with pytest.raises(AmbiguousMethodError):
        f(1, 1, 1)


def test_repeated_typevar_does_not_override_strict_specificity() -> None:
    """A repeated `TypeVar` never beats a strictly more specific method.

    `(int, int)` is strictly more specific than `(T, T)` at both arguments, so
    it wins outright -- the tie-break is reached only for an otherwise-tied
    pair and never overturns a real specificity win.
    """
    T = typing.TypeVar("T")
    f = Function("f")

    @f.register((T, T))
    def same(x, y):  # noqa: ANN001, ANN202
        return "same"

    @f.register((int, int))
    def concrete(x, y):  # noqa: ANN001, ANN202
        return "concrete"

    assert f(1, 2) == "concrete"


def test_repeated_typevar_tie_break_by_keyword() -> None:
    """The tie-break holds when the same arguments arrive by keyword.

    Grouping is over the arguments a repeated `TypeVar` lands, regardless of
    positional or keyword spelling, so `(T, T)` wins the keyword call too.
    """
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    f = Function("f")

    @f.register((), {"x": T, "y": T})
    def same(x, y):  # noqa: ANN001, ANN202
        return "same"

    @f.register((), {"x": T, "y": U})
    def indep(x, y):  # noqa: ANN001, ANN202
        return "indep"

    assert f(x=1, y=2) == "same"
    assert f(x=1, y="a") == "indep"


# --- Phase 8 (d): joint TypeVar solving through **kwargs: T -----------


def test_kwargs_typevar_groups_beat_untyped() -> None:
    """`**kwargs: T` beats an untyped `**kwargs` for a multi-keyword call.

    Every keyword a `**kwargs: T` captures lands the one variable, so the
    grouping tie-break (RFC 0001 §3) reads them as a single consistent-`T`
    block -- more constrained than the untyped `**kwargs`, whose captured
    keywords each stand alone. For a call with two surplus keywords the two
    are otherwise equally specific, so the tie-break picks the typed one.
    """
    T = typing.TypeVar("T")
    f = Function("f")

    def typed(**rest):  # noqa: ANN003, ANN202
        return "typed"

    def plain(**rest):  # noqa: ANN003, ANN202
        return "plain"

    with warnings.catch_warnings():
        # The two are genuinely ambiguous for a no-keyword call (nothing is
        # captured to group), so registration warns; that is not what is under
        # test here.
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"rest": T})(typed)
        f.register(plain)

    assert f(a=1, b=2) == "typed"


def test_kwargs_typevar_single_keyword_stays_ambiguous() -> None:
    """One captured keyword makes a group of one, which refines nothing.

    A single keyword lands a block of one, so `**kwargs: T` groups no more
    than an untyped `**kwargs` and the pair stays ambiguous -- the same rule
    that leaves `*args: T` ambiguous for a one-argument call (RFC 0001 §3).
    """
    T = typing.TypeVar("T")
    f = Function("f")

    def typed(**rest):  # noqa: ANN003, ANN202
        return "typed"

    def plain(**rest):  # noqa: ANN003, ANN202
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"rest": T})(typed)
        f.register(plain)

    with pytest.raises(AmbiguousMethodError):
        f(a=1)


def test_kwargs_unbound_typevar_rejects_mixed_values() -> None:
    """An unbound `**kwargs: T` does not bind keywords of disagreeing types.

    `**kwargs: T` solves `T` jointly across every captured keyword, exactly as
    `*args: T` does across the positionals it absorbs. Two keyword values with
    no consistent `T` leave the method inapplicable: with an untyped `**kwargs`
    sibling the call falls to it, and with only `**kwargs: T` registered there
    is nothing left to run.
    """
    T = typing.TypeVar("T")
    f = Function("f")

    def typed(**rest):  # noqa: ANN003, ANN202
        return "typed"

    def plain(**rest):  # noqa: ANN003, ANN202
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"rest": T})(typed)
        f.register(plain)

    assert f(a=1, b="x") == "plain"

    g = Function("g")

    def only(**rest):  # noqa: ANN003, ANN202
        return "only"

    g.register({"rest": T})(only)

    with pytest.raises(NoMethodError):
        g(a=1, b="x")


def test_kwargs_bound_typevar_checks_each_value() -> None:
    """A bounded `**kwargs: T` requires every captured keyword to fit.

    Each keyword must lie under the bound, so a value outside it makes the
    method inapplicable -- the same requirement the joint solve keeps for the
    keywords it groups.
    """
    T = typing.TypeVar("T", bound=int)
    f = Function("f")

    def only(**rest):  # noqa: ANN003, ANN202
        return "only"

    f.register({"rest": T})(only)

    assert f(a=1, b=2) == "only"
    with pytest.raises(NoMethodError):
        f(a=1, b="x")


def test_kwargs_typevar_grouping_not_flagged_ambiguous() -> None:
    """A multi-keyword call resolves, so dispatch picks the typed method.

    The tie-break separates `**kwargs: T` from an untyped `**kwargs` wherever
    two or more keywords are captured, matching the positional repeated-
    `TypeVar` behaviour.
    """
    T = typing.TypeVar("T")
    f = Function("f")

    def typed(x, **rest):  # noqa: ANN001, ANN003, ANN202
        return "typed"

    def plain(x, **rest):  # noqa: ANN001, ANN003, ANN202
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"rest": T})(typed)
        f.register(plain)

    assert f.dispatch(1, a=2, b=3).name == "typed"


def test_kwargs_typevar_groups_in_resolve() -> None:
    """The hint-level `resolve` solves `**kwargs: T` jointly the same way.

    Selection by hint uses the identical joint solve. Two agreeing query hints
    resolve to the `**kwargs: T` method; two disagreeing hints have no
    consistent `T`, so it is inapplicable and the untyped `**kwargs` answers.
    """
    T = typing.TypeVar("T")
    f = Function("f")

    def typed(**rest):  # noqa: ANN003, ANN202
        return "typed"

    def plain(**rest):  # noqa: ANN003, ANN202
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"rest": T})(typed)
        f.register(plain)

    assert f.resolve(a=int, b=int).name == "typed"
    assert f.resolve(a=int, b=str).name == "plain"


def test_kwargs_typevar_does_not_override_strict_specificity() -> None:
    """A concrete `**kwargs: int` beats a grouped `**kwargs: T`.

    The grouping tie-break is reached only when the landed hints are
    equivalent. `**kwargs: int` is strictly more specific than an unbound
    `**kwargs: T`, so it wins outright and the grouping never overturns it --
    the `**kwargs` twin of the positional `(int, int)` over `(T, T)`.
    """
    T = typing.TypeVar("T")
    f = Function("f")

    def grouped(**rest):  # noqa: ANN003, ANN202
        return "grouped"

    def concrete(**rest):  # noqa: ANN003, ANN202
        return "concrete"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"rest": T})(grouped)
        f.register({"rest": int})(concrete)

    assert f(a=1, b=2) == "concrete"


def test_kwargs_typevar_greatest_element_matches() -> None:
    """`**kwargs: T` matches by the greatest-element rule, not a strict join.

    The joint solve accepts a set of values whenever one type is a supertype
    of them all: `#!python {int, bool}` settles on `int` (`bool` is a subtype),
    so `f(a=1, b=True)` runs the typed method. Two values with no common
    supertype under the variable have no consistent `T`, so `f(a=1, b="x")`
    falls to the untyped `**kwargs`. This mirrors `*args: T` exactly.
    """
    T = typing.TypeVar("T")
    f = Function("f")

    def typed(**rest):  # noqa: ANN003, ANN202
        return "typed"

    def plain(**rest):  # noqa: ANN003, ANN202
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"rest": T})(typed)
        f.register(plain)

    assert f(a=1, b=True) == "typed"
    assert f(a=1, b="x") == "plain"


def test_typevar_solved_jointly_across_positional_and_kwargs() -> None:
    """One `T` spanning a positional and `**kwargs` is solved as one block.

    `(x: T, **kw: T)` ties the positional argument and every captured keyword
    to a single variable. `h(1, a="x")` mixes an `int` and a `str` under that
    one `T`, which has no consistent solution, so the typed method is
    inapplicable and the untyped-`**kw` sibling answers instead.
    """
    T = typing.TypeVar("T")
    f = Function("f")

    def typed(x, **kw):  # noqa: ANN001, ANN003, ANN202
        return "typed"

    def plain(x, **kw):  # noqa: ANN001, ANN003, ANN202
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"x": T, "kw": T})(typed)
        f.register({"x": T})(plain)

    assert f(1, a="x") == "plain"
    # Agreeing values keep the joint solve satisfied, so the typed one wins.
    assert f(1, a=2) == "typed"


def test_kwargs_distinct_typevars_stay_ambiguous() -> None:
    """`**kwargs: T` versus `**kwargs: U` is a genuine tie.

    Two single-variable catch-alls group their captured keywords the same way
    -- one block each -- so neither refines the other. A multi-keyword call is
    applicable to both and no measure separates them, so it is ambiguous.
    """
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    f = Function("f")

    def with_t(**rest):  # noqa: ANN003, ANN202
        return "t"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"rest": T})(with_t)

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register({"rest": U})
        def with_u(**rest):  # noqa: ANN003, ANN202
            return "u"

    with pytest.raises(AmbiguousMethodError):
        f(a=1, b=2)


def test_typevar_mixed_positional_kwargs_grouping_stays_ambiguous() -> None:
    """`(T, T, **kw)` versus `(T, U, **kw: T)` neither refines the other.

    The first groups the two positionals; the second groups the first
    positional with the captured keywords. Each ties a set the other leaves
    apart, so the groupings are incomparable and a call reaching both slots is
    ambiguous.
    """
    T = typing.TypeVar("T")
    U = typing.TypeVar("U")
    f = Function("f")

    def a(x, y, **kw):  # noqa: ANN001, ANN003, ANN202
        return "a"

    def b(x, y, **kw):  # noqa: ANN001, ANN003, ANN202
        return "b"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register({"x": T, "y": T})(a)
        f.register({"x": T, "y": U, "kw": T})(b)

    # A call that reaches the captured keywords witnesses the incomparable
    # groupings, so it is ambiguous even though registration cannot see it.
    with pytest.raises(AmbiguousMethodError):
        f(1, 2, p=3, q=4)
