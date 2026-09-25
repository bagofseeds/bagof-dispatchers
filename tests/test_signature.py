"""Tests for the name-aware signature layer (`_signature.py`)."""

# stdlib
import inspect
import typing

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._signature import Binding, Parameter, Signature

# --- Parameter ---------------------------------------------------------


def test_parameter_required_and_default() -> None:
    """A parameter with no default is required; one with a default is not."""
    a = Parameter("x", int, Parameter.POSITIONAL_OR_KEYWORD)
    b = Parameter("y", int, Parameter.POSITIONAL_OR_KEYWORD, default=1)
    assert a.required
    assert not b.required
    assert b.default == 1


def test_parameter_is_immutable() -> None:
    """A parameter cannot be mutated after construction."""
    p = Parameter("x", int, Parameter.POSITIONAL_OR_KEYWORD)
    with pytest.raises(AttributeError):
        p.hint = str  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del p.hint  # type: ignore[misc]


def test_parameter_equality_is_semantic() -> None:
    """Equal parameters agree on name, kind, requiredness and hint."""
    a = Parameter("x", typing.List, Parameter.POSITIONAL_OR_KEYWORD)
    b = Parameter("x", list, Parameter.POSITIONAL_OR_KEYWORD)
    assert a == b  # `List` and `list` are equivalent hints
    assert hash(a) == hash(b)
    other = Parameter("x", int, Parameter.POSITIONAL_OR_KEYWORD)
    assert a != other


# --- from_callable shapes ---------------------------------------------


def test_from_callable_positional_or_keyword() -> None:
    """A plain function's parameters are positional-or-keyword."""

    def f(x: int, y: str) -> None: ...

    sig = Signature.from_callable(f)
    assert list(sig.parameters) == ["x", "y"]
    assert sig.parameters["x"].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["x"].hint is int
    assert sig.varargs is None
    assert sig.varkw is None
    assert sig.dispatched_names == ("x", "y")


def test_from_callable_defaults() -> None:
    """A default makes a parameter optional."""

    def f(x: int, y: str = "a") -> None: ...

    sig = Signature.from_callable(f)
    assert sig.parameters["x"].required
    assert not sig.parameters["y"].required
    assert sig.parameters["y"].default == "a"


def test_from_callable_varargs() -> None:
    """`*args: H` becomes the signature's varargs hint."""

    def f(x: int, *rest: str) -> None: ...

    sig = Signature.from_callable(f)
    assert list(sig.parameters) == ["x"]
    assert sig.varargs is str
    assert sig.varkw is None


def test_from_callable_varkw() -> None:
    """`**kwargs: H` becomes the signature's varkw hint."""

    def f(x: int, **rest: str) -> None: ...

    sig = Signature.from_callable(f)
    assert list(sig.parameters) == ["x"]
    assert sig.varkw is str
    assert sig.varargs is None


def test_from_callable_keyword_only() -> None:
    """A keyword-only parameter is dispatched, by name."""

    def f(x: int, *, y: str) -> None: ...

    sig = Signature.from_callable(f)
    assert sig.parameters["y"].kind is Parameter.KEYWORD_ONLY
    assert sig.dispatched_names == ("x", "y")


def test_from_callable_positional_only() -> None:
    """A positional-only parameter is not dispatched by name."""

    def f(x: int, /, y: str) -> None: ...

    sig = Signature.from_callable(f)
    assert sig.parameters["x"].kind is Parameter.POSITIONAL_ONLY
    assert sig.dispatched_names == ("y",)


def test_from_callable_unannotated_is_any() -> None:
    """An unannotated parameter dispatches on `Any`."""

    def f(x, y: int) -> None:  # noqa: ANN001 -- unannotated on purpose
        ...

    sig = Signature.from_callable(f)
    assert sig.parameters["x"].hint is tx.Any


def test_from_callable_unannotated_varargs_is_any() -> None:
    """An unannotated `*args` reads as `Any`, and is still present."""

    def f(*args) -> None: ...

    sig = Signature.from_callable(f)
    assert sig.varargs is tx.Any


def test_from_callable_exact() -> None:
    """`Exact[...]` survives `get_type_hints`."""

    def f(x: Exact[int]) -> None: ...

    sig = Signature.from_callable(f)
    from bagof.dispatchers.core._exact import is_exact

    assert is_exact(sig.parameters["x"].hint)


def test_from_callable_newtype_resolved() -> None:
    """A `NewType` parameter resolves to its supertype."""
    UserId = tx.NewType("UserId", int)

    def f(x: UserId) -> None: ...

    sig = Signature.from_callable(f)
    assert sig.parameters["x"].hint is int


def test_from_callable_stringized_resolvable() -> None:
    """A string annotation that resolves now is not deferred."""

    def f(x: "int") -> None: ...

    sig = Signature.from_callable(f)
    assert sig.parameters["x"].hint is int


def test_return_annotation_ignored() -> None:
    """The return annotation is not a parameter."""

    def f(x: int) -> str:
        return ""

    sig = Signature.from_callable(f)
    assert list(sig.parameters) == ["x"]


# --- deferred / forward references -------------------------------------


def _make_deferred_function() -> tx.Tuple[tx.Any, dict]:
    """A function whose annotation names a class not yet in its globals."""
    namespace = {}  # type: dict
    exec("def f(x: 'Later'): pass", namespace)
    return namespace["f"], namespace


def test_forward_reference_deferred_then_resolved() -> None:
    """An unresolvable name defers, then resolves on first use."""
    fn, namespace = _make_deferred_function()
    sig = Signature.from_callable(fn)
    assert sig._deferred

    class Later:
        pass

    namespace["Later"] = Later
    # First real use settles the hints.
    assert sig.applies_to_values((Later(),), {})
    assert sig.parameters["x"].hint is Later
    assert not sig._deferred


def test_forward_reference_still_unresolved_raises() -> None:
    """A name that never becomes defined raises `NameError` on use."""
    fn, _namespace = _make_deferred_function()
    sig = Signature.from_callable(fn)
    with pytest.raises(NameError):
        sig.applies_to_values((object(),), {})


def test_deferred_binding_does_not_need_hints() -> None:
    """Binding a deferred signature works without resolving the hint."""
    fn, _namespace = _make_deferred_function()
    sig = Signature.from_callable(fn)
    assert sig.bind((object(),), {}) is not None
    assert sig.bind((), {}) is None  # missing the required argument


# --- from_hints --------------------------------------------------------


def test_from_hints_positional_and_named() -> None:
    """Positional hints are positional-only; named hints are by-name."""
    sig = Signature.from_hints(int, scale=float)
    kinds = [p.kind for p in sig.parameters.values()]
    assert Parameter.POSITIONAL_ONLY in kinds
    assert sig.dispatched_names == ("scale",)
    assert sig.applies_to_values((1,), {"scale": 1.0})
    assert sig.applies_to_values((1, 1.0), {})


# --- binder differential vs inspect.Signature.bind --------------------


def _difftest_functions() -> tx.List[tx.Callable[..., tx.Any]]:
    """A spread of functions covering every parameter kind and marker."""

    def a(x: int, y: int) -> None: ...
    def b(x: int, y: int = 5) -> None: ...
    def c(x: int, *args: int) -> None: ...
    def d(x: int, **kw: int) -> None: ...
    def e(x: int, *args: int, **kw: int) -> None: ...
    def g(x: int, *, y: int) -> None: ...
    def h(x: int, *, y: int = 5) -> None: ...
    def i(x: int, /, y: int) -> None: ...
    def j(x: int, /) -> None: ...
    def k(x: int, y: int, /, z: int, *, w: int) -> None: ...
    def m(
        x: int, y: int = 1, /, z: int = 2, *args: int, w: int, v: int = 3,
        **kw: int
    ) -> None: ...
    def n() -> None: ...
    def o(*args: int) -> None: ...
    def p(**kw: int) -> None: ...
    def q(x: int, y: int, z: int) -> None: ...

    return [a, b, c, d, e, g, h, i, j, k, m, n, o, p, q]


def _call_shapes() -> tx.List[tx.Tuple[tx.Tuple[int, ...], dict]]:
    """A spread of (args, kwargs) call shapes to bind."""
    arg_counts = range(0, 4)
    keyword_sets = [
        {},
        {"x": 1},
        {"y": 2},
        {"z": 3},
        {"w": 4},
        {"x": 1, "y": 2},
        {"y": 2, "z": 3},
        {"w": 4, "v": 5},
        {"unknown": 9},
        {"x": 1, "extra": 9},
    ]
    shapes = []  # type: tx.List[tx.Tuple[tx.Tuple[int, ...], dict]]
    for count in arg_counts:
        args = tuple(range(10, 10 + count))
        for kwargs in keyword_sets:
            shapes.append((args, dict(kwargs)))
    return shapes


def _inspect_bind(
    isig: inspect.Signature, args: tx.Tuple, kwargs: dict
) -> tx.Tuple[bool, tx.Optional[dict]]:
    """Bind with `inspect`, returning (ok, arguments-dict)."""
    try:
        bound = isig.bind(*args, **kwargs)
    except TypeError:
        return False, None
    return True, dict(bound.arguments)


def _my_arguments(
    isig: inspect.Signature, binding: Binding, args: tx.Tuple, kwargs: dict
) -> dict:
    """Reconstruct inspect-style `.arguments` from our `Binding`."""
    varargs_name = next(
        (
            name
            for name, p in isig.parameters.items()
            if p.kind is inspect.Parameter.VAR_POSITIONAL
        ),
        None,
    )
    varkw_name = next(
        (
            name
            for name, p in isig.parameters.items()
            if p.kind is inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )
    arguments = {}  # type: dict
    for key, landed in binding.slots.items():
        if landed is Parameter.VAR_POSITIONAL:
            continue
        if landed is Parameter.VAR_KEYWORD:
            continue
        value = args[key] if isinstance(key, int) else kwargs[key]
        arguments[landed] = value
    if varargs_name is not None and binding.extra_positional:
        arguments[varargs_name] = tuple(
            args[index] for index in binding.extra_positional
        )
    if varkw_name is not None and binding.extra_keywords:
        arguments[varkw_name] = dict(binding.extra_keywords)
    return arguments


@pytest.mark.parametrize("fn", _difftest_functions())
def test_binder_matches_inspect(fn: tx.Callable[..., tx.Any]) -> None:
    """Our binder agrees with `inspect.Signature.bind` exactly.

    For every call shape it must agree on success/failure and, when it
    succeeds, on which argument landed in which parameter (including
    `*args` / `**kwargs`).
    """
    isig = inspect.signature(fn)
    mysig = Signature.from_callable(fn)
    for args, kwargs in _call_shapes():
        inspect_ok, inspect_args = _inspect_bind(isig, args, kwargs)
        binding = mysig.bind(args, kwargs)
        my_ok = binding is not None
        assert my_ok == inspect_ok, (fn.__name__, args, kwargs)
        if inspect_ok:
            mine = _my_arguments(isig, binding, args, kwargs)
            assert mine == inspect_args, (fn.__name__, args, kwargs)


def test_binder_specific_cases() -> None:
    """Named failure cases the differential sweep also covers."""

    def f(x: int) -> None: ...

    sig = Signature.from_callable(f)
    assert sig.bind((1,), {"x": 1}) is None  # duplicate value
    assert sig.bind((1, 2), {}) is None  # too many positionals
    assert sig.bind((), {"z": 1}) is None  # unexpected keyword
    assert sig.bind((), {}) is None  # missing required

    def p(x: int, /, **kw: int) -> None: ...

    psig = Signature.from_callable(p)
    assert psig.bind((), {"x": 1}) is None  # positional-only as keyword
    binding = psig.bind((1,), {"x": 2})  # x=1 positional, x=2 -> **kw
    assert binding is not None
    assert binding.extra_keywords == {"x": 2}


def test_binding_defaulted() -> None:
    """A defaulted parameter is recorded but is not an argument."""

    def f(x: int, y: int = 5) -> None: ...

    sig = Signature.from_callable(f)
    binding = sig.bind((1,), {})
    assert binding is not None
    assert binding.defaulted == frozenset({"y"})
    assert "y" not in binding.slots


# --- applies_to_values / applies_to_hints -----------------------------


def test_applies_to_values_positional_and_keyword() -> None:
    """Applicability holds across positional, keyword and mixed spellings."""

    def f(x: int, y: str) -> None: ...

    sig = Signature.from_callable(f)
    assert sig.applies_to_values((1, "a"), {})
    assert sig.applies_to_values((1,), {"y": "a"})
    assert sig.applies_to_values((), {"x": 1, "y": "a"})
    assert not sig.applies_to_values((1, 2), {})  # y is not a str


def test_applies_defaults_excluded() -> None:
    """A default-filled parameter's hint is not checked."""

    def f(x: int, y: int = 5) -> None: ...

    sig = Signature.from_callable(f)
    # y is never passed, so it does not matter that it is not checked.
    assert sig.applies_to_values((1,), {})


def test_applies_varargs_and_varkw() -> None:
    """Extra positionals/keywords are checked against the catch-all hint."""

    def f(*args: int) -> None: ...

    sig = Signature.from_callable(f)
    assert sig.applies_to_values((1, 2, 3), {})
    assert not sig.applies_to_values((1, "x"), {})

    def g(**kw: int) -> None: ...

    gsig = Signature.from_callable(g)
    assert gsig.applies_to_values((), {"a": 1, "b": 2})
    assert not gsig.applies_to_values((), {"a": "x"})


def test_applies_repeated_typevar() -> None:
    """A repeated `TypeVar` must be consistent across the arguments."""
    T = tx.TypeVar("T")

    def same(x: T, y: T) -> None: ...

    sig = Signature.from_callable(same)
    assert sig.applies_to_values((1, 2), {})
    assert sig.applies_to_values((1, True), {})  # greatest element is int
    assert not sig.applies_to_values((1, "x"), {})
    # Consistency holds regardless of positional/keyword spelling.
    assert sig.applies_to_values((1,), {"y": 2})
    assert not sig.applies_to_values((1,), {"y": "x"})


def test_applies_to_hints() -> None:
    """Hint-level applicability uses `issubhint`."""

    def f(x: int, y: object) -> None: ...

    sig = Signature.from_callable(f)
    assert sig.applies_to_hints((int, str), {})
    assert sig.applies_to_hints((bool,), {"y": int})
    assert not sig.applies_to_hints((str,), {"y": int})  # str is not an int


# --- specificity (le) --------------------------------------------------


def test_le_same_names_reduces_to_positional() -> None:
    """Same names/order compares like a positional tuple."""

    def f(x: int, y: int) -> None: ...

    def g(x: int, y: object) -> None: ...

    a = Signature.from_callable(f)
    b = Signature.from_callable(g)
    shape = Signature.shape((1, 2), {})
    assert a.le(b, shape)
    assert not b.le(a, shape)


def test_le_different_names_compete_per_position() -> None:
    """Different names still compete per position for a positional call."""

    def f(a: int, b: int) -> None: ...

    def g(x: int, y: int) -> None: ...

    p = Signature.from_callable(f)
    q = Signature.from_callable(g)
    shape = Signature.shape((1, 2), {})
    assert p.le(q, shape)
    assert q.le(p, shape)


def test_le_any_is_widest() -> None:
    """`Any` is the widest, so an `int` method is strictly more specific."""

    def f(x: int) -> None: ...

    def g(x) -> None:  # noqa: ANN001 -- unannotated -> Any on purpose
        ...

    a = Signature.from_callable(f)
    b = Signature.from_callable(g)
    shape = Signature.shape((1,), {})
    assert a.le(b, shape)
    assert not b.le(a, shape)
    assert a < b


def test_le_exact_more_specific() -> None:
    """`Exact[int]` is strictly more specific than `int`."""

    def f(x: Exact[int]) -> None: ...

    def g(x: int) -> None: ...

    a = Signature.from_callable(f)
    b = Signature.from_callable(g)
    shape = Signature.shape((1,), {})
    assert a.le(b, shape)
    assert not b.le(a, shape)


def test_le_unbindable_shape_is_false() -> None:
    """A signature that cannot bind the shape is not comparable."""

    def f(x: int) -> None: ...

    def g(x: int, y: int) -> None: ...

    a = Signature.from_callable(f)
    b = Signature.from_callable(g)
    shape = Signature.shape((1, 2), {})
    assert not a.le(b, shape)  # a cannot take two positionals


# --- equality / hashing ------------------------------------------------


def test_signature_equivalence_equality() -> None:
    """Signatures are equal up to hint equivalence."""

    def f(x: typing.List) -> None: ...

    def g(x: list) -> None: ...

    a = Signature.from_callable(f)
    b = Signature.from_callable(g)
    assert a == b
    assert hash(a) == hash(b)

    def h(x: int) -> None: ...

    assert a != Signature.from_callable(h)


def test_shape_distinguishes_spellings() -> None:
    """`f(1, 2)` and `f(1, y=2)` are different shapes."""
    assert Signature.shape((1, 2), {}) != Signature.shape((1,), {"y": 2})


def test_binding_repr_and_equality() -> None:
    """A `Binding` reprs readably and compares structurally."""
    one = Binding({0: "x"}, (), {}, frozenset())
    two = Binding({0: "x"}, (), {}, frozenset())
    assert one == two
    assert "Binding(" in repr(one)


def test_parameters_mapping_is_readonly() -> None:
    """The parameters mapping cannot be mutated through the view."""

    def f(x: int) -> None: ...

    sig = Signature.from_callable(f)
    params = sig.parameters
    assert len(params) == 1
    assert "x" in params
    with pytest.raises(TypeError):
        params["y"] = None  # type: ignore[index]
