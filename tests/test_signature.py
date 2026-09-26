"""Tests for the name-aware signature layer (`_signature.py`)."""

# stdlib
import functools
import inspect
import sys
import typing
import warnings

# dependencies
import pytest
import typing_extensions as tx

# locals
import bagof.dispatchers._signature as sigmod
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


# --- deferred / forward-reference equality (defect 1) ------------------


def test_parameter_deferred_equality_does_not_raise() -> None:
    """A still-deferred hint compares by name, never through the relation."""
    kind = Parameter.POSITIONAL_OR_KEYWORD
    same_a = Parameter("x", "Later", kind)
    same_b = Parameter("x", "Later", kind)
    other = Parameter("x", "Other", kind)
    resolved = Parameter("x", int, kind)
    # None of these raise a `TypeError` (the old bug), and each gives a bool.
    assert same_a == same_b  # same forward name
    assert same_a != other  # different forward names
    assert same_a != resolved  # deferred vs resolved
    assert resolved != same_a  # and the other way round


def test_parameter_forward_ref_equality() -> None:
    """A `ForwardRef` compares by its forward name, like a raw string."""
    ref = tx.ForwardRef("Later")
    kind = Parameter.POSITIONAL_OR_KEYWORD
    assert Parameter("x", ref, kind) == Parameter("x", "Later", kind)
    assert Parameter("x", ref, kind) != Parameter("x", "Other", kind)


def test_signature_deferred_equality_no_typeerror() -> None:
    """Comparing a deferred signature never raises, and settles resolvables."""
    fn, namespace = _make_deferred_function()
    deferred = Signature.from_callable(fn)
    # A deferred, unresolvable signature compares cleanly (no TypeError,
    # no NameError) against a resolved one, and gives a clean `False`.
    assert deferred != Signature.from_callable(_difftest_int())
    assert deferred._deferred  # left deferred, not forced to resolve

    # Once the name is defined, the same function's deferred and resolved
    # signatures compare equal without raising.
    class Later:
        pass

    namespace["Later"] = Later
    resolved = Signature.from_callable(fn)
    resolved._settle()
    assert deferred == resolved
    assert not deferred._deferred  # equality settled it


def _difftest_int() -> tx.Callable[..., tx.Any]:
    def other(x: int) -> None: ...

    return other


def test_signature_deferred_same_forward_name_equal() -> None:
    """Two deferred signatures with the same unresolved name are equal."""
    a, _ = _make_deferred_function()
    b, _ = _make_deferred_function()
    assert Signature.from_callable(a) == Signature.from_callable(b)


def test_nested_forward_ref_signatures_differ_without_warning() -> None:
    """A forward reference nested in a generic is compared by name, quietly.

    `List["Later"]` and `List["Other"]` name different types, so the two
    signatures are unequal -- and the comparison must not route the unresolved
    names through the sub-hint relation, which would both warn and treat each
    name as `Any` (making the two wrongly equal).
    """

    def f(x: typing.List["Later"]) -> None: ...  # noqa: F821

    def g(x: typing.List["Other"]) -> None: ...  # noqa: F821

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Signature.from_callable(f) != Signature.from_callable(g)


def test_nested_forward_ref_signatures_same_name_equal() -> None:
    """Two signatures with the same nested forward reference are equal."""

    def f(x: typing.List["Later"]) -> None: ...  # noqa: F821

    def g(x: typing.List["Later"]) -> None: ...  # noqa: F821

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Signature.from_callable(f) == Signature.from_callable(g)


def test_signature_equality_ignores_literal_and_annotated_metadata() -> None:
    """String `Literal` members / `Annotated` metadata are not forward refs.

    They are values, so equality goes through the ordinary hint relation --
    matching the non-string-metadata cases (`Annotated[int, 1]`,
    `Literal[1, 2]`) -- rather than the structural forward-reference path,
    which would wrongly split hints that differ only in a doc string.
    """
    kind = Parameter.POSITIONAL_OR_KEYWORD

    def one(hint: tx.Any) -> Signature:
        return Signature({"x": Parameter("x", hint, kind)})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # `Annotated[int, "doc"]` is just `int` for dispatch.
        assert one(tx.Annotated[int, "doc"]) == one(int)
        # Two `Annotated` hints differing only in metadata are equal.
        assert one(tx.Annotated[int, "doc"]) == one(tx.Annotated[int, "other"])
        # A union of string `Literal`s equals the flattened `Literal`.
        assert one(tx.Union[tx.Literal["a"], tx.Literal["b"]]) == one(
            tx.Literal["a", "b"]
        )


def test_has_forward_ref_recurses_and_terminates() -> None:
    """`_has_forward_ref` finds a nested name, bottoms out on plain types."""
    assert sigmod._has_forward_ref("Later")
    assert sigmod._has_forward_ref(tx.ForwardRef("Later"))
    assert sigmod._has_forward_ref(typing.List["Later"])  # noqa: F821
    nested = typing.Optional[typing.List["Later"]]  # noqa: F821
    assert sigmod._has_forward_ref(nested)
    assert not sigmod._has_forward_ref(int)
    assert not sigmod._has_forward_ref(typing.List[int])


def test_has_forward_ref_ignores_literal_members_and_metadata() -> None:
    """A `Literal` member and `Annotated` metadata are values, not refs.

    `typing` keeps them as bare strings inside a hint, but wraps a genuine
    nested forward reference in a `ForwardRef` -- so a nested bare string is
    never a reference, and only the wrapped type of an `Annotated` is
    descended into.
    """
    # A `Literal`'s string members are values, not forward references.
    assert not sigmod._has_forward_ref(tx.Literal["a"])
    assert not sigmod._has_forward_ref(tx.Literal["a", "b"])
    assert not sigmod._has_forward_ref(typing.List[tx.Literal["a"]])
    # `Annotated` metadata is arbitrary values, so a bare string there is not
    # a reference; only the wrapped type carries one.
    assert not sigmod._has_forward_ref(tx.Annotated[int, "doc"])
    assert sigmod._has_forward_ref(
        tx.Annotated[typing.List["X"], "doc"]  # noqa: F821
    )


def test_hint_eq_resolved_hints_use_equivalence() -> None:
    """No forward reference on either side: `_hint_eq` uses equivalence."""
    assert sigmod._hint_eq(int, int)
    assert not sigmod._hint_eq(int, str)


def test_hint_eq_top_level_string_matches_forward_ref() -> None:
    """A top-level raw string and a `ForwardRef` naming it compare equal."""
    assert sigmod._hint_eq("Later", tx.ForwardRef("Later"))
    assert not sigmod._hint_eq("Later", tx.ForwardRef("Other"))


# --- PEP 585 builtin generics keep bare-string forward refs ------------
#
# A `types.GenericAlias` (`list["X"]`, `dict[str, "X"]`) keeps its forward
# reference as a *bare string* -- unlike `typing.List["X"]`, which wraps it in
# a `ForwardRef`. So a nested bare string reached during recursion is a genuine
# reference and must be treated as one, or comparing two such signatures feeds
# the unresolved name to the sub-hint relation and crashes.


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="PEP 585 builtin generics (list[...]) need Python 3.9+",
)
def test_has_forward_ref_pep585_bare_string_arg() -> None:
    """A bare string in a PEP 585 builtin generic is a forward reference."""
    assert sigmod._has_forward_ref(list["X"])  # noqa: F821
    assert sigmod._has_forward_ref(dict[str, "X"])  # noqa: F821
    assert sigmod._has_forward_ref(
        typing.Dict[str, list["X"]]  # noqa: F821
    )
    # A fully resolved builtin generic still holds no reference.
    assert not sigmod._has_forward_ref(list[int])
    assert not sigmod._has_forward_ref(dict[str, int])


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="PEP 585 builtin generics (list[...]) need Python 3.9+",
)
def test_pep585_nested_forward_ref_signatures_differ_without_warning() -> None:
    """`list["Zed"]` vs `list["Yed"]` compare unequal, no raise, no warning.

    The regression: the bare string a `types.GenericAlias` keeps was ignored
    during recursion, so the two hints fell through to the sub-hint relation,
    which raised `TypeError` trying to use the undefined name as a type.
    """

    def p(x: list["Zed"]) -> None: ...  # noqa: F821

    def q(x: list["Yed"]) -> None: ...  # noqa: F821

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Signature.from_callable(p) != Signature.from_callable(q)


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="PEP 585 builtin generics (list[...]) need Python 3.9+",
)
def test_pep585_nested_forward_ref_signatures_same_name_equal() -> None:
    """Two signatures with the same PEP 585 nested forward ref are equal."""

    def p(x: list["Zed"]) -> None: ...  # noqa: F821

    def q(x: list["Zed"]) -> None: ...  # noqa: F821

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Signature.from_callable(p) == Signature.from_callable(q)


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="PEP 585 builtin generics (list[...]) need Python 3.9+",
)
def test_pep585_dict_and_nested_builtin_generic_signatures_differ() -> None:
    """`dict[str, "X"]` / `Dict[str, list["X"]]` defer and compare by name."""

    def p(x: dict[str, "X"]) -> None: ...  # noqa: F821

    def q(x: dict[str, "Y"]) -> None: ...  # noqa: F821

    def r(x: typing.Dict[str, list["X"]]) -> None: ...  # noqa: F821

    def s(x: typing.Dict[str, list["Y"]]) -> None: ...  # noqa: F821

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Signature.from_callable(p) != Signature.from_callable(q)
        assert Signature.from_callable(r) != Signature.from_callable(s)


# --- partial / callable instances (defect 2) ---------------------------


def test_from_callable_partial_resolves_hints() -> None:
    """A `functools.partial` resolves its remaining parameters' hints."""

    def base(a: int, b: str) -> None: ...

    sig = Signature.from_callable(functools.partial(base, 1))
    assert list(sig.parameters) == ["b"]
    assert sig.parameters["b"].hint is str
    assert not sig._deferred
    assert sig.applies_to_values(("hi",), {})
    assert not sig.applies_to_values((1,), {})


def test_from_callable_callable_instance_resolves_hints() -> None:
    """A callable instance reads hints off its `__call__`."""

    class Adder:
        def __call__(self, x: int) -> int:
            return x

    sig = Signature.from_callable(Adder())
    assert list(sig.parameters) == ["x"]
    assert sig.parameters["x"].hint is int
    assert not sig._deferred
    assert sig.applies_to_values((3,), {})
    assert not sig.applies_to_values(("no",), {})


def test_from_callable_typeerror_stringized_defers() -> None:
    """A stringised spelling the running Python cannot evaluate defers."""

    def f(x: int) -> None: ...

    # A string annotation whose evaluation raises `TypeError` (as
    # `"list[int]"` does on Python 3.8) stands in for the version-specific
    # case: it must defer, not resolve to `Any`.
    f.__annotations__ = {"x": "int[str]"}
    sig = Signature.from_callable(f)
    assert sig._deferred
    with pytest.raises(NameError):
        sig.applies_to_values((1,), {})


def test_from_callable_typeerror_non_forward_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `TypeError` with already-resolved annotations resolves, not defers."""

    def f(x: int) -> None: ...

    def boom(_fn: object) -> dict:
        raise TypeError("simulated get_type_hints failure")

    monkeypatch.setattr(sigmod, "_resolve_hints", boom)
    sig = Signature.from_callable(f)
    assert not sig._deferred
    assert sig.parameters["x"].hint is int


def test_settle_resolves_variadic_hints() -> None:
    """Settling a deferred signature resolves its `*args`/`**kwargs` hints."""
    namespace = {}  # type: dict
    exec(
        "def f(x: 'Later', *args: 'Later', **kw: 'Later'): pass",
        namespace,
    )
    sig = Signature.from_callable(namespace["f"])
    assert sig._deferred

    class Later:
        pass

    namespace["Later"] = Later
    assert sig.applies_to_values((Later(), Later()), {"z": Later()})
    assert sig.varargs is Later
    assert sig.varkw is Later


def test_bind_positional_only_defaulted_passed_as_keyword() -> None:
    """A defaulted positional-only parameter cannot be given by keyword."""

    def f(a: int = 1, b: int = 2, /) -> None: ...

    sig = Signature.from_callable(f)
    assert sig.bind((), {}) is not None  # both defaulted
    assert sig.bind((), {"b": 3}) is None  # b is positional-only


def test_signature_le_operator() -> None:
    """The `<=` operator compares two signatures at their full shape."""

    def f(x: int) -> None: ...

    def g(x: object) -> None: ...

    assert Signature.from_callable(f) <= Signature.from_callable(g)


def test_raw_annotations_unreadable_falls_back() -> None:
    """An object whose `__annotations__` cannot be read yields no annotations.

    This is the shape of Python 3.14's lazy annotations, where reading them
    the ordinary way may raise.
    """

    class Weird:
        @property
        def __annotations__(self) -> dict:
            raise RuntimeError("cannot read annotations")

    # On a Python without `annotationlib` this returns `{}`; the point is that
    # it never propagates the error.
    assert isinstance(sigmod._raw_annotations(Weird()), dict)


def test_render_forward_ref_param() -> None:
    """A `ForwardRef` parameter hint renders by its name."""
    kind = Parameter.POSITIONAL_OR_KEYWORD
    sig = Signature({"x": Parameter("x", tx.ForwardRef("Later"), kind)})
    assert repr(sig) == "Signature(x: Later)"


def test_from_callable_forward_ref_still_defers() -> None:
    """A genuine forward reference still defers and resolves on first use."""
    fn, namespace = _make_deferred_function()
    sig = Signature.from_callable(fn)
    assert sig._deferred

    class Later:
        pass

    namespace["Later"] = Later
    assert sig.applies_to_values((Later(),), {})
    assert sig.parameters["x"].hint is Later


# --- variadic tails degrade to Any (defect 3) --------------------------


def test_variadic_typevartuple_is_any() -> None:
    """`*args: *Ts` keeps the `Unpack[Ts]` alias, behaving like `Any`.

    The tail is no longer flattened to `Any`: the same `Ts` may appear at a
    `Tuple[..., *Ts]` slot and be solved jointly (RFC 0001 §3), so the alias
    must survive. On its own it still accepts any positionals, with no warning.
    """
    Ts = tx.TypeVarTuple("Ts")

    def f(*args: tx.Unpack[Ts]) -> None: ...

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        sig = Signature.from_callable(f)
        assert sig.varargs == tx.Unpack[Ts]
        assert sig.applies_to_values((1, "x", object()), {})
    assert repr(sig) == "Signature(*args: Unpack[Ts])"


def test_variadic_paramspec_is_any() -> None:
    """`*args: P.args` / `**kwargs: P.kwargs` read as catch-alls."""
    P = tx.ParamSpec("P")

    def f(*args: P.args, **kwargs: P.kwargs) -> None: ...

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sig = Signature.from_callable(f)
        assert sig.varargs is tx.Any
        assert sig.varkw is tx.Any
        assert sig.applies_to_values((1, 2), {"a": "x"})
    assert repr(sig) == "Signature(*args, **kwargs)"


def test_variadic_unpack_typeddict_is_any() -> None:
    """`**kwargs: Unpack[TypedDict]` reads as an unannotated catch-all."""

    class Opts(tx.TypedDict):
        a: int

    def f(**kwargs: tx.Unpack[Opts]) -> None: ...

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sig = Signature.from_callable(f)
        assert sig.varkw is tx.Any
        assert sig.applies_to_values((), {"a": 1, "b": "anything"})
    assert repr(sig) == "Signature(**kwargs)"


# --- coverage: reprs, NotImplemented, and le with catch-alls -----------


def test_parameter_repr_and_notimplemented() -> None:
    """A parameter reprs readably and is unequal to a non-parameter."""
    kind = Parameter.POSITIONAL_OR_KEYWORD
    required = Parameter("x", int, kind)
    optional = Parameter("y", str, kind, default="a")
    assert repr(required) == "Parameter(x: int)"
    assert repr(optional) == "Parameter(y: str = 'a')"
    assert required.__eq__(object()) is NotImplemented


def test_binding_notimplemented() -> None:
    """A binding is unequal to a non-binding."""
    assert Binding({}, (), {}, frozenset()).__eq__(object()) is NotImplemented


def test_signature_repr_and_notimplemented() -> None:
    """A signature reprs readably and is unequal to a non-signature."""

    def f(x: int, y: str = "a") -> None: ...

    sig = Signature.from_callable(f)
    assert repr(sig) == "Signature(x: int, y: str = 'a')"
    assert sig.__eq__(object()) is NotImplemented
    assert sig.__le__(object()) is NotImplemented
    assert sig.__lt__(object()) is NotImplemented
    assert "'x'" in repr(sig.parameters)  # the read-only view reprs as a dict


def test_signature_eq_different_length() -> None:
    """Signatures with different parameter names are unequal."""

    def f(x: int) -> None: ...

    def g(x: int, y: int) -> None: ...

    assert Signature.from_callable(f) != Signature.from_callable(g)


def test_signature_le_and_eq_with_catch_alls() -> None:
    """`le` and `==` read the `*args` / `**kwargs` hints."""

    def f(x: int, *args: int, **kw: int) -> None: ...

    def g(x: int, *args: object, **kw: object) -> None: ...

    a = Signature.from_callable(f)
    b = Signature.from_callable(g)
    shape = Signature.shape((1, 2, 3), {"z": 4})
    assert a.le(b, shape)
    assert not b.le(a, shape)
    # A varargs/varkw hint mismatch makes two signatures unequal.
    assert a != b
    assert a == Signature.from_callable(f)


def test_signature_eq_catch_all_presence_differs() -> None:
    """A signature with `*args` is unequal to one without it."""

    def f(x: int, *args: int) -> None: ...

    def g(x: int) -> None: ...

    assert Signature.from_callable(f) != Signature.from_callable(g)


def test_applies_to_values_unbindable_is_false() -> None:
    """A call that cannot bind is not applicable."""

    def f(x: int) -> None: ...

    sig = Signature.from_callable(f)
    assert not sig.applies_to_values((1, 2), {})


def test_applies_to_hints_unbindable_and_typevar() -> None:
    """Hint-level applicability: no bind is False; repeated `TypeVar` holds."""
    T = tx.TypeVar("T")

    def f(x: int) -> None: ...

    assert not Signature.from_callable(f).applies_to_hints((int, str), {})

    def same(x: T, y: T) -> None: ...

    sig = Signature.from_callable(same)
    assert sig.applies_to_hints((int, bool), {})
    assert not sig.applies_to_hints((int, str), {})


def test_parameters_mapping_is_readonly() -> None:
    """The parameters mapping cannot be mutated through the view."""

    def f(x: int) -> None: ...

    sig = Signature.from_callable(f)
    params = sig.parameters
    assert len(params) == 1
    assert "x" in params
    with pytest.raises(TypeError):
        params["y"] = None  # type: ignore[index]


# --- from_callable on a class reads its constructor (#14) ---------------


def test_from_callable_class_reads_init_hints() -> None:
    """A class's signature comes from its `__init__`, dropping `self`."""

    class Widget:
        def __init__(self, size: int, label: str = "w") -> None: ...

    sig = Signature.from_callable(Widget)
    assert list(sig.parameters) == ["size", "label"]
    assert sig.parameters["size"].hint is int
    assert sig.parameters["label"].hint is str
    assert sig.parameters["label"].default == "w"


def test_from_callable_dataclass_reads_generated_init() -> None:
    """A dataclass dispatches on its generated `__init__` fields."""
    import dataclasses

    @dataclasses.dataclass
    class Point:
        x: int
        y: str = "o"

    sig = Signature.from_callable(Point)
    assert list(sig.parameters) == ["x", "y"]
    assert sig.parameters["x"].hint is int
    assert sig.parameters["y"].hint is str


def test_from_callable_class_reads_new_when_no_init() -> None:
    """A class defining only `__new__` reads its constructor from `__new__`."""

    class Only:
        def __new__(cls, value: float):  # noqa: ANN204
            return super().__new__(cls)

    sig = Signature.from_callable(Only)
    assert list(sig.parameters) == ["value"]
    assert sig.parameters["value"].hint is float


def test_from_callable_builtin_without_signature_falls_back() -> None:
    """A builtin with no introspectable signature gets a catch-all."""
    sig = Signature.from_callable(int)
    assert list(sig.parameters) == []
    assert sig.varargs is tx.Any
    assert sig.varkw is tx.Any


# --- same_as: structural equality vs semantic equivalence --------------


def test_same_as_distinguishes_distinct_typevars() -> None:
    """`(T, T)` and `(T, U)` are equal (both `Any`) but not `same_as`."""
    T = tx.TypeVar("T")
    U = tx.TypeVar("U")

    def same(x: T, y: T) -> None: ...

    def free(x: T, y: U) -> None: ...

    a = Signature.from_callable(same)
    b = Signature.from_callable(free)
    assert a == b  # equivalent: both are (Any, Any)
    assert not a.same_as(b)  # but spelled differently
    assert a.same_as(Signature.from_callable(same))  # identical spelling


def test_same_as_distinguishes_bound_typevar_from_bound() -> None:
    """A `TypeVar(bound=int)` is equal to `int` but not the same spelling."""
    TB = tx.TypeVar("TB", bound=int)

    def tv(x: TB) -> None: ...

    def plain(x: int) -> None: ...

    a = Signature.from_callable(tv)
    b = Signature.from_callable(plain)
    assert a == b  # a bound TypeVar is equivalent to its bound
    assert not a.same_as(b)  # but a TypeVar is not the class it is bounded by


def test_same_as_distinguishes_exact_from_plain() -> None:
    """`Exact[int]` and `int` are different spellings (a leaf, not equal)."""

    def exact(x: Exact[int]) -> None: ...

    def plain(x: int) -> None: ...

    a = Signature.from_callable(exact)
    b = Signature.from_callable(plain)
    assert not a.same_as(b)


def test_same_as_matches_identical_generic_spellings() -> None:
    """Identically-spelled generics are `same_as`; differing ones are not."""

    def ints(x: tx.List[int]) -> None: ...

    def strs(x: tx.List[str]) -> None: ...

    assert Signature.from_callable(ints).same_as(Signature.from_callable(ints))
    assert not Signature.from_callable(ints).same_as(
        Signature.from_callable(strs)
    )


def test_same_as_distinguishes_literal_members() -> None:
    """`Literal[1]` and `Literal[True]` are distinct spellings (type-aware)."""

    def one(x: tx.Literal[1]) -> None: ...

    def yes(x: tx.Literal[True]) -> None: ...

    assert not Signature.from_callable(one).same_as(
        Signature.from_callable(yes)
    )


def test_same_as_distinguishes_generic_arity() -> None:
    """Generics with the same origin but different arity are not `same_as`."""

    def one(x: tx.Tuple[int]) -> None: ...

    def two(x: tx.Tuple[int, str]) -> None: ...

    assert not Signature.from_callable(one).same_as(
        Signature.from_callable(two)
    )


def test_same_as_compares_varargs_and_varkw() -> None:
    """`same_as` compares the `*args`/`**kwargs` hints structurally."""

    def a(*args: int) -> None: ...

    def b(*args: int) -> None: ...

    def c(*args: str) -> None: ...

    assert Signature.from_callable(a).same_as(Signature.from_callable(b))
    # Same shape, different `*args` hint -> not the same.
    assert not Signature.from_callable(a).same_as(Signature.from_callable(c))


def test_same_as_varargs_presence_differs() -> None:
    """Same parameters but one has `*args` and the other does not -> differ."""

    def has_varargs(x: int, *args: int) -> None: ...

    def no_varargs(x: int) -> None: ...

    assert not Signature.from_callable(has_varargs).same_as(
        Signature.from_callable(no_varargs)
    )


def test_from_callable_empty_class_takes_object_constructor() -> None:
    """A class with neither `__init__` nor `__new__` has no dispatched args."""

    class Empty:
        pass

    sig = Signature.from_callable(Empty)
    assert list(sig.parameters) == []
