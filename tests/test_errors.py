"""Tests for dispatch error rendering (`_errors.py` + `Function`)."""

# stdlib
import typing
import warnings

# dependencies
import pytest

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._errors import (
    AmbiguousMethodError,
    DispatchError,
    NoMethodError,
    did_you_mean,
    render_ambiguous,
    render_no_method,
)
from bagof.dispatchers._function import Function
from bagof.dispatchers._method import Method
from bagof.dispatchers._signature import Signature


def _quiet(function: Function, *fns: typing.Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for fn in fns:
            function.register(fn)


# --- the highlight hook (Phase-3 renderer) -----------------------------


def test_describe_marks_a_parameter() -> None:
    """`describe(highlight)` puts a `!` on the named parameter's hint."""

    def area(x: int, y: int) -> int:
        return x * y

    method = Method(area)
    assert "x: !int" in method.describe(["x"])
    assert "y: !int" in method.describe(["y"])
    assert "!" not in method.describe()


def test_describe_marks_catch_alls() -> None:
    """The catch-all sentinels mark `*args` / `**kwargs`."""
    from bagof.dispatchers._signature import Parameter

    def f(*args: int, **kwargs: str) -> None: ...

    method = Method(f)
    marked = method.describe([Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD])
    assert "*args: !int" in marked
    assert "**kwargs: !str" in marked


def test_describe_marks_unannotated_catch_alls() -> None:
    """An unannotated `*args` / `**kwargs` still shows the mark."""
    from bagof.dispatchers._signature import Parameter

    def f(*args, **kwargs) -> None: ...

    method = Method(f)
    marked = method.describe([Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD])
    assert "*args: !Any" in marked
    assert "**kwargs: !Any" in marked


# --- no-method messages ------------------------------------------------


def test_no_method_marks_the_failing_argument() -> None:
    """The offending argument is marked with `!` in the closest candidates."""
    f = Function("area")

    def circle(x: int) -> int:
        return x

    f.register(circle)
    with pytest.raises(NoMethodError) as info:
        f("string")
    message = str(info.value)
    assert "no method matching area(str)" in message
    assert "x: !int" in message
    assert "Closest candidates:" in message


def test_no_method_reports_binding_failure_in_words() -> None:
    """A candidate that cannot bind is described in words, not with a `!`."""
    f = Function("f")

    def two(x: int, y: int) -> int:
        return x + y

    f.register(two)
    with pytest.raises(NoMethodError) as info:
        f(1)  # missing y
    assert "missing 'y'" in str(info.value)


def test_no_method_positional_only_reason() -> None:
    """A positional-only parameter passed by keyword is named as such."""
    f = Function("p")
    f._add(Method(lambda v: v, Signature.from_hints(int)))
    with pytest.raises(NoMethodError) as info:
        f(_0=3)
    assert "positional-only" in str(info.value)


def test_no_method_too_many_positionals_reason() -> None:
    """Too many positional arguments is named."""
    f = Function("f")

    def one(x: int) -> int:
        return x

    f.register(one)
    with pytest.raises(NoMethodError) as info:
        f(1, 2, 3)
    assert "positional argument" in str(info.value)


def test_no_method_unknown_keyword_did_you_mean() -> None:
    """An unknown keyword gets a did-you-mean over all parameter names."""
    f = Function("f")

    def m(scale: float) -> float:
        return scale

    f.register(m)
    with pytest.raises(NoMethodError) as info:
        f(scake=2.0)  # typo for scale
    message = str(info.value)
    assert "keyword 'scake'" in message
    assert "Did you mean 'scale'?" in message


def test_no_method_argument_given_twice() -> None:
    """The same argument passed twice is reported."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    f.register(m)
    with pytest.raises(NoMethodError) as info:
        f(1, x=1)
    assert "twice" in str(info.value)


def test_no_method_never_shows_values() -> None:
    """The message shows argument *types*, never their values."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    secret = "s3cr3t-value"
    with pytest.raises(NoMethodError) as info:
        f(secret)
    assert secret not in str(info.value)


# --- ambiguity messages ------------------------------------------------


def test_ambiguous_lists_candidates_and_fix() -> None:
    """The ambiguity message lists candidates and a possible fix."""
    f = Function("area")

    def by_x(x: float, y: object) -> int:
        return 1

    def by_y(x: object, y: float) -> int:
        return 2

    _quiet(f, by_x, by_y)
    with pytest.raises(AmbiguousMethodError) as info:
        f(2.0, 3.0)
    message = str(info.value)
    assert "area(float, float) is ambiguous" in message
    assert "Candidates:" in message
    assert "Possible fix, define" in message
    assert "area(x: float, y: float)" in message


def test_possible_fix_uses_exact_where_a_candidate_did() -> None:
    """A candidate matching with `Exact` makes the fix spell `Exact`."""
    f = Function("f")

    def exact_first(x: Exact[int], y: object) -> int:
        return 1

    def by_second(x: object, y: float) -> int:
        return 2

    _quiet(f, exact_first, by_second)
    with pytest.raises(AmbiguousMethodError) as info:
        f(3, 2.0)
    assert "Exact[int]" in str(info.value)


def test_ambiguous_never_shows_values() -> None:
    """The ambiguity message shows types, never values."""
    f = Function("f")

    def by_x(x: float, y: object) -> int:
        return 1

    def by_y(x: object, y: float) -> int:
        return 2

    _quiet(f, by_x, by_y)
    with pytest.raises(AmbiguousMethodError) as info:
        f(1.5, 2.5)
    assert "1.5" not in str(info.value)


# --- resolve error rendering (hint level) ------------------------------


def test_resolve_ambiguous_renders_hints() -> None:
    """A hint-level ambiguity renders the query and fix by hint."""
    f = Function("f")

    def by_x(x: float, y: object) -> int:
        return 1

    def by_y(x: object, y: float) -> int:
        return 2

    _quiet(f, by_x, by_y)
    with pytest.raises(AmbiguousMethodError) as info:
        f.resolve(float, float)
    assert "f(float, float) is ambiguous" in str(info.value)


def test_no_method_keyword_call_desc() -> None:
    """A keyword call renders `name=Type` in the description."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    f.register(m)
    with pytest.raises(NoMethodError) as info:
        f(y="oops")
    assert "f(y=str)" in str(info.value)


# --- the error attributes ----------------------------------------------


def test_error_attributes() -> None:
    """A dispatch error carries the function, call and candidates."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    f.register(m)
    with pytest.raises(NoMethodError) as info:
        f("a")
    error = info.value
    assert error.function == "f"
    assert "f(str)" in error.call
    assert len(error.candidates) == 1


def test_dispatch_error_hierarchy() -> None:
    """The two errors are `DispatchError`s and `TypeError`s."""
    assert issubclass(NoMethodError, DispatchError)
    assert issubclass(AmbiguousMethodError, DispatchError)
    assert issubclass(DispatchError, TypeError)


# --- the pure render helpers -------------------------------------------


def test_render_no_method_singular_plural() -> None:
    """The summary agrees in number with the method count."""
    one = render_no_method("f", "f(str)", 1, [])
    assert "has 1 method," in one
    many = render_no_method("f", "f(str)", 3, [])
    assert "has 3 methods," in many


def test_render_no_method_empty() -> None:
    """No methods at all points at the missing import."""
    text = render_no_method("f", "f(str)", 0, [])
    assert "no methods yet" in text


def test_render_ambiguous_shape() -> None:
    """The ambiguity template lays out candidates and the fix."""
    text = render_ambiguous("f(int, int)", ["a @ x:1", "b @ x:2"], "f(x: int)")
    assert text.startswith("f(int, int) is ambiguous.")
    assert "Possible fix, define" in text


def test_did_you_mean() -> None:
    """The did-you-mean helper finds a near match, or nothing."""
    assert did_you_mean("scakе".encode("ascii", "ignore").decode(), []) is None
    assert did_you_mean("scal", ["scale", "shape"]) == "scale"
    assert did_you_mean("zzzzz", ["scale", "shape"]) is None
