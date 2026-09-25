"""Tests for the method layer (`_method.py`)."""

# stdlib
import typing

# dependencies
import pytest

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._method import Method
from bagof.dispatchers._signature import Signature


def test_method_calls_the_function() -> None:
    """Calling a method runs the wrapped function."""

    def add(x: int, y: int) -> int:
        return x + y

    method = Method(add)
    assert method(2, 3) == 5


def test_method_reads_signature_from_function() -> None:
    """A method builds its signature from the function by default."""

    def f(x: int, y: str = "a") -> None: ...

    method = Method(f)
    assert list(method.signature.parameters) == ["x", "y"]


def test_method_accepts_explicit_signature() -> None:
    """An explicit signature is used as given."""
    sig = Signature.from_hints(int, scale=float)
    method = Method(lambda *a, **k: None, sig)
    assert method.signature is sig


def test_method_priority_default_and_set() -> None:
    """Priority defaults to zero and is stored as given."""

    def f(x: int) -> None: ...

    assert Method(f).priority == 0
    assert Method(f, priority=5).priority == 5


def test_method_repr_renders_named_signature() -> None:
    """The repr shows names, hints, a default, and the source location."""

    def area(shape: int, scale: float = 1.0) -> float:
        return 0.0

    text = repr(Method(area))
    assert text.startswith("area(shape: int, scale: float = 1.0) @ ")
    assert "test_method.py:" in text


def test_method_repr_markers() -> None:
    """The repr renders `/`, `*`, `*args` and `**kwargs` markers."""

    def f(a: int, /, b: int, *args: str, c: int, **kw: float) -> None: ...

    text = repr(Method(f))
    body = text.split(" @ ")[0]
    assert body == (
        "f(a: int, /, b: int, *args: str, c: int, **kwargs: float)"
    )


def test_method_repr_star_marker_without_varargs() -> None:
    """A keyword-only group with no `*args` gets a bare `*` marker."""

    def f(a: int, *, b: int) -> None: ...

    body = repr(Method(f)).split(" @ ")[0]
    assert body == "f(a: int, *, b: int)"


def test_method_repr_unannotated_is_any() -> None:
    """An unannotated parameter renders as `Any`."""

    def f(x, y: int) -> None:  # noqa: ANN001 -- unannotated on purpose
        ...

    body = repr(Method(f)).split(" @ ")[0]
    assert body == "f(x: Any, y: int)"


def test_method_repr_exact() -> None:
    """`Exact[int]` renders as `Exact[int]`, not its `Annotated` spelling."""

    def f(x: Exact[int]) -> None: ...

    body = repr(Method(f)).split(" @ ")[0]
    assert body == "f(x: Exact[int])"


def test_method_repr_generic_hint() -> None:
    """A typing generic renders without the `typing.` prefix."""

    def f(x: typing.List[int]) -> None: ...

    body = repr(Method(f)).split(" @ ")[0]
    assert body == "f(x: List[int])"


def test_method_location() -> None:
    """The location is `basename:line` for a source-backed function."""

    def f(x: int) -> None: ...

    method = Method(f)
    assert method.location.startswith("test_method.py:")
    assert method.lineno == f.__code__.co_firstlineno


def test_method_repr_deferred_uses_raw_annotation() -> None:
    """A deferred forward reference renders by name, without resolving."""
    namespace = {}  # type: dict
    exec("def f(x: 'Later'): pass", namespace)
    method = Method(namespace["f"])
    body = repr(method).split(" @ ")[0]
    assert body == "f(x: Later)"


def test_method_equality() -> None:
    """Two methods are equal when they wrap the same function and priority."""

    def f(x: int) -> None: ...

    assert Method(f) == Method(f)
    assert Method(f, priority=1) != Method(f, priority=2)

    def g(x: int) -> None: ...

    assert Method(f) != Method(g)


def test_method_repr_no_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A callable with no source file falls back to `<module>`."""

    def f(x: int) -> None: ...

    import bagof.dispatchers._method as module

    monkeypatch.setattr(
        module.inspect,
        "getsourcefile",
        lambda fn: (_ for _ in ()).throw(TypeError()),
    )
    method = Method(f)
    assert method.filename == "<module>"
