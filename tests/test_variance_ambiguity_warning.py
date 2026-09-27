"""Priority-aware ambiguity reporting (V4 of #50).

Registration warns and `Function.ambiguities()` lists a pair only when a call
could reach both with no most specific method *and* nothing resolves the tie. A
differing `priority` resolves it deterministically -- the higher one wins -- so
such a pair is neither warned at registration nor listed, though an equal-
priority clash of the same shape still is.

Every hint is spelled through `typing_extensions` and runs on 3.8, so no
`X | Y` or `list[int]`.
"""

# stdlib
import warnings

# dependencies
import pytest

# locals
from bagof.dispatchers._errors import AmbiguousMethodError
from bagof.dispatchers._function import Function


def _clashing(f, low_priority=0, high_priority=0):  # noqa: ANN001, ANN202
    """Register two incomparable overloads at the given priorities.

    `(float, object)` and `(object, float)` are incomparable: a call of two
    floats matches both, neither more specific -- a guaranteed clash.
    """

    def by_first(x: float, y: object) -> str:
        return "first"

    def by_second(x: object, y: float) -> str:
        return "second"

    f.register(by_first, priority=low_priority)
    f.register(by_second, priority=high_priority)


def test_equal_priority_clash_warns_and_is_listed() -> None:
    """An incomparable pair at equal priority warns and is listed."""
    f = Function("f")
    with pytest.warns(RuntimeWarning, match="ambiguous"):
        _clashing(f)
    assert len(f.ambiguities()) == 1
    with pytest.raises(AmbiguousMethodError):
        f(2.0, 3.0)


def test_differing_priority_is_not_ambiguous() -> None:
    """A clash split by `priority` does not warn, is not listed, and resolves.

    The higher-priority overload wins deterministically at dispatch, so the
    pair is not ambiguous at a call and must not be reported.
    """
    f = Function("f")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _clashing(f, low_priority=0, high_priority=1)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert f.ambiguities() == []
    assert f(2.0, 3.0) == "second"  # the priority=1 overload wins


def test_differing_priority_either_order() -> None:
    """The tie is resolved whichever overload carries the higher priority."""
    f = Function("f")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _clashing(f, low_priority=5, high_priority=1)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert f.ambiguities() == []
    assert f(2.0, 3.0) == "first"  # the priority=5 overload wins
