"""Tests for `Callable` variance in `issubhint`."""

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers.core import ishintstance, issubhint

C = tx.Callable

# (hint, superhint, expected)
CALLABLE_CASES = [
    # Parameters are contravariant: a callable taking a wider argument can
    # stand in for one taking a narrower argument.
    (C[[int], str], C[[bool], str], True),
    (C[[bool], str], C[[int], str], False),
    (C[[int], str], C[[int], str], True),
    # The return type is covariant.
    (C[[int], bool], C[[int], int], True),
    (C[[int], int], C[[int], bool], False),
    # Both at once.
    (C[[int], bool], C[[bool], int], True),
    (C[[bool], int], C[[int], bool], False),
    # `Callable[..., R]` accepts any parameter list, either side.
    (C[[int], str], C[..., str], True),
    (C[..., str], C[[int], str], True),
    (C[..., bool], C[..., int], True),
    (C[..., int], C[..., bool], False),
    # Arity must match for fixed lists.
    (C[[int, str], bool], C[[int], bool], False),
    (C[[int], bool], C[[int, str], bool], False),
    (C[[int, str], bool], C[[int, str], bool], True),
    # A bare `Callable` constrains nothing; a parametrised one is not a
    # sub-hint of a bare callable's... other way round.
    (C[[int], str], C, True),
    (C, C[[int], str], False),
    # A callable is a plain object.
    (C[[int], str], object, True),
    # A non-callable hint is not a sub-hint of a Callable.
    (int, C[[int], str], False),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    CALLABLE_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in CALLABLE_CASES],
)
def test_issubhint_callable(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_paramspec_degrades_to_equality() -> None:
    # `ParamSpec` / `Concatenate` parameter lists are compared for equality
    # only -- comparable to an identical one, and to nothing else, rather
    # than raising.
    P = tx.ParamSpec("P")
    assert issubhint(C[P, int], C[P, int]) is True
    assert issubhint(C[P, bool], C[P, int]) is True  # return covariant
    assert issubhint(C[P, int], C[P, bool]) is False
    Q = tx.ParamSpec("Q")
    assert issubhint(C[P, int], C[Q, int]) is False
    concat = C[tx.Concatenate[int, P], str]
    assert issubhint(concat, concat) is True
    assert issubhint(concat, C[[int], str]) is False


def test_a_function_value_is_a_callable_instance() -> None:
    assert ishintstance(print, C[[int], str]) is True
    assert ishintstance(42, C[[int], str]) is False
