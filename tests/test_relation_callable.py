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


def test_paramspec_is_a_wildcard_parameter_list() -> None:
    # A bare `ParamSpec` parameter list is a wildcard `...` on either side
    # (RFC 11.1): `Callable[P, R]` is equivalent to `Callable[..., R]`.
    P = tx.ParamSpec("P")
    Q = tx.ParamSpec("Q")
    assert issubhint(C[P, int], C[Q, int]) is True
    assert issubhint(C[P, int], C[..., int]) is True
    assert issubhint(C[..., int], C[P, int]) is True
    assert issubhint(C[P, int], C[[int], int]) is True
    # The return type stays covariant even with a ParamSpec parameter list.
    assert issubhint(C[P, bool], C[P, int]) is True
    assert issubhint(C[P, int], C[P, bool]) is False
    # `Concatenate[X, P]` is a contravariant fixed prefix followed by `...`.
    concat = C[tx.Concatenate[int, P], str]
    assert issubhint(concat, C[[int], str]) is True
    assert issubhint(concat, C[[bool], str]) is True  # prefix contravariant
    assert issubhint(C[tx.Concatenate[bool, P], str], C[[int], str]) is False
    # A fixed list shorter than the prefix cannot match.
    assert issubhint(concat, C[[], str]) is False


def test_a_function_value_is_a_callable_instance() -> None:
    assert ishintstance(print, C[[int], str]) is True
    assert ishintstance(42, C[[int], str]) is False


def test_a_callable_class_is_a_callable() -> None:
    # R2: a callable *class* -- `type`, a function type, `Type[C]`, or a class
    # with `__call__` -- is a sub-hint of a bare `Callable`.
    import types

    class Caller:
        def __call__(self) -> int:
            return 0

    assert issubhint(type, C) is True
    assert issubhint(types.FunctionType, C) is True
    assert issubhint(tx.Type[int], C) is True
    assert issubhint(Caller, C) is True
    # A callable class has no parameter list, so it cannot stand in for a
    # parametrised `Callable[...]`.
    assert issubhint(type, C[[int], str]) is False
