"""Tests for `Hint[X]`: construction, helpers, and dispatching on hints."""

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers import Exact, Function, Hint, Super, SuperHint
from bagof.dispatchers.core import ishint, ishintstance, issubhint
from bagof.dispatchers.core._hint import hint_arg, is_hint_form

# --- construction ------------------------------------------------------


def test_hint_of_a_class() -> None:
    assert tx.get_origin(Hint[int]) is Hint
    assert tx.get_args(Hint[int]) == (int,)


def test_hint_of_a_bare_special_form_uses_the_alias_fallback() -> None:
    # A bare special form must build on every version. Before Python 3.10 the
    # Generic subscription refuses one and the direct-alias fallback handles
    # it; 3.10 and later accept it through the ordinary path. Either way the
    # result is the same alias.
    for form in (tx.Union, tx.Literal, tx.Annotated):
        alias = Hint[form]
        assert tx.get_origin(alias) is Hint
        assert tx.get_args(alias) == (form,)


def test_hint_of_none() -> None:
    # `None` is accepted and read as `NoneType`, the way typing reads it.
    alias = Hint[None]
    assert tx.get_origin(alias) is Hint
    assert tx.get_args(alias) == (type(None),)


def test_hint_nested() -> None:
    alias = Hint[Hint[int]]
    assert tx.get_origin(alias) is Hint
    assert tx.get_args(alias) == (Hint[int],)


def test_hint_is_hashable_and_comparable() -> None:
    assert Hint[int] == Hint[int]
    assert hash(Hint[int]) == hash(Hint[int])
    assert Hint[tx.Union] == Hint[tx.Union]


def test_hint_nests_in_optional() -> None:
    assert tx.get_origin(tx.Optional[Hint[int]]) is tx.Union


def test_hint_of_a_non_hint_raises() -> None:
    with pytest.raises(TypeError):
        Hint[1]


def test_hint_is_not_instantiable() -> None:
    with pytest.raises(TypeError):
        Hint()
    with pytest.raises(TypeError):
        Hint[int]()


def test_hint_and_hint_arg_are_a_hint() -> None:
    assert ishint(Hint[int]) is True
    assert ishint(Hint) is True


# --- helpers -----------------------------------------------------------


def test_is_hint_form() -> None:
    assert is_hint_form(Hint) is True
    assert is_hint_form(Hint[int]) is True
    assert is_hint_form(Hint[tx.Union]) is True
    assert is_hint_form(int) is False
    assert is_hint_form(tx.List[int]) is False


def test_hint_arg() -> None:
    assert hint_arg(Hint[int]) is int
    assert hint_arg(Hint[tx.Union]) is tx.Union
    # A bare `Hint` stands for `Hint[Any]`.
    assert hint_arg(Hint) is tx.Any


def test_hint_helpers_look_through_annotated() -> None:
    wrapped = tx.Annotated[Hint[int], "meta"]
    assert is_hint_form(wrapped) is True
    assert hint_arg(wrapped) is int


def test_hint_of_exact_matches_a_bare_alias_and_its_origin() -> None:
    # `_same_hint` reduces a bare typing alias to its origin class, so
    # `Hint[Exact[list]]` matches both `list` and the bare `List`.
    assert ishintstance(list, Hint[Exact[list]]) is True
    assert ishintstance(tx.List, Hint[Exact[list]]) is True
    assert ishintstance(dict, Hint[Exact[list]]) is False


def test_hint_of_exact_never_matches_both_bottom_spellings() -> None:
    # `Never` and `NoReturn` are one bottom, so either matches the other's
    # `Hint[Exact[...]]`.
    assert ishintstance(tx.Never, Hint[Exact[tx.Never]]) is True
    assert ishintstance(tx.NoReturn, Hint[Exact[tx.Never]]) is True
    assert ishintstance(tx.Never, Hint[Exact[tx.NoReturn]]) is True
    assert ishintstance(int, Hint[Exact[tx.Never]]) is False


# --- value level -------------------------------------------------------


def test_value_level_plain() -> None:
    assert ishintstance(int, Hint[int]) is True
    assert ishintstance(bool, Hint[int]) is True
    assert ishintstance(str, Hint[int]) is False
    # A non-hint value never belongs to a `Hint`.
    assert ishintstance(1, Hint[int]) is False


def test_value_level_bare_hint_is_any() -> None:
    assert ishintstance(int, Hint) is True
    assert ishintstance(tx.Union[int, str], Hint) is True


def test_value_level_hint_of_union() -> None:
    assert ishintstance(tx.Union[int, str], Hint[tx.Union]) is True
    assert ishintstance(int, Hint[tx.Union]) is False


def test_value_level_hint_of_exact() -> None:
    assert ishintstance(int, Hint[Exact[int]]) is True
    assert ishintstance(bool, Hint[Exact[int]]) is False
    assert ishintstance(tx.Union, Hint[Exact[tx.Union]]) is True
    assert ishintstance(tx.Union[int, str], Hint[Exact[tx.Union]]) is False
    assert ishintstance(tx.Any, Hint[Exact[tx.Any]]) is True
    assert ishintstance(int, Hint[Exact[tx.Any]]) is False


def test_hint_of_exact_is_structural_not_equivalent() -> None:
    # `Hint[Exact[X]]` matches only the structurally identical hint `X`, not a
    # hint merely equivalent to it. A free `TypeVar` accepts the same values as
    # `Any` and so is equivalent to it, but it is not the hint `Any`, so it
    # does not match `Hint[Exact[Any]]`. This is what makes the exact match use
    # `_same_hint` rather than `_equivalent`.
    T = tx.TypeVar("T")
    assert ishintstance(T, Hint[Exact[tx.Any]]) is False
    assert ishintstance(tx.Any, Hint[Exact[tx.Any]]) is True
    assert issubhint(Hint[Exact[T]], Hint[Exact[tx.Any]]) is False
    assert issubhint(Hint[Exact[tx.Any]], Hint[Exact[tx.Any]]) is True


# --- hint level --------------------------------------------------------
#
# At the hint level, only another `Hint` form is ever a sub-hint of a `Hint`
# form: a plain `int` describes int *values*, while `Hint[int]` describes the
# *hints* below int, so the two are never ordered against each other. The
# covariant, value-facing behaviour lives at the value level (`ishintstance`)
# above.


def test_hint_level_plain_class_is_not_a_hint_form() -> None:
    assert issubhint(int, Hint[int]) is False
    assert issubhint(int, Hint) is False
    assert issubhint(str, Hint[int]) is False


def test_hint_level_hint_to_hint() -> None:
    assert issubhint(Hint[bool], Hint[int]) is True
    assert issubhint(Hint[int], Hint[bool]) is False
    assert issubhint(Hint[int], Hint) is True
    assert issubhint(Hint[int], Hint[tx.Any]) is True


def test_hint_level_hint_of_exact() -> None:
    assert issubhint(Hint[Exact[int]], Hint[int]) is True
    assert issubhint(Hint[Exact[int]], Hint[Exact[int]]) is True
    assert issubhint(Hint[Exact[int]], Hint[Exact[bool]]) is False
    assert issubhint(Hint[Exact[bool]], Hint[Exact[int]]) is False
    # A plain (non-exact) `Hint[int]` is not below `Hint[Exact[int]]`.
    assert issubhint(Hint[int], Hint[Exact[int]]) is False


def test_hint_of_union_forms() -> None:
    assert issubhint(Hint[Exact[tx.Union]], Hint[tx.Union]) is True
    # A parametrised union hint sits below `Hint[Union]` (it is a union).
    assert issubhint(Hint[tx.Union[int, str]], Hint[tx.Union]) is True


def test_hint_form_is_not_below_a_plain_class() -> None:
    # A `Hint` form describes hints, not values, so it is below nothing
    # ordinary.
    assert issubhint(Hint[int], object) is False
    assert issubhint(Hint, object) is False


def test_hint_form_is_below_any() -> None:
    # `Any` is the one top a `Hint` form sits below.
    assert issubhint(Hint[int], tx.Any) is True
    assert issubhint(Hint, tx.Any) is True


# --- dispatch smoke test -----------------------------------------------


def test_dispatch_on_hints() -> None:
    f = Function("classify")

    @f.register((Hint[Exact[tx.Union]],))
    def _bare_union(h: object) -> str:
        return "bare-union"

    @f.register((Hint[tx.Union],))
    def _any_union(h: object) -> str:
        return "union"

    @f.register((Hint[int],))
    def _ints(h: object) -> str:
        return "int"

    @f.register((Hint[tx.Any],))
    def _any(h: object) -> str:
        return "any"

    # The hints above `object` are `object` itself and the tops, which no
    # other method here shares, so this one is selected for `object` alone.
    @f.register((Hint[Super[object]],))
    def _above_object(h: object) -> str:
        return "above-object"

    assert f(object) == "above-object"
    assert f(int) == "int"
    assert f(bool) == "int"
    assert f(tx.Union[int, str]) == "union"
    assert f(tx.Union) == "bare-union"
    assert f(str) == "any"


def test_superhint_is_hint_of_super() -> None:
    assert SuperHint[int] == Hint[Super[int]]
    assert is_hint_form(SuperHint[int]) is True
    assert hint_arg(SuperHint[int]) == Super[int]


def test_value_level_hint_of_super() -> None:
    # `Hint[Super[X]]` matches `X` and every hint above it.
    assert ishintstance(bool, Hint[Super[bool]]) is True
    assert ishintstance(int, Hint[Super[bool]]) is True
    assert ishintstance(tx.Optional[int], Hint[Super[bool]]) is True
    assert ishintstance(str, Hint[Super[bool]]) is False
    assert ishintstance(tx.Literal[True], Hint[Super[bool]]) is False


def test_hint_level_hint_of_super() -> None:
    # Ordered contravariantly by the bound, above the exact hint, and apart
    # from a plain argument.
    assert issubhint(Hint[Super[int]], Hint[Super[bool]]) is True
    assert issubhint(Hint[Super[bool]], Hint[Super[int]]) is False
    assert issubhint(Hint[Exact[bool]], Hint[Super[bool]]) is True
    assert issubhint(Hint[bool], Hint[Super[bool]]) is False
    assert issubhint(Hint[Super[bool]], Hint[int]) is False
    assert issubhint(Hint[Super[bool]], Hint) is True


def test_resolve_non_hint_query_raises() -> None:
    f = Function("f")

    @f.register((int,))
    def _one(x: object) -> int:
        return 1

    with pytest.raises(TypeError):
        f.resolve(1)
