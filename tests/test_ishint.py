"""Tests for `ishint`, the predicate that tells a hint from a plain value."""

# stdlib
import typing

# dependencies
import typing_extensions as tx

# local
from bagof.dispatchers.core import ishint
from bagof.dispatchers.core._compat import is_plausible_hint
from bagof.dispatchers.core._exact import EXACT, Exact


def test_none_is_a_hint() -> None:
    # A bare `None` stands for `NoneType`.
    assert ishint(None) is True


def test_strings_and_forward_refs_are_hints() -> None:
    assert ishint("Foo") is True
    assert ishint(tx.ForwardRef("Foo")) is True


def test_classes_are_hints() -> None:
    assert ishint(int) is True
    assert ishint(object) is True

    class Custom:
        pass

    assert ishint(Custom) is True


def test_parametrised_generics_are_hints() -> None:
    # A subscripted generic has a typing origin.
    assert ishint(tx.List[int]) is True
    assert ishint(tx.Dict[str, int]) is True
    assert ishint(tx.Optional[int]) is True
    assert ishint(Exact[int]) is True


def test_special_forms_are_hints() -> None:
    assert ishint(tx.Any) is True
    assert ishint(tx.Union) is True
    assert ishint(tx.Literal) is True
    assert ishint(tx.Annotated) is True
    assert ishint(tx.Optional) is True


def test_typevar_family_members_are_hints() -> None:
    # Their runtime type lives in a typing module.
    assert ishint(tx.TypeVar("T")) is True
    assert ishint(tx.ParamSpec("P")) is True
    assert ishint(tx.TypeVarTuple("Ts")) is True


def test_typing_module_objects_are_hints() -> None:
    assert ishint(tx.Self) is True
    assert ishint(tx.Final) is True


def test_typeddict_marker_is_a_hint() -> None:
    assert ishint(tx.TypedDict) is True
    assert ishint(typing.TypedDict) is True


def test_newtype_is_a_hint() -> None:
    UserId = tx.NewType("UserId", int)
    assert ishint(UserId) is True


def test_plain_values_are_not_hints() -> None:
    assert ishint(1) is False
    assert ishint([]) is False
    assert ishint({}) is False
    assert ishint(object()) is False
    assert ishint(...) is False


def test_functions_are_not_hints() -> None:
    assert ishint(typing.cast) is False

    def plain() -> None:
        pass

    assert ishint(plain) is False
    assert ishint(lambda: None) is False


def test_callable_instances_are_not_hints() -> None:
    class Callable:
        def __call__(self) -> None:
            pass

    assert ishint(Callable()) is False


def test_the_exact_marker_is_not_a_hint() -> None:
    # `EXACT` is `Annotated` metadata, not a hint in its own right.
    assert ishint(EXACT) is False


def test_is_plausible_hint_is_ishint_minus_strings() -> None:
    # The two predicates never drift: a plausible hint is exactly a hint
    # that is not a bare string.
    for obj in (int, tx.Union, tx.Any, None, tx.List[int], 1, [], object()):
        assert is_plausible_hint(obj) == (
            ishint(obj) and not isinstance(obj, str)
        )
    # The one place they differ: a bare string is a hint but not plausible.
    assert ishint("Foo") is True
    assert is_plausible_hint("Foo") is False
