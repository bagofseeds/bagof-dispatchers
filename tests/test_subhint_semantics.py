"""Tests for `issubhint`/`ishintstance`'s hint semantics."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers.core import (
    get_concrete_type,
    ishintstance,
    issubhint,
    normalise_hint,
)

# --- covariant argument comparison -------------------------------------

ARG_CASES = [
    # A parametrised hint is a subhint of its own origin.
    (tx.List[int], list, True),
    (tx.Dict[str, int], dict, True),
    (tx.List[int], object, True),
    # `Annotated` is transparent, in both directions.
    (tx.Annotated[int, "meta"], int, True),
    (int, tx.Annotated[int, "meta"], True),
    (tx.Annotated[tx.List[int], "meta"], list, True),
    # A bare origin cannot stand in for a parametrised hint: it may hold
    # anything at all.
    (list, tx.List[int], False),
    (dict, tx.Dict[str, int], False),
    # Arguments are compared covariantly.
    (tx.List[bool], tx.List[int], True),
    (tx.List[str], tx.List[int], False),
    (tx.Dict[str, bool], tx.Dict[str, int], True),
    (tx.List[int], tx.List[tx.Any], True),
    # ... including through the container hierarchy.
    (tx.List[int], tx.Sequence[int], True),
    (tx.List[int], tx.Iterable[int], True),
    (tx.List[bool], tx.Sequence[int], True),
    (tx.List[str], tx.Sequence[int], False),
    # Arity must match.
    (tx.Tuple[int], tx.Tuple[int, str], False),
    (tx.Tuple[int, str], tx.Tuple[int, str], True),
    # A trailing ellipsis means "any number of these".
    (tx.Tuple[bool, bool], tx.Tuple[int, ...], True),
    (tx.Tuple[str, int], tx.Tuple[int, ...], False),
    (tx.Tuple[int, ...], tx.Tuple[int, ...], True),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    ARG_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in ARG_CASES],
)
def test_issubhint_compares_arguments(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_no_unsound_widening() -> None:
    # Every one of these used to be True, and each admits values the
    # superhint rejects.
    assert issubhint(list, tx.List[int]) is False
    assert issubhint(dict, tx.Dict[str, int]) is False
    assert issubhint(tx.Any, tx.Union[int, str]) is False


# --- the bare Union / Literal forms are structural ---------------------


@pytest.mark.parametrize(
    "hint,expected",
    [
        (tx.Union[int, str], True),
        (tx.Optional[int], True),
        (int, False),
        (None, False),
        (tx.Any, False),
        (tx.Literal[1], False),
    ],
)
def test_only_a_union_is_a_subhint_of_the_bare_union(
    hint: tx.Any, expected: bool
) -> None:
    # Matches what `_issubliteral` has always done for a bare `Literal`,
    # and is what makes `BOUND = tx.Union` mean anything.
    assert issubhint(hint, tx.Union) is expected


def test_a_parametrised_union_still_accepts_its_members() -> None:
    # The bare form asks "is this a union?"; the parametrised form asks
    # "is this one of these?". Both must keep working.
    assert issubhint(int, tx.Union[int, str]) is True
    assert issubhint(bool, tx.Union[int, str]) is True
    assert issubhint(bytes, tx.Union[int, str]) is False
    assert issubhint(tx.Union, tx.Union[int, str]) is False


def test_a_constrained_typevar_is_the_union_it_stands_for() -> None:
    constrained = tx.TypeVar("constrained", int, str)
    assert issubhint(constrained, tx.Union) is True
    assert issubhint(constrained, tx.Union[int, str, bytes]) is True
    assert issubhint(constrained, tx.Union[int, bytes]) is False


# --- Literal instance checks (PEP 586) ---------------------------------


@pytest.mark.parametrize(
    "obj,hint,expected",
    [
        (1, tx.Literal[1, 2], True),
        (3, tx.Literal[1, 2], False),
        ("a", tx.Literal["a"], True),
        ("b", tx.Literal["a"], False),
        # PEP 586 makes literal matching type-aware, so `True == 1` does
        # not make `True` a valid `Literal[1]`.
        (True, tx.Literal[1], False),
        (1, tx.Literal[True], False),
        (True, tx.Literal[True], True),
        (1.0, tx.Literal[1], False),
        # `Annotated` is transparent here too.
        (1, tx.Annotated[tx.Literal[1], "meta"], True),
        (None, tx.Literal[None], True),
    ],
)
def test_ishintstance_literal(
    obj: tx.Any, hint: tx.Any, expected: bool
) -> None:
    assert ishintstance(obj, hint) is expected


def test_a_nan_literal_matches_itself() -> None:
    nan = float("nan")
    assert ishintstance(nan, tx.Literal[nan]) is True


# --- None is NoneType --------------------------------------------------


def test_normalise_hint_replaces_a_bare_none() -> None:
    assert normalise_hint(None) is type(None)
    assert normalise_hint(int) is int


@pytest.mark.parametrize(
    "hint,superhint,expected",
    [
        (None, type(None), True),
        (type(None), None, True),
        (None, None, True),
        (int, None, False),
        (None, int, False),
        (None, tx.Optional[int], True),
    ],
)
def test_none_is_nonetype_as_a_hint(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_none_is_nonetype_for_instance_checks() -> None:
    assert ishintstance(None, None) is True
    assert ishintstance(None, type(None)) is True
    assert ishintstance(1, None) is False


def test_a_none_inside_a_hint_keeps_its_value_meaning() -> None:
    # `Literal[None]` is the *value* None, not the type.
    assert ishintstance(None, tx.Literal[None]) is True
    assert ishintstance(0, tx.Literal[None]) is False


# --- ishintstance does not check item types ---------------------------


def test_item_types_are_not_checked() -> None:
    # A value carries its type; a type carries no arguments. Python
    # refuses `isinstance(x, list[int])` for the same reason.
    assert ishintstance([1, 2], tx.List[str]) is True
    assert ishintstance([1, 2], tx.List[int]) is True
    # The origin still is checked.
    assert ishintstance((1, 2), tx.List[int]) is False


def test_ishintstance_resolves_typevars_and_any() -> None:
    bound = tx.TypeVar("bound", bound=tx.Sequence[int])
    assert ishintstance([1, 2], bound) is True
    assert ishintstance(1, bound) is False
    assert ishintstance(1, tx.TypeVar("free")) is True
    assert ishintstance(1, tx.Any) is True


def test_ishintstance_unions_check_each_member() -> None:
    assert ishintstance([1], tx.Union[tx.List[int], str]) is True
    assert ishintstance("a", tx.Union[tx.List[int], str]) is True
    assert ishintstance(1, tx.Union[tx.List[int], str]) is False


# --- get_concrete_type takes the first constraint ---------------------


def test_first_constraint_is_used() -> None:
    assert get_concrete_type(tx.TypeVar("c", int, str)) is int
    # ... in preference to the fallback.
    assert get_concrete_type(tx.TypeVar("c", int, str), list) is int


def test_a_default_still_wins_over_the_constraints() -> None:
    with_default = tx.TypeVar("d", int, str, default=str)
    assert get_concrete_type(with_default) is str


def test_an_abstract_constraint_is_skipped() -> None:
    partly_abstract = tx.TypeVar("a", tx.Sequence[int], int)
    assert get_concrete_type(partly_abstract) is int


# --- super-hints that constrain nothing, or nothing checkable ----------


def test_annotated_any_superhint_accepts_everything() -> None:
    assert issubhint(int, tx.Annotated[tx.Any, "meta"]) is True


def test_an_uncheckable_superhint_is_never_matched() -> None:
    # `Self` has no origin that can be compared against a hint.
    assert issubhint(int, tx.Self) is False


# --- ellipsis arguments ------------------------------------------------


ELLIPSIS_CASES = [
    # A repeated argument cannot stand in for a fixed-arity hint.
    (tx.Tuple[int, ...], tx.Tuple[int, str], False),
    (tx.Tuple[int, ...], tx.Tuple[int], False),
    # Repeated against repeated, compared covariantly.
    (tx.Tuple[bool, ...], tx.Tuple[int, ...], True),
    (tx.Tuple[str, ...], tx.Tuple[int, ...], False),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    ELLIPSIS_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in ELLIPSIS_CASES],
)
def test_issubhint_with_ellipsis_arguments(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


# --- the family helpers reject a super-hint of the wrong kind ----------


def test_issubnone_rejects_a_non_none_superhint() -> None:
    # locals
    from bagof.dispatchers.core._relation import _issubnone

    with pytest.raises(TypeError, match="is not a NoneType"):
        _issubnone(type(None), int)


def test_issubunion_rejects_a_non_union_superhint() -> None:
    # locals
    from bagof.dispatchers.core._relation import _issubunion

    with pytest.raises(TypeError, match="is not a Union type"):
        _issubunion(int, int)


def test_nan_maps_to_a_recognisable_marker() -> None:
    # locals
    from bagof.dispatchers.core import eq_safenan

    assert repr(eq_safenan(float("nan"))) == "<NaN>"
