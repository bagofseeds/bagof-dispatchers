"""Registration warning for incomparable overlapping overloads (V4 of #50).

After V3, two overloads that differ only in the type arguments of one origin
can be incomparable under the specificity relation yet both apply to the same
runtime value -- value dispatch is shallow, so any `list` matches every
`List[...]`. Such a pair is now flagged at registration (RFC 0001 §2.3, §9): it
warns, it is listed by `Function.ambiguities()`, and a call that hits it still
raises `AmbiguousMethodError`.

A pair that stays strictly ordered under variance -- a covariant `Sequence`, a
contravariant sink, a subclass below its base -- has an outright winner and is
*not* warned. Every hint is spelled through `typing_extensions` and runs on
3.8, so no `X | Y` or `list[int]`.
"""

# stdlib
import warnings

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers._errors import AmbiguousMethodError
from bagof.dispatchers._function import Function, _value_overlap

# --- the variance families ---------------------------------------------

_T = tx.TypeVar("_T")
_T_co = tx.TypeVar("_T_co", covariant=True)
_T_contra = tx.TypeVar("_T_contra", contravariant=True)


class Src(tx.Generic[_T_co]):
    """A covariant user generic (`covariant=True`)."""


class Snk(tx.Generic[_T_contra]):
    """A contravariant user generic (`contravariant=True`)."""


class Box(tx.Generic[_T]):
    """An invariant user generic (an unflagged `TypeVar` is invariant)."""


def _overloads(first_hint, second_hint):  # noqa: ANN001, ANN202
    """A fresh function and two one-argument overloads on the given hints."""
    f = Function("f")

    def first(x: object) -> str:
        return "first"

    def second(x: object) -> str:
        return "second"

    first.__annotations__ = {"x": first_hint, "return": str}
    second.__annotations__ = {"x": second_hint, "return": str}
    return f, first, second


def _register_quietly(f, *fns):  # noqa: ANN001, ANN002, ANN202
    """Register each overload, capturing every warning raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for fn in fns:
            f.register(fn)
    return caught


# --- WARN: incomparable, but a value matches both ----------------------


WARN_CASES = [
    ("List[int] vs List[str]", tx.List[int], tx.List[str], [True, False]),
    (
        "List[int] vs List[bool] (invariant)",
        tx.List[int],
        tx.List[bool],
        [True, False],
    ),
    (
        "Dict[str, int] vs Dict[str, bool]",
        tx.Dict[str, int],
        tx.Dict[str, bool],
        {"k": True},
    ),
    ("Box[int] vs Box[str] (user invariant)", Box[int], Box[str], Box()),
]


@pytest.mark.parametrize(
    "name, first_hint, second_hint, value",
    WARN_CASES,
    ids=[case[0] for case in WARN_CASES],
)
def test_incomparable_overlap_warns_at_registration(
    name: str,
    first_hint: tx.Any,
    second_hint: tx.Any,
    value: tx.Any,
) -> None:
    """Two parametrisations of one origin warn and are listed as ambiguous."""
    f, first, second = _overloads(first_hint, second_hint)
    f.register(first)
    with pytest.warns(RuntimeWarning, match="ambiguous") as caught:
        f.register(second)
    # The message names the offending parameter and why, not any internal name.
    message = str(caught[0].message)
    assert "parameter 'x'" in message
    assert "type arguments" in message
    assert "_ABSENT" not in message and "__magic" not in message
    assert len(f.ambiguities()) == 1


@pytest.mark.parametrize(
    "name, first_hint, second_hint, value",
    WARN_CASES,
    ids=[case[0] for case in WARN_CASES],
)
def test_incomparable_overlap_still_raises_at_call(
    name: str,
    first_hint: tx.Any,
    second_hint: tx.Any,
    value: tx.Any,
) -> None:
    """The call-time behaviour is unchanged: the clash still raises."""
    f, first, second = _overloads(first_hint, second_hint)
    _register_quietly(f, first, second)  # the warning is exercised above
    with pytest.raises(AmbiguousMethodError):
        f(value)


# --- must NOT warn: a strict winner, or no overlap at all --------------


NO_WARN_CASES = [
    # Covariant `Sequence`: `Sequence[bool]` is strictly more specific.
    ("Sequence[int] vs Sequence[bool]", tx.Sequence[int], tx.Sequence[bool]),
    # Contravariant sink: `Snk[int]` is strictly more specific.
    ("Snk[int] vs Snk[bool]", Snk[int], Snk[bool]),
    # Covariant user generic: `Src[bool]` is strictly more specific.
    ("Src[int] vs Src[bool]", Src[int], Src[bool]),
    # A subclass strictly below its base (generic origins ordered).
    ("List[int] vs Sequence[int]", tx.List[int], tx.Sequence[int]),
    # Plain classes with no common value.
    ("int vs str", int, str),
    # A subclass vs its base: `bool` is strictly more specific than `int`.
    ("bool vs int", bool, int),
]


@pytest.mark.parametrize(
    "name, first_hint, second_hint",
    NO_WARN_CASES,
    ids=[case[0] for case in NO_WARN_CASES],
)
def test_ordered_or_disjoint_pairs_do_not_warn(
    name: str,
    first_hint: tx.Any,
    second_hint: tx.Any,
) -> None:
    """A pair with a strict winner (or no overlap) neither warns nor lists."""
    f, first, second = _overloads(first_hint, second_hint)
    caught = _register_quietly(f, first, second)
    assert caught == []
    assert f.ambiguities() == []


def test_two_offending_parameters_are_both_named() -> None:
    """When two positions clash, the message names both parameters."""
    f = Function("f")

    def first(x: tx.List[int], y: tx.List[int]) -> str:
        return "first"

    def second(x: tx.List[str], y: tx.List[str]) -> str:
        return "second"

    f.register(first)
    with pytest.warns(RuntimeWarning, match="ambiguous") as caught:
        f.register(second)
    message = str(caught[0].message)
    assert "parameter 'x' and parameter 'y'" in message


def test_identical_overloads_replace_without_an_ambiguity_warning() -> None:
    """Two identically-spelled `List[int]` overloads replace, not clash.

    The second replaces the first (a replacement warning), leaving one method,
    so there is no ambiguity warning and nothing listed.
    """
    f, first, second = _overloads(tx.List[int], tx.List[int])
    caught = _register_quietly(f, first, second)
    ambiguity = [w for w in caught if "ambiguous" in str(w.message)]
    assert ambiguity == []
    assert f.ambiguities() == []


# --- the helper directly -----------------------------------------------


def test_value_overlap_reads_shallow_parametrisations() -> None:
    """`_value_overlap` is True for two parametrisations of one origin."""
    assert _value_overlap(tx.List[int], tx.List[str]) is True
    assert _value_overlap(tx.List[int], tx.List[bool]) is True
    assert _value_overlap(Box[int], Box[str]) is True
    # Comparable either way is overlap too.
    assert _value_overlap(tx.Sequence[bool], tx.Sequence[int]) is True
    assert _value_overlap(bool, int) is True


def test_value_overlap_leaves_non_generics_to_the_comparable_rule() -> None:
    """Unions, literals, bare TypeVars, Any, tuples and callables stay narrow.

    None of these overlap when incomparable: only the comparable-either-way
    reading applies, so the shallow branch never fires for any of them.
    """
    assert _value_overlap(int, str) is False
    assert _value_overlap(tx.Union[int, str], tx.Union[str, bytes]) is False
    assert _value_overlap(tx.Literal[1], tx.Literal[2]) is False
    # Bare TypeVars carry no parametrised origin: bounded to disjoint types,
    # they are incomparable and do not fall through the shallow branch.
    tv_int = tx.TypeVar("tv_int", bound=int)
    tv_str = tx.TypeVar("tv_str", bound=str)
    assert _value_overlap(tv_int, tv_str) is False
    # A tuple and a callable keep their own dedicated paths, not this branch.
    assert _value_overlap(tx.Tuple[int, str], tx.Tuple[str, int]) is False
    assert (
        _value_overlap(tx.Callable[[int], str], tx.Callable[[str], int])
        is False
    )
