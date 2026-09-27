"""Relation-level tests for spec-defined variance in `issubhint` (V3 of #50).

Variance is a property of a generic's parameter *position*, read from the
super-hint's origin: a user generic reads its declared `TypeVar`, a stdlib
generic the vendored spec table. The per-slot rule (`A` sub-side arg, `B`
super-side arg) is covariant `A <= B`, contravariant `B <= A`, invariant
`A == B` (with `Any` / a free `TypeVar` a top an invariant slot may widen to).
Value applicability stays shallow -- any list still matches every `List[...]`.

Every hint is spelled through `typing_extensions`, and every row runs on 3.8
(the docstring / oldest-supported interpreter), so no `X | Y` or `list[int]`.
"""

# stdlib
import warnings

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers._errors import AmbiguousMethodError
from bagof.dispatchers._function import Function
from bagof.dispatchers.core import ishintstance, issubhint

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


# --- the truth table (#50) ---------------------------------------------

TRUTH_TABLE = [
    # stdlib `list` is invariant: a subtype argument is not a sub-hint.
    (tx.List[bool], tx.List[int], False),
    (tx.List[int], tx.List[int], True),
    (tx.List[int], tx.List[bool], False),
    # stdlib `Sequence` is covariant.
    (tx.Sequence[bool], tx.Sequence[int], True),
    (tx.Sequence[int], tx.Sequence[bool], False),
    (tx.Sequence[int], tx.Sequence[int], True),
    # `Mapping`: key invariant, value covariant.
    (tx.Mapping[str, bool], tx.Mapping[str, int], True),
    (tx.Mapping[str, int], tx.Mapping[str, bool], False),
    (tx.Mapping[bool, int], tx.Mapping[int, int], False),
    (tx.Mapping[int, int], tx.Mapping[int, int], True),
    # a user invariant generic (`Box`): equality only.
    (Box[bool], Box[int], False),
    (Box[int], Box[bool], False),
    (Box[int], Box[int], True),
    # ... but a free `T` / `Any` on the super side is a top it may widen to.
    (Box[int], Box[_T], True),
    (Box[int], Box[tx.Any], True),
    (Box[_T], Box[int], False),
    (Box[tx.Any], Box[int], False),
    # a user covariant generic (`Src`).
    (Src[bool], Src[int], True),
    (Src[int], Src[bool], False),
    # a user contravariant generic (`Snk`).
    (Snk[int], Snk[bool], True),
    (Snk[bool], Snk[int], False),
    (Snk[int], Snk[int], True),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    TRUTH_TABLE,
    ids=[f"{h}<:{s}" for h, s, _ in TRUTH_TABLE],
)
def test_variance_truth_table(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


# --- the ends of each family: Never, Any, free `T` ---------------------

ENDS_TABLE = [
    # Covariant: `Never` is the bottom argument, `Any`/free `T` the top.
    (Src[tx.Never], Src[int], True),
    (Src[int], Src[tx.Never], False),
    (Src[int], Src[tx.Any], True),
    (Src[int], Src[_T], True),
    # Contravariant flips the ends: the `Never` argument is the *top*, and a
    # free `T` / `Any` argument the *bottom* (the gradual-consistency top lands
    # on the reversed side, since a contravariant slot asks `B <= A`).
    (Snk[int], Snk[tx.Never], True),
    (Snk[tx.Never], Snk[int], False),
    (Snk[_T], Snk[int], True),
    (Snk[tx.Any], Snk[int], True),
    (Snk[int], Snk[_T], False),
    (Snk[int], Snk[tx.Any], False),
    # Invariant: only equality or a top on the super side.
    (Box[tx.Never], Box[int], False),
    (Box[int], Box[tx.Never], False),
    (Box[tx.Never], Box[tx.Never], True),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    ENDS_TABLE,
    ids=[f"{h}<:{s}" for h, s, _ in ENDS_TABLE],
)
def test_variance_family_ends(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


# --- nested composition: the signs multiply ----------------------------

NESTED_TABLE = [
    # cov o cov = cov.
    (Src[Src[bool]], Src[Src[int]], True),
    (Src[Src[int]], Src[Src[bool]], False),
    # contra o contra = cov (bool propagates as if covariant).
    (Snk[Snk[bool]], Snk[Snk[int]], True),
    (Snk[Snk[int]], Snk[Snk[bool]], False),
    # contra o cov = contra.
    (Snk[Src[int]], Snk[Src[bool]], True),
    (Snk[Src[bool]], Snk[Src[int]], False),
    # cov o contra = contra.
    (Src[Snk[int]], Src[Snk[bool]], True),
    (Src[Snk[bool]], Src[Snk[int]], False),
    # invariant of an invariant stays equality all the way down.
    (Box[tx.List[bool]], Box[tx.List[int]], False),
    (Box[tx.List[int]], Box[tx.List[int]], True),
    # a covariant container of an invariant one composes the two rules.
    (Src[tx.List[int]], Src[tx.List[int]], True),
    (Src[tx.List[bool]], Src[tx.List[int]], False),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    NESTED_TABLE,
    ids=[f"{h}<:{s}" for h, s, _ in NESTED_TABLE],
)
def test_variance_nested_composition(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


# --- value dispatch: the flipped winner --------------------------------


def test_contravariant_overloads_flip_the_winner() -> None:
    """Two `Snk[...]` overloads pick `Snk[int]` -- contravariance reverses it.

    A plain `Snk()` value matches both overloads shallowly. `Snk` is
    contravariant, so `Snk[int] <= Snk[bool]` and `Snk[int]` is the more
    specific hint -- the opposite of the covariant reading, where `Snk[bool]`
    would have won.
    """
    f = Function("consume")

    def wants_int(x: Snk[int]) -> str:
        return "int"

    def wants_bool(x: Snk[bool]) -> str:
        return "bool"

    f.register(wants_int)
    f.register(wants_bool)
    assert f(Snk()) == "int"


def test_invariant_overloads_are_ambiguous() -> None:
    """Two `List[...]` overloads with subtype-related args are ambiguous.

    `list` is invariant, so `List[int]` and `List[bool]` are incomparable: a
    list value matches both and neither is more specific. Under the old
    covariant reading `List[bool]` would have won outright.
    """
    f = Function("handle")

    def wants_ints(x: tx.List[int]) -> str:
        return "ints"

    def wants_bools(x: tx.List[bool]) -> str:
        return "bools"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f.register(wants_ints)
        f.register(wants_bools)
    with pytest.raises(AmbiguousMethodError):
        f([True, False])


# --- value applicability stays shallow ---------------------------------


def test_value_check_is_shallow_regardless_of_variance() -> None:
    """A value carries no type arguments, so any container matches any arg.

    Variance changes hint *ordering*, never value applicability -- the shallow
    value check is unchanged (#50).
    """
    # An invariant container: a bool list is still a `List[int]` at the value
    # level, and a plain list is still a `List[str]`.
    assert ishintstance([True], tx.List[int]) is True
    assert ishintstance([1, 2], tx.List[str]) is True
    # A user generic value matches any parameterisation, co/contra/invariant.
    assert ishintstance(Snk(), Snk[int]) is True
    assert ishintstance(Snk(), Snk[bool]) is True
    assert ishintstance(Src(), Src[int]) is True
    assert ishintstance(Box(), Box[int]) is True
