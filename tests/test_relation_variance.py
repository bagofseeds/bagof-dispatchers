"""Relation-level tests for spec-defined variance in `issubhint` (V3 of #50).

Variance is a property of a generic's parameter *position*, read from the
super-hint's origin: a user generic reads its declared `TypeVar`, a stdlib
generic the vendored spec table. The per-slot rule (`A` sub-side arg, `B`
super-side arg) is covariant `A <= B`, contravariant `B <= A`, invariant
`A == B` (with `Any` / a free `TypeVar` a top an invariant slot may widen to).
A value that declares no type arguments is matched shallowly -- a plain list
still matches every `List[...]` (what a value *does* declare is read too; see
`test_declared_parametrisation.py`).

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
_TB = tx.TypeVar("_TB", bound=int)
_TBO = tx.TypeVar("_TBO", bound=object)
_TC = tx.TypeVar("_TC", int, str)
_TC2 = tx.TypeVar("_TC2", int, str)
_TC3 = tx.TypeVar("_TC3", int, str, bytes)
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
    # A *bounded* `TypeVar` on the super side is solved, as a type checker
    # does (#50, V5 -- owner decision): `T <= int` can stand for `bool`, so
    # `Box[bool] <= Box[TB]`. V3 read it as exactly its bound and answered
    # False here; a `Box[TB]` fallback now sits above its specialisations.
    (Box[bool], Box[_TB], True),
    (Box[int], Box[_TB], True),
    (Box[str], Box[_TB], False),
    # On the sub side it stands for a whole family, which no single type
    # contains: V3 had `Box[TB]` and `Box[int]` equivalent; now `Box[int]` is
    # strictly below `Box[TB]` (#50, V5).
    (Box[_TB], Box[int], False),
    (Box[_TB], Box[_TB], True),
    (Box[_TB], Box[_TBO], True),
    (Box[_TBO], Box[_TB], False),
    (Box[_TB], Box[_T], True),
    # A constrained `TypeVar` is solved to one of its constraints.
    (Box[int], Box[_TC], True),
    (Box[str], Box[_TC], True),
    (Box[bool], Box[_TC], False),
    (Box[tx.Union[int, str]], Box[_TC], False),
    (Box[_TC], Box[_TC], True),
    # Same constraints: the same family, both ways; more constraints: wider.
    (Box[_TC2], Box[_TC], True),
    (Box[_TC], Box[_TC2], True),
    (Box[_TC], Box[_TC3], True),
    (Box[_TC3], Box[_TC], False),
    (Box[_TC], Box[int], False),
    (Box[_TC], Box[_TB], False),
    (Box[_TB], Box[_TC], False),
    # Covariant slots already read a `TypeVar` as its bound, which is what
    # solving it gives.
    (Src[bool], Src[_TB], True),
    (Src[_TB], Src[int], True),
    (Src[_TB], Src[bool], False),
    # Contravariant slots keep the bound reading: solving there would ask
    # whether two types overlap, which is not transitive (#50, V5).
    (Snk[bool], Snk[_TB], False),
    (Snk[object], Snk[_TB], True),
    (Snk[_TB], Snk[bool], True),
    # A `Literal` in an invariant slot is not equal to its value's type.
    (tx.List[tx.Literal[1]], tx.List[int], False),
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


# --- mixed-sign generics leave parameterisations incomparable ----------


def test_mixed_sign_generic_is_incomparable() -> None:
    """A generic with positions of different signs can order neither way.

    `Generator[Y_co, S_contra, R_co]` has a covariant yield, a contravariant
    send and a covariant return. `Any` is the top of a covariant slot but the
    *bottom* of a contravariant one, so `Generator[int, None, None]` and
    `Generator[int, Any, Any]` are incomparable -- neither is a sub-hint of the
    other. Two such overloads are ambiguous, and this is sound: an
    antisymmetric preorder allows incomparable elements.
    """
    a = tx.Generator[int, None, None]
    b = tx.Generator[int, tx.Any, tx.Any]
    assert issubhint(a, b) is False
    assert issubhint(b, a) is False


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
    list value matches both and neither is more specific, so the call raises
    `AmbiguousMethodError`. Under the old covariant reading `List[bool]` would
    have won outright. Registration does *not* warn -- registering both is
    legitimate (a `class Child(List[int])` is dispatched precisely under
    base-parameter substitution); the tie shows only at call time, where a
    `priority` resolves it.
    """
    f = Function("handle")

    def wants_ints(x: tx.List[int]) -> str:
        return "ints"

    def wants_bools(x: tx.List[bool]) -> str:
        return "bools"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f.register(wants_ints)
        f.register(wants_bools)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    with pytest.raises(AmbiguousMethodError):
        f([True, False])


# --- value applicability stays shallow ---------------------------------


def test_value_check_is_shallow_regardless_of_variance() -> None:
    """A value declaring no type arguments matches any parametrisation.

    Variance changes hint *ordering*; a plain container, or a user generic
    built without arguments, is still matched by its class alone (#50).
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
