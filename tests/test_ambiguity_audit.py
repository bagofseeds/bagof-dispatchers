"""`Function.ambiguities()` as a dominance-aware audit.

Registration never warns about ambiguity. `ambiguities()` reports a pair only
when some call matches both methods and the full selection -- every other
registered method, priority and tightness included -- still leaves the two
tied, so a third method that wins every call the pair shares clears it in any
registration order. Each test checks the audit against what a call actually
does.

Every hint is spelled through `typing` and runs on 3.8.
"""

# stdlib
import itertools
import numbers
import random
import typing
import warnings

# dependencies
import pytest

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._errors import AmbiguousMethodError, NoMethodError
from bagof.dispatchers._function import Function


def _rank(a: float, b: object) -> str:
    return "rank"


def _order(a: object, b: float) -> str:
    return "order"


def _both(a: float, b: float) -> str:
    return "both"


def _names(f: Function) -> typing.List[typing.Tuple[str, str]]:
    return [(a.name, b.name) for a, b in f.ambiguities()]


@pytest.mark.parametrize(
    "order", list(itertools.permutations([_rank, _order, _both]))
)
def test_a_dominating_third_method_clears_the_pair(
    order: typing.Tuple[typing.Any, ...],
) -> None:
    """`(float, float)` wins every call `rank` and `order` share."""
    f = Function("f")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # registration never warns
        for fn in order:
            f.register(fn)
    assert f.ambiguities() == []
    assert f(1.0, 1.0) == "both"


def test_without_the_third_method_the_pair_is_reported() -> None:
    f = Function("f")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f.register(_rank)
        f.register(_order)
    assert _names(f) == [("_rank", "_order")]
    with pytest.raises(AmbiguousMethodError):
        f(1.0, 1.0)


def test_a_lower_priority_third_method_still_clears_the_pair() -> None:
    """Specificity is decided before priority, at dispatch and in the audit.

    `both` is strictly more specific than `rank` and `order`, so it wins the
    calls they share even at a lower priority than theirs.
    """
    f = Function("f")
    f.register(_rank)
    f.register(_order)
    f.register(_both, priority=-1)
    assert f.ambiguities() == []
    assert f(1.0, 1.0) == "both"


def test_a_higher_priority_but_wider_third_method_does_not_clear_it() -> None:
    """Priority only chooses among the most specific methods."""
    f = Function("f")
    f.register(_rank)
    f.register(_order)

    @f.register((object, object), priority=5)
    def _anything(a: object, b: object) -> str:
        return "anything"

    assert _names(f) == [("_rank", "_order")]
    with pytest.raises(AmbiguousMethodError):
        f(1.0, 1.0)


def test_the_pair_that_actually_ties_is_the_one_reported() -> None:
    """A third method can beat one of the pair and tie with the other.

    `(Real, float)` is strictly more specific than `order` but incomparable
    with `rank`. On two floats it removes `order` and ties with `rank`, so the
    audit reports `(rank, real)` rather than `(rank, order)`, as a call does.
    """
    f = Function("f")
    f.register(_rank)
    f.register(_order)

    @f.register((numbers.Real, float))
    def _real(a: object, b: object) -> str:
        return "real"

    assert _names(f) == [("_rank", "_real")]
    with pytest.raises(AmbiguousMethodError) as info:
        f(1.0, 1.0)
    assert [m.name for m in info.value.candidates] == ["_rank", "_real"]


def test_a_priority_difference_within_the_pair_clears_it() -> None:
    f = Function("f")
    f.register(_rank)
    f.register(_order, priority=1)
    assert f.ambiguities() == []
    assert f(1.0, 1.0) == "order"


def test_a_third_method_clears_a_keyword_only_pair() -> None:
    f = Function("f")

    def by_x(x: float, *, mode: object) -> str:
        return "x"

    def by_mode(x: object, *, mode: float) -> str:
        return "mode"

    f.register(by_x)
    f.register(by_mode)
    assert _names(f) == [("by_x", "by_mode")]

    @f.register
    def by_both(x: float, *, mode: float) -> str:
        return "both"

    assert f.ambiguities() == []
    assert f(1.0, mode=2.0) == "both"


def test_an_exact_third_method_does_not_clear_the_pair() -> None:
    """`(Exact[int], Exact[int])` wins only the calls of exactly `int`.

    A subclass of `int` still reaches both `(int, object)` and
    `(object, int)` with nothing to choose between them, so the pair is kept.
    """

    class MyInt(int):
        pass

    f = Function("f")
    f.register((int, object))(lambda a, b: "first")
    f.register((object, int))(lambda a, b: "second")
    f.register((Exact[int], Exact[int]))(lambda a, b: "exact")
    assert len(f.ambiguities()) == 1
    assert f(1, 2) == "exact"
    with pytest.raises(AmbiguousMethodError):
        f(MyInt(1), MyInt(2))


def test_an_unresolved_third_method_does_not_raise() -> None:
    """A method whose hints are still names does not stop the audit.

    It cannot be weighed against the pair yet, so the pair is kept.
    """
    f = Function("f")
    f.register(_rank)
    f.register(_order)

    def later(a: "NotDefinedYet", b: "NotDefinedYet") -> str:  # noqa: F821
        return "later"

    f.register(later)
    assert _names(f) == [("_rank", "_order")]


def test_a_union_covered_only_jointly_is_still_reported() -> None:
    """The known limit: no single method wins the whole overlap.

    `rank` and `order` share every call of two `int`-or-`str` arguments, and
    four narrower methods win those calls between them. None of the four wins
    all of them alone, so the pair is still reported, although every actual
    call dispatches.
    """
    either = typing.Union[int, str]
    f = Function("f")
    f.register((either, object))(lambda a, b: "rank")
    f.register((object, either))(lambda a, b: "order")
    for first, second in itertools.product((int, str), repeat=2):
        f.register((first, second))(lambda a, b: "narrow")
    assert len(f.ambiguities()) == 1
    for call in itertools.product((1, "a"), repeat=2):
        assert f(*call) == "narrow"


class _Left:
    pass


class _Right:
    pass


class _Both(_Left, _Right):
    pass


def test_a_pair_whose_overlap_one_method_rejects_is_kept() -> None:
    """The overlap is not always a query both methods accept.

    Argument by argument, `(T, T, float)` and `(_Left, _Right, object)` share
    `(_Left, _Right, float)`, which the repeated `T` rejects as a query. A
    subclass of both classes still reaches both methods, so the pair is
    reported rather than cleared.
    """
    T = typing.TypeVar("T")
    f = Function("f")

    @f.register((T, T, float))
    def same(x: object, y: object, z: object) -> str:
        return "same"

    @f.register((_Left, _Right, object))
    def split(x: object, y: object, z: object) -> str:
        return "split"

    assert _names(f) == [("same", "split")]
    assert f(_Left(), _Right(), 1.0) == "split"
    with pytest.raises(AmbiguousMethodError):
        f(_Both(), _Both(), 1.0)


def test_methods_of_different_arities_are_not_a_pair() -> None:
    f = Function("f")
    f.register((float,))(lambda a: "one")
    f.register((object, float))(lambda a, b: "two")
    assert f.ambiguities() == []


def test_a_tighter_method_is_not_ambiguous_with_a_looser_one() -> None:
    """A fixed-arity method beats one that absorbs an argument into `*args`."""
    f = Function("f")

    def fixed(a: float, b: object) -> str:
        return "fixed"

    def spread(a: object, *rest: float) -> str:
        return "spread"

    f.register(fixed)
    f.register(spread)
    assert f.ambiguities() == []
    assert f(1.0, 1.0) == "fixed"


# --- agreement with dispatch over a small class tree -------------------


class _Base:
    pass


class _Mid(_Base):
    pass


class _Leaf(_Mid):
    pass


class _Other(_Base):
    pass


_TREE = [object, _Base, _Mid, _Leaf, _Other]


def _returning(tag: int) -> typing.Callable[[object, object], int]:
    return lambda a, b: tag


def _tree_function(rng: random.Random) -> Function:
    """Register two to five distinct two-argument methods over `_TREE`."""
    f = Function("f")
    seen: typing.Set[typing.Tuple[typing.Any, ...]] = set()
    for tag in range(rng.randint(2, 5)):
        hints = (rng.choice(_TREE), rng.choice(_TREE))
        if hints in seen:
            continue
        seen.add(hints)
        f.register(hints, priority=rng.choice([0, 0, 0, 1]))(_returning(tag))
    return f


@pytest.mark.parametrize("seed", range(40))
def test_the_audit_agrees_with_every_call(seed: int) -> None:
    """A pair is reported exactly when some call finds the two tied.

    In a class tree, hints that share a value are always ordered, and every
    class in the tree has an instance to call with, so the audit is exact:
    it reports a pair if and only if one of those calls raises with both
    methods among the tied candidates.
    """
    rng = random.Random(seed)
    for _ in range(10):
        f = _tree_function(rng)
        tied: typing.Set[typing.Tuple[int, int]] = set()
        for call in itertools.product([cls() for cls in _TREE], repeat=2):
            try:
                f(*call)
            except NoMethodError:
                pass
            except AmbiguousMethodError as error:
                ids = [id(method) for method in error.candidates]
                tied.update(itertools.combinations(sorted(ids), 2))
        reported = {
            tuple(sorted((id(first), id(second))))
            for first, second in f.ambiguities()
        }
        assert reported == tied
