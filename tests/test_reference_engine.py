"""A reduce-to-positional reference-engine property harness.

RFC 0001 §2.2 states a *reduce-to-positional guarantee*: for a call with no
keywords and methods without ``*args``, name-aware dispatch coincides exactly
with a plain positional engine. This harness makes that a property test without
Hypothesis: a small deterministic generator produces method sets and calls over
a fixed hint alphabet, an independent positional reference engine computes the
expected winner (or error), and the real
[`Function`][bagof.dispatchers._function.Function] is asserted to agree -- and
to agree regardless of the order the methods were registered in.

The reference engine implements the §2.2/§3 selection directly and
positionally: per-argument applicability with ``ishintstance`` *and*
repeated-``TypeVar`` consistency; the specificity order with per-position
``issubhint``; then the documented tie-breaks (priority, then argument MRO --
with no cross-argument conflict allowed to be broken by MRO -- then tightness,
which is trivial here since every method has the same fixed arity, then the
repeated-``TypeVar`` group-count refinement of RFC 0001 §3). Incomparable
maxima are ambiguous; no applicable method is a no-match.

This catches the B2 cross-argument-conflict bug (an ``Exact[int]/object`` vs
``int/int`` pair that must stay ambiguous): on the pre-fix engine the harness
fails, on the fixed engine it passes. It also exercises the Phase 7 group
tie-break: the generated sweep includes repeated-``TypeVar`` signatures, and
the oracle re-derives their grouping independently.
"""

# stdlib
import itertools
import random
import typing
import warnings

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._errors import AmbiguousMethodError, NoMethodError
from bagof.dispatchers._function import Function
from bagof.dispatchers.core import ishintstance, issubhint, normalise_hint
from bagof.dispatchers.core._exact import exact_target, is_exact

# The fixed hint alphabet (RFC 0001 harness spec).
_ALPHABET = [
    object,
    int,
    bool,
    float,
    str,
    tx.Any,
    Exact[int],
    tx.Literal[1],
]

# Shared `TypeVar`s for the Phase 7 sweep. Reused by identity across positions,
# so a variable named twice in a signature forms a repeated group -- the whole
# point of the tie-break under test. `_TB` is bound to make sure a bounded
# repeated variable is exercised too.
_T = tx.TypeVar("_T")
_U = tx.TypeVar("_U")
_V = tx.TypeVar("_V")
_TB = tx.TypeVar("_TB", bound=int)

# A second alphabet that mixes those `TypeVar`s with concrete classes, for the
# repeated-`TypeVar` sweep. Kept small and class-only (no `Literal`/`Exact`) so
# the grouping, not value-dependence, is what varies.
_TYPEVAR_ALPHABET = [_T, _U, _V, _TB, object, int, bool]

# Concrete values to call with: enough to exercise every hint, including the
# value-dependent ones (`Exact[int]` fires for `1`/`0` but not `True`;
# `Literal[1]` for `1` only).
_VALUES = [object(), 0, 1, True, 2.5, "x"]

_PRIORITIES = [0, 0, 0, 1, 2]

# ('method', tag) | ('ambiguous', None) | ('none', None)
_Outcome = typing.Tuple[str, typing.Any]


class _Spec:
    """One generated method: its positional hints, priority and tag."""

    __slots__ = ("hints", "priority", "tag")

    def __init__(
        self,
        hints: typing.Tuple[typing.Any, ...],
        priority: int,
        tag: int,
    ) -> None:
        self.hints = hints
        self.priority = priority
        self.tag = tag


_Scenario = typing.Tuple[typing.List[_Spec], typing.Tuple[typing.Any, ...]]


# --- the independent positional reference engine -----------------------


def _ref_mro_index(
    hint: typing.Any, value_type: type
) -> typing.Optional[int]:
    """Where `hint`'s class sits in `value_type`'s MRO, or `None`.

    An `Exact[C]` counts as `C`; a plain class is looked up in the MRO;
    anything else (`Any`, a `Literal`) gives no MRO position.
    """
    hint = normalise_hint(hint)
    cls = normalise_hint(exact_target(hint)) if is_exact(hint) else hint
    if not isinstance(cls, type):
        return None
    try:
        return list(value_type.__mro__).index(cls)
    except ValueError:
        return None


def _equivalent(a: typing.Any, b: typing.Any) -> bool:
    return issubhint(a, b) and issubhint(b, a)


def _ref_solve(
    classes: typing.List[typing.Any], typevar: typing.Any
) -> typing.Any:
    """Independently solve one repeated `TypeVar`, or return `None`.

    Greatest-element for an unbound/bound variable, same-constraint for a
    constrained one -- the RFC 0001 §3 rule, re-derived here rather than reused
    from `_lattice`, so the oracle stays an independent check.
    """
    constraints = getattr(typevar, "__constraints__", ())
    if constraints:
        for constraint in constraints:
            if all(issubhint(cls, constraint) for cls in classes):
                return constraint
        return None
    for candidate in classes:
        if all(issubhint(other, candidate) for other in classes):
            return candidate
    return None


def _ref_applies(
    spec: _Spec, call: typing.Tuple[typing.Any, ...]
) -> bool:
    """Applicability: each argument an instance, repeated `TypeVar`s solved."""
    if not all(ishintstance(v, h) for v, h in zip(call, spec.hints)):
        return False
    groups = {}  # type: typing.Dict[int, typing.Tuple[typing.Any, list]]
    for pos, hint in enumerate(spec.hints):
        if isinstance(hint, tx.TypeVar):
            groups.setdefault(id(hint), (hint, []))[1].append(type(call[pos]))
    for typevar, classes in groups.values():
        if _ref_solve(classes, typevar) is None:
            return False
    return True


def _ref_labels(spec: _Spec) -> typing.List[typing.Any]:
    """A per-position label: the `TypeVar` identity, or a unique marker."""
    return [
        id(hint) if isinstance(hint, tx.TypeVar) else ("solo", pos)
        for pos, hint in enumerate(spec.hints)
    ]


def _ref_group_refines(
    more: _Spec, less: _Spec, call: typing.Tuple[typing.Any, ...]
) -> bool:
    """Whether `more`'s repeated `TypeVar`s strictly refine `less`'s (§3).

    Position-wise equivalence, then a strictly coarser grouping: every pair
    `less` ties `more` ties too, and `more` ties at least one pair `less`
    leaves independent.
    """
    for hm, hl in zip(more.hints, less.hints):
        if not _equivalent(hm, hl):
            return False
    lm, ll = _ref_labels(more), _ref_labels(less)
    strict = False
    for i in range(len(call)):
        for j in range(i + 1, len(call)):
            more_same = lm[i] == lm[j]
            less_same = ll[i] == ll[j]
            if less_same and not more_same:
                return False
            if more_same and not less_same:
                strict = True
    return strict


def _reference_select(
    specs: typing.List[_Spec], call: typing.Tuple[typing.Any, ...]
) -> _Outcome:
    """The §2.2/§3 positional selection: ('method', tag) or a marker."""
    applicable = [spec for spec in specs if _ref_applies(spec, call)]
    if not applicable:
        return ("none", None)

    def le(a: _Spec, b: _Spec) -> bool:
        return all(issubhint(ha, hb) for ha, hb in zip(a.hints, b.hints))

    maximal = [
        a
        for a in applicable
        if not any(
            b is not a and le(b, a) and not le(a, b) for b in applicable
        )
    ]
    if len(maximal) == 1:
        return ("method", maximal[0].tag)

    best_priority = max(spec.priority for spec in maximal)
    cand = [spec for spec in maximal if spec.priority == best_priority]
    if len(cand) == 1:
        return ("method", cand[0].tag)

    def dominates(a: _Spec, b: _Spec) -> bool:
        strict = False
        for pos in range(len(call)):
            ha, hb, value = a.hints[pos], b.hints[pos], call[pos]
            # Cross-argument conflict: if `b` is strictly more specific here,
            # MRO may not hand the win to `a` (the B2 rule).
            if issubhint(hb, ha) and not issubhint(ha, hb):
                return False
            ia = _ref_mro_index(ha, type(value))
            ib = _ref_mro_index(hb, type(value))
            if ia is None or ib is None:
                if not _equivalent(ha, hb):
                    return False
                continue
            if ia > ib:
                return False
            if ia < ib:
                strict = True
        return strict

    survivors = [
        a
        for a in cand
        if not any(b is not a and dominates(b, a) for b in cand)
    ]
    if len(survivors) == 1:
        return ("method", survivors[0].tag)
    # Tightness is trivial here (every method is the same fixed arity with no
    # catch-alls or defaults). The last tie-break is the repeated-`TypeVar`
    # group-count refinement (RFC 0001 §3): drop any survivor another groups
    # strictly more of the call's arguments into one consistent type.
    refined = [
        a
        for a in survivors
        if not any(
            b is not a and _ref_group_refines(b, a, call) for b in survivors
        )
    ]
    if len(refined) == 1:
        return ("method", refined[0].tag)
    return ("ambiguous", None)


# --- the real engine, driven the same way ------------------------------


def _make_impl(arity: int, tag: int) -> typing.Callable[..., int]:
    """A function of `arity` positional parameters that returns `tag`."""
    params = ", ".join(f"a{index}" for index in range(arity))
    namespace = {}  # type: typing.Dict[str, typing.Any]
    exec(f"def impl({params}): return {tag!r}", namespace)  # noqa: S102
    return namespace["impl"]


def _build_function(specs: typing.List[_Spec]) -> Function:
    """A `Function` with `specs` registered (overlay form), warnings muted."""
    f = Function("f")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for spec in specs:
            impl = _make_impl(len(spec.hints), spec.tag)
            f.register(spec.hints, priority=spec.priority)(impl)
    return f


def _real_outcome(
    specs: typing.List[_Spec], call: typing.Tuple[typing.Any, ...]
) -> _Outcome:
    """The real engine's outcome as a comparable marker tuple."""
    f = _build_function(specs)
    try:
        return ("method", f(*call))
    except AmbiguousMethodError:
        return ("ambiguous", None)
    except NoMethodError:
        return ("none", None)


# --- the property ------------------------------------------------------


def _scenarios(
    count: int,
    seed: int,
    alphabet: typing.Optional[typing.List[typing.Any]] = None,
) -> typing.Iterator[_Scenario]:
    """Deterministically generate `(specs, call)` scenarios."""
    alphabet = _ALPHABET if alphabet is None else alphabet
    rng = random.Random(seed)
    made = 0
    while made < count:
        arity = rng.randint(1, 3)
        n_methods = rng.randint(2, 4)
        specs = []  # type: typing.List[_Spec]
        seen = set()  # dedupe identical spellings (which would replace)
        for _ in range(n_methods):
            hints = tuple(rng.choice(alphabet) for _ in range(arity))
            if hints in seen:
                continue
            seen.add(hints)
            specs.append(_Spec(hints, rng.choice(_PRIORITIES), len(specs)))
        if len(specs) < 2:
            continue
        call = tuple(rng.choice(_VALUES) for _ in range(arity))
        made += 1
        yield specs, call


def _adversarial_scenarios() -> typing.List[_Scenario]:
    """Hand-picked cross-argument conflicts the random generator hits rarely.

    Each is a genuine ambiguity the pre-fix MRO tie-break resolves wrongly: an
    `Exact[C]` reads as `C` for the MRO, so one position looks tied while the
    method is in fact strictly more specific there, and the competitor's win at
    another position wrongly dominates. These make the property test itself
    (not only the dedicated conflict test) fail on the pre-fix engine.
    """
    return [
        (
            [_Spec((Exact[int], object), 0, 0), _Spec((int, int), 0, 1)],
            (1, 1),
        ),
        (
            [_Spec((object, Exact[int]), 0, 0), _Spec((int, int), 0, 1)],
            (1, 1),
        ),
        (
            [_Spec((Exact[int], object), 0, 0), _Spec((int, bool), 0, 1)],
            (1, True),
        ),
        (
            [
                _Spec((Exact[int], object, object), 0, 0),
                _Spec((int, int, int), 0, 1),
            ],
            (1, 1, 1),
        ),
    ]


def _order_permutations(
    specs: typing.List[_Spec],
) -> typing.Iterator[typing.List[_Spec]]:
    """A few registration orders: reversed and a couple of rotations."""
    yield list(reversed(specs))
    if len(specs) > 2:
        yield specs[1:] + specs[:1]
        yield specs[-1:] + specs[:-1]


def test_reference_engine_agreement_and_order_independence() -> None:
    """The real engine matches the positional reference, order-invariant."""
    checked = 0
    caught_ambiguous = 0
    scenarios = itertools.chain(
        _adversarial_scenarios(), _scenarios(count=900, seed=20240115)
    )
    for specs, call in scenarios:
        expected = _reference_select(specs, call)
        if expected[0] == "ambiguous":
            caught_ambiguous += 1
        # The real engine agrees with the reference...
        assert _real_outcome(specs, call) == expected, (
            [(s.hints, s.priority) for s in specs],
            call,
            expected,
        )
        # ...and does so regardless of registration order.
        for permuted in _order_permutations(specs):
            assert _real_outcome(permuted, call) == expected, (
                [(s.hints, s.priority) for s in permuted],
                call,
            )
        checked += 1
    assert checked >= 500  # the generator actually produced work
    # The alphabet includes cross-argument conflicts, so some scenarios are
    # genuinely ambiguous -- the case B2 must get right.
    assert caught_ambiguous > 0


def test_reference_engine_flags_the_exact_cross_argument_conflict() -> None:
    """The canonical B2 shape is in the reference engine's ambiguous set."""
    specs = [
        _Spec((Exact[int], object), 0, 0),
        _Spec((int, int), 0, 1),
    ]
    assert _reference_select(specs, (1, 1)) == ("ambiguous", None)
    assert _real_outcome(specs, (1, 1)) == ("ambiguous", None)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_reference_engine_small_smoke(seed: int) -> None:
    """A quick independent smoke run at other seeds."""
    for specs, call in itertools.islice(_scenarios(120, seed), 120):
        assert _real_outcome(specs, call) == _reference_select(specs, call)


# --- Phase 7: the repeated-TypeVar sweep -------------------------------


def _typevar_adversarial_scenarios() -> typing.List[_Scenario]:
    """Hand-picked repeated-`TypeVar` shapes the tie-break must get right."""
    return [
        # (T, T) beats (T, U) for a same-type call.
        ([_Spec((_T, _T), 0, 0), _Spec((_T, _U), 0, 1)], (1, 2)),
        # ...but a mixed call excludes (T, T) by consistency.
        ([_Spec((_T, _T), 0, 0), _Spec((_T, _U), 0, 1)], (1, "x")),
        # (T, U) vs (U, T): identical grouping, genuinely ambiguous.
        ([_Spec((_T, _U), 0, 0), _Spec((_U, _T), 0, 1)], (1, 2)),
        # Two groups beat three independents.
        ([_Spec((_T, _T, _U), 0, 0), _Spec((_T, _U, _V), 0, 1)], (1, 2, 3)),
        # Partial refinement: {0,1} vs {1,2}, incomparable, ambiguous.
        ([_Spec((_T, _T, _U), 0, 0), _Spec((_T, _U, _U), 0, 1)], (1, 1, 1)),
        # A repeated bound TypeVar against an independent bound pair.
        ([_Spec((_TB, _TB), 0, 0), _Spec((_TB, _U), 0, 1)], (1, 2)),
        # (T, T) beats a fully unannotated (Any, Any).
        ([_Spec((_T, _T), 0, 0), _Spec((object, object), 0, 1)], (1, 2)),
    ]


def test_reference_engine_typevar_sweep() -> None:
    """The real engine matches the oracle over repeated-`TypeVar` shapes.

    The oracle re-derives grouping, consistency and the §3 tie-break
    independently, so agreement over the generated sweep (plus hand-picked
    shapes) is a genuine cross-check of the Phase 7 selection, order-invariant.
    """
    checked = 0
    caught_ambiguous = 0
    scenarios = itertools.chain(
        _typevar_adversarial_scenarios(),
        _scenarios(count=700, seed=20240720, alphabet=_TYPEVAR_ALPHABET),
    )
    for specs, call in scenarios:
        expected = _reference_select(specs, call)
        if expected[0] == "ambiguous":
            caught_ambiguous += 1
        assert _real_outcome(specs, call) == expected, (
            [s.hints for s in specs],
            call,
            expected,
        )
        for permuted in _order_permutations(specs):
            assert _real_outcome(permuted, call) == expected, (
                [s.hints for s in permuted],
                call,
            )
        checked += 1
    assert checked >= 400
    # Independent groupings (`(T, U)`/`(U, T)`) and partial refinements keep
    # some scenarios genuinely ambiguous even with the tie-break in place.
    assert caught_ambiguous > 0
