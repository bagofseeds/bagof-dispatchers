"""Tests for `Function.candidates` and its relatives.

The six methods report what selection sees for a call without acting on
it: `candidates` / `itercandidates` list every applicable method, most
specific first, and `bestcandidates` / `iterbestcandidates` the methods
left standing after every tie-break. `resolve_candidates` and
`resolve_bestcandidates` do the same for a call described by hints.

Most of the checks below are invariants tying these methods to
`dispatch` and `resolve`, swept over a small corpus of functions and
calls, so that a divergence between the report and the real selection
shows up whatever shape it takes.
"""

# stdlib
import collections.abc
import itertools
import threading
import typing
import warnings

# dependencies
import pytest

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._errors import AmbiguousMethodError, NoMethodError
from bagof.dispatchers._function import Function
from bagof.dispatchers._signature import Signature

T = typing.TypeVar("T")
U = typing.TypeVar("U")


class Animal:
    pass


class Dog(Animal):
    pass


class Puppy(Dog):
    pass


class Cat(Animal):
    pass


class Pet:
    pass


class PetDog(Dog, Pet):
    """A class whose MRO puts `Dog` ahead of `Pet`."""


class Sized:
    """A class that is a `collections.abc.Sized` only structurally."""

    def __len__(self) -> int:
        return 0


def _quiet(fn: typing.Callable[[], typing.Any]) -> typing.Any:
    """Run `fn`, ignoring the registration-time ambiguity warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn()


def _add(
    f: Function,
    name: str,
    hints: typing.Tuple[typing.Any, ...],
    priority: int = 0,
    named: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> None:
    """Register a method with these explicit hints under `name`.

    The positional parameters are named after the method, so that two
    methods with the same hints are two methods rather than one replacing
    the other.
    """
    params = [f"{name}_{i}" for i in range(len(hints))] + sorted(named or {})
    namespace: typing.Dict[str, typing.Any] = {}
    exec(f"def {name}({', '.join(params)}): return {name!r}", namespace)
    spec: typing.Tuple[typing.Any, ...] = (hints, named) if named else (hints,)
    _quiet(lambda: f.register(*spec, priority=priority)(namespace[name]))


# --- the corpus ------------------------------------------------------


def _hierarchy() -> Function:
    f = Function("hierarchy")
    _add(f, "anything", (object,))
    _add(f, "animal", (Animal,))
    _add(f, "dog", (Dog,))
    _add(f, "puppy", (Puppy,))
    _add(f, "cat", (Cat,))
    _add(f, "loud_animal", (Animal,), priority=2)
    return f


def _pairs() -> Function:
    f = Function("pairs")
    _add(f, "any_any", (object, object))
    _add(f, "dog_any", (Dog, object))
    _add(f, "any_dog", (object, Dog))
    _add(f, "animal_animal", (Animal, Animal))
    _add(f, "puppy_any", (Puppy, object), priority=1)
    _add(f, "cat_cat", (Cat, Cat), priority=-1)
    return f


def _priority_tie() -> Function:
    # The reproducer for the candidates of `AmbiguousMethodError`: `sized`
    # is as specific as `left` and `right`, but loses on priority.
    f = Function("priority_tie")
    _add(f, "sized", (collections.abc.Sized, collections.abc.Sized))
    _add(f, "left", (Sized, object), priority=1)
    _add(f, "right", (object, Sized), priority=1)
    return f


def _mro() -> Function:
    f = Function("mro")
    _add(f, "as_dog", (Dog,))
    _add(f, "as_pet", (Pet,))
    _add(f, "as_object", (object,))
    return f


def _typevars() -> Function:
    f = Function("typevars")
    _add(f, "same", (T, T))
    _add(f, "different", (T, U))
    _add(f, "dogs", (Dog, Dog), priority=-1)
    return f


def _tightness() -> Function:
    f = Function("tightness")

    def fixed(x: int) -> str:
        return "fixed"

    def spread(x: int, *rest: int) -> str:
        return "spread"

    def defaulted(x: int, y: int = 0) -> str:
        return "defaulted"

    for fn in (fixed, spread, defaulted):
        _quiet(lambda fn=fn: f.register(fn))
    return f


def _exact() -> Function:
    f = Function("exact")
    _add(f, "exactly_dog", (Exact[Dog],))
    _add(f, "any_dog", (Dog,))
    _add(f, "exact_then_any", (Exact[Dog], object))
    _add(f, "dog_then_dog", (Dog, Dog))
    return f


def _keywords() -> Function:
    f = Function("keywords")
    _add(f, "kw_animal", (), named={"who": Animal})
    _add(f, "kw_dog", (), named={"who": Dog})
    _add(f, "pos_kw", (Dog,), named={"other": object})
    _add(f, "pos_kw_cat", (object,), named={"other": Cat})
    return f


def _empty() -> Function:
    return Function("empty")


CORPUS = [
    _hierarchy,
    _pairs,
    _priority_tie,
    _mro,
    _typevars,
    _tightness,
    _exact,
    _keywords,
    _empty,
]

VALUES = [
    object(),
    Animal(),
    Dog(),
    Puppy(),
    Cat(),
    PetDog(),
    Sized(),
    1,
    "s",
]

HINTS = [object, Animal, Dog, Puppy, Cat, PetDog, Sized, int, Exact[Dog]]


def _calls(
    pool: typing.Sequence[typing.Any],
) -> typing.Iterator[
    typing.Tuple[typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]]
]:
    """Every call of zero to two positionals, plus a few keyword ones."""
    for count in range(3):
        for args in itertools.product(pool, repeat=count):
            yield args, {}
    for value in pool:
        yield (), {"who": value}
        for first in pool[:4]:
            yield (first,), {"other": value}


def _outcome(
    select: typing.Callable[[], typing.Any],
) -> typing.Tuple[str, typing.Any]:
    """Run a selection and report how it ended."""
    try:
        return "ok", select()
    except AmbiguousMethodError as error:
        return "ambiguous", error
    except NoMethodError as error:
        return "none", error


def _strictly_more_specific(
    first: typing.Any, second: typing.Any, shape: typing.Any
) -> bool:
    return first.signature.le(second.signature, shape) and not (
        second.signature.le(first.signature, shape)
    )


def _expected_order(
    applicable: typing.Sequence[typing.Any],
    best: typing.Sequence[typing.Any],
    shape: typing.Any,
    registered: typing.Sequence[typing.Any],
) -> typing.List[typing.Any]:
    """The documented order, worked out from the public pieces alone."""

    def maximal(pool: typing.List[typing.Any]) -> typing.List[typing.Any]:
        return [
            m
            for m in pool
            if not any(
                _strictly_more_specific(o, m, shape)
                for o in pool
                if o is not m
            )
        ]

    def rank(m: typing.Any) -> typing.Tuple[int, int]:
        return (-m.priority, registered.index(m))

    order = list(best)
    remaining = list(applicable)
    while remaining:
        layer = maximal(remaining)
        order += [m for m in sorted(layer, key=rank) if m not in best]
        remaining = [m for m in remaining if m not in layer]
    return order


def _check(
    f: Function,
    args: typing.Tuple[typing.Any, ...],
    kwargs: typing.Dict[str, typing.Any],
    hints: bool,
) -> None:
    """Check every invariant for one call (or one hint query)."""
    shape = Signature.shape(args, kwargs)
    if hints:
        candidates = f.resolve_candidates(*args, **kwargs)
        best = f.resolve_bestcandidates(*args, **kwargs)
        kind, result = _outcome(lambda: f.resolve(*args, **kwargs))
        applies = [
            m
            for m in f.methods
            if m.signature.applies_to_hints(args, kwargs)
        ]
    else:
        candidates = f.candidates(*args, **kwargs)
        best = f.bestcandidates(*args, **kwargs)
        kind, result = _outcome(lambda: f.dispatch(*args, **kwargs))
        applies = [
            m
            for m in f.methods
            if m.signature.applies_to_values(args, kwargs)
        ]
        # 8. The iterators yield exactly the tuples' sequences.
        assert tuple(f.itercandidates(*args, **kwargs)) == candidates
        assert tuple(f.iterbestcandidates(*args, **kwargs)) == best
    assert isinstance(candidates, tuple) and isinstance(best, tuple)
    # 1. Exactly the applicable methods, each once.
    assert len(set(map(id, candidates))) == len(candidates)
    assert {id(m) for m in candidates} == {id(m) for m in applies}
    # 2. Nothing precedes a method strictly more specific than itself.
    for i, earlier in enumerate(candidates):
        for later in candidates[i + 1 :]:
            assert not _strictly_more_specific(later, earlier, shape)
    assert list(candidates) == _expected_order(
        applies, best, shape, f.methods
    )
    # 3. The best candidates are a prefix of the candidates.
    assert candidates[: len(best)] == best
    # 4. One best candidate exactly when selection succeeds, and it is the
    # method selected.
    assert (len(best) == 1) == (kind == "ok")
    if kind == "ok":
        assert best[0] is result
    # 5. An ambiguity reports exactly the best candidates, in order.
    if kind == "ambiguous":
        assert result.candidates == best
        assert len(best) > 1
    # 6. No candidates exactly when no method applies.
    assert (candidates == ()) == (kind == "none")


@pytest.mark.parametrize("build", CORPUS, ids=lambda b: b.__name__)
def test_value_invariants_over_the_corpus(build: typing.Any) -> None:
    """Invariants 1-6 and 8 hold for every call in the corpus."""
    f = build()
    for args, kwargs in _calls(VALUES):
        _check(f, args, kwargs, hints=False)


@pytest.mark.parametrize("build", CORPUS, ids=lambda b: b.__name__)
def test_hint_invariants_over_the_corpus(build: typing.Any) -> None:
    """Invariant 7: the hint twins agree with `resolve` the same way."""
    f = build()
    for args, kwargs in _calls(HINTS):
        _check(f, args, kwargs, hints=True)


def test_the_corpus_reaches_every_outcome() -> None:
    """The sweep sees successes, ties and misses at both levels."""
    seen = set()
    for build in CORPUS:
        f = build()
        for args, kwargs in _calls(VALUES):
            seen.add(("value", len(f.bestcandidates(*args, **kwargs)) > 1))
            seen.add(("value", len(f.candidates(*args, **kwargs))))
        for args, kwargs in _calls(HINTS):
            seen.add(("hint", len(f.resolve_bestcandidates(*args, **kwargs))))
    assert {("value", True), ("value", False), ("value", 0)} <= seen
    assert {("hint", 0), ("hint", 1), ("hint", 2)} <= seen


# --- specific orders -------------------------------------------------


def _names(methods: typing.Iterable[typing.Any]) -> typing.List[str]:
    return [m.name for m in methods]


def test_candidates_are_ordered_by_specificity() -> None:
    f = _hierarchy()
    assert _names(f.candidates(Puppy())) == [
        "puppy",
        "dog",
        "loud_animal",
        "animal",
        "anything",
    ]
    assert _names(f.bestcandidates(Puppy())) == ["puppy"]


def test_within_a_layer_priority_then_registration_order() -> None:
    """`loud_animal` outranks `animal`, although registered after it."""
    f = _hierarchy()
    assert _names(f.candidates(Animal())) == [
        "loud_animal",
        "animal",
        "anything",
    ]
    assert _names(f.bestcandidates(Animal())) == ["loud_animal"]


def test_the_best_candidate_leads_whatever_its_registration() -> None:
    """A tie-break winner comes first even when registered after the loser.

    `as_pet` and `as_dog` are equally specific for a `PetDog` and share a
    priority, so registration order alone would put `as_pet` first; the
    MRO tie-break picks `as_dog`, and the order follows the selection.
    """
    f = Function("late_winner")
    _add(f, "as_pet", (Pet,))
    _add(f, "as_dog", (Dog,))
    assert _names(f.candidates(PetDog())) == ["as_dog", "as_pet"]


def test_layers_follow_strict_specificity() -> None:
    """A method beaten only by a lower layer's rival is not promoted.

    For `(Puppy, Dog)`, `puppy_any` wins on priority over the other two
    methods of the first layer. `dog_any` is incomparable with those two,
    but `puppy_any` is strictly more specific than it, so it waits for the
    second layer.
    """
    f = _pairs()
    assert _names(f.candidates(Puppy(), Dog())) == [
        "puppy_any",
        "any_dog",
        "animal_animal",
        "dog_any",
        "any_any",
    ]
    assert _names(f.bestcandidates(Dog(), Dog())) == [
        "dog_any",
        "any_dog",
        "animal_animal",
    ]
    # `cat_cat` is strictly more specific, so its low priority is moot.
    assert _names(f.candidates(Cat(), Cat())) == [
        "cat_cat",
        "animal_animal",
        "any_any",
    ]


def test_the_mro_tie_break_is_reflected() -> None:
    """`PetDog` reaches `Dog` first in its MRO, so `as_dog` is best."""
    f = _mro()
    assert _names(f.bestcandidates(PetDog())) == ["as_dog"]
    assert _names(f.candidates(PetDog())) == ["as_dog", "as_pet", "as_object"]
    # A hint query has no MRO to consult, so the two stay tied.
    assert _names(f.resolve_bestcandidates(PetDog)) == ["as_dog", "as_pet"]


def test_the_typevar_refinement_is_reflected() -> None:
    """`same` refines `different`, but `dogs` beats both on specificity."""
    f = _typevars()
    assert _names(f.bestcandidates(1, 1)) == ["same"]
    assert _names(f.candidates(1, 1)) == ["same", "different"]
    assert _names(f.bestcandidates(1, "s")) == ["different"]
    assert _names(f.candidates(Dog(), Dog())) == ["dogs", "same", "different"]


def test_tightness_is_reflected() -> None:
    f = _tightness()
    assert _names(f.bestcandidates(1)) == ["fixed"]
    assert set(_names(f.candidates(1))) == {"fixed", "spread", "defaulted"}
    assert _names(f.candidates(1, 2, 3)) == ["spread"]


def test_nothing_applicable_gives_empty_results() -> None:
    f = _hierarchy()
    assert f.candidates() == ()
    assert f.bestcandidates(1, 2) == ()
    assert list(f.itercandidates(x=1)) == []
    assert list(f.iterbestcandidates()) == []
    assert f.resolve_candidates(int, int) == ()
    assert f.resolve_bestcandidates() == ()
    assert Function("empty").candidates(1) == ()


def test_resolve_twins_honour_the_exact_convenience() -> None:
    """A plain `C` query reaches a method whose parameter is `Exact[C]`."""
    f = _exact()
    assert f.resolve(Dog).name == "exactly_dog"
    assert _names(f.resolve_bestcandidates(Dog)) == ["exactly_dog"]
    assert _names(f.resolve_candidates(Dog)) == ["exactly_dog", "any_dog"]
    assert _names(f.resolve_candidates(Puppy)) == ["any_dog"]


def test_iterators_are_iterators() -> None:
    f = _hierarchy()
    it = f.itercandidates(Dog())
    assert iter(it) is it
    assert next(it).name == "dog"
    best = f.iterbestcandidates(Dog())
    assert iter(best) is best
    assert [m.name for m in best] == ["dog"]


def test_itercandidates_selects_when_called() -> None:
    """A method registered mid-iteration does not join the iteration."""
    f = _hierarchy()
    it = f.itercandidates(Dog())
    assert next(it).name == "dog"
    _add(f, "late", (Dog,), priority=5)
    assert "late" not in _names(it)
    assert f.candidates(Dog())[0].name == "late"


def test_errors_dispatch_raises_before_selection_propagate() -> None:
    """A method whose hints cannot be read fails the report as dispatch."""
    f = Function("broken")

    def broken(x: "Undefined") -> None:  # type: ignore[name-defined] # noqa: F821
        pass

    f.register(broken)
    reports = [
        f.dispatch,
        f.candidates,
        f.itercandidates,
        f.bestcandidates,
        f.iterbestcandidates,
        f.resolve_candidates,
        f.resolve_bestcandidates,
    ]
    for report in reports:
        with pytest.raises(NameError, match="Undefined"):
            report(1)


# --- the bug fix: ties are reported after the tie-breaks ---------------


def test_ambiguity_leaves_out_a_method_that_lost_on_priority() -> None:
    f = _priority_tie()
    with pytest.raises(AmbiguousMethodError) as info:
        f.dispatch(Sized(), Sized())
    assert _names(info.value.candidates) == ["left", "right"]
    message = str(info.value)
    assert "left(" in message and "right(" in message
    assert "sized(" not in message
    assert info.value.candidates == f.bestcandidates(Sized(), Sized())
    assert _names(f.candidates(Sized(), Sized())) == ["left", "right", "sized"]


def test_resolve_reports_the_survivors_too() -> None:
    f = _priority_tie()
    with pytest.raises(AmbiguousMethodError) as info:
        f.resolve(Sized, Sized)
    assert _names(info.value.candidates) == ["left", "right"]


@pytest.mark.parametrize("ambiguity", ["warn", "ignore"])
def test_resolve_takes_the_first_survivor(ambiguity: str) -> None:
    """`sized` is registered first but lost on priority: it is not taken."""
    f = _priority_tie()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chosen = f.resolve(Sized, Sized, ambiguity=ambiguity)
    assert chosen.name == "left"
    assert bool(caught) == (ambiguity == "warn")


def test_possible_fix_still_outranks_a_method_lost_on_priority() -> None:
    """An `Exact` in a priority loser still shapes the suggested fix.

    `exact` lost on priority, so it is not reported, but a new method at
    the default priority has to be strictly more specific than it too.
    """
    f = Function("fix")
    _add(f, "exact", (Exact[PetDog], object, object))
    _add(f, "first", (PetDog, Dog, object), priority=1)
    _add(f, "last", (object, Dog, Cat), priority=1)
    call = (PetDog(), Dog(), Cat())
    with pytest.raises(AmbiguousMethodError) as info:
        f.dispatch(*call)
    assert _names(info.value.candidates) == ["first", "last"]
    fix = str(info.value).splitlines()[-1].strip()
    assert fix == "fix(first_0: Exact[PetDog], first_1: Dog, first_2: Cat)"
    _add(f, "settled", (Exact[PetDog], Dog, Cat))
    assert f.dispatch(*call).name == "settled"


# --- caching and threads -----------------------------------------------


def test_interleaving_with_dispatch_and_registration() -> None:
    """Neither the call cache nor the report ever serves a stale answer."""
    f = _hierarchy()
    dog = Dog()
    assert f.dispatch(dog).name == "dog"
    assert f.bestcandidates(dog) == (f.dispatch(dog),)
    assert _names(f.candidates(dog))[0] == "dog"
    _add(f, "fancy_dog", (Dog,), priority=1)
    assert _names(f.bestcandidates(dog)) == ["fancy_dog"]
    assert f.dispatch(dog).name == "fancy_dog"
    assert _names(f.candidates(dog))[:2] == ["fancy_dog", "dog"]
    _add(f, "rival", (Dog,), priority=1)
    # Registering `rival` replaced nothing (it has another name) but ties.
    assert _names(f.bestcandidates(dog)) == ["fancy_dog", "rival"]
    with pytest.raises(AmbiguousMethodError) as info:
        f.dispatch(dog)
    assert info.value.candidates == f.bestcandidates(dog)
    f.clear_cache()
    assert f.resolve_bestcandidates(Dog) == f.bestcandidates(dog)
    assert f.resolve(Dog, ambiguity="ignore").name == "fancy_dog"


def test_concurrent_reports_and_registrations() -> None:
    """Reports taken while methods are registered are always consistent."""
    f = _hierarchy()
    errors: typing.List[BaseException] = []
    start = threading.Barrier(4)

    def report() -> None:
        start.wait()
        try:
            for _ in range(200):
                best = f.bestcandidates(Puppy())
                candidates = f.candidates(Puppy())
                assert best and candidates
                assert best[0].priority >= 0
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)

    def register() -> None:
        start.wait()
        for i in range(20):
            _add(f, f"extra{i}", (Dog,), priority=-1)

    threads = [threading.Thread(target=report) for _ in range(3)]
    threads.append(threading.Thread(target=register))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(f.candidates(Puppy())) == 5 + 20


# --- the bound view ------------------------------------------------------


def test_bound_function_fills_in_self() -> None:
    speak = Function("speak")

    class Kennel:
        pass

    class Shelter(Kennel):
        pass

    _add(speak, "anything", (object, object))
    _add(speak, "kennel_dog", (Kennel, Dog))
    _add(speak, "shelter_any", (Shelter, object))
    Kennel.speak = speak  # type: ignore[attr-defined]
    bound = Shelter().speak  # type: ignore[attr-defined]
    assert _names(bound.candidates(Puppy())) == [
        "kennel_dog",
        "shelter_any",
        "anything",
    ]
    assert tuple(bound.itercandidates(Puppy())) == bound.candidates(Puppy())
    assert _names(bound.bestcandidates(Puppy())) == [
        "kennel_dog",
        "shelter_any",
    ]
    assert tuple(bound.iterbestcandidates(Cat())) == bound.bestcandidates(
        Cat()
    )
    assert _names(bound.bestcandidates(Cat())) == ["shelter_any"]
    assert _names(bound.resolve_candidates(Dog)) == _names(
        bound.candidates(Dog())
    )
    assert _names(bound.resolve_bestcandidates(Cat)) == ["shelter_any"]
    assert bound.resolve_bestcandidates(Cat) == (bound.resolve(Cat),)
