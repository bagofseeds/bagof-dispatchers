"""Tests for `Between[L, U]`, an interval on a value, a class or a hint."""

# stdlib
import itertools
import numbers
import warnings

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers import (
    AmbiguousMethodError,
    Between,
    Exact,
    Function,
    Hint,
    NoMethodError,
    Signature,
    Super,
    SuperType,
    dispatch,
)
from bagof.dispatchers._signature import _render_hint, _structural_hint_eq
from bagof.dispatchers.core import ishintstance, issubhint, resolve_hint
from bagof.dispatchers.core._bounds import (
    _Lower,
    between_bounds,
    bounds_of,
    empty_interval_message,
    is_between,
    is_bound,
    is_unbounded_form,
    unbounded_form_message,
)
from bagof.dispatchers.core._exact import is_exact
from bagof.dispatchers.core._super import _render_target, is_super

Integral = numbers.Integral


class Animal:
    pass


class Dog(Animal):
    pass


class Puppy(Dog):
    pass


_T = tx.TypeVar("_T")


def _tb(lower: tx.Any, upper: tx.Any) -> tx.Any:
    return tx.Type[Between[lower, upper]]


def _hb(lower: tx.Any, upper: tx.Any) -> tx.Any:
    return Hint[Between[lower, upper]]


def _ts(c: tx.Any) -> tx.Any:
    return tx.Type[Super[c]]


def _hs(c: tx.Any) -> tx.Any:
    return Hint[Super[c]]


def _te(c: tx.Any) -> tx.Any:
    return tx.Type[Exact[c]]


def _he(c: tx.Any) -> tx.Any:
    return Hint[Exact[c]]


# --- construction ------------------------------------------------------


def test_between_spelling_and_detection() -> None:
    hint = Between[bool, int]
    assert hint == tx.Annotated[int, _Lower(bool)]
    assert hash(hint) == hash(Between[bool, int])
    assert repr(_Lower(bool)) == "LOWER(bool)"
    assert repr(hint).endswith("[int, LOWER(bool)]")
    assert _Lower(bool).__eq__(bool) is NotImplemented
    assert is_between(hint) is True
    assert is_between(int) is False
    assert is_between(tx.Annotated[int, "meta"]) is False
    assert between_bounds(hint) == (bool, int)
    # The markers never read as each other.
    assert is_super(hint) is False
    assert is_exact(hint) is False
    assert is_between(Super[int]) is False
    assert is_between(Exact[int]) is False


def test_is_bound_covers_super_and_between() -> None:
    assert is_bound(Super[int]) is True
    assert is_bound(Between[bool, int]) is True
    assert is_bound(int) is False
    assert is_bound(Exact[int]) is False


def test_extra_metadata_rides_alongside_the_lower_bound() -> None:
    hint = tx.Annotated[Between[bool, int], "meta"]
    assert is_between(hint) is True
    assert between_bounds(hint) == (bool, int)


@pytest.mark.parametrize(
    "form", [tx.Union, tx.Literal, tx.Annotated], ids=repr
)
def test_between_of_a_bare_special_form_constructs(form: tx.Any) -> None:
    # A bare special form must build even where `typing._type_check` refuses
    # it as an `Annotated` argument; the alias fallback handles it.
    hint = Between[tx.Never, form]
    assert is_between(hint) is True
    assert between_bounds(hint) == (tx.Never, form)


def test_a_none_bound_reads_as_nonetype() -> None:
    assert between_bounds(Between[None, object]) == (type(None), object)
    assert between_bounds(Between[tx.Never, None]) == (tx.Never, type(None))
    with pytest.raises(TypeError, match=r"^Between\[NoneType, int\] is empty"):
        Between[None, int]


def test_degenerate_intervals_are_legal() -> None:
    assert between_bounds(Between[int, int]) == (int, int)
    assert between_bounds(Between[tx.Never, tx.Never]) == (
        tx.Never,
        tx.Never,
    )
    assert between_bounds(Between[tx.Any, tx.Any]) == (tx.Any, tx.Any)
    assert between_bounds(Between[tx.Never, tx.Any]) == (tx.Never, tx.Any)


@pytest.mark.parametrize("item", [int, (int,), (bool, int, object)], ids=repr)
def test_between_takes_two_bounds(item: tx.Any) -> None:
    with pytest.raises(TypeError) as info:
        Between[item]
    assert str(info.value) == (
        "Between[...] takes two type hints, a lower and an upper bound, as "
        f"Between[L, U]; got {item!r}."
    )


def test_between_takes_only_hints() -> None:
    with pytest.raises(TypeError) as info:
        Between[1, int]
    assert str(info.value) == (
        "Between[...] takes type hints as its bounds, got 1."
    )
    with pytest.raises(TypeError, match="as its bounds, got 2"):
        Between[int, 2]


def test_empty_interval_is_refused() -> None:
    with pytest.raises(TypeError) as info:
        Between[int, bool]
    assert str(info.value) == (
        "Between[int, bool] is empty: int is not a sub-hint of bool, so no "
        "hint lies between them. Did you mean Between[bool, int]?"
    )
    with pytest.raises(TypeError) as info:
        Between[tx.Any, int]
    assert str(info.value) == (
        "Between[Any, int] is empty: Any is the top of the hint order, so "
        "nothing lies between Any and int. For no lower bound write "
        "Between[Never, int], which means the same as plain int."
    )
    with pytest.raises(TypeError) as info:
        Between[int, str]
    assert str(info.value) == (
        "Between[int, str] is empty: int is not a sub-hint of str, and str "
        "is not a sub-hint of int, so no hint lies between them."
    )
    with pytest.raises(TypeError, match=r"Did you mean Between\[Dog, Animal"):
        Between[Animal, Dog]


@pytest.mark.parametrize(
    "build, needle",
    [
        (lambda: Between[tx.Optional[Super[int]], object], "Super[int]"),
        (lambda: Between[tx.Never, tx.Optional[Super[int]]], "Super[int]"),
        (lambda: Super[tx.Optional[Between[bool, int]]], "Between[bool, int]"),
        (
            lambda: Super[tx.TypeVar("_TS", bound=Super[int])],
            "Super[int]",
        ),
    ],
)
def test_a_bound_inside_a_bound_is_refused_when_written(
    build: tx.Callable[[], tx.Any], needle: str
) -> None:
    with pytest.raises(TypeError) as info:
        build()
    assert str(info.value).startswith(
        f"{needle} cannot appear inside an Exact, Super or Between form"
    )


def test_an_unbounded_form_inside_a_bound_is_refused_when_written() -> None:
    with pytest.raises(TypeError, match=r"^Between needs two bounds"):
        Super[tx.Optional[Between]]


def test_empty_interval_message_waits_for_forward_references() -> None:
    assert empty_interval_message(int, "Later") is None
    assert empty_interval_message(int, tx.ForwardRef("Later")) is None
    assert empty_interval_message(bool, int) is None


def test_a_forward_reference_lower_bound_is_refused() -> None:
    # A lower bound travels as metadata, which nothing resolves later.
    for lower in ("Later", tx.ForwardRef("Later")):
        with pytest.raises(TypeError) as info:
            Between[lower, Animal]
        assert str(info.value) == (
            "Between[...] cannot take the forward reference 'Later' as its "
            "lower bound, because a lower bound is kept as written and "
            "never resolved. Define Later before the annotation is read, or "
            "quote the whole annotation instead, as "
            "'Type[Between[Later, Animal]]'."
        )


@pytest.mark.parametrize(
    "lower",
    [tx.Optional["Later"], tx.List["Later"]],
    ids=["optional", "list"],
)
def test_a_nested_forward_reference_lower_bound_is_refused(
    lower: tx.Any,
) -> None:
    # Refused as a quoted lower bound, not reported as an empty interval.
    # How `Optional` is spelled differs between Python versions.
    shown = _render_target(lower)
    assert "Later" in shown
    with pytest.raises(TypeError) as info:
        Between[lower, object]
    assert str(info.value) == (
        f"Between[...] cannot take {shown}, which holds a forward "
        "reference, as its lower bound, because a lower bound is kept as "
        f"written and never resolved. Define every name that {shown} "
        "refers to before the annotation is read, or quote the whole "
        f"annotation instead, as 'Type[Between[{shown}, object]]'."
    )


def test_a_literal_string_is_not_a_forward_reference() -> None:
    assert between_bounds(Between[tx.Literal["a"], str]) == (
        tx.Literal["a"],
        str,
    )


def test_a_nested_forward_reference_upper_bound_defers_the_check() -> None:
    assert empty_interval_message(int, tx.List["Later"]) is None
    lower, upper = between_bounds(Between[int, tx.List["Later"]])
    assert lower is int
    assert upper == tx.List["Later"]
    assert between_bounds(Between[tx.Never, tx.Callable[["Later"], int]])


def test_a_forward_reference_upper_bound_is_kept() -> None:
    hint = Between[Dog, "Later"]
    lower, upper = between_bounds(hint)
    assert lower is Dog
    assert upper.__forward_arg__ == "Later"


_NESTED = [
    (lambda: Between[Exact[int], object], "Exact[int]"),
    (lambda: Between[bool, Super[int]], "Super[int]"),
    (lambda: Between[Between[bool, int], object], "Between[bool, int]"),
    (lambda: Between[Super, int], "Super"),
    (lambda: Between[Exact, int], "Exact"),
    (lambda: Between[tx.Never, Between], "Between"),
    (lambda: Between[tx.Never, SuperType], "SuperType"),
]


@pytest.mark.parametrize("build, shown", _NESTED)
def test_markers_cannot_nest_inside_between(
    build: tx.Callable[[], tx.Any], shown: str
) -> None:
    with pytest.raises(TypeError) as info:
        build()
    assert str(info.value) == (
        f"Between[...] cannot take {shown} as a bound: a bound is a plain "
        "hint, and Exact, Super and Between cannot be nested. Write "
        "Between[L, U] with plain L and U. "
        "On a value, write Between[L, U] for a value whose class lies "
        "between L and U, Super[C] for a value whose class is C or a class "
        "above it, or Exact[C] for a value whose class is exactly C. Inside "
        "Type[...], write Type[Between[L, U]] for a class between L and U, "
        "Type[Super[C]] for C or any class above it, or Type[Exact[C]] for "
        "exactly C; the same holds inside Hint[...]."
    )


@pytest.mark.parametrize("outer", ["Super", "Exact"])
def test_between_cannot_nest_inside_super_or_exact(outer: str) -> None:
    form = {"Super": Super, "Exact": Exact}[outer]
    with pytest.raises(TypeError) as info:
        form[Between[bool, int]]
    assert str(info.value) == (
        f"{outer}[...] cannot take Between[bool, int]: Exact, Super and "
        "Between cannot be nested, because each of them already describes "
        "the whole hint at its position. "
        "On a value, write Between[L, U] for a value whose class lies "
        "between L and U, Super[C] for a value whose class is C or a class "
        "above it, or Exact[C] for a value whose class is exactly C. Inside "
        "Type[...], write Type[Between[L, U]] for a class between L and U, "
        "Type[Super[C]] for C or any class above it, or Type[Exact[C]] for "
        "exactly C; the same holds inside Hint[...]."
    )


def test_bounds_of_reads_between() -> None:
    assert bounds_of(Between[Dog, Animal], object) == (Dog, Animal)
    assert bounds_of(Between[bool, Integral], tx.Any) == (bool, Integral)
    # The other spellings are unchanged.
    assert bounds_of(Super[int], object) == (int, object)
    assert bounds_of(Exact[int], object) == (int, int)
    assert bounds_of(int, object) == (tx.Never, int)


# --- where an interval can stand ---------------------------------------


def test_is_unbounded_form() -> None:
    for hint in (Between, Super, SuperType):
        assert is_unbounded_form(hint) is True
    for hint in (int, Between[bool, int], Super[int], _tb(bool, int)):
        assert is_unbounded_form(hint) is False


def test_unbounded_form_messages() -> None:
    assert unbounded_form_message(Between) == (
        "Between needs two bounds: write Between[L, U] to accept a value "
        "whose class lies between L and U, Type[Between[L, U]] to accept a "
        "class between them, or Hint[Between[L, U]] to accept a hint "
        "between them."
    )
    assert unbounded_form_message(Super).startswith("Super needs a bound")


_ON_A_VALUE = [
    Between[bool, int],
    tx.Optional[Between[bool, int]],
    tx.TypeVar("_TBETWEEN", bound=Between[bool, int]),
]


@pytest.mark.parametrize("hint", _ON_A_VALUE, ids=repr)
def test_an_interval_on_a_value_is_read_by_the_relation(
    hint: tx.Any,
) -> None:
    assert ishintstance(1, hint) is True
    assert ishintstance(True, hint) is True
    assert ishintstance(object(), hint) is False
    assert issubhint(hint, int) is (hint is not _ON_A_VALUE[1])
    assert issubhint(hint, tx.Optional[int]) is True
    assert issubhint(int, hint) is False


_REFUSED = [
    (Between, "Between needs two bounds"),
    (tx.Type[tx.Union[Between[bool, int], str]], "cannot be a member of a"),
    # `Between` written around the whole form is not rewritten.
    (
        Between[tx.Type[bool], tx.Type[int]],
        "To bound the class passed to a Type[...] parameter write "
        "Type[Between[bool, int]].",
    ),
]


@pytest.mark.parametrize("hint, needle", _REFUSED, ids=repr)
def test_misplaced_between_is_refused_by_the_relation(
    hint: tx.Any, needle: str
) -> None:
    for call in (
        lambda: issubhint(hint, _tb(bool, int)),
        lambda: issubhint(_tb(bool, int), hint),
        lambda: ishintstance(int, hint),
    ):
        with pytest.raises(TypeError) as info:
            call()
        assert needle in str(info.value)


def test_between_in_an_invariant_argument_is_a_range() -> None:
    assert issubhint(tx.List[bool], tx.List[Between[bool, int]]) is True
    assert issubhint(tx.List[Integral], tx.List[Between[bool, int]]) is False
    assert issubhint(tx.List[Between[bool, int]], tx.List[int]) is False


def test_an_interval_is_a_hint_value() -> None:
    f = Function("f")

    @f.register((Hint[tx.Any],))
    def _any(h: object) -> str:
        return "any"

    assert f(Between[bool, int]) == "any"
    # An interval on a value is below the upper bound it names.
    assert ishintstance(Between[bool, int], Hint[int]) is True
    assert ishintstance(Between[bool, int], _hb(bool, Integral)) is False


def _fn(annotation: tx.Any, kind: str = "x") -> tx.Any:
    if kind == "args":

        def f(*args: annotation) -> None: ...

    elif kind == "kwargs":

        def f(**kwargs: annotation) -> None: ...

    else:

        def f(x: annotation) -> None: ...

    return f


@pytest.mark.parametrize(
    "hint, needle",
    [
        (
            tx.Sequence[Between[bool, int]],
            "Between[bool, int] puts a lower bound on argument 1 of Sequence",
        ),
        (
            tx.Dict[str, tx.Optional[Between[bool, int]]],
            "Between[bool, int] cannot be a member of a union, or the bound "
            "of a TypeVar, inside a type argument of Dict",
        ),
        (Between, "Between needs two bounds"),
        (tx.Type[Between], "Between needs two bounds"),
        (Hint[Between], "Between needs two bounds"),
        (
            tx.Type[Between[tx.Never, tx.Sequence[Super[int]]]],
            "Super[int] puts a lower bound on argument 1 of Sequence",
        ),
        (
            # What `Between[Never, "Later"]` becomes once `Later` resolves to
            # a hint holding a bound: the resolved hint is never built by
            # `Between`.
            tx.Type[tx.Annotated[tx.Optional[Super[int]], _Lower(tx.Never)]],
            "Super[int] cannot appear inside an Exact, Super or Between form",
        ),
        (
            Between[tx.Never, tx.List[int]],
            "Between[Never, List[int]] cannot bound a value with List[int]",
        ),
        (
            Between[tx.Type[bool], tx.Type[int]],
            "write Type[Between[bool, int]]",
        ),
        (
            Between[Hint[bool], tx.Any],
            "write Hint[Between[bool, Any]]",
        ),
    ],
    ids=repr,
)
def test_misplaced_between_is_refused_at_registration(
    hint: tx.Any, needle: str
) -> None:
    with pytest.raises(TypeError) as info:
        dispatch(_fn(hint))
    assert str(info.value).startswith("'x' of f: ")
    assert needle in str(info.value)

    f = Function("g")
    with pytest.raises(TypeError) as info:
        f.register((hint,))(lambda y: y)
    assert str(info.value).startswith("'y' of <lambda>: ")
    assert needle in str(info.value)


@pytest.mark.parametrize("hint", _ON_A_VALUE, ids=repr)
def test_between_on_a_value_is_accepted_at_registration(
    hint: tx.Any,
) -> None:
    f = Function("f")
    f.register((hint,))(lambda x: "between")
    assert f(1) == "between"
    assert f(True) == "between"
    with pytest.raises(NoMethodError):
        f(object())


@pytest.mark.parametrize("kind", ["args", "kwargs"])
def test_between_is_accepted_on_a_catch_all(kind: str) -> None:
    f = dispatch(_fn(Between[Dog, Animal], kind))
    if kind == "args":
        assert f(Dog(), Animal()) is None
        with pytest.raises(NoMethodError):
            f(Dog(), Puppy())
    else:
        assert f(a=Dog(), b=Animal()) is None
        with pytest.raises(NoMethodError):
            f(a=Puppy())


def test_between_is_accepted_by_from_hints() -> None:
    sig = Signature.from_hints(Between[bool, int], scale=Between[bool, int])
    assert sig.dispatched_names == ("scale",)


def test_between_inside_type_or_hint_is_accepted_at_registration() -> None:
    f = Function("f")

    @f.register((tx.Optional[_tb(Dog, Animal)], _hb(bool, Integral)))
    def _both(cls: object, h: object) -> str:
        return "both"

    # A bound is also read as the whole argument of an invariant slot.
    @f.register(
        (
            tx.Dict[str, Between[bool, int]],
            tx.Type[Between[tx.Never, tx.List[Super[int]]]],
        )
    )
    def _slots(d: object, cls: object) -> str:
        return "slots"

    assert f(Dog, int) == "both"
    assert f(None, bool) == "both"


def test_forward_reference_empty_interval_is_refused_at_settle() -> None:
    # The upper bound is a forward reference when the method is registered,
    # so the interval can only be checked once it resolves.
    namespace: tx.Dict[str, tx.Any] = {
        "Type": tx.Type,
        "Between": Between,
        "Animal": Animal,
    }
    exec("def f(cls: Type[Between[Animal, 'Later']]): pass", namespace)
    sig = Signature.from_callable(namespace["f"])
    assert sig._deferred
    namespace["Later"] = Puppy
    with pytest.raises(TypeError) as info:
        sig._settle()
    assert str(info.value) == (
        "'cls' of f: Between[Animal, Puppy] is empty: Animal is not a "
        "sub-hint of Puppy, so no hint lies between them. Did you mean "
        "Between[Puppy, Animal]?"
    )


def test_forward_reference_empty_interval_is_refused_at_registration(
) -> None:
    # Resolved by the time the method is registered: refused straight away.
    namespace: tx.Dict[str, tx.Any] = {
        "Type": tx.Type,
        "Between": Between,
        "Animal": Animal,
        "Later": Puppy,
    }
    exec("def f(cls: Type[Between[Animal, 'Later']]): pass", namespace)
    with pytest.raises(TypeError, match=r"^'cls' of f: Between\[Animal, P"):
        Function("g").register(namespace["f"])


def test_a_quoted_annotation_with_an_empty_interval_names_the_function(
) -> None:
    # A wholly quoted annotation builds the interval when it is resolved, so
    # the refusal arrives through the resolution error, naming the function.
    namespace: tx.Dict[str, tx.Any] = {
        "Type": tx.Type,
        "Between": Between,
        "Animal": Animal,
    }
    exec("def f(cls: 'Type[Between[Later, Animal]]'): pass", namespace)
    sig = Signature.from_callable(namespace["f"])
    namespace["Later"] = object
    with pytest.raises(NameError) as info:
        sig._settle()
    assert str(info.value).startswith("cannot resolve the type hints of f: ")
    assert "Did you mean Between[Animal, object]?" in str(info.value)


def test_a_forward_reference_inside_between_settles_later() -> None:
    namespace: tx.Dict[str, tx.Any] = {
        "Type": tx.Type,
        "Between": Between,
        "Dog": Dog,
    }
    exec("def f(cls: Type[Between[Dog, 'Later']]): return 'in'", namespace)
    f = Function("f")
    f.register(namespace["f"])
    namespace["Later"] = Animal
    assert f(Dog) == "in"
    assert f(Animal) == "in"
    with pytest.raises(NoMethodError):
        f(object)


# --- value level -------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (Dog, True),
        (Animal, True),
        (Puppy, False),
        (object, False),
        (int, False),
        (Dog(), False),
    ],
    ids=repr,
)
def test_value_level_type_between(value: tx.Any, expected: bool) -> None:
    assert ishintstance(value, _tb(Dog, Animal)) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (int, True),
        (bool, True),
        (Integral, True),
        (object, False),
        (tx.Any, False),
        (tx.Union[bool, str], False),
        (str, False),
        ("Foo", False),
        (1, False),
    ],
    ids=repr,
)
def test_value_level_hint_between(value: tx.Any, expected: bool) -> None:
    assert ishintstance(value, _hb(bool, Integral)) is expected


# --- hint level --------------------------------------------------------

_TYPE_ROWS = [
    _tb(Dog, Animal),
    _tb(Puppy, Dog),
    _tb(Puppy, Animal),
    _ts(Dog),
    _ts(Puppy),
    _te(Dog),
    tx.Type[Animal],
    tx.Type[Dog],
]

_TYPE_COLUMNS = _TYPE_ROWS[:6] + [
    tx.Type[Animal],
    tx.Type[Dog],
    tx.Type,
    tx.Any,
    object,
]

# Rows are sub-hints, columns super-hints, in the order of `_TYPE_COLUMNS`.
_TYPE_TABLE = [
    "T F T T T F T F T T T",
    "F T T F T F T T T T T",
    "F F T F T F T F T T T",
    "F F F T T F F F T T T",
    "F F F F T F F F T T T",
    "T T T T T T T T T T T",
    "F F F F F F T F T T T",
    "F F F F F F T T T T T",
]

_TYPE_CASES = [
    (sub, sup, cell == "T")
    for sub, row in zip(_TYPE_ROWS, _TYPE_TABLE)
    for sup, cell in zip(_TYPE_COLUMNS, row.split())
]


@pytest.mark.parametrize("sub, sup, expected", _TYPE_CASES)
def test_issubhint_type_between_table(
    sub: tx.Any, sup: tx.Any, expected: bool
) -> None:
    assert issubhint(sub, sup) is expected


_HINT_ROWS = [
    _hb(bool, Integral),
    _hb(bool, int),
    _hs(int),
    _hs(bool),
    _he(int),
    Hint[Integral],
    Hint[int],
    Hint,
]

_HINT_COLUMNS = _HINT_ROWS + [tx.Any, object]

_HINT_TABLE = [
    "T F F T F T F T T F",
    "T T F T F T T T T F",
    "F F T T F F F T T F",
    "F F F T F F F T T F",
    "T T T T T T T T T F",
    "F F F F F T F T T F",
    "F F F F F T T T T F",
    "F F F F F F F T T F",
]

_HINT_CASES = [
    (sub, sup, cell == "T")
    for sub, row in zip(_HINT_ROWS, _HINT_TABLE)
    for sup, cell in zip(_HINT_COLUMNS, row.split())
]


@pytest.mark.parametrize("sub, sup, expected", _HINT_CASES)
def test_issubhint_hint_between_table(
    sub: tx.Any, sup: tx.Any, expected: bool
) -> None:
    assert issubhint(sub, sup) is expected


def _equivalent(a: tx.Any, b: tx.Any) -> bool:
    return issubhint(a, b) and issubhint(b, a)


def test_the_other_spellings_are_intervals() -> None:
    assert _equivalent(_tb(tx.Never, Animal), tx.Type[Animal])
    assert _equivalent(_tb(Dog, object), _ts(Dog))
    assert _equivalent(_hb(tx.Never, int), Hint[int])
    assert _equivalent(_hb(int, tx.Any), _hs(int))
    # `Exact[C]` is narrower than the interval of hints equivalent to `C`.
    assert issubhint(_te(Dog), _tb(Dog, Dog)) is True
    assert issubhint(_tb(Dog, Dog), _te(Dog)) is False
    # The bottom holds no class, so `Type[Never]` is below every interval;
    # the hint `Never` is a value of `Hint[Never]`, so there the bounds decide.
    assert issubhint(tx.Type[tx.Never], _tb(Dog, Animal)) is True
    assert issubhint(Hint[tx.Never], _hb(bool, int)) is False
    assert issubhint(Hint[tx.Never], _hb(tx.Never, int)) is True


# The empty-ended intervals hold no class, so they sit below every
# interval, as `Type[Never]` does.
_TYPE_CORPUS = _TYPE_COLUMNS[:10] + [
    tx.Type[tx.Never],
    _tb(tx.Never, tx.Never),
    _te(tx.Never),
]
_HINT_CORPUS = _HINT_ROWS + [tx.Any]


@pytest.mark.parametrize("corpus", [_TYPE_CORPUS, _HINT_CORPUS])
def test_between_order_is_a_preorder(corpus: tx.List[tx.Any]) -> None:
    for hint in corpus:
        assert issubhint(hint, hint) is True
    for a, b, c in itertools.product(corpus, repeat=3):
        if issubhint(a, b) and issubhint(b, c):
            assert issubhint(a, c), (a, b, c)


def test_value_consistency_type_between() -> None:
    values = (object, Animal, Dog, Puppy, int)
    for a, b in itertools.product(_TYPE_CORPUS, repeat=2):
        if not issubhint(a, b):
            continue
        for value in values:
            if ishintstance(value, a):
                assert ishintstance(value, b), (value, a, b)


def test_value_consistency_hint_between() -> None:
    values = (object, Integral, int, bool, str, tx.Any, tx.Union[bool, str])
    for a, b in itertools.product(_HINT_CORPUS, repeat=2):
        if not issubhint(a, b):
            continue
        for value in values:
            if ishintstance(value, a):
                assert ishintstance(value, b), (value, a, b)


def test_a_typevar_or_union_below_an_interval_is_read_by_its_bound() -> None:
    ordinary = tx.TypeVar("ordinary", bound=Dog)
    exact = tx.TypeVar("exact", bound=Exact[Dog])
    assert issubhint(tx.Type[ordinary], _tb(Dog, Animal)) is False
    assert issubhint(tx.Type[exact], _tb(Dog, Animal)) is True
    both = tx.Type[tx.Union[Exact[Dog], Exact[Animal]]]
    assert issubhint(both, _tb(Dog, Animal)) is True
    assert issubhint(both, _tb(Puppy, Dog)) is False


# --- dispatch ----------------------------------------------------------


def test_dispatch_type_between() -> None:
    f = Function("f")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        @f.register((_tb(Dog, Animal),))
        def _between(cls: object) -> str:
            return "between"

        @f.register((_te(Dog),))
        def _exact(cls: object) -> str:
            return "exact"

        @f.register((tx.Type[Puppy],))
        def _puppy(cls: object) -> str:
            return "puppy"

    assert f(Animal) == "between"
    assert f(Dog) == "exact"
    assert f(Puppy) == "puppy"
    with pytest.raises(NoMethodError):
        f(object)


def test_dispatch_between_vs_plain_warns_and_priority_settles() -> None:
    f = Function("f")

    @f.register((tx.Type[Dog],))
    def _plain(cls: object) -> str:
        return "plain"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register((_tb(Dog, Animal),))
        def _between(cls: object) -> str:
            return "between"

    with pytest.raises(AmbiguousMethodError):
        f(Dog)
    assert f(Puppy) == "plain"
    assert f(Animal) == "between"

    g = Function("g")

    @g.register((tx.Type[Dog],))
    def _g_plain(cls: object) -> str:
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        @g.register((_tb(Dog, Animal),), priority=1)
        def _g_between(cls: object) -> str:
            return "between"

    assert g(Dog) == "between"
    assert g(Puppy) == "plain"


class _DiaC:
    pass


class _DiaD:
    pass


class _DiaX(_DiaC, _DiaD):
    pass


class _DiaA(_DiaX):
    pass


class _DiaB(_DiaX):
    pass


def test_dispatch_between_pair_sharing_a_class_off_both_ends_warns() -> None:
    # `_DiaX` lies in both intervals, although no end of either one does.
    f = Function("f")

    @f.register((_tb(_DiaA, _DiaC),))
    def _left(cls: object) -> str:
        return "left"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register((_tb(_DiaB, _DiaD),))
        def _right(cls: object) -> str:
            return "right"

    with pytest.raises(AmbiguousMethodError):
        f(_DiaX)
    assert f(_DiaA) == "left"
    assert f(_DiaB) == "right"


def test_dispatch_between_in_one_of_two_positions() -> None:
    f = Function("f")

    @f.register((tx.Type[Dog], _tb(Dog, Animal)))
    def _first(a: object, b: object) -> str:
        return "first"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register((_tb(Dog, Animal), tx.Type[Dog]))
        def _second(a: object, b: object) -> str:
            return "second"

    with pytest.raises(AmbiguousMethodError):
        f(Dog, Dog)
    assert f(Puppy, Animal) == "first"
    assert f(Animal, Puppy) == "second"

    # A second position the two methods cannot share settles it.
    g = Function("g")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        @g.register((tx.Type[Dog], _tb(Dog, Animal)))
        def _g_first(a: object, b: object) -> str:
            return "first"

        @g.register((_tb(Dog, Animal), tx.Type[Puppy]))
        def _g_second(a: object, b: object) -> str:
            return "second"

    assert g(Dog, Dog) == "first"
    assert g(Dog, Puppy) == "second"


@pytest.mark.parametrize(
    "wrapped",
    [
        tx.Optional[_tb(Dog, Animal)],
        tx.TypeVar("_TVB", bound=_tb(Dog, Animal)),
    ],
    ids=["optional", "typevar"],
)
def test_dispatch_wrapped_between_warns(wrapped: tx.Any) -> None:
    f = Function("f")

    @f.register((tx.Type[Dog],))
    def _plain(cls: object) -> str:
        return "plain"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register((wrapped,))
        def _wrapped(cls: object) -> str:
            return "wrapped"

    with pytest.raises(AmbiguousMethodError):
        f(Dog)


def test_a_narrower_interval_wins() -> None:
    f = Function("f")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        @f.register((_ts(Dog),))
        def _above(cls: object) -> str:
            return "super"

        @f.register((_tb(Dog, Animal),))
        def _between(cls: object) -> str:
            return "between"

    assert f(Dog) == "between"
    assert f(Animal) == "between"
    assert f(object) == "super"


def test_dispatch_hint_between() -> None:
    f = Function("f")

    @f.register((_hb(bool, Integral),))
    def _between(h: object) -> str:
        return "between"

    @f.register((_he(bool),))
    def _exact(h: object) -> str:
        return "exact"

    @f.register((Hint[tx.Any],))
    def _any(h: object) -> str:
        return "any"

    assert f(int) == "between"
    assert f(Integral) == "between"
    assert f(bool) == "exact"
    assert f(object) == "any"
    assert f(tx.Any) == "any"


def test_between_renders() -> None:
    f = Function("f")

    @f.register((_tb(bool, int), _hb(bool, Integral)))
    def _between(cls: object, h: object) -> None: ...

    (method,) = f.methods
    text = method.describe()
    assert "Type[Between[bool, int]]" in text
    assert "Hint[Between[bool, Integral]]" in text
    optional = _render_hint(tx.Optional[int]).replace(
        "int", "Type[Between[Dog, Animal]]"
    )
    assert _render_hint(tx.Optional[_tb(Dog, Animal)]) == optional


def test_two_spellings_of_one_interval_are_the_same_hint() -> None:
    assert _structural_hint_eq(
        Between[tx.List[int], object], Between[tx.List[int], object]
    )
    assert not _structural_hint_eq(
        Between[tx.List[int], object], Between[tx.List[str], object]
    )
    assert not _structural_hint_eq(Between[bool, int], Super[int])


# --- registries --------------------------------------------------------


def test_resolve_hint_between_key() -> None:
    registry = {_tb(Dog, Animal): 1, tx.Type[Puppy]: 2}
    assert resolve_hint(_te(Dog), registry) == 1
    assert resolve_hint(_tb(Dog, Animal), registry) == 1
    assert resolve_hint(tx.Type[Puppy], registry) == 2
    assert resolve_hint(tx.Type[Animal], registry, default=0) == 0


def test_resolve_hint_bare_between_query_raises() -> None:
    with pytest.raises(TypeError, match=r"Between needs two bounds"):
        resolve_hint(Between, {int: 1})
