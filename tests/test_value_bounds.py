"""Tests for `Super[C]` and `Between[L, U]` on a value parameter.

On a value, a bound constrains the value's own class, exactly as
`Type[...]` of the same bound constrains a class passed in: `Super[C]`
accepts a value whose class is `C` or above it, and `Between[L, U]` one
whose class lies between `L` and `U`.
"""

# stdlib
import abc
import itertools
import numbers
import typing
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
    dispatch,
)
from bagof.dispatchers._lattice import overlaps
from bagof.dispatchers.core import (
    ishintstance,
    issubhint,
    mro_index,
    resolve_hint,
)
from bagof.dispatchers.core._bounds import (
    _Lower,
    find_bound,
    holds_bound,
    value_bound_message,
)
from bagof.dispatchers.core._relation import is_class_hint
from bagof.dispatchers.core._super import SUPER

Integral = numbers.Integral


class Animal:
    pass


class Dog(Animal):
    def speak(self) -> str:
        return "woof"


class Puppy(Dog):
    pass


class Cat(Animal):
    pass


@tx.runtime_checkable
class Speaks(tx.Protocol):
    def speak(self) -> str: ...


@tx.runtime_checkable
class HasName(tx.Protocol):
    name: str


class Movie(tx.TypedDict):
    title: str


_T = tx.TypeVar("_T")


class Box(tx.Generic[_T]):
    pass


UserId = tx.NewType("UserId", int)
T_A = tx.TypeVar("T_A", bound=Animal)
T_S = tx.TypeVar("T_S", bound=Super[Dog])
B, S, E = Between, Super, Exact


def _fn(annotation: tx.Any) -> tx.Any:
    def f(x: annotation) -> None: ...

    return f


# --- membership --------------------------------------------------------

_COLUMNS = [
    B[Dog, Animal],
    B[Puppy, Dog],
    S[Dog],
    S[Puppy],
    E[Dog],
    Dog,
    Animal,
    S[tx.Never],
    S[object],
    S[tx.Any],
]
_TABLE = [
    (Puppy(), "FTFTFTTTFF"),
    (Dog(), "TTTTTTTTFF"),
    (Animal(), "TFTTFFTTFF"),
    (object(), "FFTTFFFTTF"),
    (Cat(), "FFFFFFTTFF"),
    (1, "FFFFFFFTFF"),
]
_MEMBERSHIP = [
    (value, hint, row[column] == "T")
    for value, row in _TABLE
    for column, hint in enumerate(_COLUMNS)
]


@pytest.mark.parametrize("value, hint, expected", _MEMBERSHIP, ids=repr)
def test_ishintstance_value_table(
    value: tx.Any, hint: tx.Any, expected: bool
) -> None:
    assert ishintstance(value, hint) is expected


def test_optional_and_typevar_bound_read_through() -> None:
    optional = tx.Optional[S[Dog]]
    for value, expected in [
        (None, True),
        (Dog(), True),
        (Animal(), True),
        (object(), True),
        (Puppy(), False),
        (Cat(), False),
    ]:
        assert ishintstance(value, optional) is expected
        assert ishintstance(value, T_S) is (expected and value is not None)


def test_between_c_c_is_equivalence_and_exact_is_identity() -> None:
    class First(abc.ABC):  # noqa: B024 -- equivalence is the point
        @classmethod
        def __subclasshook__(cls, other: type) -> bool:
            return other is Second or NotImplemented

    class Second(abc.ABC):  # noqa: B024 -- equivalence is the point
        @classmethod
        def __subclasshook__(cls, other: type) -> bool:
            return other is First or NotImplemented

    # Each is a subclass of the other, so the two classes are equivalent
    # without being the same class.
    assert issubclass(First, Second) and issubclass(Second, First)
    assert ishintstance(Second(), B[First, First]) is True
    assert ishintstance(Second(), E[First]) is False
    assert ishintstance(First(), B[First, First]) is True
    assert ishintstance(First(), E[First]) is True
    assert ishintstance(1, B[int, int]) is True
    assert ishintstance(True, B[int, int]) is False


def test_super_never_equals_object_on_every_interpreter() -> None:
    values = (object(), Dog(), 1, None, "a", typing.Any, tx.Any, int)
    for value in values:
        assert ishintstance(value, S[tx.Never]) is ishintstance(value, object)
    assert issubhint(S[tx.Never], object) is True
    assert issubhint(object, S[tx.Never]) is True


# --- endpoints ---------------------------------------------------------


_LEGAL_ENDS = [
    Dog,
    Integral,
    object,
    type,
    tx.Type,
    typing.Type,
    type(None),
    UserId,
    tx.List,
    tx.Sequence,
    tx.Callable,
    tx.Tuple,
    tx.Never,
    tx.Any,
    tx.Union[Dog, Cat],
    T_A,
    Speaks,
    tx.Annotated[Dog, "metadata"],
]
_ILLEGAL_ENDS = [
    HasName,
    Movie,
    tx.Literal[1],
    Hint[int],
    tx.Type[int],
    tx.List[int],
    Box[int],
    tx.Tuple[int],
    tx.Callable[[int], str],
    tx.Union,
    tx.Literal,
    tx.LiteralString,
    tx.TypeGuard[int],
    tx.Optional[tx.List[int]],
    tx.TypeVar("_TL", bound=tx.List[int]),
]


@pytest.mark.parametrize("end", _LEGAL_ENDS, ids=repr)
def test_a_class_like_bound_is_legal_on_a_value(end: tx.Any) -> None:
    assert is_class_hint(end) is True
    for hint in (S[end], B[tx.Never, end]):
        assert value_bound_message(hint) is None
        f = Function("f")
        f.register((hint,))(lambda x: "bounded")
        Signature.from_hints(hint)
        issubhint(hint, object)
        issubhint(E[Dog], hint)


@pytest.mark.parametrize("end", _ILLEGAL_ENDS, ids=repr)
def test_a_bound_read_against_a_value_is_refused(end: tx.Any) -> None:
    assert is_class_hint(end) is False
    # `Super[Type[C]]` and `Super[Hint[C]]` are rewritten to the legal
    # `Type[Super[C]]` and `Hint[Super[C]]`, so only `Between` keeps such an
    # end on a value.
    hints = [B[tx.Never, end]]
    if tx.get_origin(end) not in (type, Hint):
        hints.append(S[end])
    for hint in hints:
        message = value_bound_message(hint)
        assert message is not None
        assert "cannot bound a value with" in message
        with pytest.raises(TypeError, match="cannot bound a value with"):
            dispatch(_fn(hint))
        with pytest.raises(TypeError, match="cannot bound a value with"):
            Function("f").register((hint,))(lambda x: x)
        with pytest.raises(TypeError, match="cannot bound a value with"):
            Signature.from_hints(hint)
        with pytest.raises(TypeError, match="cannot bound a value with"):
            issubhint(hint, object)
        with pytest.raises(TypeError, match="cannot bound a value with"):
            issubhint(tx.Never, hint)
        with pytest.raises(TypeError, match="cannot bound a value with"):
            ishintstance(1, hint)


def test_a_bare_type_end_reads_like_type() -> None:
    # A bare `Type` is the same hint as `type`, so it bounds a value alike.
    for end in (tx.Type, typing.Type):
        assert issubhint(end, type) and issubhint(type, end)
        for value in (int, Dog, object(), Dog(), 1):
            assert ishintstance(value, S[end]) is ishintstance(value, S[type])
            assert ishintstance(value, B[end, object]) is ishintstance(
                value, B[type, object]
            )
        assert issubhint(S[end], S[type]) and issubhint(S[type], S[end])
        assert issubhint(S[object], S[end]) is True
        assert issubhint(S[end], S[object]) is False


def test_a_marked_hint_is_not_class_like() -> None:
    for hint in (S[Dog], B[Dog, Animal], E[Dog], S):
        assert is_class_hint(hint) is False
    # `Exact` can stand in a union written as the bound of a `Super`, where
    # it is compared against a value rather than against the value's class.
    with pytest.raises(TypeError, match="cannot bound a value with"):
        ishintstance(Dog(), S[tx.Union[E[Dog], Cat]])


def test_a_typevar_bounded_by_an_unbounded_form_is_refused() -> None:
    bare = tx.TypeVar("_TBARE", bound=Super)
    with pytest.raises(TypeError, match="Super needs a bound"):
        ishintstance(1, bare)
    with pytest.raises(TypeError, match="Super needs a bound"):
        dispatch(_fn(bare))


def test_the_endpoint_message() -> None:
    assert value_bound_message(S[tx.Literal[1]]) == (
        "Super[Literal[1]] cannot bound a value with Literal[1]: a bound on "
        "a value parameter constrains the value's class, so each bound has "
        "to be a hint that a class can be compared against, such as a "
        "class, a union of classes, Never or Any. Literal[1] is matched "
        "against a value rather than against the value's class. Inside "
        "Type[...] or Hint[...], where the value passed is itself a class "
        "or a hint, any hint can be a bound."
    )
    for hint, advice in [
        (
            B[tx.Type[Dog], tx.Type[Animal]],
            "write Type[Between[Dog, Animal]].",
        ),
        (B[tx.Never, tx.Type[Animal]], "write Type[Between[Never, Animal]]."),
        (B[Hint[bool], Hint[int]], "write Hint[Between[bool, int]]."),
        (B[Hint[bool], tx.Any], "write Hint[Between[bool, Any]]."),
        # A lone `Type` end next to a class does not suggest the rewrite.
        (B[tx.Type[Dog], object], "any hint can be a bound."),
    ]:
        message = value_bound_message(hint)
        assert message is not None and message.endswith(advice)


def test_an_end_is_spelled_as_written_in_the_message() -> None:
    # A `Hint[...]` end reads as written, not by its qualified class name.
    message = value_bound_message(B[Hint[Dog], Hint[Animal]])
    assert message is not None
    assert message.startswith(
        "Between[Hint[Dog], Hint[Animal]] cannot bound a value with "
        "Hint[Dog]: "
    )
    assert message.endswith("write Hint[Between[Dog, Animal]].")
    with pytest.raises(TypeError) as info:
        ishintstance(Dog(), B[Hint[Dog], Hint[Animal]])
    assert str(info.value) == message
    # An `Exact` in a union end reads as `Exact[Dog]`, not as `Annotated`.
    message = value_bound_message(S[tx.Union[E[Dog], Cat]])
    assert message is not None
    spelled = next(
        each
        for each in ("Union[Exact[Dog], Cat]", "Exact[Dog] | Cat")
        if message.startswith(f"Super[{each}] cannot bound a value with ")
    )
    assert f"with {spelled}: " in message
    assert "Annotated" not in message and __name__ not in message


def _forward_dispatch(name: str, later: tx.Any) -> tx.Any:
    # Each call needs its own function name, since `dispatch` joins a function
    # of the same name in the same module, and its own forward reference,
    # since `typing` caches `Between[int, "Later"]` and the `ForwardRef`
    # inside it remembers the first value it resolved to.
    forward = name.capitalize()
    namespace: tx.Dict[str, tx.Any] = {"dispatch": dispatch, "Between": B}
    exec(
        f"@dispatch\ndef {name}(x: Between[int, '{forward}']):\n"
        "    return 'between'",
        namespace,
    )
    namespace[forward] = later
    return namespace[name]


def test_a_forward_reference_end_is_checked_once_it_resolves() -> None:
    assert value_bound_message(B[Dog, "Later"]) is None
    # Bound after the method is defined, the end is read at the first call.
    f = _forward_dispatch("literal", tx.Literal[1])
    with pytest.raises(TypeError, match="cannot bound a value with"):
        f(1)
    f = _forward_dispatch("integral", Integral)
    assert f(1) == "between"


def test_a_bound_is_legal_anywhere_inside_type_or_hint() -> None:
    # The same objects that are refused on a value stand inside `Hint`.
    f = Function("f")
    f.register((Hint[B[tx.Literal[1], int]],))(lambda h: "hint")
    assert f(tx.Literal[1]) == "hint"
    assert f(int) == "hint"
    assert ishintstance(tx.Type[int], Hint[S[tx.Type[bool]]]) is True


# --- the order ---------------------------------------------------------

_ROWS = [
    B[Dog, Animal],
    B[Puppy, Dog],
    B[Puppy, Animal],
    S[Dog],
    S[Puppy],
    E[Dog],
    Dog,
    Animal,
    object,
    tx.Never,
]
_SUPERS = [
    B[Dog, Animal],
    B[Puppy, Dog],
    B[Puppy, Animal],
    S[Dog],
    S[Puppy],
    E[Dog],
    Dog,
    Animal,
    tx.Optional[Animal],
    object,
    tx.Any,
    T_A,
    tx.Never,
]
_ORDER_TABLE = """
TFTTTFFTTTTTF
FTTFTFTTTTTTF
FFTFTFFTTTTTF
FFFTTFFFFTTFF
FFFFTFFFFTTFF
TTTTTTTTTTTTF
FFFFFFTTTTTTF
FFFFFFFTTTTTF
FFFFFFFFFTTFF
TTTTTTTTTTTTT
""".split()
_ORDER = [
    (sub, sup, row[column] == "T")
    for sub, row in zip(_ROWS, _ORDER_TABLE)
    for column, sup in enumerate(_SUPERS)
]


@pytest.mark.parametrize("sub, sup, expected", _ORDER, ids=repr)
def test_issubhint_value_table(
    sub: tx.Any, sup: tx.Any, expected: bool
) -> None:
    assert issubhint(sub, sup) is expected


@pytest.mark.parametrize(
    "sub, sup, expected",
    [
        (tx.Optional[S[Dog]], S[Dog], False),
        (tx.Optional[S[Dog]], tx.Optional[S[Puppy]], True),
        (tx.Optional[S[Dog]], object, True),
        (T_S, S[Puppy], True),
        (Dog, T_S, False),
        (S[Dog], _T, True),
        (B[Dog, tx.Union[Animal, int]], tx.Union[Animal, int], True),
        (B[Dog, Animal], tx.Union[Exact[Dog], Animal], True),
        (S[Dog], tx.Union[Exact[Dog], Animal], False),
        (tx.Union[E[Dog], E[Animal]], S[Dog], True),
        (tx.List[int], B[tx.Never, list], True),
        (tx.List[int], S[list], False),
        (S[Dog], Hint[tx.Any], False),
        (S[Dog], tx.Annotated, False),
        (B[Dog, Animal], tx.Literal[1], False),
        (tx.Any, S[tx.Never], False),
        (S[Dog], E[Dog], False),
        (tx.Type[Dog], B[tx.Never, type], True),
        (tx.Type[Dog], S[type], False),
    ],
    ids=repr,
)
def test_further_rows(sub: tx.Any, sup: tx.Any, expected: bool) -> None:
    assert issubhint(sub, sup) is expected


@pytest.mark.parametrize(
    "sub, sup, expected",
    [
        (tx.Literal[1], S[int], True),
        (tx.Literal[1], S[bool], True),
        (tx.Literal[1], B[bool, int], True),
        (tx.Literal[True], S[int], False),
        (tx.Literal[True], B[bool, int], True),
        (tx.Literal[1, "a"], S[int], False),
        (tx.Literal, S[int], False),
    ],
    ids=repr,
)
def test_literal_rows(sub: tx.Any, sup: tx.Any, expected: bool) -> None:
    assert issubhint(sub, sup) is expected


@pytest.mark.parametrize(
    "a, b",
    [
        (B[tx.Never, Animal], Animal),
        (B[Dog, object], S[Dog]),
        (S[tx.Never], object),
        (B[tx.Never, tx.Never], tx.Never),
        (B[tx.Never, tx.Any], tx.Any),
        (B[tx.Never, tx.Union[Dog, Cat]], tx.Union[Dog, Cat]),
    ],
    ids=repr,
)
def test_equivalences(a: tx.Any, b: tx.Any) -> None:
    assert issubhint(a, b) is True
    assert issubhint(b, a) is True


# A corpus mixing value-level bounds with the hints they are compared
# against: plain classes, an ABC, `Exact`, unions, `Optional`, `TypeVar`s,
# `Literal`s, a method-only protocol, `Any`, `object` and `Never`.
_CORPUS = [
    S[Dog],
    S[Puppy],
    S[Animal],
    S[int],
    S[Integral],
    S[Speaks],
    S[tx.Never],
    S[object],
    S[tx.Any],
    B[Dog, Animal],
    B[Puppy, Dog],
    B[Puppy, Animal],
    B[Dog, Dog],
    B[bool, int],
    B[int, Integral],
    B[tx.Never, int],
    B[tx.Never, Animal],
    B[tx.Never, Speaks],
    B[tx.Never, tx.Never],
    B[tx.Never, tx.Any],
    B[Dog, tx.Union[Animal, int]],
    tx.Optional[S[Dog]],
    tx.Union[S[Dog], str],
    T_S,
    T_A,
    E[Dog],
    E[Animal],
    E[int],
    E[bool],
    Dog,
    Animal,
    Puppy,
    Cat,
    int,
    bool,
    Integral,
    str,
    Speaks,
    tx.Optional[Animal],
    tx.Union[Dog, int],
    tx.Literal[1],
    tx.Literal[True],
    tx.List[int],
    list,
    type(None),
    object,
    tx.Any,
    tx.Never,
]
_VALUES = (
    object(),
    Animal(),
    Dog(),
    Puppy(),
    Cat(),
    1,
    True,
    "a",
    None,
    [1],
    typing.Any,
)


def test_value_order_is_a_preorder() -> None:
    failures = [a for a in _CORPUS if not issubhint(a, a)]
    below = {
        (a, b): issubhint(_CORPUS[a], _CORPUS[b])
        for a, b in itertools.product(range(len(_CORPUS)), repeat=2)
    }
    failures += [
        (_CORPUS[a], _CORPUS[b], _CORPUS[c])
        for a, b, c in itertools.product(range(len(_CORPUS)), repeat=3)
        if below[a, b] and below[b, c] and not below[a, c]
    ]
    assert failures == []


def test_value_consistency() -> None:
    inside = {
        (index, position): ishintstance(value, hint)
        for index, hint in enumerate(_CORPUS)
        for position, value in enumerate(_VALUES)
    }
    failures = [
        (_CORPUS[a], _CORPUS[b], _VALUES[position])
        for a, b in itertools.product(range(len(_CORPUS)), repeat=2)
        if issubhint(_CORPUS[a], _CORPUS[b])
        for position in range(len(_VALUES))
        if inside[a, position] and not inside[b, position]
    ]
    assert failures == []


# --- positions ---------------------------------------------------------


def test_typevar_constraint_is_refused() -> None:
    constrained = tx.TypeVar("_TC", S[Dog], Cat)
    with pytest.raises(TypeError) as info:
        dispatch(_fn(constrained))
    assert str(info.value) == (
        "'x' of f: Super[Dog] cannot be a constraint of a TypeVar: a "
        "constrained variable is solved to the one constraint an argument's "
        "class is below, and a class is never below a bound, so such a "
        'variable would apply to no call. Write TypeVar("T", '
        "bound=Super[Dog]) to bound the variable, or a Union of the "
        "alternatives."
    )
    nested = tx.TypeVar("_TN", tx.Optional[S[Dog]], Cat)
    with pytest.raises(TypeError, match="cannot be a constraint"):
        dispatch(_fn(nested))
    # A constraint that holds no bound is walked like any other hint.
    listed = tx.TypeVar("_TLS", tx.Sequence[S[Dog]], Cat)
    with pytest.raises(TypeError, match="puts a lower bound on argument 1"):
        dispatch(_fn(listed))


_MEMBER = [
    (tx.Type[tx.Union[S[Dog], Cat]], tx.Type[Animal]),
    (tx.Type[tx.TypeVar("_TM", bound=S[Dog])], tx.Type[Animal]),
    (Hint[tx.Optional[B[Dog, Animal]]], Hint[tx.Any]),
]


@pytest.mark.parametrize("hint, other", _MEMBER, ids=repr)
def test_member_positions_inside_type_are_refused(
    hint: tx.Any, other: tx.Any
) -> None:
    needle = "cannot be a member of a union, or the bound of a TypeVar"
    for call in (
        lambda: issubhint(hint, other),
        lambda: issubhint(other, hint),
        lambda: ishintstance(Dog, hint),
        lambda: dispatch(_fn(hint)),
    ):
        with pytest.raises(TypeError, match=needle):
            call()


def test_member_message() -> None:
    with pytest.raises(TypeError) as info:
        ishintstance(Dog, tx.Type[tx.Union[S[Dog], Cat]])
    assert str(info.value) == (
        "Super[Dog] cannot be a member of a union, or the bound of a "
        "TypeVar, inside the argument of Type[...] or Hint[...]: a bound "
        "there has to be the whole argument, because the class or hint "
        "passed is compared against it as a class or hint rather than as a "
        "value. Write Union[Type[Super[Dog]], Type[B]] in place of "
        "Type[Union[Super[Dog], B]], and Type[Super[Dog]] in place of a "
        "Type[T] whose T is bounded by Super[Dog]; the same holds inside "
        "Hint[...]."
    )


def test_a_bound_nested_in_a_type_argument_bound_is_refused() -> None:
    # What `Type[Between[Never, "Later"]]` becomes once `Later` resolves to a
    # hint holding a bound.
    nested = tx.Type[tx.Annotated[tx.Optional[S[Dog]], _Lower(tx.Never)]]
    needle = "Super[Dog] cannot appear inside an Exact, Super or Between form"
    for call in (
        lambda: issubhint(nested, tx.Type[Animal]),
        lambda: ishintstance(Dog, nested),
        lambda: ishintstance(1, tx.Annotated[tx.Optional[S[Dog]], SUPER]),
        lambda: E[tx.Optional[S[Dog]]],
        lambda: E[tx.TypeVar("_TE", bound=S[Dog])],
    ):
        with pytest.raises(TypeError) as info:
            call()
        assert str(info.value).startswith(needle)


class Row(tx.Sequence[S[int]]):
    def __getitem__(self, index: tx.Any) -> tx.Any:
        raise IndexError(index)

    def __len__(self) -> int:
        return 0


_P = tx.ParamSpec("_P")
_SLOTS = [
    (tx.Tuple[S[int]], "cannot be an element of Tuple[...]"),
    (tx.Tuple[int, S[int]], "cannot be an element of Tuple[...]"),
    (
        tx.Callable[[S[int]], None],
        "cannot be a parameter or the return type of Callable[...]",
    ),
    (tx.Callable[[int], S[int]], "cannot be a parameter or the return type"),
    (
        tx.Callable[tx.Concatenate[S[int], _P], None],
        "cannot be a parameter or the return type",
    ),
]


@pytest.mark.parametrize("hint, needle", _SLOTS, ids=repr)
def test_tuple_and_callable_positions_are_refused(
    hint: tx.Any, needle: str
) -> None:
    other = {
        tuple: tx.Tuple[int],
        abc_callable(): tx.Callable[[int], None],
    }[tx.get_origin(hint)]
    for call in (
        lambda: issubhint(hint, other),
        lambda: issubhint(other, hint),
        lambda: issubhint(hint, object),
        lambda: ishintstance(object(), hint),
        lambda: dispatch(_fn(hint)),
    ):
        with pytest.raises(TypeError) as info:
            call()
        assert needle in str(info.value)


def abc_callable() -> tx.Any:
    return tx.get_origin(tx.Callable[[int], None])


def test_a_bound_written_in_a_covariant_base_is_refused() -> None:
    needle = (
        "Row derives from Sequence[Super[int]]: Super[int] puts a lower "
        "bound on argument 1 of Sequence"
    )
    for call in (
        lambda: issubhint(Row, tx.Sequence[int]),
        lambda: ishintstance(Row(), tx.Sequence[int]),
    ):
        with pytest.raises(TypeError) as info:
            call()
        assert str(info.value).startswith(needle)


def test_a_bound_inside_type_inside_a_generic_is_legal() -> None:
    assert issubhint(tx.List[tx.Type[S[int]]], tx.List[tx.Type[int]]) is False
    assert issubhint(tx.List[tx.Type[S[int]]], tx.List[tx.Type[S[int]]])
    f = Function("f")
    f.register((tx.List[tx.Type[S[int]]],))(lambda x: "list")
    assert f([int]) == "list"


def test_unhashable_hints_are_searched_without_the_cache() -> None:
    unhashable = tx.Annotated[int, []]
    assert find_bound(unhashable) is None
    assert holds_bound(tx.Annotated[S[int], []]) is True
    assert issubhint(tx.List[unhashable], tx.List[int]) is True


# --- dispatch ----------------------------------------------------------


def _quiet_register(f: Function, *fns: tx.Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for fn in fns:
            f.register(fn)


def test_dispatch_value_super() -> None:
    f = Function("f")

    def above(x: S[Dog]) -> str:
        return "super"

    def exact(x: E[Dog]) -> str:
        return "exact"

    def puppy(x: Puppy) -> str:
        return "puppy"

    _quiet_register(f, above, exact, puppy)
    assert f(Animal()) == "super"
    assert f(object()) == "super"
    assert f(Dog()) == "exact"
    assert f(Puppy()) == "puppy"
    with pytest.raises(NoMethodError):
        f(Cat())


def test_dispatch_plain_vs_super_warns_and_is_ambiguous() -> None:
    f = Function("f")

    @f.register
    def plain(x: Animal) -> str:
        return "plain"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register
        def above(x: S[Dog]) -> str:
            return "super"

    for value, name in [(Dog(), "Dog"), (Animal(), "Animal")]:
        with pytest.raises(AmbiguousMethodError) as info:
            f(value)
        assert f"f(x: Exact[{name}])" in str(info.value)
    assert f(Puppy()) == "plain"
    assert f(object()) == "super"

    g = Function("g")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g.register(plain, priority=1)
        g.register(above)
    assert g(Dog()) == "plain"
    assert g(object()) == "super"


def test_dog_vs_super_dog_is_ambiguous_for_dog() -> None:
    f = Function("f")

    @f.register
    def plain(x: Dog) -> str:
        return "plain"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register
        def above(x: S[Dog]) -> str:
            return "super"

    with pytest.raises(AmbiguousMethodError):
        f(Dog())
    assert f(Puppy()) == "plain"
    assert f(Animal()) == "super"

    @f.register
    def exact(x: E[Dog]) -> str:
        return "exact"

    assert f(Dog()) == "exact"


def test_a_bound_on_variadic_parameters() -> None:
    f = Function("f")

    @f.register
    def args(*args: S[Dog]) -> str:
        return "args"

    assert f(Animal(), Dog()) == "args"
    with pytest.raises(NoMethodError):
        f(Puppy())

    g = Function("g")

    @g.register
    def kwargs(**kwargs: B[Dog, Animal]) -> str:
        return "kwargs"

    assert g(a=Animal(), b=Dog()) == "kwargs"
    with pytest.raises(NoMethodError):
        g(a=Puppy())


def test_a_bound_among_several_parameters() -> None:
    def plain(x: Animal, y: int) -> str:
        return "plain"

    # `str` and `int` never overlap, so the methods cannot both match.
    def text(x: S[Dog], y: str) -> str:
        return "text"

    _quiet_register(Function("f"), plain, text)

    def flag(x: S[Dog], y: bool) -> str:
        return "flag"

    f = Function("f")
    f.register(plain)
    with pytest.warns(RuntimeWarning, match="ambiguous"):
        f.register(flag)
    with pytest.raises(AmbiguousMethodError) as info:
        f(Dog(), True)
    assert "f(x: Exact[Dog], y: bool)" in str(info.value)

    def optional(x: Animal, y: tx.Optional[S[Dog]]) -> str:
        return "optional"

    def reversed_(x: S[Dog], y: Animal) -> str:
        return "reversed"

    g = Function("g")
    g.register(optional)
    with pytest.warns(RuntimeWarning, match="ambiguous"):
        g.register(reversed_)
    with pytest.raises(AmbiguousMethodError) as info:
        g(Dog(), Dog())
    assert "g(x: Exact[Dog], y: Exact[Dog])" in str(info.value)


def test_a_constrained_typevar_as_an_end() -> None:
    constrained = tx.TypeVar("_TC", Animal, str)
    # A class is above the variable only when it is above every constraint.
    hint = S[constrained]
    assert ishintstance(object(), hint) is True
    for value in (Animal(), Dog(), "text", 1):
        assert ishintstance(value, hint) is False
    assert issubhint(B[Dog, Animal], constrained) is True
    assert issubhint(S[Dog], constrained) is False


def test_between_below_plain_wins() -> None:
    f = Function("f")

    def plain(x: Animal) -> str:
        return "plain"

    def between(x: B[Dog, Animal]) -> str:
        return "between"

    _quiet_register(f, plain, between)
    assert f(Dog()) == "between"
    assert f(Animal()) == "between"
    assert f(Puppy()) == "plain"
    assert f(Cat()) == "plain"


def test_mro_tiebreak_ignores_bounds() -> None:
    assert mro_index(S[Dog], Dog) is None
    assert mro_index(B[Dog, Animal], Puppy) is None
    assert mro_index(tx.Optional[S[Dog]], Dog) is None
    assert mro_index(Animal, Dog) == 1


def test_candidates_and_bestcandidates_with_bounds() -> None:
    f = Function("f")

    @f.register
    def plain(x: Animal) -> str:
        return "plain"

    with pytest.warns(RuntimeWarning):

        @f.register
        def above(x: S[Dog]) -> str:
            return "super"

    @f.register
    def between(x: B[Dog, Animal]) -> str:
        return "between"

    best = f.bestcandidates(Dog())
    assert [m.function for m in best] == [between]
    assert [m.function for m in f.candidates(Dog())][0] is between
    assert {m.function for m in f.candidates(Dog())} == {plain, above, between}
    assert [m.function for m in f.bestcandidates(object())] == [above]
    assert [m.function for m in f.bestcandidates(Puppy())] == [plain]
    assert f.candidates(1) == ()


def test_resolve_with_exact_query_reaches_a_bounded_method() -> None:
    f = Function("f")

    @f.register
    def above(x: S[Dog]) -> str:
        return "super"

    assert f.resolve(E[Animal]).function is above
    assert f.resolve(B[Dog, Animal]).function is above
    # A plain class query holds the instances of its subclasses, which a
    # lower bound does not accept.
    with pytest.raises(NoMethodError):
        f.resolve(Animal)
    registry = {S[Dog]: 1, Animal: 2}
    assert resolve_hint(E[Dog], registry, ambiguity="ignore") in (1, 2)
    assert resolve_hint(B[Puppy, Puppy], registry, default=0) == 2
    assert resolve_hint(S[Animal], registry) == 1


def test_bound_as_a_hint_value() -> None:
    assert ishintstance(S[int], Hint[tx.Any]) is True
    assert ishintstance(S[int], Hint[int]) is False
    assert ishintstance(S[int], Hint[S[int]]) is False
    assert ishintstance(B[Dog, Animal], Hint[Animal]) is True


def test_a_hint_value_is_checked_for_a_bound_at_its_top_level() -> None:
    # A bound at the top of a hint passed as a value is always refused.
    for sup in (Hint[tx.Any], Hint[object]):
        with pytest.raises(TypeError, match="cannot bound a value with"):
            ishintstance(S[tx.Literal[1]], sup)
    # A nested one is found only when the comparison reaches it, which a
    # comparison with `Any` never does.
    for nested in (
        tx.Optional[S[tx.Literal[1]]],
        tx.Sequence[S[int]],
        tx.Type[tx.Union[S[int], str]],
    ):
        assert ishintstance(nested, Hint[tx.Any]) is True
    with pytest.raises(TypeError, match="cannot bound a value with"):
        ishintstance(tx.Optional[S[tx.Literal[1]]], Hint[object])
    with pytest.raises(TypeError, match="puts a lower bound on argument 1"):
        ishintstance(tx.Sequence[S[int]], Hint[object])
    # A bound that a type argument can hold is no error at all.
    assert ishintstance(tx.List[S[int]], Hint[object]) is True


def test_value_bound_renders() -> None:
    f = Function("f")
    f.register((S[Dog],))(lambda x: None)
    f.register((B[Dog, Animal], tx.Optional[S[int]]))(lambda x, y: None)
    text = "\n".join(method.describe() for method in f.methods)
    assert "Super[Dog]" in text
    assert "Between[Dog, Animal]" in text
    assert "Super[int]" in text


def test_call_cache_keys_by_type_only() -> None:
    class Pet(abc.ABC):  # noqa: B024 -- register, not subclass
        pass

    class Stray:
        pass

    f = Function("f")

    @f.register
    def above(x: S[Stray]) -> str:
        return "super"

    @f.register
    def anything(x: object) -> str:
        return "object"

    assert f.dispatch(Stray()) is f.dispatch(Stray())
    assert f(Pet()) == "object"
    # Registering `Stray` under `Pet` puts `Pet` above it, which the cached
    # selection must not survive.
    Pet.register(Stray)
    assert f(Pet()) == "super"


# --- overlap -----------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (Dog, S[Dog], True),
        (Animal, S[Dog], True),
        (Puppy, S[Dog], False),
        (Dog, B[Dog, Animal], True),
        (Cat, B[Dog, Animal], False),
        (tx.Optional[S[Dog]], Animal, True),
        (E[Dog], S[Dog], False),
        (int, S[Dog], False),
        # A documented miss: a `Literal` names no class to try.
        (tx.Literal[1], S[int], False),
        (tx.Type[Dog], S[Dog], False),
        (tx.TypeVar("_TO", bound=Animal), S[Dog], True),
        (tx.Annotated[Animal, "metadata"], S[Dog], True),
        # A bound built on a union is one interval, not its members.
        (B[Dog, tx.Union[Animal, int]], Dog, True),
        (B[Puppy, tx.Union[Dog, int]], Cat, False),
    ],
    ids=repr,
)
def test_overlaps_value_rows(a: tx.Any, b: tx.Any, expected: bool) -> None:
    assert overlaps(a, b) is expected
    assert overlaps(b, a) is expected
