"""Tests for `Super[C]`, a lower bound on a value, a class or a hint."""

# stdlib
import itertools
import sys
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
    SuperHint,
    SuperType,
    dispatch,
)
from bagof.dispatchers._signature import _render_hint
from bagof.dispatchers.core import (
    ishintstance,
    issubhint,
    normalise_hint,
    resolve_hint,
)
from bagof.dispatchers.core._bounds import bounds_of
from bagof.dispatchers.core._exact import is_exact
from bagof.dispatchers.core._super import (
    SUPER,
    bare_super_message,
    is_bare_super,
    is_super,
    super_target,
)


class Animal:
    pass


class Dog(Animal):
    pass


class Puppy(Dog):
    pass


_T = tx.TypeVar("_T")


def _ts(c: tx.Any) -> tx.Any:
    return tx.Type[Super[c]]


def _te(c: tx.Any) -> tx.Any:
    return tx.Type[Exact[c]]


def _hs(c: tx.Any) -> tx.Any:
    return Hint[Super[c]]


def _he(c: tx.Any) -> tx.Any:
    return Hint[Exact[c]]


# --- construction ------------------------------------------------------


def test_super_spelling_and_detection() -> None:
    assert Super[int] == tx.Annotated[int, SUPER]
    assert is_super(Super[int]) is True
    assert is_super(int) is False
    assert is_super(tx.Annotated[int, "meta"]) is False
    assert super_target(Super[int]) is int
    # The two markers never read as each other.
    assert is_exact(Super[int]) is False
    assert is_super(Exact[int]) is False
    assert repr(SUPER) == "SUPER"
    assert SUPER is type(SUPER)()


def test_super_of_bare_special_form_constructs() -> None:
    # A bare special form must build even where `typing._type_check` refuses
    # it as an `Annotated` argument; the alias fallback handles it.
    for form in (tx.Union, tx.Literal, tx.Annotated):
        hint = Super[form]
        assert is_super(hint) is True
        assert super_target(hint) is form


def test_super_of_none() -> None:
    assert super_target(Super[None]) is type(None)


def test_super_takes_only_a_hint() -> None:
    with pytest.raises(TypeError, match=r"Super\[\.\.\.\] takes a type hint"):
        Super[1]


def test_super_is_idempotent() -> None:
    hint = Super[int]
    assert Super[hint] is hint
    assert Super[Super[int]] == Super[int]


@pytest.mark.parametrize(
    "build",
    [
        lambda: Super[Exact[int]],
        lambda: Exact[Super[int]],
        lambda: SuperType[Exact[int]],
        lambda: SuperHint[Exact[int]],
        # `Annotated` flattens, so the marker is found under extra metadata.
        lambda: Exact[tx.Annotated[Super[int], "meta"]],
    ],
)
def test_exact_and_super_cannot_combine(build: tx.Any) -> None:
    with pytest.raises(TypeError) as info:
        build()
    message = str(info.value)
    assert "cannot be combined" in message
    assert "Type[Exact[C]]" in message
    assert "Type[Super[C]]" in message


@pytest.mark.parametrize(
    "build, outer",
    [
        (lambda: Super[Between[Dog, Animal]], "Super"),
        (lambda: SuperType[Between[Dog, Animal]], "Super"),
        (lambda: SuperHint[Between[Dog, Animal]], "Super"),
        (lambda: Exact[Between[Dog, Animal]], "Exact"),
    ],
)
def test_super_and_exact_cannot_take_between(
    build: tx.Callable[[], tx.Any], outer: str
) -> None:
    with pytest.raises(TypeError) as info:
        build()
    message = str(info.value)
    assert message.startswith(f"{outer}[...] cannot take Between[Dog, Animal]")
    assert "cannot be nested" in message
    assert "Type[Between[L, U]]" in message


def test_combination_message_names_the_inner_form() -> None:
    with pytest.raises(TypeError, match=r"^Super\[\.\.\.\] cannot take Exact"):
        Super[Exact[int]]
    with pytest.raises(TypeError, match=r"^Exact\[\.\.\.\] cannot take Super"):
        Exact[Super[int]]


def test_aliases_expand() -> None:
    assert SuperType[int] == tx.Type[Super[int]]
    assert SuperHint[int] == Hint[Super[int]]
    with pytest.raises(TypeError, match=r"SuperType\[\.\.\.\] takes"):
        SuperType[1]
    with pytest.raises(TypeError, match=r"SuperHint\[\.\.\.\] takes"):
        SuperHint[1]


def test_outer_super_normalises_to_inner() -> None:
    assert normalise_hint(Super[tx.Type[int]]) == tx.Type[Super[int]]
    assert normalise_hint(Super[Hint[int]]) == Hint[Super[int]]
    # A bare target names no bound, so it stays as written, to be refused.
    assert normalise_hint(Super[tx.Type]) == Super[tx.Type]
    assert normalise_hint(Super[int]) == Super[int]
    with pytest.raises(TypeError, match="cannot be combined"):
        normalise_hint(Super[tx.Type[Exact[int]]])
    with pytest.raises(TypeError, match="cannot be combined"):
        normalise_hint(Exact[tx.Type[Super[int]]])


def test_bounds_of_reads_each_argument_as_an_interval() -> None:
    assert bounds_of(Super[int], object) == (int, object)
    assert bounds_of(Super[int], tx.Any) == (int, tx.Any)
    assert bounds_of(Exact[int], object) == (int, int)
    assert bounds_of(int, object) == (tx.Never, int)


# --- where a lower bound can stand ------------------------------------


def test_is_bare_super() -> None:
    for hint in (Super, SuperType, SuperHint):
        assert is_bare_super(hint) is True
    for hint in (
        int,
        Super[int],
        tx.Type[Super[int]],
        Hint[Super[int]],
        Exact[int],
    ):
        assert is_bare_super(hint) is False


def test_bare_super_messages() -> None:
    assert bare_super_message(Super) == (
        "Super needs a bound: write Super[C] to accept a value whose class "
        "is C or a class above it, Type[Super[C]] to accept the class C or "
        "any class above it, or Hint[Super[C]] to accept the hint C or any "
        "hint above it."
    )
    assert bare_super_message(SuperType).startswith("SuperType needs a bound")
    assert bare_super_message(SuperHint).startswith("SuperHint needs a bound")


@pytest.mark.parametrize("hint", [Super, SuperType, SuperHint], ids=repr)
def test_bare_super_is_refused_by_the_relation(hint: tx.Any) -> None:
    with pytest.raises(TypeError, match=r"needs a bound"):
        issubhint(hint, int)
    with pytest.raises(TypeError, match=r"needs a bound"):
        issubhint(int, hint)
    with pytest.raises(TypeError, match=r"needs a bound"):
        ishintstance(1, hint)


_ON_A_VALUE = [
    Super[int],
    tx.Optional[Super[int]],
    tx.TypeVar("_TSUPER", bound=Super[int]),
]


@pytest.mark.parametrize("hint", _ON_A_VALUE, ids=repr)
def test_a_lower_bound_on_a_value_is_read_by_the_relation(
    hint: tx.Any,
) -> None:
    assert ishintstance(1, hint) is True
    assert ishintstance(object(), hint) is True
    assert ishintstance(True, hint) is False
    assert issubhint(hint, int) is False
    assert issubhint(hint, object) is True
    assert issubhint(Exact[int], hint) is True
    assert issubhint(int, hint) is False


def test_super_in_an_invariant_argument_is_refused_by_the_relation(
) -> None:
    needle = r"Super\[int\] is not supported as a type argument of list"
    with pytest.raises(TypeError, match=needle):
        issubhint(tx.List[Super[int]], tx.List[int])
    with pytest.raises(TypeError, match=needle):
        issubhint(tx.List[int], tx.List[Super[int]])
    with pytest.raises(TypeError, match=needle):
        ishintstance([1], tx.List[Super[int]])


def test_super_below_a_union_or_typevar_inside_type_is_refused() -> None:
    # Only the whole argument of `Type` may be a lower bound: one reached
    # through a union member or a `TypeVar` bound is refused, whichever side
    # of the comparison carries the other lower bound.
    bounded = tx.TypeVar("bounded", bound=Super[Dog])
    needle = r"Super\[Dog\] cannot be a member of a union"
    with pytest.raises(TypeError, match=needle):
        issubhint(tx.Type[tx.Union[Super[Dog], int]], _ts(Animal))
    with pytest.raises(TypeError, match=needle):
        issubhint(tx.Type[bounded], _ts(Animal))
    with pytest.raises(TypeError, match=needle):
        issubhint(_ts(Animal), tx.Type[tx.Union[Super[Dog], int]])
    with pytest.raises(TypeError, match=needle):
        ishintstance(Dog, tx.Type[tx.Union[Super[Dog], int]])
    with pytest.raises(TypeError, match=needle):
        ishintstance(int, Hint[tx.Optional[Super[Dog]]])


def test_a_typevar_below_a_lower_bound_is_read_by_its_bound() -> None:
    # A `TypeVar` sub-argument stands for its bound: a free or ordinarily
    # bounded one is never above a class, and one bounded by `Exact[C]` is
    # the single class `C`.
    exact = tx.TypeVar("exact", bound=Exact[Dog])
    ordinary = tx.TypeVar("ordinary", bound=Dog)
    assert issubhint(tx.Type[_T], _ts(Dog)) is False
    assert issubhint(tx.Type[ordinary], _ts(Dog)) is False
    assert issubhint(tx.Type[exact], _ts(Dog)) is True
    assert issubhint(tx.Type[exact], _ts(Puppy)) is True
    assert issubhint(tx.Type[exact], _ts(Animal)) is False


def test_a_union_below_a_lower_bound_distributes() -> None:
    both = tx.Type[tx.Union[Exact[Dog], Exact[Animal]]]
    assert issubhint(both, _ts(Dog)) is True
    assert issubhint(both, _ts(Animal)) is False
    assert issubhint(tx.Type[tx.Union[Dog, Exact[Animal]]], _ts(Dog)) is False


def test_super_of_a_union_is_one_interval() -> None:
    above = tx.Type[Super[tx.Union[Dog, int]]]
    assert issubhint(above, _ts(Dog)) is True
    assert issubhint(_ts(Dog), above) is False
    assert ishintstance(object, above) is True
    assert ishintstance(Dog, above) is False


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
        (SuperType, "SuperType needs a bound"),
        (
            tx.Type[tx.Union[Super[int], str]],
            "Super[int] cannot be a member of a union",
        ),
        (
            tx.List[Super[int]],
            "Super[int] is not supported as a type argument of list[...]",
        ),
        (
            tx.Callable[[Super[int]], int],
            "Super[int] cannot be a parameter or the return type of Callable",
        ),
        (tx.Tuple[Super[int]], "Super[int] cannot be an element of Tuple"),
        (
            tx.TypeVar("_TCON", Super[int], str),
            "Super[int] cannot be a constraint of a TypeVar",
        ),
        (
            tx.Type[Super[tx.List[Super[int]]]],
            "Super[int] is not supported as a type argument of list[...]",
        ),
        (
            # What `Super["Later"]` becomes once `Later` resolves to a hint
            # holding a bound: the resolved hint is never built by `Super`.
            tx.Annotated[tx.Optional[Super[int]], SUPER],
            "Super[int] cannot appear inside an Exact, Super or Between form",
        ),
        (
            Super[tx.Literal[1]],
            "Super[Literal[1]] cannot bound a value with Literal[1]",
        ),
        (tx.Type[Super], "Super needs a bound"),
    ],
    ids=repr,
)
def test_misplaced_super_is_refused_at_registration(
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
def test_super_on_a_value_is_accepted_at_registration(hint: tx.Any) -> None:
    f = Function("f")
    f.register((hint,))(lambda x: "above int")
    assert f(1) == "above int"
    assert f(object()) == "above int"
    with pytest.raises(NoMethodError):
        f(True)


@pytest.mark.parametrize("kind", ["args", "kwargs"])
def test_super_is_accepted_on_a_catch_all(kind: str) -> None:
    f = dispatch(_fn(Super[int], kind))
    if kind == "args":
        assert f(1, object()) is None
        with pytest.raises(NoMethodError):
            f(1, True)
    else:
        assert f(a=1, b=object()) is None
        with pytest.raises(NoMethodError):
            f(a=True)


def test_super_is_accepted_by_from_hints() -> None:
    sig = Signature.from_hints(Super[int], scale=Super[int])
    assert sig.dispatched_names == ("scale",)


@pytest.mark.parametrize(
    "source",
    [
        "def f(x: 'Later'): pass",
        "def f(*args: 'Later'): pass",
        "def f(**kw: 'Later'): pass",
    ],
)
def test_a_forward_reference_to_a_lower_bound_is_checked_once_resolved(
    source: str,
) -> None:
    namespace: tx.Dict[str, tx.Any] = {}
    exec(source, namespace)
    sig = Signature.from_callable(namespace["f"])
    assert sig._deferred
    namespace["Later"] = Super[int]
    sig._settle()
    assert not sig._deferred

    namespace = {}
    exec(source, namespace)
    sig = Signature.from_callable(namespace["f"])
    namespace["Later"] = tx.List[Super[int]]
    with pytest.raises(TypeError, match=r"not supported as a type argument"):
        sig._settle()


def test_super_inside_type_or_hint_is_accepted_at_registration() -> None:
    f = Function("f")

    @f.register((tx.Optional[tx.Type[Super[int]]],))
    def _types(x: object) -> str:
        return "type"

    @f.register((Hint[Super[int]], tx.Callable[[int], tx.Any]))
    def _hints(x: object, y: object) -> str:
        return "hint"

    # A forward reference inside a hint is left for later, not refused.
    @f.register((tx.List["Later"], tx.Literal["a"], tx.Literal[1]))
    def _later(x: object, y: object, z: object) -> str:
        return "later"

    assert f(None) == "type"
    assert f(object) == "type"


# --- value level -------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (Animal, True),
        (object, True),
        (Dog, True),
        (Puppy, False),
        (int, False),
        (Dog(), False),
        (tx.Any, False),  # a class from 3.11, but not below `object`
    ],
    ids=repr,
)
def test_value_level_type_super(value: tx.Any, expected: bool) -> None:
    assert ishintstance(value, tx.Type[Super[Dog]]) is expected
    assert ishintstance(value, SuperType[Dog]) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (int, True),
        (object, True),
        (tx.Any, True),
        (_T, True),
        (tx.Union[bool, str], True),
        (tx.Literal[True], False),
        (bool, True),
        (str, False),
        (1, False),
        ("Foo", False),
    ],
    ids=repr,
)
def test_value_level_hint_super(value: tx.Any, expected: bool) -> None:
    assert ishintstance(value, Hint[Super[bool]]) is expected
    assert ishintstance(value, SuperHint[bool]) is expected


def test_value_level_outer_super_is_read_as_inner() -> None:
    assert ishintstance(Animal, Super[tx.Type[Dog]]) is True
    assert ishintstance(Puppy, Super[tx.Type[Dog]]) is False


def test_super_degenerate_targets() -> None:
    # Nothing is above `Any` among classes; only `object` is above `object`.
    for value in (object, int, type):
        assert ishintstance(value, tx.Type[Super[tx.Any]]) is False
    assert ishintstance(object, tx.Type[Super[object]]) is True
    assert ishintstance(int, tx.Type[Super[object]]) is False
    # Among hints, only the tops are above `Any`.
    assert ishintstance(tx.Any, Hint[Super[tx.Any]]) is True
    assert ishintstance(_T, Hint[Super[tx.Any]]) is True
    assert ishintstance(int, Hint[Super[tx.Any]]) is False
    # Every class is above the bottom.
    for value in (object, int, Dog):
        assert ishintstance(value, tx.Type[Super[tx.Never]]) is True


# --- hint level --------------------------------------------------------

_U_TYPE = tx.Union[_ts(Dog), str]

_TYPE_COLUMNS = [
    _ts(Dog),
    _ts(Animal),
    _te(Dog),
    _te(Animal),
    tx.Type[Dog],
    tx.Type[Animal],
    tx.Type,
    tx.Type[tx.Any],
    tx.Any,
    object,
    _T,
    _U_TYPE,
]

# Rows are sub-hints, columns super-hints, in the order of `_TYPE_COLUMNS`.
_TYPE_TABLE = [
    "T F F F F F T T T T T T",
    "T T F F F F T T T T T T",
    "T F T F T T T T T T T T",
    "T T F T F T T T T T T T",
    "F F F F T T T T T T T F",
    "F F F F F T T T T T T F",
    "F F F F F F T F T T T F",
    "F F F F F F T T T T T F",
    "F F F F F F F F T F T F",
    "F F F F F F F F T T T F",
    "F F F F F F F F T F T F",
    "F F F F F F F F T T T T",
]

_TYPE_CASES = [
    (sub, sup, cell == "T")
    for sub, row in zip(_TYPE_COLUMNS, _TYPE_TABLE)
    for sup, cell in zip(_TYPE_COLUMNS, row.split())
]


@pytest.mark.parametrize("sub, sup, expected", _TYPE_CASES)
def test_issubhint_type_super_table(
    sub: tx.Any, sup: tx.Any, expected: bool
) -> None:
    assert issubhint(sub, sup) is expected


_U_HINT = tx.Union[_hs(bool), str]

_HINT_ROWS = [
    _hs(bool),
    _hs(int),
    _he(bool),
    _he(int),
    Hint[bool],
    Hint[int],
    Hint,
]

_HINT_COLUMNS = _HINT_ROWS + [tx.Any, object, _T, _U_HINT]

_HINT_TABLE = [
    "T F F F F F T T F T T",
    "T T F F F F T T F T T",
    "T F T F T T T T F T T",
    "T T F T F T T T F T T",
    "F F F F T T T T F T F",
    "F F F F F T T T F T F",
    "F F F F F F T T F T F",
]

_HINT_CASES = [
    (sub, sup, cell == "T")
    for sub, row in zip(_HINT_ROWS, _HINT_TABLE)
    for sup, cell in zip(_HINT_COLUMNS, row.split())
]


@pytest.mark.parametrize("sub, sup, expected", _HINT_CASES)
def test_issubhint_hint_super_table(
    sub: tx.Any, sup: tx.Any, expected: bool
) -> None:
    assert issubhint(sub, sup) is expected


_TYPE_CORPUS = _TYPE_COLUMNS[:10]
_HINT_CORPUS = _HINT_ROWS + [tx.Any]


@pytest.mark.parametrize("corpus", [_TYPE_CORPUS, _HINT_CORPUS])
def test_super_order_is_a_preorder_with_exact_and_plain(
    corpus: tx.List[tx.Any],
) -> None:
    for hint in corpus:
        assert issubhint(hint, hint) is True
    for a, b, c in itertools.product(corpus, repeat=3):
        if issubhint(a, b) and issubhint(b, c):
            assert issubhint(a, c), (a, b, c)


def test_value_consistency_type_super() -> None:
    values = (object, Animal, Dog, Puppy, int)
    for a, b in itertools.product(_TYPE_CORPUS, repeat=2):
        if not issubhint(a, b):
            continue
        for value in values:
            if ishintstance(value, a):
                assert ishintstance(value, b), (value, a, b)


def test_value_consistency_hint_super() -> None:
    values = (object, int, bool, str, tx.Any, tx.Union[bool, str])
    for a, b in itertools.product(_HINT_CORPUS, repeat=2):
        if not issubhint(a, b):
            continue
        for value in values:
            if ishintstance(value, a):
                assert ishintstance(value, b), (value, a, b)


def test_a_bottom_argument_is_below_every_lower_bound() -> None:
    # `Type[Never]` holds no class, so it is below every `Type` form.
    assert issubhint(tx.Type[tx.Never], _ts(Dog)) is True
    assert issubhint(_ts(Dog), tx.Type[tx.Never]) is False


def test_a_bottom_hint_argument_is_ordered_by_its_interval() -> None:
    # The hint `Never` is itself a value of `Hint[Never]`, so the intervals
    # decide: `Hint[Never]` is below `Hint[Super[C]]` only when `C` is too.
    assert issubhint(Hint[tx.Never], _hs(bool)) is False
    assert issubhint(Hint[tx.Never], _hs(tx.Never)) is True
    assert ishintstance(tx.Never, Hint[tx.Never]) is True
    assert ishintstance(tx.Never, _hs(bool)) is False
    assert ishintstance(tx.Never, _hs(tx.Never)) is True


def test_super_forms_are_ordered_against_a_bare_type() -> None:
    assert issubhint(_ts(Dog), type) is True
    assert issubhint(type, _ts(Dog)) is False


# --- dispatch ----------------------------------------------------------


def test_dispatch_type_super() -> None:
    f = Function("f")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        @f.register((tx.Type[Super[Dog]],))
        def _above(cls: object) -> str:
            return "super"

        @f.register((tx.Type[Exact[Dog]],))
        def _exact(cls: object) -> str:
            return "exact"

        @f.register((tx.Type[Puppy],))
        def _puppy(cls: object) -> str:
            return "puppy"

    assert f(Animal) == "super"
    assert f(object) == "super"
    assert f(Dog) == "exact"  # `Exact[Dog]` beats `Super[Dog]`
    assert f(Puppy) == "puppy"
    with pytest.raises(NoMethodError):
        f(int)


def test_dispatch_plain_vs_super_is_ambiguous_and_exact_resolves_it() -> None:
    f = Function("f")

    @f.register((tx.Type[Animal],))
    def _plain(cls: object) -> str:
        return "plain"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register((tx.Type[Super[Dog]],))
        def _above(cls: object) -> str:
            return "super"

    with pytest.raises(AmbiguousMethodError):
        f(Dog)
    assert f(Puppy) == "plain"
    assert f(object) == "super"

    @f.register((tx.Type[Exact[Dog]],))
    def _exact(cls: object) -> str:
        return "exact"

    assert f(Dog) == "exact"
    with pytest.raises(AmbiguousMethodError):
        f(Animal)

    g = Function("g")

    @g.register((tx.Type[Animal],))
    def _g_plain(cls: object) -> str:
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        @g.register((tx.Type[Super[Dog]],), priority=1)
        def _g_above(cls: object) -> str:
            return "super"

    assert g(Dog) == "super"
    assert g(Animal) == "super"
    assert g(Puppy) == "plain"


def test_a_second_argument_keeps_the_overlap_meaningful() -> None:
    # The overlap at the class argument combines with an ordered second
    # argument: `(Dog, True)` fits both methods and neither is more specific.
    f = Function("f")

    @f.register((tx.Type[Animal], int))
    def _plain(cls: object, n: object) -> str:
        return "plain"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register((tx.Type[Super[Dog]], bool))
        def _above(cls: object, n: object) -> str:
            return "super"

    with pytest.raises(AmbiguousMethodError):
        f(Dog, True)

    # Without a shared value at the class argument there is nothing to warn.
    g = Function("g")

    @g.register((tx.Type[Puppy], int))
    def _g_plain(cls: object, n: object) -> str:
        return "plain"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        @g.register((tx.Type[Super[Dog]], bool))
        def _g_above(cls: object, n: object) -> str:
            return "super"


def test_an_optional_lower_bound_is_ambiguous_with_a_plain_type() -> None:
    f = Function("f")

    @f.register((tx.Type[Animal],))
    def _plain(cls: object) -> str:
        return "plain"

    with pytest.warns(RuntimeWarning, match="ambiguous"):

        @f.register((tx.Optional[tx.Type[Super[Dog]]],))
        def _above(cls: object) -> str:
            return "super"

    with pytest.raises(AmbiguousMethodError) as caught:
        f(Dog)
    spelled = "Type[Super[Dog]]"
    optional = _render_hint(tx.Optional[int]).replace("int", spelled)
    assert optional in str(caught.value)
    assert f(None) == "super"


def test_a_lower_bound_is_a_hint_value() -> None:
    f = Function("f")

    @f.register((Hint[tx.Any],))
    def _any(h: object) -> str:
        return "any"

    assert f(Super[int]) == "any"
    # The values of `Super[int]` are those whose class is above `int`, so it
    # is not below `int`, whose values include `True`, and `int` is not below
    # it either.
    assert ishintstance(Super[int], Hint[int]) is False
    assert ishintstance(Super[int], Hint[Super[int]]) is False


def test_a_forward_reference_to_a_lower_bound_settles_later() -> None:
    namespace: tx.Dict[str, tx.Any] = {"Type": tx.Type, "Super": Super}
    exec("def f(cls: 'Type[Super[Later]]'): return 'super'", namespace)
    f = Function("f")
    f.register(namespace["f"])
    exec("class Later: pass\nclass Sooner(Later): pass", namespace)
    later, sooner = namespace["Later"], namespace["Sooner"]
    assert f(later) == "super"
    assert f(object) == "super"
    with pytest.raises(NoMethodError):
        f(sooner)


def test_dispatch_hint_super() -> None:
    f = Function("f")

    @f.register((Hint[Super[bool]],))
    def _above(h: object) -> str:
        return "super"

    @f.register((Hint[Exact[bool]],))
    def _exact(h: object) -> str:
        return "exact"

    @f.register((Hint[tx.Any],))
    def _any(h: object) -> str:
        return "any"

    assert f(int) == "super"
    assert f(object) == "super"
    assert f(tx.Any) == "super"
    assert f(bool) == "exact"
    assert f(str) == "any"


def test_dispatch_with_the_aliases() -> None:
    @dispatch
    def describe(cls: SuperType[Dog]) -> str:
        return "an ancestor of Dog"

    @dispatch
    def describe(cls: tx.Type[Puppy]) -> str:  # noqa: F811
        return "a kind of puppy"

    assert describe(Animal) == "an ancestor of Dog"
    assert describe(Puppy) == "a kind of puppy"


def test_super_renders() -> None:
    f = Function("f")

    @f.register((tx.Type[Super[int]], Hint[Super[bool]]))
    def _above(cls: object, h: object) -> None: ...

    (method,) = f.methods
    text = method.describe()
    assert "Type[Super[int]]" in text
    assert "Hint[Super[bool]]" in text


@pytest.mark.parametrize("marker", ["Super", "Exact"])
def test_a_marked_form_renders_inside_a_union(marker: str) -> None:
    form = {"Super": Super, "Exact": Exact}[marker]
    spelled = f"Type[{marker}[Dog]]"
    optional = _render_hint(tx.Optional[int]).replace("int", spelled)
    union = _render_hint(tx.Union[int, str]).replace("int", spelled)
    assert _render_hint(tx.Optional[tx.Type[form[Dog]]]) == optional
    assert _render_hint(tx.Union[tx.Type[form[Dog]], str]) == union
    f = Function("f")

    @f.register((tx.Optional[tx.Type[form[Dog]]],))
    def _above(cls: object) -> None: ...

    (method,) = f.methods
    assert optional in method.describe()


@pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 unions")
def test_a_pep_604_union_renders_its_members() -> None:
    union = eval("type[Super[Dog]] | None", {"Super": Super, "Dog": Dog})
    assert _render_hint(union) == "Type[Super[Dog]] | NoneType"


# --- registries --------------------------------------------------------


def test_resolve_hint_super_key() -> None:
    registry = {tx.Type[Super[Dog]]: 1, tx.Type[Puppy]: 2}
    assert resolve_hint(tx.Type[Exact[Animal]], registry) == 1
    assert resolve_hint(tx.Type[Super[Animal]], registry) == 1
    assert resolve_hint(tx.Type[Puppy], registry) == 2
    assert resolve_hint(tx.Type[Animal], registry, default=0) == 0


@pytest.mark.parametrize("query", [SuperType, Super])
def test_resolve_hint_bare_super_query_raises(query: tx.Any) -> None:
    with pytest.raises(TypeError, match=r"needs a bound"):
        resolve_hint(query, {int: 1, query: 2})
