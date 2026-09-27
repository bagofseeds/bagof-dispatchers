"""Declared parametrisations: base substitution and value dispatch (V5 of #50).

Three layers, each with its own section:

* **hint level** -- a sub-hint whose origin differs from the super-hint's is
  re-expressed through the bases its class was written with (`class
  IntBox(Box[int])` is `Box[int]`), then compared slot by slot with the
  super-hint's variance;
* **value level** -- `ishintstance(v, G[args])` reads what the value declares,
  its `__orig_class__` (`Box[int]()`) and then its class's bases (`class
  Child(List[int])`), and stays shallow when it declares nothing;
* **cache** -- a position whose hint is a parametrised user generic keys on
  the value's type *and* recorded parametrisation, never on the instance, and
  every other position keeps its type-only key.

Every class is built twice, once from `typing` and once from
`typing_extensions`, since the two `TypeVar`s differ on some versions. No
`X | Y` or `list[int]` outside a version guard: this runs on 3.8.
"""

# stdlib
import collections
import collections.abc
import sys
import types
import typing
import warnings

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._errors import AmbiguousMethodError
from bagof.dispatchers._function import (
    Function,
    _call_key,
    _Plan,
    _SameObject,
)
from bagof.dispatchers._lattice import (
    is_declaration_dependent,
    is_value_dependent,
)
from bagof.dispatchers.core import ishintstance, issubhint
from bagof.dispatchers.core._relation import (
    _as_base_args,
    _is_fully_declared,
)

# --- the families, in both spellings -----------------------------------


def _family(module: tx.Any) -> types.SimpleNamespace:
    """The V5 classes, with `Generic`/`TypeVar` taken from `module`."""
    T = module.TypeVar("T")
    U = module.TypeVar("U")
    A = module.TypeVar("A")
    B = module.TypeVar("B")
    T_co = module.TypeVar("T_co", covariant=True)

    class Box(module.Generic[T]):
        """Invariant: an unflagged `TypeVar`."""

    class Src(module.Generic[T_co]):
        """Covariant."""

    class IntBox(Box[int]):
        """Declares `Box[int]` through its base."""

    class Leaf(IntBox):
        """Declares nothing itself; inherits `Box[int]` through `IntBox`."""

    class Sub(Box[T]):
        """Passes its argument on to `Box`."""

    class Plain(Sub):
        """A non-generic subclass of `Sub` written bare: `T` stays open."""

    class Pair(module.Generic[A, B]):
        """Two invariant parameters."""

    class Flip(Pair[B, A], module.Generic[A, B]):
        """`Flip[X, Y]` is `Pair[Y, X]`."""

    class FlipImplicit(Pair[B, A]):
        """Its parameters are `(B, A)`, in the order the base lists them."""

    class WideBox(Box[T], module.Generic[T, U]):
        """Takes a parameter `Box` does not."""

    class Mixin:
        """A plain class among a generic's bases."""

    class Mixed(Mixin, Box[T]):
        """A generic with a plain base beside its parametrised one."""

    class Fixed(Box[int], module.Generic[T]):
        """A generic whose `Box` base is already concrete."""

    class Child(module.List[int]):
        """A `list` declaring `List[int]` through its base."""

    class OpenChild(module.List[T]):
        """A `list` whose base leaves `T` open."""

    class SrcInt(Src[int]):
        """Declares the covariant `Src[int]`."""

    return types.SimpleNamespace(**locals())


FAMILIES = [
    pytest.param(_family(typing), id="typing"),
    pytest.param(_family(tx), id="typing_extensions"),
]


# --- hint level: base-parameter substitution ---------------------------


def _hint_rows(
    f: types.SimpleNamespace,
) -> tx.List[tx.Tuple[tx.Any, tx.Any, bool]]:
    return [
        (f.IntBox, f.Box[int], True),
        (f.IntBox, f.Box[str], False),
        (f.IntBox, f.Box[bool], False),
        (f.Leaf, f.Box[int], True),
        (f.Leaf, f.Box[str], False),
        (f.Sub[bool], f.Box[bool], True),
        # `Box` is invariant, so `Sub[bool]` is not a `Box[int]`.
        (f.Sub[bool], f.Box[int], False),
        (f.Sub[int], f.Box[f.T], True),
        (f.Flip[int, str], f.Pair[str, int], True),
        (f.Flip[int, str], f.Pair[int, str], False),
        # Without the explicit `Generic[A, B]`, `FlipImplicit` takes `(B, A)`:
        # its first argument fills `B`, so it lands in the first slot.
        (f.FlipImplicit[int, str], f.Pair[int, str], True),
        (f.FlipImplicit[int, str], f.Pair[str, int], False),
        (f.WideBox[int, str], f.Box[int], True),
        (f.WideBox[int, str], f.Box[str], False),
        (f.Mixed[int], f.Box[int], True),
        (f.Fixed[str], f.Box[int], True),
        (f.Fixed[str], f.Box[str], False),
        (f.SrcInt, f.Src[int], True),
        # `Src` is covariant: `SrcInt` is a `Src[object]`, not a `Src[bool]`.
        (f.SrcInt, f.Src[object], True),
        (f.SrcInt, f.Src[bool], False),
        (f.Child, tx.List[int], True),
        (f.Child, tx.List[float], False),
        # `list` is invariant, `Sequence` covariant.
        (f.Child, tx.List[object], False),
        (f.Child, tx.Sequence[object], True),
        (f.Child, tx.Sequence[int], True),
        (f.Child, tx.Sequence[str], False),
        (f.Child, tx.Iterable[int], True),
        (f.Child, list, True),
        # Written bare, a generic class declares nothing: as a bare `Box` is
        # not a `Box[int]`, neither is a bare `Sub`, nor a class extending one.
        (f.Sub, f.Box[int], False),
        (f.Sub, f.Box[f.T], False),
        (f.Plain, f.Box[int], False),
        (f.OpenChild, tx.List[int], False),
        # The reverse direction is untouched: a base is not a subclass.
        (f.Box[int], f.IntBox, False),
        (tx.List[int], f.Child, False),
    ]


@pytest.mark.parametrize("f", FAMILIES)
def test_base_substitution_truth_table(f: types.SimpleNamespace) -> None:
    for sub, sup, expected in _hint_rows(f):
        assert issubhint(sub, sup) is expected, (sub, sup)


@pytest.mark.parametrize("f", FAMILIES)
def test_as_base_args_reads_what_a_class_declares(
    f: types.SimpleNamespace,
) -> None:
    assert _as_base_args(f.IntBox, f.Box) == (int,)
    assert _as_base_args(f.Flip[int, str], f.Pair) == (str, int)
    assert _as_base_args(f.Child, list) == (int,)
    assert _as_base_args(f.Child, collections.abc.Sequence) == (int,)
    # The origin itself: its own arguments, or nothing when it has none.
    assert _as_base_args(f.Box[int], f.Box) == (int,)
    assert _as_base_args(f.Box, f.Box) is None
    assert _as_base_args(list, list) is None
    # A generic class written without arguments leaves its bases open.
    assert _as_base_args(f.Sub, f.Box) is None


def test_runtime_stdlib_subclass_falls_back_positionally() -> None:
    """A `Counter` records no parametrised base: nothing maps onto `dict`."""
    assert _as_base_args(collections.Counter, dict) is None
    assert issubhint(collections.Counter, tx.Dict[str, int]) is False
    # A parametrised stdlib origin is read positionally, as it always was,
    # and only against a target taking as many arguments.
    assert _as_base_args(tx.DefaultDict[str, int], dict) == (str, int)
    assert _as_base_args(tx.Dict[str, int], collections.abc.Iterable) is None
    assert _as_base_args(tx.Tuple[int, int], collections.abc.Sequence) is None
    assert issubhint(tx.DefaultDict[str, int], tx.Dict[str, int]) is True
    assert issubhint(tx.List[int], tx.Sequence[int]) is True


def test_a_base_that_refuses_its_arguments_never_raises() -> None:
    """A substitution that raises drops that base; the walk goes on."""
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    class Refusing(typing._GenericAlias, _root=True):  # type: ignore
        """`Box[T]`, but refusing to have `T` filled in."""

        def __getitem__(self, params: tx.Any) -> tx.Any:
            raise TypeError("refused")

    class Sub(Box[T]):
        pass

    Sub.__orig_bases__ = (Refusing(Box, (T,)),)
    assert _as_base_args(Sub[int], Box) is None
    # Nothing maps, so the arguments are compared positionally, as before.
    assert issubhint(Sub[int], Box[int]) is True
    assert issubhint(Sub[int], Box[str]) is False


def test_a_paramspec_generic_is_not_substituted() -> None:
    """A `ParamSpec` generic's arguments do not pair one to one."""
    P = tx.ParamSpec("P")

    class Hook(tx.Generic[P]):
        pass

    class SubHook(Hook[P]):
        pass

    # Before 3.10 the backport refuses a parameter list here; one type is
    # the spelling it accepts.
    params = [int] if sys.version_info >= (3, 10) else int
    assert _as_base_args(SubHook[params], Hook) is None


def test_same_origin_tuple_callable_type_are_unchanged() -> None:
    """The dedicated paths never see the base walk."""
    assert issubhint(tx.Tuple[bool], tx.Tuple[int]) is True
    assert issubhint(tx.Tuple[int, ...], tx.Tuple[int]) is False
    assert issubhint(tx.Callable[[int], bool], tx.Callable[[bool], int])
    assert issubhint(tx.Type[bool], tx.Type[int]) is True
    assert issubhint(tx.List[bool], tx.List[int]) is False

    class Point(tx.Tuple[int, int]):
        pass

    # A tuple subclass keeps the tuple path's answer.
    assert issubhint(Point, tx.Tuple[int, int]) is False
    assert issubhint(Point, tuple) is True


# --- value level: the declared parametrisation -------------------------


@pytest.mark.parametrize("f", FAMILIES)
def test_value_reads_class_declared_arguments(
    f: types.SimpleNamespace,
) -> None:
    assert ishintstance(f.Child(), tx.List[int]) is True
    assert ishintstance(f.Child(), tx.List[float]) is False
    assert ishintstance(f.Child(), tx.Sequence[int]) is True
    assert ishintstance(f.Child(), tx.Sequence[str]) is False
    assert ishintstance(f.IntBox(), f.Box[int]) is True
    assert ishintstance(f.IntBox(), f.Box[str]) is False
    assert ishintstance(f.Leaf(), f.Box[int]) is True
    assert ishintstance(f.SrcInt(), f.Src[object]) is True
    assert ishintstance(f.SrcInt(), f.Src[bool]) is False


@pytest.mark.parametrize("f", FAMILIES)
def test_value_reads_instance_declared_arguments(
    f: types.SimpleNamespace,
) -> None:
    assert ishintstance(f.Box[int](), f.Box[int]) is True
    assert ishintstance(f.Box[int](), f.Box[str]) is False
    assert ishintstance(f.Sub[bool](), f.Box[bool]) is True
    assert ishintstance(f.Sub[bool](), f.Box[int]) is False
    assert ishintstance(f.Flip[int, str](), f.Pair[str, int]) is True
    assert ishintstance(f.Flip[int, str](), f.Pair[int, str]) is False
    # A free `TypeVar` in the hint is the invariant slot's top.
    assert ishintstance(f.Box[int](), f.Box[f.T]) is True
    # Through the unions and `TypeVar`s the value check walks.
    assert ishintstance(f.Box[str](), tx.Optional[f.Box[int]]) is False
    assert ishintstance(f.Box[int](), tx.Optional[f.Box[int]]) is True
    bounded = tx.TypeVar("bounded", bound=f.Box[int])
    assert ishintstance(f.Box[str](), bounded) is False


@pytest.mark.parametrize("f", FAMILIES)
def test_value_that_declares_nothing_stays_shallow(
    f: types.SimpleNamespace,
) -> None:
    assert ishintstance([1], tx.List[str]) is True
    assert ishintstance(f.Box(), f.Box[int]) is True
    assert ishintstance(f.Box(), f.Box[str]) is True
    # A base that leaves `T` open declares nothing.
    assert ishintstance(f.OpenChild(), tx.List[str]) is True
    assert ishintstance(f.Plain(), f.Box[str]) is True
    # `Any` says nothing either -- anywhere inside the arguments.
    assert ishintstance(f.Box[tx.Any](), f.Box[int]) is True
    assert ishintstance(f.Box[tx.List[tx.Any]](), f.Box[tx.List[int]])
    # A bare generic hint asks for the class alone.
    assert ishintstance(f.Sub[str](), f.Box) is True


def test_a_standard_library_hint_never_reads_the_record() -> None:
    """An instance's record is read only against a user-generic hint.

    That is what lets a `Sequence[int]` position keep its type-only cache
    key: two instances of one class then always match it alike. What the
    class declares through its bases is still read.
    """
    T = tx.TypeVar("T")

    class Row(tx.Sequence[T]):
        def __getitem__(self, index: tx.Any) -> tx.Any:
            raise IndexError(index)

        def __len__(self) -> int:
            return 0

    class IntRow(Row[int]):
        pass

    assert ishintstance(Row[int](), tx.Sequence[str]) is True
    assert ishintstance(IntRow(), tx.Sequence[str]) is False
    assert ishintstance(IntRow(), tx.Sequence[int]) is True


def test_callable_arguments_are_read_through() -> None:
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    handler = tx.Callable[[int], str]
    assert ishintstance(Box[handler](), Box[handler]) is True
    assert ishintstance(Box[handler](), Box[tx.Callable[[str], str]]) is False
    assert ishintstance(
        Box[tx.Callable[[], str]](), Box[tx.Callable[[], str]]
    )
    # A `TypeVar` inside the parameter list leaves it undeclared.
    assert ishintstance(Box[tx.Callable[[T], str]](), Box[handler]) is True
    assert _is_fully_declared((tx.Literal[1],)) is True
    assert _is_fully_declared((tx.ParamSpec("P"),)) is False
    assert _is_fully_declared(()) is False
    assert _is_fully_declared(None) is False


def test_slotted_generic_falls_back_to_shallow() -> None:
    """`__slots__` without `__dict__` leaves nowhere to record `Box[int]`."""
    T = tx.TypeVar("T")

    class Slotted(tx.Generic[T]):
        __slots__ = ()

    value = Slotted[int]()
    assert not hasattr(value, "__orig_class__")
    assert ishintstance(value, Slotted[str]) is True


def test_untrustworthy_records_are_ignored() -> None:
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    class Other(tx.Generic[T]):
        pass

    class Loud(Box[int]):
        def __getattr__(self, name: str) -> tx.Any:
            raise RuntimeError(name)

    bare = Box()
    bare.__orig_class__ = Box  # no arguments: says nothing
    assert ishintstance(bare, Box[str]) is True
    stray = Box()
    stray.__orig_class__ = Other[int]  # not a class `stray` is an instance of
    assert ishintstance(stray, Box[str]) is True
    # A raising `__getattr__` reads as no record; the class still declares.
    assert ishintstance(Loud(), Box[int]) is True
    assert ishintstance(Loud(), Box[str]) is False


def test_subclasses_stay_shallow_where_arity_differs() -> None:
    class Table(tx.Dict[str, int]):
        pass

    class Point(tx.Tuple[int, int]):
        pass

    assert ishintstance(Table(), tx.Mapping[str, int]) is True
    assert ishintstance(Table(), tx.Mapping[bytes, int]) is False
    # `Dict[K, V]` does not line up with `Iterable[T]`: shallow.
    assert ishintstance(Table(), tx.Iterable[bytes]) is True
    # A tuple keeps its own shape check, and does not line up with
    # `Sequence[T]`: shallow.
    assert ishintstance(Point((1, 2)), tx.Tuple[str, str]) is True
    assert ishintstance(Point((1, 2)), tx.Sequence[str]) is True


@pytest.mark.skipif(sys.version_info < (3, 9), reason="PEP 585 is 3.9+")
def test_pep585_spellings() -> None:
    child_585 = types.new_class("Child585", (list[int],))  # type: ignore
    assert issubhint(child_585, list[int]) is True  # type: ignore
    assert issubhint(child_585, list[float]) is False  # type: ignore
    assert ishintstance(child_585(), tx.List[float]) is False
    # `list[int]([1])` is a plain list: it declares nothing.
    assert ishintstance(list[int]([1]), list[str]) is True  # type: ignore


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 is 3.12+")
def test_pep695_generic() -> None:
    namespace = {}  # type: tx.Dict[str, tx.Any]
    exec(  # noqa: S102 -- the syntax does not parse before 3.12
        "class Box[T]: pass\nclass IntBox(Box[int]): pass\n", namespace
    )
    box, int_box = namespace["Box"], namespace["IntBox"]
    assert issubhint(int_box, box[int]) is True
    assert issubhint(int_box, box[str]) is False
    f = Function("f")
    f.register((box[int],))(lambda b: "int")
    f.register((box[str],))(lambda b: "str")
    assert f(box[int]()) == "int"
    assert f(box[str]()) == "str"
    assert f(int_box()) == "int"


# --- dispatch: the owner's acceptance ----------------------------------


@pytest.mark.parametrize("f", FAMILIES)
def test_child_list_dispatches_on_its_declared_argument(
    f: types.SimpleNamespace,
) -> None:
    handle = Function("handle")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        handle.register((tx.List[int],))(lambda xs: "ints")
        handle.register((tx.List[float],))(lambda xs: "floats")
    assert handle.ambiguities() == []
    assert handle(f.Child()) == "ints"
    with pytest.raises(AmbiguousMethodError):
        handle([1])


@pytest.mark.parametrize("f", FAMILIES)
def test_box_instances_dispatch_on_their_recorded_argument(
    f: types.SimpleNamespace,
) -> None:
    unbox = Function("unbox")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unbox.register((f.Box[int],))(lambda b: "int")
        unbox.register((f.Box[str],))(lambda b: "str")
    assert unbox.ambiguities() == []
    assert unbox(f.Box[int]()) == "int"
    assert unbox(f.Box[str]()) == "str"
    assert unbox(f.IntBox()) == "int"
    with pytest.raises(AmbiguousMethodError):
        unbox(f.Box())


@pytest.mark.parametrize("f", FAMILIES)
def test_declared_subclass_is_more_specific(f: types.SimpleNamespace) -> None:
    """`IntBox <= Box[int]` makes an `IntBox` overload win over `Box[int]`."""
    g = Function("g")
    g.register((f.Box[int],))(lambda b: "box")
    g.register((f.IntBox,))(lambda b: "intbox")
    assert g(f.IntBox()) == "intbox"
    assert g(f.Box[int]()) == "box"


# --- cache: keyed on the declared parametrisation, not the instance ----


def test_declaration_dependence_truth_table() -> None:
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    dependent = [
        Box[int],
        tx.Optional[Box[int]],
        tx.Annotated[Box[int], "m"],
        tx.TypeVar("TB", bound=Box[int]),
    ]
    independent = [
        Box,
        int,
        tx.List[int],
        tx.Sequence[int],
        tx.Optional[tx.List[int]],
        Exact[int],
        tx.TypeVar("TI", bound=int),
    ]
    for hint in dependent:
        assert is_declaration_dependent(hint) is True, hint
    for hint in independent:
        assert is_declaration_dependent(hint) is False, hint


def _plan_of(f: Function, shape: tx.Any) -> _Plan:
    return f._cache.shape_plans[shape]


def test_instances_of_one_parametrisation_share_an_entry() -> None:
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    f = Function("f")
    f.register((Box[int],))(lambda b: "int")
    f.register((Box[str],))(lambda b: "str")
    first, second = Box[int](), Box[int]()
    assert f(first) == "int"
    assert f(second) == "int"
    assert len(f._cache.call_cache) == 1
    plan = _plan_of(f, ((1, ())))
    assert plan.declared == frozenset({0})
    assert _call_key((first,), {}, plan) == _call_key((second,), {}, plan)


def test_a_new_parametrisation_is_never_served_a_stale_method() -> None:
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    f = Function("f")
    f.register((Box[int],))(lambda b: "int")
    f.register((Box[str],))(lambda b: "str")
    for _ in range(2):
        assert f(Box[int]()) == "int"
        assert f(Box[str]()) == "str"
    # Keyword arguments key the same way.
    g = Function("g")
    g.register(lambda *, b: "any")

    @g.register({"b": Box[int]})
    def _int(*, b: tx.Any) -> str:
        return "int"

    assert g(b=Box[int]()) == "int"
    assert g(b=Box[str]()) == "any"
    assert g(b=Box[int]()) == "int"


def test_a_standard_library_position_keeps_its_type_only_key() -> None:
    f = Function("f")
    f.register((tx.List[int],))(lambda xs: "ints")
    f.register((tx.Tuple[str, ...],))(lambda xs: "strs")
    assert f([1]) == "ints"
    plan = _plan_of(f, (1, ()))
    assert plan.declared == frozenset()
    assert plan.value_dependent == frozenset()
    assert _call_key(([1],), {}, plan) == (1, list)


def test_full_value_key_subsumes_the_declared_one() -> None:
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    hint = tx.Union[tx.Literal[1], Box[int]]
    assert is_value_dependent(hint) and is_declaration_dependent(hint)
    f = Function("f")
    f.register((hint,))(lambda x: "one")
    assert f(1) == "one"
    plan = _plan_of(f, (1, ()))
    assert plan.value_dependent == frozenset({0})
    assert plan.declared == frozenset()


def test_declared_key_survives_a_raising_getattr() -> None:
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    class Loud(Box[int]):
        def __getattr__(self, name: str) -> tx.Any:
            raise RuntimeError(name)

    f = Function("f")
    f.register((Box[int],))(lambda b: "int")
    assert f(Loud()) == "int"
    assert f(Loud()) == "int"
    key = _call_key((Loud(),), {}, _plan_of(f, (1, ())))
    assert key == (1, (Loud, None))


def test_same_object_compares_by_identity() -> None:
    record = tx.List[int]
    assert _SameObject(record) == _SameObject(record)
    assert hash(_SameObject(record)) == hash(_SameObject(record))
    assert _SameObject(record) != _SameObject(tx.List[str])
    assert _SameObject(record).__eq__(record) is NotImplemented
