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

A last section covers classes written against a PEP 585 alias (`class
GL(list[T])`, #60) through all three layers, built only on 3.9+.
"""

# stdlib
import collections
import collections.abc
import dataclasses
import gc
import sys
import types
import typing
import warnings
import weakref

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
    _record_key,
    _SameObject,
)
from bagof.dispatchers._lattice import (
    is_declaration_dependent,
    is_value_dependent,
)
from bagof.dispatchers.core import ishintstance, issubhint
from bagof.dispatchers.core._introspect import (
    _PEP585_ALIAS,
    _class_parameters,
    _generic_variances,
    _is_pep585_alias,
    _memoised_variances,
)
from bagof.dispatchers.core._relation import (
    _RECORDER_REFS,
    _RECORDERS,
    _as_base_args,
    _is_fully_declared,
    _may_record_parametrisation,
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
    assert _as_base_args(f.IntBox, f.Box) == ((int,),)
    assert _as_base_args(f.Flip[int, str], f.Pair) == ((str, int),)
    assert _as_base_args(f.Child, list) == ((int,),)
    assert _as_base_args(f.Child, collections.abc.Sequence) == ((int,),)
    # The origin itself: its own arguments, or nothing when it has none.
    assert _as_base_args(f.Box[int], f.Box) == ((int,),)
    assert _as_base_args(f.Box, f.Box) == ()
    assert _as_base_args(list, list) == ()
    # A generic class written without arguments leaves its bases open.
    assert _as_base_args(f.Sub, f.Box) == ()


def test_runtime_stdlib_subclass_falls_back_positionally() -> None:
    """A `Counter` records no parametrised base: nothing maps onto `dict`."""
    assert _as_base_args(collections.Counter, dict) == ()
    assert issubhint(collections.Counter, tx.Dict[str, int]) is False
    # A parametrised stdlib origin is read positionally, as it always was,
    # and only against a target taking as many arguments.
    assert _as_base_args(tx.DefaultDict[str, int], dict) == ((str, int),)
    assert _as_base_args(tx.Dict[str, int], collections.abc.Iterable) == ()
    assert _as_base_args(tx.Tuple[int, int], collections.abc.Sequence) == ()
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
    assert _as_base_args(Sub[int], Box) == ()
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
    assert _as_base_args(SubHook[params], Hook) == ()


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


def test_a_standard_library_hint_reads_the_record_too() -> None:
    """A `Generic` instance's record is read against a stdlib hint as well.

    An earlier cut read the record only against a user-generic hint (so
    `Row[int]()` matched `Sequence[str]`); the owner chose to read it against
    any parametrised class hint, gated on the value being a `Generic`
    instance, with the cache keyed to match (#50, V5). What the class
    declares through its bases is read as before.
    """
    T = tx.TypeVar("T")

    class Row(tx.Sequence[T]):
        def __getitem__(self, index: tx.Any) -> tx.Any:
            raise IndexError(index)

        def __len__(self) -> int:
            return 0

    class IntRow(Row[int]):
        pass

    class GenericList(tx.List[T]):
        pass

    assert ishintstance(Row[int](), tx.Sequence[str]) is False
    assert ishintstance(Row[int](), tx.Sequence[int]) is True
    assert ishintstance(Row[int](), tx.Iterable[object]) is True
    # A `Generic` subclass of `list` declares through its record too.
    assert ishintstance(GenericList[int](), tx.List[str]) is False
    assert ishintstance(GenericList[int](), tx.List[int]) is True
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
        # A stdlib generic too: a `Generic` instance's record is read against
        # it (#50, V5 -- owner decision).
        tx.List[int],
        tx.Sequence[int],
        tx.Optional[tx.List[int]],
    ]
    independent = [
        Box,
        int,
        list,
        Exact[int],
        tx.TypeVar("TI", bound=int),
        # The shapes and forms with checks of their own.
        tx.Tuple[int],
        tx.Callable[[int], str],
        tx.Type[int],
        tx.Literal[1],
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


def test_a_plain_value_at_a_standard_library_position_is_not_probed() -> None:
    """A `List[int]` position is declaration-dependent (#50, V5).

    A plain list is not a `Generic` instance, so it is never asked for a
    record and keys as its type and `None`; a tuple-only position keeps the
    bare type key.
    """
    f = Function("f")
    f.register((tx.List[int],))(lambda xs: "ints")
    f.register((tx.Tuple[str, ...],))(lambda xs: "strs")
    assert f([1]) == "ints"
    plan = _plan_of(f, (1, ()))
    assert plan.declared == frozenset({0})
    assert plan.value_dependent == frozenset()
    assert _call_key(([1],), {}, plan) == (1, (list, None))
    g = Function("g")
    g.register((tx.Tuple[str, ...],))(lambda xs: "strs")
    assert g(("a",)) == "strs"
    assert _call_key((("a",),), {}, _plan_of(g, (1, ()))) == (1, tuple)


def test_a_position_that_is_both_keys_on_value_and_record() -> None:
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
    assert plan.declared == frozenset({0})


def test_value_equality_does_not_stand_in_for_the_record() -> None:
    """A dataclass generic's `==` ignores `__orig_class__`: key on both."""
    T = tx.TypeVar("T")

    @dataclasses.dataclass(unsafe_hash=True)
    class DBox(tx.Generic[T]):
        v: tx.Any

    f = Function("f")
    f.register((tx.Literal["auto"],))(lambda x: "auto")
    f.register((DBox[int],))(lambda x: "int")
    f.register((DBox[str],))(lambda x: "str")
    assert DBox[int](1) == DBox[str](1)
    assert f(DBox[int](1)) == "int"
    assert f(DBox[str](1)) == "str"
    assert f("auto") == "auto"
    # The keyword spelling keys the same way.
    g = Function("g")
    g.register({"b": tx.Literal["auto"]})(lambda *, b: "auto")
    g.register({"b": DBox[int]})(lambda *, b: "int")
    g.register({"b": DBox[str]})(lambda *, b: "str")
    assert g(b=DBox[int](1)) == "int"
    assert g(b=DBox[str](1)) == "str"


def test_only_generic_instances_are_asked_for_a_record() -> None:
    """The key probes no value the value check would not read."""
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    probed = []  # type: tx.List[str]

    class Proxy:
        def __getattr__(self, name: str) -> tx.Any:
            probed.append(name)
            raise AttributeError(name)

    f = Function("f")
    f.register((tx.Union[Box[int], str, Proxy],))(lambda x: "ok")
    assert f("s") == "ok"
    assert f(Proxy()) == "ok"
    assert "__orig_class__" not in probed
    plan = _plan_of(f, (1, ()))
    assert plan.declared == frozenset({0})
    assert _call_key(("s",), {}, plan) == (1, (str, None))


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


# --- review follow-ups: shapes that must stay shallow or tighten -------


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="a ParamSpec list argument is 3.10+"
)
def test_a_paramspec_generic_value_stays_shallow() -> None:
    P = tx.ParamSpec("P")

    class Hook(tx.Generic[P]):
        pass

    value = Hook[[int]]()
    assert ishintstance(value, Hook[[int]]) is True
    assert ishintstance(value, Hook[[str]]) is True
    hook = Function("hook")
    hook.register((Hook[[int]],))(lambda h: "hook")
    assert hook(value) == "hook"


def test_an_unresolved_name_declares_nothing() -> None:
    """`Box["int"]()` records a `ForwardRef`, which says nothing yet."""
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    value = Box["int"]()
    assert ishintstance(value, Box[int]) is True
    assert ishintstance(value, Box[str]) is True
    f = Function("f")
    f.register((Box[int],))(lambda b: "int")
    assert f(value) == "int"
    assert _is_fully_declared(("int",)) is False


def test_a_frozen_dataclass_generic_records_nothing() -> None:
    T = tx.TypeVar("T")

    @dataclasses.dataclass(frozen=True)
    class Frozen(tx.Generic[T]):
        v: int = 0

    value = Frozen[int]()
    assert not hasattr(value, "__orig_class__")
    assert ishintstance(value, Frozen[str]) is True


def test_self_inside_init_declares_nothing_yet() -> None:
    """The record is written only after `__init__` returns."""
    T = tx.TypeVar("T")
    seen = []  # type: tx.List[bool]

    class Box(tx.Generic[T]):
        def __init__(self) -> None:
            seen.append(ishintstance(self, Box[str]))

    value = Box[int]()
    assert seen == [True]
    assert ishintstance(value, Box[str]) is False


def test_invariant_positions_tighten_declared_values() -> None:
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    class Strs(tx.List[str]):
        pass

    value = Box[int]()
    assert ishintstance(value, Box[object]) is False
    assert ishintstance(value, Box[tx.Union[int, str]]) is False
    assert ishintstance(value, Box[tx.Optional[int]]) is False
    assert ishintstance(Strs(), tx.List[object]) is False
    assert ishintstance(Strs(), tx.List[tx.Any]) is True
    assert ishintstance(Strs(), list) is True
    assert ishintstance(Strs(), tx.Sequence[object]) is True


def test_a_diamond_is_every_base_it_reaches() -> None:
    """An ill-typed diamond is below each parametrisation it declares."""
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    class Ints(Box[int]):
        pass

    class Strs(Box[str]):
        pass

    class Both(Ints, Strs):
        pass

    assert _as_base_args(Both, Box) == ((int,), (str,))
    assert issubhint(Both, Box[int]) is True
    assert issubhint(Both, Box[str]) is True
    assert issubhint(Both, Box[bytes]) is False
    assert ishintstance(Both(), Box[str]) is True
    assert ishintstance(Both(), Box[bytes]) is False


def test_a_diamond_through_a_generic_is_walked_per_parametrisation() -> None:
    """A class reached twice with other arguments is walked each time, so
    `Both <= Strs <= Mid[str] <= Box[str]` gives `Both <= Box[str]`."""
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    class Mid(Box[T]):
        pass

    class Ints(Mid[int]):
        pass

    class Strs(Mid[str]):
        pass

    class Both(Ints, Strs):
        pass

    assert _as_base_args(Both, Box) == ((int,), (str,))
    assert issubhint(Both, Mid[str]) is True
    assert issubhint(Mid[str], Box[str]) is True
    assert issubhint(Both, Box[str]) is True
    assert issubhint(Both, Box[bytes]) is False


# --- owner round 2: solved TypeVar slots, stdlib hints read the record -


def test_a_declared_value_solves_a_typevar_slot() -> None:
    """An invariant `TypeVar` slot is solved, as a type checker does."""
    import numbers

    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    to_object = tx.TypeVar("to_object", bound=object)
    int_or_str = tx.TypeVar("int_or_str", int, str)
    real = tx.TypeVar("real", bound=numbers.Real)
    to_float = tx.TypeVar("to_float", bound=float)
    value = Box[int]()
    assert ishintstance(value, Box[to_object]) is True
    assert ishintstance(value, Box[int_or_str]) is True
    assert ishintstance(value, Box[real]) is True
    # No numeric tower: `int` is not below `float`.
    assert ishintstance(value, Box[to_float]) is False
    assert ishintstance(Box[bytes](), Box[int_or_str]) is False
    f = Function("f")
    f.register((Box[int_or_str],))(lambda b: "int or str")
    f.register((Box[to_object],))(lambda b: "anything")
    assert f(Box[int]()) == "int or str"
    assert f(Box[bytes]()) == "anything"


def test_a_row_dispatches_on_its_recorded_argument() -> None:
    T = tx.TypeVar("T")

    class Row(tx.Sequence[T]):
        def __getitem__(self, index: tx.Any) -> tx.Any:
            raise IndexError(index)

        def __len__(self) -> int:
            return 0

    h = Function("h")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h.register((tx.Sequence[int],))(lambda s: "ints")
        h.register((tx.Sequence[str],))(lambda s: "strs")
    assert h(Row[int]()) == "ints"
    for _ in range(2):
        assert h(Row[str]()) == "strs"
        assert h(Row[int]()) == "ints"
    with pytest.raises(AmbiguousMethodError):
        h([1])


# --- PEP 585 generic subclasses (#60) ----------------------------------

# `class GL(list[T])` has no `Generic` in its MRO and lists no
# `__parameters__`, yet `GL[int]()` records `GL[int]`. Every class below is
# built inside `_pep585_family`, which only a 3.9+ test calls, so this module
# still imports on 3.8.
_PEP585 = pytest.mark.skipif(
    sys.version_info < (3, 9), reason="PEP 585 is 3.9+"
)


def _pep585_family() -> types.SimpleNamespace:
    """Classes written against PEP 585 aliases, and their neighbours."""
    T = tx.TypeVar("T")
    U = tx.TypeVar("U")
    K = tx.TypeVar("K")
    T_co = tx.TypeVar("T_co", covariant=True)

    class GL(list[T]):  # type: ignore[misc]
        """`T` fills `list`'s invariant slot."""

    class GD(dict[str, T]):  # type: ignore[misc]
        """Fills one of `dict`'s two parameters."""

    class GM(collections.abc.Mapping[K, T]):  # type: ignore[misc]
        """A key-invariant, value-covariant ABC, both slots open."""

        def __getitem__(self, key: tx.Any) -> tx.Any:
            raise KeyError(key)

        def __iter__(self) -> tx.Iterator[tx.Any]:
            return iter(())

        def __len__(self) -> int:
            return 0

    class Mixed(list[T], tx.Generic[T]):  # type: ignore[misc]
        """A `Generic` beside the PEP 585 base: worked before #60."""

    class Sub(GL[int]):  # type: ignore[misc]
        """Not generic: declares `GL[int]`, hence `list[int]`, by its base."""

    class Sub2(GL[T]):  # type: ignore[misc]
        """Passes its argument on to `GL`."""

    class Leaf(GL):  # type: ignore[misc]
        """Written bare: takes no parameters, declares nothing."""

    class Two(list[T], collections.abc.Container[U]):  # type: ignore[misc]
        """Parameters in order of first appearance across the bases."""

    class Twice(list[T], collections.abc.Container[T]):  # type: ignore[misc]
        """One parameter, mentioned by two bases."""

    class Cov(list[T_co]):  # type: ignore[misc]
        """A covariant `TypeVar` in an invariant slot: a checker flags it."""

    class Slotted(list[T]):  # type: ignore[misc]
        """No `__dict__`: nowhere to record `__orig_class__`."""

        __slots__ = ()

    @dataclasses.dataclass(frozen=True)
    class Frozen(GL[T]):  # type: ignore[misc]
        """A frozen subclass: refuses `__orig_class__`."""

    return types.SimpleNamespace(**locals())


@pytest.fixture
def pep() -> types.SimpleNamespace:
    return _pep585_family()


def test_pep585_helpers_on_every_version() -> None:
    """The new helpers import and answer on 3.8 too, where nothing is 585."""
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    assert _class_parameters(Box) == (T,)
    assert _class_parameters(int) == ()
    assert _is_pep585_alias(tx.List[int]) is False
    assert _may_record_parametrisation(Box) is True
    assert _may_record_parametrisation(list) is False
    assert _may_record_parametrisation(int) is False
    assert _record_key(tx.List[int]) == _SameObject(tx.List[int])
    # Not a class: no parameters to read.
    assert _generic_variances(tx.Union) is None
    if sys.version_info < (3, 9):
        assert _PEP585_ALIAS is None


@_PEP585
def test_pep585_parameters_and_variances(pep: types.SimpleNamespace) -> None:
    assert _class_parameters(pep.GL) == (pep.T,)
    assert _class_parameters(pep.GD) == (pep.T,)
    assert _class_parameters(pep.GM) == (pep.K, pep.T)
    assert _class_parameters(pep.Sub2) == (pep.T,)
    assert _class_parameters(pep.Two) == (pep.T, pep.U)
    assert _class_parameters(pep.Twice) == (pep.T,)
    assert _class_parameters(pep.Sub) == ()
    assert _class_parameters(pep.Leaf) == ()
    assert _generic_variances(pep.GL) == ("invariant",)
    assert _generic_variances(pep.GD) == ("invariant",)
    assert _generic_variances(pep.GM) == ("invariant", "invariant")
    # The declared variance, not the slot's (see `_generic_variances`).
    assert _generic_variances(pep.Cov) == ("covariant",)
    assert _generic_variances(pep.Sub) is None
    assert _generic_variances(pep.Leaf) is None
    # What `Generic` computes, where it is there to ask.
    assert pep.Mixed.__parameters__ == (pep.T,)


def _pep585_hint_rows(
    pep: types.SimpleNamespace,
) -> tx.List[tx.Tuple[tx.Any, tx.Any, bool]]:
    return [
        # `list` is invariant, and so is `GL`'s unflagged `T`.
        (pep.GL[bool], pep.GL[int], False),
        (pep.GL[int], pep.GL[int], True),
        (pep.GL[int], list[int], True),
        (pep.GL[int], tx.List[int], True),
        (pep.GL[int], list[str], False),
        (pep.GL[bool], list[int], False),
        (pep.GL[bool], tx.Sequence[int], True),
        (pep.GL[int], list, True),
        (pep.GL, list[int], False),
        (pep.GD[int], tx.Dict[str, int], True),
        (pep.GD[int], dict[str, str], False),
        (pep.GD[int], tx.Mapping[str, object], True),
        (pep.GD[bool], pep.GD[int], False),
        (pep.GM[int, bool], tx.Mapping[int, int], True),
        (pep.GM[int, bool], tx.Mapping[bool, bool], False),
        (pep.Sub, pep.GL[int], True),
        (pep.Sub, list[int], True),
        (pep.Sub, tx.List[str], False),
        (pep.Sub2[int], pep.GL[int], True),
        (pep.Sub2[int], list[int], True),
        (pep.Sub2[int], list[str], False),
        (pep.Leaf, pep.GL[int], False),
        (pep.Two[int, str], list[int], True),
        (pep.Two[int, str], list[str], False),
        # Below `Container[X]` through either base (#64).
        (pep.Two[int, str], tx.Container[int], True),
        (pep.Two[int, str], tx.Container[str], True),
        (pep.Two[int, str], tx.Container[bytes], False),
        (pep.Mixed[bool], pep.Mixed[int], False),
        (pep.Mixed[int], list[int], True),
        # The reverse direction is untouched: a base is not a subclass.
        (list[int], pep.GL[int], False),
    ]


@_PEP585
def test_pep585_hint_truth_table(pep: types.SimpleNamespace) -> None:
    for sub, sup, expected in _pep585_hint_rows(pep):
        assert issubhint(sub, sup) is expected, (sub, sup)


@_PEP585
def test_a_flagged_typevar_is_taken_at_its_word(
    pep: types.SimpleNamespace,
) -> None:
    """`class Cov(list[T_co])` is an error a type checker reports.

    The class says `Cov` is covariant, and `Cov[bool] <= Cov[int]` follows,
    while each still maps onto the invariant `list` it is written against.
    """
    assert issubhint(pep.Cov[bool], pep.Cov[int]) is True
    assert issubhint(pep.Cov[bool], list[bool]) is True
    assert issubhint(pep.Cov[bool], list[int]) is False


@_PEP585
def test_pep585_instances_read_their_record(
    pep: types.SimpleNamespace,
) -> None:
    assert ishintstance(pep.GL[int](), list[int]) is True
    assert ishintstance(pep.GL[int](), list[str]) is False
    assert ishintstance(pep.GL[int](), tx.List[str]) is False
    assert ishintstance(pep.GL[int](), pep.GL[str]) is False
    assert ishintstance(pep.GL[int](), tx.Sequence[object]) is True
    assert ishintstance(pep.GD[int](), tx.Dict[str, int]) is True
    assert ishintstance(pep.GD[int](), tx.Dict[str, str]) is False
    assert ishintstance(pep.GM[int, str](), tx.Mapping[int, str]) is True
    assert ishintstance(pep.GM[int, str](), tx.Mapping[str, str]) is False
    assert ishintstance(pep.Sub2[int](), list[str]) is False
    # Not generic: its class declares `list[int]`.
    assert ishintstance(pep.Sub(), list[int]) is True
    assert ishintstance(pep.Sub(), list[str]) is False
    assert ishintstance(pep.Mixed[int](), list[str]) is False


@_PEP585
def test_pep585_values_that_declare_nothing_stay_shallow(
    pep: types.SimpleNamespace,
) -> None:
    # Built from the bare class, or with `T` left open.
    assert ishintstance(pep.GL(), list[str]) is True
    assert ishintstance(pep.GL[pep.T](), list[str]) is True
    assert ishintstance(pep.Leaf(), list[str]) is True
    # A plain list declares nothing, and `list[int]([1])` is one.
    assert ishintstance([1], list[str]) is True
    assert ishintstance(list[int]([1]), list[str]) is True
    # Nowhere to record the parametrisation.
    for value in (pep.Slotted[int](), pep.Frozen[int]()):
        assert not hasattr(value, "__orig_class__")
        assert ishintstance(value, list[str]) is True


@_PEP585
def test_pep585_instances_dispatch_on_their_record(
    pep: types.SimpleNamespace,
) -> None:
    m = Function("m")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.register((list[int],))(lambda xs: "ints")
        m.register((list[str],))(lambda xs: "strs")
    # Ambiguous before #60: `GL[int]()` was matched as a bare list.
    assert m(pep.GL[int]()) == "ints"
    assert m(pep.GL[str]()) == "strs"
    assert m(pep.Sub()) == "ints"
    assert m(pep.Sub2[str]()) == "strs"
    assert m(pep.Mixed[int]()) == "ints"
    for undeclared in ([1], pep.GL(), pep.Slotted[int]()):
        with pytest.raises(AmbiguousMethodError):
            m(undeclared)
    d = Function("d")
    d.register((tx.Dict[str, int],))(lambda xs: "ints")
    d.register((tx.Dict[str, str],))(lambda xs: "strs")
    assert d(pep.GD[int]()) == "ints"
    assert d(pep.GD[str]()) == "strs"


@_PEP585
def test_pep585_records_are_never_served_a_stale_method(
    pep: types.SimpleNamespace,
) -> None:
    m = Function("m")
    m.register((list[int],))(lambda xs: "ints")
    m.register((list[str],))(lambda xs: "strs")
    for _ in range(3):
        assert m(pep.GL[int]()) == "ints"
        assert m(pep.GL[str]()) == "strs"
    # Keyword arguments key the same way.
    g = Function("g")
    g.register({"xs": tx.List[int]})(lambda *, xs: "ints")
    g.register({"xs": tx.List[str]})(lambda *, xs: "strs")
    for _ in range(3):
        assert g(xs=pep.GL[int]()) == "ints"
        assert g(xs=pep.GL[str]()) == "strs"


@_PEP585
def test_pep585_records_share_an_entry(pep: types.SimpleNamespace) -> None:
    """Each `GL[int]` is a new alias object, but keys like the last one."""
    m = Function("m")
    m.register((list[int],))(lambda xs: "ints")
    m.register((list[str],))(lambda xs: "strs")
    first, second = pep.GL[int](), pep.GL[int]()
    assert first.__orig_class__ is not second.__orig_class__
    assert m(first) == "ints"
    assert m(second) == "ints"
    assert len(m._cache.call_cache) == 1
    plan = _plan_of(m, (1, ()))
    assert plan.declared == frozenset({0})

    def key(value: tx.Any) -> tx.Any:
        return _call_key((value,), {}, plan)

    assert key(first) == key(second)
    assert key(first) != key(pep.GL[str]())
    assert key(pep.GL[list[int]]()) == key(pep.GL[list[int]]())
    assert key(pep.GL[list[int]]()) != key(pep.GL[list[str]]())
    # A plain list is not probed, and keys as its type and `None`.
    assert key([1]) == (1, (list, None))


@_PEP585
def test_pep585_record_keys() -> None:
    """Keyed by parts: equal parts share a key, and nothing else does."""
    mapping = collections.abc.Mapping
    same = [
        (dict[str, list[int]], dict[str, list[int]]),  # type: ignore[misc]
        (mapping[int, str], mapping[int, str]),  # type: ignore[index]
        (list[tx.List[int]], list[tx.List[int]]),  # type: ignore[misc]
    ]
    for first, second in same:
        assert first is not second
        assert _record_key(first) == _record_key(second), first
        assert hash(_record_key(first)) == hash(_record_key(second))
    different = [
        (dict[str, list[int]], dict[str, list[str]]),  # type: ignore[misc]
        (dict[str, list[int]], dict[bytes, list[int]]),  # type: ignore
        (mapping[int, str], collections.abc.Sequence[int]),  # type: ignore
        (list[int], set[int]),  # type: ignore[misc]
        (list[tx.Literal[1]], list[tx.Literal[True]]),  # type: ignore[misc]
    ]
    for first, second in different:
        assert _record_key(first) != _record_key(second), (first, second)
    # A plain class is its own key.
    assert _record_key(int) is int


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="an unpacked alias is 3.11+"
)
def test_an_unpacked_record_keys_apart_from_a_packed_one() -> None:
    packed = tuple[int]  # type: ignore[misc]
    unpacked = next(iter(packed))
    assert _record_key(unpacked) != _record_key(packed)
    assert _record_key(tuple[int]) == _record_key(packed)  # type: ignore


@_PEP585
def test_pep585_plain_subclasses_are_not_probed() -> None:
    """Only a class written against a PEP 585 alias is asked for a record."""
    probed = []  # type: tx.List[str]

    class Probe(list):  # type: ignore[type-arg]
        def __getattr__(self, name: str) -> tx.Any:
            probed.append(name)
            raise AttributeError(name)

    assert _may_record_parametrisation(Probe) is False
    m = Function("m")
    m.register((list[int],))(lambda xs: "ints")
    m.register((object,))(lambda xs: "any")
    assert m(Probe()) == "ints"
    assert m(Probe()) == "ints"
    assert "__orig_class__" not in probed
    key = _call_key((Probe(),), {}, _plan_of(m, (1, ())))
    assert key == (1, (Probe, None))


@_PEP585
def test_pep585_declaration_dependence(pep: types.SimpleNamespace) -> None:
    for hint in (
        pep.GL[int],
        pep.GD[int],
        pep.GM[int, str],
        list[int],
        tx.List[int],
        tx.Optional[pep.GL[int]],
    ):
        assert is_declaration_dependent(hint) is True, hint
    for hint in (pep.GL, pep.Sub, pep.Leaf[int], list):
        assert is_declaration_dependent(hint) is False, hint


@_PEP585
def test_pep585_class_with_an_unhashable_metaclass() -> None:
    """An unhashable PEP 585 subclass is gated, read and dispatched on.

    The gate's memo keys a class by `id`, and the variance memo reads an
    unhashable class afresh (#64), so `Odd[int]` is compared with `Odd[str]`
    as well as with the `list[...]` it is written against.
    """
    T = tx.TypeVar("T")

    class Odd(list[T], metaclass=_unhashable_meta()):  # type: ignore[misc]
        pass

    with pytest.raises(TypeError):
        hash(Odd)
    assert _may_record_parametrisation(Odd) is True
    assert _RECORDERS[id(Odd)] is True
    assert ishintstance(Odd[int](), list[int]) is True
    assert ishintstance(Odd[int](), list[str]) is False
    _check_an_unhashable_generic(Odd)


def test_the_gate_memo_does_not_keep_a_class_alive() -> None:
    """A class dropped at runtime leaves the memo when it is collected."""
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    class Plain:
        pass

    keys = [id(Box), id(Plain)]
    assert _may_record_parametrisation(Box) is True
    assert _may_record_parametrisation(Plain) is False
    assert all(key in _RECORDERS for key in keys)
    watch = weakref.ref(Box), weakref.ref(Plain)
    del Box, Plain
    gc.collect()
    assert [ref() for ref in watch] == [None, None]
    assert not any(key in _RECORDERS for key in keys)
    assert not any(key in _RECORDER_REFS for key in keys)


# --- a class whose metaclass makes it unhashable (#64) -----------------


def _unhashable_meta() -> type:
    """A metaclass defining `__eq__` alone, so its classes are unhashable."""

    class Meta(type):
        def __eq__(cls, other: tx.Any) -> bool:
            return cls is other

    assert Meta.__hash__ is None
    return Meta


def _check_an_unhashable_generic(odd: tx.Any) -> None:
    """`odd` takes one unflagged `T`, and cannot be hashed."""
    with pytest.raises(TypeError):
        hash(odd)
    assert _generic_variances(odd) == ("invariant",)
    assert issubhint(odd[int], odd[int]) is True
    assert issubhint(odd[int], odd[str]) is False
    assert issubhint(odd[bool], odd[int]) is False
    assert ishintstance(odd[int](), odd[int]) is True
    assert ishintstance(odd[int](), odd[str]) is False
    f = Function("f")
    f.register((odd[int],))(lambda x: "int")
    f.register((odd[str],))(lambda x: "str")
    f.register((object,))(lambda x: "any")
    assert [f(odd[int]()), f(odd[str]()), f(odd[int]()), f(1)] == [
        "int",
        "str",
        "int",
        "any",
    ]


@pytest.mark.parametrize("module", [typing, tx], ids=["typing", "tx"])
def test_generic_class_with_an_unhashable_metaclass(module: tx.Any) -> None:
    """A `Generic` subclass that cannot key a memo is read afresh."""
    T = module.TypeVar("T")

    class Odd(module.Generic[T], metaclass=_unhashable_meta()):
        pass

    _check_an_unhashable_generic(Odd)


def test_the_variance_memo_keeps_a_hashable_origin() -> None:
    """Only an origin that cannot be hashed skips the memo."""
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    before = _memoised_variances.cache_info().misses
    assert _generic_variances(Box) == ("invariant",)
    assert _generic_variances(Box) == ("invariant",)
    assert _memoised_variances.cache_info().misses == before + 1


# --- a class reaching the origin through several bases (#64) -----------


@pytest.mark.parametrize("module", [typing, tx], ids=["typing", "tx"])
def test_a_class_is_below_each_parametrisation_its_bases_reach(
    module: tx.Any,
) -> None:
    """`Two[int, str]` is a `Container[int]` through `List[T]` and a
    `Container[str]` through its own base; either is enough."""
    T = module.TypeVar("T")
    U = module.TypeVar("U")

    class Two(module.List[T], module.Container[U]):
        pass

    _check_two(Two)


@_PEP585
def test_a_pep585_class_is_below_each_parametrisation_its_bases_reach() -> (
    None
):
    """`Two` again, with its `list` base spelled as a PEP 585 alias."""
    T = tx.TypeVar("T")
    U = tx.TypeVar("U")

    class Two(list[T], collections.abc.Container[U]):  # type: ignore[misc]
        pass

    _check_two(Two)


def _check_two(two: tx.Any) -> None:
    """The hint, value and cache levels of a `Two[T, U]` class."""
    container = tx.Container
    assert _as_base_args(two[int, str], collections.abc.Container) == (
        (int,),
        (str,),
    )
    # Hint level: either base's parametrisation satisfies the super-hint.
    assert issubhint(two[int, str], container[int]) is True
    assert issubhint(two[int, str], container[str]) is True
    # `Container` is covariant.
    assert issubhint(two[bool, str], container[int]) is True
    assert issubhint(two[int, str], container[object]) is True
    assert issubhint(two[int, str], container[bytes]) is False
    assert issubhint(two[int, str], container[bool]) is False
    # Value level: an instance reads the parametrisation it recorded.
    assert ishintstance(two[int, str](), container[int]) is True
    assert ishintstance(two[int, str](), container[str]) is True
    assert ishintstance(two[int, str](), container[bytes]) is False
    # Cache: keyed by the recorded parametrisation, so a second one with
    # the same type does not reuse the first one's answer.
    f = Function("f")
    f.register((container[str],))(lambda x: "str")
    f.register((object,))(lambda x: "any")
    calls = [two[int, str](), two[int, bytes](), two[str, int]()]
    assert [f(each) for each in calls * 2] == ["str", "any", "str"] * 2


def test_a_class_declaring_one_base_fully_is_read_by_its_declaration() -> None:
    """One fully declared base is enough for the value to be checked."""
    T = tx.TypeVar("T")
    U = tx.TypeVar("U")

    class Two(tx.List[T], tx.Container[U]):
        pass

    class Half(Two[int, tx.Any]):
        pass

    # Through `List[int]` the class declares `Container[int]`; through its
    # own `Container[Any]` it declares nothing.
    assert _as_base_args(Half, collections.abc.Container) == (
        (int,),
        (tx.Any,),
    )
    assert ishintstance(Half(), tx.Container[int]) is True
    assert ishintstance(Half(), tx.Container[bytes]) is False
    assert issubhint(Half, tx.Container[bytes]) is False

    class Other(Two[tx.Any, str]):
        pass

    # The first base declares nothing, the second declares `str`: the value
    # is still read by its declaration.
    assert ishintstance(Other(), tx.Container[str]) is True
    assert ishintstance(Other(), tx.Container[bytes]) is False
