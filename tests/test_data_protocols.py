"""Runtime-checkable protocols with data members (#56).

Python refuses `issubclass()` against a protocol that declares a data member
(`name: str`), so dispatch used to answer False for every value and a method
registered on one never ran. Three layers, each with its own section:

* **value level** -- `ishintstance(v, P)` reads each data member off the
  value (its instance `__dict__` or its class) and each method off its class;
  a class that names `P` among its bases is one outright;
* **hint level** -- `issubhint(C, P)` holds when every instance of `C` is
  one: `C` names `P` among its bases, or defines every member itself (a bare
  annotation does not count; a dataclass field does);
* **cache** -- such a position keys on the value's type and which of the
  protocol's data members it has, never on the value, and combines with a
  value- or declaration-dependent key at the same position.

Every protocol is built twice, from `typing` and from `typing_extensions`.
No `X | Y` or `list[int]`: this runs on 3.8.
"""

# stdlib
import dataclasses
import types
import typing

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import NoMethodError
from bagof.dispatchers._function import Function, _call_key, _Plan
from bagof.dispatchers._lattice import (
    instance_members,
    is_declaration_dependent,
    is_value_dependent,
)
from bagof.dispatchers.core import ishintstance, issubhint

# --- the family, in both spellings -------------------------------------


def _family(module: tx.Any) -> types.SimpleNamespace:
    """The protocols and classes of this file, built from `module`."""

    @module.runtime_checkable
    class HasName(module.Protocol):
        name: str

    @module.runtime_checkable
    class HasNameAge(HasName, module.Protocol):
        age: int

    @module.runtime_checkable
    class OtherName(module.Protocol):
        """The same member as `HasName`, but not a sub-protocol of it."""

        name: str

    @module.runtime_checkable
    class SupportsClose(module.Protocol):
        def close(self) -> None: ...

    @module.runtime_checkable
    class NameClose(module.Protocol):
        name: str

        def close(self) -> None: ...

    @module.runtime_checkable
    class NamedCloser(SupportsClose, module.Protocol):
        name: str

    @module.runtime_checkable
    class AnnotatedClose(module.Protocol):
        """`close` is a data member here: an annotation, not a method."""

        name: str
        close: tx.Callable[[], None]

    class NotRuntime(module.Protocol):
        name: str

    class Plain:
        """Declares nothing: an instance has `name` only if it is set."""

    class ClassAttr:
        name = "class-level"

    class Annotated:
        """A bare annotation: promises nothing at runtime."""

        name: str

    @dataclasses.dataclass
    class Record:
        name: str

    @dataclasses.dataclass
    class Person:
        name: str
        age: int

    class Prop:
        @property
        def name(self) -> str:
            return "prop"

    class Slotted:
        __slots__ = ("name",)

    class Nominal(HasName):
        """Names `HasName` among its bases, and sets nothing."""

    class Closer:
        name = "c"

        def close(self) -> None:
            pass

    class NoClose:
        name = "c"
        close = None

    return types.SimpleNamespace(**locals())


@pytest.fixture(params=[typing, tx], ids=["typing", "tx"])
def p(request: tx.Any) -> types.SimpleNamespace:
    return _family(request.param)


def _with_name(cls: type, value: tx.Any = "x") -> tx.Any:
    obj = cls()
    obj.name = value
    return obj


# --- value level -------------------------------------------------------


def test_the_issue_repro(p: types.SimpleNamespace) -> None:
    """#56: an attribute on the instance or on its class both count."""
    assert ishintstance(_with_name(p.Plain), p.HasName) is True
    assert ishintstance(p.ClassAttr(), p.HasName) is True
    assert ishintstance(p.Plain(), p.HasName) is False


@pytest.mark.parametrize(
    "make, expected",
    [
        (lambda p: _with_name(p.Plain), True),
        (lambda p: p.ClassAttr(), True),
        (lambda p: p.Record("r"), True),
        (lambda p: p.Prop(), True),
        (lambda p: p.Nominal(), True),
        (lambda p: p.Plain(), False),
        (lambda p: p.Annotated(), False),
        (lambda p: _with_name(p.Annotated), True),
        (lambda p: _with_name(p.Plain, None), True),
    ],
    ids=[
        "instance-attribute",
        "class-attribute",
        "dataclass-field",
        "property",
        "names-it-as-a-base",
        "missing",
        "annotated-but-unset",
        "annotated-and-set",
        "set-to-none",
    ],
)
def test_value_matches_isinstance(
    p: types.SimpleNamespace, make: tx.Any, expected: bool
) -> None:
    value = make(p)
    assert ishintstance(value, p.HasName) is expected
    # Where every Python and both spellings agree, so does the relation.
    assert isinstance(value, p.HasName) is expected


def test_members_are_found_without_running_code(
    p: types.SimpleNamespace,
) -> None:
    """A property is found, not called; `__getattr__` is never asked.

    As Python's `isinstance` reads a protocol from 3.12 on; `typing`'s own
    before 3.12 calls `hasattr`, so the two can differ there.
    """

    class Raising:
        @property
        def name(self) -> str:
            raise AttributeError("not now")

    class Dynamic:
        def __getattr__(self, attr: str) -> str:
            return "dynamic"

    assert ishintstance(Raising(), p.HasName) is True
    assert ishintstance(Dynamic(), p.HasName) is False


def test_slots(p: types.SimpleNamespace) -> None:
    """A slot is defined by the class, set or not, as Python reads it."""

    class Bare:
        __slots__ = ()

    assert ishintstance(p.Slotted(), p.HasName) is True
    assert ishintstance(Bare(), p.HasName) is False


def test_a_class_that_redefines_its_dict_is_read_statically(
    p: types.SimpleNamespace,
) -> None:
    class Shadowed:
        @property
        def __dict__(self) -> tx.Dict[str, tx.Any]:  # type: ignore[override]
            raise AssertionError("the value's own code must not run")

    class ShadowedNamed(Shadowed):
        name = "n"

    assert ishintstance(Shadowed(), p.HasName) is False
    assert ishintstance(ShadowedNamed(), p.HasName) is True


def test_a_class_object_is_read_through_its_bases_and_metaclass(
    p: types.SimpleNamespace,
) -> None:
    class Base:
        name = "b"

    class Child(Base):
        pass

    assert ishintstance(Child, p.HasName) is True
    assert ishintstance(p.Plain, p.HasName) is False


def test_an_unhashable_class_is_still_read(p: types.SimpleNamespace) -> None:
    class Meta(type):
        def __eq__(cls, other: tx.Any) -> bool:
            return cls is other

    assert Meta.__hash__ is None

    class Odd(metaclass=Meta):
        pass

    assert ishintstance(_with_name(Odd), p.HasName) is True
    assert ishintstance(Odd(), p.HasName) is False
    # The class itself, which `inspect.getattr_static` cannot read from
    # 3.13 on (and `isinstance` raises for): nothing is found, never an error.
    assert ishintstance(Odd, p.HasName) is False


def test_methods_are_read_off_the_class(p: types.SimpleNamespace) -> None:
    """A method must be defined by the class, and not as `None`."""
    assert ishintstance(p.Closer(), p.NameClose) is True
    assert ishintstance(p.NoClose(), p.NameClose) is False
    assert ishintstance(p.ClassAttr(), p.NameClose) is False
    # A method set on the instance alone is not counted: only the data
    # members depend on the instance, which is all the cache keys on.
    only = p.ClassAttr()
    only.close = lambda: None
    assert ishintstance(only, p.NameClose) is False


def test_a_method_only_protocol_is_unchanged(
    p: types.SimpleNamespace,
) -> None:
    assert ishintstance(p.Closer(), p.SupportsClose) is True
    assert ishintstance(p.Plain(), p.SupportsClose) is False
    assert issubhint(p.Closer, p.SupportsClose) is True
    assert instance_members(p.SupportsClose) == ()


def test_a_non_runtime_protocol_is_unchanged(
    p: types.SimpleNamespace,
) -> None:
    """Out of scope (#56): it still matches nothing, and says so."""
    assert ishintstance(p.ClassAttr(), p.NotRuntime) is False
    assert issubhint(p.ClassAttr, p.NotRuntime) is False
    assert instance_members(p.NotRuntime) == ()
    f = Function("f")
    f.register((p.NotRuntime,))(lambda x: "never")
    with pytest.raises(NoMethodError):
        f(p.ClassAttr())


def test_a_sub_protocol_needs_every_member(p: types.SimpleNamespace) -> None:
    both = _with_name(p.Plain)
    both.age = 3
    assert ishintstance(both, p.HasNameAge) is True
    assert ishintstance(_with_name(p.Plain), p.HasNameAge) is False
    assert ishintstance(p.Person("n", 1), p.HasNameAge) is True


def test_type_of_a_data_protocol(p: types.SimpleNamespace) -> None:
    """`Type[P]` asks the class question `issubhint` answers."""
    assert ishintstance(p.ClassAttr, tx.Type[p.HasName]) is True
    assert ishintstance(p.Record, tx.Type[p.HasName]) is True
    assert ishintstance(p.Plain, tx.Type[p.HasName]) is False
    assert issubhint(tx.Type[p.ClassAttr], tx.Type[p.HasName]) is True
    assert issubhint(tx.Type[p.Plain], tx.Type[p.HasName]) is False


# --- hint level --------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ClassAttr", True),
        ("Record", True),
        ("Person", True),
        ("Prop", True),
        ("Slotted", True),
        ("Nominal", True),
        ("HasNameAge", True),
        ("HasName", True),
        ("Plain", False),
        ("Annotated", False),
        ("OtherName", False),
        ("SupportsClose", False),
    ],
)
def test_what_is_below_a_data_protocol(
    p: types.SimpleNamespace, name: str, expected: bool
) -> None:
    assert issubhint(getattr(p, name), p.HasName) is expected


def test_an_annotated_but_unset_attribute_is_not_declared(
    p: types.SimpleNamespace,
) -> None:
    """Soundness: `Annotated <= HasName` would let an instance of a sub-hint
    fail the super-hint's value check."""
    value = p.Annotated()
    assert ishintstance(value, p.Annotated) is True
    assert ishintstance(value, p.HasName) is False
    assert issubhint(p.Annotated, p.HasName) is False


def test_protocols_order_by_their_bases(p: types.SimpleNamespace) -> None:
    assert issubhint(p.HasNameAge, p.HasName) is True
    assert issubhint(p.HasName, p.HasNameAge) is False
    assert issubhint(p.HasName, p.OtherName) is False
    assert issubhint(p.OtherName, p.HasName) is False
    assert issubhint(p.HasName, object) is True
    assert issubhint(object, p.HasName) is False
    assert issubhint(p.HasName, tx.Union[p.HasName, int]) is True
    assert issubhint(tx.Union[p.ClassAttr, p.Record], p.HasName) is True
    assert issubhint(tx.Optional[p.ClassAttr], p.HasName) is False
    # A hint that is not a class declares nothing.
    assert issubhint(tx.Union, p.HasName) is False
    assert issubhint(
        tx.Type[tx.Union[p.ClassAttr, int]], tx.Type[p.HasName]
    ) is False


def test_methods_are_declared_by_the_class(p: types.SimpleNamespace) -> None:
    assert issubhint(p.Closer, p.NameClose) is True
    assert issubhint(p.NoClose, p.NameClose) is False
    assert issubhint(p.ClassAttr, p.NameClose) is False


def test_a_data_protocol_below_a_method_only_one(
    p: types.SimpleNamespace,
) -> None:
    """By its bases, or by its methods -- never by a data member."""
    assert issubhint(p.NamedCloser, p.SupportsClose) is True
    assert issubhint(p.NameClose, p.SupportsClose) is True
    assert issubhint(p.AnnotatedClose, p.SupportsClose) is False
    assert issubhint(p.HasName, p.SupportsClose) is False
    # An instance of `AnnotatedClose` may hold `close` on itself alone, so
    # its class need not satisfy `SupportsClose`.
    value = _with_name(p.Plain)
    value.close = lambda: None
    assert ishintstance(value, p.AnnotatedClose) is True
    assert ishintstance(value, p.SupportsClose) is False


_VALUES = [
    lambda p: p.Plain(),
    lambda p: _with_name(p.Plain),
    lambda p: p.ClassAttr(),
    lambda p: p.Annotated(),
    lambda p: _with_name(p.Annotated),
    lambda p: p.Record("r"),
    lambda p: p.Person("r", 1),
    lambda p: p.Prop(),
    lambda p: p.Slotted(),
    lambda p: p.Nominal(),
    lambda p: p.Closer(),
    lambda p: p.NoClose(),
]

_HINTS = [
    "HasName",
    "HasNameAge",
    "OtherName",
    "SupportsClose",
    "NameClose",
    "NamedCloser",
    "AnnotatedClose",
    "Plain",
    "ClassAttr",
    "Annotated",
    "Record",
    "Person",
    "Prop",
    "Slotted",
    "Nominal",
    "Closer",
    "NoClose",
]


def test_dispatch_soundness(p: types.SimpleNamespace) -> None:
    """`v in Sub and Sub <= Base` implies `v in Base`, over the family."""
    values = [make(p) for make in _VALUES]
    hints = [getattr(p, name) for name in _HINTS] + [object]
    for sub in hints:
        for base in hints:
            if not issubhint(sub, base):
                continue
            for value in values:
                if ishintstance(value, sub):
                    assert ishintstance(value, base), (value, sub, base)


# --- dispatch and the cache --------------------------------------------


def _plan_of(f: Function, shape: tx.Any) -> _Plan:
    return f._cache.shape_plans[shape]


def _named_or_not(p: types.SimpleNamespace) -> Function:
    f = Function("f")
    f.register((p.HasName,))(lambda x: "named")
    f.register((object,))(lambda x: "object")
    return f


def test_a_data_protocol_overload_beats_object(
    p: types.SimpleNamespace,
) -> None:
    f = _named_or_not(p)
    assert f(_with_name(p.Plain)) == "named"
    assert f(p.Plain()) == "object"
    assert f(p.ClassAttr()) == "named"
    assert f(p.Record("r")) == "named"
    assert f(p.Annotated()) == "object"
    assert f.resolve(p.ClassAttr).function(None) == "named"
    assert f.resolve(p.Annotated).function(None) == "object"


def test_the_more_specific_protocol_wins(p: types.SimpleNamespace) -> None:
    f = Function("f")
    f.register((p.HasName,))(lambda x: "name")
    f.register((p.HasNameAge,))(lambda x: "name-age")
    f.register((object,))(lambda x: "object")
    both = _with_name(p.Plain)
    both.age = 1
    assert f(both) == "name-age"
    assert f(_with_name(p.Plain)) == "name"
    assert f(p.Plain()) == "object"
    assert f(p.Person("n", 2)) == "name-age"


def test_a_class_that_declares_the_members_is_more_specific(
    p: types.SimpleNamespace,
) -> None:
    f = Function("f")
    f.register((p.HasName,))(lambda x: "protocol")
    f.register((p.Record,))(lambda x: "record")
    assert f(p.Record("r")) == "record"
    assert f(p.ClassAttr()) == "protocol"


def test_a_union_with_a_data_protocol(p: types.SimpleNamespace) -> None:
    f = Function("f")
    f.register((tx.Union[p.HasName, int],))(lambda x: "named-or-int")
    f.register((object,))(lambda x: "object")
    for _ in range(2):
        assert f(1) == "named-or-int"
        assert f(_with_name(p.Plain)) == "named-or-int"
        assert f(p.Plain()) == "object"
        assert f("s") == "object"


def test_keyword_binding(p: types.SimpleNamespace) -> None:
    f = Function("f")

    @f.register
    def _named(*, x: p.HasName) -> str:  # type: ignore[name-defined]
        return "named"

    @f.register
    def _other(*, x: object) -> str:
        return "object"

    for _ in range(2):
        assert f(x=_with_name(p.Plain)) == "named"
        assert f(x=p.Plain()) == "object"


def test_one_class_alternating_never_gets_a_stale_method(
    p: types.SimpleNamespace,
) -> None:
    """Instances of one class, with and without the attribute, alternate."""
    f = _named_or_not(p)
    for _ in range(3):
        assert f(_with_name(p.Plain)) == "named"
        assert f(p.Plain()) == "object"
        assert f(x=_with_name(p.Plain)) == "named"
        assert f(x=p.Plain()) == "object"
    # One entry per (shape, presence), however many instances were seen.
    assert len(f._cache.call_cache) == 4


def test_the_key_is_the_type_and_the_members_present(
    p: types.SimpleNamespace,
) -> None:
    f = _named_or_not(p)
    f(p.Plain())
    plan = _plan_of(f, (1, ()))
    assert plan.members == {0: ("name",)}
    assert plan.value_dependent == frozenset()
    assert plan.declared == frozenset()
    assert _call_key((p.Plain(),), {}, plan) == (1, (p.Plain, (False,)))
    first, second = _with_name(p.Plain, 1), _with_name(p.Plain, 2)
    assert _call_key((first,), {}, plan) == (1, (p.Plain, (True,)))
    assert _call_key((first,), {}, plan) == _call_key((second,), {}, plan)
    f(x=first)
    (keywords,) = [
        each for shape, each in f._cache.shape_plans.items() if shape[1]
    ]
    assert keywords.members == {"x": ("name",)}
    assert _call_key((), {"x": first}, keywords) == (
        0,
        ("x", p.Plain, (True,)),
    )


def test_every_protocol_at_a_position_is_keyed(
    p: types.SimpleNamespace,
) -> None:
    f = Function("f")
    f.register((p.HasNameAge,))(lambda x: "name-age")
    f.register((p.NameClose,))(lambda x: "name-close")
    f.register((object,))(lambda x: "object")
    assert f(p.Plain()) == "object"
    assert _plan_of(f, (1, ())).members == {0: ("age", "name")}


def test_members_through_wrappers(p: types.SimpleNamespace) -> None:
    bound = tx.TypeVar("bound", bound=p.HasName)
    assert instance_members(p.HasName) == ("name",)
    assert instance_members(p.HasNameAge) == ("age", "name")
    assert instance_members(p.NameClose) == ("name",)
    assert instance_members(tx.Optional[p.HasName]) == ("name",)
    assert instance_members(tx.Union[p.HasNameAge, p.OtherName]) == (
        "age",
        "name",
    )
    assert instance_members(tx.Annotated[p.HasName, "m"]) == ("name",)
    assert instance_members(bound) == ("name",)
    assert instance_members(int) == ()
    assert instance_members(tx.Literal[1]) == ()
    from bagof.dispatchers import Exact

    assert instance_members(Exact[p.ClassAttr]) == ()
    # Neither of the other two dependences.
    assert is_value_dependent(p.HasName) is False
    assert is_declaration_dependent(p.HasName) is False


def test_a_typevar_bounded_by_a_data_protocol_dispatches(
    p: types.SimpleNamespace,
) -> None:
    bound = tx.TypeVar("bound", bound=p.HasName)
    f = Function("f")
    f.register((bound,))(lambda x: "named")
    f.register((object,))(lambda x: "object")
    for _ in range(2):
        assert f(_with_name(p.Plain)) == "named"
        assert f(p.Plain()) == "object"


class _AllEqual:
    """Every instance equal to every other, with one hash: `==` alone
    cannot tell a named instance from an unnamed one."""

    def __eq__(self, other: tx.Any) -> bool:
        return isinstance(other, _AllEqual)

    def __hash__(self) -> int:
        return 0


def test_a_value_dependent_position_also_keys_on_members(
    p: types.SimpleNamespace,
) -> None:
    hint = tx.Union[tx.Literal[1], p.HasName]
    assert is_value_dependent(hint)
    assert instance_members(hint) == ("name",)
    f = Function("f")
    f.register((hint,))(lambda x: "one-or-named")
    f.register((object,))(lambda x: "object")
    plan = None
    for _ in range(3):
        assert f(1) == "one-or-named"
        assert f(2) == "object"
        assert f(_with_name(_AllEqual)) == "one-or-named"
        assert f(_AllEqual()) == "object"
        plan = _plan_of(f, (1, ()))
    assert plan is not None
    assert plan.value_dependent == frozenset({0})
    assert plan.members == {0: ("name",)}
    key = _call_key((_AllEqual(),), {}, plan)
    assert key[1][0] is _AllEqual and key[1][-1] == (False,)


def test_a_declaration_dependent_position_also_keys_on_members(
    p: types.SimpleNamespace,
) -> None:
    T = tx.TypeVar("T")

    class Box(tx.Generic[T]):
        pass

    hint = tx.Union[Box[int], p.HasName]
    assert is_declaration_dependent(hint)
    assert instance_members(hint) == ("name",)
    f = Function("f")
    f.register((hint,))(lambda x: "int-box-or-named")
    f.register((object,))(lambda x: "object")
    for _ in range(3):
        assert f(Box[int]()) == "int-box-or-named"
        assert f(Box[str]()) == "object"
        assert f(_with_name(Box[str])) == "int-box-or-named"
        assert f(Box[str]()) == "object"
        assert f(x=_with_name(Box[str])) == "int-box-or-named"
        assert f(x=Box[str]()) == "object"
    plan = _plan_of(f, (1, ()))
    assert plan.declared == frozenset({0})
    assert plan.members == {0: ("name",)}


def test_a_generic_data_protocol() -> None:
    T = tx.TypeVar("T")

    @tx.runtime_checkable
    class HasItem(tx.Protocol[T]):
        item: T

    class Holder:
        pass

    held = Holder()
    held.item = 1
    assert ishintstance(held, HasItem[int]) is True
    assert ishintstance(Holder(), HasItem[int]) is False
    assert instance_members(HasItem[int]) == ("item",)
    f = Function("f")
    f.register((HasItem[int],))(lambda x: "item")
    f.register((object,))(lambda x: "object")
    for _ in range(2):
        assert f(held) == "item"
        assert f(Holder()) == "object"
