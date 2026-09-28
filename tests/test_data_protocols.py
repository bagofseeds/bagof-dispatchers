"""Runtime-checkable protocols with data members (#56).

Python refuses `issubclass()` against a protocol that declares a data member
(`name: str`), so dispatch used to answer False for every value and a method
registered on one never ran. Three layers, each with its own section:

* **value level** -- `ishintstance(v, P)` reads each data member off the
  value (its instance `__dict__` or its class) and each method off its class;
  a class that names `P` among its bases is one outright;
* **hint level** -- `issubhint(C, P)` holds when `C` is a structural subtype
  of `P` as a type checker reads it: `C` names `P` among its bases, or
  holds or declares every member -- an annotation declares one, set or not,
  and the value level then counts it as present on every instance, so the
  order stays sound; a `ClassVar` member is declared only as a `ClassVar`;
* **cache** -- such a position keys on the value's type and which of the
  protocol's data members it has, never on the value, and combines with a
  value- or declaration-dependent key at the same position.

Every protocol is built twice, from `typing` and from `typing_extensions`.
No `X | Y` or `list[int]`: this runs on 3.8.
"""

# stdlib
import dataclasses
import gc
import inspect
import sys
import types
import typing
import weakref

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.dispatchers import AmbiguousMethodError, NoMethodError
from bagof.dispatchers._function import Function, _call_key, _Plan
from bagof.dispatchers._lattice import (
    instance_members,
    is_declaration_dependent,
    is_value_dependent,
)
from bagof.dispatchers.core import _relation, ishintstance, issubhint

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

    @module.runtime_checkable
    class HasKind(module.Protocol):
        """A class variable: read off the class alone."""

        kind: module.ClassVar[str]

    @module.runtime_checkable
    class HasNameKind(module.Protocol):
        """An instance variable and a class variable."""

        name: str
        kind: module.ClassVar[str]

    class NameClassVar:
        """Declares `name` as a class variable, and holds no value."""

        name: module.ClassVar[str]

    class NameClassVarSet:
        """A class variable with a value: a class attribute."""

        name: module.ClassVar[str] = "class-level"

    class KindDeclared:
        """Declares `kind` as a class variable, with no value."""

        kind: module.ClassVar[str]

    class KindSet:
        kind = "class-level"

    class KindInstance:
        """Declares `kind` as an instance variable: not a class variable."""

        kind: str

    class NameKind:
        name: str
        kind: module.ClassVar[str]

    class Plain:
        """Declares nothing: an instance has `name` only if it is set."""

    class ClassAttr:
        name = "class-level"

    class Annotated:
        """A bare annotation: declares `name`, as a type checker reads it."""

        name: str

    class User:
        """Annotates `name` and sets it in `__init__`."""

        name: str

        def __init__(self, name: str) -> None:
            self.name = name

    class Inherits(Annotated):
        """Declares `name` through its base's annotation."""

    @dataclasses.dataclass
    class Record:
        name: str

    @dataclasses.dataclass
    class Person:
        name: str
        age: int

    @dataclasses.dataclass
    class InitFalse:
        """`__init__` never sets `name`, but the field annotates it."""

        name: str = dataclasses.field(init=False)

    @dataclasses.dataclass
    class InitFalseDefault:
        """`name` has a plain default, so the class holds it."""

        name: str = dataclasses.field(init=False, default="d")

    class Prop:
        @property
        def name(self) -> str:
            return "prop"

    class Slotted:
        __slots__ = ("name",)

    @dataclasses.dataclass
    class WithInitVar:
        """`name` is an init-only variable: never an attribute."""

        name: dataclasses.InitVar[str]

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


def _with_kind(cls: type) -> tx.Any:
    obj = cls()
    obj.kind = "instance-level"
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


def _unhashable_meta() -> type:
    class Meta(type):
        def __eq__(cls, other: tx.Any) -> bool:
            return cls is other

    assert Meta.__hash__ is None
    return Meta


def test_an_instance_of_an_unhashable_class_is_read(
    p: types.SimpleNamespace,
) -> None:
    """A class whose metaclass makes it unhashable cannot key the memo."""

    class Odd(metaclass=_unhashable_meta()):
        pass

    with pytest.raises(TypeError):
        hash(Odd)
    assert ishintstance(_with_name(Odd), p.HasName) is True
    assert ishintstance(Odd(), p.HasName) is False
    # Python's own `isinstance` raises for such an instance.
    with pytest.raises(TypeError):
        isinstance(Odd(), p.HasName)
    # The class itself is an ordinary value: its type is hashable.
    assert ishintstance(Odd, p.HasName) is False
    assert isinstance(Odd, p.HasName) is False


def test_an_unhashable_class_that_redefines_its_dict_is_read(
    p: types.SimpleNamespace,
) -> None:
    """Read through its class alone, which `getattr_static` cannot do from
    3.13 on: it memoises on the class too."""

    class Odd(metaclass=_unhashable_meta()):
        @property
        def __dict__(self) -> tx.Dict[str, tx.Any]:  # type: ignore[override]
            raise AssertionError("the value's own code must not run")

    class Named(Odd):
        name = "n"

    if sys.version_info >= (3, 13):
        with pytest.raises(TypeError):
            inspect.getattr_static(Odd(), "name")
    assert ishintstance(Odd(), p.HasName) is False
    assert ishintstance(Named(), p.HasName) is True


def test_a_class_made_at_runtime_is_not_kept_alive(
    p: types.SimpleNamespace,
) -> None:
    """The lookup memo holds the types of dispatched values weakly."""

    def make() -> weakref.ReferenceType:
        throwaway = type("Throwaway", (), {})
        assert ishintstance(_with_name(throwaway), p.HasName) is True
        assert throwaway in _relation._CLASS_READINGS
        return weakref.ref(throwaway)

    ref = make()
    gc.collect()
    assert ref() is None


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
        ("Annotated", True),
        ("User", True),
        ("Inherits", True),
        ("Record", True),
        ("Person", True),
        ("InitFalse", True),
        ("InitFalseDefault", True),
        ("NameClassVarSet", True),
        ("Prop", True),
        ("Slotted", True),
        ("Nominal", True),
        ("HasNameAge", True),
        ("HasName", True),
        ("Plain", False),
        ("NameClassVar", False),
        ("WithInitVar", False),
        ("OtherName", False),
        ("SupportsClose", False),
    ],
)
def test_what_is_below_a_data_protocol(
    p: types.SimpleNamespace, name: str, expected: bool
) -> None:
    assert issubhint(getattr(p, name), p.HasName) is expected


def test_an_annotation_declares_the_member(
    p: types.SimpleNamespace,
) -> None:
    """A type checker accepts `Annotated` as a `HasName`, set or not.

    So the value level counts an annotated member as present on every
    instance -- the order is sound only if an instance of a sub-hint is an
    instance of the super-hint. That is where it parts from `isinstance`,
    which finds no attribute on an instance that never set it.
    """
    value = p.Annotated()
    assert issubhint(p.Annotated, p.HasName) is True
    assert ishintstance(value, p.Annotated) is True
    assert ishintstance(value, p.HasName) is True
    assert isinstance(value, p.HasName) is False
    # Declared by a base, it is declared by the subclass too.
    assert issubhint(p.Inherits, p.HasName) is True
    assert ishintstance(p.Inherits(), p.HasName) is True
    # A dataclass field `__init__` never sets is annotated all the same.
    assert issubhint(p.InitFalse, p.HasName) is True
    assert ishintstance(p.InitFalse(), p.HasName) is True
    assert isinstance(p.InitFalse(), p.HasName) is False


def test_what_an_annotation_does_not_declare(
    p: types.SimpleNamespace,
) -> None:
    """A `ClassVar` is not an instance variable, and an `InitVar` is no
    attribute at all: a type checker rejects both as a `HasName`."""
    for cls in (p.NameClassVar, p.WithInitVar):
        assert issubhint(cls, p.HasName) is False
    assert ishintstance(p.NameClassVar(), p.HasName) is False
    assert ishintstance(p.WithInitVar("n"), p.HasName) is False
    # Holding a value, the class variable is a class attribute, which
    # counts as it always has: the value has it.
    assert issubhint(p.NameClassVarSet, p.HasName) is True
    assert ishintstance(p.NameClassVarSet(), p.HasName) is True


def test_a_class_variable_member(p: types.SimpleNamespace) -> None:
    """A `ClassVar` member is held or declared by the class, and only as a
    class variable -- never by an instance, never by an instance
    variable."""
    assert issubhint(p.KindDeclared, p.HasKind) is True
    assert issubhint(p.KindSet, p.HasKind) is True
    assert issubhint(p.KindInstance, p.HasKind) is False
    assert issubhint(p.Plain, p.HasKind) is False
    assert ishintstance(p.KindDeclared(), p.HasKind) is True
    assert ishintstance(p.KindSet(), p.HasKind) is True
    assert ishintstance(p.KindInstance(), p.HasKind) is False
    assert ishintstance(_with_kind(p.Plain), p.HasKind) is False
    # Read off the class alone, so it adds nothing to the call cache's key.
    assert instance_members(p.HasKind) == ()
    assert instance_members(p.HasNameKind) == ("name",)


def test_an_instance_and_a_class_variable_together(
    p: types.SimpleNamespace,
) -> None:
    assert issubhint(p.NameKind, p.HasNameKind) is True
    assert ishintstance(p.NameKind(), p.HasNameKind) is True
    assert issubhint(p.Annotated, p.HasNameKind) is False
    assert issubhint(p.KindDeclared, p.HasNameKind) is False
    named = _with_name(p.KindSet)
    assert ishintstance(named, p.HasNameKind) is True
    assert ishintstance(p.KindSet(), p.HasNameKind) is False
    f = Function("f")
    f.register((p.HasNameKind,))(lambda x: "name-kind")
    f.register((object,))(lambda x: "object")
    for _ in range(2):
        assert f(named) == "name-kind"
        assert f(p.KindSet()) == "object"
        assert f(p.NameKind()) == "name-kind"


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
    lambda p: p.User("u"),
    lambda p: p.User.__new__(p.User),
    lambda p: p.Inherits(),
    lambda p: p.WithInitVar("n"),
    lambda p: p.NameClassVar(),
    lambda p: p.NameClassVarSet(),
    lambda p: p.KindDeclared(),
    lambda p: p.KindSet(),
    lambda p: _with_name(p.KindSet),
    lambda p: _with_kind(p.Plain),
    lambda p: p.KindInstance(),
    lambda p: p.NameKind(),
    lambda p: p.Record("r"),
    lambda p: p.Person("r", 1),
    lambda p: p.InitFalse(),
    lambda p: p.InitFalseDefault(),
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
    "User",
    "Inherits",
    "WithInitVar",
    "NameClassVar",
    "NameClassVarSet",
    "HasKind",
    "HasNameKind",
    "KindDeclared",
    "KindSet",
    "KindInstance",
    "NameKind",
    "Record",
    "Person",
    "InitFalse",
    "InitFalseDefault",
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


# --- what an annotation declares ---------------------------------------


@pytest.mark.parametrize(
    "annotation, kind",
    [
        (str, _relation._INSTANCE_VARIABLE),
        (tx.List[int], _relation._INSTANCE_VARIABLE),
        (tx.Annotated[str, "m"], _relation._INSTANCE_VARIABLE),
        (typing.ClassVar, _relation._CLASS_VARIABLE),
        (typing.ClassVar[int], _relation._CLASS_VARIABLE),
        (tx.ClassVar[int], _relation._CLASS_VARIABLE),
        (dataclasses.InitVar, _relation._NOT_AN_ATTRIBUTE),
        (dataclasses.InitVar[int], _relation._NOT_AN_ATTRIBUTE),
        (tx.ForwardRef("Undefined"), _relation._INSTANCE_VARIABLE),
        (tx.ForwardRef("ClassVar[int]"), _relation._CLASS_VARIABLE),
        ("int", _relation._INSTANCE_VARIABLE),
        ("ClassVar", _relation._CLASS_VARIABLE),
        ("ClassVar[int]", _relation._CLASS_VARIABLE),
        (" typing . ClassVar [int]", _relation._CLASS_VARIABLE),
        ("tx.Annotated[ClassVar[int], 'm']", _relation._CLASS_VARIABLE),
        ("Annotated[int, 'm']", _relation._INSTANCE_VARIABLE),
        ("Annotated", _relation._INSTANCE_VARIABLE),
        ("dataclasses.InitVar[int]", _relation._NOT_AN_ATTRIBUTE),
        ("KW_ONLY", _relation._NOT_AN_ATTRIBUTE),
        ("List[ClassVar[int]]", _relation._INSTANCE_VARIABLE),
        ("", _relation._INSTANCE_VARIABLE),
        ("'int'", _relation._INSTANCE_VARIABLE),
    ],
)
def test_the_kind_an_annotation_declares(
    annotation: tx.Any, kind: int
) -> None:
    """Read by its marker; text is read by name, never evaluated."""
    assert _relation._annotation_kind(annotation) == kind


def test_a_class_variable_inside_annotated() -> None:
    try:
        annotation = tx.Annotated[tx.ClassVar[int], "m"]
    except TypeError:  # pragma: no cover -- refused before 3.11
        pytest.skip("Annotated[ClassVar[...]] is refused on this Python")
    assert _relation._annotation_kind(annotation) == _relation._CLASS_VARIABLE


@pytest.mark.skipif(sys.version_info < (3, 10), reason="KW_ONLY is 3.10+")
def test_a_kw_only_marker_declares_nothing(p: types.SimpleNamespace) -> None:
    @dataclasses.dataclass
    class KeywordOnly:
        _: dataclasses.KW_ONLY  # type: ignore[name-defined]
        name: str

    @tx.runtime_checkable
    class HasUnderscore(tx.Protocol):
        _: int

    assert issubhint(KeywordOnly, p.HasName) is True
    assert issubhint(KeywordOnly, HasUnderscore) is False
    assert ishintstance(KeywordOnly(name="k"), HasUnderscore) is False


def test_annotations_written_as_text(p: types.SimpleNamespace) -> None:
    """Under `from __future__ import annotations` every annotation is a
    string; what it declares is read from its text."""
    import _future_annotations_protocols as fa

    for cls in (fa.Annotated, fa.Quoted):
        assert issubhint(cls, p.HasName) is True
        assert ishintstance(cls(), p.HasName) is True
    for cls in (
        fa.NameClassVar,
        fa.NameDottedClassVar,
        fa.NameAnnotatedClassVar,
        fa.WithInitVar,
    ):
        assert issubhint(cls, p.HasName) is False, cls
    assert ishintstance(fa.WithInitVar("n"), p.HasName) is False
    assert issubhint(fa.KindDeclared, p.HasKind) is True
    assert issubhint(fa.KindInstance, p.HasKind) is False


def test_a_protocol_written_as_text() -> None:
    """A protocol's own `ClassVar` members are read the same way."""
    namespace = {}  # type: tx.Dict[str, tx.Any]
    exec(  # noqa: S102 -- a fixed source, for the future import
        "from __future__ import annotations\n"
        "import typing_extensions as tx\n"
        "@tx.runtime_checkable\n"
        "class HasKind(tx.Protocol):\n"
        "    kind: tx.ClassVar[str]\n"
        "    name: str\n",
        namespace,
    )
    members = _relation._data_protocol_members(namespace["HasKind"])
    assert members is not None
    assert members.data == ("name",)
    assert members.class_variables == ("kind",)


def test_a_sub_protocol_redeclares_a_member() -> None:
    """The nearest annotation says which kind a protocol's member is."""

    @tx.runtime_checkable
    class HasKind(tx.Protocol):
        kind: tx.ClassVar[str]

    @tx.runtime_checkable
    class HasInstanceKind(HasKind, tx.Protocol):
        kind: str  # type: ignore[misc]

    assert _relation._data_protocol_members(HasKind).class_variables == (
        "kind",
    )
    members = _relation._data_protocol_members(HasInstanceKind)
    assert members.data == ("kind",)
    assert members.class_variables == ()


def test_a_typeddict_declares_keys_not_attributes(
    p: types.SimpleNamespace,
) -> None:
    """A `TypedDict` value is a plain `dict`, which has no `name`."""

    class Movie(tx.TypedDict):
        name: str

    assert issubhint(Movie, p.HasName) is False
    assert ishintstance(Movie(name="m"), p.HasName) is False


def test_a_metaclass_annotation_declares_for_its_classes(
    p: types.SimpleNamespace,
) -> None:
    """A class object is an instance of its metaclass: the metaclass's
    annotations are what declare its members, not the class's own."""

    class Meta(type):
        name: str

    class Made(metaclass=Meta):
        pass

    assert issubhint(Meta, p.HasName) is True
    assert ishintstance(Made, p.HasName) is True
    assert ishintstance(p.Annotated, p.HasName) is False


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="annotations are lazy from 3.14"
)
def test_an_annotation_that_cannot_be_evaluated_declares_nothing(
    p: types.SimpleNamespace,
) -> None:
    namespace = {}  # type: tx.Dict[str, tx.Any]
    exec(  # noqa: S102 -- raises only when the annotation is evaluated
        "class Broken:\n    name: 1 / 0\n", namespace
    )
    broken = namespace["Broken"]
    assert issubhint(broken, p.HasName) is False
    assert ishintstance(broken(), p.HasName) is False


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
    assert f(p.Annotated()) == "named"
    assert f.resolve(p.ClassAttr).function(None) == "named"
    assert f.resolve(p.Annotated).function(None) == "named"
    assert f.resolve(p.WithInitVar).function(None) == "object"


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


def _user_or_named(p: types.SimpleNamespace, cls: type) -> Function:
    f = Function("f")
    f.register((p.HasName,))(lambda x: "protocol")
    f.register((cls,))(lambda x: "class")
    return f


def test_a_class_that_annotates_the_members_is_more_specific(
    p: types.SimpleNamespace,
) -> None:
    """`User` annotates `name` and sets it in `__init__`: a type checker
    accepts it as a `HasName`, so its overload is the more specific one."""
    assert issubhint(p.User, p.HasName) is True
    assert issubhint(p.HasName, p.User) is False
    f = _user_or_named(p, p.User)
    for _ in range(2):
        assert f(p.User("u")) == "class"
        assert f(_with_name(p.Plain)) == "protocol"
    g = Function("g")

    @g.register
    def _protocol(*, x: p.HasName) -> str:  # type: ignore[name-defined]
        return "protocol"

    @g.register
    def _class(*, x: p.User) -> str:  # type: ignore[name-defined]
        return "class"

    for _ in range(2):
        assert g(x=p.User("u")) == "class"
        assert g(x=_with_name(p.Plain)) == "protocol"
    assert f.ambiguities() == []


def test_an_annotated_member_never_set(p: types.SimpleNamespace) -> None:
    """The value check and dispatch take the annotation at its word."""
    unset = p.User.__new__(p.User)
    assert ishintstance(unset, p.User) is True
    assert ishintstance(unset, p.HasName) is True
    f = _user_or_named(p, p.User)
    assert f(unset) == "class"
    g = _named_or_not(p)
    assert g(unset) == "named"
    assert g(p.Annotated()) == "named"


def test_a_class_that_declares_nothing_stays_ambiguous(
    p: types.SimpleNamespace,
) -> None:
    """`Plain` declares no `name`: an instance that gains one is in both,
    and neither overload is more specific."""
    assert issubhint(p.Plain, p.HasName) is False
    assert issubhint(p.HasName, p.Plain) is False
    f = _user_or_named(p, p.Plain)
    for _ in range(2):
        with pytest.raises(AmbiguousMethodError):
            f(_with_name(p.Plain))
        assert f(p.Plain()) == "class"


def test_an_annotated_member_keys_the_same_for_every_instance(
    p: types.SimpleNamespace,
) -> None:
    """Declared by the class, `name` is present on every instance, so set
    or not, they share one key -- and one cache entry."""
    f = _named_or_not(p)
    f(p.User("u"))
    plan = _plan_of(f, (1, ()))
    unset = p.User.__new__(p.User)
    assert _call_key((unset,), {}, plan) == (1, (p.User, (True,)))
    assert _call_key((p.User("u"),), {}, plan) == (1, (p.User, (True,)))
    f(unset)
    assert len(f._cache.call_cache) == 1


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


def test_a_constrained_typevar_is_solved_from_the_class(
    p: types.SimpleNamespace,
) -> None:
    """A documented limit: the constraint is chosen from `type(v)`."""
    constrained = tx.TypeVar("constrained", p.HasName, int)
    bound = tx.TypeVar("bound", bound=p.HasName)
    f = Function("f")
    f.register((constrained,))(lambda x: "constrained")
    f.register((object,))(lambda x: "object")
    g = Function("g")
    g.register((bound,))(lambda x: "bound")
    g.register((object,))(lambda x: "object")
    # By its class, the constraint is chosen...
    assert f(p.ClassAttr()) == "constrained"
    # ...but not by what the instance alone holds; a bound reads that.
    assert f(_with_name(p.Plain)) == "object"
    assert g(_with_name(p.Plain)) == "bound"


def test_a_generic_data_protocol_checks_presence_only() -> None:
    """A structural value declares no arguments to compare."""
    T = tx.TypeVar("T")

    @tx.runtime_checkable
    class HasItem(tx.Protocol[T]):
        item: T

    class Holder:
        pass

    held = Holder()
    held.item = "not an int"
    assert ishintstance(held, HasItem[int]) is True
