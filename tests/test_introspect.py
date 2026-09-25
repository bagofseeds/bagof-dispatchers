"""Tests for the hint-introspection helpers."""

# stdlib
import sys

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers.core import (
    get_concrete_type,
    ishintstance,
    issubclassable,
    issubscriptable,
    safe_isinstance,
    safe_issubclass,
    type2hint,
    unwrap,
)
from bagof.dispatchers.core._introspect import (
    _typing_spelling,
    _unwrap_typevar,
)


class Base(tx.TypedDict):
    a: int


class Middle(Base):
    b: int


class Leaf(Middle):
    c: int


# --- ishintstance against `type[T]` -----------------------------------


@pytest.mark.parametrize(
    "obj,hint,expected",
    [
        # Regression: the argument of `type[T]` used to be ignored, so
        # every class validated against every `type[...]`.
        (bool, tx.Type[int], True),
        (str, tx.Type[int], False),
        (int, tx.Type[int], True),
        # A bare `type` still accepts any class...
        (str, tx.Type, True),
        # ... and neither form accepts a non-class.
        (3, tx.Type[int], False),
        (3, tx.Type, False),
        # `Annotated` is transparent here, like everywhere else.
        (bool, tx.Annotated[tx.Type[int], "meta"], True),
        (str, tx.Annotated[tx.Type[int], "meta"], False),
    ],
)
def test_ishintstance_type(obj: tx.Any, hint: tx.Any, expected: bool) -> None:
    assert ishintstance(obj, hint) is expected


# --- typevar cycles ----------------------------------------------------


def test_the_reentrancy_guard_terminates() -> None:
    # Regression: the guard returned the typevar, which sent `unwrap`
    # straight back into it - turning one infinite recursion into
    # another. Drive the guard directly, so this holds on every Python.
    typevar = tx.TypeVar("typevar")
    assert _unwrap_typevar(typevar, (typevar,)) is tx.Any


def test_unwrap_terminates_on_a_typevar_cycle() -> None:
    first = tx.TypeVar("first")
    second = tx.TypeVar("second")
    try:
        first.__default__ = second
        second.__default__ = first
    except AttributeError:  # pragma: no cover
        # `__default__` is a read-only slot from python 3.13 on, so the
        # cycle cannot be built there. The guard itself is covered above.
        pytest.skip("TypeVar.__default__ is not writable")
    assert unwrap(first, (tx.Annotated, tx.TypeVar)) is tx.Any


# --- tuples, like the builtins ----------------------------------------


@pytest.mark.parametrize(
    "obj,classes,expected",
    [
        (1, (int, str), True),
        (1.5, (int, str), False),
        (1, (), False),
        # Still safe: a non-type member is skipped, not raised on.
        (1, (int, "not a type"), True),
        (1.5, (int, "not a type"), False),
        # A TypedDict cannot be instance-checked at all, so a dict is
        # not an instance of one.
        ({"a": 1}, (Base,), False),
    ],
)
def test_safe_isinstance_accepts_a_tuple(
    obj: tx.Any, classes: tx.Any, expected: bool
) -> None:
    assert safe_isinstance(obj, classes) is expected


@pytest.mark.parametrize(
    "subcls,classes,expected",
    [
        (bool, (int, str), True),
        (float, (int, str), False),
        (bool, (), False),
        (bool, (int, "not a type"), True),
    ],
)
def test_safe_issubclass_accepts_a_tuple(
    subcls: tx.Any, classes: tx.Any, expected: bool
) -> None:
    assert safe_issubclass(subcls, classes) is expected


# --- TypedDicts are not instance-checkable ----------------------------


def test_typeddict_is_not_instance_checkable() -> None:
    # Python refuses `isinstance(value, SomeTypedDict)` outright, and a
    # TypedDict leaves no trace on the dict it describes - so there is
    # nothing to recognise at runtime.
    with pytest.raises(TypeError):
        isinstance({"a": 1}, Base)
    assert safe_isinstance({"a": 1}, Base) is False
    assert safe_isinstance({"wrong": 1}, Base) is False


def test_dict_is_not_a_subclass_of_a_typeddict() -> None:
    # A TypedDict is a dict; a dict is not a TypedDict.
    assert safe_issubclass(dict, Base) is False
    assert safe_issubclass(Base, tx.TypedDict) is True
    assert safe_issubclass(Middle, Base) is True


# --- Any is never type-like -------------------------------------------


def test_any_is_not_subclassable_on_any_version() -> None:
    # `typing.Any` became a class in 3.11, so `isinstance(Any, type)`
    # answers differently across the versions this package supports.
    # Pin the answer instead of inheriting it.
    assert issubclassable(tx.Any) is False
    assert safe_issubclass(tx.Any, object) is False
    assert safe_issubclass(int, tx.Any) is False


# --- get_concrete_type -------------------------------------------------


def test_get_concrete_type_uses_the_fallback() -> None:
    # A union has no concrete origin, so the fallback is used.
    assert get_concrete_type(tx.Union[int, str], list) is list


def test_get_concrete_type_without_a_usable_fallback_raises() -> None:
    with pytest.raises(TypeError, match="Cannot get concrete type"):
        get_concrete_type(tx.Union[int, str])
    with pytest.raises(TypeError, match="Cannot get concrete type"):
        get_concrete_type(tx.Union[int, str], "not a type")


def test_get_concrete_type_skips_an_abstract_or_special_origin() -> None:
    # stdlib
    from collections import abc

    # `Sequence` is abstract, so the fallback wins.
    assert get_concrete_type(tx.Sequence[int], list) is list
    assert get_concrete_type(abc.Sequence, list) is list
    # `Union` is a class from 3.14 on, but still not instantiable.
    assert get_concrete_type(tx.Union, list) is list


def test_get_concrete_type_of_a_constrained_typevar() -> None:
    assert get_concrete_type(tx.TypeVar("T", int, str)) is int
    # An unconstrained typevar has no constraint to fall back on.
    with pytest.raises(TypeError):
        get_concrete_type(tx.TypeVar("T"))


# --- safe_issubclass ---------------------------------------------------


def test_safe_issubclass_with_a_non_type_second_argument() -> None:
    assert safe_issubclass(int, "not a type") is False
    assert safe_issubclass("not a type", int) is False


# --- ishintstance, continued -------------------------------------------


def test_ishintstance_of_a_bare_union_asks_whether_it_is_one() -> None:
    # A bare `Union` is not a type to check against: the question becomes
    # "is this value's type a union?", which no value's type ever is.
    assert ishintstance(1, tx.Union) is False


def test_ishintstance_type_rejects_a_non_type_hint() -> None:
    # local
    from bagof.dispatchers.core._relation import _ishintstance_type

    with pytest.raises(TypeError, match="is not a type"):
        _ishintstance_type(int, int)


# --- unwrap ------------------------------------------------------------


def test_unwrap_with_no_origins_is_a_no_op() -> None:
    hint = tx.Annotated[int, "meta"]
    assert unwrap(hint, None) is hint
    assert unwrap(hint, ()) is hint


# --- type2hint ---------------------------------------------------------


def test_type2hint_leaves_a_subscriptable_value_alone() -> None:
    value = [1, 2, 3]
    assert type2hint(value) is value


def test_type2hint_leaves_an_unhashable_value_alone() -> None:
    # A set is neither subscriptable nor hashable, so there is no key to
    # look up and nothing to convert.
    value = {1, 2}
    assert type2hint(value) is value


# --- issubscriptable ---------------------------------------------------


def test_issubscriptable() -> None:
    # A class with `__class_getitem__` is subscriptable. Use an explicit
    # class rather than `list`, whose `__class_getitem__` only exists on
    # Python 3.9+ (PEP 585).
    class Sub:
        def __class_getitem__(cls, item: tx.Any) -> type:
            return cls

    assert issubscriptable(Sub) is True
    # An instance with `__getitem__` is too.
    assert issubscriptable([1, 2]) is True
    # A plain value is not.
    assert issubscriptable(3) is False


# --- unwrap follows a typevar bound ------------------------------------


def test_unwrap_typevar_follows_a_bound() -> None:
    # A bound typevar (no default, no constraints) unwraps to its bound.
    bound = tx.TypeVar("bound", bound=int)
    assert _unwrap_typevar(bound) is int
    assert unwrap(bound, tx.TypeVar) is int


# --- _typing_spelling: new-style generics rewritten to `typing` --------


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="list[int] needs PEP 585 (3.9+)"
)
def test_typing_spelling_rewrites_new_style_generics() -> None:
    # stdlib
    import collections.abc as cabc

    assert _typing_spelling(list[int]) == tx.List[int]
    assert _typing_spelling(dict[str, int]) == tx.Dict[str, int]
    # Nested inside Annotated, Union, and Callable.
    assert _typing_spelling(tx.Annotated[list[int], "m"]) == tx.Annotated[
        tx.List[int], "m"
    ]
    assert _typing_spelling(tx.Union[list[int], str]) == tx.Union[
        tx.List[int], str
    ]
    assert _typing_spelling(cabc.Callable[[list[int]], int]) == tx.Callable[
        [tx.List[int]], int
    ]
    # `tuple[()]` is the empty-tuple type.
    assert _typing_spelling(tuple[()]) == tx.Tuple[()]


def test_typing_spelling_leaves_others_unchanged() -> None:
    # No origin, or a `Literal` (its arguments are values), is left alone.
    assert _typing_spelling(int) is int
    assert _typing_spelling(tx.Literal[1]) == tx.Literal[1]
    # An Annotated whose inner type needs no rewrite is returned unchanged.
    assert _typing_spelling(tx.Annotated[int, "m"]) == tx.Annotated[int, "m"]
