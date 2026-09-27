"""Malformed `TypedDict`s (PEP 728) are refused at registration (#42).

`issubhint` on a concrete `TypedDict` is nominal: it reads the base chain, not
closedness. For a *well-formed* PEP 728 `TypedDict` that is sound, because a
type checker forbids a subclass that would break `v in Sub and Sub <= Base =>
v in Base`. The `typing_extensions` runtime still lets you build such a
subclass, so a **malformed** one -- one a type checker rejects -- is refused
when a method is registered with it, rather than left to mis-dispatch.

The four malformed shapes (per PEP 728):

1. adds a declared key to a *closed* base;
2. adds a declared key incompatible with a base's `extra_items`;
3. widens a base's `extra_items`;
4. reopens a closed base (with `extra_items=` or `closed=False`).

The `closed=` / `extra_items=` keywords exist only on new-enough
`typing_extensions`; tests that need them are skipped where they cannot be
expressed, mirroring `test_typeddict_closed.py`.
"""

# stdlib
import typing

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers._function import Function
from bagof.dispatchers._signature import Signature


def _closed_supported() -> bool:
    """Whether this `typing_extensions` can express a closed `TypedDict`."""
    try:

        class _Probe(tx.TypedDict, closed=True):
            a: int

    except Exception:
        return False
    return getattr(_Probe, "__closed__", None) is True


def _extra_items_supported() -> bool:
    """Whether this `typing_extensions` can express `extra_items=`."""
    try:

        class _Probe(tx.TypedDict, extra_items=int):
            a: int

    except Exception:
        return False
    return getattr(_Probe, "__extra_items__", None) is int


CLOSED = pytest.mark.skipif(
    not _closed_supported(),
    reason="typing_extensions here cannot express closed=True",
)
EXTRA = pytest.mark.skipif(
    not _extra_items_supported(),
    reason="typing_extensions here cannot express extra_items=",
)


def _register_param(hint: typing.Any) -> Function:
    """Register a one-parameter method whose parameter is `hint`."""
    function = Function("f")

    def impl(x: hint) -> int:  # noqa: ANN001 -- the hint under test
        return 0

    # The annotation is already an object here, not a string, so there is no
    # forward-reference deferral: the malformed check fires at registration.
    function.register(impl)
    return function


# --- 1: adds a declared key to a closed base ---------------------------


@CLOSED
def test_adds_key_to_closed_base_is_rejected() -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Adds(Base):
        b: str

    with pytest.raises(TypeError) as info:
        _register_param(Adds)
    message = str(info.value)
    assert "Adds" in message
    assert "malformed" in message
    assert "'b'" in message
    assert "closed base Base" in message


# --- 2: adds a key incompatible with extra_items ------------------------


@EXTRA
def test_adds_incompatible_key_to_typed_base_is_rejected() -> None:
    class Base(tx.TypedDict, extra_items=int):
        a: int

    class Adds(Base):
        b: str  # str is not an int

    with pytest.raises(TypeError) as info:
        _register_param(Adds)
    message = str(info.value)
    assert "Adds" in message
    assert "malformed" in message
    assert "'b'" in message
    assert "extra_items" in message
    assert "Base" in message


# --- 3: widens extra_items ----------------------------------------------


@EXTRA
def test_widening_extra_items_is_rejected() -> None:
    class Base(tx.TypedDict, extra_items=int):
        a: int

    class Widens(Base, extra_items=object):
        pass

    with pytest.raises(TypeError) as info:
        _register_param(Widens)
    message = str(info.value)
    assert "Widens" in message
    assert "malformed" in message
    assert "widens" in message
    assert "extra_items" in message
    assert "Base" in message


# --- 4a: closed -> typed (reopen with extra_items) ----------------------


@CLOSED
@EXTRA
def test_setting_extra_items_on_closed_base_is_rejected() -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Reopens(Base, extra_items=int):
        pass

    with pytest.raises(TypeError) as info:
        _register_param(Reopens)
    message = str(info.value)
    assert "Reopens" in message
    assert "malformed" in message
    assert "extra_items" in message
    assert "closed base Base" in message


# --- 4b: reopen a closed base with closed=False -------------------------


@CLOSED
def test_reopening_closed_base_is_rejected() -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Reopens(Base, closed=False):
        z: int

    with pytest.raises(TypeError) as info:
        _register_param(Reopens)
    message = str(info.value)
    assert "Reopens" in message
    assert "malformed" in message
    # Reopening is reported (adding a key to the still-closed base would be
    # too; either names the class and the closed base).
    assert "Base" in message


# --- well-formed subclasses are accepted --------------------------------


@CLOSED
def test_plain_subclass_of_closed_base_is_accepted() -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Plain(Base):
        pass

    # Registration must not raise.
    _register_param(Plain)


@EXTRA
def test_narrowing_extra_items_is_accepted() -> None:
    class Base(tx.TypedDict, extra_items=int):
        a: int

    class Narrows(Base, extra_items=bool):
        pass

    _register_param(Narrows)


@EXTRA
def test_compatible_added_key_under_typed_base_is_accepted() -> None:
    class Base(tx.TypedDict, extra_items=int):
        a: int

    class Adds(Base):
        b: bool  # bool is an int

    _register_param(Adds)


def test_subclass_of_open_base_is_accepted() -> None:
    class OpenBase(tx.TypedDict):
        a: int

    class Adds(OpenBase):
        b: str

    _register_param(Adds)


@CLOSED
def test_standalone_closed_typeddict_is_accepted() -> None:
    class Closed(tx.TypedDict, closed=True):
        a: int

    _register_param(Closed)


@EXTRA
def test_standalone_extra_items_typeddict_is_accepted() -> None:
    class Typed(tx.TypedDict, extra_items=int):
        a: int

    _register_param(Typed)


# --- reachable through the other registration paths ---------------------


@CLOSED
def test_from_hints_rejects_a_malformed_typeddict() -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Adds(Base):
        b: str

    with pytest.raises(TypeError) as info:
        Signature.from_hints(Adds)
    assert "malformed" in str(info.value)


@CLOSED
def test_overlay_register_rejects_a_malformed_typeddict() -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Adds(Base):
        b: str

    function = Function("f")
    with pytest.raises(TypeError) as info:

        @function.register((Adds,))
        def impl(x: typing.Any) -> int:
            return 0

    assert "malformed" in str(info.value)


@CLOSED
def test_top_level_optional_of_malformed_is_rejected() -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Adds(Base):
        b: str

    with pytest.raises(TypeError) as info:
        _register_param(tx.Optional[Adds])
    assert "malformed" in str(info.value)


# --- a well-formed TypedDict still dispatches by shape ------------------


@CLOSED
def test_registered_well_formed_closed_typeddict_dispatches() -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Plain(Base):
        pass

    function = Function("f")

    def on_plain(x: Plain) -> str:
        return "plain"

    function.register(on_plain)
    assert function({"a": 1}) == "plain"
