"""TypedDicts (PEP 728) that break dispatch soundness are refused (#42).

`issubhint` / `ishintstance` on a concrete `TypedDict` are nominal: they read
the base chain, not closedness. For a *well-formed* PEP 728 `TypedDict` that is
sound. The `typing_extensions` runtime still lets you build a subclass whose
nominal hint order disagrees with the value-level shape check, so a value could
be `in Sub` and `Sub <= Base` yet not `in Base`. Such a class is refused when a
method is registered with it, rather than left to mis-dispatch.

The gate rejects a **dispatch-sound subset** -- registrations that would break
`v in Sub and Sub <= Base => v in Base` under this library's relation -- which
is not the same as what a type checker enforces:

1. adds a key to a *closed* base;
2. adds a key whose value type a base's `extra_items` does not admit (including
   a key a sibling open base contributes in a diamond);
3. widens a base's `extra_items`;
4. reopens a closed base (with `extra_items=` or `closed=False`).

Two deviations from a type checker follow from resting on this library's
relation: an `Any`-typed key or `extra_items` is accepted (lenient), and there
is no numeric-tower promotion, so a key typed `int` under `extra_items=float`
*is* rejected (stricter). Some PEP-illegal but dispatch-safe shapes are
likewise accepted.

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
from bagof.dispatchers.core._relation import _malformed_typeddict_reason


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
def test_narrowing_extra_items_is_accepted_dispatch_safe() -> None:
    # Narrowing `extra_items` keeps every value of the subclass a value of the
    # base, so it is dispatch-safe and accepted.
    class Base(tx.TypedDict, extra_items=int):
        a: int

    class Narrows(Base, extra_items=bool):
        pass

    _register_param(Narrows)


@EXTRA
def test_compatible_added_key_under_typed_base_is_accepted_dispatch_safe(
) -> None:
    # A key whose type the base's `extra_items` admits keeps the subclass a
    # sub-shape of the base, so it is dispatch-safe and accepted.
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


# --- Any keys are accepted (lenient) ------------------------------------
#
# An `Any`-typed key or `extra_items` admits every value, so it can never
# carry something a base refuses: it is dispatch-safe and accepted, even
# though a type checker may still object.


@EXTRA
def test_any_added_key_under_typed_base_is_accepted_lenient() -> None:
    class Typed(tx.TypedDict, extra_items=int):
        a: int

    class Adds(Typed):
        b: tx.NotRequired[tx.Any]  # Any, through the qualifier

    # Accepted: without the Any-skip this would be rejected as incompatible.
    _register_param(Adds)


@EXTRA
def test_any_added_key_under_readonly_typed_base_is_accepted_lenient(
) -> None:
    class Typed(tx.TypedDict, extra_items=tx.ReadOnly[int]):
        a: int

    class Adds(Typed):
        b: tx.Any

    _register_param(Adds)


@EXTRA
def test_any_extra_items_is_accepted_lenient() -> None:
    class Typed(tx.TypedDict, extra_items=int):
        a: int

    class Widens(Typed, extra_items=tx.Any):
        pass

    # An `Any` `extra_items` admits every value, so widening is skipped.
    _register_param(Widens)


# --- no numeric-tower promotion (stricter than a type checker) -----------


@EXTRA
def test_int_key_under_float_extra_items_is_rejected_no_promotion() -> None:
    # `issubhint(int, float)` is False -- this library has no numeric-tower
    # promotion -- so a value-level check would genuinely mis-dispatch, and the
    # key is rejected (stricter than a type checker, which accepts it).
    class Typed(tx.TypedDict, extra_items=float):
        a: int

    class Adds(Typed):
        b: int

    with pytest.raises(TypeError) as info:
        _register_param(Adds)
    assert "malformed" in str(info.value)
    assert "'b'" in str(info.value)


# --- 5: the diamond hazard (a real dispatch break) ----------------------


@CLOSED
def test_diamond_open_sibling_key_under_closed_base_is_rejected() -> None:
    # `issubhint(D, Closed)` is nominally True, but `{"a": 1, "c": "x"}` is in
    # D (D admits `c`, inherited from the open sibling) and not in Closed,
    # which admits no extra key -- a dispatch break. D is refused.
    class Closed(tx.TypedDict, closed=True):
        a: int

    class OpenWithC(tx.TypedDict):
        c: str

    class D(Closed, OpenWithC):
        pass

    with pytest.raises(TypeError) as info:
        _register_param(D)
    message = str(info.value)
    assert "D" in message
    assert "malformed" in message
    assert "'c'" in message
    assert "closed base Closed" in message


@CLOSED
def test_diamond_open_sibling_incompatible_key_under_typed_base_rejected(
) -> None:
    class Typed(tx.TypedDict, extra_items=int):
        a: int

    class OpenWithC(tx.TypedDict):
        c: str  # str is not an int

    class D(Typed, OpenWithC):
        pass

    with pytest.raises(TypeError) as info:
        _register_param(D)
    assert "malformed" in str(info.value)
    assert "'c'" in str(info.value)


@CLOSED
def test_diamond_of_two_plain_subclasses_adding_nothing_is_accepted(
) -> None:
    # Two plain subclasses of a closed base that add nothing form a legal
    # diamond -- no key crosses into the closed base -- so it stays accepted.
    class Closed(tx.TypedDict, closed=True):
        a: int

    class Left(Closed):
        pass

    class Right(Closed):
        pass

    class D(Left, Right):
        pass

    _register_param(D)


# --- 6: parametrised generic malformed TypedDict (through its origin) ----


@CLOSED
def test_parametrised_generic_malformed_typeddict_is_rejected() -> None:
    T = tx.TypeVar("T")
    try:

        class G(tx.TypedDict, tx.Generic[T], closed=True):
            a: T

        class GA(G[T]):
            b: str  # adds a key to the closed base

    except Exception:
        pytest.skip("generic closed TypedDict not expressible here")

    with pytest.raises(TypeError) as info:
        _register_param(GA[int])
    assert "malformed" in str(info.value)
    assert "'b'" in str(info.value)


@CLOSED
def test_optional_of_parametrised_generic_malformed_is_rejected() -> None:
    T = tx.TypeVar("T")
    try:

        class G(tx.TypedDict, tx.Generic[T], closed=True):
            a: T

        class GA(G[T]):
            b: str

    except Exception:
        pytest.skip("generic closed TypedDict not expressible here")

    with pytest.raises(TypeError) as info:
        _register_param(tx.Optional[GA[int]])
    assert "malformed" in str(info.value)


# --- message quality ----------------------------------------------------


@CLOSED
def test_from_hints_positional_names_the_slot_not_the_internal_name(
) -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Adds(Base):
        b: str

    with pytest.raises(TypeError) as info:
        Signature.from_hints(Adds)
    message = str(info.value)
    assert "positional hint 0" in message
    # The synthetic internal name never surfaces.
    assert "'_0'" not in message


@CLOSED
def test_error_names_the_function_by_name() -> None:
    class Base(tx.TypedDict, closed=True):
        a: int

    class Adds(Base):
        b: str

    def impl(x: Adds) -> int:
        return 0

    function = Function("f")
    with pytest.raises(TypeError) as info:
        function.register(impl)
    # Rendered as the function's name, not its full repr.
    assert "of impl" in str(info.value)


# --- well-formed redundant re-closing is accepted -----------------------


@CLOSED
def test_redeclaring_closed_on_a_closed_base_is_accepted() -> None:
    # A subclass that repeats `closed=True` adds no key and reopens nothing, so
    # it is well-formed and accepted (it exercises the own-policy read).
    class Base(tx.TypedDict, closed=True):
        a: int

    class Sub(Base, closed=True):
        pass

    _register_param(Sub)


# --- an added forward-ref key is left unjudged --------------------------


@CLOSED
def test_added_unresolvable_forward_ref_key_is_not_judged() -> None:
    # A key whose hint is an unresolvable forward reference carries no type to
    # compare, so it is skipped rather than guessed at, and the class is
    # accepted at registration.
    class Base(tx.TypedDict, closed=True):
        a: int

    class Adds(Base):
        b: "_NeverDefinedName"  # noqa: F821 -- intentionally unresolvable

    _register_param(Adds)


# --- the reason helper's guards -----------------------------------------


def test_reason_is_none_for_a_non_typeddict() -> None:
    # Not a concrete TypedDict: there is nothing to judge.
    assert _malformed_typeddict_reason(int) is None
    assert _malformed_typeddict_reason(tx.TypedDict) is None
