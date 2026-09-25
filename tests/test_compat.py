"""Tests for the version-pinning and special-form recognition in `_compat`."""

# dependencies
import typing_extensions as tx

# local
from bagof.dispatchers.core._compat import (
    is_plausible_hint,
    is_special_form,
    spellings,
)

# --- is_special_form: the markers it must recognise and must not -------


def test_generic_protocol_typeddict_markers_are_not_special_forms() -> None:
    # `Generic`, `Protocol` and the `TypedDict` marker are real bases a user
    # subclasses, not special forms.
    assert is_special_form(tx.Generic) is False
    assert is_special_form(tx.Protocol) is False
    assert is_special_form(tx.TypedDict) is False


def test_a_generic_subclass_is_not_a_special_form() -> None:
    # `typing.IO` is a `Generic` subclass: reached through the MRO branch.
    assert is_special_form(tx.IO) is False


def test_a_protocol_is_not_a_special_form() -> None:
    assert is_special_form(tx.SupportsInt) is False
    assert is_special_form(tx.SupportsIndex) is False


def test_a_future_typing_class_form_is_a_special_form() -> None:
    # The structural fallback's positive case: a plain class living in
    # `typing` that is not Generic/Protocol/TypedDict/ABC stands in for a
    # future class-shaped special form and is recognised as one.
    class _FutureForm:
        pass

    _FutureForm.__module__ = "typing"
    assert is_special_form(_FutureForm) is True


def test_the_explicit_special_forms_are_recognised() -> None:
    assert is_special_form(tx.Any) is True
    assert is_special_form(tx.Literal) is True


# --- is_plausible_hint -------------------------------------------------


def test_is_plausible_hint_accepts_hints() -> None:
    assert is_plausible_hint(int) is True  # a class
    assert is_plausible_hint(tx.Any) is True  # a special form (a class ≥3.11)
    # A special form that is *not* a class (so it reaches the special-form
    # branch rather than the `isinstance(obj, type)` one).
    assert is_plausible_hint(tx.Literal) is True
    assert is_plausible_hint(tx.TypeVar("T")) is True  # TypeVar family
    assert is_plausible_hint(tx.ParamSpec("P")) is True
    assert is_plausible_hint(tx.ForwardRef("Foo")) is True
    assert is_plausible_hint(tx.Self) is True  # a typing-module object


def test_is_plausible_hint_rejects_non_hints() -> None:
    assert is_plausible_hint(123) is False
    assert is_plausible_hint("Foo") is False
    assert is_plausible_hint([1, 2]) is False
    assert is_plausible_hint(lambda: None) is False


# --- spellings ---------------------------------------------------------


def test_spellings_deduplicates() -> None:
    # Every entry is distinct, and `Any` is among them.
    forms = spellings("Any")
    assert tx.Any in forms
    assert len(forms) == len(set(map(id, forms)))
