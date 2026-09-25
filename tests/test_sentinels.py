"""Tests for the shared `Unset` / `UNSET` sentinel."""

# local
from bagof.dispatchers.core import UNSET, Unset


def test_unset_is_falsey_and_reprs() -> None:
    assert bool(UNSET) is False
    assert not UNSET
    assert repr(UNSET) == "<UNSET>"
    assert str(UNSET) == "<UNSET>"


def test_unset_is_a_singleton() -> None:
    assert Unset() is UNSET
