"""Tests for the hint-keyed lookup (`core/_registry.py`)."""

# stdlib
import typing

# dependencies
import pytest

# locals
from bagof.dispatchers import Exact
from bagof.dispatchers._errors import AmbiguousMethodError, NoMethodError
from bagof.dispatchers.core import UNSET, resolve_hint


def test_exact_match_wins() -> None:
    """A key equal to the query is returned before the relation is read."""
    registry = {int: "int", object: "any"}
    assert resolve_hint(int, registry) == "int"


def test_nearest_superhint() -> None:
    """A subclass query finds its nearest registered superclass."""
    registry = {int: "number", object: "any"}
    assert resolve_hint(bool, registry) == "number"
    assert resolve_hint(str, registry) == "any"


def test_any_key_is_reachable() -> None:
    """An `Any` key is the catch-all every query reaches."""
    registry = {typing.Any: "anything"}
    assert resolve_hint(int, registry) == "anything"


def test_union_key_matches_a_member() -> None:
    """A `Union` key matches any of its members."""
    registry = {typing.Union[int, str]: "either"}
    assert resolve_hint(int, registry) == "either"
    assert resolve_hint(str, registry) == "either"


def test_covariant_generic_key() -> None:
    """A `List[int]` key matches a `List[bool]` query (covariant)."""
    registry = {typing.List[int]: "ints"}
    assert resolve_hint(typing.List[bool], registry) == "ints"


def test_typing_spelling_bridges_builtin_and_typing() -> None:
    """A key written one spelling is reached by a query written the other."""
    registry = {typing.List[int]: "ints"}
    assert resolve_hint(list, {list: "list"}) == "list"
    # A new-style query hits its typing-spelled exact key.
    assert resolve_hint(typing.List[int], registry) == "ints"


def test_unhashable_key_does_not_raise() -> None:
    """A key that cannot be hashed is simply not an exact key."""
    registry = {int: "int"}
    # A bare list is unhashable; iterating over the keys still works.
    assert resolve_hint(bool, registry) == "int"


def test_unhashable_query_hint() -> None:
    """An unhashable query hint is not an exact key and falls through."""
    # A list is unhashable; it cannot be an exact key, and nothing accepts it.
    assert resolve_hint([1, 2], {int: "n"}, default="fallback") == "fallback"


def test_none_is_read_as_nonetype() -> None:
    """A bare `None` query is matched as `NoneType`."""
    registry = {type(None): "none"}
    assert resolve_hint(None, registry) == "none"


def test_default_when_nothing_matches() -> None:
    """A default is returned when no key accepts the query."""
    assert resolve_hint(list, {int: "n"}, default="fallback") == "fallback"


def test_no_match_raises_without_default() -> None:
    """Without a default, no match raises `NoMethodError`."""
    with pytest.raises(NoMethodError):
        resolve_hint(list, {int: "n"})


def test_no_match_empty_registry() -> None:
    """An empty registry names itself in the error."""
    with pytest.raises(NoMethodError, match="registry is empty"):
        resolve_hint(int, {})


def test_ambiguous_keys_raise() -> None:
    """Two equally specific accepting keys raise by default."""
    registry = {typing.Union[int, str]: "a", typing.Union[int, bytes]: "b"}
    with pytest.raises(AmbiguousMethodError):
        resolve_hint(int, registry)


def test_ambiguous_keys_warn_takes_first() -> None:
    """`ambiguity='warn'` takes the first registered and warns."""
    registry = {typing.Union[int, str]: "a", typing.Union[int, bytes]: "b"}
    with pytest.warns(RuntimeWarning):
        assert resolve_hint(int, registry, ambiguity="warn") == "a"


def test_ambiguous_keys_ignore_is_silent() -> None:
    """`ambiguity='ignore'` takes the first silently."""
    registry = {typing.Union[int, str]: "a", typing.Union[int, bytes]: "b"}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_hint(int, registry, ambiguity="ignore") == "a"


def test_exact_key_convenience() -> None:
    """An `Exact[C]` key answers a plain-`C` query (RFC §4 convenience)."""
    registry = {Exact[int]: "exactly int"}
    assert resolve_hint(int, registry) == "exactly int"


def test_more_specific_key_beats_broader() -> None:
    """The most specific accepting key wins over a broader one."""
    registry = {object: "any", int: "int", bool: "bool"}
    assert resolve_hint(bool, registry) == "bool"


def test_unset_is_the_default_sentinel() -> None:
    """`UNSET` is what marks 'no default given'."""
    with pytest.raises(NoMethodError):
        resolve_hint(list, {int: "n"}, default=UNSET)
