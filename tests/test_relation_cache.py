"""Tests for the `issubhint` result cache in `core/_relation.py`.

The cache must only ever change how fast the relation answers, never what
it answers, so these tests check that a hit returns the same result, that
an ABC registration invalidates it, that an unhashable pair is answered
without caching, and that the bounded working set evicts oldest-first.
"""

# stdlib
import abc
import sys
import typing

# dependencies
import pytest
import typing_extensions as tx

# local
import bagof.dispatchers.core._relation as rel
from bagof.dispatchers import Between, Exact, Super
from bagof.dispatchers.core import issubhint

_key = rel._relation_key


def test_a_repeated_query_is_a_cache_hit() -> None:
    rel.clear_relation_cache()
    calls = []
    original = rel._issubhint

    def counting(hint: object, superhint: object) -> bool:
        calls.append((hint, superhint))
        return original(hint, superhint)

    rel._issubhint = counting
    try:
        assert issubhint(bool, int) is True
        assert issubhint(bool, int) is True
    finally:
        rel._issubhint = original
    # The body ran once; the second call was served from the cache.
    assert calls.count((bool, int)) == 1
    assert _key(bool, int) in rel._RELATION_CACHE


def test_a_lower_bound_keys_its_own_entry() -> None:
    # `Type[Super[int]]` and `Type[int]` are different hints with different
    # answers, so each is cached under its own key.
    rel.clear_relation_cache()
    assert issubhint(tx.Type[Super[int]], tx.Type[object]) is True
    assert issubhint(tx.Type[int], tx.Type[object]) is True
    assert issubhint(tx.Type[Super[int]], tx.Type[int]) is False
    assert issubhint(tx.Type[int], tx.Type[int]) is True
    above = _key(tx.Type[Super[int]], tx.Type[int])
    assert rel._RELATION_CACHE[above] is False
    assert rel._RELATION_CACHE[_key(tx.Type[int], tx.Type[int])] is True


def test_an_interval_keys_its_own_entry() -> None:
    # `Type[Between[bool, int]]`, `Type[Super[int]]` and `Type[int]` are three
    # different hints, so each is cached under its own key.
    rel.clear_relation_cache()
    interval = tx.Type[Between[bool, int]]
    assert issubhint(interval, tx.Type[int]) is True
    assert issubhint(tx.Type[Super[int]], tx.Type[int]) is False
    assert issubhint(tx.Type[int], tx.Type[int]) is True
    above = _key(tx.Type[Super[int]], tx.Type[int])
    assert rel._RELATION_CACHE[_key(interval, tx.Type[int])] is True
    assert rel._RELATION_CACHE[above] is False
    assert _key(interval, tx.Type[int]) != _key(tx.Type[int], tx.Type[int])


def test_cache_stores_the_result() -> None:
    rel.clear_relation_cache()
    assert issubhint(str, int) is False
    assert rel._RELATION_CACHE[_key(str, int)] is False


def test_abc_register_invalidates_the_cache() -> None:
    # The cache must never mask an answer that changed: after `register`,
    # `issubhint(C, ABC)` flips from False to True because the cache-token
    # change clears the stale entry.
    class MyABC(abc.ABC):  # noqa: B024 -- register, not subclass, is the point
        pass

    class C:
        pass

    rel.clear_relation_cache()
    assert issubhint(C, MyABC) is False
    MyABC.register(C)
    assert issubhint(C, MyABC) is True


def test_result_computed_across_a_token_change_is_not_stored() -> None:
    # If an ABC registration lands while the answer is being computed, the
    # answer may already be stale, so it is dropped rather than stored under
    # the old regime; the token change clears the cache on the next call.
    rel.clear_relation_cache()
    original = rel._issubhint

    class _Bumper(abc.ABC):  # noqa: B024 -- register bumps the cache token
        pass

    class _X:
        pass

    def bump_then_answer(hint: object, superhint: object) -> bool:
        _Bumper.register(_X)  # changes abc.get_cache_token()
        return original(hint, superhint)

    rel._issubhint = bump_then_answer
    try:
        assert issubhint(_X, object) is True
    finally:
        rel._issubhint = original
    assert _key(_X, object) not in rel._RELATION_CACHE


def test_an_unhashable_pair_is_answered_correctly() -> None:
    unhashable = tx.Annotated[int, [1, 2]]  # a list metadata is unhashable
    rel.clear_relation_cache()
    # It must not crash and must give the right answer either way.
    assert issubhint(unhashable, int) is True
    if not rel._LITERAL_EQ_MERGES:
        # Where the cache keys by value, an unhashable pair cannot be a key, so
        # nothing is stored. Where it keys by identity (old Pythons), the pair
        # is hashable through the identity wrapper and may be cached; both are
        # correct.
        assert len(rel._RELATION_CACHE) == 0


def test_the_cache_evicts_oldest_first_when_full() -> None:
    rel.clear_relation_cache()
    saved = rel.RELATION_CACHE_SIZE
    rel.RELATION_CACHE_SIZE = 2
    try:
        assert issubhint(bool, int) is True
        assert issubhint(str, object) is True
        # A third distinct key evicts the oldest, (bool, int).
        assert issubhint(float, object) is True
        assert len(rel._RELATION_CACHE) == 2
        assert _key(bool, int) not in rel._RELATION_CACHE
        assert _key(float, object) in rel._RELATION_CACHE
    finally:
        rel.RELATION_CACHE_SIZE = saved
        rel.clear_relation_cache()


@pytest.mark.skipif(
    sys.version_info >= (3, 9, 1),
    reason="typing.Literal merges 1 and True only before 3.9.1",
)
def test_literal_cache_does_not_collide_on_old_python() -> None:
    # On 3.8 and 3.9.0, `typing.Literal[1] == typing.Literal[True]` holds with
    # equal hashes. Once `typing`'s own subscription cache has evicted
    # `Literal[1]`, a later `Literal[True]` is a distinct object that still
    # compares equal to it. A `==`-keyed cache would then read the stored
    # answer for `Literal[1]` back for `Literal[True]`, even though the two
    # differ under `Exact[int]`. Identity keying keeps them apart.
    rel.clear_relation_cache()
    one = typing.Literal[1]
    assert issubhint(one, Exact[int]) is True  # stored under `one`
    try:
        # Evict `Literal[1]` from `typing`'s subscription cache so the next
        # `Literal[True]` is a fresh, distinct object.
        for value in range(2000):
            typing.Literal[value]
        true = typing.Literal[True]
        assert true is not one
        assert true == one and hash(true) == hash(one)
        assert issubhint(true, Exact[int]) is False
    finally:
        # Creating `Literal[True]` last leaves it in `typing`'s shared 1/True
        # cache slot, so `typing.Literal[1]` would keep returning it and
        # mislead later tests. Evict it and repopulate the slot with a real
        # `Literal[1]` object to restore the normal state.
        for value in range(2000, 4000):
            typing.Literal[value]
        typing.Literal[1]


def test_clear_empties_the_cache() -> None:
    issubhint(bool, int)
    assert rel._RELATION_CACHE
    rel.clear_relation_cache()
    assert rel._RELATION_CACHE == {}


def test_the_cache_never_changes_an_answer() -> None:
    # Cold and warm answers agree for a spread of hint shapes.
    pairs = [
        (bool, int),
        (int, str),
        (tx.List[bool], tx.List[int]),
        (tx.Sequence[bool], tx.Sequence[int]),
        (tx.Union[int, str], tx.Union[int, str, bytes]),
        (tx.Literal[1], tx.Literal[1, 2]),
    ]
    rel.clear_relation_cache()
    cold = [issubhint(a, b) for a, b in pairs]
    warm = [issubhint(a, b) for a, b in pairs]
    assert cold == warm
