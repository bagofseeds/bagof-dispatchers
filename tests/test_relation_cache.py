"""Tests for the `issubhint` result cache in `core/_relation.py`.

The cache must only ever change how fast the relation answers, never what
it answers, so these tests check that a hit returns the same result, that
an ABC registration invalidates it, that an unhashable pair is answered
without caching, and that the bounded working set evicts oldest-first.
"""

# stdlib
import abc

# dependencies
import typing_extensions as tx

# local
import bagof.dispatchers.core._relation as rel
from bagof.dispatchers.core import issubhint


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
    assert (bool, int) in rel._RELATION_CACHE


def test_cache_stores_the_result() -> None:
    rel.clear_relation_cache()
    assert issubhint(str, int) is False
    assert rel._RELATION_CACHE[(str, int)] is False


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


def test_an_unhashable_pair_is_answered_without_caching() -> None:
    unhashable = tx.Annotated[int, [1, 2]]  # a list metadata is unhashable
    rel.clear_relation_cache()
    assert issubhint(unhashable, int) is True
    # The pair could not be a key, so nothing was stored for it.
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
        assert (bool, int) not in rel._RELATION_CACHE
        assert (float, object) in rel._RELATION_CACHE
    finally:
        rel.RELATION_CACHE_SIZE = saved
        rel.clear_relation_cache()


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
