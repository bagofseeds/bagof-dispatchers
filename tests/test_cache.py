"""Tests for dispatch caching and concurrency (`_function.py`)."""

# stdlib
import abc
import threading
import typing

# dependencies
import pytest

# locals
from bagof.dispatchers._function import Function, _call_key, _KeyValue, _Plan

# --- caching correctness -----------------------------------------------


def test_repeated_call_is_cached() -> None:
    """The same call shape and types resolves to the same method twice."""
    f = Function("f")

    def m(x: int) -> int:
        return x

    f.register(m)
    first = f.dispatch(1)
    second = f.dispatch(2)
    assert first is second  # same cached method object


def test_cache_dropped_on_register() -> None:
    """A new registration invalidates the cache, so selection updates."""
    f = Function("f")

    def general(x: object) -> str:
        return "object"

    f.register(general)
    assert f(3) == "object"

    def specific(x: int) -> str:
        return "int"

    f.register(specific)
    assert f(3) == "int"  # the newly-registered, more specific method wins


def test_value_dependent_cache_keys_on_value() -> None:
    """A `Literal` argument keys the cache on the value, not just the type."""
    f = Function("f")

    def one(x: typing.Literal[1]) -> str:
        return "one"

    def other(x: int) -> str:
        return "int"

    f.register(one)
    f.register(other)
    assert f(1) == "one"
    assert f(2) == "int"  # same type (int), different value, different method


def test_unhashable_value_dependent_argument_is_uncached() -> None:
    """An unhashable value at a value-dependent argument still dispatches."""
    f = Function("f")

    def for_type(x: typing.Type[int]) -> str:
        return "type"

    def anything(x: object) -> str:
        return "object"

    f.register(for_type)
    f.register(anything)
    # A plain object at a value-dependent (type[...]) position: not a class,
    # so it takes the `object` method, and repeats without caching issues.
    assert f(object()) == "object"
    assert f(int) == "type"


def test_unhashable_at_planned_value_dependent_shape() -> None:
    """An unhashable value at a value-dependent, already-planned shape works.

    The first call plans the shape and caches; the second reaches the cache
    with an unhashable value-dependent argument, so both the read and the
    write fall back to resolving without caching.
    """
    f = Function("f")

    def one(x: typing.Literal[1]) -> str:
        return "one"

    def other(x: object) -> str:
        return "object"

    f.register(one)
    f.register(other)
    assert f(1) == "one"  # plans the shape and caches
    assert f([1, 2]) == "object"  # unhashable list, uncached, still resolves
    assert f([3, 4]) == "object"  # again -- the hot-path read also falls back


def test_keyword_value_dependent_key() -> None:
    """A value-dependent keyword argument keys the cache on its value."""
    f = Function("f")

    def a(x: int, *, mode: typing.Literal["a"]) -> str:
        return "a"

    def other(x: int, *, mode: str) -> str:
        return "other"

    f.register(a)
    f.register(other)
    assert f(1, mode="a") == "a"
    assert f(1, mode="b") == "other"  # same type, different value


def test_non_value_dependent_call_key_uses_types() -> None:
    """A plain call keys on argument types only."""
    f = Function("f")

    def m(x: int, y: str) -> str:
        return "m"

    f.register(m)
    cache = f._refresh()
    shape = (2, ())
    plan = f._build_plan(shape, cache)
    key = _call_key((1, "a"), {}, plan)
    assert key == (2, int, str)


# --- abc.register() invalidation ---------------------------------------


def test_abc_register_invalidates_cache() -> None:
    """A virtual `abc.register` changes selection and the cache honours it."""

    class Interface(abc.ABC):  # noqa: B024 -- a purely virtual base
        pass

    class Plain:
        pass

    f = Function("f")

    def for_interface(x: Interface) -> str:
        return "interface"

    def for_object(x: object) -> str:
        return "object"

    f.register(for_interface)
    f.register(for_object)

    plain = Plain()
    assert f(plain) == "object"  # not yet an Interface; cached

    Interface.register(Plain)  # now Plain is a virtual subclass
    # The abc cache token has moved, so the stale entry is not served.
    assert f(plain) == "interface"


# --- concurrency -------------------------------------------------------


def test_concurrent_register_and_dispatch_never_tears() -> None:
    """Registering while dispatching never sees a half-updated method group."""
    f = Function("f")

    def base(x: object) -> str:
        return "object"

    f.register(base)

    # Each registration takes a distinct type, so the signatures differ and
    # every one is kept rather than replacing the last.
    types = [type(f"T{index}", (), {}) for index in range(40)]

    stop = threading.Event()
    errors = []  # type: typing.List[BaseException]

    def dispatcher() -> None:
        while not stop.is_set():
            try:
                assert f(3) == "object"  # 3 is an int, not any T-type
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                return

    def registrar() -> None:
        try:
            for index, made_type in enumerate(types):

                def made(x: made_type) -> str:  # noqa: ANN001
                    return "made"

                made.__name__ = f"m{index}"
                made.__qualname__ = f"m{index}"
                f.register(made)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    readers = [threading.Thread(target=dispatcher) for _ in range(2)]
    writer = threading.Thread(target=registrar)
    for reader in readers:
        reader.start()
    writer.start()
    writer.join()
    stop.set()
    for reader in readers:
        reader.join()
    assert not errors
    # Every distinct registration was kept (each a different signature).
    assert len(f.methods) == 41


# --- the key wrapper ---------------------------------------------------


def test_keyvalue_hashes_and_compares_by_value() -> None:
    """`_KeyValue` delegates hashing and equality to its value."""
    assert _KeyValue(1) == _KeyValue(1)
    assert hash(_KeyValue(1)) == hash(1)
    assert _KeyValue(1) != _KeyValue(2)


def test_keyvalue_distinguishes_type() -> None:
    """`1` and `True` hash alike but are different keys (`Literal` safety)."""
    assert _KeyValue(1) != _KeyValue(True)


def test_keyvalue_notimplemented_for_other_types() -> None:
    """Compared to a non-`_KeyValue`, equality defers."""
    assert _KeyValue(1).__eq__(1) is NotImplemented


def test_call_key_builds_but_defers_hash_when_unhashable() -> None:
    """A value-dependent unhashable value still builds a key; hash defers.

    `_call_key` always returns a tuple -- it never signals "uncacheable" with
    `None`. An unhashable value at a value-dependent argument is wrapped so the
    tuple builds fine and the `TypeError` surfaces only when the key is hashed
    (on the surrounding `dict` access), which the engine catches to leave the
    call uncached.
    """
    # Fabricate a plan whose single positional is value-dependent.
    plan = _Plan((), {}, frozenset({0}))
    key = _call_key(([1, 2],), {}, plan)  # a list is unhashable
    assert isinstance(key, tuple)  # the tuple always builds
    with pytest.raises(TypeError):
        hash(key)  # hashing it is what raises, and dispatch catches that
