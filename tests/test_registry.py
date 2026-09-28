"""Tests for the hint-keyed lookup (`core/_registry.py`)."""

# stdlib
import enum
import re
import sys
import typing
import warnings
from collections import abc

# dependencies
import pytest
import typing_extensions as tx

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
    """A covariant container's key matches a subtype-argument query (#50).

    `Sequence` is covariant, so a `Sequence[int]` key is reached by a
    `Sequence[bool]` query. `list` is invariant, so a `List[int]` key is
    **not** reached by a `List[bool]` query -- the two describe incomparable
    lists (the spec-variance reversal of #50).
    """
    covariant = {typing.Sequence[int]: "ints"}
    assert resolve_hint(typing.Sequence[bool], covariant) == "ints"
    invariant = {typing.List[int]: "ints"}
    assert resolve_hint(typing.List[bool], invariant, default=None) is None


def test_typing_spelling_bridges_builtin_and_typing() -> None:
    """A key written one spelling is reached by a query written the other."""
    registry = {typing.List[int]: "ints"}
    assert resolve_hint(list, {list: "list"}) == "list"
    # A new-style query hits its typing-spelled exact key.
    assert resolve_hint(typing.List[int], registry) == "ints"


def test_bare_tuple_query_does_not_match_the_empty_tuple_key() -> None:
    """A bare `Tuple` query is not the empty-tuple type `Tuple[()]`.

    `_typing_spelling` used to rewrite a bare `typing.Tuple` to `Tuple[()]`,
    so the exact-key path matched an `Tuple[()]` entry that the relation does
    not accept.
    """
    # `Tuple[()]` is not an accepting key for a bare `Tuple` query.
    empty_only = {tx.Tuple[()]: "empty"}
    assert resolve_hint(tx.Tuple, empty_only, default=None) is None
    # With a bare `tuple` key present, that is the only accepting one.
    registry = {tx.Tuple[()]: "empty", tuple: "any"}
    assert resolve_hint(tx.Tuple, registry, default=None) == "any"


def test_unhashable_key_does_not_raise() -> None:
    """A key that cannot be hashed is simply not an exact key."""
    registry = {int: "int"}
    # A bare list is unhashable; iterating over the keys still works.
    assert resolve_hint(bool, registry) == "int"


def test_unhashable_query_hint() -> None:
    """An unhashable query hint is not an exact key and falls through."""
    # A list is unhashable; it cannot be an exact key, and nothing accepts it.
    assert resolve_hint([1, 2], {int: "n"}, default="fallback") == "fallback"


def test_non_hint_query_no_longer_reaches_any() -> None:
    """A non-hint query no longer matches an `Any` key (0.2.0).

    `issubhint` now refuses a non-hint, so a non-hint query no longer
    slips through to an `Any` catch-all as it did while `issubhint(1, Any)`
    was wrongly True. With nothing accepting it and no default, the lookup
    raises, and `NoMethodError` is a `TypeError`.
    """
    with pytest.raises(TypeError):
        resolve_hint(1, {tx.Any: "anything"})


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


# --- MRO tie-break for equally specific class keys (#19) ----------------


class _MroColor(str, enum.Enum):
    RED = "red"


class _MroB:
    pass


class _MroC:
    pass


class _MroD(_MroB, _MroC):
    pass


def test_mro_tiebreak_enum_str_prefers_str() -> None:
    """`{Enum, str}` resolves `class Color(str, Enum)` to `str`, either order.

    `Enum` and `str` are incomparable under the relation, but `str` is the
    nearer base of `Color` in its MRO, so it wins (RFC 0001 §2.2 step 3) --
    order-independently, matching the old `get_from_registry`.
    """
    str_first = {str: "str", enum.Enum: "enum"}
    enum_first = {enum.Enum: "enum", str: "str"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no ambiguity warning
        assert resolve_hint(_MroColor, str_first) == "str"
        assert resolve_hint(_MroColor, enum_first) == "str"
    assert resolve_hint(_MroColor, enum_first, ambiguity="raise") == "str"


def test_mro_tiebreak_diamond_prefers_nearer_base() -> None:
    """Diamond `D(B, C)` resolves `{C, B}` to `B`, either order."""
    b_first = {_MroB: "b", _MroC: "c"}
    c_first = {_MroC: "c", _MroB: "b"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_hint(_MroD, b_first) == "b"
        assert resolve_hint(_MroD, c_first) == "b"
    assert resolve_hint(_MroD, c_first, ambiguity="raise") == "b"


def test_mro_tiebreak_leaves_a_clear_win_alone() -> None:
    """A plain, non-ambiguous class query still resolves by specificity."""
    registry = {object: "any", int: "int", bool: "bool"}
    assert resolve_hint(bool, registry) == "bool"


def test_mro_tiebreak_does_not_apply_to_non_class_keys() -> None:
    """A genuinely incomparable non-class tie still warns and order-falls-back.

    The query is a class here, but the two accepting keys are `Union`s, which
    name no MRO position -- so MRO cannot break the tie and the caller's order
    fallback (plus warning) still applies.
    """
    a_first = {typing.Union[int, str]: "a", typing.Union[int, bytes]: "b"}
    b_first = {typing.Union[int, bytes]: "b", typing.Union[int, str]: "a"}
    with pytest.warns(RuntimeWarning):
        assert resolve_hint(int, a_first, ambiguity="warn") == "a"
    with pytest.warns(RuntimeWarning):
        assert resolve_hint(int, b_first, ambiguity="warn") == "b"
    with pytest.raises(AmbiguousMethodError):
        resolve_hint(int, a_first)


def test_mro_tiebreak_equidistant_class_keys_stay_ambiguous() -> None:
    """Class keys at the same MRO position leave the tie unbroken.

    `typing.Sequence` and `collections.abc.Sequence` name the *same* position
    in the query's MRO, so MRO cannot separate them -- the lookup keeps the
    order fallback and still warns, either registration order.
    """
    class _Seq(abc.Sequence):
        def __getitem__(self, index: int) -> int:
            raise IndexError

        def __len__(self) -> int:
            return 0

    abc_first = {abc.Sequence: "abc", typing.Sequence: "typing"}
    typing_first = {typing.Sequence: "typing", abc.Sequence: "abc"}
    with pytest.warns(RuntimeWarning):
        assert resolve_hint(_Seq, abc_first, ambiguity="warn") == "abc"
    with pytest.warns(RuntimeWarning):
        assert resolve_hint(_Seq, typing_first, ambiguity="warn") == "typing"
    with pytest.raises(AmbiguousMethodError):
        resolve_hint(_Seq, abc_first)


def test_mro_tiebreak_nearest_group_excludes_a_farther_key() -> None:
    """A farther key never wins on order when the nearest position ties.

    `class S2(abc.Sequence, B)` reaches `Sequence` nearer than `B` in its MRO.
    The two `Sequence` spellings tie at that nearest position, so the tie is
    restricted to *them* -- `B` is dropped, never picked up by the order
    fallback -- and the result is one of the `Sequence` values, either order.
    """
    class _S2(abc.Sequence, _MroB):
        def __getitem__(self, index: int) -> int:
            raise IndexError

        def __len__(self) -> int:
            return 0

    b_first = {_MroB: "b", typing.Sequence: "typing", abc.Sequence: "abc"}
    abc_first = {abc.Sequence: "abc", typing.Sequence: "typing", _MroB: "b"}
    with pytest.warns(RuntimeWarning):
        assert resolve_hint(_S2, b_first, ambiguity="warn") == "typing"
    with pytest.warns(RuntimeWarning):
        assert resolve_hint(_S2, abc_first, ambiguity="warn") == "abc"
    with pytest.raises(AmbiguousMethodError):
        resolve_hint(_S2, b_first)


def test_mro_tiebreak_parametrised_generic_query_is_version_consistent() -> (
    None
):
    """A `G[int]` query refines exactly like a bare `G` on every version.

    A parametrised generic alias is `isinstance(_, type)` on 3.9/3.10 but not
    on 3.11+, so the tie-break resolves the query to its origin class -- the
    same class a `G[int]` value has at runtime -- rather than depending on that
    quirk.
    """
    _T = typing.TypeVar("_T")

    class _Gen(_MroB, _MroC, typing.Generic[_T]):
        pass

    # `{C, B}` is a genuine tie for `_Gen` (it subclasses both), so the query
    # must reach the MRO refinement -- where `B` is the nearer base.
    registry = {_MroC: "c", _MroB: "b"}
    bare = resolve_hint(_Gen, registry)
    parametrised = resolve_hint(_Gen[int], registry)
    assert bare == parametrised == "b"


def test_non_class_query_tie_is_ambiguous_by_design() -> None:
    """A bare `Literal` key vs a class stays ambiguous (RFC 0001 §8.1 (3)).

    `Literal[1]` is a sub-hint of both bare `Literal` ("any literal") and
    `int`, but the two keys are incomparable and neither is a class in a value
    MRO, so MRO cannot decide -- it raises by default and, under
    `ambiguity="warn"`, warns and takes the first registered.
    """
    lit_first = {tx.Literal: "literal", int: "number"}
    int_first = {int: "number", tx.Literal: "literal"}
    with pytest.raises(AmbiguousMethodError):
        resolve_hint(tx.Literal[1], lit_first)
    with pytest.warns(RuntimeWarning):
        assert resolve_hint(tx.Literal[1], lit_first, ambiguity="warn") == (
            "literal"
        )
    with pytest.warns(RuntimeWarning):
        assert resolve_hint(tx.Literal[1], int_first, ambiguity="warn") == (
            "number"
        )


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


# =======================================================================
# Parity suite ported from bagof-core-magic's `get_from_registry`
# (test_introspection.py registry tests). `resolve_hint(..., default=None,
# ambiguity="warn")` is exactly the `get_from_registry` shim (RFC 0001 §8.1),
# so `_get` mirrors it. Where the relation-based lookup deliberately differs
# from the old summed-distance one, the case is flagged with issue #19 and
# asserts the NEW documented behavior.
# =======================================================================


def _get(hint: typing.Any, registry: typing.Mapping) -> typing.Any:
    """The `get_from_registry` shim: default None, ambiguity warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return resolve_hint(hint, registry, default=None, ambiguity="warn")


class _Base(tx.TypedDict):
    a: int


class _Middle(_Base):
    b: int


class _Leaf(_Middle):
    c: int


def test_parity_prefers_the_nearest_typeddict() -> None:
    assert _get(_Leaf, {_Middle: "near", _Base: "far"}) == "near"


def test_parity_result_is_insertion_order_independent() -> None:
    forwards = _get(_Leaf, {_Middle: "near", _Base: "far"})
    backwards = _get(_Leaf, {_Base: "far", _Middle: "near"})
    assert forwards == backwards == "near"


def test_parity_still_prefers_an_exact_match() -> None:
    assert _get(_Leaf, {_Leaf: "exact", _Base: "far"}) == "exact"


def test_parity_documented_example_is_unchanged() -> None:
    registry = {int: "number", object: "any"}
    assert _get(bool, registry) == "number"
    assert _get(str, registry) == "any"


_T = tx.TypeVar("_T")
_S = tx.TypeVar("_S")


@pytest.mark.parametrize(
    "query,expected",
    [
        (tx.Union[int, str], "union"),
        (tx.Union[str, int], "union"),  # membership order-insensitive
        (tx.Literal["a", "b"], "literal"),
        (_T, "typevar-T"),
        (tx.List[int], "list-int"),
        # #19 (TypeVar-key ordering): a *different* unbound TypeVar `_S` is now
        # a sub-hint of the registered unbound `_T` (an unbound TypeVar is
        # equivalent to `Any`), so it resolves to the `_T` entry rather than to
        # the bare `TypeVar` catch-all the old summed-distance lookup chose.
        (_S, "typevar-T"),
    ],
)
def test_parity_matches_a_specific_hint_key(
    query: typing.Any, expected: str
) -> None:
    registry = {
        tx.Union[int, str]: "union",
        tx.Literal["a", "b"]: "literal",
        _T: "typevar-T",
        tx.List[int]: "list-int",
        tx.TypeVar: "typevar-any",
        object: "any",
    }
    assert _get(query, registry) == expected


def test_parity_specific_key_survives_an_annotated_wrapper() -> None:
    registry = {tx.Union[int, str]: "union", object: "any"}
    query = tx.Annotated[tx.Union[int, str], "meta"]
    assert _get(query, registry) == "union"


def test_parity_does_not_crash_on_an_unhashable_query() -> None:
    registry = {int: "number", object: "any"}
    annotated = tx.Annotated[int, [1, 2]]  # unhashable metadata
    assert _get(annotated, registry) == "number"
    # A bare unhashable object is no key and matches no origin -- None.
    assert _get([1, 2], registry) is None


def test_parity_exact_pass_is_purely_additive() -> None:
    registry = {
        tx.Union: "union-origin",
        tx.Literal: "literal-origin",
        tx.TypeVar: "typevar-origin",
        object: "any",
    }
    assert _get(tx.Union[int, str], registry) == "union-origin"
    assert _get(tx.Literal[1, 2], registry) == "literal-origin"
    assert _get(_T, registry) == "typevar-origin"


def test_parity_annotated_key_delta() -> None:
    # A bare `Annotated` super-hint is now structural (0.2.0): every
    # `Annotated[...]` form is a sub-hint of it, and an ordinary type is not.
    # An `Annotated[int, ...]` query therefore matches both the bare
    # `Annotated` key (as an `Annotated` form) and the `int` key (its inner
    # type), and the two are incomparable, so neither is strictly more
    # specific. Under the default the query is ambiguous; in warn mode the
    # first-registered of the tied keys is taken.
    registry = {tx.Annotated: "annotated", int: "number", object: "any"}
    with pytest.raises(AmbiguousMethodError):
        resolve_hint(tx.Annotated[int, "meta"], registry)
    assert _get(tx.Annotated[int, "meta"], registry) == "annotated"
    # A plain `int` query is not an `Annotated` form, so the bare key does not
    # catch it: `issubhint(int, Annotated)` is now False.
    assert _get(int, registry) == "number"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="list[int] needs PEP 585 (3.9+)"
)
def test_parity_matches_a_new_style_generic_to_its_typing_key() -> None:
    registry = {
        tx.List[int]: "list-int",
        tx.Dict[str, int]: "dict-str-int",
        tx.Type[int]: "type-int",
        object: "any",
    }
    assert _get(list[int], registry) == "list-int"
    assert _get(dict[str, int], registry) == "dict-str-int"
    assert _get(type[int], registry) == "type-int"
    assert _get(tx.List[int], registry) == "list-int"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="list[int] needs PEP 585 (3.9+)"
)
def test_parity_rewrites_a_new_style_generic_recursively() -> None:
    registry = {
        tx.Optional[tx.List[int]]: "optional",
        tx.Union[tx.List[int], tx.Dict[str, int]]: "union",
        tx.List[tx.List[int]]: "list-of-list",
        tx.Annotated[tx.List[int], "m"]: "annotated",
        tx.Dict[str, tx.Optional[tx.List[int]]]: "deep",
        object: "any",
    }
    assert _get(tx.Optional[list[int]], registry) == "optional"
    if sys.version_info >= (3, 10):  # PEP 604 `X | Y`
        assert _get(list[int] | None, registry) == "optional"
    both = tx.Union[list[int], dict[str, int]]
    assert _get(both, registry) == "union"
    assert _get(list[list[int]], registry) == "list-of-list"
    assert _get(tx.Annotated[list[int], "m"], registry) == "annotated"
    deep = dict[str, tx.Optional[list[int]]]
    assert _get(deep, registry) == "deep"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="list[int] needs PEP 585 (3.9+)"
)
def test_parity_rewrites_inside_callable() -> None:
    registry = {
        tx.Callable[[tx.List[int]], str]: "params",
        tx.Callable[..., tx.List[int]]: "ret",
        object: "any",
    }
    assert _get(tx.Callable[[list[int]], str], registry) == "params"
    assert _get(tx.Callable[..., list[int]], registry) == "ret"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="abc/re generics need __class_getitem__"
)
def test_parity_new_style_generic_from_abc_and_re_origins() -> None:
    registry = {
        tx.Collection[int]: "collection",
        tx.Pattern[str]: "pattern",
        object: "any",
    }
    assert _get(abc.Collection[int], registry) == "collection"
    assert _get(re.Pattern[str], registry) == "pattern"


def test_parity_ranks_an_annotated_query_by_its_inner_type() -> None:
    registry = {int: "number", object: "any"}
    assert _get(tx.Annotated[bool, "x"], registry) == "number"


def test_parity_matches_none_as_nonetype() -> None:
    none_type = type(None)
    assert _get(None, {none_type: "none", object: "any"}) == "none"
    assert _get(None, {object: "any"}) == "any"


def test_parity_survives_an_origin_that_refuses_the_rewrite() -> None:
    class Picky:
        def __class_getitem__(cls, item: object) -> object:
            raise ValueError("no generics here")

    assert _get(Picky, {Picky: "picky", object: "any"}) == "picky"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="list[int] needs PEP 585 (3.9+)"
)
def test_parity_new_style_generic_still_falls_back_to_its_origin() -> None:
    assert _get(list[int], {list: "bare", object: "any"}) == "bare"


def test_parity_typeddict_over_dict_wins_order_independently() -> None:
    # #19 (TypedDict-key ordering): the bare `TypedDict` marker is a sub-hint
    # of `dict` (every TypedDict value is a dict), so `TypedDict` is the unique
    # most-specific key over `dict` for a TypedDict-subclass query -- resolved
    # order-independently, restoring the old `get_from_registry` preference.
    td_first = {tx.TypedDict: "typeddict", dict: "dict"}
    dict_first = {dict: "dict", tx.TypedDict: "typeddict"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no ambiguity warning is emitted
        assert resolve_hint(_Base, td_first) == "typeddict"
        assert resolve_hint(_Base, dict_first) == "typeddict"
    # It is unambiguous: `ambiguity="raise"` still returns it, either order.
    assert resolve_hint(_Base, td_first, ambiguity="raise") == "typeddict"
    assert resolve_hint(_Base, dict_first, ambiguity="raise") == "typeddict"


def test_parity_ignores_an_unrelated_typeddict_key() -> None:
    assert _get(int, {tx.TypedDict: "typeddict"}) is None
    assert _get(int, {tx.TypedDict: "typeddict", int: "number"}) == "number"


# --- defensive: a malformed key never fails the whole lookup (B4) -------


def test_resolve_hint_tolerates_a_malformed_key() -> None:
    """A key that is not a usable hint does not fail the lookup; it misses.

    The relation raises for a value that is not a type or typing construct; a
    registry that happens to hold one still resolves the other keys, but warns
    rather than skipping the bad key in silence.
    """
    from bagof.dispatchers.core import UnknownHintWarning

    registry = {5: "bad-key", int: "number", object: "any"}
    with pytest.warns(UnknownHintWarning, match="not a usable type hint"):
        assert resolve_hint(bool, registry) == "number"


def test_accepts_warns_and_strictly_below_is_quiet_on_a_bad_key() -> None:
    """`_accepts` warns for a bad key; `_strictly_below` reads it as not-below.

    A key `_accepts` cannot read raises a relation `TypeError`, read as "does
    not accept" *and* warned about, so the skip is not silent.
    `_strictly_below` only ever sees keys `_accepts` already vetted, so it
    stays quiet on the same signal.
    """
    from bagof.dispatchers.core import UnknownHintWarning
    from bagof.dispatchers.core._registry import _accepts, _strictly_below

    with pytest.warns(UnknownHintWarning, match="not a usable type hint"):
        assert _accepts(bool, 5) is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _strictly_below(int, 5) is False


def test_accepts_does_not_swallow_a_genuine_error() -> None:
    """A non-`TypeError` from the relation (real fault) is not hidden.

    A key whose `__subclasscheck__` raises a `RuntimeError` is a genuine error,
    so it propagates rather than being read as "does not accept".
    """
    from bagof.dispatchers.core._registry import _accepts

    class Raising(type):
        def __subclasscheck__(cls, other: object) -> bool:
            raise RuntimeError("boom")

    class Key(metaclass=Raising):
        pass

    with pytest.raises(RuntimeError, match="boom"):
        _accepts(int, Key)
