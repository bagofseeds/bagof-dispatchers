"""`typing.TypedDict` and `typing_extensions.TypedDict` are interchangeable."""

# stdlib
import typing

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers._lattice import equivalent
from bagof.dispatchers.core import (
    is_typeddict,
    issubclassable,
    issubhint,
    safe_issubclass,
    typeddict_required_keys,
)

# The two are distinct objects, and a class built from one never mentions
# the other in its `__orig_bases__` -- which is what made them dispatch
# differently.
SPELLINGS = [("typing_extensions", tx.TypedDict), ("typing", typing.TypedDict)]


def _build(TD: tx.Any) -> tx.Dict[str, tx.Any]:
    class Movie(TD):
        title: str
        year: int

    class Extended(Movie):
        rating: int

    class Partial(TD, total=False):
        a: int

    class MixedChild(Partial):
        b: int

    return {
        "plain": Movie,
        "inherited": Extended,
        "total=False": Partial,
        "mixed-total child": MixedChild,
    }


@pytest.mark.parametrize("name,TD", SPELLINGS)
def test_typeddict_itself_is_recognised(name: str, TD: tx.Any) -> None:
    assert is_typeddict(TD)
    assert issubclassable(TD)


@pytest.mark.parametrize("name,TD", SPELLINGS)
@pytest.mark.parametrize(
    "case", ["plain", "inherited", "total=False", "mixed-total child"]
)
def test_subclasses_are_recognised(
    name: str, TD: tx.Any, case: str
) -> None:
    cls = _build(TD)[case]
    assert is_typeddict(cls)


@pytest.mark.parametrize("key_name,key", SPELLINGS)
def test_a_plain_dict_is_not_a_typeddict(key_name: str, key: tx.Any) -> None:
    assert not safe_issubclass(dict, key)


@pytest.mark.parametrize("name,TD", SPELLINGS)
def test_subclass_relation_holds_across_spellings(
    name: str, TD: tx.Any
) -> None:
    cls = _build(TD)["plain"]
    for _, other in SPELLINGS:
        assert safe_issubclass(cls, other)
    # ... and a TypedDict is still a dict.
    assert safe_issubclass(cls, dict)


def _tracks_mixed_totality(TD: tx.Any) -> bool:
    """Whether this runtime records requiredness across a `total=` change.

    A capability probe rather than a version check: what matters is what
    the class in front of us actually reports.
    """

    class Base(TD, total=False):
        x: int

    class Child(Base):
        y: int

    return set(getattr(Child, "__required_keys__", ())) == {"y"}


@pytest.mark.parametrize("name,TD", SPELLINGS)
def test_required_keys(name: str, TD: tx.Any) -> None:
    cases = _build(TD)
    assert typeddict_required_keys(cases["plain"]) == {"title", "year"}
    assert typeddict_required_keys(cases["total=False"]) == frozenset()


@pytest.mark.parametrize("name,TD", SPELLINGS)
def test_required_keys_across_a_totality_change(
    name: str, TD: tx.Any
) -> None:
    cases = _build(TD)
    required = typeddict_required_keys(cases["mixed-total child"])
    if _tracks_mixed_totality(TD):
        # Only the child's own key is required. `__total__` alone reports
        # this wrongly, which is why `__required_keys__` is preferred.
        assert required == {"b"}
    else:
        # An older `typing.TypedDict` does not record which class declared
        # a key, and a subclass has no `__orig_bases__` to walk -- so the
        # true answer is not recoverable and every inherited key is
        # reported required. Loudly conservative: a valid value fails,
        # rather than an invalid one passing.
        assert required == {"a", "b"}


def test_typing_extensions_always_tracks_mixed_totality() -> None:
    # The reason `typing_extensions` reimplements the class at all.
    assert _tracks_mixed_totality(tx.TypedDict)


def test_required_keys_handles_notrequired_under_annotated() -> None:
    class Wrapped(tx.TypedDict):
        outside: tx.Annotated[tx.NotRequired[int], "meta"]
        inside: tx.NotRequired[tx.Annotated[int, "meta"]]
        plain: int

    assert typeddict_required_keys(Wrapped) == {"plain"}


def test_required_keys_falls_back_to_total() -> None:
    # `typing.TypedDict` gained `__required_keys__` only in Python 3.9.
    # Before that, requiredness came from `total=` alone.
    class OldStyle(dict):
        __annotations__ = {"a": int, "b": str}
        __total__ = True

    class OldStylePartial(dict):
        __annotations__ = {"a": int}
        __total__ = False

    assert typeddict_required_keys(OldStyle) == {"a", "b"}
    assert typeddict_required_keys(OldStylePartial) == frozenset()


# --- the bare `TypedDict` marker is a sub-hint of `dict` (#19) -----------


@pytest.mark.parametrize("name,TD", SPELLINGS)
def test_bare_typeddict_marker_is_subhint_of_dict(
    name: str, TD: tx.Any
) -> None:
    """A bare `TypedDict` marker means "any TypedDict", and every TypedDict
    value is a `dict`, so the marker is a sub-hint of `dict` -- but a plain
    `dict` is not the marker's sub-hint, so the marker stays strictly below
    `dict`."""
    assert issubhint(TD, dict)
    assert not issubhint(dict, TD)


@pytest.mark.parametrize("name,TD", SPELLINGS)
def test_bare_typeddict_marker_is_not_below_unrelated_types(
    name: str, TD: tx.Any
) -> None:
    """The marker ranks as `dict`, so it is not a sub-hint of an unrelated
    class or of `int`."""
    assert not issubhint(TD, int)
    assert not issubhint(TD, list)


@pytest.mark.parametrize("name,TD", SPELLINGS)
def test_concrete_typeddict_is_subhint_of_dict(
    name: str, TD: tx.Any
) -> None:
    """A concrete TypedDict is a sub-hint of `dict` too."""
    cls = _build(TD)["plain"]
    assert issubhint(cls, dict)


def test_bare_markers_are_interchangeable_across_spellings() -> None:
    """The two bare `TypedDict` markers stay equivalent both ways (#19).

    The `TypedDict <= dict` redirect applies only against a plain-class
    super-hint, so a marker super-hint still goes through the nominal path --
    keeping `typing.TypedDict` and `typing_extensions.TypedDict`
    interchangeable (which `canonical_typeddict` and this file rely on)."""
    assert issubhint(typing.TypedDict, tx.TypedDict)
    assert issubhint(tx.TypedDict, typing.TypedDict)
    assert equivalent(typing.TypedDict, tx.TypedDict)
    assert equivalent(tx.TypedDict, typing.TypedDict)


@pytest.mark.parametrize("name,TD", SPELLINGS)
def test_bare_marker_stays_subhint_when_contained(
    name: str, TD: tx.Any
) -> None:
    """A super-hint that merely *contains* the marker still matches it (#19).

    Moving the `TypedDict <= dict` redirect below the union / `TypeVar` /
    `Annotated` handling is what keeps these True -- a containing super-hint is
    not silently replaced by `dict`."""
    assert issubhint(TD, tx.Optional[TD])
    assert issubhint(TD, tx.Union[TD, int])
    assert issubhint(TD, tx.Annotated[TD, "m"])
    assert issubhint(tx.Annotated[TD, "m"], TD)
    bound = tx.TypeVar("bound", bound=TD)
    assert issubhint(TD, bound)
