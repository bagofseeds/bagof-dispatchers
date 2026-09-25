"""Tests for modern typing constructs and forward tolerance."""

# stdlib
import sys
import typing
import warnings

# dependencies
import pytest
import typing_extensions as tx

# local
from bagof.dispatchers.core import (
    ishintstance,
    issubclassable,
    issubhint,
    normalise_hint,
    safe_issubclass,
)
from bagof.dispatchers.core._compat import UnknownHintWarning
from bagof.dispatchers.core._introspect import resolve_alias, resolve_newtype

# --- NewType -----------------------------------------------------------


def test_newtype_resolves_to_its_supertype() -> None:
    NT = tx.NewType("NT", int)
    assert normalise_hint(NT) is int
    assert resolve_newtype(NT) is int
    assert issubhint(NT, int) is True
    assert issubhint(bool, NT) is True
    assert issubhint(str, NT) is False
    assert ishintstance(1, NT) is True
    assert ishintstance("x", NT) is False


def test_newtype_of_newtype_is_followed() -> None:
    Inner = tx.NewType("Inner", int)
    Outer = tx.NewType("Outer", Inner)
    assert resolve_newtype(Outer) is int


# --- TypeAliasType (typing_extensions backport) ------------------------


def test_type_alias_type_resolves_to_its_value() -> None:
    MyInt = tx.TypeAliasType("MyInt", int)
    assert normalise_hint(MyInt) is int
    assert issubhint(MyInt, int) is True
    assert issubhint(bool, MyInt) is True


def test_generic_alias_substitutes_its_arguments() -> None:
    T = tx.TypeVar("T")
    L = tx.TypeAliasType("L", tx.List[T], type_params=(T,))
    assert resolve_alias(L[int]) == tx.List[int]
    assert resolve_alias(L[bool]) == tx.List[bool]
    assert issubhint(L[int], list) is True
    # The argument survives, so two parameterisations are distinguished.
    assert issubhint(L[bool], L[int]) is True
    assert issubhint(L[str], L[int]) is False


def test_alias_of_alias_is_followed() -> None:
    MyInt = tx.TypeAliasType("MyInt", int)
    Again = tx.TypeAliasType("Again", MyInt)
    assert normalise_hint(Again) is int


# --- native PEP 695 (3.12+) --------------------------------------------


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="native PEP 695 `type` needs 3.12+"
)
def test_native_pep695_alias() -> None:
    # exec-guarded so collection still works on 3.8.
    namespace = {}  # type: dict
    exec(
        "type MyStr = str\n"
        "type Box[T] = list[T]\n",
        namespace,
    )
    MyStr = namespace["MyStr"]
    Box = namespace["Box"]
    assert normalise_hint(MyStr) is str
    assert issubhint(MyStr, str) is True
    # A native `type Box[T] = list[T]` resolves to the builtin `list[int]`,
    # which is never `== typing.List[int]`; compare spelling-agnostically.
    resolved = resolve_alias(Box[int])
    assert tx.get_origin(resolved) is list and tx.get_args(resolved) == (int,)
    assert issubhint(resolved, tx.List[int]) and issubhint(
        tx.List[int], resolved
    )
    assert issubhint(Box[bool], Box[int]) is True


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="native PEP 695 `type` needs 3.12+"
)
def test_native_recursive_alias_stops() -> None:
    namespace = {}  # type: dict
    exec("type Tree = int | list[Tree]\n", namespace)
    Tree = namespace["Tree"]
    # Resolving must terminate rather than loop on the self-reference.
    resolved = normalise_hint(Tree)
    assert issubhint(int, resolved) is True


# --- transparent qualifiers --------------------------------------------


@pytest.mark.parametrize(
    "wrap",
    [tx.Final, tx.ClassVar, tx.Required, tx.NotRequired, tx.ReadOnly],
)
def test_qualifiers_unwrap_to_inner(wrap: tx.Any) -> None:
    assert normalise_hint(wrap[int]) is int
    assert issubhint(wrap[bool], int) is True
    assert issubhint(int, wrap[int]) is True


# --- bottom, guards, and simple substitutions --------------------------


def test_never_is_a_bottom_type() -> None:
    assert issubhint(tx.Never, int) is True
    assert issubhint(tx.NoReturn, int) is True
    assert issubhint(int, tx.Never) is False
    assert issubhint(tx.Never, tx.Never) is True
    assert ishintstance(1, tx.Never) is False


def test_typeguard_and_typeis_are_bool() -> None:
    assert issubhint(tx.TypeGuard[int], bool) is True
    assert issubhint(bool, tx.TypeGuard[int]) is True
    assert issubhint(tx.TypeIs[int], bool) is True
    assert ishintstance(True, tx.TypeGuard[int]) is True


def test_literalstring_is_str() -> None:
    assert issubhint(tx.LiteralString, str) is True
    assert issubhint(str, tx.LiteralString) is True
    assert ishintstance("x", tx.LiteralString) is True


# --- PEP 696 defaults are ignored --------------------------------------


def test_pep696_default_is_ignored_for_dispatch() -> None:
    # The bound wins; the default (`int`) is a static-checker fallback only,
    # so an `int` value does not match a `float`-bound typevar.
    TB = tx.TypeVar("TB", bound=float, default=int)
    assert issubhint(TB, float) is True
    assert ishintstance(1, TB) is False
    assert ishintstance(1.0, TB) is True


def test_no_dot_has_default_is_called() -> None:
    # A native 3.12 PEP 695 TypeVar lacks `has_default`; the relation must
    # read `__default__` through `getattr`, never call `.has_default()`.
    import inspect

    from bagof.dispatchers.core import _relation

    source = inspect.getsource(_relation)
    assert ".has_default(" not in source


# --- Unpack / ParamSpec degradation ------------------------------------


def test_unpack_and_paramspec_degrade_without_crashing() -> None:
    Ts = tx.TypeVarTuple("Ts")
    P = tx.ParamSpec("P")
    # A bare unpack / paramspec form as a super-hint is opaque, not a crash.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnknownHintWarning)
        assert issubhint(int, tx.Unpack[Ts]) is True
        assert issubhint(int, P) is True


# --- the GenericAlias trap ---------------------------------------------


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="list[int] needs PEP 585 (3.9+)"
)
def test_generic_alias_is_not_mistaken_for_a_class() -> None:
    # `isinstance(list[int], type)` is True on 3.9/3.10; the relation must
    # not treat `list[int]` as a subclassable class.
    assert issubclassable(list[int]) is False
    assert safe_issubclass(list[int], object) is False
    # A real class is still recognised.
    assert issubclassable(list) is True


# --- Protocol guard ----------------------------------------------------


def test_non_runtime_protocol_answers_false_not_raises() -> None:
    class NotRuntime(tx.Protocol):
        def foo(self) -> int: ...

    # `issubclass(int, NotRuntime)` would raise TypeError; the relation
    # answers False instead.
    assert issubhint(int, NotRuntime) is False
    assert ishintstance(1, NotRuntime) is False


def test_runtime_protocol_dispatches_structurally() -> None:
    @tx.runtime_checkable
    class Sized(tx.Protocol):
        def __len__(self) -> int: ...

    assert issubhint(list, Sized) is True
    assert issubhint(int, Sized) is False


# --- Union <= constrained TypeVar symmetry -----------------------------


def test_union_is_a_subhint_of_a_constrained_typevar() -> None:
    TC = tx.TypeVar("TC", int, str)
    # The symmetric partner of `TC <= Union[int, str]`.
    assert issubhint(tx.Union[int, str], TC) is True
    assert issubhint(TC, tx.Union[int, str]) is True
    assert issubhint(tx.Union[int, bool], TC) is True  # members subhints
    assert issubhint(tx.Union[int, bytes], TC) is False


# --- forward tolerance: opaque + one warning ---------------------------


def test_unknown_form_is_opaque_and_warns_once() -> None:
    # A future/unknown special form is accepted as a super-hint (treated as
    # `Any`) and warned about exactly once.
    marker = tx.Self  # a form the structural relation has no branch for
    from bagof.dispatchers.core import _relation

    _relation._WARNED_UNKNOWN.discard(_relation._warn_key(marker))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert issubhint(int, marker) is True
        assert issubhint(str, marker) is True  # second sighting: no warning
    unknowns = [
        w for w in caught if issubclass(w.category, UnknownHintWarning)
    ]
    assert len(unknowns) == 1
    # Opaque is a sub-hint only of itself and `Any`.
    assert issubhint(marker, int) is False
    assert issubhint(marker, tx.Any) is True


def test_unknown_form_warns_once_per_form_not_per_repr() -> None:
    # Two hints built from one unknown form (`Unpack[Ts]`, `Unpack[Us]`) have
    # different reprs but share an origin, so the warning fires once, not
    # twice.
    from bagof.dispatchers.core import _relation

    Ts = tx.TypeVarTuple("Ts")
    Us = tx.TypeVarTuple("Us")
    a, b = tx.Unpack[Ts], tx.Unpack[Us]
    _relation._WARNED_UNKNOWN.discard(_relation._warn_key(a))
    _relation._WARNED_UNKNOWN.discard(_relation._warn_key(b))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert issubhint(int, a) is True
        assert issubhint(int, b) is True
    unknowns = [
        w for w in caught if issubclass(w.category, UnknownHintWarning)
    ]
    assert len(unknowns) == 1


# --- R1: real typing classes are not swallowed as special forms --------


def test_typing_protocols_and_abcs_are_not_special_forms() -> None:
    from bagof.dispatchers.core import issubclassable

    assert issubhint(int, tx.SupportsInt) is True
    assert issubhint(int, tx.SupportsIndex) is True
    assert issubclassable(tx.SupportsIndex) is True
    buffer = getattr(tx, "Buffer", None)
    if buffer is not None:
        assert issubhint(bytes, buffer) is True


# --- R3: `typing` vs `typing_extensions` spellings of Any/Literal ------


@pytest.mark.parametrize("any_form", list({typing.Any, tx.Any}))
def test_any_spellings_accept_everything(any_form: tx.Any) -> None:
    assert issubhint(int, any_form) is True
    assert ishintstance(1, any_form) is True


@pytest.mark.parametrize("literal", list({typing.Literal, tx.Literal}))
def test_literal_spellings_are_understood(literal: tx.Any) -> None:
    # A `Literal` super-hint must be read by its values, not treated as an
    # opaque accept-everything form.
    assert issubhint(int, literal[1]) is False
    assert issubhint(literal[1], literal[1, 2]) is True
    assert ishintstance(1, literal[1]) is True
    assert ishintstance(2, literal[1]) is False


# --- R4: the bare TypedDict marker is not opaque -----------------------


def test_bare_typeddict_marker_is_not_opaque() -> None:
    class Movie(tx.TypedDict):
        title: str

    assert issubhint(dict, tx.TypedDict) is False
    assert issubhint(Movie, tx.TypedDict) is True
    assert ishintstance({"title": "x"}, tx.TypedDict) is False


# --- D2: obvious non-hints raise, typing-shaped forms stay opaque ------


def test_obvious_non_hints_raise_typeerror() -> None:
    with pytest.raises(TypeError):
        issubhint(int, "Foo")  # a bare string is an unresolvable forward ref
    with pytest.raises(TypeError):
        ishintstance(1, 123)
    with pytest.raises(TypeError):
        issubhint(int, [1, 2])


def test_typing_shaped_unknown_stays_opaque() -> None:
    from bagof.dispatchers.core import _relation

    class _FutureForm:
        __module__ = "typing"

        def __repr__(self) -> str:
            return "SomeFutureForm"

    marker = _FutureForm()
    _relation._WARNED_UNKNOWN.discard(_relation._warn_key(marker))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert issubhint(int, marker) is True
    unknowns = [
        w for w in caught if issubclass(w.category, UnknownHintWarning)
    ]
    assert len(unknowns) == 1
