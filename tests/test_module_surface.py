"""The public surface: the top-level API and the ``core`` facade.

`bagof.dispatchers` exposes exactly the dispatch API, and
`bagof.dispatchers.core` exposes the relation and introspection helpers. The
two are disjoint (apart from `Exact`, whose home is the top level), and each is
complete -- every name in `__all__` resolves. These assertions guard the
boundary the RFC draws, so an accidental export or a dropped name is a test
failure rather than a doc page that quietly disagrees.
"""

# locals
import bagof.dispatchers as pkg
import bagof.dispatchers.core as core

# The dispatch API, exactly (RFC 0001 §6). Order matters for the reference.
_TOP_LEVEL = [
    "dispatch",
    "Dispatcher",
    "Function",
    "Method",
    "Signature",
    "Parameter",
    "DispatchError",
    "NoMethodError",
    "AmbiguousMethodError",
    "Exact",
]


def test_top_level_all_is_exact() -> None:
    """`bagof.dispatchers.__all__` is the dispatch API and nothing else."""
    assert pkg.__all__ == _TOP_LEVEL


def test_top_level_names_resolve() -> None:
    """Every name in `__all__` is importable from the package."""
    for name in pkg.__all__:
        assert hasattr(pkg, name), name


def test_core_names_resolve() -> None:
    """Every name in `core.__all__` is importable from the facade."""
    for name in core.__all__:
        assert hasattr(core, name), name


def test_no_engine_internals_leak() -> None:
    """The top level exports no engine internals beyond the API list."""
    internals = ["Binding", "resolve_hint", "issubhint", "ishintstance"]
    for name in internals:
        assert name not in pkg.__all__, name


def test_namespaces_are_disjoint_except_exact() -> None:
    """The top-level and `core` `__all__`s share only `Exact`.

    `Exact` is understood by the relation in `core` but documented once, at the
    top level; every other name belongs to exactly one namespace.
    """
    shared = set(pkg.__all__) & set(core.__all__)
    assert shared <= {"Exact"}


def test_exact_is_one_object() -> None:
    """`Exact` reached either way is the same object."""
    from bagof.dispatchers.core._exact import Exact as core_exact

    assert pkg.Exact is core_exact


def test_public_names_come_from_private_modules() -> None:
    """Each public dispatch name is re-exported from a private engine module.

    A class carries its home module; `dispatch` is an instance, so its type's
    module is checked instead.
    """
    homes = {
        "dispatch": "bagof.dispatchers._dispatcher",
        "Dispatcher": "bagof.dispatchers._dispatcher",
        "Function": "bagof.dispatchers._function",
        "Method": "bagof.dispatchers._method",
        "Signature": "bagof.dispatchers._signature",
        "Parameter": "bagof.dispatchers._signature",
        "DispatchError": "bagof.dispatchers._errors",
        "NoMethodError": "bagof.dispatchers._errors",
        "AmbiguousMethodError": "bagof.dispatchers._errors",
    }
    for name, module in homes.items():
        obj = getattr(pkg, name)
        home = obj if isinstance(obj, type) else type(obj)
        assert home.__module__ == module, (name, home.__module__)
