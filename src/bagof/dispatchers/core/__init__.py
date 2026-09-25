"""The hint subtype relation and introspection helpers.

`bagof.dispatchers.core` is the shared, dependency-light root of the bagof
family: the hint-level subtype relation ([`issubhint`][], [`ishintstance`][])
and the introspection helpers they are built on. The dispatch engine at the
top level, and the sibling bags, are built on these names.
"""

# local
from ._compat import UNION_TYPES, NoneType, UnionType
from ._introspect import (
    eq_safenan,
    get_args_uw,
    get_concrete_type,
    get_origin_uw,
    is_typeddict,
    issubclassable,
    issubscriptable,
    normalise_hint,
    safe_get_args,
    safe_get_origin,
    safe_isinstance,
    safe_issubclass,
    type2hint,
    typeddict_required_keys,
    unwrap,
)
from ._relation import ishintstance, issubhint
from ._sentinels import UNSET, Unset

__all__ = [
    "issubhint",
    "ishintstance",
    "safe_get_origin",
    "safe_get_args",
    "get_origin_uw",
    "get_args_uw",
    "unwrap",
    "normalise_hint",
    "is_typeddict",
    "typeddict_required_keys",
    "safe_issubclass",
    "safe_isinstance",
    "issubclassable",
    "issubscriptable",
    "get_concrete_type",
    "type2hint",
    "eq_safenan",
    "Unset",
    "UNSET",
    "NoneType",
    "UnionType",
    "UNION_TYPES",
]
