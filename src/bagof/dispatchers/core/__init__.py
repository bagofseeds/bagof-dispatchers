"""Comparing type hints to each other, and values to type hints.

`bagof.dispatchers.core` depends on nothing beyond `typing_extensions`,
which is what lets the rest of the `bagof` family sit on top of it
without pulling in the dispatch machinery above it. Its central exports
are [`issubhint`][], which decides whether one hint describes a set of
values narrower than another hint's, and [`ishintstance`][], which
decides whether a given value falls inside the set a hint describes.
Both are assembled from the smaller introspection functions exported
alongside them, which normalise a hint, take apart a generic alias, or
recognise a `TypedDict`. The dispatch engine defined at the top of this
package is one consumer of this layer, built with no assumption that it
is the only one.
"""

# local
from ._compat import UNION_TYPES, NoneType, UnionType, UnknownHintWarning
from ._introspect import (
    eq_safenan,
    get_args_uw,
    get_concrete_type,
    get_origin_uw,
    is_typeddict,
    issubclassable,
    issubscriptable,
    mro_index,
    normalise_hint,
    resolve_alias,
    resolve_newtype,
    safe_get_args,
    safe_get_origin,
    safe_isinstance,
    safe_issubclass,
    type2hint,
    typeddict_field_hints,
    typeddict_required_keys,
    unwrap,
)
from ._registry import resolve_hint
from ._relation import ishintstance, issubhint
from ._sentinels import UNSET, Unset

__all__ = [
    "issubhint",
    "ishintstance",
    "resolve_hint",
    "safe_get_origin",
    "safe_get_args",
    "get_origin_uw",
    "get_args_uw",
    "unwrap",
    "normalise_hint",
    "resolve_alias",
    "resolve_newtype",
    "is_typeddict",
    "typeddict_required_keys",
    "typeddict_field_hints",
    "safe_issubclass",
    "safe_isinstance",
    "issubclassable",
    "issubscriptable",
    "mro_index",
    "get_concrete_type",
    "type2hint",
    "eq_safenan",
    "Unset",
    "UNSET",
    "NoneType",
    "UnionType",
    "UNION_TYPES",
    "UnknownHintWarning",
]
