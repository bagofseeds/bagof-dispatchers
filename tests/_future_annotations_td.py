"""A TypedDict defined under `from __future__ import annotations`.

Every annotation here is stored as a plain string, so it stands in for the
real-world case the per-field fallback in `typeddict_field_hints` exists for:
one unresolvable field must not drop its resolvable siblings.
"""

# future
from __future__ import annotations

# dependencies
import typing_extensions as tx


class FutureTD(tx.TypedDict):
    year: int  # stored as the string "int"; resolves against builtins
    missing: _UndefinedForwardName  # noqa: F821 -- deliberately unresolvable
