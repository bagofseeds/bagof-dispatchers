"""Classes that annotate a protocol's members under the future import.

Every annotation here is stored as a plain string, so what each one declares
-- an instance variable, a class variable, or no attribute -- has to be read
from its text, without evaluating it.
"""

# future
from __future__ import annotations

# stdlib
import dataclasses
import typing
import typing as t
from typing import ClassVar
from typing import ClassVar as CV

# dependencies
import typing_extensions as tx


class Annotated:
    name: str


class Quoted:
    name: "Undefined"  # noqa: F821, UP037 -- never evaluated


class NameClassVar:
    name: ClassVar[str]


class NameDottedClassVar:
    name: typing.ClassVar[str]


class NameAnnotatedClassVar:
    name: tx.Annotated[ClassVar[str], "m"]


class KindDeclared:
    kind: ClassVar[str]


class KindInstance:
    kind: tx.Annotated[str, "m"]


@dataclasses.dataclass
class WithInitVar:
    name: dataclasses.InitVar[str]


class InitVar:
    """The module's own class, named like the dataclass marker."""


class AliasClassVar:
    name: CV[str]


class QualifiedClassVar:
    name: t.ClassVar[str]


class BodyAliasClassVar:
    Shared = ClassVar
    name: Shared[str]


class ShadowedInitVar:
    name: InitVar


class AliasKind:
    kind: CV[str]
