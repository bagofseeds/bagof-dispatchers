"""The exceptions a dispatch call fails with, and their message text.

A call can fail to resolve to a single method in one of two ways: no
registered method accepts the arguments given, which raises
[`NoMethodError`][], or more than one method accepts them equally well,
which raises [`AmbiguousMethodError`][]. Both inherit from
[`DispatchError`][], which is itself a [`TypeError`][], since calling a
function with arguments of the wrong type already means `TypeError` in
ordinary Python; code that catches `TypeError` around a call therefore
keeps working once that call goes through dispatch.

The functions in this module build the message text for both errors.
Each message states the function's name, describes the call by the
types of the arguments rather than their values, and lists the
implementations that were closest to matching, marking with a leading
`#!python !` whichever argument kept each one from succeeding. It then
closes with a signature that would resolve the situation if it were
defined, following the same "possible fix" convention that Julia's
multiple-dispatch errors use.
"""

# stdlib
import difflib

# dependencies
import typing_extensions as tx

__all__ = ["DispatchError", "NoMethodError", "AmbiguousMethodError"]


class DispatchError(TypeError):
    """The base of both errors a dispatch call can raise.

    A call reaches `DispatchError` when it cannot be pinned down to
    exactly one method, whether because none matched or because several
    matched equally well; [`NoMethodError`][] and
    [`AmbiguousMethodError`][] cover those two cases respectively.
    Inheriting from [`TypeError`][] keeps a dispatch failure inside the
    exception hierarchy that an ordinary Python call already raises for
    a type mismatch, so a broad `#!python except TypeError:` around a
    call needs no change to keep working.

    Attributes
    ----------
    function : str or None
        The name of the function that was called.
    call : Any
        A description of the call: the argument types for a value
        dispatch, or the query hints for a hint resolution.
    candidates : tuple
        The methods, or registry keys, that competed or came closest.
    """

    def __init__(
        self,
        message: str,
        *,
        function: tx.Optional[str] = None,
        call: tx.Any = None,
        candidates: tx.Sequence[tx.Any] = (),
    ) -> None:
        super().__init__(message)
        self.function = function
        self.call = call
        self.candidates = tuple(candidates)


class NoMethodError(DispatchError):
    """Raised when the call matches none of a function's methods.

    A method can fall short of a call in two different ways before
    dispatch ever gets to compare types. It can fail to bind the
    arguments at all, for instance because it takes no such keyword, is
    missing a required argument, or was handed more positional arguments
    than it accepts. Or it can bind successfully and then reject one of
    the argument values against its declared types. Either way, the
    message names the function, describes the call by argument type, and
    lists whichever methods came closest, marking the argument that
    defeated each one with a leading `#!python !`.
    """


class AmbiguousMethodError(DispatchError):
    """Raised when more than one method matches a call equally well.

    This happens when two methods both accept the call and neither is
    more specific than the other, so picking one over the other would
    depend on the order the methods happened to be registered in rather
    than on anything about their signatures. Dispatch refuses to make
    that arbitrary choice and raises instead, listing the competing
    methods and suggesting a signature specific enough to take
    precedence over both.

    The competing methods are the ones still tied after every tie-break
    that dispatch applies, so a method that was equally specific but
    lost on `priority`, for example, is not among them. The error's
    `candidates` attribute holds these methods in registration order,
    which is the same tuple that
    [`Function.bestcandidates`][bagof.dispatchers.Function.bestcandidates]
    returns for the same call.
    """


def render_no_method(
    name: str,
    call_desc: str,
    method_count: int,
    closest: tx.Sequence[str],
    note: tx.Optional[str] = None,
) -> str:
    """Compose the text of a [`NoMethodError`][] message.

    Parameters
    ----------
    name
        The function's name.
    call_desc
        The call, rendered by the types of its arguments, for example
        `#!python "area(str)"`.
    method_count
        The number of methods currently registered under `name`, used to
        distinguish a function with no methods at all from one whose
        methods simply do not cover this combination of types.
    closest
        Lines already rendered for whichever methods came closest to
        matching, in the order they should appear.
    note
        An extra line placed between the summary and the candidate list,
        used to carry a "did you mean" suggestion when the call used an
        unrecognised keyword close to one a method declares.
    """
    head = f"no method matching {call_desc}."
    if method_count == 0:
        return (
            f"{head}\n`{name}` has no methods yet: the module that defines "
            f"them has not been imported, or it was reached by name from a "
            f"different module than the one that registered it."
        )
    plural = "s" if method_count != 1 else ""
    parts = [
        head,
        f"`{name}` has {method_count} method{plural}, but none is defined "
        f"for this combination of argument types.",
    ]
    if note:
        parts.append(note)
    if closest:
        parts.append("Closest candidates:")
        parts.extend(f"  {line}" for line in closest)
    return "\n".join(parts)


def render_ambiguous(
    call_desc: str,
    candidates: tx.Sequence[str],
    possible_fix: tx.Optional[str],
) -> str:
    """Compose the text of an [`AmbiguousMethodError`][] message.

    Parameters
    ----------
    call_desc
        The call, rendered by the types of its arguments, for example
        `#!python "area(int, int)"`.
    candidates
        Lines already rendered for the methods tied for the best match,
        in the order they should appear.
    possible_fix
        A signature that, if a method were defined with it, would be
        specific enough to settle the ambiguity. When no such signature
        can be built from the call, this is [`None`][], and the message
        ends after the list of candidates.
    """
    parts = [f"{call_desc} is ambiguous.", "Candidates:"]
    parts.extend(f"  {line}" for line in candidates)
    if possible_fix is not None:
        parts.append("Possible fix, define")
        parts.append(f"  {possible_fix}")
    return "\n".join(parts)


def did_you_mean(name: str, options: tx.Iterable[str]) -> tx.Optional[str]:
    """Find the name in `options` closest to `name`, if one is close enough.

    Returns `None` when nothing in `options` resembles `name` closely
    enough to be worth suggesting.
    """
    matches = difflib.get_close_matches(name, list(options), n=1)
    return matches[0] if matches else None
