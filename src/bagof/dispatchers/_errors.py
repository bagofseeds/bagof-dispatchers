"""What a dispatch call can raise, and how its message is written.

When no registered method accepts a call, dispatch raises
[`NoMethodError`][]. When two methods are equally specific and both
accept the call, it raises [`AmbiguousMethodError`][]. Both are
[`DispatchError`][], and `DispatchError` is itself a [`TypeError`][]:
calling a function with argument types it does not support is
ordinarily a `TypeError` in Python, so existing code that guards a call
with `#!python except TypeError:` keeps catching a dispatch failure
too.

Every message is written for someone who has just hit the error. It
names the function, shows the call by the types of its arguments
rather than their values, lists the methods that competed or came
closest (marking with `#!python !` the argument that did not fit), and
ends with the signature that would resolve the problem if it were
defined, the same "possible fix" line Julia prints for the same
situation.
"""

# stdlib
import difflib

# dependencies
import typing_extensions as tx

__all__ = ["DispatchError", "NoMethodError", "AmbiguousMethodError"]


class DispatchError(TypeError):
    """A call could not be resolved to exactly one method.

    `DispatchError` is the common base of [`NoMethodError`][] and
    [`AmbiguousMethodError`][]. It subclasses [`TypeError`][], so code
    that already guards a call with `#!python except TypeError:`
    catches a dispatch failure as well.

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
    """Raised when no registered method accepts the call.

    Every method either fails to bind the call at all, because it
    declares no such keyword, is missing a required argument, or was
    given too many positional arguments, or rejects one of the argument
    types once bound. The message names the function, shows the call by
    the types of its arguments, and lists the closest methods with
    `#!python !` marking each argument that did not fit.
    """


class AmbiguousMethodError(DispatchError):
    """Raised when two equally specific methods both accept the call.

    Neither method is more specific than the other, so choosing between
    them would make the result depend on the order in which they
    happened to be registered rather than on their signatures. The
    message lists the competing methods and suggests a more specific
    signature that would take precedence over both.
    """


def render_no_method(
    name: str,
    call_desc: str,
    method_count: int,
    closest: tx.Sequence[str],
    note: tx.Optional[str] = None,
) -> str:
    """Build the message for a [`NoMethodError`][].

    Parameters
    ----------
    name
        The function's name.
    call_desc
        The call rendered by argument type, for example
        `#!python "area(str)"`.
    method_count
        How many methods the function has registered.
    closest
        The already-rendered lines describing the closest candidates,
        in order.
    note
        An extra line inserted between the summary and the candidates,
        used for the "did you mean" suggestion when an unrecognised
        keyword is close to a known one.
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
    possible_fix: str,
) -> str:
    """Build the message for an [`AmbiguousMethodError`][].

    Parameters
    ----------
    call_desc
        The call rendered by argument type, for example
        `#!python "area(int, int)"`.
    candidates
        The already-rendered lines describing the competing methods,
        in order.
    possible_fix
        The signature that, if defined, would resolve the ambiguity.
    """
    parts = [f"{call_desc} is ambiguous.", "Candidates:"]
    parts.extend(f"  {line}" for line in candidates)
    parts.append("Possible fix, define")
    parts.append(f"  {possible_fix}")
    return "\n".join(parts)


def did_you_mean(name: str, options: tx.Iterable[str]) -> tx.Optional[str]:
    """Return the declared name closest to `name`, if any is close enough."""
    matches = difflib.get_close_matches(name, list(options), n=1)
    return matches[0] if matches else None
