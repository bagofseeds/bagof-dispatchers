"""What a dispatch can raise, and how it reads.

A call that no method accepts raises
[`NoMethodError`][bagof.dispatchers.NoMethodError]; a call that two equally
specific methods both accept raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError]. Both are
[`DispatchError`][bagof.dispatchers.DispatchError]s, and a `DispatchError` is a
[`TypeError`][] -- calling a function with argument types it does not support
is a `TypeError` in Python, so an existing `#!python except TypeError:` keeps
working.

The messages are built to help someone stuck: they name the function, show the
call by the **types** of its arguments (never their values), list the methods
that competed or came closest with a `#!python !` on the argument that did not
fit, and end with the signature to define to resolve it -- Julia's "possible
fix".
"""

# stdlib
import difflib

# dependencies
import typing_extensions as tx

__all__ = ["DispatchError", "NoMethodError", "AmbiguousMethodError"]


class DispatchError(TypeError):
    """A call could not be dispatched to a single method.

    The base of [`NoMethodError`][bagof.dispatchers.NoMethodError] and
    [`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError]. It is a
    [`TypeError`][], so code that already guards a call with
    `#!python except TypeError:` catches a dispatch failure too.

    Attributes
    ----------
    function : str or None
        The name of the function that was called.
    call : Any
        A description of the call -- the argument types for a value dispatch,
        or the query hints for a hint resolution.
    candidates : tuple
        The methods (or registry keys) that competed or came closest.
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
    """No registered method accepts the call.

    Raised when every method either cannot bind the call (a keyword it does
    not declare, a missing argument, too many positionals) or rejects one of
    the argument types. The message names the function, shows the call by the
    types of its arguments, and lists the closest methods with a
    `#!python !` on each argument that did not fit.
    """


class AmbiguousMethodError(DispatchError):
    """Two equally specific methods both accept the call.

    Neither is more specific than the other, so choosing either would make
    the result depend on the order the methods were registered. The message
    lists the competing methods and suggests the more specific method to
    define -- which would win over both.
    """


def render_no_method(
    name: str,
    call_desc: str,
    method_count: int,
    closest: tx.Sequence[str],
    note: tx.Optional[str] = None,
) -> str:
    """Build a [`NoMethodError`][] message.

    Parameters
    ----------
    name
        The function's name.
    call_desc
        The call rendered by argument type, e.g. `#!python "area(str)"`.
    method_count
        How many methods the function has.
    closest
        The already-rendered closest-candidate lines, in order.
    note
        An extra line between the summary and the candidates -- the
        unknown-keyword did-you-mean, when there is one.
    """
    head = f"no method matching {call_desc}."
    if method_count == 0:
        return (
            f"{head}\n`{name}` has no methods yet: the module that defines "
            f"them has not been imported."
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
    """Build an [`AmbiguousMethodError`][] message.

    Parameters
    ----------
    call_desc
        The call rendered by argument type, e.g. `#!python "area(int, int)"`.
    candidates
        The already-rendered competing-method lines, in order.
    possible_fix
        The signature to define to resolve the ambiguity.
    """
    parts = [f"{call_desc} is ambiguous.", "Candidates:"]
    parts.extend(f"  {line}" for line in candidates)
    parts.append("Possible fix, define")
    parts.append(f"  {possible_fix}")
    return "\n".join(parts)


def did_you_mean(name: str, options: tx.Iterable[str]) -> tx.Optional[str]:
    """The closest declared name to `name`, if one is close enough."""
    matches = difflib.get_close_matches(name, list(options), n=1)
    return matches[0] if matches else None
