"""Every ``pycon`` block in the docs and public docstrings runs and is true.

The examples the docs and API reference show are executed here, top to bottom,
each source sharing one namespace -- so a later block sees an earlier block's
imports and definitions, the way a reader following the page does. A block that
does not reproduce its stated output fails the test. CI runs this on the oldest
supported Python too, so an example that needs a newer one is caught.
"""

# stdlib
import doctest
import inspect
import re
from pathlib import Path

# dependencies
import pytest
import typing_extensions as tx

# locals
import bagof.dispatchers as pkg

_ROOT = Path(__file__).resolve().parent.parent
_PYCON = re.compile(r"```pycon\n(.*?)```", re.S)

# Some hints print with a `typing.` prefix on new Pythons and a
# `typing_extensions.` prefix on old ones (`Annotated`, `Union`, ...); the
# object is the same, so an example that prints one reads true on either.
_OPTIONFLAGS = doctest.ELLIPSIS


class _Checker(doctest.OutputChecker):
    """An output checker that ignores the typing-module prefix drift."""

    @staticmethod
    def _canon(text: str) -> str:
        return text.replace("typing_extensions.", "typing.")

    def check_output(
        self, want: str, got: str, optionflags: int
    ) -> bool:
        return super().check_output(
            self._canon(want), self._canon(got), optionflags
        )


def _blocks(text: tx.Optional[str]) -> tx.List[str]:
    """The bodies of every ``pycon`` fenced block in `text`, in order."""
    if not text:
        return []
    return _PYCON.findall(text)


def _seed() -> tx.Dict[str, tx.Any]:
    """A namespace pre-loaded with the public API, for docstring examples."""
    import warnings

    namespace = {name: getattr(pkg, name) for name in pkg.__all__}
    namespace["tx"] = tx
    namespace["warnings"] = warnings
    return namespace


def _run(
    name: str, blocks: tx.Sequence[str], seed: tx.Dict[str, tx.Any]
) -> None:
    """Run `blocks` in one shared namespace and assert none fails."""
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(
        checker=_Checker(), optionflags=_OPTIONFLAGS
    )
    globs = dict(seed)
    # Give each source its own module name, so a `@dispatch def` in one
    # source's examples keys by that module and never collides with a
    # same-named one in another source's -- the module-level `dispatch` keys
    # by (module, qualname), and every source would otherwise share the one
    # bucket of definitions made with no module name.
    globs["__name__"] = "doctest_" + re.sub(r"\W+", "_", name)
    for lineno, block in enumerate(blocks):
        test = parser.get_doctest(block, globs, name, None, lineno)
        runner.run(test, clear_globs=False)
        globs.update(test.globs)
    failures = runner.summarize(verbose=False).failed
    assert failures == 0, f"{failures} pycon example(s) failed in {name}"


def _markdown_sources() -> tx.List[Path]:
    """The docs pages whose examples are executed: README and ``docs/*.md``."""
    sources = [_ROOT / "README.md"]
    sources.extend(sorted((_ROOT / "docs").glob("*.md")))
    return [path for path in sources if path.exists()]


@pytest.mark.parametrize(
    "path", _markdown_sources(), ids=lambda p: p.name
)
def test_markdown_examples(path: Path) -> None:
    """Every ``pycon`` block in a docs page runs and prints what it claims."""
    _run(str(path.relative_to(_ROOT)), _blocks(path.read_text()), {})


def _docstring_sources() -> tx.List[tx.Tuple[str, str]]:
    """`(name, docstring)` for every public object and method with examples."""
    sources = []  # type: tx.List[tx.Tuple[str, str]]
    for name in pkg.__all__:
        obj = getattr(pkg, name)
        doc = inspect.getdoc(obj)
        if doc and "```pycon" in doc:
            sources.append((name, doc))
        if inspect.isclass(obj):
            for member_name, member in inspect.getmembers(obj):
                if member_name.startswith("_"):
                    continue
                doc = inspect.getdoc(member)
                if doc and "```pycon" in doc:
                    sources.append((f"{name}.{member_name}", doc))
    return sources


_DOCSTRING_SOURCES = _docstring_sources()


@pytest.mark.parametrize(
    "name, doc",
    _DOCSTRING_SOURCES,
    ids=[name for name, _ in _DOCSTRING_SOURCES],
)
def test_docstring_examples(name: str, doc: str) -> None:
    """Every ``pycon`` block in a public docstring runs and is true."""
    _run(name, _blocks(doc), _seed())
