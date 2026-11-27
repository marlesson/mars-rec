"""Griffe extension: render Sphinx/reStructuredText docstring markup as Markdown.

The mars-rec docstrings are written with Sphinx conventions, while mkdocstrings
renders docstrings as Markdown. This extension rewrites the constructs that
would otherwise show up verbatim, before the docstring is parsed:

- cross-reference roles such as ``:class:`ProjectConfig``` or
  ``:func:`~mars_rec.inference.run_inference```. Fully-qualified ``mars_rec.*``
  targets become autorefs links (``[`run_inference`][mars_rec.inference.run_inference]``)
  and any other target becomes inline code (``` `ProjectConfig` ```);
- ``:math:`...``` roles become inline code;
- literal blocks introduced by a trailing ``::`` (``Algorithm::`` followed by an
  indented block) become fenced code blocks.

Docs-only: it is loaded by ``mkdocs.yml`` and is not part of the package.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

import griffe

_ROLE = re.compile(r":(?:py:)?(?:class|func|meth|mod|attr|data|exc|obj|const):`(?P<target>[^`]+)`")
_MATH = re.compile(r":math:`(?P<expr>[^`]+)`")
_PYTHON_HINT = re.compile(r"^\s*(def |class |import |from \S+ import |return |@\w)|\w+\s*=\s*\w+\(", re.MULTILINE)


def _replace_role(match: re.Match) -> str:
    target = match.group("target").strip()
    short = target.startswith("~")
    target = target.lstrip("~").rstrip("()")
    # Sphinx "title <target>" syntax.
    explicit = re.fullmatch(r"(?P<title>.+?)\s*<(?P<ref>[^>]+)>", target)
    if explicit:
        title, target = explicit.group("title"), explicit.group("ref")
    else:
        title = target.rsplit(".", 1)[-1] if short else target
    if target.startswith("mars_rec."):
        return f"[`{title}`][{target}]"
    return f"`{title}`"


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _fence_literal_blocks(text: str) -> str:
    """Turn reST literal blocks (``para::`` + indented block) into fenced code blocks."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        if not stripped.endswith("::") or stripped.endswith(":::"):
            out.append(line)
            i += 1
            continue
        base = _indent_of(line)
        # Find the indented block that follows (after at least one blank line).
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j == i + 1 or j >= len(lines) or _indent_of(lines[j]) <= base:
            out.append(line)  # not a literal block
            i += 1
            continue
        block_end = j
        while block_end < len(lines) and (not lines[block_end].strip() or _indent_of(lines[block_end]) > base):
            block_end += 1
        while block_end > j and not lines[block_end - 1].strip():
            block_end -= 1
        block = textwrap.dedent("\n".join(lines[j:block_end]))
        lang = "python" if _PYTHON_HINT.search(block) else "text"
        intro = stripped[:-2].rstrip()  # "Algorithm::" -> "Algorithm"; bare "::" -> ""
        pad = " " * base
        if intro.strip():
            out.append(intro + ":")
            out.append("")
        out.append(f"{pad}```{lang}")
        out.extend(f"{pad}{b}" if b else "" for b in block.splitlines())
        out.append(f"{pad}```")
        i = block_end
    return "\n".join(out)


def convert(text: str) -> str:
    """Apply every reST → Markdown rewrite to a docstring."""
    if "::" in text:
        text = _fence_literal_blocks(text)
    if ":" in text:
        text = _MATH.sub(lambda m: f"`{m.group('expr')}`", text)
        text = _ROLE.sub(_replace_role, text)
    return text


class SphinxRoles(griffe.Extension):
    """Rewrite reST markup in every docstring loaded by griffe."""

    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.docstring:
            obj.docstring.value = convert(obj.docstring.value)
