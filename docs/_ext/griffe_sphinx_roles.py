"""Griffe extension: render Sphinx cross-reference roles as Markdown.

The mars-rec docstrings use Sphinx roles such as ``:class:`ProjectConfig``` or
``:func:`~mars_rec.inference.run_inference```. mkdocstrings renders docstrings
as Markdown, where these roles would show up verbatim. This extension rewrites
them before the docstring is parsed:

- fully-qualified ``mars_rec.*`` targets become autorefs links
  (``[`run_inference`][mars_rec.inference.run_inference]``);
- any other target becomes inline code (``` `ProjectConfig` ```).

Docs-only: it is loaded by ``mkdocs.yml`` and is not part of the package.
"""

from __future__ import annotations

import re
from typing import Any

import griffe

_ROLE = re.compile(r":(?:py:)?(?:class|func|meth|mod|attr|data|exc|obj|const):`(?P<target>[^`]+)`")


def _replace(match: re.Match) -> str:
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


class SphinxRoles(griffe.Extension):
    """Rewrite Sphinx roles in every docstring loaded by griffe."""

    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.docstring and ":" in obj.docstring.value:
            obj.docstring.value = _ROLE.sub(_replace, obj.docstring.value)
