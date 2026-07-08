"""Convert generated ``[[wikilinks]]`` to bundle-relative markdown links.

OKF recommends standard markdown links (bundle-relative or page-relative)
over wiki-specific syntaxes. The LLM prompts still ask for ``[[wikilinks]]``
— models emit them reliably and the ghost-link whitelist operates on them —
so the conversion happens deterministically at page-write time, after ghost
stripping. Obsidian and the webui wiki browser both resolve relative
markdown links, and OKF consumers get real relationships.

Emission is config-gated: ``link_style: markdown`` (default) converts;
``link_style: wikilinks`` keeps the raw ``[[...]]`` syntax.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

# Wiki subdirectories whose pages are valid bare-stem link targets.
WIKI_PAGE_DIRS = ("summaries", "concepts", "entities", "explorations")


def link_style(config: dict) -> str:
    """Return the effective link style ("markdown" | "wikilinks")."""
    style = str(config.get("link_style", "markdown")).lower()
    return style if style in ("markdown", "wikilinks") else "markdown"


def _resolve_target(target: str, wiki_dir: Path | None) -> str | None:
    """Resolve a wikilink target to a wiki-root-relative ``.md`` path.

    Dir-prefixed targets (``concepts/attention``) resolve by pure path math —
    no existence check, because the target may be written later in the same
    compile run. Bare stems resolve only when exactly one page dir already
    holds ``<stem>.md``; otherwise ``None`` (leave the wikilink alone — the
    lint pass owns ghost handling).
    """
    t = target.strip().strip("/")
    if not t or t.startswith(("http://", "https://")):
        return None
    if t == "index":
        return "index.md"
    if "/" in t:
        head = t.split("/", 1)[0]
        if head not in WIKI_PAGE_DIRS:
            return None
        resolved = t if t.endswith(".md") else f"{t}.md"
        # Topic-tree: a flat concepts/<stem> address may live nested under
        # concepts/<topic>/. Prefer the file that actually exists.
        if wiki_dir is not None and head == "concepts" and not (wiki_dir / resolved).exists():
            stem = resolved.rsplit("/", 1)[-1]
            hits = [p for p in (wiki_dir / "concepts").rglob(stem) if p.is_file()]
            if len(hits) == 1:
                return str(hits[0].relative_to(wiki_dir))
        return resolved
    if wiki_dir is not None:
        dir_hits = [sub for sub in WIKI_PAGE_DIRS if (wiki_dir / sub / f"{t}.md").exists()]
        if len(dir_hits) == 1:
            return f"{dir_hits[0]}/{t}.md"
    return None


def wikilinks_to_markdown(text: str, own_dir: str, wiki_dir: Path | None = None) -> str:
    """Rewrite ``[[target]]`` / ``[[target|label]]`` as relative markdown links.

    ``own_dir`` is the page's directory relative to the wiki root (empty
    string for a page at the wiki root, e.g. ``index.md``). Unresolvable
    targets are left as wikilinks.
    """

    def _repl(m: re.Match) -> str:
        target, label = m.group(1), m.group(2)
        resolved = _resolve_target(target, wiki_dir)
        if resolved is None:
            return m.group(0)
        if not label:
            label = posixpath.basename(resolved)[: -len(".md")]
        rel = posixpath.relpath(resolved, own_dir or ".")
        return f"[{label.strip()}]({rel})"

    return _WIKILINK_RE.sub(_repl, text)
