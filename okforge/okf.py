"""OKF bundle maintenance: conformance checking and link retargeting.

Split from ``okforge.lint`` (file-size gate): these operate on the wiki as
an Open Knowledge Format bundle rather than linting its content.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

from okforge import frontmatter
from okforge.lint import _EXCLUDED_FILES, _MDLINK_RE, _all_wiki_pages, _read_md
from okforge.locks import atomic_write_text


def okf_check(wiki: Path) -> list[str]:
    """OKF conformance issues for the wiki bundle.

    Checks (per the Open Knowledge Format spec, v0.1):
    - every ``.md`` page carries parseable YAML frontmatter with a
      non-empty ``type`` (reserved files ``index.md`` / ``log.md`` are
      exempt — their role is positional, not typed);
    - the reserved files exist and open with a heading / log format.

    Returns a sorted list of human-readable issue strings (empty = clean).
    """
    issues: list[str] = []
    reserved = {"index.md", "log.md"}

    for md in sorted(wiki.rglob("*.md")):
        rel_parts = md.relative_to(wiki).parts
        # dot-dirs (.trash from `remove`, .obsidian, …) are workspace
        # state, not part of the OKF bundle; reports/ is generated lint
        # output, same as the structural linter's exclusion
        if any(p.startswith(".") for p in rel_parts[:-1]) or rel_parts[0] == "reports":
            continue
        rel = "/".join(rel_parts)
        if md.name in reserved and len(rel_parts) == 1:
            continue
        text = _read_md(md)
        fm = frontmatter.parse(text)
        if not fm:
            issues.append(f"{rel}: missing or malformed frontmatter")
        elif not str(fm.get("type", "") or "").strip():
            issues.append(f"{rel}: frontmatter lacks a non-empty 'type'")

    index_md = wiki / "index.md"
    if not index_md.exists():
        issues.append("index.md: missing (OKF reserved file)")
    elif not _read_md(index_md).lstrip().startswith("#"):
        issues.append("index.md: does not open with a Markdown heading")

    log_md = wiki / "log.md"
    if not log_md.exists():
        issues.append("log.md: missing (OKF reserved file)")

    return sorted(issues)


def retarget_md_links(wiki: Path) -> int:
    """Rewrite relative markdown links whose target file has MOVED.

    Wikilinks are address-by-name and survive a concept moving into a
    topic dir; markdown links (the `link_style: markdown` default since
    v0.5.1) are physical relative paths and break. For every page link
    that no longer resolves but whose basename uniquely matches a page
    elsewhere in the wiki (e.g. flat ``../concepts/x.md`` after x moved
    to ``concepts/<topic>/x.md``), rewrite the href to the new relative
    path. Returns the number of links rewritten.
    """

    pages = _all_wiki_pages(wiki)
    # unique basename -> page path (ambiguous stems dropped)
    by_stem: dict[str, Path | None] = {}
    for p in set(pages.values()):
        by_stem[p.stem] = None if p.stem in by_stem else p

    rewritten = 0
    for md in wiki.rglob("*.md"):
        rel_parts = md.relative_to(wiki).parts
        if md.name in _EXCLUDED_FILES or (rel_parts and rel_parts[0] in ("reports", "sources")):
            continue
        if any(p.startswith(".") for p in rel_parts[:-1]):
            continue
        own_dir = "/".join(rel_parts[:-1])
        text = _read_md(md)

        def _repl(m: re.Match) -> str:
            nonlocal rewritten
            raw = m.group(1)
            if raw.startswith(("http://", "https://", "/")):
                return m.group(0)
            resolved = posixpath.normpath(posixpath.join(own_dir, raw))
            if resolved.startswith("..") or (wiki / resolved).exists():
                return m.group(0)
            # A moved page's own relative links skew by the depth change
            # (../summaries/X.md -> concepts/summaries/X): the longest
            # existing TAIL of the wrong path is the intended target.
            parts = resolved[: -len(".md")].split("/")
            target = None
            for i in range(1, len(parts)):
                tail = "/".join(parts[i:])
                hit = pages.get(tail)
                if hit is not None:
                    target = hit
                    break
            if target is None:
                target = by_stem.get(parts[-1])
            if target is None:
                return m.group(0)
            new_rel = posixpath.relpath(str(target.relative_to(wiki)), own_dir or ".")
            rewritten += 1
            return m.group(0).replace(raw, new_rel)

        new_text = _MDLINK_RE.sub(_repl, text)
        if new_text != text:
            atomic_write_text(md, new_text)
    return rewritten
