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

OKF_VERSION = "0.2"

_STATUS_VALUES = {"draft", "stable", "deprecated"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LOG_DATE_HEADING_RE = re.compile(r"^##\s+\d{4}-\d{2}-\d{2}\s*$", re.MULTILINE)
_ISO_DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _is_actor(value) -> bool:
    """Whether ``value`` reads as an OKF §7 actor.

    ``human:<id>`` / ``process:<id>`` for people and automated processes,
    ``<producer>/<version>`` for agents and tools.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    v = value.strip()
    return v.startswith(("human:", "process:", "team:")) or "/" in v


def _check_verified(rel: str, verified) -> list[str]:
    """Issues for a ``verified`` value (§5.2).

    A bare ``{by, at}`` mapping is a conformant one-element list — §11 requires
    consumers to accept it — so it is normalized here rather than flagged.
    """
    entries = [verified] if isinstance(verified, dict) else verified
    if not isinstance(entries, list):
        return [f"{rel}: 'verified' must be a mapping or a list of mappings (OKF §5.2)"]
    out: list[str] = []
    for e in entries:
        if not isinstance(e, dict):
            out.append(f"{rel}: 'verified' entry is not a mapping (OKF §5.2)")
            continue
        if not _is_actor(e.get("by")):
            out.append(f"{rel}: 'verified.by' is not an OKF actor (OKF §7)")
        at = e.get("at")
        if at is not None and not _ISO_DT_RE.match(str(at)):
            out.append(f"{rel}: 'verified.at' is not an ISO 8601 datetime (OKF §5.2)")
    return out


def _check_optional_families(rel: str, fm: dict) -> list[str]:
    """Issues for the optional trust / lifecycle / provenance families (§5).

    Only checked when present: §11 forbids rejecting a concept for *missing*
    any optional family, so absence is never an issue here.
    """
    issues: list[str] = []

    sources = fm.get("sources")
    if sources is not None:
        if not isinstance(sources, list):
            issues.append(f"{rel}: 'sources' must be a list (OKF §5.1)")
        else:
            for entry in sources:
                if not isinstance(entry, dict):
                    issues.append(
                        f"{rel}: 'sources' entry is not a mapping — v0.1 shape, "
                        f"needs a 'resource' key (OKF §5.1)"
                    )
                elif not str(entry.get("resource", "") or "").strip():
                    issues.append(f"{rel}: 'sources' entry lacks a 'resource' (OKF §5.1)")

    generated = fm.get("generated")
    if generated is not None:
        if not isinstance(generated, dict):
            issues.append(f"{rel}: 'generated' must be a mapping (OKF §5.2)")
        else:
            if not _is_actor(generated.get("by")):
                issues.append(f"{rel}: 'generated.by' is not an OKF actor (OKF §7)")
            at = generated.get("at")
            if at is not None and not _ISO_DT_RE.match(str(at)):
                issues.append(f"{rel}: 'generated.at' is not an ISO 8601 datetime (OKF §5.2)")

    if fm.get("verified") is not None:
        issues.extend(_check_verified(rel, fm["verified"]))

    status = fm.get("status")
    if status is not None and str(status).strip() not in _STATUS_VALUES:
        issues.append(f"{rel}: 'status' must be draft|stable|deprecated (OKF §5.4)")

    stale_after = fm.get("stale_after")
    if stale_after is not None and not _DATE_RE.match(str(stale_after).strip()):
        issues.append(f"{rel}: 'stale_after' must be an absolute YYYY-MM-DD date (OKF §5.5)")

    return issues


def okf_check(wiki: Path) -> list[str]:
    """OKF conformance issues for the wiki bundle.

    Checks (per the Open Knowledge Format spec, v0.2):
    - every ``.md`` page carries parseable YAML frontmatter with a
      non-empty ``type`` (reserved files ``index.md`` / ``log.md`` are
      exempt at *any* depth — their role is positional, not typed);
    - the reserved files exist and follow §8 / §9;
    - the optional trust / lifecycle / provenance families, *when present*,
      are well-formed.

    Deliberately permissive, per §11: unknown types, unknown extra keys,
    broken cross-links and missing optional fields are never issues.

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
        # §3.1 reserves these filenames at every level of the hierarchy, not
        # just the bundle root — topic-tree mode nests directories that may
        # carry their own index.md.
        if md.name in reserved:
            continue
        text = _read_md(md)
        fm = frontmatter.parse(text)
        if not fm:
            issues.append(f"{rel}: missing or malformed frontmatter")
            continue
        if not str(fm.get("type", "") or "").strip():
            issues.append(f"{rel}: frontmatter lacks a non-empty 'type'")
        issues.extend(_check_optional_families(rel, fm))

    issues.extend(_check_index(wiki))
    issues.extend(_check_log(wiki))
    return sorted(issues)


def _check_index(wiki: Path) -> list[str]:
    """Issues for the bundle-root ``index.md`` (§8, §12)."""
    index_md = wiki / "index.md"
    if not index_md.exists():
        return ["index.md: missing (OKF reserved file)"]
    text = _read_md(index_md)
    fm = frontmatter.parse(text)
    if fm:
        # §8: index.md carries no frontmatter, with exactly one exception —
        # a bundle-root okf_version (§12).
        extra = sorted(k for k in fm if k != "okf_version")
        if extra:
            return [f"index.md: frontmatter may only carry 'okf_version' (OKF §8): {extra}"]
        return []
    if not text.lstrip().startswith("#"):
        return ["index.md: does not open with a Markdown heading"]
    return []


def _check_log(wiki: Path) -> list[str]:
    """Issues for the bundle-root ``log.md`` (§9)."""
    log_md = wiki / "log.md"
    if not log_md.exists():
        return ["log.md: missing (OKF reserved file)"]
    text = _read_md(log_md)
    headings = [ln for ln in text.split("\n") if ln.startswith("## ")]
    if not headings:
        # An empty log (freshly-initialized wiki) has nothing to violate.
        return []
    if not _LOG_DATE_HEADING_RE.search(text):
        return ["log.md: entries must be grouped under '## YYYY-MM-DD' headings (OKF §9)"]
    bad = [h for h in headings if not _LOG_DATE_HEADING_RE.match(h)]
    if bad:
        return [f"log.md: {len(bad)} heading(s) are not a bare '## YYYY-MM-DD' date (OKF §9)"]
    return []


def _migrate_frontmatter(fm_block: str, actor: str) -> str:
    """Rewrite one v0.1 frontmatter block into v0.2 shape.

    Two changes, both line-local so the resulting git diff stays readable:

    - ``sources: ["a.md"]`` → ``sources: [{"id": "a", "resource": "a.md"}]``
      (§5.1). Entries already in mapping form are left untouched.
    - ``timestamp: <iso>`` → ``generated: {"by": <actor>, "at": <iso>}``
      (§5.2/§13.1), replacing the line in place and preserving the original
      instant. Skipped when ``generated`` is already present.

    Re-dumping the whole block through yaml would reorder and requote every
    key, so the block is edited line by line instead.
    """
    lines = fm_block.split("\n")
    has_generated = any(ln.startswith("generated:") for ln in lines)

    for i, line in enumerate(lines):
        if line.startswith("sources:"):
            items = frontmatter.parse_list_value(line)
            if items and any(not isinstance(x, dict) for x in items):
                paths = [frontmatter.source_resource(x) for x in items]
                entries = [frontmatter.okf_source_entry(p) for p in paths]
                lines[i] = frontmatter.list_line("sources", entries)
        elif line.startswith("timestamp:") and not has_generated:
            raw = line.split(":", 1)[1].strip().strip("\"'")
            lines[i] = frontmatter.flow_map_line("generated", {"by": actor, "at": raw})

    return "\n".join(lines)


_LEGACY_LOG_RE = re.compile(
    r"^## \[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\] (\S+) \| (.*)$",
)


def _migrate_log(text: str) -> str | None:
    """Reshape a legacy ``log.md`` into OKF §9 form, or None if nothing to do.

    The v0.1 line ``## [2026-07-22 18:30:02] ingest | doc.md`` already carries
    every field §9 wants — date, time, operation, description — so regrouping
    it under bare ``## YYYY-MM-DD`` headings is lossless reformatting, not a
    rewrite of history. Returns None when no legacy heading is present, so
    already-migrated logs are left byte-identical.
    """
    by_day: dict[str, list[str]] = {}
    found = False
    for line in text.split("\n"):
        m = _LEGACY_LOG_RE.match(line)
        if not m:
            continue
        found = True
        day, clock, operation, description = m.groups()
        by_day.setdefault(day, []).append(f"* **{operation.title()}**: {description} ({clock})")
    if not found:
        return None

    from okforge.log import _HEADER

    out = [_HEADER, ""]
    for day in sorted(by_day, reverse=True):  # §9: newest first
        out.append(f"## {day}")
        out.extend(reversed(by_day[day]))
        out.append("")
    return "\n".join(out)


def okf_migrate(wiki: Path, actor: str, dry_run: bool = False) -> list[str]:
    """Migrate a wiki bundle's frontmatter from OKF v0.1 to v0.2 in place.

    ``actor`` is recorded as ``generated.by`` on pages that only carried a
    v0.1 ``timestamp``; callers should pass the KB's own configured model
    (via :func:`okforge.frontmatter.okf_actor`) rather than a synthetic
    identity, since that is what actually wrote the content.

    The bundle-root ``index.md`` gains the ``okf_version`` declaration (§12),
    and ``log.md`` is regrouped into §9 form — a lossless reshape of lines
    that already carry every field the spec asks for.

    Returns the bundle-relative paths that changed (or would change, under
    ``dry_run``).
    """
    changed: list[str] = []

    for md in sorted(wiki.rglob("*.md")):
        rel_parts = md.relative_to(wiki).parts
        if any(p.startswith(".") for p in rel_parts[:-1]) or rel_parts[0] == "reports":
            continue
        if md.name in {"index.md", "log.md"}:
            continue
        rel = "/".join(rel_parts)
        text = _read_md(md)
        parts = frontmatter.split(text)
        if parts is None:
            continue
        fm_block, body = parts
        new_block = _migrate_frontmatter(fm_block, actor)
        if new_block == fm_block:
            continue
        changed.append(rel)
        if not dry_run:
            atomic_write_text(md, new_block + body)

    index_md = wiki / "index.md"
    if index_md.exists():
        text = _read_md(index_md)
        if frontmatter.split(text) is None:
            changed.append("index.md")
            if not dry_run:
                atomic_write_text(index_md, f'---\nokf_version: "{OKF_VERSION}"\n---\n\n{text}')

    log_md = wiki / "log.md"
    if log_md.exists():
        migrated = _migrate_log(_read_md(log_md))
        if migrated is not None:
            changed.append("log.md")
            if not dry_run:
                atomic_write_text(log_md, migrated)

    return changed


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
