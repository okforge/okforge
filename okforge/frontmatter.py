"""Shared helpers for the YAML frontmatter blocks of okforge wiki pages.

Single source of truth for building, splitting, parsing, and mutating the
``---`` frontmatter used by summaries / concepts / entities. The closing
delimiter is always matched at the start of a line (``\\n---``), so a ``---``
that appears inside a quoted value never truncates the block — the failure
mode that ad-hoc ``text.find("---", 3)`` parsing was prone to.
"""

from __future__ import annotations

import json
import re

import yaml


def kv_line(key: str, value: str) -> str:
    """Render ``key: "value"`` with the value JSON-quoted (always single-line).

    JSON strings are a strict subset of YAML: always single-line, always
    correctly escaped (newlines, quotes, colons, control chars), and never
    auto-promoted to a block scalar.
    """
    return f"{key}: {json.dumps(value, ensure_ascii=False)}"


def list_line(key: str, items) -> str:
    """Render ``key: ["a", "b"]`` as JSON-style YAML (always single-line)."""
    return f"{key}: {json.dumps(list(items), ensure_ascii=False)}"


def block(lines: list[str]) -> str:
    """Assemble a complete frontmatter block (with delimiters + trailing blank)."""
    return "---\n" + "\n".join(lines) + "\n---\n\n"


def parse_list_value(line: str) -> list | None:
    """Parse the right-hand side of ``key: [...]`` into a list.

    Entries are returned as-is: strings stay strings, and OKF ``sources``
    mappings (``{id, resource}``, §5.1) stay dicts. Coercing everything to
    ``str`` would turn a mapping into ``"{'id': ...}"`` and silently destroy
    it on the next round-trip.

    Returns ``None`` when the value cannot be interpreted as a list — callers
    treat that as "leave the frontmatter alone".
    """
    colon = line.find(":")
    if colon == -1:
        return None
    try:
        parsed = yaml.safe_load(line[colon + 1 :])
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def split(text: str) -> tuple[str, str] | None:
    """Split ``text`` into ``(frontmatter_block, body)``.

    ``frontmatter_block`` includes both ``---`` delimiters and the newline that
    ends the closing delimiter line; ``body`` is everything after, so
    ``frontmatter_block + body == text`` exactly (lossless).

    Returns ``None`` when ``text`` has no well-formed frontmatter: no leading
    ``---`` or no line-anchored closing ``---``. Because the closing delimiter
    must start a line (``\\n---``), a ``---`` inside a quoted value is ignored.
    """
    if not text.startswith("---"):
        return None
    nl = text.find("\n---", 3)
    if nl == -1:
        return None
    after = text.find("\n", nl + 1)  # newline ending the closing '---' line
    if after == -1:
        return text, ""
    return text[: after + 1], text[after + 1 :]


def parse(text: str) -> dict:
    """Return the frontmatter as a dict (``{}`` when absent or malformed)."""
    parts = split(text)
    if parts is None:
        return {}
    fm_block = parts[0]
    inner = fm_block[3:]  # drop opening '---'
    close = inner.rfind("\n---")  # drop closing '---' line
    if close != -1:
        inner = inner[:close]
    try:
        data = yaml.safe_load(inner)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def set_raw_line(fm_block: str, key: str, line: str) -> str:
    """Set or insert an already-rendered ``key: …`` line in a frontmatter block.

    Replaces an existing line for ``key``; otherwise inserts it right after the
    opening ``---``. A lambda replacement is used so values containing regex
    backrefs (``\\1``, ``\\g<…>``) are inserted literally.

    ``line`` must be a single line: the match is anchored per-line, so a
    multi-line value would leave its continuation lines orphaned.
    """
    if re.search(rf"^{re.escape(key)}:", fm_block, flags=re.MULTILINE):
        return re.sub(
            rf"^{re.escape(key)}:.*", lambda _m: line, fm_block, count=1, flags=re.MULTILINE
        )
    return fm_block.replace("---\n", f"---\n{line}\n", 1)


def set_line(fm_block: str, key: str, value: str) -> str:
    """Set or insert a single scalar ``key:`` line in a frontmatter block."""
    return set_raw_line(fm_block, key, kv_line(key, value))


def drop_line(fm_block: str, key: str) -> str:
    """Remove any ``key:`` line from a frontmatter block (no-op if absent)."""
    return re.sub(rf"^{re.escape(key)}:.*\n?", "", fm_block, flags=re.MULTILINE)


def okf_timestamp() -> str:
    """Current time as ISO-8601 with offset, second precision (OKF field)."""
    import datetime

    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# Actor for pages written without an LLM in the loop (OKF §7 ``process:<id>``).
CONVERT_ACTOR = "process:okforge-convert"


def okf_actor(model: str) -> str:
    """An OKF §7 actor for content an LLM produced: ``<producer>/<version>``.

    The provider prefix LiteLLM needs (``openai/Qwen3.6-27B-MTP``) is routing
    detail, not identity — the actor names okforge as the producer and the
    model as the version, so ``openai/Qwen3.6-27B-MTP`` becomes
    ``okforge/Qwen3.6-27B-MTP``.
    """
    version = (model or "").rsplit("/", 1)[-1].strip()
    return f"okforge/{version}" if version else "okforge"


def flow_map_line(key: str, mapping: dict) -> str:
    """Render ``key: {"a": "b"}`` as a single-line YAML flow mapping.

    JSON is a strict subset of YAML, so a JSON object is a valid flow mapping.
    Staying on one line is what keeps the line-oriented helpers here
    (``set_line``, ``drop_line``) and the compiler's ``sources:`` scanners
    working — a block-style mapping would split across lines and they would
    silently stop matching.
    """
    return f"{key}: {json.dumps(mapping, ensure_ascii=False)}"


def okf_source_entry(path: str) -> dict:
    """An OKF §5.1 ``sources`` entry for a bundle-relative page ``path``.

    ``resource`` is required by the spec; ``id`` is the file stem, a stable
    key for per-claim attribution that survives the list being reordered
    (the spec's stated reason for keying attribution rather than indexing it).
    """
    stem = path.rsplit("/", 1)[-1]
    if stem.endswith(".md"):
        stem = stem[: -len(".md")]
    return {"id": stem, "resource": path}


def source_resource(entry) -> str:
    """The ``resource`` of a ``sources`` entry, tolerating the v0.1 shape.

    v0.1 bundles (and any not-yet-migrated wiki) carry bare strings where
    v0.2 carries ``{id, resource}`` mappings. Readers go through here so both
    shapes resolve to a path.
    """
    if isinstance(entry, dict):
        return str(entry.get("resource", "") or "")
    return str(entry)


def okf_generated_line(actor: str) -> str:
    """The OKF §5.2 ``generated: {by, at}`` line for a page written now."""
    return flow_map_line("generated", {"by": actor, "at": okf_timestamp()})


def okf_meta_lines(title: str, actor: str) -> list[str]:
    """OKF-recommended metadata lines: ``title`` + ``generated`` (§5.2).

    Supersedes the v0.1 ``timestamp`` field, which OKF v0.2 §13.1 retires in
    favour of ``generated.at``.
    """
    return [kv_line("title", title), okf_generated_line(actor)]


def refresh_okf_meta(fm_block: str, title: str, actor: str) -> str:
    """Set/refresh ``title`` + ``generated`` in an existing frontmatter block.

    A legacy ``timestamp`` line is dropped: ``generated.at`` now records the
    content's last meaningful change, and leaving both would let them drift.
    """
    fm_block = set_line(fm_block, "title", title)
    fm_block = drop_line(fm_block, "timestamp")
    return set_raw_line(fm_block, "generated", okf_generated_line(actor))


def body(text: str) -> str:
    """Return ``text`` without its frontmatter block (unchanged when absent)."""
    parts = split(text)
    if parts is None:
        return text
    return parts[1].lstrip("\n")
