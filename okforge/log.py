"""Append-only operation log for the wiki (log.md)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_HEADER = "# Operations Log"


def append_log(wiki_dir: Path, operation: str, description: str) -> None:
    """Append an entry to wiki/log.md.

    Follows OKF §9: entries are grouped under bare ``## YYYY-MM-DD`` date
    headings, newest first, as ``* **Operation**: description (HH:MM:SS)``.
    The clock time rides in the entry rather than the heading because §9
    requires the heading be a date and nothing else.

    A new date heading is opened only when the most recent one is not today's,
    so a day's operations accumulate under a single heading.
    """
    log_path = wiki_dir / "log.md"
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    entry = f"* **{operation.title()}**: {description} ({now.strftime('%H:%M:%S')})"

    if not log_path.exists():
        log_path.write_text(f"{_HEADER}\n\n## {day}\n{entry}\n", encoding="utf-8")
        return

    text = log_path.read_text(encoding="utf-8")
    heading = f"## {day}"
    lines = text.split("\n")

    # Newest-first: the run of entries under today's heading ends at the next
    # heading, so the new entry goes immediately below the heading line.
    for i, line in enumerate(lines):
        if line.strip() == heading:
            lines.insert(i + 1, entry)
            log_path.write_text("\n".join(lines), encoding="utf-8")
            return

    # No heading for today yet — open one above the existing days, keeping the
    # file header (if any) on top.
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("## "):
            insert_at = i
            break
    else:
        # No date headings at all (fresh or legacy header-only file): append.
        while lines and lines[-1] == "":
            lines.pop()
        lines.extend(["", heading, entry, ""])
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines[insert_at:insert_at] = [heading, entry, ""]
    log_path.write_text("\n".join(lines), encoding="utf-8")
