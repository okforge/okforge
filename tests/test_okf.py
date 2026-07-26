"""Tests for OKF v0.2 conformance checking and the v0.1 → v0.2 migration.

Covers ``okforge.okf.okf_check`` (the ``okf-lint`` engine) and
``okforge.okf.okf_migrate`` (the ``okf-migrate`` command), plus the one
behaviour a wrong ``sources`` shape breaks silently: ``okforge remove``'s
reverse index.
"""

from __future__ import annotations

from okforge import frontmatter
from okforge.agent.compiler import (
    _prepend_source_to_frontmatter,
    _remove_source_from_frontmatter,
    scan_affected_pages,
)
from okforge.okf import okf_check, okf_migrate

ACTOR = "okforge/test-model"


def _bundle(tmp_path, pages=None):
    """A minimal conformant v0.2 bundle; ``pages`` maps rel path -> text."""
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "index.md").write_text('---\nokf_version: "0.2"\n---\n\n# Index\n')
    (wiki / "log.md").write_text(
        "# Operations Log\n\n## 2026-07-22\n* **Ingest**: a.md (10:00:00)\n"
    )
    for rel, text in (pages or {}).items():
        (wiki / rel).write_text(text)
    return wiki


class TestConformance:
    def test_clean_v02_bundle_has_no_issues(self, tmp_path):
        wiki = _bundle(
            tmp_path,
            {
                "concepts/a.md": (
                    '---\ntype: "Concept"\n'
                    'sources: [{"id": "a", "resource": "summaries/a.md"}]\n'
                    'generated: {"by": "okforge/gpt-5.4", "at": "2026-07-22T10:00:00+00:00"}\n'
                    "status: stable\nstale_after: 2026-12-31\n---\n\nBody.\n"
                )
            },
        )
        assert okf_check(wiki) == []

    def test_v01_sources_shape_is_reported(self, tmp_path):
        wiki = _bundle(
            tmp_path,
            {"concepts/a.md": '---\ntype: "Concept"\nsources: ["summaries/a.md"]\n---\n\nBody.\n'},
        )
        issues = okf_check(wiki)
        assert any("not a mapping" in i and "concepts/a.md" in i for i in issues)

    def test_sources_entry_without_resource_is_reported(self, tmp_path):
        wiki = _bundle(
            tmp_path,
            {"concepts/a.md": '---\ntype: "Concept"\nsources: [{"id": "a"}]\n---\n\nBody.\n'},
        )
        assert any("lacks a 'resource'" in i for i in okf_check(wiki))

    def test_bare_verified_mapping_is_accepted(self, tmp_path):
        """§11: consumers MUST treat a bare mapping as a one-element list."""
        wiki = _bundle(
            tmp_path,
            {
                "concepts/a.md": (
                    '---\ntype: "Concept"\n'
                    'verified: {"by": "human:dana", "at": "2026-07-22T10:00:00Z"}\n---\n\nB.\n'
                )
            },
        )
        assert okf_check(wiki) == []

    def test_non_actor_verifier_is_reported(self, tmp_path):
        wiki = _bundle(
            tmp_path,
            {"concepts/a.md": '---\ntype: "Concept"\nverified: {"by": "dana"}\n---\n\nB.\n'},
        )
        assert any("not an OKF actor" in i for i in okf_check(wiki))

    def test_bad_status_and_stale_after_are_reported(self, tmp_path):
        wiki = _bundle(
            tmp_path,
            {"concepts/a.md": '---\ntype: "Concept"\nstatus: live\nstale_after: soon\n---\n\nB.\n'},
        )
        issues = okf_check(wiki)
        assert any("draft|stable|deprecated" in i for i in issues)
        assert any("YYYY-MM-DD" in i for i in issues)

    def test_permissive_about_unknown_types_keys_and_broken_links(self, tmp_path):
        """§11 forbids rejecting a bundle for any of these."""
        wiki = _bundle(
            tmp_path,
            {
                "concepts/a.md": (
                    '---\ntype: "Wholly Invented Type"\nsome_extension_key: 7\n---\n\n'
                    "[missing](/concepts/nope.md)\n"
                )
            },
        )
        assert okf_check(wiki) == []

    def test_nested_reserved_files_are_exempt_at_any_depth(self, tmp_path):
        """§3.1 reserves index.md / log.md at every level, not just the root."""
        wiki = _bundle(tmp_path)
        nested = wiki / "concepts" / "attention"
        nested.mkdir()
        (nested / "index.md").write_text("# Attention\n\n* [a](a.md) - a concept\n")
        (nested / "log.md").write_text("# Log\n\n## 2026-07-22\n* **Update**: x\n")
        assert okf_check(wiki) == []

    def test_index_frontmatter_limited_to_okf_version(self, tmp_path):
        wiki = _bundle(tmp_path)
        (wiki / "index.md").write_text('---\nokf_version: "0.2"\ntitle: "Nope"\n---\n\n# Index\n')
        assert any("only carry 'okf_version'" in i for i in okf_check(wiki))

    def test_legacy_log_heading_is_reported(self, tmp_path):
        wiki = _bundle(tmp_path)
        (wiki / "log.md").write_text("# Operations Log\n\n## [2026-07-22 18:30:02] ingest | a.md\n")
        assert any("YYYY-MM-DD" in i and "log.md" in i for i in okf_check(wiki))

    def test_empty_log_is_not_an_issue(self, tmp_path):
        wiki = _bundle(tmp_path)
        (wiki / "log.md").write_text("# Operations Log\n\n")
        assert okf_check(wiki) == []


class TestMigration:
    def _v01_bundle(self, tmp_path):
        wiki = tmp_path / "wiki"
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "concepts" / "a.md").write_text(
            '---\ntype: "Concept"\nsources: ["summaries/doc.md"]\n'
            'description: "A thing."\ntitle: "a"\n'
            'timestamp: "2026-07-22T18:46:45+00:00"\n---\n\n# A\n\nBody.\n'
        )
        (wiki / "index.md").write_text("# Knowledge Base Index\n\n## Concepts\n")
        (wiki / "log.md").write_text(
            "# Operations Log\n\n"
            "## [2026-07-22 18:30:02] ingest | doc.md\n\n"
            "## [2026-07-23 09:00:00] query | what is a\n\n"
        )
        return wiki

    def test_migrates_sources_timestamp_index_and_log(self, tmp_path):
        wiki = self._v01_bundle(tmp_path)
        assert okf_check(wiki) != []

        changed = okf_migrate(wiki, ACTOR)
        assert set(changed) == {"concepts/a.md", "index.md", "log.md"}
        assert okf_check(wiki) == []

        fm = frontmatter.parse((wiki / "concepts" / "a.md").read_text())
        assert fm["sources"] == [{"id": "doc", "resource": "summaries/doc.md"}]
        # The original instant is preserved, not restamped.
        assert fm["generated"] == {"by": ACTOR, "at": "2026-07-22T18:46:45+00:00"}
        assert "timestamp" not in fm
        # Untouched keys survive.
        assert fm["description"] == "A thing." and fm["title"] == "a"

    def test_log_migration_is_lossless_and_newest_first(self, tmp_path):
        wiki = self._v01_bundle(tmp_path)
        okf_migrate(wiki, ACTOR)
        text = (wiki / "log.md").read_text()
        assert text.index("## 2026-07-23") < text.index("## 2026-07-22")
        assert "* **Ingest**: doc.md (18:30:02)" in text
        assert "* **Query**: what is a (09:00:00)" in text

    def test_dry_run_writes_nothing(self, tmp_path):
        wiki = self._v01_bundle(tmp_path)
        before = (wiki / "concepts" / "a.md").read_text()
        changed = okf_migrate(wiki, ACTOR, dry_run=True)
        assert changed
        assert (wiki / "concepts" / "a.md").read_text() == before

    def test_is_idempotent(self, tmp_path):
        wiki = self._v01_bundle(tmp_path)
        okf_migrate(wiki, ACTOR)
        after_first = (wiki / "concepts" / "a.md").read_text()
        assert okf_migrate(wiki, ACTOR) == []
        assert (wiki / "concepts" / "a.md").read_text() == after_first

    def test_existing_generated_is_not_overwritten(self, tmp_path):
        wiki = self._v01_bundle(tmp_path)
        (wiki / "concepts" / "b.md").write_text(
            '---\ntype: "Concept"\n'
            'generated: {"by": "human:dana", "at": "2026-01-01T00:00:00+00:00"}\n'
            'timestamp: "2026-07-22T18:46:45+00:00"\n---\n\nB\n'
        )
        okf_migrate(wiki, ACTOR)
        fm = frontmatter.parse((wiki / "concepts" / "b.md").read_text())
        assert fm["generated"]["by"] == "human:dana"


class TestSourcesShapeDoesNotBreakRemove:
    """The `sources` reverse index is what `okforge remove` prunes by.

    A shape it can't parse fails *silently* — pages simply stop being pruned —
    so these assert on parsed entries rather than substring matches.
    """

    def _page(self, *paths):
        line = frontmatter.list_line("sources", [frontmatter.okf_source_entry(p) for p in paths])
        return f'---\ntype: "Concept"\n{line}\n---\n\nbody\n'

    def test_remove_shrinks_by_exactly_one(self):
        text = self._page("summaries/a.md", "summaries/b.md")
        rewritten, empty = _remove_source_from_frontmatter(text, "summaries/a.md")
        assert empty is False
        sources = frontmatter.parse(rewritten)["sources"]
        assert [frontmatter.source_resource(s) for s in sources] == ["summaries/b.md"]

    def test_remove_last_source_marks_empty(self):
        text = self._page("summaries/a.md")
        rewritten, empty = _remove_source_from_frontmatter(text, "summaries/a.md")
        assert empty is True
        assert frontmatter.parse(rewritten)["sources"] == []

    def test_prepend_is_deduplicated(self):
        text = self._page("summaries/a.md")
        once = _prepend_source_to_frontmatter(text, "summaries/b.md")
        twice = _prepend_source_to_frontmatter(once, "summaries/b.md")
        assert once == twice
        sources = frontmatter.parse(twice)["sources"]
        assert [frontmatter.source_resource(s) for s in sources] == [
            "summaries/b.md",
            "summaries/a.md",
        ]

    def test_v01_pages_still_prune(self):
        """An unmigrated wiki must keep working — no forced migration."""
        text = '---\ntype: "Concept"\nsources: ["summaries/a.md", "summaries/b.md"]\n---\n\nbody\n'
        rewritten, empty = _remove_source_from_frontmatter(text, "summaries/a.md")
        assert empty is False
        sources = frontmatter.parse(rewritten)["sources"]
        assert [frontmatter.source_resource(s) for s in sources] == ["summaries/b.md"]

    def test_dry_run_preview_sees_both_shapes(self, tmp_path):
        pages = tmp_path / "concepts"
        pages.mkdir()
        (pages / "v2.md").write_text(self._page("summaries/a.md", "summaries/b.md"))
        (pages / "v1.md").write_text(
            '---\ntype: "Concept"\nsources: ["summaries/a.md"]\n---\n\nbody\n'
        )
        assert scan_affected_pages(pages, "summaries/a.md") == [("v1", 0), ("v2", 1)]
