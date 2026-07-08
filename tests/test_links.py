"""Tests for openkb.links — wikilink → relative markdown link conversion."""

from __future__ import annotations

from openkb.links import link_style, wikilinks_to_markdown


class TestLinkStyle:
    def test_default_is_markdown(self):
        assert link_style({}) == "markdown"

    def test_wikilinks_opt_out(self):
        assert link_style({"link_style": "wikilinks"}) == "wikilinks"

    def test_garbage_falls_back_to_markdown(self):
        assert link_style({"link_style": "obsidian++"}) == "markdown"


class TestWikilinksToMarkdown:
    def test_dir_prefixed_target_from_summary(self):
        out = wikilinks_to_markdown("See [[concepts/attention]].", "summaries")
        assert out == "See [attention](../concepts/attention.md)."

    def test_alias_label_kept(self):
        out = wikilinks_to_markdown("[[entities/fort-marion|Fort Marion]]", "summaries")
        assert out == "[Fort Marion](../entities/fort-marion.md)"

    def test_same_dir_target(self):
        out = wikilinks_to_markdown("[[concepts/attention]]", "concepts")
        assert out == "[attention](attention.md)"

    def test_index_page_own_dir_empty(self):
        out = wikilinks_to_markdown("[[summaries/paper]]", "")
        assert out == "[paper](summaries/paper.md)"

    def test_index_target(self):
        out = wikilinks_to_markdown("[[index]]", "concepts")
        assert out == "[index](../index.md)"

    def test_unknown_dir_prefix_left_alone(self):
        text = "[[private/notes]]"
        assert wikilinks_to_markdown(text, "summaries") == text

    def test_bare_stem_resolved_when_file_exists(self, tmp_path):
        (tmp_path / "concepts").mkdir()
        (tmp_path / "concepts" / "attention.md").write_text("x")
        out = wikilinks_to_markdown("[[attention]]", "summaries", tmp_path)
        assert out == "[attention](../concepts/attention.md)"

    def test_bare_stem_unresolved_left_alone(self, tmp_path):
        (tmp_path / "concepts").mkdir()
        text = "[[nonexistent]]"
        assert wikilinks_to_markdown(text, "summaries", tmp_path) == text

    def test_bare_stem_ambiguous_left_alone(self, tmp_path):
        for sub in ("concepts", "entities"):
            (tmp_path / sub).mkdir()
            (tmp_path / sub / "acme.md").write_text("x")
        text = "[[acme]]"
        assert wikilinks_to_markdown(text, "summaries", tmp_path) == text

    def test_multiple_links_in_one_body(self):
        out = wikilinks_to_markdown("[[concepts/a]] and [[entities/b|B Corp]]", "summaries")
        assert "[a](../concepts/a.md)" in out
        assert "[B Corp](../entities/b.md)" in out
