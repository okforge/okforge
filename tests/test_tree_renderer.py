"""Tests for okforge.tree_renderer."""

from __future__ import annotations

from okforge.tree_renderer import render_summary_md

# ---------------------------------------------------------------------------
# render_summary_md
# ---------------------------------------------------------------------------


class TestRenderSummaryMd:
    def test_has_yaml_frontmatter(self, sample_tree):
        output = render_summary_md(sample_tree, "Sample Document", "doc-abc")
        assert output.startswith("---\n")
        assert "doc_type: pageindex" in output
        assert 'full_text: "sources/Sample Document.json"' in output

    def test_top_level_nodes_are_h1(self, sample_tree):
        output = render_summary_md(sample_tree, "Sample Document", "doc-abc")
        assert "# Introduction" in output
        assert "# Conclusion" in output

    def test_nested_nodes_are_h2(self, sample_tree):
        output = render_summary_md(sample_tree, "Sample Document", "doc-abc")
        assert "## Background" in output
        assert "## Motivation" in output

    def test_page_range_included(self, sample_tree):
        output = render_summary_md(sample_tree, "Sample Document", "doc-abc")
        assert "(pages 0–120)" in output
        assert "(pages 121–200)" in output

    def test_summary_and_source_text_both_included(self, sample_tree):
        output = render_summary_md(sample_tree, "Sample Document", "doc-abc")
        assert "Summary: Overview of the document topic." in output
        assert "Summary: Historical context." in output
        # The real per-node source text is now quoted too, not just a
        # paraphrase — IndexConfig(if_add_node_text=True) already fetches
        # it, the old renderer just silently discarded it.
        assert "Source text:" in output
        assert "> This document introduces the core concepts of the system." in output

    def test_node_without_text_has_no_source_text_block(self):
        tree = {
            "structure": [
                {"title": "Intro", "start_index": 1, "end_index": 2, "summary": "x", "nodes": []}
            ]
        }
        output = render_summary_md(tree, "my-doc", "doc-123")
        assert "Source text:" not in output

    def test_internal_pageindex_image_refs_are_stripped_from_source_text(self):
        # PageIndex's own image refs point into its private
        # .okforge/files/{doc_id}/images/... cache, which never resolves from
        # a wiki page, so they're stripped rather than quoted verbatim.
        tree = {
            "structure": [
                {
                    "title": "Intro",
                    "start_index": 1,
                    "end_index": 2,
                    "summary": "x",
                    "text": "Some text.\n![fig](/private/cache/img.png)\nMore text.",
                    "nodes": [],
                }
            ]
        }
        output = render_summary_md(tree, "my-doc", "doc-123")
        assert "![fig]" not in output
        assert "> Some text." in output
        assert "> More text." in output


def test_summary_md_has_type_and_description():
    tree = {
        "structure": [
            {"title": "Intro", "start_index": 1, "end_index": 2, "summary": "x", "nodes": []}
        ]
    }
    md = render_summary_md(tree, "my-doc", "doc-123", description="Quarterly report.")
    assert 'type: "Summary"' in md
    assert 'description: "Quarterly report."' in md
    assert "doc_type: pageindex" in md
    assert 'full_text: "sources/my-doc.json"' in md


def test_overlong_title_is_truncated_in_heading():
    long_title = (
        "This is an entire source sentence copied verbatim into the title "
        "field because the no-TOC fallback found no natural short heading "
        "to extract from this section of the document."
    )
    tree = {
        "structure": [
            {
                "title": long_title,
                "start_index": 1,
                "end_index": 2,
                "summary": "x",
                "nodes": [],
            }
        ]
    }
    md = render_summary_md(tree, "my-doc", "doc-123")
    assert long_title not in md
    assert f"# {long_title[:80]}…" in md


def test_short_title_is_not_truncated():
    tree = {
        "structure": [
            {"title": "Background", "start_index": 1, "end_index": 2, "summary": "x", "nodes": []}
        ]
    }
    md = render_summary_md(tree, "my-doc", "doc-123")
    assert "# Background (pages 1–2)" in md


def test_duplicate_sibling_summaries_collapse_to_a_pointer():
    # Two sibling nodes on the same physical page can be handed the exact
    # same LLM-written summary (PageIndex#340). The second occurrence should
    # collapse to a pointer instead of repeating the block verbatim.
    tree = {
        "structure": [
            {
                "title": "1.1 First item",
                "start_index": 5,
                "end_index": 5,
                "summary": "Shared duplicate summary text.",
                "nodes": [],
            },
            {
                "title": "1.2 Second item",
                "start_index": 5,
                "end_index": 5,
                "summary": "Shared duplicate summary text.",
                "nodes": [],
            },
        ]
    }
    md = render_summary_md(tree, "my-doc", "doc-123")
    assert md.count("Summary: Shared duplicate summary text.") == 1
    assert '_(same content as "1.1 First item" above)_' in md


def test_duplicate_summaries_collapse_across_cousins_not_just_siblings():
    # The collision isn't confined to direct siblings — a node nested under
    # a different parent, seen later in document order, can repeat the same
    # summary too.
    tree = {
        "structure": [
            {
                "title": "Parent A",
                "start_index": 1,
                "end_index": 1,
                "summary": "",
                "nodes": [
                    {
                        "title": "Child A.1",
                        "start_index": 1,
                        "end_index": 1,
                        "summary": "Cousin duplicate.",
                        "nodes": [],
                    }
                ],
            },
            {
                "title": "Parent B",
                "start_index": 2,
                "end_index": 2,
                "summary": "",
                "nodes": [
                    {
                        "title": "Child B.1",
                        "start_index": 2,
                        "end_index": 2,
                        "summary": "Cousin duplicate.",
                        "nodes": [],
                    }
                ],
            },
        ]
    }
    md = render_summary_md(tree, "my-doc", "doc-123")
    assert md.count("Summary: Cousin duplicate.") == 1
    assert '_(same content as "Child A.1" above)_' in md


def test_distinct_summaries_are_not_collapsed():
    tree = {
        "structure": [
            {
                "title": "A",
                "start_index": 1,
                "end_index": 1,
                "summary": "Summary one.",
                "nodes": [],
            },
            {
                "title": "B",
                "start_index": 2,
                "end_index": 2,
                "summary": "Summary two.",
                "nodes": [],
            },
        ]
    }
    md = render_summary_md(tree, "my-doc", "doc-123")
    assert "Summary: Summary one." in md
    assert "Summary: Summary two." in md
    assert "same content as" not in md


def test_node_images_are_embedded_from_pages(sample_tree):
    # "Introduction" spans pages 0-120; page 5 is inside that range.
    pages = [{"page": 5, "content": "...", "images": [{"path": "sources/images/doc/p5.png"}]}]
    md = render_summary_md(sample_tree, "Sample Document", "doc-abc", pages=pages)
    assert "![image](sources/images/doc/p5.png)" in md


def test_no_pages_argument_means_no_images(sample_tree):
    md = render_summary_md(sample_tree, "Sample Document", "doc-abc")
    assert "![image]" not in md


def test_image_on_a_page_shared_by_two_nodes_is_not_duplicated():
    # Two sibling nodes both cover page 3 (a "no TOC" fallback can split one
    # physical page across several nodes) — the image on that page should
    # only be attached to the first of them, not repeated at both.
    tree = {
        "structure": [
            {"title": "A", "start_index": 3, "end_index": 3, "summary": "", "nodes": []},
            {"title": "B", "start_index": 3, "end_index": 3, "summary": "", "nodes": []},
        ]
    }
    pages = [{"page": 3, "content": "...", "images": [{"path": "sources/images/doc/p3.png"}]}]
    md = render_summary_md(tree, "my-doc", "doc-123", pages=pages)
    assert md.count("![image](sources/images/doc/p3.png)") == 1


def test_multiple_images_on_one_page_all_render_in_order():
    tree = {
        "structure": [{"title": "A", "start_index": 1, "end_index": 1, "summary": "", "nodes": []}]
    }
    pages = [
        {
            "page": 1,
            "content": "...",
            "images": [
                {"path": "sources/images/doc/a.png"},
                {"path": "sources/images/doc/b.png"},
            ],
        }
    ]
    md = render_summary_md(tree, "my-doc", "doc-123", pages=pages)
    a_idx = md.index("a.png")
    b_idx = md.index("b.png")
    assert a_idx < b_idx


def test_summary_full_text_quoted_yaml_safe():
    import yaml

    tree = {"structure": []}
    md = render_summary_md(tree, "weird: name", "doc-1", description="d")
    # full_text is JSON-quoted, so a source name with a colon stays valid YAML
    assert 'full_text: "sources/weird: name.json"' in md
    fm = yaml.safe_load(md.split("---")[1])
    assert fm["full_text"] == "sources/weird: name.json"
    assert fm["type"] == "Summary"
