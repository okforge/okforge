from pathlib import Path

from openkb import topic_tree as tt
from openkb.lint import list_existing_wiki_targets, strip_ghost_wikilinks


def _mk(p: Path, text: str = "x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_targets_include_bare_stem_for_nested_concept(tmp_path):
    wiki = tmp_path / "wiki"
    _mk(wiki / "concepts" / "attention-and-transformers" / "self-attention.md")
    targets = list_existing_wiki_targets(wiki)
    assert "self-attention" in targets  # bare stem resolves
    assert "concepts/attention-and-transformers/self-attention" in targets


def test_dir_prefixed_link_resolves_for_nested_concept(tmp_path):
    """Compiler-generated ``[[concepts/<stem>]]`` links must still resolve after
    a concept is nested under a topic dir (the form real concept bodies use)."""
    wiki = tmp_path / "wiki"
    _mk(wiki / "concepts" / "transformer" / "self-attention.md")
    targets = list_existing_wiki_targets(wiki)
    assert "concepts/self-attention" in targets
    out, ghosts = strip_ghost_wikilinks("see [[concepts/self-attention]]", targets)
    assert ghosts == []
    assert "[[concepts/self-attention]]" in out


def test_bare_stem_link_not_stripped_when_nested(tmp_path):
    wiki = tmp_path / "wiki"
    _mk(wiki / "concepts" / "topic" / "self-attention.md")
    targets = list_existing_wiki_targets(wiki)
    out, ghosts = strip_ghost_wikilinks("see [[self-attention]]", targets)
    assert ghosts == []  # link survives despite living in a subfolder
    assert "[[self-attention]]" in out


def test_link_resolves_after_split_move(tmp_path):
    wiki = tmp_path / "wiki"
    root = wiki / "concepts"
    tt.write_topic_md(root, "root", 0)
    (root / "self-attention.md").write_text("# self-attention\n", encoding="utf-8")
    tt.split_node(
        root,
        cluster=lambda items: {"attention": ["self-attention"]},
        summarize=lambda n, b: "s",
    )
    targets = list_existing_wiki_targets(wiki)
    out, ghosts = strip_ghost_wikilinks("see [[self-attention]]", targets)
    assert ghosts == []  # bare-stem link still resolves after the move


def test_all_wiki_pages_aliases_nested_concepts(tmp_path):
    from openkb.lint import find_broken_links

    wiki = tmp_path / "wiki"
    _mk(wiki / "concepts" / "transformer" / "self-attention.md")
    (wiki / "entities").mkdir()
    (wiki / "entities" / "bert.md").write_text("see [[concepts/self-attention]]", encoding="utf-8")
    assert find_broken_links(wiki) == []


def test_retarget_md_links_follows_moved_concept(tmp_path):
    from openkb.okf import retarget_md_links

    wiki = tmp_path / "wiki"
    _mk(wiki / "concepts" / "transformer" / "self-attention.md")
    (wiki / "entities").mkdir()
    page = wiki / "entities" / "bert.md"
    page.write_text("see [self-attention](../concepts/self-attention.md)", encoding="utf-8")
    n = retarget_md_links(wiki)
    assert n == 1
    assert "(../concepts/transformer/self-attention.md)" in page.read_text(encoding="utf-8")


def test_links_resolve_dir_prefixed_to_nested(tmp_path):
    from openkb.links import wikilinks_to_markdown

    (tmp_path / "concepts" / "transformer").mkdir(parents=True)
    (tmp_path / "concepts" / "transformer" / "self-attention.md").write_text("x")
    out = wikilinks_to_markdown("[[concepts/self-attention]]", "summaries", tmp_path)
    assert out == "[self-attention](../concepts/transformer/self-attention.md)"


def test_retarget_fixes_moved_pages_own_outgoing_links(tmp_path):
    """A concept moved one level deeper has its OWN ../summaries/X.md links
    skewed by the depth change; the tail-match must repair them even though
    the stem is ambiguous (summaries/ and sources/ share stems)."""
    from openkb.okf import retarget_md_links

    wiki = tmp_path / "wiki"
    _mk(wiki / "summaries" / "doc1.md")
    _mk(wiki / "sources" / "doc1.md")  # same stem — ambiguity trap
    page = wiki / "concepts" / "topic" / "idea.md"
    _mk(page, "see [doc1](../summaries/doc1.md)")
    n = retarget_md_links(wiki)
    assert n == 1
    assert "(../../summaries/doc1.md)" in page.read_text(encoding="utf-8")
