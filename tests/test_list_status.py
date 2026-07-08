"""Tests for okforge list and okforge status CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from okforge.cli import cli


def _setup_kb(tmp_path: Path) -> Path:
    """Create a minimal KB structure and return kb_dir."""
    kb_dir = tmp_path
    (kb_dir / "raw").mkdir()
    (kb_dir / "wiki" / "sources" / "images").mkdir(parents=True)
    (kb_dir / "wiki" / "summaries").mkdir(parents=True)
    (kb_dir / "wiki" / "concepts").mkdir(parents=True)
    (kb_dir / "wiki" / "entities").mkdir(parents=True)
    (kb_dir / "wiki" / "reports").mkdir(parents=True)
    openkb_dir = kb_dir / ".okforge"
    openkb_dir.mkdir()
    (openkb_dir / "config.yaml").write_text("model: gpt-4o-mini\n")
    (openkb_dir / "hashes.json").write_text(json.dumps({}))
    (kb_dir / "wiki" / "index.md").write_text(
        "# Knowledge Base Index\n\n## Documents\n\n## Concepts\n"
    )
    return kb_dir


class TestListCommand:
    def test_list_no_kb(self, tmp_path):
        runner = CliRunner()
        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("okforge.cli._find_kb_dir", return_value=None),
        ):
            result = runner.invoke(cli, ["list"])
            assert "No knowledge base found" in result.output

    def test_list_empty_kb(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["list"])
            assert "No documents indexed yet" in result.output

    def test_list_shows_documents(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        hashes = {
            "abc123": {"name": "paper.pdf", "type": "pdf"},
            "def456": {"name": "notes.md", "type": "md"},
        }
        (kb_dir / ".okforge" / "hashes.json").write_text(json.dumps(hashes))

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["list"])

        assert "paper.pdf" in result.output
        assert "notes.md" in result.output
        assert "pdf" in result.output
        assert "md" in result.output

    def test_list_shows_concepts(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        hashes = {"abc": {"name": "paper.pdf", "type": "pdf"}}
        (kb_dir / ".okforge" / "hashes.json").write_text(json.dumps(hashes))
        (kb_dir / "wiki" / "concepts" / "attention.md").write_text("# Attention")
        (kb_dir / "wiki" / "concepts" / "transformer.md").write_text("# Transformer")

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["list"])

        assert "attention" in result.output
        assert "transformer" in result.output

    def test_list_no_concepts_section_when_empty(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        hashes = {"abc": {"name": "paper.pdf", "type": "pdf"}}
        (kb_dir / ".okforge" / "hashes.json").write_text(json.dumps(hashes))

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        # No concepts in output since none exist
        assert "Concepts:" not in result.output

    def test_list_shows_entities(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        hashes = {"abc": {"name": "paper.pdf", "type": "pdf"}}
        (kb_dir / ".okforge" / "hashes.json").write_text(json.dumps(hashes))
        (kb_dir / "wiki" / "entities" / "ada-lovelace.md").write_text("# Ada")
        (kb_dir / "wiki" / "entities" / "openai.md").write_text("# OpenAI")

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["list"])

        assert "Entities (2):" in result.output
        assert "ada-lovelace" in result.output
        assert "openai" in result.output

    def test_list_no_entities_section_when_empty(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        hashes = {"abc": {"name": "paper.pdf", "type": "pdf"}}
        (kb_dir / ".okforge" / "hashes.json").write_text(json.dumps(hashes))

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Entities:" not in result.output
        assert "Entities (" not in result.output


class TestStatusCommand:
    def test_status_no_kb(self, tmp_path):
        runner = CliRunner()
        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("okforge.cli._find_kb_dir", return_value=None),
        ):
            result = runner.invoke(cli, ["status"])
            assert "No knowledge base found" in result.output

    def test_status_shows_directory_counts(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        # Add some files
        (kb_dir / "wiki" / "sources" / "doc1.md").write_text("# Doc 1")
        (kb_dir / "wiki" / "sources" / "doc2.md").write_text("# Doc 2")
        (kb_dir / "wiki" / "summaries" / "sum1.md").write_text("# Sum 1")
        (kb_dir / "wiki" / "concepts" / "concept1.md").write_text("# Concept")

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["status"])

        assert "sources" in result.output
        assert "summaries" in result.output
        assert "concepts" in result.output
        assert "entities" in result.output
        assert "reports" in result.output

    def test_status_shows_total_indexed(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        hashes = {
            "abc": {"name": "a.pdf", "type": "pdf"},
            "def": {"name": "b.pdf", "type": "pdf"},
            "ghi": {"name": "c.md", "type": "md"},
        }
        (kb_dir / ".okforge" / "hashes.json").write_text(json.dumps(hashes))

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["status"])

        assert "3" in result.output  # total indexed count

    def test_status_shows_raw_count(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        (kb_dir / "raw" / "file1.pdf").write_bytes(b"PDF")
        (kb_dir / "raw" / "file2.pdf").write_bytes(b"PDF")

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["status"])

        assert "raw" in result.output

    def test_status_exit_code_zero(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0


class TestStatusKbPath:
    """Status output must lead with the active KB path so agents and
    scripts can locate the wiki when invoked from outside the KB root."""

    def test_status_prints_kb_path_first(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        # First non-empty line carries the path in a parseable form:
        #   "Knowledge base: /path/to/kb"
        first_line = result.output.splitlines()[0]
        assert first_line.startswith("Knowledge base: ")
        assert first_line.split(": ", 1)[1] == str(kb_dir)


class TestListJson:
    def test_list_json_payload(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        hashes = {
            "abc123": {"name": "paper.pdf", "type": "long_pdf", "doc_name": "paper", "pages": 30},
            "def456": {"name": "notes.md", "type": "md"},
        }
        (kb_dir / ".okforge" / "hashes.json").write_text(json.dumps(hashes))
        (kb_dir / "wiki" / "summaries" / "paper.md").write_text("s")
        (kb_dir / "wiki" / "concepts" / "attention.md").write_text("c")
        (kb_dir / "wiki" / "entities" / "acme.md").write_text("e")

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["kb_dir"] == str(kb_dir)
        by_name = {d["name"]: d for d in data["documents"]}
        assert by_name["paper.pdf"]["doc_name"] == "paper"
        assert by_name["paper.pdf"]["display_type"] == "pageindex"
        assert by_name["paper.pdf"]["pages"] == 30
        assert by_name["notes.md"]["doc_name"] == "notes"
        assert by_name["notes.md"]["pages"] is None
        assert data["summaries"] == ["paper"]
        assert data["concepts"] == ["attention"]
        assert data["entities"] == ["acme"]

    def test_list_json_empty_kb(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["list", "--json"])
        data = json.loads(result.output)
        assert data["documents"] == []
        assert data["summaries"] == []

    def test_list_json_no_kb_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("okforge.cli._find_kb_dir", return_value=None),
        ):
            result = runner.invoke(cli, ["list", "--json"])
        assert result.exit_code == 1
        assert json.loads(result.output) == {"error": "no_kb"}

    def test_list_json_long_names_not_truncated(self, tmp_path):
        # The human Documents table truncates at 40 chars; JSON must not.
        kb_dir = _setup_kb(tmp_path)
        long_name = "a-very-long-document-name-that-goes-well-past-forty-characters.md"
        hashes = {"h1": {"name": long_name, "type": "md"}}
        (kb_dir / ".okforge" / "hashes.json").write_text(json.dumps(hashes))

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["list", "--json"])
        data = json.loads(result.output)
        assert data["documents"][0]["name"] == long_name


class TestStatusJson:
    def test_status_json_payload(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        (kb_dir / "wiki" / "summaries" / "doc.md").write_text("s")
        (kb_dir / "raw" / "doc.pdf").write_bytes(b"x")
        (kb_dir / ".okforge" / "hashes.json").write_text(json.dumps({"h": {"name": "doc.pdf"}}))

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["status", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["kb_dir"] == str(kb_dir)
        assert data["counts"]["summaries"] == 1
        assert data["counts"]["raw"] == 1
        assert data["total_indexed"] == 1
        # summaries/doc.md is a compiled page → ISO-8601 timestamp present
        assert data["last_compile"] is not None
        assert "T" in data["last_compile"]
        assert data["last_lint"] is None

    def test_status_json_no_kb_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("okforge.cli._find_kb_dir", return_value=None),
        ):
            result = runner.invoke(cli, ["status", "--json"])
        assert result.exit_code == 1
        assert json.loads(result.output) == {"error": "no_kb"}


class TestInitJson:
    def test_init_json_is_noninteractive_and_reports(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
            with patch("okforge.cli.register_kb"):
                # No stdin provided: --json must never prompt.
                result = runner.invoke(cli, ["init", "-m", "gpt-4o-mini", "-l", "en", "--json"])
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert data["created"] is True
            assert data["kb_dir"] == str(Path(fs).resolve())
            assert data["model"] == "gpt-4o-mini"
            assert data["language"] == "en"
            assert (Path(fs) / ".okforge" / "config.yaml").exists()
            assert (Path(fs) / "wiki" / "index.md").exists()

    def test_init_json_already_initialised(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path(".okforge").mkdir()
            result = runner.invoke(cli, ["init", "--json"])
            data = json.loads(result.output)
            assert data["created"] is False


class TestOkfLint:
    def test_clean_kb_passes(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        (kb_dir / "wiki" / "log.md").write_text("# Operations Log\n\n")
        (kb_dir / "wiki" / "summaries" / "doc.md").write_text(
            '---\ntype: "Summary"\n---\n\n# Doc\n'
        )
        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["okf-lint", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["conformant"] is True
        assert data["issues"] == []

    def test_missing_type_flagged_and_exit_1(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        (kb_dir / "wiki" / "log.md").write_text("# Operations Log\n\n")
        (kb_dir / "wiki" / "sources").mkdir(exist_ok=True)
        (kb_dir / "wiki" / "sources" / "raw.md").write_text("# No frontmatter\n")
        (kb_dir / "wiki" / "concepts" / "c.md").write_text('---\ndescription: "x"\n---\n\nbody\n')
        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["okf-lint", "--json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert any("sources/raw.md" in i and "frontmatter" in i for i in data["issues"])
        assert any("concepts/c.md" in i and "type" in i for i in data["issues"])

    def test_missing_reserved_files_flagged(self, tmp_path):
        kb_dir = _setup_kb(tmp_path)
        (kb_dir / "wiki" / "index.md").unlink()
        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["okf-lint", "--json"])
        data = json.loads(result.output)
        assert any("index.md: missing" in i for i in data["issues"])
        assert any("log.md: missing" in i for i in data["issues"])


def test_okf_lint_ignores_dot_dirs(tmp_path):
    from okforge.okf import okf_check

    wiki = tmp_path / "wiki"
    (wiki / ".trash").mkdir(parents=True)
    (wiki / ".trash" / "2026-07-07.md").write_text("no frontmatter here")
    (wiki / "index.md").write_text("# Index\n")
    (wiki / "log.md").write_text("# Log\n")
    assert okf_check(wiki) == []
