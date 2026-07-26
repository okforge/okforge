"""Tests for the `add` CLI command (Task 10)."""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from okforge.cli import SUPPORTED_EXTENSIONS, _find_kb_dir, cli


class TestSupportedExtensions:
    def test_pdf_supported(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_md_supported(self):
        assert ".md" in SUPPORTED_EXTENSIONS

    def test_txt_supported(self):
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_unknown_not_supported(self):
        assert ".xyz" not in SUPPORTED_EXTENSIONS

    def test_markitdown_formats_dropped(self):
        # Pre-conversion owns these now; the MarkItDown path was stripped.
        for ext in (".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".csv"):
            assert ext not in SUPPORTED_EXTENSIONS


class TestFindKbDir:
    def test_finds_openkb_dir(self, tmp_path, monkeypatch):
        (tmp_path / ".okforge").mkdir()
        monkeypatch.chdir(tmp_path)
        result = _find_kb_dir()
        assert result is not None

    def test_returns_none_if_no_openkb(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("okforge.cli.load_global_config", return_value={}):
            result = _find_kb_dir()
            assert result is None


class TestAddCommand:
    def _setup_kb(self, tmp_path):
        """Create a minimal KB structure."""
        (tmp_path / "raw").mkdir()
        (tmp_path / "wiki" / "sources" / "images").mkdir(parents=True)
        (tmp_path / "wiki" / "summaries").mkdir(parents=True)
        (tmp_path / "wiki" / "concepts").mkdir(parents=True)
        (tmp_path / "wiki" / "reports").mkdir(parents=True)
        openkb_dir = tmp_path / ".okforge"
        openkb_dir.mkdir()
        (openkb_dir / "config.yaml").write_text("model: gpt-4o-mini\n")
        (openkb_dir / "hashes.json").write_text(json.dumps({}))
        return tmp_path

    def test_add_missing_init(self, tmp_path):
        runner = CliRunner()
        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("okforge.cli._find_kb_dir", return_value=None),
        ):
            result = runner.invoke(cli, ["add", "somefile.pdf"])
            assert "No knowledge base found" in result.output

    def test_add_single_file_calls_helper(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "test.md"
        doc.write_text("# Hello")

        runner = CliRunner()
        with (
            patch("okforge.cli.add_single_file") as mock_add,
            patch("okforge.cli._find_kb_dir", return_value=kb_dir),
        ):
            runner.invoke(cli, ["add", str(doc)])
            mock_add.assert_called_once_with(doc, kb_dir)

    def test_add_single_file_compile_failure_rolls_back_converted_artifacts(self, tmp_path):
        from okforge.cli import add_single_file
        from okforge.state import HashRegistry

        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "notes.md"
        doc.write_text("# Notes\n\nBody", encoding="utf-8")

        with (
            patch("okforge.agent.compiler.compile_short_doc", side_effect=RuntimeError("boom")),
            patch("okforge.cli.time.sleep"),
            patch("okforge.cli._setup_llm_key"),
        ):
            outcome = add_single_file(doc, kb_dir)

        assert outcome == "failed"
        assert not (kb_dir / "raw" / "notes.md").exists()
        assert not (kb_dir / "wiki" / "sources" / "notes.md").exists()
        assert HashRegistry(kb_dir / ".okforge" / "hashes.json").all_entries() == {}

    def test_add_rejects_pdf_over_threshold(self, tmp_path):
        """A raw long PDF is rejected during conversion, with a clear message
        pointing at pre-chunking — okforge no longer auto-indexes these."""
        from unittest.mock import MagicMock

        from okforge.cli import add_single_file

        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "long.pdf"
        doc.write_bytes(b"%PDF-1.4 fake long content")

        fake_doc = MagicMock()
        fake_doc.page_count = 200
        fake_doc.__enter__ = MagicMock(return_value=fake_doc)
        fake_doc.__exit__ = MagicMock(return_value=False)

        with (
            patch("okforge.converter.pymupdf.open", return_value=fake_doc),
            patch("okforge.cli._setup_llm_key"),
        ):
            outcome = add_single_file(doc, kb_dir)

        assert outcome == "failed"
        assert not (kb_dir / "raw" / "long.pdf").exists()

    def test_add_directory_calls_helper_for_each_file(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text("# A")
        (docs_dir / "b.txt").write_text("B content")
        (docs_dir / "ignore.xyz").write_text("skip me")

        runner = CliRunner()
        with (
            patch("okforge.cli.add_single_file") as mock_add,
            patch("okforge.cli._find_kb_dir", return_value=kb_dir),
        ):
            runner.invoke(cli, ["add", str(docs_dir)])
            # Should be called for .md and .txt but not .xyz
            assert mock_add.call_count == 2
            called_names = {call.args[0].name for call in mock_add.call_args_list}
            assert "a.md" in called_names
            assert "b.txt" in called_names
            assert "ignore.xyz" not in called_names

    def test_add_unsupported_extension(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "file.xyz"
        doc.write_text("content")

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["add", str(doc)])
            assert "Unsupported file type" in result.output

    def test_add_nonexistent_path(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)

        runner = CliRunner()
        with patch("okforge.cli._find_kb_dir", return_value=kb_dir):
            result = runner.invoke(cli, ["add", str(tmp_path / "nonexistent.pdf")])
            assert "does not exist" in result.output

    def test_add_skipped_file(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "test.md"
        doc.write_text("# Hello")

        from okforge.converter import ConvertResult

        mock_result = ConvertResult(skipped=True)

        runner = CliRunner()
        with (
            patch("okforge.cli._find_kb_dir", return_value=kb_dir),
            patch("okforge.cli.convert_document", return_value=mock_result),
            patch("okforge.cli.asyncio.run") as mock_arun,
        ):
            result = runner.invoke(cli, ["add", str(doc)])
            assert "SKIP" in result.output
            mock_arun.assert_not_called()

    def test_add_short_doc_runs_compiler(self, tmp_path):
        kb_dir = self._setup_kb(tmp_path)
        doc = tmp_path / "test.md"
        doc.write_text("# Hello")

        source_path = kb_dir / "wiki" / "sources" / "test.md"
        source_path.write_text("# Hello converted")

        from okforge.converter import ConvertResult

        mock_result = ConvertResult(
            raw_path=kb_dir / "raw" / "test.md",
            source_path=source_path,
            file_hash="deadbeef00" * 8,
            doc_name="test",
        )

        # An edited doc arrives with a new content hash; the stale entry
        # for the same doc_name must be replaced, leaving exactly ONE entry.
        from okforge.state import HashRegistry

        HashRegistry(kb_dir / ".okforge" / "hashes.json").add(
            "stale-old-hash", {"name": "test.md", "doc_name": "test", "type": "md"}
        )

        compile_calls = []

        async def compile_noop(*args, **kwargs):
            compile_calls.append((args, kwargs))

        runner = CliRunner()
        with (
            patch("okforge.cli._find_kb_dir", return_value=kb_dir),
            patch("okforge.cli.convert_document", return_value=mock_result),
            patch("okforge.agent.compiler.compile_short_doc", new=compile_noop),
        ):
            result = runner.invoke(cli, ["add", str(doc)])
            assert len(compile_calls) == 1
            assert "OK" in result.output

        import json as json_mod

        hashes = json_mod.loads((kb_dir / ".okforge" / "hashes.json").read_text(encoding="utf-8"))
        meta = hashes[mock_result.file_hash]
        assert meta["doc_name"] == "test"
        assert meta["raw_path"] == "raw/test.md"
        assert meta["source_path"] == "wiki/sources/test.md"
        assert "path" in meta
        assert "stale-old-hash" not in hashes

    def test_add_oldest_legacy_entry_converges_to_single_entry(self, tmp_path):
        """Editing a pre-doc_name-era document must not fork the registry.

        convert_document backfills doc_name/path onto the legacy entry on
        disk; the cli's registry instance must see that backfill (i.e. be
        constructed after convert), otherwise its full-file rewrite clobbers
        the backfill and leaves two entries for one document.
        """
        import json as json_mod

        from okforge.state import HashRegistry

        kb_dir = self._setup_kb(tmp_path)
        # oldest-generation entry: name only, no doc_name, no path
        HashRegistry(kb_dir / ".okforge" / "hashes.json").add(
            "old-hash", {"name": "notes.md", "type": "md"}
        )
        doc = tmp_path / "notes.md"
        doc.write_text("# Notes, edited")  # new content hash != "old-hash"

        # Compilation mocked out, but convert_document REAL so
        # the legacy backfill actually happens on disk mid-pipeline.
        def close_coro(coro):
            if hasattr(coro, "close"):
                coro.close()

        runner = CliRunner()
        with (
            patch("okforge.cli._find_kb_dir", return_value=kb_dir),
            patch("okforge.cli.asyncio.run", side_effect=close_coro),
        ):
            result = runner.invoke(cli, ["add", str(doc)])
            assert "OK" in result.output

        hashes = json_mod.loads((kb_dir / ".okforge" / "hashes.json").read_text(encoding="utf-8"))
        assert "old-hash" not in hashes  # stale entry replaced…
        new_entries = [m for m in hashes.values() if m.get("doc_name") == "notes"]
        assert len(new_entries) == 1  # …exactly one entry survives
        assert new_entries[0]["path"]  # with path identity persisted


class TestAddExitCode:
    """`add` must report failure through its exit code.

    The exit code is the only failure signal a non-interactive caller —
    the webui job runner, a shell script, CI — receives. Returning 0 after
    "[ERROR] Compilation failed" made a wholly failed ingest indistinguishable
    from a clean one (okforge/okforge#2): four add jobs and a describe job all
    reported done against a knowledge base with zero content.
    """

    def _kb(self, tmp_path):
        (tmp_path / ".okforge").mkdir()
        (tmp_path / ".okforge" / "config.yaml").write_text("model: gpt-4o-mini\n")
        (tmp_path / ".okforge" / "hashes.json").write_text("{}")
        (tmp_path / "raw").mkdir()
        (tmp_path / "wiki" / "summaries").mkdir(parents=True)
        (tmp_path / "wiki" / "sources").mkdir(parents=True)
        (tmp_path / "wiki" / "log.md").write_text("")
        return tmp_path

    def _failing_compile(self):
        async def fail(*args, **kwargs):
            raise RuntimeError("LLM 503")

        return fail

    def test_single_file_failure_exits_1(self, tmp_path):
        kb = self._kb(tmp_path)
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n\nBody.\n")
        runner = CliRunner()
        with (
            patch("okforge.cli._find_kb_dir", return_value=kb),
            patch("okforge.agent.compiler.compile_short_doc", new=self._failing_compile()),
            patch("okforge.cli.time.sleep"),
        ):
            result = runner.invoke(cli, ["add", str(doc)])
        assert result.exit_code == 1, result.output
        assert "[ERROR] Compilation failed" in result.output

    def test_directory_reports_which_files_failed(self, tmp_path):
        """A bad document must not strand the rest of the batch, but the
        run as a whole still has to fail."""
        kb = self._kb(tmp_path)
        src = tmp_path / "docs"
        src.mkdir()
        (src / "a.md").write_text("# A\n\nAlpha.\n")
        (src / "b.md").write_text("# B\n\nBeta.\n")
        runner = CliRunner()
        with (
            patch("okforge.cli._find_kb_dir", return_value=kb),
            patch("okforge.agent.compiler.compile_short_doc", new=self._failing_compile()),
            patch("okforge.cli.time.sleep"),
        ):
            result = runner.invoke(cli, ["add", str(src)])
        assert result.exit_code == 1, result.output
        # Both were attempted, and both are named in the summary.
        assert "2 of 2 file(s) failed" in result.output
        assert "a.md" in result.output and "b.md" in result.output

    def test_success_still_exits_0(self, tmp_path):
        """Guard the other direction: the fix must not make a good ingest
        look failed."""
        kb = self._kb(tmp_path)
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n\nBody.\n")

        def close_coro(coro):
            if hasattr(coro, "close"):
                coro.close()

        runner = CliRunner()
        with (
            patch("okforge.cli._find_kb_dir", return_value=kb),
            patch("okforge.cli.asyncio.run", side_effect=close_coro),
        ):
            result = runner.invoke(cli, ["add", str(doc)])
        assert result.exit_code == 0, result.output

    def test_skip_exits_0(self, tmp_path):
        """A dedup skip is not a failure."""
        kb = self._kb(tmp_path)
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n\nBody.\n")

        def close_coro(coro):
            if hasattr(coro, "close"):
                coro.close()

        runner = CliRunner()
        with (
            patch("okforge.cli._find_kb_dir", return_value=kb),
            patch("okforge.cli.asyncio.run", side_effect=close_coro),
        ):
            runner.invoke(cli, ["add", str(doc)])
            result = runner.invoke(cli, ["add", str(doc)])
        assert result.exit_code == 0, result.output
        assert "SKIP" in result.output
