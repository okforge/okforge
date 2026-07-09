"""Tests for the engine-native MCP server (okforge.agent.mcp_server).

Bound to a single KB (unlike the separate manager deployment's own
multi-project MCP layer) — these tests exercise the tool set through
FastMCP's own call_tool/list_tools, not just the underlying tools.py
functions directly, so a regression in the MCP wiring itself (wrong
tool name, dropped guard, wrong default) would be caught here even if
tools.py's own unit tests still pass.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from okforge.agent.mcp_server import build_mcp_server
from okforge.cli import cli


def _write_config(kb_dir, *, topic_tree: bool = False) -> None:
    extra = "\ntopic_tree: true\n" if topic_tree else "\n"
    (kb_dir / ".okforge" / "config.yaml").write_text(f"model: openai/test-model{extra}")


def _tool_text(result) -> str:
    """Unwrap a FastMCP call_tool() return into the plain text a tool
    actually returned. Shape varies by the tool's return annotation: a
    str-returning tool gets wrapped as (content_blocks, structured_result);
    a dict-returning tool (e.g. status) comes back as a bare content list."""
    content = result[0] if isinstance(result, tuple) else result
    return content[0].text


@pytest.fixture
def wired_kb(kb_dir):
    """kb_dir (from conftest) plus enough wiki content for grep/read tests."""
    (kb_dir / "wiki" / "concepts" / "widgets.md").write_text(
        "---\nname: widgets\n---\nWidgets are small mechanical parts. (p. 3)\n"
    )
    (kb_dir / "wiki" / "index.md").write_text("# Index\n")
    _write_config(kb_dir)
    return kb_dir


class TestToolset:
    @pytest.mark.asyncio
    async def test_default_toolset_excludes_read_topic(self, wired_kb):
        server = build_mcp_server(wired_kb, "openai/test-model")
        names = {t.name for t in await server.list_tools()}
        assert names == {"query", "grep_wiki", "read_wiki_page", "status"}

    @pytest.mark.asyncio
    async def test_topic_tree_enabled_adds_read_topic(self, kb_dir):
        _write_config(kb_dir, topic_tree=True)
        server = build_mcp_server(kb_dir, "openai/test-model")
        names = {t.name for t in await server.list_tools()}
        assert "read_topic" in names

    @pytest.mark.asyncio
    async def test_no_write_tools_ever_exposed(self, wired_kb):
        """Read-only by design — matches the manager MCP layer's own stated
        principle; a public package is not the place to relax it."""
        server = build_mcp_server(wired_kb, "openai/test-model")
        names = {t.name for t in await server.list_tools()}
        assert "write_kb_file" not in names
        assert "write_wiki_file" not in names


class TestGrepWiki:
    @pytest.mark.asyncio
    async def test_finds_match(self, wired_kb):
        server = build_mcp_server(wired_kb, "openai/test-model")
        result = await server.call_tool("grep_wiki", {"pattern": "Widgets"})
        text = _tool_text(result)
        assert "concepts/widgets.md" in text
        assert "Widgets are small mechanical parts" in text

    @pytest.mark.asyncio
    async def test_no_match_returns_message_not_error(self, wired_kb):
        server = build_mcp_server(wired_kb, "openai/test-model")
        result = await server.call_tool("grep_wiki", {"pattern": "nonexistent-term-xyz"})
        assert "No matches" in _tool_text(result)


class TestReadWikiPage:
    @pytest.mark.asyncio
    async def test_reads_expected_content(self, wired_kb):
        server = build_mcp_server(wired_kb, "openai/test-model")
        result = await server.call_tool("read_wiki_page", {"path": "concepts/widgets.md"})
        assert "Widgets are small mechanical parts" in _tool_text(result)

    @pytest.mark.asyncio
    async def test_path_escape_is_denied(self, wired_kb):
        """The guard lives in tools.py's read_wiki_file, but this confirms
        it still holds when called through the MCP tool-call layer, not
        just when the underlying function is called directly."""
        server = build_mcp_server(wired_kb, "openai/test-model")
        result = await server.call_tool("read_wiki_page", {"path": "../../../etc/passwd"})
        assert "Access denied" in _tool_text(result)

    @pytest.mark.asyncio
    async def test_missing_file_returns_message_not_error(self, wired_kb):
        server = build_mcp_server(wired_kb, "openai/test-model")
        result = await server.call_tool("read_wiki_page", {"path": "concepts/nope.md"})
        assert "File not found" in _tool_text(result)


class TestStatus:
    @pytest.mark.asyncio
    async def test_matches_collect_list_data(self, wired_kb):
        from okforge.cli import collect_list_data

        server = build_mcp_server(wired_kb, "openai/test-model")
        result = await server.call_tool("status", {})
        payload = json.loads(_tool_text(result))
        assert payload == collect_list_data(wired_kb)
        assert "widgets" in payload["concepts"]


class TestQueryTool:
    @pytest.mark.asyncio
    async def test_returns_agent_final_output(self, wired_kb):
        mock_result = MagicMock()
        mock_result.final_output = "Widgets are small mechanical parts (p. 3)."
        with patch("okforge.agent.query.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            server = build_mcp_server(wired_kb, "openai/test-model")
            result = await server.call_tool("query", {"question": "What are widgets?"})
        assert _tool_text(result) == "Widgets are small mechanical parts (p. 3)."

    @pytest.mark.asyncio
    async def test_empty_question_errors(self, wired_kb):
        server = build_mcp_server(wired_kb, "openai/test-model")
        with pytest.raises(Exception, match="empty question"):
            await server.call_tool("query", {"question": "   "})


class TestMcpCommandStdoutHygiene:
    """The stdio transport requires stdout to carry *only* newline-delimited
    JSON-RPC once the server starts — any stray print (e.g. the missing-API-
    key warning _setup_llm_key emits via click.echo) corrupts the stream for
    every real MCP client. A live subprocess protocol test caught this once
    already; this pins the fix so it can't silently regress."""

    def test_setup_warnings_go_to_stderr_not_stdout(self, wired_kb, monkeypatch):
        # LLM_API_KEY plus every provider-specific var _setup_llm_key checks
        # (it self-propagates LLM_API_KEY into these on a successful run
        # elsewhere in the suite, via a real os.environ write monkeypatch
        # can't auto-revert since it wasn't the one who set it) — clear all
        # of them so this test doesn't depend on suite run order.
        from okforge.cli import _KNOWN_PROVIDER_KEYS

        for var in ("LLM_API_KEY", *_KNOWN_PROVIDER_KEYS):
            monkeypatch.delenv(var, raising=False)
        fake_server = MagicMock()
        with patch(
            "okforge.agent.mcp_server.build_mcp_server", return_value=fake_server
        ) as mock_build:
            runner = CliRunner()
            result = runner.invoke(cli, ["--kb-dir", str(wired_kb), "mcp"])
        mock_build.assert_called_once()
        fake_server.run.assert_called_once_with(transport="stdio")
        assert result.stdout == ""
        assert "No LLM API key found" in result.stderr

    def test_no_kb_found_reports_to_stderr_and_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["--kb-dir", str(tmp_path), "mcp"])
        assert result.exit_code != 0
        assert result.stdout == ""
        assert "No knowledge base found" in result.stderr


class TestMcpCommandTransport:
    """--transport http is the Streamable HTTP path; stdio stays the
    default so existing `claude mcp add --transport stdio ...` setups
    don't change behavior."""

    def test_transport_http_runs_streamable_http(self, wired_kb, monkeypatch):
        from okforge.cli import _KNOWN_PROVIDER_KEYS

        for var in ("LLM_API_KEY", *_KNOWN_PROVIDER_KEYS):
            monkeypatch.delenv(var, raising=False)
        fake_server = MagicMock()
        with patch(
            "okforge.agent.mcp_server.build_mcp_server", return_value=fake_server
        ) as mock_build:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--kb-dir",
                    str(wired_kb),
                    "mcp",
                    "--transport",
                    "http",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9001",
                ],
            )
        assert result.exit_code == 0
        mock_build.assert_called_once_with(wired_kb, "openai/test-model", host="0.0.0.0", port=9001)
        fake_server.run.assert_called_once_with(transport="streamable-http")

    def test_transport_stdio_still_default(self, wired_kb, monkeypatch):
        from okforge.cli import _KNOWN_PROVIDER_KEYS

        for var in ("LLM_API_KEY", *_KNOWN_PROVIDER_KEYS):
            monkeypatch.delenv(var, raising=False)
        fake_server = MagicMock()
        with patch(
            "okforge.agent.mcp_server.build_mcp_server", return_value=fake_server
        ) as mock_build:
            runner = CliRunner()
            result = runner.invoke(cli, ["--kb-dir", str(wired_kb), "mcp"])
        assert result.exit_code == 0
        mock_build.assert_called_once_with(
            wired_kb, "openai/test-model", host="127.0.0.1", port=8000
        )
        fake_server.run.assert_called_once_with(transport="stdio")


class TestBuildMcpServerHostPort:
    def test_host_port_passed_through_to_fastmcp_settings(self, wired_kb):
        server = build_mcp_server(wired_kb, "openai/test-model", host="0.0.0.0", port=9001)
        assert server.settings.host == "0.0.0.0"
        assert server.settings.port == 9001

    def test_defaults_bind_loopback(self, wired_kb):
        server = build_mcp_server(wired_kb, "openai/test-model")
        assert server.settings.host == "127.0.0.1"
        assert server.settings.port == 8000
