"""MCP server exposing a single okforge knowledge base.

Bound to one KB at server-start time — same resolution every other CLI
command uses (cwd walk-up or ``--kb-dir``). Unlike the separate,
private "manager" deployment's own MCP layer (which picks between many
sibling KB directories), the engine has no multi-KB registry, so this
server is deliberately scoped to the one KB it's started against. Run
via ``okforge mcp``.

Read-only by design, matching the same principle the manager's MCP
layer already documents: mutating a KB (``add``, ``remove``, ``lint
--fix``, ...) stays a deliberate CLI action, never something an MCP
client can trigger.

No authentication. stdio transport is inherently local (this process's
own pipes — nothing listens on the network), which is what makes that
safe by default. If it's ever bridged to be reachable remotely, that
must only happen over a network already trusted end-to-end (SSH
tunnel, VPN) — never a directly-exposed port. Anyone who can reach it
gets full read access to the bound KB, including ``query`` (which
runs the configured LLM on the caller's behalf), with no login step.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from okforge.agent.query import run_query
from okforge.agent.tools import grep_wiki_files, read_topic_node, read_wiki_file
from okforge.config import load_config, state_dir


def build_mcp_server(kb_dir: Path, model: str) -> FastMCP:
    """Build a FastMCP server bound to *kb_dir*, using *model* for ``query``.

    Caller is responsible for resolving ``kb_dir`` and calling
    ``_setup_llm_key`` beforehand — this function only wires tools.
    """
    wiki_root = str(kb_dir / "wiki")

    mcp = FastMCP(
        "okforge",
        instructions=(
            f"Query the okforge knowledge base at {kb_dir}. Prefer "
            "grep_wiki for fast lexical fact lookups (names, dates, "
            "places) and read_wiki_page to pull a specific page found "
            "that way — both are instant, no LLM cost. Use query only "
            "when you need synthesis across sources: it runs a full "
            "retrieval-and-generation pass on an LLM and is slower. "
            "Answers from query cite source pages as (p. N) when "
            "available."
        ),
    )

    @mcp.tool()
    def status() -> dict:
        """Content stats for this knowledge base: documents, concepts,
        entities, and their names — same payload as `okforge list --json`."""
        from okforge.cli import collect_list_data

        return collect_list_data(kb_dir)

    @mcp.tool()
    def grep_wiki(pattern: str, ignore_case: bool = True, fixed_string: bool = False) -> str:
        """Fast lexical search over the wiki's markdown — no LLM, sub-second.
        Use this FIRST for fact lookups; only fall back to query() when you
        need synthesis. Case-insensitive by default. `pattern` is an
        extended regular expression unless fixed_string is True. Returns
        matches as `relative/path.md:LINE:text`."""
        return grep_wiki_files(
            pattern, wiki_root, ignore_case=ignore_case, fixed_string=fixed_string
        )

    @mcp.tool()
    def read_wiki_page(path: str) -> str:
        """Read one wiki page by its wiki-relative path (as returned by
        grep_wiki), e.g. 'summaries/doc.md' or 'entities/fort-marion.md'.
        Cheap and instant — prefer grep_wiki + this over query() for
        lookups you can answer by reading a page directly."""
        return read_wiki_file(path, wiki_root)

    # read_topic only makes sense (and is only correct) for a KB that
    # actually has the nested topic-tree layout enabled — registering it
    # unconditionally would offer a tool that's silently wrong for every
    # flat-concepts KB, so it's opt-in based on this KB's own config.
    config = load_config(state_dir(kb_dir) / "config.yaml")
    if config.get("topic_tree", False):

        @mcp.tool()
        def read_topic(rel: str = "") -> str:
            """Navigate the concept topic tree top-down: start at "" (root),
            pick a child topic from the result, call again with its path,
            until you reach the concept leaves you need (read those with
            read_wiki_page). Only available because this KB has topic_tree
            enabled."""
            return read_topic_node(rel, wiki_root)

    @mcp.tool()
    async def query(question: str) -> str:
        """Ask this knowledge base a question; returns a synthesized answer
        with (p. N) page citations when available. Expensive: a full
        retrieval + generation pass on an LLM — for simple fact lookups use
        grep_wiki + read_wiki_page instead."""
        question = question.strip()
        if not question:
            raise ValueError("empty question")
        return await run_query(question, kb_dir, model)

    return mcp
