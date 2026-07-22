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

No authentication, on either transport. stdio is inherently local
(this process's own pipes — nothing listens on the network), which is
what makes that safe by default. The ``streamable-http`` transport
does listen on a socket — it defaults to 127.0.0.1 so it's no more
exposed than stdio out of the box, but if you point ``--host`` at
anything other than loopback, or put it behind a reverse proxy, that
must only happen over a network already trusted end-to-end (SSH
tunnel, VPN, an authenticating proxy in front of it) — never a
directly-exposed port. Anyone who can reach it gets full read access
to the bound KB, including ``query`` (which runs the configured LLM on
the caller's behalf), with no login step.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from okforge.agent.query import run_query
from okforge.agent.tools import grep_wiki_files, read_topic_node, read_wiki_file
from okforge.config import load_config, state_dir


def build_mcp_server(
    kb_dir: Path,
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP:
    """Build a FastMCP server bound to *kb_dir*, using *model* for ``query``.

    Caller is responsible for resolving ``kb_dir`` and calling
    ``_setup_llm_key`` beforehand — this function only wires tools.

    *host*/*port* only take effect for the ``streamable-http`` transport
    (FastMCP reads them off the server at construction time, not at
    ``run()``); the ``stdio`` transport ignores them.
    """
    wiki_root = str(kb_dir / "wiki")

    mcp = FastMCP(
        "okforge",
        # This block is the ONE place the grep-vs-query routing rule is
        # stated. Repeating it in the tool docstrings gave weak client
        # models the same judgment call from four angles to re-litigate;
        # observed live, one spent its whole thinking budget re-deciding
        # and never called anything. Keep the rule here, phrased as a
        # rule; keep docstrings descriptive.
        instructions=(
            f"Query the okforge knowledge base at {kb_dir} — a "
            "citation-backed wiki compiled from source documents.\n\n"
            "Choosing a tool: default to grep_wiki, then read_wiki_page on "
            "the hits worth citing. Use query only when the question calls "
            "for a summary, comparison, or explanation spanning multiple "
            "documents. If this KB offers read_topic, it walks the concept "
            "topic tree top-down when browsing beats searching.\n\n"
            "Wiki pages cite their source pages as (p. N) where the "
            "documents were ingested page by page. Carry those citations "
            "into your own answer, next to the claim each one supports — "
            "tracing a statement back to its source page is the point of "
            "this knowledge base, and an uncited answer throws that away. "
            "In video-transcript knowledge bases page N is the N-th "
            "5-minute block of the video, so give the timestamp too: "
            "(p. 14) = minutes 65-70."
        ),
        host=host,
        port=port,
    )

    @mcp.tool()
    def status() -> dict:
        """Content stats for this knowledge base: documents, concepts,
        entities, and their names — same payload as `okforge list --json`."""
        from okforge.cli import collect_list_data

        return collect_list_data(kb_dir)

    @mcp.tool()
    def grep_wiki(pattern: str, ignore_case: bool = True, fixed_string: bool = False) -> str:
        """Lexical search over the wiki's markdown: no LLM, sub-second.
        Finds names, dates, places, part numbers and the lines they appear
        on. Case-insensitive by default. `pattern` is an extended regular
        expression unless fixed_string is True. Returns matches as
        `relative/path.md:LINE:text`. Read a whole hit with
        read_wiki_page(path)."""
        return grep_wiki_files(
            pattern, wiki_root, ignore_case=ignore_case, fixed_string=fixed_string
        )

    @mcp.tool()
    def read_wiki_page(path: str) -> str:
        """Read one wiki page by its wiki-relative path (as returned by
        grep_wiki), e.g. 'summaries/doc.md' or 'entities/fort-marion.md'."""
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
        """Ask this knowledge base a question; returns a written answer
        citing source pages as (p. N) where the documents were ingested
        page by page. Runs a full retrieval and generation pass on an LLM."""
        question = question.strip()
        if not question:
            raise ValueError("empty question")
        return await run_query(question, kb_dir, model)

    return mcp
