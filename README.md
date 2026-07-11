# okforge

A local-first LLM knowledge-base engine. Point it at your documents;
it builds an interlinked wiki — per-document summaries, cross-document
concept and entity pages, extracted images, and real page citations
back to source. The wiki is plain Markdown with YAML frontmatter,
readable in Obsidian or any editor, and queryable from a CLI, a chat
REPL, any MCP client, or the companion
[okforge-webui](https://github.com/okforge/okforge-webui) web app.

## Why

Most retrieval setups hand an LLM a pile of raw chunks and hope. okforge
instead compiles your sources into curated pages *ahead of time* —
concepts and entities that already synthesize what's spread across
many documents, each claim traceable back to a `(p. N)` citation in
the original source. That matters most for models with limited
context, including small models running entirely on your own hardware:
they don't have to reconstruct an answer from scratch every query, and
what they do say is checkable against a specific page, not just
plausible-sounding.

The output follows the [Open Knowledge Format (OKF)](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) —
typed frontmatter, relative links, a predictable directory layout — so
the wiki a KB produces is portable, not locked to okforge itself.

## Install

```bash
pip install git+https://github.com/okforge/okforge@main
```

## Quick start

```bash
mkdir my-kb && cd my-kb
okforge init          # scaffold raw/, wiki/, .okforge/  (--json for scripts)
okforge add paper.md  # ingest (pre-convert non-md/pdf inputs first)
okforge query "What does the paper conclude?"
okforge chat          # interactive REPL over the wiki
okforge list --json   # machine-readable state (also: status, okf-lint)
okforge describe "One line about this project."   # curated description
```

The query agent reads curated pages first, then drills for detail with
a built-in `grep_wiki` lexical search (locate-then-read) rather than
re-embedding everything. `okf-lint` checks a wiki bundle's OKF
conformance.

Configuration lives in `.okforge/config.yaml` (model, language, entity
types, …) and `~/.config/okforge/global.yaml` (KB registry, default
KB). The LLM endpoint is configured litellm-style — any
OpenAI-compatible server works, including a local llama.cpp instance.
Hosted providers work through the same model string: e.g.
`okforge init --model openrouter/qwen/qwen3.6-27b` with
`LLM_API_KEY=sk-or-...` in the KB's `.env` runs the whole pipeline
against [OpenRouter](https://openrouter.ai/), no other changes.

One tip for thinking-capable models (the Qwen3 family in particular):
left alone they spend a hidden reasoning pass on **every** pipeline
call, and each serving stack spells "don't reason" differently. Put the
dialect your endpoint understands in the KB's `config.yaml` — for
llama.cpp/vLLM:

```yaml
llm_extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

and for OpenRouter:

```yaml
llm_extra_body:
  reasoning:
    enabled: false
```

The wrong (ignored) dialect fails silently — the wiki still builds, you
just pay reasoning-token cost and latency on every call. Measured on
`qwen3.6-27b` via OpenRouter: a trivial completion drops from 199
tokens to 2 with the block in place.

### Ingesting scans and non-text documents

`okforge add` accepts Markdown, plain text, and PDF directly. Anything
else — docx, pptx, scanned pages, photo catalogs — needs converting to
Markdown first, by a tool that knows your material. For scanned pages
specifically, [**okforge-vision-ocr**](https://github.com/okforge/okforge-vision-ocr)
(`pip install okforge-vision-ocr`) is built for exactly this: one
vision-LLM call per page produces both a clean Markdown transcription
and extracted photos/figures, plus a sibling `<doc>.pages.json` page
array that `okforge add` reads directly for real `(p. N)` citations in
the compiled summaries — no separate wiring needed.

A PDF at/above `pageindex_threshold` pages (default 20) is rejected the
same way — okforge doesn't auto-chunk long documents, so pre-chunk it
into smaller page ranges first, the same as any other large source.

```bash
okforge-vision-ocr scanned.pdf raw/book.md   # OCR + photo extraction
okforge add raw/book.md                      # ingest, with page citations
```

It works against any OpenAI-compatible vision-language model (tuned
against a locally-hosted Qwen3.6-27B-MTP, but not tied to it).

### MCP server

`okforge mcp` starts an MCP server over the KB it's run against — same
resolution as every other command (cwd walk-up, or `--kb-dir`). Tools:
`query`, `grep_wiki`, `read_wiki_page`, `status`, and `read_topic` when
the KB has `topic_tree` enabled. Read-only — ingest stays a deliberate
CLI action.

Two transports, picked with `--transport`:

```bash
# stdio (default) — the client spawns this process itself
claude mcp add --transport stdio okforge -- okforge --kb-dir /path/to/kb mcp

# Streamable HTTP — this process listens on a socket for clients to connect to
okforge --kb-dir /path/to/kb mcp --transport http --port 8000
claude mcp add --transport http okforge http://127.0.0.1:8000/mcp
```

Works the same way with any MCP client that supports stdio or
Streamable HTTP.

**No authentication, on either transport.** stdio is safe by default
(it's just this process's own pipes, nothing listens on the network).
`--transport http` binds `127.0.0.1` by default for the same reason —
if you change `--host` or put it behind a reverse proxy, only do so
over a network you already trust end-to-end (SSH tunnel, VPN, an
authenticating proxy) — never a directly-exposed port. Anyone who can
reach it gets full read access to the KB, including `query` (which
runs your configured LLM on your behalf), with no login step.

**Testing the http transport directly** (e.g. with `curl`, without a
real MCP client) means implementing the Streamable HTTP session
handshake yourself — a real client (`claude mcp add --transport http
...`) does all of this for you automatically:

- `Accept` must include both `application/json` and `text/event-stream`
  — either alone gets `406 Not Acceptable`.
- Call `initialize` first and capture the `Mcp-Session-Id` response
  header; send it back as a header on every subsequent request. There's
  no session without it.
- Responses are SSE (`event: message` / `data: {...}`) — strip the
  `data: ` prefix and parse the rest as JSON.
- `query`'s argument key is `question`, not `query`.

### Topic tree (experimental, per-KB opt-in)

For knowledge bases that outgrow a flat concept list: set
`topic_tree: true` in `.okforge/config.yaml`, then run `okforge
reindex`. Existing concepts cluster into named `concepts/<topic>/`
directories, each with a `_topic.md` summary node; later ingests place
new concepts by tree descent, and queries gain a `read_topic`
navigation tool for browsing top-down instead of scanning a flat list.

## Wiki layout

```
wiki/
  index.md              # document + concept index
  summaries/<doc>.md    # per-document summary (page citations when available)
  concepts/<name>.md    # cross-document concept pages
  entities/<name>.md    # named people/places/organizations/works
  sources/<doc>.md      # ingested source text
  sources/<doc>.json    # per-page text + images (when page-aware)
  sources/images/<doc>/ # extracted images
  log.md                # append-only ingest log
```

## Roadmap

A web UI already exists as a separate companion app:
[okforge-webui](https://github.com/okforge/okforge-webui) drives the
whole scan-to-wiki pipeline from a browser — PDF inbox, OCR pilots,
chunked ingest jobs with live progress, wiki browsing and query, and
static-site publishing — over the same one-directory-per-KB layout
this README describes (it discovers KBs by scanning a `kbs/` root, so
CLI-created KBs placed there just appear).

Planned, not yet built:

- **Multiple LLM endpoints** — a KB config today points `model:` at one
  litellm-style endpoint; planned support for registering several
  (different hosts, different models) and choosing between them per
  call rather than being locked to one for the whole KB.
- **Parallel processing** — ingest and query are fully serial today
  (one `add` at a time, one generation call at a time); planned
  concurrency for page-level work and multi-document ingest, bounded
  by each endpoint's slot budget so throughput scales with the
  hardware actually available.

## Development

```bash
uv run --extra dev python -m pytest tests/   # test suite
uv run --extra dev ruff check okforge tests  # lint
uv run --extra dev ruff format okforge tests # format
```

## Origins

okforge began as a hard fork of [VectifyAI/OpenKB](https://github.com/VectifyAI/OpenKB),
in the spirit of Karpathy's LLM-wiki idea, and has since diverged
deliberately rather than tracking it — local-only by default, its own
document-conversion boundary, and OKF conformance as a first-class
goal rather than an incidental format.

## License

Apache-2.0. Portions originate from the upstream OpenKB project
(copyright the original authors); okforge-specific changes are
maintained at [okforge/okforge](https://github.com/okforge/okforge).
