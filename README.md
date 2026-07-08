# okforge

A local-first LLM knowledge-base engine. Point it at your documents;
it builds an interlinked wiki — per-document summaries, cross-document
concept and entity pages, extracted images, and real page citations
back to source. The wiki is plain Markdown with YAML frontmatter,
readable in Obsidian or any editor, and queryable from a CLI, a chat
REPL, or any MCP client.

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
okforge init          # scaffold raw/, wiki/, .openkb/  (--json for scripts)
okforge add paper.md  # ingest (pre-convert non-md/pdf inputs first)
okforge query "What does the paper conclude?"
okforge chat          # interactive REPL over the wiki
okforge list --json   # machine-readable state (also: status, okf-lint)
okforge describe "One line about this project."   # curated description
```

Non-Markdown, non-PDF inputs (docx, pptx, scans, photo catalogs, …)
need converting to Markdown first, by a tool that understands your
material — a page-aware OCR script, for example. A sibling
`<doc>.pages.json` page array is what enables real `(p. N)` citations
in the generated summaries.

The query agent reads curated pages first, then drills for detail with
a built-in `grep_wiki` lexical search (locate-then-read) rather than
re-embedding everything. `okf-lint` checks a wiki bundle's OKF
conformance.

Configuration lives in `.openkb/config.yaml` (model, language, entity
types, …) and `~/.config/openkb/global.yaml` (KB registry, default
KB). The LLM endpoint is configured litellm-style — any
OpenAI-compatible server works, including a local llama.cpp instance.

### Topic tree (experimental, per-KB opt-in)

For knowledge bases that outgrow a flat concept list: set
`topic_tree: true` in `.openkb/config.yaml`, then run `okforge
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

## Development

```bash
uv run --extra dev python -m pytest tests/   # test suite
uv run --extra dev ruff check openkb tests   # lint
uv run --extra dev ruff format openkb tests  # format
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
