# Getting Started with okforge and okforge-vision-ocr

A step-by-step guide for beginners to set up a Python virtual environment, install okforge's tools, and turn a scanned PDF into a queryable wiki.

**Works on Windows, macOS, and Linux.**

Two separate tools are involved, each doing one job:

- **[okforge-vision-ocr](https://github.com/okforge/okforge-vision-ocr)** — turns a scanned PDF into Markdown text plus extracted photos, one page at a time, using a vision-language model.
- **[okforge](https://github.com/okforge/okforge)** — compiles Markdown documents into an interlinked wiki: per-document summaries, cross-document concepts, named entities, all cross-linked and citeable back to the source page.

You'll use the first only for scanned/image PDFs. Plain text, Markdown, and PDFs with a real text layer can skip straight to okforge.

---

## Step 0: Create a Project Directory

Create a dedicated project directory with **no spaces in the name**. Paths with spaces can cause unexpected issues with some tools.

### Windows (Command Prompt)

```cmd
mkdir C:\Users\YOURNAME\mykb
cd C:\Users\YOURNAME\mykb
```

### macOS / Linux

```bash
mkdir ~/mykb
cd ~/mykb
```

Check your Python version — both tools need **Python 3.10 or newer**:

```bash
python3 --version
```

Inside your project directory, create a `source` subdirectory to hold your original PDFs and the output from okforge-vision-ocr:

```bash
mkdir source
```

Copy your PDFs into `source/`:

```bash
# Windows
copy "C:\Users\YOURNAME\Documents\My Folder\My Document.pdf" source\mydoc.pdf

# macOS / Linux
cp "~/Documents/My Folder/My Document.pdf" source/mydoc.pdf
```

> **Tip:** Always use a `source/` directory with no spaces. It keeps your originals organized and avoids path problems.

---

## Step 1: Create a Python Virtual Environment

A virtual environment (venv) keeps your project's Python packages isolated from the rest of your system.

### Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verify it worked** — your prompt should now show `(.venv)`:

```bash
python --version
```

---

## Step 2: Install okforge and okforge-vision-ocr

With the venv activated, install both packages from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install okforge okforge-vision-ocr
```

**Verify the installation:**

```bash
okforge --help
okforge-vision-ocr --help
```

Both commands should display usage information without errors.

---

## Step 3: Configure Your LLM Endpoint

Both tools need access to a **vision-language model** (VLM) — required for okforge-vision-ocr (it reads scanned pages), and useful for okforge too if you want it to describe extracted images. You have two options.

Both tools read the **same** `.env` file, so one setup covers both — but note they use *different* variables to pick which model to call: okforge's model is set once at `okforge init --model ...` and saved into its config; okforge-vision-ocr reads a separate `OKFORGE_VISION_MODEL` variable on every run. Keep both in sync when you change providers.

### Option A: Local Model (Recommended if you have GPU hardware)

If you have a local model running via llama.cpp with an OpenAI-compatible server, create a `.env` file in your project directory:

```
OPENAI_API_BASE=http://localhost:8080/v1
LLM_API_KEY=no-key
```

You'll pass the model name directly to each tool (`--model` for okforge, `OKFORGE_VISION_MODEL` for okforge-vision-ocr) in the steps below.

### Option B: OpenRouter (Cloud — No GPU Needed)

If you don't have a local model, [OpenRouter](https://openrouter.ai/) gives you an OpenAI-compatible endpoint in front of many hosted models.

1. **Sign up** at [openrouter.ai](https://openrouter.ai/) and create a free account.
2. **Get an API key** from [openrouter.ai/keys](https://openrouter.ai/keys).
3. **Pick a vision-capable model** at [openrouter.ai/models](https://openrouter.ai/models) — filter for "vision" support, since okforge-vision-ocr needs to send images. Check current availability and pricing there rather than trusting a hardcoded example list; a few reasonable starting points as of this writing:
   - `qwen/qwen3.5-32b-uncensored` — Qwen VLM family, matches what okforge-vision-ocr was tuned against.
   - `google/gemini-2.5-flash-preview` — fast and cheap.
   - `anthropic/claude-sonnet-4-6` — higher quality, higher cost.
4. **Create a `.env` file** in your project directory with **both** the endpoint and the vision model:

```
OPENAI_API_BASE=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-your-key-here
OKFORGE_VISION_MODEL=qwen/qwen3.5-32b-uncensored
```

`OKFORGE_VISION_MODEL` is easy to miss and okforge-vision-ocr won't warn you if it's wrong — it'll just fail (or silently fall back to a default model name your endpoint doesn't recognize) the first time it tries to call the API. Set it explicitly.

> **Note:** `OPENAI_API_BASE` and `LLM_API_KEY` route okforge-vision-ocr's calls (it talks to your endpoint directly). okforge itself is told which provider to use via the `--model provider/model` string you pass to `okforge init` in Step 5 — for OpenRouter that's an `openrouter/...` model string, which routes through okforge's own provider layer rather than `OPENAI_API_BASE`. You don't need to change anything for this to work, but if you're wondering why `OPENAI_API_BASE` doesn't affect okforge's own calls, that's why.
>
> Prefer setting environment variables in your terminal instead of a `.env` file? That works too, on all platforms:
>
> **Windows:** `set OPENAI_API_BASE=https://openrouter.ai/api/v1`
> **macOS / Linux:** `export OPENAI_API_BASE=https://openrouter.ai/api/v1`

---

## Step 4: Test with okforge-vision-ocr

This tool converts scanned PDFs into Markdown with extracted images. Skip this step if your PDF already has a real text layer — `okforge add` can ingest it directly (Step 5).

### Basic usage

Run the command from your project directory, pointing to the PDF in `source/`:

```bash
okforge-vision-ocr source/mydoc.pdf source/mydoc.md --figures
```

The output files are created alongside the PDF in `source/`. The `--figures` flag extracts drawings, engravings, and diagrams in addition to photographs — leave it off if your document is photos-only and you want the tighter default scope.

### What it produces

```
source/
├── mydoc.pdf              # Original PDF
├── mydoc.md                # Full markdown transcription
├── mydoc.pages.json        # Page-by-page content + citations
└── mydoc_images/           # Extracted image crops
    ├── p2_img1.jpg
    ├── p5_img1.jpg
    └── ...
```

`mydoc.pages.json` matters even though nothing in this guide opens it directly — it's what lets okforge attach real `(p. N)` citations to facts in the compiled summary, instead of an uncited paraphrase.

### Page-by-page processing

Useful for testing on a few pages before committing to a long document, or for reprocessing a handful of pages that came out wrong:

```bash
okforge-vision-ocr source/doc.pdf source/doc.md --pages 1-5   # First 5 pages only
okforge-vision-ocr source/doc.pdf source/doc.md --pages 10    # Just page 10
```

### Difficult tables

If a page has a complex table (multi-level headers, merged cells), add `--think --tables` — it turns on model reasoning and an information-first prompt that prioritizes getting the table's meaning right over mimicking its exact grid layout. Slower and more expensive per page; use it selectively via `--pages`.

```bash
okforge-vision-ocr source/doc.pdf source/doc.md --pages 12 --think --tables
```

---

## Step 5: Initialize a Knowledge Base (okforge)

okforge compiles your documents into a wiki with cross-linked concepts and entities.

### Initialize the KB

```bash
okforge init --model openai/Qwen3.6-27B-MTP --language en --json
```

Replace the model with whatever you're actually using. For a local llama.cpp server, the `openai/` prefix is correct (it's a generic OpenAI-compatible endpoint, pointed at by `OPENAI_API_BASE`). For OpenRouter, use the `openrouter/` prefix with the same model you set as `OKFORGE_VISION_MODEL` in Step 3:

```bash
okforge init --model openrouter/qwen/qwen3.5-32b-uncensored --language en --json
```

### Add your OCR output

```bash
okforge add source/mydoc.md
```

(Or, for a text-layer PDF you didn't need to OCR: `okforge add source/mydoc.pdf` directly.)

This runs the compile pipeline:
1. **Summary** — the model reads the document and writes a per-document summary, citing pages where `mydoc.pages.json` is available.
2. **Concept plan** — identifies which cross-document concepts and named entities this document touches, and whether each is new or should merge into an existing page.
3. **Generate** — writes (or updates) each concept/entity page.
4. **Index** — updates `wiki/index.md` and the `[[wikilink]]` cross-references between pages.

### Check the result

```bash
okforge status
okforge list
```

---

## Step 6: Explore Your Wiki

### Chat with your knowledge base

```bash
okforge chat
```

Ask questions about your documents in natural language, in an interactive session that keeps context across turns.

### Query a specific question

```bash
okforge query "What are the main conclusions?"
```

One-shot version of the above — no persistent session, useful for scripting.

### Visualize the wiki graph

```bash
okforge visualize
```

Writes a self-contained HTML file showing the `[[wikilink]]` graph connecting all your wiki pages — open it in a browser.

### Generate a slide deck

```bash
okforge deck new my-deck "A short intro deck on the document's key findings"
```

`my-deck` is a slug for the output folder; the quoted text tells the model what the deck should focus on. Generates a single-file HTML presentation from your wiki content — add `--critique` for a slower second pass that reviews and tightens the result.

---

## Full End-to-End Example

### Windows (Command Prompt)

```cmd
# 0. Create project directory and source folder
mkdir C:\Users\YOURNAME\mykb
cd C:\Users\YOURNAME\mykb
mkdir source

# Copy your PDF into source/ (use quotes if the source path has spaces)
copy "C:\Users\YOURNAME\Documents\My Folder\My Document.pdf" source\mydoc.pdf

# 1. Create and activate venv
python -m venv .venv
.venv\Scripts\activate

# 2. Install packages
python -m pip install --upgrade pip
python -m pip install okforge okforge-vision-ocr

# 3. Set LLM endpoint (local model)
set OPENAI_API_BASE=http://localhost:8080/v1
set LLM_API_KEY=no-key

# 4. OCR the PDF (output goes into source/)
okforge-vision-ocr source\mydoc.pdf source\mydoc.md --figures

# 5. Initialize knowledge base and ingest
okforge init --model openai/Qwen3.6-27B-MTP --language en --json
okforge add source\mydoc.md

# 6. Check results and ask a question
okforge list
okforge query "What is the main topic?"
```

### macOS / Linux

```bash
# 0. Create project directory and source folder
mkdir ~/mykb
cd ~/mykb
mkdir source

# Copy your PDF into source/ (use quotes if the source path has spaces)
cp "~/Documents/My Folder/My Document.pdf" source/mydoc.pdf

# 1. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install packages
python3 -m pip install --upgrade pip
python3 -m pip install okforge okforge-vision-ocr

# 3. Set LLM endpoint (local model)
export OPENAI_API_BASE=http://localhost:8080/v1
export LLM_API_KEY=no-key

# 4. OCR the PDF (output goes into source/)
okforge-vision-ocr source/mydoc.pdf source/mydoc.md --figures

# 5. Initialize knowledge base and ingest
okforge init --model openai/Qwen3.6-27B-MTP --language en --json
okforge add source/mydoc.md

# 6. Check results and ask a question
okforge list
okforge query "What is the main topic?"
```

### Resulting directory structure

```
mykb/
├── .venv/                  # Python virtual environment
├── .env                    # LLM endpoint + key (Step 3)
├── source/                 # Your originals and OCR output
│   ├── mydoc.pdf
│   ├── mydoc.md
│   ├── mydoc.pages.json
│   └── mydoc_images/
├── .okforge/                # KB state: config.yaml, hash registry
├── raw/                    # okforge's own archival copy of each ingested file
│   └── mydoc.md
└── wiki/                   # Generated wiki — this is the actual output
    ├── index.md
    ├── concepts/
    ├── entities/
    ├── summaries/
    └── sources/
        └── images/
```

---

## Troubleshooting

### "Python was not found"

Download Python from [python.org](https://python.org) and check "Add Python to PATH" during installation (Windows). On macOS, install via Homebrew: `brew install python`. On Linux, use your package manager: `sudo apt install python3 python3-venv`.

### "ModuleNotFoundError: No module named 'okforge'"

Your venv isn't activated. Your prompt should show `(.venv)`. Re-activate with:

- **Windows (CMD):** `.venv\Scripts\activate`
- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

### LLM call fails with a connection error

- **Local model:** verify the endpoint is actually up: `curl http://localhost:8080/v1/models`
- **OpenRouter:** check your API key is correct and has credit; confirm the model slug in `.env` matches one currently listed at [openrouter.ai/models](https://openrouter.ai/models)

### LLM call fails with an "invalid model" or similar error, specifically from okforge-vision-ocr

Almost always means `OKFORGE_VISION_MODEL` isn't set (or is set to a slug your endpoint doesn't serve) — it defaults to `Qwen3.6-27B-MTP`, which only exists on a local llama.cpp server configured with that model. Set `OKFORGE_VISION_MODEL` explicitly in `.env` (Step 3, Option B).

### `okforge deck` does nothing / shows a usage error

`deck` needs a subcommand and two arguments: `okforge deck new <name> "<what the deck should be about>"`. Bare `okforge deck` just shows help.

### Compilation is slow

- Use `--pages 1-3` on okforge-vision-ocr to test with a few pages first, and ingest a single short document before a whole book.
- Larger/pricier models generally produce better summaries but are slower — this is a real tradeoff, not a bug.
- `--think` (okforge-vision-ocr) enables deeper per-page reasoning at the cost of more tokens and time; reserve it for pages that actually need it (complex tables, dense layouts), not the whole document.

---

## Checking Cost Before You Commit (OpenRouter)

Cost depends entirely on which model you pick, and OpenRouter pricing changes — don't rely on a number in a guide. Instead, verify it yourself cheaply before running a whole book through the pipeline:

1. Run okforge-vision-ocr on a **single page** (`--pages 1`) and `okforge add` a **single short document**.
2. Check your actual spend on the [OpenRouter dashboard](https://openrouter.ai/activity) — it shows per-request cost immediately.
3. Multiply by your real page/document count to estimate the full run before committing to it.

A local model removes the per-call cost entirely at the expense of needing GPU hardware and being slower per call on modest hardware.
