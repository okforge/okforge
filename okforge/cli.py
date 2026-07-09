"""okforge CLI — command-line interface for the knowledge base workflow."""

from __future__ import annotations

# Silence import-time warnings from heavy third-party imports (litellm and
# friends clobber warning filters during their own import, so the filters are
# re-applied after all imports below).
import warnings

warnings.filterwarnings("ignore")

import asyncio
import contextlib
import json
import logging
import shutil
import sys
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Literal

import os

from agents import set_tracing_disabled

set_tracing_disabled(True)
# Use local model cost map — skip fetching from GitHub on every invocation
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import click
import yaml


# Silence LiteLLM's "could not pre-load <aws-service> response stream
# shape" warnings — they fire at import time when ``botocore`` isn't
# installed, but botocore is only needed for AWS Bedrock / SageMaker
# users. Filter must be attached before ``import litellm`` runs.
class _SuppressLiteLLMPreloadWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "could not pre-load" not in record.getMessage()


logging.getLogger("LiteLLM").addFilter(_SuppressLiteLLMPreloadWarnings())

import litellm

litellm.suppress_debug_info = True
from dotenv import load_dotenv

from okforge.agent.compiler import compile_long_doc
from okforge.config import (
    DEFAULT_CONFIG,
    LEGACY_STATE_DIR_NAME,
    STATE_DIR_NAME,
    load_config,
    save_config,
    load_global_config,
    register_kb,
    resolve_extra_headers,
    resolve_llm_extra_body,
    set_extra_headers,
    set_llm_extra_body,
    resolve_timeout,
    set_timeout,
    resolve_litellm_settings,
    state_dir,
)
from okforge.converter import _registry_path, _sanitize_stem, convert_document
from okforge.locks import atomic_write_json, atomic_write_text, kb_ingest_lock, kb_read_lock
from okforge.log import append_log
from okforge.mutation import MutationSnapshot, publish_staged_tree, snapshot_paths
from okforge.schema import AGENTS_MD, INDEX_SEED, PAGE_CONTENT_DIRS
from okforge.topic_tree import bootstrap as tt_bootstrap

# Suppress warnings after all imports — some deps override filters at import time
import warnings

warnings.filterwarnings("ignore")

load_dotenv()  # load from cwd (covers running inside the KB dir)

logger = logging.getLogger(__name__)


_KNOWN_PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "MOONSHOT_API_KEY",
    "ZHIPUAI_API_KEY",
    "DASHSCOPE_API_KEY",
)

# Providers that authenticate via OAuth device flow (subscription login
# handled by LiteLLM itself) — no API key env var is needed, so the
# missing-key warning would be a false alarm for them.
_OAUTH_PROVIDERS = {"chatgpt", "github_copilot"}


def _extract_provider(model: str) -> str | None:
    """Extract the LiteLLM provider name from a model string.

    ``model`` uses ``provider/model`` LiteLLM format.
    OpenAI models can omit the prefix; default to ``"openai"``.
    """
    model = model.strip()
    if not model:
        return None
    if "/" in model:
        return model.split("/")[0].lower()
    return "openai"


def _apply_litellm_settings(settings: dict) -> None:
    """Set each ``litellm:`` key verbatim onto the litellm module (process-wide
    globals, so they reach every LiteLLM call). Skips with a warning a key the
    installed litellm doesn't define, or one that is a litellm function (e.g.
    ``completion``) since overwriting it would break later calls. Applied, never
    reset — the values persist for the life of the process.
    """
    for key, value in settings.items():
        if not hasattr(litellm, key):
            logger.warning(
                "config: LiteLLM has no setting %r — ignoring it "
                "(check the spelling or your installed litellm version).",
                key,
            )
            continue
        if callable(getattr(litellm, key)):
            logger.warning(
                "config: 'litellm.%s' is a LiteLLM function, not a setting — "
                "refusing to overwrite it from the litellm: config block.",
                key,
            )
            continue
        setattr(litellm, key, value)


def _setup_llm_key(kb_dir: Path | None = None) -> None:
    """Set LiteLLM API key from LLM_API_KEY env var if present.

    Load order (override=False, so first one wins):
    1. System environment variables (already set)
    2. KB-local .env  (kb_dir/.env)
    3. Global .env    (~/.config/okforge/.env)

    Also propagates to provider-specific env vars (OPENAI_API_KEY, etc.)
    so that the Agents SDK litellm provider can pick them up.
    Provider is auto-detected from the KB config when available; otherwise
    a common provider set is used as a fallback.
    """
    if kb_dir is not None:
        env_file = kb_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)

    from okforge.config import GLOBAL_CONFIG_DIR

    global_env = GLOBAL_CONFIG_DIR / ".env"
    if global_env.exists():
        load_dotenv(global_env, override=False)

    api_key = os.environ.get("LLM_API_KEY", "")

    # Try to resolve the active provider, extra headers, and request timeout
    # from the KB config
    provider: str | None = None
    extra_headers: dict[str, str] = {}
    timeout: float | None = None
    litellm_settings: dict = {}
    llm_extra_body: dict = {}
    if kb_dir is not None:
        config_path = state_dir(kb_dir) / "config.yaml"
        if config_path.exists():
            config = load_config(config_path)
            model = config.get("model", "")
            provider = _extract_provider(str(model))
            extra_headers = resolve_extra_headers(config)
            timeout = resolve_timeout(config)
            llm_extra_body = resolve_llm_extra_body(config)
            litellm_settings = resolve_litellm_settings(config)
            # `timeout` / `extra_headers` in the block route to the per-call
            # stashes (replacing the legacy top-level keys); the rest are globals.
            if "extra_headers" in litellm_settings:
                extra_headers = resolve_extra_headers(
                    {"extra_headers": litellm_settings.pop("extra_headers")}
                )
            if "timeout" in litellm_settings:
                timeout = resolve_timeout({"timeout": litellm_settings.pop("timeout")})
    set_extra_headers(extra_headers)
    set_timeout(timeout)
    set_llm_extra_body(llm_extra_body)
    _apply_litellm_settings(litellm_settings)

    if not api_key:
        # Check if any provider key is already set. OAuth-based providers
        # (ChatGPT subscription, GitHub Copilot) don't use API keys at all,
        # so the warning is skipped for them.
        check_keys = (f"{provider.upper()}_API_KEY",) if provider else _KNOWN_PROVIDER_KEYS
        has_key = any(os.environ.get(k) for k in check_keys)
        if not has_key and provider not in _OAUTH_PROVIDERS:
            click.echo(
                "Warning: No LLM API key found. Set one of:\n"
                f"  1. {kb_dir / '.env' if kb_dir else '<kb_dir>/.env'} — LLM_API_KEY=sk-...\n"
                f"  2. {GLOBAL_CONFIG_DIR / '.env'} — LLM_API_KEY=sk-...\n"
                "  3. Export LLM_API_KEY in your shell profile"
            )
    else:
        litellm.api_key = api_key

        # Dynamically set the provider-specific env var when possible
        if provider:
            provider_env = f"{provider.upper()}_API_KEY"
            if not os.environ.get(provider_env):
                os.environ[provider_env] = api_key

        # Fallback: also set common provider keys so multi-provider
        # configs (e.g. PageIndex Cloud) still work
        for env_var in _KNOWN_PROVIDER_KEYS:
            if not os.environ.get(env_var):
                os.environ[env_var] = api_key


# Supported document extensions for the `add` command. Everything else
# (docx, pptx, html, scans, …) is expected to be pre-converted to Markdown
# before ingest — this fork stripped the MarkItDown conversion path.
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".md",
    ".markdown",
    ".txt",
}

# Map raw doc types to display types
_TYPE_DISPLAY_MAP = {
    "long_pdf": "pageindex",
    "pageindex_cloud": "pageindex",
}

# Registry types that were compiled via the long-doc pipeline (tree + per-page
# JSON source), as opposed to short docs (markdown source). Both the local
# long-PDF type and cloud imports belong here — they share the long-doc
# summary/source layout and recompile path.
_LONG_DOC_TYPES = {"long_pdf", "pageindex_cloud"}


def _is_long_doc(meta: dict) -> bool:
    return meta.get("type") in _LONG_DOC_TYPES


_SHORT_DOC_TYPES = {
    "pdf",
    "docx",
    "md",
    "markdown",
    "html",
    "htm",
    "txt",
    "csv",
    "pptx",
    "xlsx",
    "xls",
}


def _display_type(raw_type: str) -> str:
    """Map a raw stored doc type to a display type string."""
    if raw_type in _TYPE_DISPLAY_MAP:
        return _TYPE_DISPLAY_MAP[raw_type]
    if raw_type in _SHORT_DOC_TYPES:
        return "short"
    return raw_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_kb_dir(candidate: Path) -> bool:
    """True if candidate has a state dir — .okforge/, or legacy .openkb/."""
    return state_dir(candidate).is_dir()


def _find_kb_dir(override: Path | None = None) -> Path | None:
    """Find the KB root: explicit override → walk up from cwd → global default_kb."""
    # 0. Explicit override (--kb-dir or OPENKB_DIR)
    if override is not None:
        if _is_kb_dir(override):
            return override
        return None
    # 1. Walk up from cwd
    current = Path.cwd().resolve()
    while True:
        if _is_kb_dir(current):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # 2. Fall back to global config default_kb
    gc = load_global_config()
    default = gc.get("default_kb")
    if default:
        p = Path(default)
        if _is_kb_dir(p):
            return p
    return None


def _validate_skill_name(name: str) -> str | None:
    """Validate a skill slug. Returns None if OK, an error message if not.

    Rules: lowercase ``[a-z0-9-]``, no leading/trailing dash, no consecutive
    dashes, 1-64 characters. This matches the directory name we'll create
    under ``<kb>/output/skills/`` and the ``name:`` frontmatter field.
    """
    if not name:
        return "Skill name must not be empty."
    if len(name) > 64:
        return "Skill name must be at most 64 characters."
    if not all(("a" <= c <= "z") or ("0" <= c <= "9") or c == "-" for c in name):
        return "Skill name must contain only lowercase letters, digits, and dashes."
    if name.startswith("-"):
        return "Skill name must not have a leading dash."
    if name.endswith("-"):
        return "Skill name must not have a trailing dash."
    if "--" in name:
        return "Skill name must not contain consecutive dashes."
    return None


def _preflight_skill_new(kb_dir: Path, name: str) -> str | None:
    """Run shared safety gates for ``okforge skill new`` / ``/skill new``.

    Checks (in order):
      * skill name is a valid kebab-case slug
      * ``<kb>/wiki`` exists
      * any of ``<kb>/wiki/{summaries,concepts,entities}`` has at least
        one file (i.e. some document has been ingested + compiled)

    Returns ``None`` if all gates pass, else a single-line error message
    suitable to print to the user.

    Overwrite handling is NOT done here — the CLI handles it with
    ``-y`` + ``click.confirm``; chat refuses overwrite outright.
    """
    err = _validate_skill_name(name)
    if err:
        return err

    wiki = kb_dir / "wiki"
    if not wiki.is_dir():
        return "No wiki found in this KB. Run `okforge add <source>` to ingest documents first."

    has_content = any(
        (wiki / sub).is_dir() and any((wiki / sub).iterdir()) for sub in PAGE_CONTENT_DIRS
    )
    if not has_content:
        return (
            "Wiki has no compiled content yet. Ingest at least one "
            "document with `okforge add` first."
        )

    return None


def _clear_existing_skill_dir(kb_dir: Path, name: str) -> None:
    """Delete an existing ``<kb>/output/skills/<name>/`` directory."""
    from okforge.skill import skill_dir

    target = skill_dir(kb_dir, name)
    if target.exists():
        shutil.rmtree(target)


def _staging_dir_for(kb_dir: Path, file_path: Path) -> Path:
    safe = _sanitize_stem(file_path.stem)
    path = state_dir(kb_dir) / "staging" / f"add-{safe}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _cleanup_staging(path: Path | None) -> None:
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def _final_artifact_paths(result, kb_dir: Path) -> tuple[Path | None, Path | None]:
    final_raw = None
    final_source = None
    if result.raw_path is not None:
        final_raw = kb_dir / "raw" / result.raw_path.name
    if result.source_path is not None:
        final_source = kb_dir / "wiki" / "sources" / result.source_path.name
    return final_raw, final_source


def _snapshot_add_paths(
    kb_dir: Path,
    doc_name: str,
    final_raw: Path | None,
    final_source: Path | None,
) -> list[Path]:
    # NOTE: .okforge/files (the PageIndex blob store) is intentionally NOT
    # snapshotted here. It is append-only by {doc_id}, and the doc_id is only
    # assigned during indexing (after this snapshot). Eagerly snapshotting the
    # whole tree cost one os.link per existing blob on every add; instead the
    # long-doc add path registers just the new blob via snapshot.track_new()
    # once indexing has run.
    kb_state_dir = state_dir(kb_dir)
    paths = [
        kb_state_dir / "hashes.json",
        kb_state_dir / "pageindex.db",
        kb_state_dir / "pageindex.db-wal",
        kb_state_dir / "pageindex.db-shm",
        kb_state_dir / "pageindex.db-journal",
        kb_dir / "wiki" / "summaries" / f"{doc_name}.md",
        kb_dir / "wiki" / "sources" / f"{doc_name}.json",
        kb_dir / "wiki" / "sources" / "images" / doc_name,
        kb_dir / "wiki" / "concepts",
        kb_dir / "wiki" / "entities",
        kb_dir / "wiki" / "index.md",
        kb_dir / "wiki" / "log.md",
    ]
    if final_raw is not None:
        paths.append(final_raw)
    if final_source is not None:
        paths.append(final_source)
    return paths


def _run_compile_with_retry(coro_factory, label: str) -> None:
    click.echo(f"  {label}...")
    for attempt in range(2):
        try:
            asyncio.run(coro_factory())
            return
        except Exception as exc:
            if attempt == 0:
                click.echo("  Retrying compilation in 2s...")
                time.sleep(2)
            else:
                click.echo(f"  [ERROR] Compilation failed: {exc}")
                logger.debug("Compilation traceback:", exc_info=True)
                raise


def add_single_file(
    file_path: Path, kb_dir: Path, *, stage: bool = True
) -> Literal["added", "skipped", "failed"]:
    """Convert, index, and compile a single document under the KB mutation lock."""
    with kb_ingest_lock(state_dir(kb_dir)):
        return _add_single_file_locked(file_path, kb_dir, stage=stage)


def _add_single_file_locked(
    file_path: Path, kb_dir: Path, *, stage: bool = True
) -> Literal["added", "skipped", "failed"]:
    """Convert, index, and compile a single document into the knowledge base.

    Steps:
    1. Load config to get the model name.
    2. Convert the document (hash-check; skip if already known).
    3. If long doc: run PageIndex then compile_long_doc.
    4. Else: compile_short_doc.

    Returns:
        ``"added"`` on full success, ``"skipped"`` when the file's hash
        is already in the registry (dedup), or ``"failed"`` when any
        pipeline stage raised. URL-ingest distinguishes these so it can
        unlink the just-downloaded raw file on dedup (it would otherwise
        be an orphan) while preserving it on failure so the user can
        retry without re-downloading.
    """
    from okforge.agent.compiler import compile_short_doc
    from okforge.state import HashRegistry

    kb_state_dir = state_dir(kb_dir)
    config = load_config(kb_state_dir / "config.yaml")
    _setup_llm_key(kb_dir)
    model: str = config.get("model", DEFAULT_CONFIG["model"])

    staging_dir = _staging_dir_for(kb_dir, file_path) if stage else None
    snapshot: MutationSnapshot | None = None

    # 2. Convert document into staging when possible.
    click.echo(f"Adding: {file_path.name}")
    try:
        result = convert_document(file_path, kb_dir, staging_dir=staging_dir)
    except Exception as exc:
        click.echo(f"  [ERROR] Conversion failed: {exc}")
        logger.debug("Conversion traceback:", exc_info=True)
        _cleanup_staging(staging_dir)
        return "failed"

    if result.skipped:
        click.echo(f"  [SKIP] Already in knowledge base: {file_path.name}")
        _cleanup_staging(staging_dir)
        return "skipped"

    doc_name = result.doc_name or file_path.stem
    index_result = None  # populated only on the long-doc branch

    final_raw, final_source = _final_artifact_paths(result, kb_dir)
    try:
        snapshot = snapshot_paths(
            kb_dir,
            _snapshot_add_paths(kb_dir, doc_name, final_raw, final_source),
            operation="add",
            details={
                "file_hash": result.file_hash,
                "name": file_path.name,
                "doc_name": doc_name,
            },
            hardlink_dirs={
                kb_dir / "wiki" / "concepts",
                kb_dir / "wiki" / "entities",
            },
        )
        publish_staged_tree(staging_dir, kb_dir)
        if final_raw is not None:
            result.raw_path = final_raw
        if final_source is not None:
            result.source_path = final_source

        # 3/4. Index and compile
        if result.is_long_doc:
            if result.raw_path is None:
                raise RuntimeError(f"Converted long document has no raw artifact: {file_path.name}")
            click.echo("  Long document detected — indexing with PageIndex...")
            # PageIndex content-dedups: if the same content is already indexed
            # (e.g. hashes.json and pageindex.db diverged after a remove whose
            # PageIndex cleanup failed), col.add() returns the EXISTING doc_id
            # and writes no new blob. Capture the blob set *before* indexing so
            # we register only blobs THIS add actually created — otherwise
            # rollback would delete a prior document's blob.
            files_root = kb_state_dir / "files"
            blobs_before = set(files_root.glob("*/*")) if files_root.exists() else set()
            try:
                from okforge.indexer import index_long_document

                index_result = index_long_document(result.raw_path, kb_dir, doc_name=doc_name)
            except Exception as exc:
                click.echo(f"  [ERROR] Indexing failed: {exc}")
                logger.debug("Indexing traceback:", exc_info=True)
                raise

            # Register only the newly-created blob artifacts for this doc (the
            # {doc_id} file + its images dir) — the append-only store means the
            # name isn't known until now — so rollback + crash recovery remove
            # exactly this add's blob, never a pre-existing one, instead of
            # snapshotting the whole store up front. The doc_id guard + the
            # blobs_before diff keep a dedup hit (or an unexpected empty doc_id)
            # from registering — and later deleting — existing blobs.
            if index_result.doc_id and files_root.exists():
                snapshot.track_new(
                    [
                        p
                        for p in files_root.glob(f"*/{index_result.doc_id}*")
                        if p not in blobs_before
                    ]
                )

            summary_path = kb_dir / "wiki" / "summaries" / f"{doc_name}.md"
            _run_compile_with_retry(
                lambda: compile_long_doc(
                    doc_name,
                    summary_path,
                    index_result.doc_id,
                    kb_dir,
                    model,
                    doc_description=index_result.description,
                ),
                label=f"Compiling long doc (doc_id={index_result.doc_id})",
            )
        else:
            if result.source_path is None:
                raise RuntimeError(f"Converted document has no source artifact: {file_path.name}")
            source_path = result.source_path
            _run_compile_with_retry(
                lambda: compile_short_doc(doc_name, source_path, kb_dir, model),
                label="Compiling short doc",
            )

        # Register hash only after successful compilation.
        if result.file_hash:
            registry = HashRegistry(kb_state_dir / "hashes.json")
            doc_type = "long_pdf" if result.is_long_doc else file_path.suffix.lstrip(".")
            meta = {
                "name": file_path.name,
                "doc_name": doc_name,
                "type": doc_type,
                "path": _registry_path(file_path, kb_dir),
            }
            if result.raw_path is not None:
                meta["raw_path"] = _registry_path(result.raw_path, kb_dir)
            if result.source_path is not None:
                meta["source_path"] = _registry_path(result.source_path, kb_dir)
            if index_result is not None:
                meta["doc_id"] = index_result.doc_id
            registry.remove_by_doc_name(doc_name)
            for existing_hash, existing_meta in list(registry.all_entries().items()):
                if (
                    existing_hash != result.file_hash
                    and not existing_meta.get("doc_name")
                    and existing_meta.get("name") == file_path.name
                ):
                    registry.remove_by_hash(existing_hash)
            registry.add(result.file_hash, meta)

        snapshot.mark_committed()
    except Exception:
        if snapshot is None:
            click.echo(f"  [ERROR] Failed to prepare mutation snapshot for {file_path.name}.")
            _cleanup_staging(staging_dir)
            return "failed"
        rollback_error = snapshot.rollback_best_effort()
        if rollback_error is None:
            snapshot.discard_best_effort()
        else:
            click.echo(
                "  [ERROR] Rollback failed; mutation journal retained for recovery: "
                f"{snapshot.journal_path}"
            )
        _cleanup_staging(staging_dir)
        return "failed"
    finally:
        _cleanup_staging(staging_dir)

    try:
        append_log(kb_dir / "wiki", "ingest", file_path.name)
    except Exception as exc:
        logger.warning("Failed to append ingest log for %s: %s", file_path.name, exc)
    cleanup_error = snapshot.discard_best_effort()
    if cleanup_error is not None:
        click.echo(
            f"  [WARN] {file_path.name} added, but mutation journal cleanup failed: {cleanup_error}"
        )
    click.echo(f"  [OK] {file_path.name} added to knowledge base.")
    return "added"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose logging.")
@click.option(
    "--kb-dir",
    "kb_dir_override",
    default=None,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to a KB root directory (overrides auto-detection).",
)
@click.pass_context
def cli(ctx, verbose, kb_dir_override):
    """okforge — local LLM knowledge-base workflow (hard fork of OpenKB)."""
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.WARNING,
    )
    if verbose:
        logging.getLogger("okforge").setLevel(logging.DEBUG)
    ctx.ensure_object(dict)
    if kb_dir_override:
        ctx.obj["kb_dir_override"] = Path(kb_dir_override)
    else:
        env_kb = os.environ.get("OPENKB_DIR")
        if env_kb:
            ctx.obj["kb_dir_override"] = Path(env_kb).resolve()
        else:
            ctx.obj["kb_dir_override"] = None


def _with_kb_lock(*, exclusive: bool):
    """Wrap a Click command in the appropriate KB lock when a KB exists."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
            if kb_dir is None:
                return fn(ctx, *args, **kwargs)
            if exclusive:
                with kb_ingest_lock(state_dir(kb_dir)):
                    return fn(ctx, *args, **kwargs)
            with kb_read_lock(state_dir(kb_dir)):
                return fn(ctx, *args, **kwargs)

        return wrapper

    return decorator


@cli.command()
@click.argument("path", default=".")
def use(path):
    """Set PATH as the default knowledge base."""
    target = Path(path).resolve()
    if not _is_kb_dir(target):
        click.echo(f"Not a knowledge base: {target}")
        return
    register_kb(target)
    click.echo(f"Default KB set to: {target}")


def _verify_migrated_state(new_dir: Path) -> tuple[bool, str]:
    """Sanity-check a freshly-copied state dir is genuinely readable,
    not just present. Checked before the legacy dir is renamed aside."""
    config_path = new_dir / "config.yaml"
    if not config_path.exists():
        return False, "config.yaml missing"
    try:
        load_config(config_path)
    except Exception as exc:
        return False, f"config.yaml unreadable: {exc}"
    hashes_path = new_dir / "hashes.json"
    if hashes_path.exists():
        try:
            json.loads(hashes_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"hashes.json unreadable: {exc}"
    return True, "ok"


def _migrate_one_kb(kb_dir: Path, *, dry_run: bool) -> None:
    new_dir = kb_dir / STATE_DIR_NAME
    old_dir = kb_dir / LEGACY_STATE_DIR_NAME
    new_exists = new_dir.is_dir()
    old_exists = old_dir.is_dir()

    if new_exists and old_exists:
        click.echo(
            f"[ERROR] {kb_dir}: both {STATE_DIR_NAME}/ and {LEGACY_STATE_DIR_NAME}/ "
            "exist — not migrating automatically. Inspect both by hand and remove "
            "whichever is stale before re-running."
        )
        return
    if new_exists:
        click.echo(f"{kb_dir}: already migrated ({STATE_DIR_NAME}/ present).")
        return
    if not old_exists:
        click.echo(
            f"[ERROR] {kb_dir}: not an okforge KB "
            f"(no {STATE_DIR_NAME}/ or {LEGACY_STATE_DIR_NAME}/ found)."
        )
        return

    files = [p for p in old_dir.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    if dry_run:
        click.echo(
            f"{kb_dir}: would migrate {LEGACY_STATE_DIR_NAME}/ -> {STATE_DIR_NAME}/ "
            f"({len(files)} file(s), {total_bytes} bytes)."
        )
        return

    import datetime

    with kb_ingest_lock(old_dir):
        shutil.copytree(old_dir, new_dir, dirs_exist_ok=False)
        ok, msg = _verify_migrated_state(new_dir)
        if not ok:
            shutil.rmtree(new_dir)
            click.echo(
                f"[ERROR] {kb_dir}: migration verification failed ({msg}); aborted — "
                f"{LEGACY_STATE_DIR_NAME}/ untouched."
            )
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = kb_dir / f"{LEGACY_STATE_DIR_NAME}.migrated-{timestamp}"
        old_dir.rename(backup_dir)

    click.echo(
        f"{kb_dir}: migrated {len(files)} file(s), {total_bytes} bytes -> "
        f"{STATE_DIR_NAME}/. Old state preserved at {backup_dir.name}/ — safe to "
        f"delete manually once you've verified (e.g. `okforge status`)."
    )


def _migrate_global_config(*, dry_run: bool) -> None:
    # Local import (not module-level): must re-read the current attribute
    # off the config module at call time, not a value bound once when
    # this module was imported — otherwise tests that monkeypatch
    # okforge.config.GLOBAL_CONFIG_DIR silently no-op and this function
    # falls through to the real ~/.config paths instead.
    import okforge.config as config_mod

    new_dir = config_mod.GLOBAL_CONFIG_DIR
    old_dir = config_mod.LEGACY_GLOBAL_CONFIG_DIR
    new_exists = new_dir.is_dir()
    old_exists = old_dir.is_dir()

    if new_exists and old_exists:
        click.echo(
            f"[ERROR] global config: both {new_dir} and {old_dir} exist — "
            "not migrating automatically. Inspect both by hand."
        )
        return
    if new_exists:
        click.echo(f"Global config: already migrated ({new_dir} present).")
        return
    if not old_exists:
        click.echo("Global config: nothing to migrate (no ~/.config/openkb found).")
        return

    if dry_run:
        click.echo(f"Would migrate {old_dir} -> {new_dir}.")
        return

    import datetime

    shutil.copytree(old_dir, new_dir, dirs_exist_ok=False)
    if not (new_dir / "global.yaml").exists() or not yaml.safe_load(
        (new_dir / "global.yaml").read_text(encoding="utf-8")
    ):
        shutil.rmtree(new_dir)
        click.echo("[ERROR] global config: migration verification failed; aborted.")
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = old_dir.parent / f"openkb.migrated-{timestamp}"
    old_dir.rename(backup_dir)
    click.echo(f"Global config migrated -> {new_dir}. Old copy preserved at {backup_dir}.")


def _migrate_global_skills(*, dry_run: bool) -> None:
    import okforge.config as config_mod  # see _migrate_global_config for why local

    old_dir = Path.home() / ".openkb" / "skills"
    new_dir = config_mod.GLOBAL_CONFIG_DIR / "skills"
    new_exists = new_dir.is_dir()
    old_exists = old_dir.is_dir()

    if new_exists and old_exists:
        click.echo(
            f"[ERROR] global skills: both {new_dir} and {old_dir} exist — "
            "not migrating automatically. Inspect both by hand."
        )
        return
    if new_exists or not old_exists:
        return  # already migrated, or nothing there to begin with — quiet no-op

    if dry_run:
        click.echo(f"Would migrate {old_dir} -> {new_dir}.")
        return

    new_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(old_dir, new_dir, dirs_exist_ok=False)
    old_dir.rename(old_dir.parent / f"{old_dir.name}.migrated")
    click.echo(f"Global skills migrated -> {new_dir}.")


@cli.command()
@click.argument("kb_path", required=False, default=None, type=click.Path(path_type=Path))
@click.option("--dry-run", is_flag=True, help="Report what would change without touching anything.")
@click.option(
    "--global",
    "do_global",
    is_flag=True,
    help="Also migrate ~/.config/openkb (global registry) and ~/.openkb/skills "
    "to their ~/.config/okforge equivalents.",
)
def migrate(kb_path, dry_run, do_global):
    """Migrate a KB from the pre-rename .openkb/ layout to .okforge/.

    Copies KB_PATH/.openkb/ to KB_PATH/.okforge/, verifies the copy is
    genuinely readable, then renames the original aside to
    KB_PATH/.openkb.migrated-<timestamp>/ (kept as a backup, not deleted).
    Safe to re-run — a KB that's already migrated is a no-op.
    """
    if kb_path is None and not do_global:
        click.echo(
            "Specify a KB directory to migrate, or pass --global for the global config/skills dirs."
        )
        return
    if kb_path is not None:
        _migrate_one_kb(kb_path.resolve(), dry_run=dry_run)
    if do_global:
        _migrate_global_config(dry_run=dry_run)
        _migrate_global_skills(dry_run=dry_run)


_LANGUAGE_MAX_LEN = 50
_MODEL_MAX_LEN = 100


def _coerce_language(value: str | None) -> str | None:
    """Strip a language string; treat blanks as unset; reject unsafe values.

    The language string is interpolated into LLM system prompts (see
    ``_SYSTEM_TEMPLATE`` in ``okforge/agent/compiler.py`` and the query agent's
    instructions), so values with newlines or excessive length would let an
    external caller smuggle instructions into the prompt. Capping at
    ``_LANGUAGE_MAX_LEN`` and rejecting control characters is enough to close
    that vector while still allowing common forms ("en", "ko", "Korean",
    "Simplified Chinese").

    Returns the cleaned string, or ``None`` if the input was missing or blank
    after stripping. Raises ``click.BadParameter`` on unsafe input.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > _LANGUAGE_MAX_LEN or any(c in value for c in "\n\r\t"):
        raise click.BadParameter(
            f"language must be {_LANGUAGE_MAX_LEN} characters or fewer with no control characters",
            param_hint="'--language'",
        )
    return value


def _language_option_callback(_ctx, _param, value):
    return _coerce_language(value)


def _coerce_model(value: str | None) -> str | None:
    """Strip a model string; treat blanks as unset; reject unsafe values.

    Mirrors ``_coerce_language``. The model string is passed to LiteLLM and
    also echoed in logs/CLI output, so embedded control characters would
    corrupt that output. Capping length keeps pathological values out of
    config.yaml.

    Returns the cleaned string, or ``None`` if the input was missing or blank
    after stripping. Raises ``click.BadParameter`` on unsafe input.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > _MODEL_MAX_LEN or any(c in value for c in "\n\r\t"):
        raise click.BadParameter(
            f"model must be {_MODEL_MAX_LEN} characters or fewer with no control characters",
            param_hint="'--model'",
        )
    return value


def _model_option_callback(_ctx, _param, value):
    return _coerce_model(value)


def _stdin_is_tty() -> bool:
    """Return True when stdin is a real terminal.

    Used to skip optional ``okforge init`` prompts when input is piped or
    redirected, so existing automation (e.g. ``printf '\\n\\n' | okforge init``)
    keeps working as new prompts are added. Mirrors ``_stream_to_tty`` from #45.
    """
    return sys.stdin.isatty()


@cli.command()
@click.option(
    "--model",
    "-m",
    "model",
    default=None,
    metavar="MODEL",
    callback=_model_option_callback,
    help=(
        "LLM in LiteLLM provider/model format "
        "(e.g. 'gpt-5.4', 'anthropic/claude-sonnet-4-6'). "
        "Skips the interactive prompt when set."
    ),
)
@click.option(
    "--language",
    "-l",
    "language",
    default=None,
    metavar="LANG",
    callback=_language_option_callback,
    help="Wiki output language (e.g. 'en', 'ko'). Skips the interactive prompt when set.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Non-interactive machine mode: no prompts, JSON result on stdout.",
)
def init(model, language, as_json):
    """Initialise a new knowledge base in the current directory."""
    kb_state_dir = Path(STATE_DIR_NAME)
    if _is_kb_dir(Path.cwd()):
        if as_json:
            click.echo(json.dumps({"created": False, "kb_dir": str(Path.cwd())}))
            return
        click.echo("Knowledge base already initialized.")
        return

    api_key = ""
    if as_json:
        # Machine mode: no prompts — model/language from flags or defaults,
        # never an API-key prompt (configure the key via .env yourself).
        if not model:
            model = DEFAULT_CONFIG["model"]
        if not language:
            language = DEFAULT_CONFIG["language"]
    else:
        # Interactive prompts
        click.echo("Pick an LLM in `provider/model` LiteLLM format:")
        click.echo("  OpenAI:    gpt-5.4, gpt-5.4-mini")
        click.echo("  Anthropic: anthropic/claude-sonnet-4-6, anthropic/claude-opus-4-6")
        click.echo("  Gemini:    gemini/gemini-3.1-pro-preview, gemini/gemini-3-flash-preview")
        click.echo("  DeepSeek:  deepseek/deepseek-v4-flash, deepseek/deepseek-v4-pro")
        click.echo("  Others:    see https://docs.litellm.ai/docs/providers")
        click.echo()
        if model is None and _stdin_is_tty():
            model = _coerce_model(
                click.prompt(
                    f"Model (enter for default {DEFAULT_CONFIG['model']})",
                    default=DEFAULT_CONFIG["model"],
                    show_default=False,
                )
            )
        if not model:
            model = DEFAULT_CONFIG["model"]
        api_key = click.prompt(
            "LLM API Key (saved to .env, enter to skip)",
            default="",
            hide_input=True,
            show_default=False,
        ).strip()
        if language is None and _stdin_is_tty():
            language = _coerce_language(
                click.prompt(
                    f"Wiki language (enter for default {DEFAULT_CONFIG['language']})",
                    default=DEFAULT_CONFIG["language"],
                    show_default=False,
                )
            )
        if not language:
            language = DEFAULT_CONFIG["language"]
    # Create directory structure
    Path("raw").mkdir(exist_ok=True)
    Path("wiki/sources/images").mkdir(parents=True, exist_ok=True)
    Path("wiki/summaries").mkdir(parents=True, exist_ok=True)
    Path("wiki/concepts").mkdir(parents=True, exist_ok=True)
    Path("wiki/entities").mkdir(parents=True, exist_ok=True)

    # Write wiki files
    atomic_write_text(Path("wiki/AGENTS.md"), AGENTS_MD)
    atomic_write_text(Path("wiki/index.md"), INDEX_SEED)
    atomic_write_text(Path("wiki/log.md"), "# Operations Log\n\n")

    # Create .okforge/ state directory
    kb_state_dir.mkdir()
    config = {
        "model": model,
        "language": language,
        "pageindex_threshold": DEFAULT_CONFIG["pageindex_threshold"],
    }
    save_config(kb_state_dir / "config.yaml", config)
    atomic_write_json(kb_state_dir / "hashes.json", {})

    # Write API key to KB-local .env (0600) if the user provided one
    if api_key:
        env_path = Path(".env")
        if env_path.exists():
            click.echo(".env already exists, skipping write. Add LLM_API_KEY manually if needed.")
        else:
            env_path.write_text(f"LLM_API_KEY={api_key}\n", encoding="utf-8")
            os.chmod(env_path, 0o600)
            click.echo("Saved LLM API key to .env.")

    # Register this KB in the global config
    register_kb(Path.cwd())

    if as_json:
        click.echo(
            json.dumps(
                {
                    "created": True,
                    "kb_dir": str(Path.cwd()),
                    "model": model,
                    "language": language,
                }
            )
        )
        return
    click.echo("Knowledge base initialized.")


@cli.command()
@click.argument("path", required=True)
@click.pass_context
@_with_kb_lock(exclusive=True)
def add(ctx, path):
    """Add a document or directory of documents at PATH to the knowledge base.

    PATH may be a local file, a local directory (which is walked
    recursively for supported extensions), or an http(s) URL. URLs are
    fetched into ``raw/`` first: PDF responses (by Content-Type and
    magic-byte sniff) are saved as ``.pdf``; HTML responses are run
    through trafilatura's main-content extractor and saved as ``.md``.

    Other formats (docx, pptx, scans, …) are expected to be pre-converted
    to Markdown before ingest — a page-aware pre-conversion also enables
    real page citations via a sibling .pages.json.
    """
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.")
        return

    # URL ingest: download into raw/ first, then call add_single_file explicitly.
    # Keep staged conversion enabled so converted source artifacts do not touch
    # the live KB before the mutation snapshot exists. The tri-state outcome
    # still lets us clean up the just-downloaded raw file on dedup.
    from okforge.url_ingest import looks_like_url, fetch_url_to_raw

    if looks_like_url(path):
        fetched = fetch_url_to_raw(path, kb_dir)
        if fetched is None:
            return
        outcome = add_single_file(fetched, kb_dir)
        # Only clean up on dedup-skip. On "failed" we keep the file so
        # the user can retry (e.g. transient LLM error during compile)
        # without re-downloading — and so they don't lose data when
        # indexing has already succeeded but compilation didn't.
        if outcome == "skipped":
            fetched.unlink(missing_ok=True)
        return

    target = Path(path)
    if not target.exists():
        click.echo(f"Path does not exist: {path}")
        return

    if target.is_dir():
        files = [
            f
            for f in sorted(target.rglob("*"))
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            click.echo(f"No supported files found in {path}.")
            return
        total = len(files)
        click.echo(f"Found {total} supported file(s) in {path}.")
        for i, f in enumerate(files, 1):
            click.echo(f"\n[{i}/{total}] ", nl=False)
            add_single_file(f, kb_dir)
    else:
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            click.echo(
                f"Unsupported file type: {target.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            return
        add_single_file(target, kb_dir)


def _stream_to_tty() -> bool:
    """Return True when stdout is a real terminal.

    Used to auto-disable streaming output when ``okforge query`` is piped,
    redirected to a file, or run as a subprocess — streaming output emits
    interleaved tool-call lines that are noisy for non-interactive callers,
    and the non-streaming branch returns just the final answer string.
    """
    return sys.stdout.isatty()


@cli.command()
@click.argument("question")
@click.option("--save", is_flag=True, default=False, help="Save the answer to wiki/explorations/.")
@click.option(
    "--raw",
    "raw",
    is_flag=True,
    default=False,
    help="Show raw markdown source instead of rendered output (keeps tool-call colors).",
)
@click.pass_context
def query(ctx, question, save, raw):
    """Query the knowledge base with QUESTION."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.")
        return

    from okforge.agent.query import run_query

    config = load_config(state_dir(kb_dir) / "config.yaml")
    _setup_llm_key(kb_dir)
    model: str = config.get("model", DEFAULT_CONFIG["model"])

    stream = _stream_to_tty()
    try:
        answer = asyncio.run(run_query(question, kb_dir, model, stream=stream, raw=raw))
        if not stream and answer:
            click.echo(answer)
    except Exception as exc:
        click.echo(f"[ERROR] Query failed: {exc}")
        return

    append_log(kb_dir / "wiki", "query", question)

    if save and answer:
        import re
        from okforge.lint import list_existing_wiki_targets, strip_ghost_wikilinks

        slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60]
        explore_dir = kb_dir / "wiki" / "explorations"
        explore_dir.mkdir(parents=True, exist_ok=True)
        explore_path = explore_dir / f"{slug}.md"
        # Strip ghost wikilinks the agent may have emitted to non-existent
        # concept/summary pages — the schema_md in the agent's instructions
        # encourages [[wikilinks]] but the agent's view of "which pages
        # exist" can drift from disk reality.
        known = list_existing_wiki_targets(kb_dir / "wiki")
        cleaned_answer, _ = strip_ghost_wikilinks(answer, known)
        explore_path.write_text(
            f'---\nquery: "{question}"\n---\n\n{cleaned_answer}\n',
            encoding="utf-8",
        )
        click.echo(f"\nSaved to {explore_path}")


@cli.command()
@click.pass_context
def mcp(ctx):
    """Start an MCP server (stdio transport) exposing this knowledge base.

    Read-only: query, grep_wiki, read_wiki_page, status, and — when this
    KB has topic_tree enabled — read_topic. Bound to whichever KB the
    usual resolution finds (cwd walk-up, or --kb-dir). Point an MCP
    client at this command, e.g.:

        claude mcp add --transport stdio okforge -- okforge --kb-dir /path/to/kb mcp

    No authentication. stdio is safe as-is (it's just this process's
    pipes, nothing listens on the network) — but if you bridge it to be
    reachable remotely, only do so over a network you already trust
    (SSH tunnel, VPN), never a directly-exposed port: anyone who can
    reach it gets full read access to this KB, including query (which
    runs your configured LLM on your behalf) with no login required.
    """
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.", err=True)
        ctx.exit(1)

    from okforge.agent.mcp_server import build_mcp_server

    # stdio transport requires stdout to carry *only* newline-delimited
    # JSON-RPC once the server starts — any stray click.echo (e.g.
    # _setup_llm_key's missing-API-key warning) would corrupt the stream
    # for every real MCP client. Redirect anything printed during setup
    # to stderr instead, where a human running this directly still sees it.
    with contextlib.redirect_stdout(sys.stderr):
        config = load_config(state_dir(kb_dir) / "config.yaml")
        _setup_llm_key(kb_dir)
    model: str = config.get("model", DEFAULT_CONFIG["model"])

    server = build_mcp_server(kb_dir, model)
    server.run(transport="stdio")


def _cleanup_pageindex(
    kb_state_dir: Path,
    kb_dir: Path,
    doc_name: str,
    doc_id: str | None,
) -> tuple[bool, str]:
    """Drop a long-doc entry from PageIndex's local SQLite + remove its
    managed files. Returns ``(did_cleanup, message)``.

    No-op (returns ``(False, "no PageIndex state")``) when no
    ``pageindex.db`` exists — short-doc-only KBs never created any.

    Falls back to matching by ``doc_name`` via ``list_documents()`` when
    the registry entry pre-dates PR #51's ``doc_id`` field. Ambiguous
    multi-match cases are skipped with a warning rather than guessed.
    """
    if not (kb_state_dir / "pageindex.db").exists():
        return False, "no PageIndex state"

    from pageindex import PageIndexClient

    _setup_llm_key(kb_dir)
    config = load_config(kb_state_dir / "config.yaml")
    model = config.get("model", DEFAULT_CONFIG.get("model", "gpt-5.4"))
    client = PageIndexClient(model=model, storage_path=str(kb_state_dir))
    col = client.collection()

    if doc_id is None:
        candidates = [d for d in col.list_documents() if d.get("doc_name") == doc_name]
        if not candidates:
            return False, "no PageIndex doc to delete"
        if len(candidates) > 1:
            return False, (
                f"{len(candidates)} PageIndex docs match doc_name='{doc_name}'; "
                "skipping (re-add to refresh)"
            )
        doc_id = candidates[0]["doc_id"]

    col.delete_document(doc_id)
    return True, f"deleted PageIndex doc ({doc_id[:12]}…)"


def _resolve_doc_identifier(registry, identifier: str) -> list[tuple[str, dict]]:
    """Find registry entries matching ``identifier``.

    Match precedence (returns immediately on the first non-empty bucket):
      1. Exact match on ``metadata['name']`` (the original filename).
      2. Exact match on ``metadata['doc_name']`` (the slug).
      3. Case-insensitive substring match on either field.

    Returns ``[(file_hash, metadata), ...]``. Callers handle the empty,
    single, and multi-match cases.
    """
    entries = registry.all_entries()

    exact_name = [(h, m) for h, m in entries.items() if m.get("name") == identifier]
    if exact_name:
        return exact_name

    exact_slug = [(h, m) for h, m in entries.items() if m.get("doc_name") == identifier]
    if exact_slug:
        return exact_slug

    needle = identifier.lower()
    fuzzy = [
        (h, m)
        for h, m in entries.items()
        if needle in (m.get("name") or "").lower() or needle in (m.get("doc_name") or "").lower()
    ]
    return fuzzy


@cli.command()
@click.argument("identifier")
@click.option(
    "--keep-raw", is_flag=True, default=False, help="Don't delete the original file from raw/."
)
@click.option(
    "--keep-empty",
    "--keep-empty-concepts",
    "keep_empty",
    is_flag=True,
    default=False,
    help="Keep concept AND entity pages whose only source was the "
    "removed doc (leaving an empty sources: [] list). Useful "
    "when replacing the doc with a newer version. "
    "(--keep-empty-concepts is a backward-compatible alias.)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be done without modifying anything.",
)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the confirmation prompt.")
@click.pass_context
@_with_kb_lock(exclusive=True)
def remove(ctx, identifier, keep_raw, keep_empty, dry_run, yes):
    """Remove a document from the knowledge base.

    IDENTIFIER may be the original filename ("paper.pdf"), the doc_name
    slug ("paper-a1b2c3d4e5f6"), or a substring that uniquely matches one.

    Deletes the doc's summary and source files, prunes the doc from
    concept- and entity-page frontmatter and Related Documents sections,
    drops the Documents entry from index.md, removes the hash entry, and
    finally runs `lint --fix` to clean any dangling wikilinks.

    Concept and entity pages whose only source was this doc are deleted by
    default; use --keep-empty to retain them.
    """
    from okforge.agent.compiler import (
        remove_doc_from_concept_pages,
        remove_doc_from_entity_pages,
        remove_doc_from_index,
        scan_affected_pages,
    )
    from okforge.lint import fix_broken_links
    from okforge.state import HashRegistry

    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.")
        return

    kb_state_dir = state_dir(kb_dir)
    registry = HashRegistry(kb_state_dir / "hashes.json")

    matches = _resolve_doc_identifier(registry, identifier)
    if not matches:
        click.echo(f"No document matching '{identifier}' found in the KB.")
        click.echo("Try `okforge list` to see indexed documents.")
        return
    if len(matches) > 1:
        click.echo(f"'{identifier}' matches multiple documents:")
        for _, m in matches:
            click.echo(f"  - {m.get('name', '?')}  (doc_name: {m.get('doc_name', '?')})")
        click.echo("Use a more specific name or the exact doc_name slug.")
        return

    file_hash, meta = matches[0]
    name = meta.get("name", "?")
    doc_name = meta.get("doc_name") or Path(name).stem
    doc_type = meta.get("type", "")
    wiki_dir = kb_dir / "wiki"

    # ----- Build the plan (no side effects) -----
    actions: list[tuple[str, str]] = []

    summary_path = wiki_dir / "summaries" / f"{doc_name}.md"
    if summary_path.exists():
        actions.append(("DELETE", str(summary_path.relative_to(kb_dir))))

    source_md = wiki_dir / "sources" / f"{doc_name}.md"
    source_json = wiki_dir / "sources" / f"{doc_name}.json"
    if source_md.exists():
        actions.append(("DELETE", str(source_md.relative_to(kb_dir))))
    if source_json.exists():
        actions.append(("DELETE", str(source_json.relative_to(kb_dir))))

    # Per-doc extracted-images directory (PDF page images + base64 images
    # from docx/pptx + copied relative refs from .md inputs). Created by
    # okforge.images during ingest, keyed by doc_name.
    images_dir = wiki_dir / "sources" / "images" / doc_name
    if images_dir.is_dir():
        actions.append(
            (
                "DELETE",
                f"{images_dir.relative_to(kb_dir)}/  (images directory)",
            )
        )

    # Scan concept pages to predict which will be edited vs. deleted.
    # Only frontmatter ``sources:`` membership drives the plan — body-only
    # references (e.g. a stray ``See also:`` line a user added by hand
    # without updating sources) are cleaned by the executor but don't
    # affect the delete/edit classification, so the plan reflects what
    # the executor will actually do.
    source_file_marker = f"summaries/{doc_name}.md"
    affected_concepts = scan_affected_pages(wiki_dir / "concepts", source_file_marker)

    concept_deletes = [s for s, r in affected_concepts if r == 0 and not keep_empty]
    concept_edits = [s for s, r in affected_concepts if r > 0 or keep_empty]
    for slug in concept_deletes:
        actions.append(("DELETE", f"wiki/concepts/{slug}.md  (only source: this doc)"))
    for slug in concept_edits:
        actions.append(("MODIFY", f"wiki/concepts/{slug}.md  (drop this doc from sources)"))

    # Scan entity pages with the same frontmatter logic as concepts. The
    # executor calls ``remove_doc_from_entity_pages``; this only makes the
    # preview/summary truthful about what it will delete vs. edit.
    affected_entities = scan_affected_pages(wiki_dir / "entities", source_file_marker)

    entity_deletes = [s for s, r in affected_entities if r == 0 and not keep_empty]
    entity_edits = [s for s, r in affected_entities if r > 0 or keep_empty]
    for slug in entity_deletes:
        actions.append(("DELETE", f"wiki/entities/{slug}.md  (only source: this doc)"))
    for slug in entity_edits:
        actions.append(("MODIFY", f"wiki/entities/{slug}.md  (drop this doc from sources)"))

    if (wiki_dir / "index.md").exists():
        actions.append(("MODIFY", "wiki/index.md  (remove Documents entry)"))

    actions.append(("REGISTRY", f"remove hash entry  ({file_hash[:12]}…)"))

    # Long PDFs leave state in PageIndex's local store (`.okforge/pageindex.db`
    # row + `.okforge/files/<collection>/<doc_id>.pdf` + extracted images).
    # Only flag this when both the registry says long_pdf and PageIndex
    # state exists on disk — short-doc-only KBs never created any.
    pageindex_doc_id = meta.get("doc_id")
    pageindex_state_exists = (kb_state_dir / "pageindex.db").exists()
    cleanup_pageindex = doc_type == "long_pdf" and pageindex_state_exists
    if cleanup_pageindex:
        if pageindex_doc_id:
            actions.append(
                (
                    "PAGEINDEX",
                    f"delete document ({pageindex_doc_id[:12]}…)",
                )
            )
        else:
            actions.append(
                (
                    "PAGEINDEX",
                    "delete document (lookup by doc_name; legacy entry)",
                )
            )

    raw_path = None
    if not keep_raw:
        raw_dir = kb_dir / "raw"
        # Raw copies are named by doc_name since the collision fix: use the
        # recorded raw_path when present. Only pre-upgrade entries (no
        # raw_path field) fall back to the original filename — a recorded
        # path that no longer exists must NOT fall through, or it could
        # delete a same-named raw file belonging to another document.
        if meta.get("raw_path"):
            candidate = kb_dir / meta["raw_path"]
        else:
            candidate = raw_dir / name
        if candidate.exists():
            raw_path = candidate
            actions.append(("DELETE", str(candidate.relative_to(kb_dir))))

    # ----- Print the plan -----
    click.echo(f"Removing '{name}' (doc_name: {doc_name}, type: {doc_type or '?'}).")
    click.echo("")
    for tag, target in actions:
        click.echo(f"  {tag:<8} {target}")
    if concept_deletes:
        click.echo("")
        click.echo(
            f"  {len(concept_deletes)} concept(s) will be DELETED because this is their only source."
        )
        click.echo("  Pass --keep-empty to retain them instead.")
    if entity_deletes:
        click.echo("")
        click.echo(
            f"  {len(entity_deletes)} entity(s) will be DELETED because this is their only source."
        )
        click.echo("  Pass --keep-empty to retain them instead.")
    click.echo("")

    if dry_run:
        click.echo("(dry-run — nothing modified)")
        return

    if not yes:
        if not click.confirm("Proceed?", default=False):
            click.echo("Aborted.")
            return

    # ----- Execute -----
    # Ordering rationale: every step before the registry write is
    # idempotent (``unlink(missing_ok=True)``, ``shutil.rmtree(
    # ignore_errors=True)``, concept/index helpers that no-op on
    # already-clean state, and PageIndex's own delete-by-doc_id which
    # uses ``missing_ok`` + ``if dir.exists()`` internally). The
    # registry write is therefore the *commit point*: if anything
    # before it raises (including PageIndex), the entry plus its
    # ``doc_id`` survive and the user can simply re-run ``okforge
    # remove`` to retry from a clean slate.
    summary_path.unlink(missing_ok=True)
    source_md.unlink(missing_ok=True)
    source_json.unlink(missing_ok=True)
    if images_dir.is_dir():
        shutil.rmtree(images_dir, ignore_errors=True)

    concept_result = remove_doc_from_concept_pages(
        wiki_dir,
        doc_name,
        keep_empty=keep_empty,
    )

    entity_result = remove_doc_from_entity_pages(
        wiki_dir,
        doc_name,
        keep_empty=keep_empty,
    )

    remove_doc_from_index(
        wiki_dir, doc_name, concept_result["deleted"], entity_slugs_deleted=entity_result["deleted"]
    )

    # Strip dangling wikilinks now so a retry (after a PageIndex
    # failure below) finds a clean wiki — no point in re-running this
    # on every attempt.
    #
    # Scope: only the pages this remove actually touched (modified
    # concept + entity pages ∪ index.md). Previously this swept the whole
    # wiki via ``fix_broken_links(wiki_dir)``, which silently stripped
    # pre-existing dangling links in unrelated pages — see issue #58
    # (Bug 2). Users who want a wiki-wide sweep can still run
    # ``okforge lint --fix`` explicitly.
    lint_scope: list[Path] = [
        wiki_dir / "concepts" / f"{slug}.md" for slug in concept_result["modified"]
    ]
    lint_scope += [wiki_dir / "entities" / f"{slug}.md" for slug in entity_result["modified"]]
    index_md = wiki_dir / "index.md"
    if index_md.exists():
        lint_scope.append(index_md)
    files_changed, ghosts = fix_broken_links(wiki_dir, restrict_to=lint_scope)
    if files_changed:
        click.echo(f"  lint --fix cleaned {ghosts} dangling wikilink(s) in {files_changed} file(s)")

    # Free PageIndex's local managed state for long PDFs *before* the
    # registry write so the user can retry on failure — leaving the
    # entry intact preserves the ``doc_id`` we need for the second
    # attempt. PageIndex's local dedup is SHA-256 based, so a stale row
    # left behind here would silently re-bind on the next ``okforge
    # add`` and the user would get the old parse back without warning.
    if cleanup_pageindex:
        try:
            cleaned, msg = _cleanup_pageindex(
                kb_state_dir,
                kb_dir,
                doc_name,
                pageindex_doc_id,
            )
            click.echo(f"  PageIndex: {msg}")
        except Exception as exc:
            click.echo(
                f"  [WARN] PageIndex cleanup failed: {exc} "
                f"— registry entry kept; re-run `okforge remove {name}` to retry"
            )
            logging.getLogger(__name__).debug(
                "PageIndex cleanup traceback:",
                exc_info=True,
            )
            return

    # ----- Commit point -----
    # Prune by hash, not by ``doc_name``: legacy registry entries
    # (ingested before commit c504e26) carry only ``{name, type}`` and
    # would silently no-op under ``remove_by_doc_name``. See issue #58.
    registry.remove_by_hash(file_hash)

    if raw_path is not None:
        raw_path.unlink(missing_ok=True)

    append_log(wiki_dir, "remove", name)
    click.echo(f"  [OK] {name} removed from knowledge base.")


def _refresh_schema(wiki_dir: Path) -> bool:
    """Back up + overwrite ``wiki/AGENTS.md`` with the current ``AGENTS_MD``.

    If the on-disk schema differs from the bundled default, copy it to
    ``wiki/AGENTS.md.bak`` then overwrite with ``AGENTS_MD``. No-op when the
    file is missing or already identical. Returns True if it overwrote.
    """
    agents_file = wiki_dir / "AGENTS.md"
    if not agents_file.exists():
        # No-op when missing: get_agents_md() already falls back to the
        # bundled AGENTS_MD default at runtime, so there is nothing to refresh.
        return False
    current = agents_file.read_text(encoding="utf-8")
    if current == AGENTS_MD:
        return False
    backup = wiki_dir / "AGENTS.md.bak"
    backup.write_text(current, encoding="utf-8")
    click.echo(f"  Backed up existing schema to {backup.relative_to(wiki_dir.parent)}")
    agents_file.write_text(AGENTS_MD, encoding="utf-8")
    click.echo("  Refreshed wiki/AGENTS.md to the current schema.")
    return True


@cli.command()
@click.argument("doc_name", required=False)
@click.option(
    "--all", "all_docs", is_flag=True, default=False, help="Recompile every indexed document."
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List the docs that would be recompiled; no LLM calls, no writes.",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip the --all confirmation prompt."
)
@click.option(
    "--refresh-schema",
    "refresh_schema",
    is_flag=True,
    default=False,
    help="Overwrite wiki/AGENTS.md with the bundled schema (backs up "
    "the old one to AGENTS.md.bak) if it differs.",
)
@click.pass_context
@_with_kb_lock(exclusive=True)
def recompile(ctx, doc_name, all_docs, dry_run, yes, refresh_schema):
    """Re-run the current compile pipeline on already-indexed documents.

    Recompiling re-runs the same ``compile_short_doc`` / ``compile_long_doc``
    that ``okforge add`` uses, so pre-feature KBs gain the ``entities/`` layer
    and pages refresh to the current format. It does NOT re-run PageIndex or
    re-convert raw files — it reuses the on-disk ``wiki/sources/`` and
    ``wiki/summaries/`` content (and the registry's PageIndex ``doc_id``).

    DOC_NAME recompiles one doc (resolved like ``okforge remove`` — filename,
    slug, or unique substring). ``--all`` recompiles every indexed doc.
    Exactly one of DOC_NAME or ``--all`` is required.

    Side effect: this regenerates summaries (short docs) and rewrites concept
    pages with the current logic — manual edits to those pages are overwritten.
    """
    from okforge.state import HashRegistry

    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.")
        return

    if all_docs and doc_name:
        click.echo("Specify either a DOC_NAME or --all, not both.")
        return
    if not all_docs and not doc_name:
        click.echo("Specify a document name or pass --all to recompile every doc.")
        return

    kb_state_dir = state_dir(kb_dir)
    wiki_dir = kb_dir / "wiki"
    registry = HashRegistry(kb_state_dir / "hashes.json")

    # Resolve the set of docs to recompile.
    if all_docs:
        entries = list(registry.all_entries().values())
        if not entries:
            click.echo("No documents indexed yet. Run `okforge add` first.")
            return
        targets = entries
    else:
        matches = _resolve_doc_identifier(registry, doc_name)
        if not matches:
            click.echo(f"No document matching '{doc_name}' found in the KB.")
            click.echo("Try `okforge list` to see indexed documents.")
            return
        if len(matches) > 1:
            click.echo(f"'{doc_name}' matches multiple documents:")
            for _, m in matches:
                click.echo(f"  - {m.get('name', '?')}  (doc_name: {m.get('doc_name', '?')})")
            click.echo("Use a more specific name or the exact doc_name slug.")
            return
        targets = [matches[0][1]]

    def _classify(meta: dict) -> str:
        return "long" if _is_long_doc(meta) else "short"

    # --dry-run: enumerate only, no LLM calls, no writes.
    if dry_run:
        click.echo(f"Would recompile {len(targets)} document(s):")
        for meta in targets:
            name = meta.get("doc_name") or meta.get("name", "?")
            click.echo(f"  - {name}  ({_classify(meta)})")
        click.echo(
            "\nNote: recompiling regenerates summaries (short docs) and rewrites "
            "concept pages — manual edits would be overwritten."
        )
        click.echo("(dry-run — nothing modified)")
        return

    # --all confirmation (the summary/concept-regeneration side effect).
    if all_docs and not yes:
        click.echo(
            f"This will recompile {len(targets)} document(s), regenerating "
            "summaries and rewriting concept pages with the current logic.\n"
            "Manual edits to those pages will be overwritten."
        )
        if not click.confirm("Proceed?", default=False):
            click.echo("Aborted.")
            return

    if refresh_schema:
        _refresh_schema(wiki_dir)

    _setup_llm_key(kb_dir)
    config = load_config(kb_state_dir / "config.yaml")
    model: str = config.get("model", DEFAULT_CONFIG["model"])

    # Import lazily and reference via the module so tests can patch
    # ``okforge.agent.compiler.compile_*`` and see the call.
    from okforge.agent import compiler

    recompiled = 0
    skipped = 0
    total = len(targets)
    for i, meta in enumerate(targets, 1):
        name = meta.get("doc_name") or Path(meta.get("name", "")).stem
        if not name:
            click.echo(f"[{i}/{total}] [SKIP] registry entry has no doc_name.")
            skipped += 1
            continue

        if _is_long_doc(meta):
            summary_path = wiki_dir / "summaries" / f"{name}.md"
            doc_id = meta.get("doc_id")
            if not doc_id:
                click.echo(
                    f"[{i}/{total}] [SKIP] {name}: legacy long-doc entry without a "
                    "doc_id — re-add to refresh."
                )
                skipped += 1
                continue
            if not summary_path.exists():
                click.echo(
                    f"[{i}/{total}] [SKIP] {name}: missing summary at "
                    f"{summary_path.relative_to(kb_dir)}."
                )
                skipped += 1
                continue
            click.echo(f"[{i}/{total}] Recompiling long doc {name}...")
            start = time.time()
            try:
                asyncio.run(compiler.compile_long_doc(name, summary_path, doc_id, kb_dir, model))
            except Exception as exc:
                click.echo(f"  [ERROR] Compilation failed: {exc}")
                logging.getLogger(__name__).debug("Recompile traceback:", exc_info=True)
                skipped += 1
                continue
            click.echo(f"  [OK] {name} ({time.time() - start:.1f}s)")
            recompiled += 1
        else:
            source_path = wiki_dir / "sources" / f"{name}.md"
            if not source_path.exists():
                click.echo(
                    f"[{i}/{total}] [SKIP] {name}: missing source at "
                    f"{source_path.relative_to(kb_dir)}."
                )
                skipped += 1
                continue
            click.echo(f"[{i}/{total}] Recompiling short doc {name}...")
            start = time.time()
            try:
                asyncio.run(compiler.compile_short_doc(name, source_path, kb_dir, model))
            except Exception as exc:
                click.echo(f"  [ERROR] Compilation failed: {exc}")
                logging.getLogger(__name__).debug("Recompile traceback:", exc_info=True)
                skipped += 1
                continue
            click.echo(f"  [OK] {name} ({time.time() - start:.1f}s)")
            recompiled += 1

    click.echo(f"\nDone: recompiled {recompiled}, skipped {skipped}.")
    append_log(wiki_dir, "recompile", f"recompiled {recompiled}, skipped {skipped}")


@cli.command()
@click.option(
    "--resume",
    "-r",
    "resume",
    is_flag=False,
    flag_value="__latest__",
    default=None,
    metavar="[ID]",
    help="Resume the latest chat session, or a specific one by id or prefix.",
)
@click.option(
    "--list",
    "list_sessions_flag",
    is_flag=True,
    default=False,
    help="List chat sessions.",
)
@click.option(
    "--delete",
    "delete_id",
    default=None,
    metavar="ID",
    help="Delete a chat session by id or prefix.",
)
@click.option(
    "--no-color",
    "no_color",
    is_flag=True,
    default=False,
    help="Disable colored output.",
)
@click.option(
    "--raw",
    "raw",
    is_flag=True,
    default=False,
    help="Show raw markdown source instead of rendered output (keeps prompt and tool-call colors).",
)
@click.pass_context
def chat(ctx, resume, list_sessions_flag, delete_id, no_color, raw):
    """Start an interactive chat with the knowledge base."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.")
        return

    from okforge.agent.chat_session import (
        ChatSession,
        delete_session,
        list_sessions,
        load_session,
        relative_time,
        resolve_session_id,
    )

    if list_sessions_flag:
        sessions = list_sessions(kb_dir)
        if not sessions:
            click.echo("No chat sessions yet.")
            return
        click.echo(f"  {'ID':<22} {'TURNS':<6} {'UPDATED':<12} TITLE")
        click.echo(f"  {'-' * 22} {'-' * 6} {'-' * 12} {'-' * 30}")
        for s in sessions:
            rel = relative_time(s.get("updated_at", ""))
            title = s.get("title") or "(empty)"
            click.echo(f"  {s['id']:<22} {s['turn_count']:<6} {rel:<12} {title}")
        click.echo(f"\n{len(sessions)} session(s) in {state_dir(kb_dir) / 'chats'}")
        return

    if delete_id is not None:
        try:
            resolved = resolve_session_id(kb_dir, delete_id)
        except ValueError as exc:
            click.echo(f"[ERROR] {exc}")
            return
        if not resolved:
            click.echo(f"No matching session: {delete_id}")
            return
        if delete_session(kb_dir, resolved):
            click.echo(f"Deleted session {resolved}")
        else:
            click.echo(f"Could not delete session: {resolved}")
        return

    config = load_config(state_dir(kb_dir) / "config.yaml")
    _setup_llm_key(kb_dir)

    if resume is not None:
        try:
            resolved = resolve_session_id(kb_dir, resume)
        except ValueError as exc:
            click.echo(f"[ERROR] {exc}")
            return
        if not resolved:
            if resume == "__latest__":
                click.echo("No previous chat sessions to resume.")
            else:
                click.echo(f"No matching session: {resume}")
            return
        session = load_session(kb_dir, resolved)
    else:
        model: str = config.get("model", DEFAULT_CONFIG["model"])
        language: str = config.get("language", "en")
        session = ChatSession.new(kb_dir, model, language)

    from okforge.agent.chat import run_chat

    try:
        asyncio.run(run_chat(kb_dir, session, no_color=no_color, raw=raw))
    except Exception as exc:
        click.echo(f"[ERROR] Chat failed: {exc}")


@cli.command()
@click.pass_context
def watch(ctx):
    """Watch the raw/ directory for new documents and process them automatically."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.")
        return

    from okforge.watcher import watch_directory

    raw_dir = kb_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    def on_new_files(paths):
        for p in paths:
            fp = Path(p)
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                click.echo(
                    f"Skipping unsupported file type: {fp.suffix}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
                continue
            add_single_file(fp, kb_dir)

    click.echo(f"Watching {raw_dir} for new documents. Press Ctrl+C to stop.")
    watch_directory(raw_dir, on_new_files)


async def run_lint(kb_dir: Path) -> Path | None:
    """Run structural + knowledge lint, write report, return report path.

    Returns ``None`` if the KB has no indexed documents (nothing to lint).
    Async because knowledge lint uses an LLM agent. Usable from CLI
    (via ``asyncio.run``) and directly from the chat REPL.
    """
    from okforge.lint import run_structural_lint
    from okforge.agent.linter import run_knowledge_lint

    kb_state_dir = state_dir(kb_dir)

    with kb_read_lock(kb_state_dir):
        # Skip lint entirely when the KB has no indexed documents
        hashes_file = kb_state_dir / "hashes.json"
        if hashes_file.exists():
            hashes = json.loads(hashes_file.read_text(encoding="utf-8"))
        else:
            hashes = {}
        if not hashes:
            click.echo("Nothing to lint — no documents indexed yet. Run `okforge add` first.")
            return

        config = load_config(kb_state_dir / "config.yaml")
        _setup_llm_key(kb_dir)
        model: str = config.get("model", DEFAULT_CONFIG["model"])

        click.echo("Running structural lint...")
        structural_report = run_structural_lint(kb_dir)
        click.echo(structural_report)

        click.echo("Running knowledge lint...")
        try:
            knowledge_report = await run_knowledge_lint(kb_dir, model)
        except Exception as exc:
            knowledge_report = f"Knowledge lint failed: {exc}"
        click.echo(knowledge_report)

    # Write combined report
    with kb_ingest_lock(kb_state_dir):
        reports_dir = kb_dir / "wiki" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"lint_{timestamp}.md"
        report_content = f"# Lint Report — {timestamp}\n\n## Structural\n\n{structural_report}\n\n## Semantic\n\n{knowledge_report}\n"
        report_path.write_text(report_content, encoding="utf-8")
        append_log(kb_dir / "wiki", "lint", f"report → {report_path.name}")
    click.echo(f"\nReport written to {report_path}")
    return report_path


@cli.command()
@click.option(
    "--fix",
    is_flag=True,
    default=False,
    help="Rewrite broken [[wikilinks]] in place (fuzzy match) or "
    "strip to plain text when no match. Runs before the report.",
)
@click.pass_context
def lint(ctx, fix):
    """Lint the knowledge base for structural and semantic inconsistencies."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.")
        return
    if fix:
        from okforge.lint import fix_broken_links

        with kb_ingest_lock(state_dir(kb_dir)):
            files_changed, ghosts = fix_broken_links(kb_dir / "wiki")
        if files_changed:
            click.echo(f"Fixed {ghosts} wikilink(s) across {files_changed} file(s).")
        else:
            click.echo("Nothing to fix — all wikilinks resolve.")
    asyncio.run(run_lint(kb_dir))


@cli.command()
@click.pass_context
def reindex(ctx):
    """Build the concept topic tree from the existing flat wiki/concepts/ (experimental).

    No-op unless `topic_tree: true` is set in .okforge/config.yaml.
    """
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.")
        return
    kb_state_dir = state_dir(kb_dir)
    config = load_config(kb_state_dir / "config.yaml")
    if not bool(config.get("topic_tree", False)):
        click.echo(
            "topic_tree is not enabled. Set `topic_tree: true` in .okforge/config.yaml first."
        )
        return
    _setup_llm_key(kb_dir)
    model = config.get("model", DEFAULT_CONFIG["model"])
    from okforge.topic_tree_llm import make_cluster, make_summarize

    concepts_root = kb_dir / "wiki" / "concepts"
    with kb_ingest_lock(kb_state_dir):
        n = tt_bootstrap(
            concepts_root,
            cluster=make_cluster(model),
            summarize=make_summarize(model),
        )
    from okforge.okf import retarget_md_links

    moved_links = retarget_md_links(kb_dir / "wiki")
    click.echo(
        f"Reindexed {n} concept(s) into the topic tree; "
        f"retargeted {moved_links} markdown link(s) to moved pages."
    )


@cli.command()
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open the graph in your browser after generating (default: on; --no-open for headless).",
)
@click.pass_context
@_with_kb_lock(exclusive=False)
def visualize(ctx, open_browser):
    """Render the wiki's [[wikilink]] graph as a self-contained interactive HTML page."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.")
        return
    from okforge import visualize as viz

    graph = viz.build_graph(kb_dir / "wiki")
    if not graph["nodes"]:
        click.echo("No wiki pages to visualize yet. Run `okforge add` first.")
        return
    out = kb_dir / "output" / "visualize" / "graph.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(viz.render_html(graph), encoding="utf-8")
    click.echo(
        f"Graph written to {out}  ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)"
    )
    if open_browser:
        import webbrowser

        try:
            opened = webbrowser.open(
                out.resolve().as_uri()
            )  # resolve() so a relative --kb-dir still yields a valid file URI
        except Exception:
            opened = False
        if not opened:
            click.echo(
                "(couldn't launch a browser — open the file above manually, or use --no-open)"
            )


def collect_list_data(kb_dir: Path) -> dict:
    """Machine-readable `list` payload: registry documents + wiki page stems.

    This is the stable interface scripts and the webui consume via
    ``list --json`` — prefer extending it over parsing the human tables.
    """
    hashes_file = state_dir(kb_dir) / "hashes.json"
    hashes: dict = {}
    if hashes_file.exists():
        hashes = json.loads(hashes_file.read_text(encoding="utf-8"))

    documents = []
    for meta in hashes.values():
        raw_type = meta.get("type", "unknown")
        documents.append(
            {
                "name": meta.get("name", "unknown"),
                "doc_name": meta.get("doc_name") or Path(meta.get("name", "")).stem,
                "type": raw_type,
                "display_type": _display_type(raw_type),
                "pages": meta.get("pages") or None,
            }
        )

    wiki_dir = kb_dir / "wiki"

    def _stems(subdir: str) -> list[str]:
        # rglob: topic-tree KBs nest concepts; _topic.md are tree nodes
        d = wiki_dir / subdir
        if not d.exists():
            return []
        return sorted(p.stem for p in d.rglob("*.md") if p.name != "_topic.md")

    config = load_config(state_dir(kb_dir) / "config.yaml")

    return {
        "kb_dir": str(kb_dir),
        "description": config.get("description") or None,
        "documents": documents,
        "summaries": _stems("summaries"),
        "concepts": _stems("concepts"),
        "entities": _stems("entities"),
        "reports": sorted(p.name for p in (wiki_dir / "reports").glob("*.md"))
        if (wiki_dir / "reports").exists()
        else [],
    }


def print_list(kb_dir: Path) -> None:
    """Print all documents in the knowledge base. Usable from CLI and chat REPL."""
    hashes_file = state_dir(kb_dir) / "hashes.json"
    if not hashes_file.exists():
        click.echo("No documents indexed yet.")
        return

    hashes = json.loads(hashes_file.read_text(encoding="utf-8"))
    if not hashes:
        click.echo("No documents indexed yet.")
        return

    # Display documents table with count in header
    doc_count = len(hashes)
    click.echo(f"Documents ({doc_count}):")
    click.echo(f"  {'Name':<40} {'Type':<12} {'Pages':<8}")
    click.echo(f"  {'-' * 40} {'-' * 12} {'-' * 8}")
    for file_hash, meta in hashes.items():
        name = meta.get("name", "unknown")
        raw_type = meta.get("type", "unknown")
        display = _display_type(raw_type)
        pages = meta.get("pages", "")
        pages_str = str(pages) if pages else ""
        click.echo(f"  {name:<40} {display:<12} {pages_str:<8}")

    # Display summaries
    summaries_dir = kb_dir / "wiki" / "summaries"
    if summaries_dir.exists():
        summaries = sorted(p.stem for p in summaries_dir.glob("*.md"))
        if summaries:
            click.echo(f"\nSummaries ({len(summaries)}):")
            for s in summaries:
                click.echo(f"  - {s}")

    # Display concepts
    concepts_dir = kb_dir / "wiki" / "concepts"
    if concepts_dir.exists():
        concepts = sorted(p.stem for p in concepts_dir.glob("*.md"))
        if concepts:
            click.echo(f"\nConcepts ({len(concepts)}):")
            for c in concepts:
                click.echo(f"  - {c}")

    # Display entities
    entities_dir = kb_dir / "wiki" / "entities"
    if entities_dir.exists():
        entities = sorted(p.stem for p in entities_dir.glob("*.md"))
        if entities:
            click.echo(f"\nEntities ({len(entities)}):")
            for e in entities:
                click.echo(f"  - {e}")

    # Display reports
    reports_dir = kb_dir / "wiki" / "reports"
    if reports_dir.exists():
        reports = sorted(p.name for p in reports_dir.glob("*.md"))
        if reports:
            click.echo(f"\nReports ({len(reports)}):")
            for r in reports:
                click.echo(f"  - {r}")


@cli.command(name="list")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit machine-readable JSON.")
@click.pass_context
@_with_kb_lock(exclusive=False)
def list_cmd(ctx, as_json):
    """List all documents in the knowledge base."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        if as_json:
            click.echo(json.dumps({"error": "no_kb"}))
            ctx.exit(1)
        click.echo("No knowledge base found. Run `okforge init` first.")
        return
    if as_json:
        click.echo(json.dumps(collect_list_data(kb_dir), ensure_ascii=False))
        return
    print_list(kb_dir)


def collect_status_data(kb_dir: Path) -> dict:
    """Machine-readable `status` payload (counts + ISO-8601 timestamps)."""
    import datetime

    wiki_dir = kb_dir / "wiki"

    def _iso(ts: float) -> str:
        return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")

    counts = {}
    for subdir in ("sources", "summaries", "concepts", "entities", "reports"):
        path = wiki_dir / subdir
        counts[subdir] = len(list(path.glob("*.md"))) if path.exists() else 0

    raw_dir = kb_dir / "raw"
    counts["raw"] = len([f for f in raw_dir.iterdir() if f.is_file()]) if raw_dir.exists() else 0

    hashes_file = state_dir(kb_dir) / "hashes.json"
    total_indexed = 0
    if hashes_file.exists():
        total_indexed = len(json.loads(hashes_file.read_text(encoding="utf-8")))

    compiled_pages = [
        p
        for sub in PAGE_CONTENT_DIRS
        for p in (wiki_dir / sub).glob("*.md")
        if (wiki_dir / sub).exists()
    ]
    last_compile = _iso(max(p.stat().st_mtime for p in compiled_pages)) if compiled_pages else None

    reports = list((wiki_dir / "reports").glob("*.md")) if (wiki_dir / "reports").exists() else []
    last_lint = _iso(max(p.stat().st_mtime for p in reports)) if reports else None

    return {
        "kb_dir": str(kb_dir),
        "counts": counts,
        "total_indexed": total_indexed,
        "last_compile": last_compile,
        "last_lint": last_lint,
    }


def print_status(kb_dir: Path) -> None:
    """Print knowledge base status. Usable from CLI and chat REPL."""
    wiki_dir = kb_dir / "wiki"
    subdirs = ["sources", "summaries", "concepts", "entities", "reports"]

    # Print the active KB path as the first line. Agents and scripts
    # parse this to locate the wiki without assuming cwd == KB root.
    click.echo(f"Knowledge base: {kb_dir}")
    click.echo("")
    click.echo("Knowledge Base Status:")
    click.echo(f"  {'Directory':<20} {'Files':<10}")
    click.echo(f"  {'-' * 20} {'-' * 10}")

    for subdir in subdirs:
        path = wiki_dir / subdir
        if path.exists():
            count = len(list(path.glob("*.md")))
        else:
            count = 0
        click.echo(f"  {subdir:<20} {count:<10}")

    # Raw files
    raw_dir = kb_dir / "raw"
    if raw_dir.exists():
        raw_count = len([f for f in raw_dir.iterdir() if f.is_file()])
        click.echo(f"  {'raw':<20} {raw_count:<10}")

    # Hash registry summary
    hashes_file = state_dir(kb_dir) / "hashes.json"
    if hashes_file.exists():
        hashes = json.loads(hashes_file.read_text(encoding="utf-8"))
        click.echo(f"\n  Total indexed: {len(hashes)} document(s)")

    # Last compile time: newest compiled page across summaries/, concepts/,
    # and entities/ (an entity-only compile must still bump the shown time).
    compiled_pages = [
        p
        for sub in PAGE_CONTENT_DIRS
        for p in (wiki_dir / sub).glob("*.md")
        if (wiki_dir / sub).exists()
    ]
    if compiled_pages:
        newest_page = max(compiled_pages, key=lambda p: p.stat().st_mtime)
        import datetime

        mtime = datetime.datetime.fromtimestamp(newest_page.stat().st_mtime)
        click.echo(f"  Last compile:  {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Last lint time: newest file in wiki/reports/
    reports_dir = wiki_dir / "reports"
    if reports_dir.exists():
        reports = list(reports_dir.glob("*.md"))
        if reports:
            newest_report = max(reports, key=lambda p: p.stat().st_mtime)
            import datetime

            mtime = datetime.datetime.fromtimestamp(newest_report.stat().st_mtime)
            click.echo(f"  Last lint:     {mtime.strftime('%Y-%m-%d %H:%M:%S')}")


@cli.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit machine-readable JSON.")
@click.pass_context
@_with_kb_lock(exclusive=False)
def status(ctx, as_json):
    """Show the current status of the knowledge base.

    Output starts with a ``Knowledge base: <path>`` line so agents and
    scripts can locate the wiki without assuming cwd == KB root.
    """
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        if as_json:
            click.echo(json.dumps({"error": "no_kb"}))
            ctx.exit(1)
        click.echo("No knowledge base found. Run `okforge init` first.")
        return
    if as_json:
        click.echo(json.dumps(collect_status_data(kb_dir), ensure_ascii=False))
        return
    print_status(kb_dir)


@cli.command()
@click.argument("text", required=False)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit machine-readable JSON.")
@click.pass_context
def describe(ctx, text, as_json):
    """Show or set the project-level description.

    With TEXT, saves it to .okforge/config.yaml as the curated one-liner for
    what this whole project covers — preferred by consumers (e.g. MCP
    list_projects) over guessing from the first few document summaries,
    which misleads once a project holds many books.
    """
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        if as_json:
            click.echo(json.dumps({"error": "no_kb"}))
            ctx.exit(1)
        click.echo("No knowledge base found. Run `okforge init` first.")
        return
    config_path = state_dir(kb_dir) / "config.yaml"
    config = load_config(config_path)
    if text:
        config["description"] = text.strip()
        save_config(config_path, config)
    description = config.get("description") or None
    if as_json:
        click.echo(json.dumps({"description": description}, ensure_ascii=False))
    else:
        click.echo(description or "(no description set)")


@cli.command(name="okf-lint")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit machine-readable JSON.")
@click.pass_context
@_with_kb_lock(exclusive=False)
def okf_lint(ctx, as_json):
    """Check the wiki bundle for Open Knowledge Format conformance.

    Fast, structural, no LLM: frontmatter present and typed on every page,
    reserved files (index.md / log.md) present and well-formed. Exit code 1
    when issues are found.
    """
    from okforge.okf import okf_check

    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        if as_json:
            click.echo(json.dumps({"error": "no_kb"}))
            ctx.exit(1)
        click.echo("No knowledge base found. Run `okforge init` first.")
        return
    issues = okf_check(kb_dir / "wiki")
    if as_json:
        click.echo(json.dumps({"issues": issues, "conformant": not issues}, ensure_ascii=False))
    elif issues:
        click.echo(f"OKF conformance: {len(issues)} issue(s)")
        for issue in issues:
            click.echo(f"  - {issue}")
    else:
        click.echo("OKF conformance: clean")
    if issues:
        ctx.exit(1)


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------

_FEEDBACK_REPO = "designcomputer/okforge"
_FEEDBACK_TYPES = ("bug", "feature", "question", "other")
_FEEDBACK_LABEL_MAP = {
    "bug": "bug",
    "feature": "enhancement",
    "question": "question",
    "other": "",
}


def _openkb_version() -> str:
    """Return the installed okforge package version.

    Delegates to ``okforge.__version__`` so the chat REPL, feedback issue
    body, and any future caller all surface the same fallback string
    (``0.0.0+unknown`` from ``okforge/__init__.py``). Mirrors
    ``okforge.agent.chat._openkb_version``.
    """
    from okforge import __version__

    return __version__


def _collect_feedback_diagnostics(ctx) -> dict[str, str]:
    """Auto-collect non-sensitive environment info to attach to a feedback
    issue. Kept deliberately small — no paths, no API keys, no usernames.
    """
    import platform

    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override") if ctx.obj else None)
    return {
        "okforge": _openkb_version(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "kb_initialised": "yes" if kb_dir else "no",
    }


def _build_feedback_url(
    message: str,
    feedback_type: str,
    diagnostics: dict[str, str],
) -> str:
    """Build a GitHub issue URL with title / body / labels prefilled."""
    from urllib.parse import urlencode

    first_line = message.splitlines()[0] if message else ""
    truncated = first_line[:60] + ("…" if len(first_line) > 60 else "")
    title_prefix = f"[{feedback_type}] " if feedback_type != "other" else ""
    title = f"{title_prefix}{truncated}" if truncated else f"{title_prefix}Feedback from CLI"

    if diagnostics:
        diag_block = "\n".join(f"- **{k}**: {v}" for k, v in diagnostics.items())
        body = (
            f"{message}\n\n"
            "---\n\n"
            "<details>\n"
            "<summary>Diagnostics (auto-collected by <code>okforge feedback</code>)</summary>\n\n"
            f"{diag_block}\n"
            "</details>\n"
        )
    else:
        body = message

    params = {"title": title, "body": body}
    label = _FEEDBACK_LABEL_MAP.get(feedback_type, "")
    if label:
        params["labels"] = label

    return f"https://github.com/{_FEEDBACK_REPO}/issues/new?{urlencode(params)}"


@cli.command()
@click.argument("message", required=False)
@click.option(
    "--type",
    "feedback_type",
    type=click.Choice(_FEEDBACK_TYPES),
    default=None,
    help="Feedback type — sets the GitHub issue label.",
)
@click.pass_context
def feedback(ctx, message, feedback_type):
    """Submit feedback by opening a prefilled GitHub issue.

    Examples:

      \b
      okforge feedback                              # interactive
      okforge feedback "okforge add hangs on .docx"  # one-line bug report
      okforge feedback --type feature "..."         # tags the issue 'enhancement'

    The command does not send anything to okforge maintainers directly —
    it opens GitHub in your browser with title, body, and label prefilled.
    You log in with your own GitHub account and submit the issue.
    """
    if not message:
        click.echo(
            "What's your feedback? End with an empty line + Ctrl-D "
            "(Unix) or Ctrl-Z+Enter (Windows). Ctrl-C cancels."
        )
        message = sys.stdin.read().strip()

    if not message:
        click.echo("No feedback provided. Aborted.")
        ctx.exit(1)
        return

    if feedback_type is None:
        # Skip the prompt in non-TTY contexts (CI / piped stdin) so
        # ``echo "msg" | okforge feedback`` doesn't hang on the second
        # prompt after consuming all piped input for the message body.
        # Mirrors the ``_stdin_is_tty()`` gate added in PR #48.
        if _stdin_is_tty():
            feedback_type = click.prompt(
                "Type",
                default="other",
                type=click.Choice(_FEEDBACK_TYPES),
                show_default=True,
                show_choices=True,
            )
        else:
            feedback_type = "other"

    diagnostics = _collect_feedback_diagnostics(ctx)
    url = _build_feedback_url(message, feedback_type, diagnostics)

    click.echo("Copy this URL into a browser if the auto-open below fails:")
    click.echo(f"  {url}")

    import webbrowser

    try:
        opened = webbrowser.open(url)
    except Exception as exc:
        # webbrowser.open rarely raises but be defensive — the printed URL
        # above is the fallback path.
        click.echo(f"  (browser auto-open failed: {exc})", err=True)
        return

    # ``webbrowser.open`` returns False on headless boxes (no GUI, no
    # ``BROWSER`` env) without raising. Without this check we'd silently
    # print "Opened" and the user would think the issue was filed.
    if opened:
        click.echo("Opened GitHub in your browser.")
    else:
        click.echo(
            "  (no browser available — copy the URL above to file the issue)",
            err=True,
        )


# ---------------------------------------------------------------------------
# `okforge skill ...` — skill factory (v0.1)
# ---------------------------------------------------------------------------


@cli.group()
def skill():
    """Compile knowledge into a redistributable Anthropic Skill."""


@skill.command("new")
@click.argument("name")
@click.argument("intent")
@click.option(
    "-y",
    "--yes",
    "yes_flag",
    is_flag=True,
    default=False,
    help="Overwrite existing output/skills/<name>/ without prompting.",
)
@click.pass_context
def skill_new(ctx, name, intent, yes_flag):
    """Compile a new skill from this KB's wiki.

    NAME is a kebab-case slug used for the output directory and skill name.
    INTENT is a natural-language description of what this skill should do.

    Example:

      okforge skill new karpathy-thinking "Reason about transformers like Karpathy"
    """
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.", err=True)
        ctx.exit(1)

    err = _preflight_skill_new(kb_dir, name)
    if err:
        click.echo(f"[ERROR] {err}", err=True)
        ctx.exit(1)

    # Verify LLM key + load config BEFORE touching existing output. Any
    # failure here (missing API key, malformed config) must leave the old
    # skill directory intact — we can't replace it if we can't proceed.
    try:
        _setup_llm_key(kb_dir)
    except RuntimeError as exc:
        click.echo(f"[ERROR] {exc}", err=True)
        ctx.exit(1)
    config = load_config(state_dir(kb_dir) / "config.yaml")
    model = config.get("model", DEFAULT_CONFIG["model"])

    # Overwrite handling (CLI-specific). Done AFTER key/config so a
    # missing key doesn't wipe the user's existing skill output.
    #
    # When overwriting, we don't destroy the old skill — we copy it
    # into <kb>/output/skills/<name>-workspace/iteration-N/ first, so
    # the user can roll back via `okforge skill rollback`. See
    # ``okforge/skill/workspace.py``.
    from okforge.skill import skill_dir
    from okforge.skill.workspace import save_iteration, write_diff

    target = skill_dir(kb_dir, name)
    saved_iteration: Path | None = None
    if target.exists():
        if yes_flag:
            saved_iteration = save_iteration(kb_dir, name)
            _clear_existing_skill_dir(kb_dir, name)
        elif sys.stdin.isatty():
            if not click.confirm(
                f"output/skills/{name}/ already exists. Overwrite?",
                default=False,
            ):
                click.echo("Aborted.")
                ctx.exit(1)
            saved_iteration = save_iteration(kb_dir, name)
            _clear_existing_skill_dir(kb_dir, name)
        else:
            click.echo(
                f"[ERROR] output/skills/{name}/ exists. Pass -y to overwrite "
                f"in non-interactive contexts.",
                err=True,
            )
            ctx.exit(1)

    # Run the generator. Generator.run handles compile -> validate ->
    # marketplace publish, so both CLI and chat get the same quality gate.
    from okforge.skill.generator import Generator

    click.echo(f"Compiling skill '{name}'...")
    gen = Generator(
        target_type="skill",
        name=name,
        intent=intent,
        kb_dir=kb_dir,
        model=model,
    )
    try:
        asyncio.run(gen.run())
    except RuntimeError as exc:
        click.echo(f"[ERROR] {exc}", err=True)
        ctx.exit(1)

    # Drop a structural diff inside the saved iteration so the user
    # can see what changed since the previous compile.
    if saved_iteration is not None:
        try:
            write_diff(saved_iteration, target, saved_iteration / "diff.md")
        except Exception as exc:  # diff is best-effort; never block success
            logging.getLogger(__name__).debug("diff generation failed: %s", exc, exc_info=True)

    # Surface validation issues. Don't block — files are on disk and
    # the user can fix or rollback.
    result = gen.validation
    if result is not None and (result.errors or result.warnings):
        click.echo("\n[WARN] Validation found issues:")
        for err in result.errors:
            click.echo(f"  ERROR:   {err}")
        for warn in result.warnings:
            click.echo(f"  WARN:    {warn}")
        click.echo(
            f"\nRun `okforge skill validate {name}` to re-check, or "
            f"`okforge skill rollback {name}` to revert."
        )

    click.echo(f"\nSaved: output/skills/{name}/")
    if saved_iteration is not None:
        rel = saved_iteration.relative_to(kb_dir)
        click.echo(f"Previous version: {rel}/  (run `okforge skill rollback {name}` to restore)")
    click.echo("Manifest: .claude-plugin/marketplace.json updated")
    click.echo("\nInstall locally:")
    click.echo(f"  cp -r output/skills/{name} ~/.claude/skills/")
    click.echo("\nShare (push KB to GitHub, then):")
    click.echo("  npx skills@latest add <owner>/<repo>")


@skill.command("history")
@click.argument("name")
@click.pass_context
def skill_history(ctx, name):
    """List previous iterations of a skill."""
    import datetime as _dt

    from okforge.skill.workspace import list_iterations

    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.", err=True)
        ctx.exit(1)

    err = _validate_skill_name(name)
    if err:
        click.echo(f"[ERROR] {err}", err=True)
        ctx.exit(1)

    iters = list_iterations(kb_dir, name)
    if not iters:
        click.echo(f"No previous iterations for '{name}'.")
        return

    click.echo(f"Iterations of '{name}' ({len(iters)} total):\n")
    click.echo("  N  Path                                                  Created")
    click.echo("  -  --------------------------------------------------    -------")
    for path in iters:
        n = int(path.name.split("-", 1)[1])
        rel = path.relative_to(kb_dir)
        try:
            mtime = _dt.datetime.fromtimestamp(path.stat().st_mtime)
            stamp = mtime.strftime("%Y-%m-%d %H:%M")
        except OSError:
            stamp = "-"
        click.echo(f"  {n}  {rel}  {stamp}")

    from okforge.skill import skill_dir

    current = skill_dir(kb_dir, name)
    if current.is_dir():
        rel_curr = current.relative_to(kb_dir)
        click.echo(f"\n  Current: {rel_curr}/")

    latest_n = int(iters[-1].name.split("-", 1)[1])
    click.echo("\nRestore an iteration:")
    click.echo(f"  okforge skill rollback {name}          # restore latest (iteration-{latest_n})")
    click.echo(f"  okforge skill rollback {name} --to 1   # restore iteration-1")


@skill.command("rollback")
@click.argument("name")
@click.option(
    "--to",
    "to_n",
    default=None,
    type=int,
    help="Iteration number to restore. Defaults to latest.",
)
@click.option(
    "-y",
    "--yes",
    "yes_flag",
    is_flag=True,
    default=False,
    help="Skip confirmation.",
)
@click.pass_context
def skill_rollback(ctx, name, to_n, yes_flag):
    """Restore a previous iteration as the current skill."""
    from okforge.skill.marketplace import regenerate_marketplace
    from okforge.skill.workspace import list_iterations, restore_iteration

    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.", err=True)
        ctx.exit(1)

    err = _validate_skill_name(name)
    if err:
        click.echo(f"[ERROR] {err}", err=True)
        ctx.exit(1)

    iters = list_iterations(kb_dir, name)
    if not iters:
        click.echo(
            f"[ERROR] No iterations exist for '{name}'. Nothing to roll back.",
            err=True,
        )
        ctx.exit(1)

    target_n = to_n if to_n is not None else int(iters[-1].name.split("-", 1)[1])
    target_label = f"iteration-{target_n}"
    if not any(p.name == target_label for p in iters):
        click.echo(
            f"[ERROR] Iteration {target_n} not found for '{name}'. "
            f"Run `okforge skill history {name}` to see available iterations.",
            err=True,
        )
        ctx.exit(1)

    from okforge.skill import skill_dir

    current = skill_dir(kb_dir, name)
    if current.exists():
        prompt = f"This will overwrite output/skills/{name}/ with {target_label}. Continue?"
        if yes_flag:
            pass
        elif sys.stdin.isatty():
            if not click.confirm(prompt, default=False):
                click.echo("Aborted.")
                ctx.exit(1)
        else:
            click.echo(
                f"[ERROR] output/skills/{name}/ exists. Pass -y to overwrite "
                f"in non-interactive contexts.",
                err=True,
            )
            ctx.exit(1)

    try:
        restore_iteration(kb_dir, name, n=to_n)
    except FileNotFoundError as exc:
        click.echo(f"[ERROR] {exc}", err=True)
        ctx.exit(1)

    regenerate_marketplace(kb_dir)
    click.echo(f"Restored output/skills/{name}/ from {target_label}.")
    click.echo("Manifest: .claude-plugin/marketplace.json updated")


@skill.command("validate")
@click.argument("name", required=False)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat warnings as failures (exit non-zero).",
)
@click.pass_context
def skill_validate(ctx, name, strict):
    """Validate one skill (by name) or all compiled skills in this KB."""
    from okforge.skill import skill_dir, skills_root
    from okforge.skill.validator import validate_skill

    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.", err=True)
        ctx.exit(1)

    root = skills_root(kb_dir)
    if not root.is_dir():
        click.echo("No skills found. Compile one with `okforge skill new`.")
        return

    if name:
        target = skill_dir(kb_dir, name)
        if not target.is_dir():
            click.echo(f"[ERROR] Skill '{name}' not found.", err=True)
            ctx.exit(1)
        targets = [target]
    else:
        targets = sorted(
            d for d in root.iterdir() if d.is_dir() and not d.name.endswith("-workspace")
        )

    any_failed = False
    for t in targets:
        result = validate_skill(t, strict=strict)
        passed = result.passed_strict if strict else result.passed
        prefix = "[OK]" if passed else "[FAIL]"
        click.echo(f"{prefix} {t.name}")
        for err in result.errors:
            click.echo(f"  ERROR:   {err}")
        for warn in result.warnings:
            click.echo(f"  WARN:    {warn}")
        if not passed:
            any_failed = True

    if any_failed:
        ctx.exit(1)


@skill.command("eval")
@click.argument("name")
@click.option(
    "--save",
    "save_flag",
    is_flag=True,
    default=False,
    help="Persist the generated eval set to .okforge/eval-sets/<name>.json",
)
@click.option(
    "--eval-set",
    "eval_set_path",
    default=None,
    type=click.Path(),
    help="Use a saved eval set instead of generating fresh prompts.",
)
@click.option(
    "--count",
    default=10,
    type=int,
    help="Number of should-trigger + should-not prompts (each).",
)
@click.pass_context
def skill_eval(ctx, name, save_flag, eval_set_path, count):
    """Measure how accurately a compiled skill's description fires.

    Generates trigger-eval prompts via LLM, then asks a grader LLM whether
    the description should activate the skill for each prompt. Prints pass
    rate + miss list.
    """
    from okforge.skill.evaluator import (
        run_eval,
        save_eval_set,
        load_eval_set,
        EvalPrompt,
    )

    from okforge.skill import skill_dir as _skill_dir

    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.", err=True)
        ctx.exit(1)

    skill_dir = _skill_dir(kb_dir, name)
    if not skill_dir.is_dir():
        click.echo(f"[ERROR] Skill '{name}' not found.", err=True)
        ctx.exit(1)

    try:
        _setup_llm_key(kb_dir)
    except RuntimeError as exc:
        click.echo(f"[ERROR] {exc}", err=True)
        ctx.exit(1)
    config = load_config(state_dir(kb_dir) / "config.yaml")
    model = config.get("model", DEFAULT_CONFIG["model"])

    eval_set: list[EvalPrompt] | None = None
    if eval_set_path:
        eval_set = load_eval_set(Path(eval_set_path))
        click.echo(f"Loaded eval set from {eval_set_path} ({len(eval_set)} prompts).")
    else:
        click.echo(f"Generating eval set for '{name}' (count={count} per side)...")

    try:
        result = asyncio.run(
            run_eval(
                skill_dir,
                model=model,
                eval_set=eval_set,
                count=count,
            )
        )
    except RuntimeError as exc:
        click.echo(f"[ERROR] {exc}", err=True)
        ctx.exit(1)

    click.echo(f"\nEval set: {result.total} prompts")
    click.echo(
        f"Trigger accuracy: {result.passed}/{result.trigger_scored} "
        f"({result.pass_rate * 100:.0f}%)  "
        f"— does the description fire on the right questions?"
    )
    coverage_scored = (
        result.trigger_questions - len(result.coverage_ambiguous) - len(result.coverage_errors)
    )
    click.echo(
        f"Body coverage:    {result.coverage_passed}/{coverage_scored} "
        f"({result.coverage_rate * 100:.0f}%)  "
        f"— does SKILL.md actually support what the description promises?"
    )

    if result.misses:
        click.echo(f"\nTrigger misses ({len(result.misses)}):")
        for miss in result.misses:
            click.echo(f"  - {miss.label} {miss.prompt.question}")

    if result.coverage_misses:
        click.echo(f"\nCoverage gaps ({len(result.coverage_misses)}):")
        for gap in result.coverage_misses:
            tail = f" — {gap.reason}" if gap.reason else ""
            click.echo(f"  - {gap.prompt.question}{tail}")

    if result.coverage_ambiguous:
        click.echo(
            f"\n[WARN] Coverage grader returned unparseable output on "
            f"{len(result.coverage_ambiguous)} prompt(s) — excluded from "
            f"the body-coverage score. Try a more capable model:"
        )
        for amb in result.coverage_ambiguous:
            tail = f" — {amb.reason}" if amb.reason else ""
            click.echo(f"  - {amb.prompt.question}{tail}")

    if result.trigger_errors or result.coverage_errors:
        click.echo(
            f"\n[WARN] {len(result.trigger_errors)} trigger and "
            f"{len(result.coverage_errors)} coverage grader call(s) "
            f"failed and are excluded from the scores above:"
        )
        for err in result.trigger_errors:
            click.echo(f"  - trigger:  {err.prompt.question} — {err.reason}")
        for err in result.coverage_errors:
            click.echo(f"  - coverage: {err.prompt.question} — {err.reason}")

    if (
        not result.misses
        and not result.coverage_misses
        and not result.coverage_ambiguous
        and not result.trigger_errors
        and not result.coverage_errors
    ):
        click.echo("\nAll prompts graded correctly with full body support.")

    if save_flag and eval_set is None:
        path = save_eval_set(kb_dir, name, result.prompts)
        click.echo(f"\nEval set persisted to {path}")


# ---------------------------------------------------------------------------
# `okforge deck ...` — deck factory (v0.2)
# ---------------------------------------------------------------------------


@cli.group()
def deck():
    """Generate a polished single-file HTML slide deck from the wiki."""


@deck.command("new")
@click.argument("name")
@click.argument("intent")
@click.option(
    "-y",
    "--yes",
    "yes_flag",
    is_flag=True,
    default=False,
    help="Overwrite existing output/decks/<name>/ without prompting.",
)
@click.option(
    "--critique",
    "critique_flag",
    is_flag=True,
    default=False,
    help="Opt-in second-pass review via a critic agent (slower, higher quality).",
)
@click.option(
    "--skill",
    "skill_name",
    metavar="SKILL_NAME",
    default=None,
    # NOTE: 'openkb-deck-neon' below must stay in sync with
    # DEFAULT_DECK_SKILL in okforge/deck/creator.py.
    help=(
        "Which deck skill to use. Defaults to 'openkb-deck-neon' "
        "(the built-in). Pass e.g. 'deck-guizang-editorial' to route to "
        "a third-party skill installed under ~/.config/okforge/skills/."
    ),
)
@click.pass_context
def deck_new(ctx, name, intent, yes_flag, critique_flag, skill_name):
    """Generate a new HTML deck from this KB's wiki.

    NAME is a kebab-case slug used for the output directory.
    INTENT is a natural-language description of what the deck is about.

    Example:

      okforge deck new transformers-pitch "Explain attention to engineers"
      okforge deck new transformers-pitch "Explain attention to engineers" --critique
      okforge deck new transformers-pitch "..." --skill deck-guizang-editorial
    """
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `okforge init` first.", err=True)
        ctx.exit(1)

    # Reuse the shared safety gates: name validation + wiki content check.
    # Matches chat's `/deck new` so users see the same errors in both UIs.
    err = _preflight_skill_new(kb_dir, name)
    if err:
        # _preflight_skill_new returns messages like "Skill name must not be empty."
        # and "Wiki at ... is empty — add documents with `okforge add` first."
        err = err.replace("Skill name", "Deck name")
        # Only append the kebab-case hint when the failure is actually about
        # the slug, not the wiki-content gate.
        if "kebab" not in err.lower() and "Wiki" not in err and "wiki" not in err:
            err = err + " Use a kebab-case slug like 'my-deck'."
        click.echo(f"[ERROR] {err}", err=True)
        ctx.exit(1)

    # Verify LLM key + load config BEFORE touching existing output. Any
    # failure here (missing API key, malformed config) must leave the old
    # deck directory intact — we can't replace it if we can't proceed.
    try:
        _setup_llm_key(kb_dir)
    except RuntimeError as exc:
        click.echo(f"[ERROR] {exc}", err=True)
        ctx.exit(1)
    config = load_config(state_dir(kb_dir) / "config.yaml")
    model = config.get("model", DEFAULT_CONFIG["model"])

    # Overwrite handling — inline because okforge.skill.workspace.save_iteration
    # is hard-wired to skill paths (uses skill_dir / skill_workspace_dir from
    # okforge.skill). Mirror its iteration-N copy-then-rmtree behavior here
    # using deck_workspace_dir so users keep rollback safety without coupling
    # deck CLI to skill internals.
    from okforge.deck import deck_dir as _deck_dir

    target = _deck_dir(kb_dir, name)
    if target.exists():
        if yes_flag:
            _save_deck_iteration(kb_dir, name)
            shutil.rmtree(target)
        elif sys.stdin.isatty():
            if not click.confirm(
                f"output/decks/{name}/ already exists. Overwrite?",
                default=False,
            ):
                click.echo("Aborted.")
                ctx.exit(1)
            _save_deck_iteration(kb_dir, name)
            shutil.rmtree(target)
        else:
            click.echo(
                f"[ERROR] output/decks/{name}/ exists. Pass -y to overwrite "
                f"in non-interactive contexts.",
                err=True,
            )
            ctx.exit(1)

    # Run the generator.
    from okforge.skill.generator import Generator
    from okforge.deck.creator import DEFAULT_DECK_SKILL

    skill_label = skill_name if skill_name else f"{DEFAULT_DECK_SKILL} (default)"
    click.echo(f"Generating deck '{name}' via skill {skill_label}...")
    gen = Generator(
        target_type="deck",
        name=name,
        intent=intent,
        kb_dir=kb_dir,
        model=model,
        critique=critique_flag,
        skill_name=skill_name,
    )
    try:
        asyncio.run(gen.run())
    except RuntimeError as exc:
        click.echo(f"[ERROR] {exc}", err=True)
        ctx.exit(1)

    # Surface validation result.
    if gen.validation:
        for w in gen.validation.warnings:
            click.echo(f"[WARN] {w}", err=True)
        for e in gen.validation.errors:
            click.echo(f"[ERROR] {e}", err=True)
        if gen.validation.errors:
            click.echo(
                f"Deck written to {gen.output_dir / 'index.html'} but failed validation. "
                f"Inspect and re-run.",
                err=True,
            )
            ctx.exit(1)

    click.echo(f"Deck written to {gen.output_dir / 'index.html'}")


def _save_deck_iteration(kb_dir: Path, deck_name: str) -> Path | None:
    """Copy ``<kb>/output/decks/<name>/`` to the next iteration slot.

    Mirrors ``okforge.skill.workspace.save_iteration`` but uses
    ``deck_workspace_dir`` so deck rollback history stays separate from
    skill history. Returns the saved iteration path, or ``None`` if there's
    no current deck to save.
    """
    import re
    from okforge.deck import deck_dir as _deck_dir, deck_workspace_dir as _deck_workspace_dir

    src = _deck_dir(kb_dir, deck_name)
    if not src.is_dir():
        return None

    ws = _deck_workspace_dir(kb_dir, deck_name)
    ws.mkdir(parents=True, exist_ok=True)

    iter_re = re.compile(r"^iteration-(\d+)$")
    existing_ns: list[int] = []
    for child in ws.iterdir():
        if child.is_dir():
            m = iter_re.match(child.name)
            if m:
                existing_ns.append(int(m.group(1)))
    next_n = (max(existing_ns) if existing_ns else 0) + 1

    dest = ws / f"iteration-{next_n}"
    shutil.copytree(src, dest)
    return dest
