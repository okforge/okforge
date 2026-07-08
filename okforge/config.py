from __future__ import annotations

import contextlib
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterator

import yaml

from okforge.locks import atomic_write_text, flock, funlock

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "gpt-5.4",
    "language": "en",
    "pageindex_threshold": 20,
}

# Default entity-type vocabulary. Overridable per-KB via the optional
# ``entity_types:`` config key (see ``resolve_entity_types``).
DEFAULT_ENTITY_TYPES: tuple[str, ...] = (
    "person",
    "organization",
    "place",
    "product",
    "work",
    "event",
    "other",
)

# Per-KB state directory. LEGACY_STATE_DIR_NAME exists only for the
# migrate command and the temporary discovery-compat fallback in
# cli._find_kb_dir — drop it once the whole fleet is confirmed migrated.
STATE_DIR_NAME = ".okforge"
LEGACY_STATE_DIR_NAME = ".openkb"


def state_dir(kb_root: Path) -> Path:
    """Path to a KB's state directory (config, hash registry, locks).

    Resolves to ``.okforge/`` if present; falls back to the legacy
    ``.openkb/`` for a KB not yet migrated (see the ``migrate`` command)
    so every call site — locking, config reads, hash registry, discovery
    — transparently keeps working against an unmigrated KB without any
    special-casing. For a KB with neither yet (a brand-new KB about to
    be created by ``init``), returns the ``.okforge/`` path: the location
    a fresh init should create.

    This is a temporary migration-compat mechanism, not permanent dual
    support — see LEGACY_STATE_DIR_NAME.
    """
    new = kb_root / STATE_DIR_NAME
    if new.is_dir():
        return new
    legacy = kb_root / LEGACY_STATE_DIR_NAME
    if legacy.is_dir():
        return legacy
    return new


GLOBAL_CONFIG_DIR = Path.home() / ".config" / "okforge"
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "global.yaml"
GLOBAL_CONFIG_LOCK_PATH = GLOBAL_CONFIG_DIR / "global.lock"
LEGACY_GLOBAL_CONFIG_DIR = Path.home() / ".config" / "openkb"


@contextlib.contextmanager
def _with_global_config_lock() -> Iterator[None]:
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with GLOBAL_CONFIG_LOCK_PATH.open("a+", encoding="utf-8") as fh:
        flock(fh, exclusive=True)
        try:
            yield
        finally:
            funlock(fh)


def _atomic_yaml_dump(path: Path, config: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
    )


def _load_global_config_unlocked() -> dict[str, Any]:
    if GLOBAL_CONFIG_PATH.exists():
        with GLOBAL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def resolve_entity_types(config: dict) -> list[str]:
    """Resolve the effective entity-type list from a loaded config dict.

    If ``config["entity_types"]`` is a non-empty list, each string item is
    cleaned (lowercased, trimmed, restricted to ``[a-z0-9 _-]`` so a stray
    brace/punctuation can't leak into a prompt template or frontmatter value);
    non-string items (YAML nulls, numbers) are skipped. The cleaned list is
    de-duped (order preserving) and ``"other"`` is always appended when missing
    (it is the coercion fallback). Otherwise — key absent, not a list, empty,
    or fully malformed — :data:`DEFAULT_ENTITY_TYPES` is returned, so behavior
    is byte-identical to the default. A warning is logged only when
    ``entity_types`` was present-but-malformed.
    """
    raw = config.get("entity_types")
    if raw is None:
        return list(DEFAULT_ENTITY_TYPES)
    if not isinstance(raw, list):
        logger.warning(
            "config: 'entity_types' must be a list of strings, got %s — "
            "falling back to the default entity types.",
            type(raw).__name__,
        )
        return list(DEFAULT_ENTITY_TYPES)
    cleaned: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue  # skip YAML nulls/numbers (str(None) would become "none")
        s = re.sub(r"[^a-z0-9 _-]+", "", x.strip().lower()).strip()
        if s and s not in cleaned:
            cleaned.append(s)
    if not cleaned:
        logger.warning(
            "config: 'entity_types' was present but yielded no usable values — "
            "falling back to the default entity types.",
        )
        return list(DEFAULT_ENTITY_TYPES)
    if "other" not in cleaned:
        cleaned.append("other")
    return cleaned


def resolve_extra_headers(config: dict) -> dict[str, str]:
    """Resolve the optional ``extra_headers:`` config key into a str→str dict.

    Some LiteLLM providers need extra HTTP headers on every request (e.g.
    GitHub Copilot's ``Editor-Version`` IDE-auth headers). Users opt in via
    an ``extra_headers:`` mapping in config.yaml; the result is forwarded to
    LiteLLM's ``extra_headers`` parameter on all LLM calls.

    Values are stringified (YAML may parse version-like values as numbers).
    Entries with a non-string/empty key or a non-scalar value are skipped.
    A non-mapping ``extra_headers`` is ignored entirely. Warnings are logged
    only when the key was present but malformed.
    """
    raw = config.get("extra_headers")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "config: 'extra_headers' must be a mapping of header name to "
            "value, got %s — ignoring it.",
            type(raw).__name__,
        )
        return {}
    headers: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            logger.warning(
                "config: skipping 'extra_headers' entry with non-string or empty key: %r",
                key,
            )
            continue
        if value is None or not isinstance(value, (str, int, float, bool)):
            logger.warning(
                "config: skipping 'extra_headers' entry %r with non-scalar value: %r",
                key,
                value,
            )
            continue
        headers[key.strip()] = str(value)
    return headers


def resolve_timeout(config: dict) -> float | None:
    """Resolve the optional ``timeout:`` key to a finite positive number of seconds.

    Returns ``None`` (use LiteLLM's default) when absent or invalid; rejects
    bools and ``nan``/``inf``, warning when present but unusable.
    """
    raw = config.get("timeout")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        logger.warning(
            "config: 'timeout' must be a positive number of seconds, got %s — ignoring it.",
            type(raw).__name__,
        )
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "config: 'timeout' must be a positive number of seconds, got %r — ignoring it.",
            raw,
        )
        return None
    if not math.isfinite(value) or value <= 0:
        logger.warning(
            "config: 'timeout' must be a finite positive number of seconds, got %s — ignoring it.",
            value,
        )
        return None
    return value


def resolve_litellm_settings(config: dict) -> dict[str, Any]:
    """Resolve the optional ``litellm:`` mapping of LiteLLM module settings.

    Values are forwarded verbatim (the user owns them); only the container shape
    is enforced — returns ``{}`` if absent or not a mapping, and drops non-string
    keys. ``cli._apply_litellm_settings`` applies them.
    """
    raw = config.get("litellm")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "config: 'litellm' must be a mapping of LiteLLM settings, got %s — ignoring it.",
            type(raw).__name__,
        )
        return {}
    settings: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            logger.warning("config: skipping 'litellm' entry with non-string key %r.", key)
            continue
        settings[key] = value
    return settings


# Process-wide extra headers for LLM requests, resolved from the active KB's
# config by the CLI entry points (cli._setup_llm_key). LLM call sites read it
# via get_extra_headers() so the value doesn't have to be threaded through
# every compile/agent call chain — mirroring how the API key is applied
# globally via litellm.api_key / provider env vars.
_runtime_extra_headers: dict[str, str] = {}


def set_extra_headers(headers: dict[str, str]) -> None:
    """Set the process-wide extra headers for LLM requests."""
    global _runtime_extra_headers
    _runtime_extra_headers = dict(headers)


def get_extra_headers() -> dict[str, str]:
    """Return a copy of the process-wide extra headers for LLM requests."""
    return dict(_runtime_extra_headers)


def resolve_llm_extra_body(config: dict) -> dict:
    """Resolve the optional ``llm_extra_body:`` mapping from config.yaml.

    Provider-specific request-body extras forwarded verbatim to LiteLLM's
    ``extra_body`` on every compile-pipeline LLM call. The motivating case:
    llama.cpp serving a Qwen3-family model with reasoning enabled by default
    — a KB opts out per request with::

        llm_extra_body:
          chat_template_kwargs:
            enable_thinking: false

    Values are the user's to own (forwarded untouched); only the container
    shape is enforced.
    """
    raw = config.get("llm_extra_body")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "config: 'llm_extra_body' must be a mapping, got %s — ignoring it.",
            type(raw).__name__,
        )
        return {}
    body: dict = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            logger.warning(
                "config: skipping 'llm_extra_body' entry with non-string or empty key: %r",
                key,
            )
            continue
        body[key] = value
    return body


# Process-wide extra request body for LLM calls, resolved from the active
# KB's config by the CLI entry points — same pattern as the extra headers.
_runtime_llm_extra_body: dict = {}


def set_llm_extra_body(body: dict) -> None:
    """Set the process-wide extra request body for LLM calls."""
    global _runtime_llm_extra_body
    _runtime_llm_extra_body = dict(body)


def get_llm_extra_body() -> dict:
    """Return a copy of the process-wide extra request body for LLM calls."""
    return dict(_runtime_llm_extra_body)


# Process-wide LLM request timeout (seconds), set from config by the CLI and
# read at the call sites via get_timeout(). None = use LiteLLM's default.
_runtime_timeout: float | None = None


def set_timeout(timeout: float | None) -> None:
    """Set the process-wide LLM request timeout in seconds; ``None`` clears it."""
    global _runtime_timeout
    _runtime_timeout = timeout


def get_timeout() -> float | None:
    """Return the process-wide LLM request timeout in seconds, or ``None``."""
    return _runtime_timeout


def get_timeout_extra_args() -> dict[str, float] | None:
    """Timeout as Agents-SDK ``ModelSettings.extra_args`` (it has no ``timeout``
    field), or ``None``. The LiteLLM provider forwards it to the completion call.
    """
    return {"timeout": _runtime_timeout} if _runtime_timeout is not None else None


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from config_path, merged with DEFAULT_CONFIG.

    If the file does not exist, returns a copy of the defaults.
    """
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        config.update(data)
    return config


def save_config(config_path: Path, config: dict) -> None:
    """Persist config dict to YAML, creating parent directories as needed."""
    _atomic_yaml_dump(config_path, config)


def load_global_config() -> dict[str, Any]:
    """Load the global config from ~/.config/okforge/global.yaml."""
    return _load_global_config_unlocked()


def save_global_config(config: dict[str, Any]) -> None:
    """Save the global config to ~/.config/okforge/global.yaml."""
    with _with_global_config_lock():
        _atomic_yaml_dump(GLOBAL_CONFIG_PATH, config)


def register_kb(kb_path: Path) -> None:
    """Register a KB path in the global config's known_kbs list."""
    with _with_global_config_lock():
        gc = _load_global_config_unlocked()
        known = gc.get("known_kbs", [])
        resolved = str(kb_path.resolve())
        if resolved not in known:
            known.append(resolved)
            gc["known_kbs"] = known
        gc["default_kb"] = resolved
        _atomic_yaml_dump(GLOBAL_CONFIG_PATH, gc)
