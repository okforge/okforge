import json

import pytest


@pytest.fixture(autouse=True)
def _reset_extra_headers():
    """Keep the process-wide LLM extra-headers / timeout stashes from leaking across tests."""
    from okforge.config import set_extra_headers, set_llm_extra_body, set_timeout

    yield
    set_extra_headers({})
    set_llm_extra_body({})
    set_timeout(None)


_KB_CONFIG_YAML = """\
version: "0.1.0"
embedding_model: text-embedding-3-small
llm_model: gpt-4o-mini
chunk_size: 512
chunk_overlap: 64
"""


def _make_kb_dirs(tmp_path):
    """Shared raw/ + wiki/ scaffolding for both kb_dir and legacy_kb_dir."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki" / "sources" / "images").mkdir(parents=True)
    (tmp_path / "wiki" / "summaries").mkdir(parents=True)
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    (tmp_path / "wiki" / "explorations").mkdir(parents=True)
    (tmp_path / "wiki" / "reports").mkdir(parents=True)


@pytest.fixture
def kb_dir(tmp_path):
    """Create a minimal knowledge base directory structure for testing.

    Uses the current .okforge/ state-dir name — the default for any KB
    created fresh (post-rename). See legacy_kb_dir for a KB still on the
    pre-rename .openkb/ layout (migration/discovery-compat tests).
    """
    _make_kb_dirs(tmp_path)

    state_dir = tmp_path / ".okforge"
    state_dir.mkdir()
    (state_dir / "config.yaml").write_text(_KB_CONFIG_YAML)
    (state_dir / "hashes.json").write_text(json.dumps({}))

    return tmp_path


@pytest.fixture
def legacy_kb_dir(tmp_path):
    """A KB still on the pre-rename .openkb/ state-dir layout, not yet
    migrated — for testing discovery-compat fallback and the migrate
    command itself."""
    _make_kb_dirs(tmp_path)

    state_dir = tmp_path / ".openkb"
    state_dir.mkdir()
    (state_dir / "config.yaml").write_text(_KB_CONFIG_YAML)
    (state_dir / "hashes.json").write_text(json.dumps({}))

    return tmp_path
