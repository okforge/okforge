import contextlib
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from okforge.config import (
    DEFAULT_CONFIG,
    get_extra_headers,
    get_timeout,
    list_endpoint_models,
    load_config,
    resolve_compile_concurrency,
    resolve_extra_headers,
    resolve_litellm_settings,
    resolve_model,
    resolve_timeout,
    save_config,
    set_extra_headers,
    set_timeout,
    split_model,
)


def test_default_config_keys():
    assert "model" in DEFAULT_CONFIG
    assert "language" in DEFAULT_CONFIG
    assert "pageindex_threshold" in DEFAULT_CONFIG


def test_default_config_values():
    assert DEFAULT_CONFIG["model"] == "gpt-5.4"
    assert DEFAULT_CONFIG["language"] == "en"
    assert DEFAULT_CONFIG["pageindex_threshold"] == 20


def test_load_missing_file_returns_defaults(tmp_path):
    missing = tmp_path / "nonexistent" / "config.yaml"
    config = load_config(missing)
    assert config == DEFAULT_CONFIG


def test_save_creates_parent_dirs(tmp_path):
    config_path = tmp_path / "nested" / "dir" / "config.yaml"
    save_config(config_path, DEFAULT_CONFIG)
    assert config_path.exists()


def test_save_load_roundtrip(tmp_path):
    config_path = tmp_path / "config.yaml"
    custom = {"model": "gpt-3.5-turbo", "language": "fr"}
    save_config(config_path, custom)
    loaded = load_config(config_path)
    # Custom values override defaults
    assert loaded["model"] == "gpt-3.5-turbo"
    assert loaded["language"] == "fr"
    # Defaults fill in missing keys
    assert loaded["pageindex_threshold"] == DEFAULT_CONFIG["pageindex_threshold"]


def test_load_overrides_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    save_config(config_path, {"model": "claude-3", "pageindex_threshold": 100})
    loaded = load_config(config_path)
    assert loaded["model"] == "claude-3"
    assert loaded["pageindex_threshold"] == 100
    # Non-overridden defaults still present
    assert loaded["language"] == "en"


# --- extra_headers -----------------------------------------------------------


def test_resolve_extra_headers_absent_returns_empty():
    assert resolve_extra_headers({}) == {}


def test_resolve_extra_headers_valid_mapping():
    config = {
        "extra_headers": {
            "Editor-Version": "vscode/1.95.0",
            "Copilot-Integration-Id": "vscode-chat",
        }
    }
    assert resolve_extra_headers(config) == {
        "Editor-Version": "vscode/1.95.0",
        "Copilot-Integration-Id": "vscode-chat",
    }


def test_resolve_extra_headers_stringifies_scalar_values():
    # YAML may parse version-ish values as numbers.
    config = {"extra_headers": {"X-Api-Version": 2024, "X-Ratio": 1.5}}
    assert resolve_extra_headers(config) == {"X-Api-Version": "2024", "X-Ratio": "1.5"}


def test_resolve_extra_headers_non_mapping_ignored():
    assert resolve_extra_headers({"extra_headers": ["Editor-Version: x"]}) == {}
    assert resolve_extra_headers({"extra_headers": "Editor-Version: x"}) == {}


def test_resolve_extra_headers_skips_bad_entries():
    config = {
        "extra_headers": {
            "Good": "value",
            "": "empty-key-skipped",
            "NoneValue": None,
            "ListValue": ["a"],
            123: "non-string-key-skipped",
        }
    }
    assert resolve_extra_headers(config) == {"Good": "value"}


def test_extra_headers_stash_roundtrip_and_isolation():
    set_extra_headers({"A": "1"})
    got = get_extra_headers()
    assert got == {"A": "1"}
    # Mutating the returned copy must not affect the stash.
    got["B"] = "2"
    assert get_extra_headers() == {"A": "1"}
    set_extra_headers({})
    assert get_extra_headers() == {}


# --- timeout -----------------------------------------------------------------


def test_resolve_timeout_absent_returns_none():
    assert resolve_timeout({}) is None


def test_resolve_timeout_int_and_float():
    assert resolve_timeout({"timeout": 1200}) == 1200.0
    assert resolve_timeout({"timeout": 0.5}) == 0.5


def test_resolve_timeout_numeric_string_coerced():
    assert resolve_timeout({"timeout": "1200"}) == 1200.0


def test_resolve_timeout_rejects_non_positive():
    assert resolve_timeout({"timeout": 0}) is None
    assert resolve_timeout({"timeout": -10}) is None


def test_resolve_timeout_rejects_bool():
    # bool is a subclass of int; True/False are not durations.
    assert resolve_timeout({"timeout": True}) is None


def test_resolve_timeout_rejects_non_numeric():
    assert resolve_timeout({"timeout": "soon"}) is None
    assert resolve_timeout({"timeout": [1200]}) is None


def test_resolve_timeout_rejects_nan_and_inf():
    # nan/inf pass a naive `<= 0` check; YAML's .nan/.inf yield real floats.
    assert resolve_timeout({"timeout": float("inf")}) is None
    assert resolve_timeout({"timeout": float("nan")}) is None
    assert resolve_timeout({"timeout": "inf"}) is None
    assert resolve_timeout({"timeout": "nan"}) is None


def test_timeout_stash_roundtrip_and_reset():
    set_timeout(1200.0)
    assert get_timeout() == 1200.0
    set_timeout(None)
    assert get_timeout() is None


# --- compile_concurrency -------------------------------------------------------


def test_resolve_compile_concurrency_absent_returns_none():
    assert resolve_compile_concurrency({}) is None


def test_resolve_compile_concurrency_int_and_numeric_string():
    assert resolve_compile_concurrency({"compile_concurrency": 1}) == 1
    assert resolve_compile_concurrency({"compile_concurrency": "3"}) == 3


def test_resolve_compile_concurrency_rejects_non_positive():
    assert resolve_compile_concurrency({"compile_concurrency": 0}) is None
    assert resolve_compile_concurrency({"compile_concurrency": -1}) is None


def test_resolve_compile_concurrency_rejects_bool():
    assert resolve_compile_concurrency({"compile_concurrency": True}) is None


def test_resolve_compile_concurrency_rejects_non_integer():
    assert resolve_compile_concurrency({"compile_concurrency": 1.5}) is None
    assert resolve_compile_concurrency({"compile_concurrency": "soon"}) is None
    assert resolve_compile_concurrency({"compile_concurrency": [1]}) is None
    assert resolve_compile_concurrency({"compile_concurrency": float("inf")}) is None
    assert resolve_compile_concurrency({"compile_concurrency": float("nan")}) is None


def test_resolve_litellm_settings_absent_returns_empty():
    assert resolve_litellm_settings({}) == {}


def test_resolve_litellm_settings_passes_mapping_through_verbatim():
    # Values are forwarded as-is — no validation or coercion.
    config = {"litellm": {"drop_params": True, "num_retries": 3, "ssl_verify": False}}
    assert resolve_litellm_settings(config) == {
        "drop_params": True,
        "num_retries": 3,
        "ssl_verify": False,
    }


def test_resolve_litellm_settings_non_mapping_ignored():
    assert resolve_litellm_settings({"litellm": ["drop_params"]}) == {}
    assert resolve_litellm_settings({"litellm": "drop_params=true"}) == {}
    assert resolve_litellm_settings({"litellm": True}) == {}


def test_resolve_litellm_settings_drops_non_string_keys():
    assert resolve_litellm_settings({"litellm": {5: "x", "drop_params": True}}) == {
        "drop_params": True
    }


def test_resolve_litellm_settings_warns_on_non_mapping(caplog):
    with caplog.at_level(logging.WARNING, logger="okforge.config"):
        assert resolve_litellm_settings({"litellm": ["drop_params"]}) == {}
    assert "must be a mapping" in caplog.text


def test_resolve_litellm_settings_warns_on_non_string_key(caplog):
    with caplog.at_level(logging.WARNING, logger="okforge.config"):
        resolve_litellm_settings({"litellm": {5: "x", "drop_params": True}})
    assert "non-string key" in caplog.text


# --------------------------------------------------------------- model


def _fake_lister(*ids):
    """Stand in for list_endpoint_models, recording how often it was called."""
    calls = []

    def lister(api_base, timeout=None):
        calls.append(api_base)
        return list(ids)

    lister.calls = calls
    return lister


def _patch_lister(monkeypatch, lister):
    import okforge.config as cfg

    cfg.reset_model_resolution_cache()
    monkeypatch.setattr(cfg, "list_endpoint_models", lister)


BASE = "http://host:8080/v1"


def test_split_model_bare_name_defaults_to_openai():
    assert split_model("Qwen3.8-27B-MTP") == ("openai", "Qwen3.8-27B-MTP")


def test_split_model_keeps_nested_model_name():
    assert split_model("openrouter/qwen/qwen3.6-27b") == ("openrouter", "qwen/qwen3.6-27b")


def test_split_model_empty():
    assert split_model("  ") == ("", "")


def test_resolve_model_without_optin_never_probes(monkeypatch):
    lister = _fake_lister("Qwen3.8-27B-MTP")
    _patch_lister(monkeypatch, lister)
    config = {"model": "openai/Qwen3.6-27B-MTP"}
    assert resolve_model(config, api_base=BASE) == "openai/Qwen3.6-27B-MTP"
    assert lister.calls == []


def test_resolve_model_available_is_left_alone(monkeypatch):
    _patch_lister(monkeypatch, _fake_lister("Qwen3.6-27B-MTP", "Qwen3.8-27B-MTP"))
    config = {"model": "openai/Qwen3.6-27B-MTP", "model_fallback": True}
    assert resolve_model(config, api_base=BASE) == "openai/Qwen3.6-27B-MTP"


def test_resolve_model_falls_back_to_closest_name(monkeypatch, caplog):
    _patch_lister(monkeypatch, _fake_lister("Qwen3.8-27B-MTP"))
    config = {"model": "openai/Qwen3.6-27B-MTP", "model_fallback": True}
    with caplog.at_level(logging.WARNING):
        assert resolve_model(config, api_base=BASE) == "openai/Qwen3.8-27B-MTP"
    assert "not available" in caplog.text


def test_resolve_model_keeps_configured_when_no_close_match(monkeypatch, caplog):
    _patch_lister(monkeypatch, _fake_lister("gemma-4-E2B", "nomic-embed-text-v1.5"))
    config = {"model": "openai/Qwen3.6-27B-MTP", "model_fallback": True}
    with caplog.at_level(logging.WARNING):
        assert resolve_model(config, api_base=BASE) == "openai/Qwen3.6-27B-MTP"
    assert "no close match" in caplog.text


def test_resolve_model_unreachable_endpoint_keeps_configured(monkeypatch):
    _patch_lister(monkeypatch, _fake_lister())  # [] == could not tell
    config = {"model": "openai/Qwen3.6-27B-MTP", "model_fallback": True}
    assert resolve_model(config, api_base=BASE) == "openai/Qwen3.6-27B-MTP"


def test_resolve_model_skips_non_openai_provider(monkeypatch):
    lister = _fake_lister("qwen/qwen3.8-27b")
    _patch_lister(monkeypatch, lister)
    config = {"model": "openrouter/qwen/qwen3.6-27b", "model_fallback": True}
    assert resolve_model(config, api_base=BASE) == "openrouter/qwen/qwen3.6-27b"
    assert lister.calls == []


def test_resolve_model_without_api_base_keeps_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _patch_lister(monkeypatch, _fake_lister("Qwen3.8-27B-MTP"))
    config = {"model": "openai/Qwen3.6-27B-MTP", "model_fallback": True}
    assert resolve_model(config) == "openai/Qwen3.6-27B-MTP"


def test_resolve_model_reads_api_base_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", BASE)
    _patch_lister(monkeypatch, _fake_lister("Qwen3.8-27B-MTP"))
    config = {"model": "openai/Qwen3.6-27B-MTP", "model_fallback": True}
    assert resolve_model(config) == "openai/Qwen3.8-27B-MTP"


def test_resolve_model_is_memoized_per_endpoint(monkeypatch):
    lister = _fake_lister("Qwen3.8-27B-MTP")
    _patch_lister(monkeypatch, lister)
    config = {"model": "openai/Qwen3.6-27B-MTP", "model_fallback": True}
    for _ in range(3):
        assert resolve_model(config, api_base=BASE) == "openai/Qwen3.8-27B-MTP"
    assert len(lister.calls) == 1


def test_resolve_model_bare_configured_name_still_matches(monkeypatch):
    _patch_lister(monkeypatch, _fake_lister("Qwen3.8-27B-MTP"))
    config = {"model": "Qwen3.6-27B-MTP", "model_fallback": True}
    assert resolve_model(config, api_base=BASE) == "openai/Qwen3.8-27B-MTP"


# ------------------------------------------------- list_endpoint_models
#
# Every test above stubs list_endpoint_models out, so these are the only
# ones that exercise its real urllib path. They run it against a throwaway
# HTTP server on localhost rather than a mock, because the branches that
# matter most here are the failure ones — a non-200, a truncated body, a
# host that never answers — and each has to degrade to "cannot tell" ([]),
# never to "the endpoint has no models".


class _StubHandler(BaseHTTPRequestHandler):
    body = b'{"data": []}'
    status = 200
    delay = 0.0

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler's spelling)
        if self.delay:
            time.sleep(self.delay)
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass  # keep pytest output clean


@contextlib.contextmanager
def _stub_endpoint(body=b'{"data": []}', status=200, delay=0.0):
    """Serve one canned /models response; yields the api_base to probe."""
    handler = type(
        "_Handler", (_StubHandler,), {"body": body, "status": status, "delay": delay}
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    # Small poll interval: shutdown() waits up to one interval, and the
    # default 0.5s would dominate the runtime of every test below.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


MODELS_BODY = json.dumps(
    {"object": "list", "data": [{"id": "Qwen3.8-27B-MTP"}, {"id": "gemma-4-E2B"}]}
).encode()


def test_list_endpoint_models_returns_ids():
    with _stub_endpoint(MODELS_BODY) as base:
        assert list_endpoint_models(base) == ["Qwen3.8-27B-MTP", "gemma-4-E2B"]


def test_list_endpoint_models_tolerates_trailing_slash():
    with _stub_endpoint(MODELS_BODY) as base:
        assert list_endpoint_models(base + "/") == ["Qwen3.8-27B-MTP", "gemma-4-E2B"]


def test_list_endpoint_models_skips_entries_without_a_string_id():
    body = json.dumps(
        {"data": [{"id": "good"}, {"object": "model"}, {"id": 7}, "not-a-dict"]}
    ).encode()
    with _stub_endpoint(body) as base:
        assert list_endpoint_models(base) == ["good"]


def test_list_endpoint_models_empty_list_is_reported_as_unknown():
    with _stub_endpoint(b'{"data": []}') as base:
        assert list_endpoint_models(base) == []


def test_list_endpoint_models_on_error_status():
    with _stub_endpoint(b"upstream exploded", status=500) as base:
        assert list_endpoint_models(base) == []


def test_list_endpoint_models_on_malformed_json():
    with _stub_endpoint(b'{"data": [{"id": "trunc') as base:
        assert list_endpoint_models(base) == []


def test_list_endpoint_models_on_unexpected_shape():
    for body in (b'{"data": "nope"}', b"[]", b'{"models": ["a"]}', b"null"):
        with _stub_endpoint(body) as base:
            assert list_endpoint_models(base) == []


def test_list_endpoint_models_on_unreachable_host():
    with _stub_endpoint() as base:
        pass  # server is now closed, so the port refuses
    assert list_endpoint_models(base) == []


def test_list_endpoint_models_on_timeout():
    with _stub_endpoint(MODELS_BODY, delay=5.0) as base:
        assert list_endpoint_models(base, timeout=0.1) == []


def test_list_endpoint_models_refuses_non_http_scheme(tmp_path):
    # api_base is read from a KB's .env, so a file:// value must not turn
    # a model probe into a local file read.
    models = tmp_path / "models"
    models.write_text(MODELS_BODY.decode(), encoding="utf-8")
    assert list_endpoint_models(tmp_path.as_uri()) == []


def test_resolve_model_falls_back_over_a_real_http_probe(monkeypatch):
    """The whole path unmocked: config -> HTTP /models -> substitution."""
    import okforge.config as cfg

    cfg.reset_model_resolution_cache()
    with _stub_endpoint(MODELS_BODY) as base:
        config = {"model": "openai/Qwen3.6-27B-MTP", "model_fallback": True}
        assert resolve_model(config, api_base=base) == "openai/Qwen3.8-27B-MTP"
