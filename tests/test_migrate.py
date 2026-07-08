"""Tests for `okforge migrate` — the .openkb/ -> .okforge/ state-dir migration."""

from __future__ import annotations

import threading

from click.testing import CliRunner

import okforge.config as config_mod
from okforge.cli import cli
from okforge.locks import kb_ingest_lock


def test_migrate_moves_legacy_state_dir(legacy_kb_dir):
    assert (legacy_kb_dir / ".openkb").is_dir()
    assert not (legacy_kb_dir / ".okforge").is_dir()

    res = CliRunner().invoke(cli, ["migrate", str(legacy_kb_dir)])
    assert res.exit_code == 0, res.output
    assert "migrated" in res.output

    assert (legacy_kb_dir / ".okforge").is_dir()
    assert (legacy_kb_dir / ".okforge" / "config.yaml").is_file()
    assert (legacy_kb_dir / ".okforge" / "hashes.json").is_file()

    # Old dir renamed aside, not deleted.
    assert not (legacy_kb_dir / ".openkb").is_dir()
    backups = list(legacy_kb_dir.glob(".openkb.migrated-*"))
    assert len(backups) == 1
    assert (backups[0] / "config.yaml").is_file()


def test_migrate_dry_run_touches_nothing(legacy_kb_dir):
    res = CliRunner().invoke(cli, ["migrate", str(legacy_kb_dir), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "would migrate" in res.output

    assert (legacy_kb_dir / ".openkb").is_dir()
    assert not (legacy_kb_dir / ".okforge").is_dir()
    assert not list(legacy_kb_dir.glob(".openkb.migrated-*"))


def test_migrate_is_idempotent(kb_dir):
    """A KB already on .okforge/ (no .openkb/ at all) is a no-op, not an error."""
    res = CliRunner().invoke(cli, ["migrate", str(kb_dir)])
    assert res.exit_code == 0, res.output
    assert "already migrated" in res.output
    assert (kb_dir / ".okforge").is_dir()


def test_migrate_rerun_after_success_is_idempotent(legacy_kb_dir):
    first = CliRunner().invoke(cli, ["migrate", str(legacy_kb_dir)])
    assert first.exit_code == 0, first.output

    second = CliRunner().invoke(cli, ["migrate", str(legacy_kb_dir)])
    assert second.exit_code == 0, second.output
    assert "already migrated" in second.output
    # Rerunning must not create a second backup dir.
    assert len(list(legacy_kb_dir.glob(".openkb.migrated-*"))) == 1


def test_migrate_refuses_when_both_dirs_present(legacy_kb_dir):
    (legacy_kb_dir / ".okforge").mkdir()
    (legacy_kb_dir / ".okforge" / "config.yaml").write_text("model: gpt-5.4\n")

    res = CliRunner().invoke(cli, ["migrate", str(legacy_kb_dir)])
    assert res.exit_code == 0, res.output
    assert "both" in res.output.lower()
    assert "ERROR" in res.output
    # Neither directory touched — no guessing.
    assert (legacy_kb_dir / ".openkb").is_dir()
    assert (legacy_kb_dir / ".okforge").is_dir()
    assert not list(legacy_kb_dir.glob(".openkb.migrated-*"))


def test_migrate_errors_on_non_kb_directory(tmp_path):
    empty = tmp_path / "not-a-kb"
    empty.mkdir()
    res = CliRunner().invoke(cli, ["migrate", str(empty)])
    assert res.exit_code == 0, res.output
    assert "ERROR" in res.output
    assert "not an okforge KB" in res.output


def test_migrate_blocks_while_ingest_lock_held(legacy_kb_dir):
    """migrate must take the same kb_ingest_lock as add/remove/recompile —
    it should block, not race, against a concurrent mutation."""
    ready = threading.Event()
    release = threading.Event()

    def hold_lock():
        with kb_ingest_lock(legacy_kb_dir / ".openkb"):
            ready.set()
            release.wait(timeout=2)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert ready.wait(timeout=2)

    result = {}

    def run_migrate():
        result["res"] = CliRunner().invoke(cli, ["migrate", str(legacy_kb_dir)])

    migrator = threading.Thread(target=run_migrate)
    migrator.start()
    migrator.join(timeout=0.3)
    assert migrator.is_alive(), "migrate should block while the ingest lock is held"

    release.set()
    holder.join(timeout=2)
    migrator.join(timeout=2)
    assert not migrator.is_alive()
    assert result["res"].exit_code == 0, result["res"].output
    assert (legacy_kb_dir / ".okforge").is_dir()


def test_migrate_verifies_before_removing_legacy_dir(legacy_kb_dir, monkeypatch):
    """If the freshly-copied state dir doesn't parse, abort before touching
    the original — never leave a KB with neither a valid .openkb/ nor a
    valid .okforge/."""
    import okforge.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_verify_migrated_state", lambda new_dir: (False, "boom"))

    res = CliRunner().invoke(cli, ["migrate", str(legacy_kb_dir)])
    assert res.exit_code == 0, res.output
    assert "verification failed" in res.output

    assert (legacy_kb_dir / ".openkb").is_dir()
    assert not (legacy_kb_dir / ".okforge").is_dir()
    assert not list(legacy_kb_dir.glob(".openkb.migrated-*"))


def test_migrate_requires_target_or_global(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_DIR", tmp_path / "no-global")
    res = CliRunner().invoke(cli, ["migrate"])
    assert res.exit_code == 0, res.output
    assert "Specify a KB directory" in res.output


def test_migrate_global_moves_config_dir(tmp_path, monkeypatch):
    old_global = tmp_path / "old-config" / "openkb"
    new_global = tmp_path / "new-config" / "okforge"
    old_global.mkdir(parents=True)
    (old_global / "global.yaml").write_text("default_kb: /tmp/somewhere\nknown_kbs: []\n")

    monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_DIR", new_global)
    monkeypatch.setattr(config_mod, "LEGACY_GLOBAL_CONFIG_DIR", old_global)

    res = CliRunner().invoke(cli, ["migrate", "--global"])
    assert res.exit_code == 0, res.output
    assert "Global config migrated" in res.output

    assert new_global.is_dir()
    assert (new_global / "global.yaml").is_file()
    assert not old_global.is_dir()
    backups = list(old_global.parent.glob("openkb.migrated-*"))
    assert len(backups) == 1


def test_migrate_global_is_idempotent(tmp_path, monkeypatch):
    new_global = tmp_path / "config" / "okforge"
    new_global.mkdir(parents=True)
    (new_global / "global.yaml").write_text("default_kb: null\nknown_kbs: []\n")
    monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_DIR", new_global)
    monkeypatch.setattr(config_mod, "LEGACY_GLOBAL_CONFIG_DIR", tmp_path / "config" / "openkb")

    res = CliRunner().invoke(cli, ["migrate", "--global"])
    assert res.exit_code == 0, res.output
    assert "already migrated" in res.output


def test_migrate_global_refuses_when_both_present(tmp_path, monkeypatch):
    old_global = tmp_path / "config" / "openkb"
    new_global = tmp_path / "config" / "okforge"
    old_global.mkdir(parents=True)
    new_global.mkdir(parents=True)
    (old_global / "global.yaml").write_text("default_kb: null\nknown_kbs: []\n")
    (new_global / "global.yaml").write_text("default_kb: null\nknown_kbs: []\n")

    monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_DIR", new_global)
    monkeypatch.setattr(config_mod, "LEGACY_GLOBAL_CONFIG_DIR", old_global)

    res = CliRunner().invoke(cli, ["migrate", "--global"])
    assert res.exit_code == 0, res.output
    assert "both" in res.output.lower()
    assert old_global.is_dir()
    assert new_global.is_dir()


def test_migrate_global_skills_moves_when_present(tmp_path, monkeypatch):
    old_global = tmp_path / "config" / "openkb"
    new_global = tmp_path / "config" / "okforge"
    old_global.mkdir(parents=True)
    (old_global / "global.yaml").write_text("default_kb: null\nknown_kbs: []\n")
    monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_DIR", new_global)
    monkeypatch.setattr(config_mod, "LEGACY_GLOBAL_CONFIG_DIR", old_global)

    fake_home = tmp_path / "home"
    old_skills = fake_home / ".openkb" / "skills" / "my-skill"
    old_skills.mkdir(parents=True)
    (old_skills / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")
    monkeypatch.setenv("HOME", str(fake_home))

    res = CliRunner().invoke(cli, ["migrate", "--global"])
    assert res.exit_code == 0, res.output
    assert "Global skills migrated" in res.output

    new_skills = new_global / "skills" / "my-skill"
    assert new_skills.is_dir()
    assert (new_skills / "SKILL.md").is_file()
    assert not (fake_home / ".openkb" / "skills").exists()


def test_migrate_global_skills_noop_when_absent(tmp_path, monkeypatch):
    new_global = tmp_path / "config" / "okforge"
    new_global.mkdir(parents=True)
    (new_global / "global.yaml").write_text("default_kb: null\nknown_kbs: []\n")
    monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_DIR", new_global)
    monkeypatch.setattr(config_mod, "LEGACY_GLOBAL_CONFIG_DIR", tmp_path / "config" / "openkb")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

    res = CliRunner().invoke(cli, ["migrate", "--global"])
    assert res.exit_code == 0, res.output
    assert not (new_global / "skills").exists()


def test_discovery_finds_legacy_kb(legacy_kb_dir):
    """A not-yet-migrated KB must still be fully usable — status, list,
    query, etc. all resolve it via the discovery-compat fallback."""
    res = CliRunner().invoke(cli, ["--kb-dir", str(legacy_kb_dir), "status"])
    assert res.exit_code == 0, res.output
    assert "Knowledge base" in res.output


def test_discovery_finds_migrated_kb(kb_dir):
    res = CliRunner().invoke(cli, ["--kb-dir", str(kb_dir), "status"])
    assert res.exit_code == 0, res.output
    assert "Knowledge base" in res.output
