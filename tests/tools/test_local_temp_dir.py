"""Tests for ``LocalEnvironment.get_temp_dir`` temp-dir redirect.

Hermes exposes ``terminal.temp_dir`` (mirrored to ``TERMINAL_TEMP_DIR``) so
users on RAM-based tmpfs ``/tmp`` can point session temp files (background
logs/pid/exit files, code-execution sandboxes) at real storage.
"""

import os
import sys

import pytest

from tools.environments.local import LocalEnvironment


def _make_local_env(env: dict) -> LocalEnvironment:
    """Construct a LocalEnvironment without running init_session (no bash)."""
    obj = LocalEnvironment.__new__(LocalEnvironment)
    obj.env = dict(env)
    return obj


def test_temp_dir_override_honored(tmp_path):
    target = str(tmp_path)
    env = _make_local_env({"TERMINAL_TEMP_DIR": target})
    assert env.get_temp_dir() == target


def test_temp_dir_from_process_env(tmp_path):
    target = str(tmp_path)
    env = _make_local_env({})
    prev = os.environ.get("TERMINAL_TEMP_DIR")
    os.environ["TERMINAL_TEMP_DIR"] = target
    try:
        assert env.get_temp_dir() == target
    finally:
        if prev is None:
            os.environ.pop("TERMINAL_TEMP_DIR", None)
        else:
            os.environ["TERMINAL_TEMP_DIR"] = prev


def test_temp_dir_non_existent_falls_through(tmp_path):
    """A configured path that does not exist must not be used."""
    missing = str(tmp_path / "does-not-exist")
    env = _make_local_env({"TERMINAL_TEMP_DIR": missing})
    # Should fall through to the standard TMPDIR//tmp//gettempdir chain, not
    # return the missing path.
    assert env.get_temp_dir() != missing


def test_temp_dir_empty_falls_through(tmp_path, monkeypatch):
    """An empty/relative terminal.temp_dir must not redirect."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    env = _make_local_env({"TERMINAL_TEMP_DIR": ""})
    assert env.get_temp_dir() == str(tmp_path)


def test_default_is_hermes_cache_not_tmp(tmp_path, monkeypatch):
    """With no overrides at all, the default temp root is real storage under
    HERMES_HOME (cache/terminal), NOT tmpfs /tmp."""
    import hermes_constants  # noqa: F401 — resolves HERMES_HOME per call

    for var in ("TERMINAL_TEMP_DIR", "TMPDIR", "TMP", "TEMP"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    env = _make_local_env({})
    result = env.get_temp_dir()
    assert result == str(tmp_path / ".hermes" / "cache" / "terminal")
    assert os.path.isdir(result)


def test_tmpdir_still_beats_default(tmp_path, monkeypatch):
    """An explicit TMPDIR keeps winning over the managed default."""
    monkeypatch.delenv("TERMINAL_TEMP_DIR", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    env = _make_local_env({})
    assert env.get_temp_dir() == str(tmp_path)


def test_cleanup_terminal_temp_cache(tmp_path, monkeypatch):
    """Old artifacts are pruned; fresh ones and live bg groups survive."""
    import time

    from tools.environments import local as local_mod

    root = tmp_path / "cache" / "terminal"
    root.mkdir(parents=True)
    monkeypatch.setattr(local_mod, "_default_terminal_temp_dir", lambda: root)

    old = time.time() - 100 * 3600
    fresh = time.time()

    # Stale loose artifact — pruned.
    stale = root / "hermes-snap-deadbeef.sh"
    stale.write_text("x")
    os.utime(stale, (old, old))

    # Fresh artifact — kept.
    keep = root / "hermes-snap-cafef00d.sh"
    keep.write_text("x")

    # Live bg group: stale .pid but fresh .log — WHOLE group kept.
    live_pid = root / "hermes_bg_live1.pid"
    live_pid.write_text("123")
    os.utime(live_pid, (old, old))
    live_log = root / "hermes_bg_live1.log"
    live_log.write_text("running")
    os.utime(live_log, (fresh, fresh))

    # Dead bg group: everything stale — pruned.
    for suffix in ("log", "pid", "exit"):
        f = root / f"hermes_bg_dead1.{suffix}"
        f.write_text("x")
        os.utime(f, (old, old))

    removed = local_mod.cleanup_terminal_temp_cache(max_age_hours=72)
    assert removed == 4  # stale snap + 3 dead-group files
    assert keep.exists()
    assert live_pid.exists() and live_log.exists()
    assert not stale.exists()
    assert not (root / "hermes_bg_dead1.log").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
