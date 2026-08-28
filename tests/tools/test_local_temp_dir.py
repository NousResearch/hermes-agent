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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
