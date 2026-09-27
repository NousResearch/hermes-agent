"""Background local spawns get the same PATH augmentation as foreground calls (#124820).

Foreground ``terminal`` calls go through ``_make_run_env``, which appends
``_SANE_PATH``, the Hermes-managed runtime dirs (``$HERMES_HOME/bin`` — the managed
``uv`` — and the pm store's node/npm) and ``~/.local/bin``. Background spawns went
through ``ProcessRegistry._spawn_env`` with an identity PATH transform, so on an
install whose only ``uv`` is the managed one the same command worked in the
foreground and exited 127 (``command not found: uv``) in the background.
"""
import os

import pytest


@pytest.mark.platforms("posix")
def test_background_spawn_env_appends_sane_path(tmp_path, monkeypatch):
    from tools.process_registry import ProcessRegistry

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "bin").mkdir(parents=True)
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin:/bin"})

    env = ProcessRegistry._spawn_env({})

    entries = env["PATH"].split(":")
    assert str(home / "bin") in entries  # the managed uv resolves in the background too
    assert "/usr/bin" in entries and entries.index("/usr/bin") < entries.index(str(home / "bin"))
    assert env["PYTHONUNBUFFERED"] == "1"


@pytest.mark.platforms("posix")
def test_background_spawn_env_drops_empty_entries_and_duplicates(tmp_path, monkeypatch):
    from tools.process_registry import ProcessRegistry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin::/usr/bin:/bin"})

    env = ProcessRegistry._spawn_env({})

    entries = env["PATH"].split(":")
    assert "" not in entries  # shells read an empty entry as cwd
    assert entries.count("/usr/bin") == 1  # duplicates collapsed, first wins
