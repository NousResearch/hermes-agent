"""Managed scope pins the per-turn terminal policy the way load_config() pins the standalone bridge.

A sandbox launcher pins ``terminal.backend`` through managed scope and withholds the profile's .env.
The per-turn scope takes the managed ``terminal:`` values over the profile's own, merged per key, and
an unreadable .env blocks turns only while nothing pins the backend.
"""

import builtins
import io
import json
import os
from pathlib import Path

import pytest

from hermes_cli import managed_scope
from tools.terminal_scope import TerminalPolicyUnavailable, build_profile_terminal_scope


@pytest.fixture
def managed(tmp_path, monkeypatch):
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    managed_scope.invalidate_managed_cache()
    yield md
    managed_scope.invalidate_managed_cache()


def _write_managed(md: Path, body: str) -> None:
    (md / "config.yaml").write_text(body, encoding="utf-8")
    managed_scope.invalidate_managed_cache()


def _profile(tmp_path: Path, *, config_yaml: str = "", dotenv: str = "") -> Path:
    home = tmp_path / "profiles" / "work"
    home.mkdir(parents=True)
    if config_yaml:
        (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    if dotenv:
        (home / ".env").write_text(dotenv, encoding="utf-8")
    return home


def _deny_reads(monkeypatch, path: Path) -> None:
    """Make *path* exist but refuse every read, as Windows does for a file a sandbox denies."""
    denied = os.path.abspath(path)
    real_open = io.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and os.path.abspath(file) == denied:
            raise PermissionError(13, "Access is denied", os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)


def test_managed_backend_beats_profile_config(tmp_path, managed):
    _write_managed(managed, "terminal:\n  backend: busybox\n")
    home = _profile(tmp_path, config_yaml="terminal:\n  backend: local\n  timeout: 77\n")

    scope = build_profile_terminal_scope(home)

    assert scope["TERMINAL_ENV"] == "busybox"
    assert scope["TERMINAL_TIMEOUT"] == "77"


def test_managed_mapping_merges_with_the_profile_mapping(tmp_path, managed):
    _write_managed(managed, "terminal:\n  docker_env:\n    A: managed\n")
    home = _profile(tmp_path, config_yaml="terminal:\n  docker_env:\n    A: profile\n    B: keep\n")

    scope = build_profile_terminal_scope(home)

    assert json.loads(scope["TERMINAL_DOCKER_ENV"]) == {"A": "managed", "B": "keep"}


def test_managed_docker_image_pins_the_image(tmp_path, managed):
    _write_managed(managed, "terminal:\n  docker_image: lab/image:1\n")
    home = _profile(tmp_path)

    scope = build_profile_terminal_scope(home)

    assert scope["TERMINAL_DOCKER_IMAGE"] == "lab/image:1"
    assert scope["TERMINAL_DOCKER_IMAGE_PINNED"] == "1"


def test_null_managed_values_pin_nothing(tmp_path, managed):
    _write_managed(managed, "terminal:\n  backend:\n  docker_image:\n")
    home = _profile(tmp_path, config_yaml="terminal:\n  backend: docker\n")

    scope = build_profile_terminal_scope(home)

    assert scope["TERMINAL_ENV"] == "docker"
    assert scope["TERMINAL_DOCKER_IMAGE_PINNED"] == "0"


def test_unreadable_env_reads_as_empty_under_a_pinned_backend(tmp_path, managed, monkeypatch):
    _write_managed(managed, "terminal:\n  backend: busybox\n")
    home = _profile(tmp_path, dotenv="TERMINAL_ENV=docker\nTERMINAL_TIMEOUT=999\n")
    _deny_reads(monkeypatch, home / ".env")

    scope = build_profile_terminal_scope(home, env_overlay={"TERMINAL_TIMEOUT": "42"})

    assert scope["TERMINAL_ENV"] == "busybox"
    assert scope["TERMINAL_TIMEOUT"] == "42"


@pytest.mark.parametrize("managed_yaml", [
    None,
    "terminal:\n  timeout: 5\n",
    "terminal:\n  backend:\n",
    "terminal:\n  backend: ''\n",
    "terminal:\n  backend: '   '\n",
])
def test_unreadable_env_fails_closed_without_a_pinned_backend(tmp_path, managed, monkeypatch, managed_yaml):
    if managed_yaml is not None:
        _write_managed(managed, managed_yaml)
    home = _profile(tmp_path, config_yaml="terminal:\n  backend: local\n", dotenv="TERMINAL_ENV=docker\n")
    _deny_reads(monkeypatch, home / ".env")

    with pytest.raises(TerminalPolicyUnavailable):
        build_profile_terminal_scope(home)
