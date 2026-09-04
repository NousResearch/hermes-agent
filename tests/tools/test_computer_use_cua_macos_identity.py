"""macOS CuaDriver.app path and signing identity contracts."""

from __future__ import annotations

import os
import subprocess

import pytest

from tools.computer_use import cua_backend


def _codesign_proc(
    *,
    team_id: str,
    identifier: str = "com.trycua.driver",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        ["codesign"],
        returncode,
        stdout="",
        stderr=(
            f"Identifier={identifier}\n"
            f"TeamIdentifier={team_id}\n"
        ),
    )


def _patch_codesign(monkeypatch, proc):
    monkeypatch.setattr(cua_backend.shutil, "which", lambda name: "/usr/bin/codesign")
    monkeypatch.setattr(cua_backend.subprocess, "run", lambda *args, **kwargs: proc)


@pytest.mark.skipif(os.name == "nt", reason="macOS bundle paths use POSIX separators")
def test_resolve_app_path_follows_real_symlink_and_is_idempotent(tmp_path):
    app = tmp_path / "CuaDriver.app"
    executable = app / "Contents" / "MacOS" / "cua-driver"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    shim = tmp_path / "cua-driver"
    try:
        shim.symlink_to(executable)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this host")

    assert cua_backend._resolve_cua_driver_app_path(str(shim)) == str(app)
    assert cua_backend._resolve_cua_driver_app_path(str(executable)) == str(app)


def test_resolve_app_path_follows_standard_driver_symlink(monkeypatch):
    symlink = "/Users/test/.local/bin/cua-driver"
    executable = "/Applications/CuaDriver.app/Contents/MacOS/cua-driver"

    monkeypatch.setattr(cua_backend.os.path, "realpath", lambda path: executable)
    monkeypatch.setattr(cua_backend.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cua_backend.os, "access", lambda path, mode: True)

    assert cua_backend._resolve_cua_driver_app_path(symlink) == "/Applications/CuaDriver.app"


def test_resolve_app_path_does_not_fall_back_to_an_unrelated_bundle(monkeypatch):
    monkeypatch.setattr(
        cua_backend.os.path,
        "realpath",
        lambda path: "/usr/local/bin/cua-driver",
    )

    assert cua_backend._resolve_cua_driver_app_path("cua-driver") is None


@pytest.mark.parametrize("team_id", ["4YEC26S9KF", "YCK386LBJ7"])
def test_driver_signature_accepts_official_team_ids(monkeypatch, team_id):
    _patch_codesign(monkeypatch, _codesign_proc(team_id=team_id))

    cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_driver_signature_still_rejects_unrecognised_team(monkeypatch):
    _patch_codesign(monkeypatch, _codesign_proc(team_id="EVIL000000"))

    with pytest.raises(RuntimeError, match="signed by team"):
        cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_driver_signature_still_requires_exact_bundle_identifier(monkeypatch):
    _patch_codesign(
        monkeypatch,
        _codesign_proc(
            team_id="YCK386LBJ7",
            identifier="com.trycua.driver.evil",
        ),
    )

    with pytest.raises(RuntimeError, match="has identifier"):
        cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_driver_signature_rejects_unsigned_by_default(monkeypatch):
    _patch_codesign(monkeypatch, _codesign_proc(team_id="not set"))
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {})

    with pytest.raises(RuntimeError, match="signed by team"):
        cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_driver_signature_allows_unsigned_only_with_opt_in(monkeypatch):
    _patch_codesign(monkeypatch, _codesign_proc(team_id="not set"))
    monkeypatch.setattr(
        cua_backend,
        "_computer_use_cfg",
        lambda: {"allow_unsigned_driver": True},
    )

    cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_unsigned_opt_in_still_requires_exact_bundle_identifier(monkeypatch):
    _patch_codesign(
        monkeypatch,
        _codesign_proc(
            team_id="not set",
            identifier="com.trycua.driver.evil",
        ),
    )
    monkeypatch.setattr(
        cua_backend,
        "_computer_use_cfg",
        lambda: {"allow_unsigned_driver": True},
    )

    with pytest.raises(RuntimeError, match="has identifier"):
        cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_driver_signature_requires_codesign(monkeypatch):
    monkeypatch.setattr(cua_backend.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="codesign is required"):
        cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_driver_signature_rejects_codesign_failure(monkeypatch):
    _patch_codesign(
        monkeypatch,
        _codesign_proc(team_id="", returncode=1),
    )

    with pytest.raises(RuntimeError, match="not code-signed"):
        cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_embedded_spawn_resolves_shim_and_accepts_current_team(monkeypatch):
    executable = "/Applications/CuaDriver.app/Contents/MacOS/cua-driver"
    monkeypatch.setattr(cua_backend.os.path, "realpath", lambda path: executable)
    monkeypatch.setattr(cua_backend.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cua_backend.os, "access", lambda path, mode: True)
    _patch_codesign(monkeypatch, _codesign_proc(team_id="YCK386LBJ7"))

    command = cua_backend._embedded_daemon_spawn_command(
        "/Users/test/.local/bin/cua-driver",
        ["serve", "--embedded", "--socket", "/tmp/private.sock"],
        platform="darwin",
    )

    assert command == [
        "/usr/bin/open",
        "-n",
        "-g",
        "-a",
        "/Applications/CuaDriver.app",
        "--args",
        "serve",
        "--embedded",
        "--socket",
        "/tmp/private.sock",
    ]
