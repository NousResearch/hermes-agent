"""macOS CuaDriver.app path and signing identity contracts."""

from __future__ import annotations

import subprocess

import pytest

from tools.computer_use import cua_backend


def _codesign_proc(
    *,
    team_id: str,
    identifier: str = "com.trycua.driver",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        ["codesign"],
        0,
        stdout="",
        stderr=(
            f"Identifier={identifier}\n"
            f"TeamIdentifier={team_id}\n"
        ),
    )


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
    monkeypatch.setattr(cua_backend.shutil, "which", lambda name: "/usr/bin/codesign")
    monkeypatch.setattr(
        cua_backend.subprocess,
        "run",
        lambda *args, **kwargs: _codesign_proc(team_id=team_id),
    )

    cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_driver_signature_still_rejects_unrecognised_team(monkeypatch):
    monkeypatch.setattr(cua_backend.shutil, "which", lambda name: "/usr/bin/codesign")
    monkeypatch.setattr(
        cua_backend.subprocess,
        "run",
        lambda *args, **kwargs: _codesign_proc(team_id="EVIL000000"),
    )

    with pytest.raises(RuntimeError, match="signed by team"):
        cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")


def test_driver_signature_still_requires_exact_bundle_identifier(monkeypatch):
    monkeypatch.setattr(cua_backend.shutil, "which", lambda name: "/usr/bin/codesign")
    monkeypatch.setattr(
        cua_backend.subprocess,
        "run",
        lambda *args, **kwargs: _codesign_proc(
            team_id="YCK386LBJ7",
            identifier="com.trycua.driver.evil",
        ),
    )

    with pytest.raises(RuntimeError, match="has identifier"):
        cua_backend._validate_cua_driver_app_signature("/Applications/CuaDriver.app")
