"""Tests for opt-in SMB drive bootstrap in the Windows local terminal backend."""

import subprocess

from tools.environments import local as local_mod


def test_windows_network_drive_mapping_runs_once_per_environment(monkeypatch):
    monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(local_mod.subprocess, "run", fake_run)

    local_mod.ensure_windows_network_drives(
        [
            {"drive": "e", "remote": r"\\fileserver\engineering"},
            {"drive": "D:", "remote": r"\\fileserver\documents"},
        ]
    )

    assert [call[0] for call in calls] == [
        ["net", "use", "E:", "/delete", "/y"],
        ["net", "use", "E:", r"\\fileserver\engineering", "/persistent:no"],
        ["net", "use", "D:", "/delete", "/y"],
        ["net", "use", "D:", r"\\fileserver\documents", "/persistent:no"],
    ]
    assert all(call[1]["check"] is False for call in calls)
    assert all(call[1]["timeout"] == 15 for call in calls)


def test_windows_network_drive_mapping_ignores_invalid_entries(monkeypatch):
    monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
    calls = []
    monkeypatch.setattr(
        local_mod.subprocess,
        "run",
        lambda args, **kwargs: calls.append(args) or subprocess.CompletedProcess(args, 0),
    )

    local_mod.ensure_windows_network_drives(
        [
            {"drive": "invalid", "remote": r"\\server\share"},
            {"drive": "F", "remote": "not-a-unc-path"},
            "E: \\server\share",
            {"drive": "G", "remote": r"\\server\share"},
        ]
    )

    assert calls == [
        ["net", "use", "G:", "/delete", "/y"],
        ["net", "use", "G:", r"\\server\share", "/persistent:no"],
    ]


def test_windows_network_drive_mapping_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
    monkeypatch.setattr(
        local_mod.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    local_mod.ensure_windows_network_drives(
        [{"drive": "E", "remote": r"\\fileserver\engineering"}]
    )


def test_local_environment_bootstraps_configured_windows_drives(monkeypatch, tmp_path):
    monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
    mappings = [{"drive": "E", "remote": r"\\fileserver\engineering"}]
    observed = []
    monkeypatch.setattr(
        local_mod,
        "ensure_windows_network_drives",
        lambda value: observed.append(value),
    )
    monkeypatch.setattr(local_mod.LocalEnvironment, "init_session", lambda self: None)

    local_mod.LocalEnvironment(cwd=str(tmp_path), windows_network_drives=mappings)

    assert observed == [mappings]


def test_local_environment_does_not_bootstrap_without_mappings(monkeypatch, tmp_path):
    monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        local_mod,
        "ensure_windows_network_drives",
        lambda value: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    monkeypatch.setattr(local_mod.LocalEnvironment, "init_session", lambda self: None)

    local_mod.LocalEnvironment(cwd=str(tmp_path))
