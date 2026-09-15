"""Regression coverage for Desktop sessions using profile terminal.cwd."""

import tui_gateway.server as server


def test_profile_configured_cwd_is_explicit_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: str(workspace))
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: None)
    assert server._session_create_explicit_cwd({}, tmp_path / "profile") is True


def test_missing_configured_cwd_remains_launch_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: None)
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: None)
    assert server._session_create_explicit_cwd({}, tmp_path / "profile") is False