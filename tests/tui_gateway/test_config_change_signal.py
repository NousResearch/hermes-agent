"""Config invalidation pushes without exposing config contents (regression for #127374)."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tui_gateway import server


@pytest.fixture
def config_homes(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    served = tmp_path / "served"
    launch.mkdir()
    served.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", {served})
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_change_sigs", {})
    monkeypatch.setattr(server, "_change_checked_at", {})
    monkeypatch.setattr(server, "_change_broadcast_at", {})
    monkeypatch.setattr(server, "_sessions_db_sig_cache", {})
    monkeypatch.setattr(server, "_pairing_roots_cache", None)
    monkeypatch.setattr(server, "_bot_relay_outbox_seen", 0)
    frames = []
    # Keep the real broadcaster and event-contract validation, replacing only I/O.
    monkeypatch.setattr(server, "_live_transports", [SimpleNamespace(write=frames.append)])
    return launch, served, frames


@pytest.mark.parametrize("initially_present", [True, False])
def test_config_file_changes_push_once_across_launch_and_served_homes(config_homes, initially_present):
    launch, served, frames = config_homes
    if initially_present:
        for home in (launch, served):
            (home / "config.yaml").write_text("display: {}\n", encoding="utf-8")
        # A newer launch timestamp must not hide changes to an older served file.
        os.utime(launch / "config.yaml", (2_000_000_000, 2_000_000_000))

    now = 0.0

    def tick():
        nonlocal now
        server._broadcast_watched_changes(now=now)
        now += 10.0
        signals = [frame["params"] for frame in frames if frame["params"]["type"] == "config.changed"]
        frames.clear()
        return signals

    assert tick() == []  # Boot seeds even a missing config silently.
    assert tick() == []
    for home in (launch, served, launch):
        path = home / "config.yaml"
        path.write_text("display: {density: compact}\napi_key: never-broadcast-this\n", encoding="utf-8")
        assert tick() == [{"type": "config.changed", "session_id": "", "payload": {}}]
        assert tick() == []
        path.unlink()
        assert tick() == [{"type": "config.changed", "session_id": "", "payload": {}}]
        assert tick() == []


@pytest.mark.parametrize("config_exists", [True, False])
def test_config_get_mtime_advertises_push_support_through_real_handler(config_homes, config_exists):
    launch, _, _ = config_homes
    path = launch / "config.yaml"
    if config_exists:
        path.write_text("display: {}\n", encoding="utf-8")
    response = server._methods["config.get"]("config-capability", {"key": "mtime"})
    assert "error" not in response
    result = response["result"]
    assert result.get("change_events") is True
    assert result["mtime"] == (path.stat().st_mtime if config_exists else 0)
    assert isinstance(result["mcp_rev"], str)
