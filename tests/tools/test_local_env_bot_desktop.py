"""Terminal child envs route GUI launches to the running Bot Desktop (#125830)."""

import pytest

from tools.environments.local import _make_run_env


def _publish(monkeypatch, value):
    monkeypatch.setattr("tools.bot_desktop.runtime.published_env", lambda: value)


def test_running_bot_desktop_display_rides_along(monkeypatch):
    _publish(
        monkeypatch,
        {
            "DISPLAY": ":20",
            "XAUTHORITY": "/run/hermes/bot-desktop/xauth",
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/hermes/bot-desktop/bus",
        },
    )
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    env = _make_run_env({})
    assert env["DISPLAY"] == ":20"
    assert env["XAUTHORITY"] == "/run/hermes/bot-desktop/xauth"
    assert env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/hermes/bot-desktop/bus"
    # X11 desktop: a leaked Wayland socket flips GTK/Chromium backends
    assert "WAYLAND_DISPLAY" not in env


def test_running_bot_desktop_beats_the_session_snapshot(monkeypatch):
    _publish(monkeypatch, {"DISPLAY": ":20"})
    monkeypatch.setenv("DISPLAY", ":0")
    env = _make_run_env({"DISPLAY": ":0"})  # login snapshot captured the seat display
    assert env["DISPLAY"] == ":20"


def test_stopped_bot_desktop_leaves_the_seat_env_alone(monkeypatch):
    _publish(monkeypatch, {})
    monkeypatch.setenv("DISPLAY", ":0")
    env = _make_run_env({})
    assert env["DISPLAY"] == ":0"
    assert "XAUTHORITY" not in env or env.get("XAUTHORITY") == ""


def test_routing_does_not_stamp_bot_desktop_activity(monkeypatch):
    stamps = []
    _publish(monkeypatch, {"DISPLAY": ":20"})
    monkeypatch.setattr(
        "tools.bot_desktop.runtime.touch_activity", lambda: stamps.append(1)
    )
    _make_run_env({})
    # A plain terminal command is not screen use; idle_stop_minutes must stay effective.
    assert stamps == []


def test_a_wayland_hosts_scratch_tmpdir_and_runtime_dir_are_scrubbed(
    monkeypatch, tmp_path
):
    _publish(monkeypatch, {"DISPLAY": ":20"})
    monkeypatch.setattr("tools.bot_desktop.runtime.state_dir", lambda: tmp_path)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    deep = (
        "/tmp/" + "s" * 60
    )  # past SOCKET_TMPDIR_MAX_LEN: Chromium's singleton socket cannot bind under it
    monkeypatch.setenv("TMPDIR", deep)
    env = _make_run_env({})
    assert env["TMPDIR"] != deep
    assert len(env["TMPDIR"]) <= 50  # the 108-byte sun_path budget is respected again
    runtime_dir = tmp_path / "runtime-dir"
    # wl_display_connect(NULL)'s implicit wayland-0 fallback fails closed on the screen-owned dir
    assert env["XDG_RUNTIME_DIR"] == str(runtime_dir)
    assert runtime_dir.is_dir()
    assert runtime_dir.stat().st_mode & 0o777 == 0o700
    assert "WAYLAND_DISPLAY" not in env


def test_a_short_tmpdir_is_kept_but_the_runtime_dir_is_still_owned(
    monkeypatch, tmp_path
):
    _publish(monkeypatch, {"DISPLAY": ":20"})
    monkeypatch.setattr("tools.bot_desktop.runtime.state_dir", lambda: tmp_path)
    monkeypatch.setenv("TMPDIR", "/tmp")
    env = _make_run_env({})
    assert env["TMPDIR"] == "/tmp"
    assert env["XDG_RUNTIME_DIR"] == str(tmp_path / "runtime-dir")


def test_an_uncreatable_runtime_dir_is_dropped_not_kept(monkeypatch, tmp_path):
    _publish(monkeypatch, {"DISPLAY": ":20"})
    blocker = tmp_path / "file"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setattr(
        "tools.bot_desktop.runtime.state_dir", lambda: blocker / "under"
    )
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    env = _make_run_env({})
    # The host desktop's socket must not survive a failed runtime-dir creation
    assert "XDG_RUNTIME_DIR" not in env


def test_desktop_env_applies_the_same_screen_fixups(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tools.bot_desktop.runtime.published_env", lambda: {"DISPLAY": ":20"}
    )
    monkeypatch.setattr("tools.bot_desktop.runtime.touch_activity", lambda: None)
    monkeypatch.setattr("tools.bot_desktop.runtime.state_dir", lambda: tmp_path)
    monkeypatch.setattr("tools.bot_desktop.browser.env_for_agent", lambda env: env)
    from tools.bot_desktop.runtime import desktop_env

    env = desktop_env(
        {
            "DISPLAY": ":0",
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "TMPDIR": "/tmp/" + "s" * 60,
        }
    )
    assert env["DISPLAY"] == ":20"
    assert "WAYLAND_DISPLAY" not in env
    assert len(env["TMPDIR"]) <= 50
    assert env["XDG_RUNTIME_DIR"] == str(tmp_path / "runtime-dir")
