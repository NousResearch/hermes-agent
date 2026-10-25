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
