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


def test_published_xdg_dirs_do_not_leak_into_the_terminal(monkeypatch):
    # The launcher publishes its private XDG_* trio alongside the display keys; a GUI app
    # launched from the terminal must keep the user's own ~/.config state, not the Bot
    # Screen profile's directories (AI review: unintended scope in the full merge).
    _publish(
        monkeypatch,
        {
            "DISPLAY": ":20",
            "XDG_CONFIG_HOME": "/run/hermes/bot-desktop/xdg/config",
            "XDG_CACHE_HOME": "/run/hermes/bot-desktop/xdg/cache",
            "XDG_DATA_HOME": "/run/hermes/bot-desktop/xdg/data",
        },
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", "/home/user/.config")
    monkeypatch.setenv("XDG_CACHE_HOME", "/home/user/.cache")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    env = _make_run_env({})
    assert env["DISPLAY"] == ":20"
    assert env["XDG_CONFIG_HOME"] == "/home/user/.config"
    assert env["XDG_CACHE_HOME"] == "/home/user/.cache"
    assert "XDG_DATA_HOME" not in env  # not injected when the seat never set one


def test_running_desktop_pins_the_agent_browser_identity(monkeypatch):
    # Same pin as runtime.desktop_env(): the image's boot hook exports a headless-shell
    # AGENT_BROWSER_EXECUTABLE_PATH; with the terminal now routing to the Bot Screen's
    # DISPLAY, keeping it would launch the browser with no window (singleton over one
    # user-data-dir). A user's own pin must survive.
    _publish(monkeypatch, {"DISPLAY": ":20"})
    monkeypatch.setattr(
        "tools.bot_desktop.browser.profile_dir",
        lambda: __import__("pathlib").Path("/run/hermes/bot-desktop/profile"),
    )
    monkeypatch.setattr(
        "tools.bot_desktop.browser.executable", lambda: "/opt/hermes/chromium"
    )
    monkeypatch.setenv(
        "AGENT_BROWSER_EXECUTABLE_PATH", "/usr/bin/chrome-headless-shell"
    )
    monkeypatch.delenv("AGENT_BROWSER_PROFILE", raising=False)
    env = _make_run_env({})
    assert env["AGENT_BROWSER_EXECUTABLE_PATH"] == "/opt/hermes/chromium"
    assert env["AGENT_BROWSER_PROFILE"] == "/run/hermes/bot-desktop/profile"

    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", "/opt/custom/firefox")
    env = _make_run_env({})
    assert (
        env["AGENT_BROWSER_EXECUTABLE_PATH"] == "/opt/custom/firefox"
    )  # user pin wins


def test_routing_does_not_stamp_bot_desktop_activity(monkeypatch):
    stamps = []
    _publish(monkeypatch, {"DISPLAY": ":20"})
    monkeypatch.setattr(
        "tools.bot_desktop.runtime.touch_activity", lambda: stamps.append(1)
    )
    _make_run_env({})
    # A plain terminal command is not screen use; idle_stop_minutes must stay effective.
    assert stamps == []
