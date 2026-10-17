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


def test_terminal_commands_get_the_x11_display_layer_only(monkeypatch):
    """While a screen runs, every terminal command is pointed at its X11 — the backend pins
    launcher.sh already exports for dock launches — but the session's real XDG_RUNTIME_DIR/TMPDIR
    stay: ``systemctl --user``, rootless podman, agent sockets and ``pactl`` live under them, and
    the scratch dir keeps its isolation/cleanup (#136677 review scope pin)."""
    _publish(monkeypatch, {"DISPLAY": ":20"})
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("GDK_BACKEND", "wayland")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    deep = "/tmp/" + "s" * 60  # past SOCKET_TMPDIR_MAX_LEN — still kept on this path
    monkeypatch.setenv("TMPDIR", deep)
    env = _make_run_env({})
    assert "WAYLAND_DISPLAY" not in env
    assert env["XDG_SESSION_TYPE"] == "x11"
    assert env["GDK_BACKEND"] == "x11"
    assert env["QT_QPA_PLATFORM"] == "xcb"
    assert env["XDG_RUNTIME_DIR"] == "/run/user/1000"  # ordinary commands keep their sockets
    assert env["TMPDIR"] == deep  # and their scratch dir


def _screen_spawns(monkeypatch, tmp_path):
    """A live screen for ``desktop_env()``: published display env, no activity stamp, state dir
    under tmp_path, and the browser env pass-through neutralized."""
    monkeypatch.setattr("tools.bot_desktop.runtime.published_env", lambda: {"DISPLAY": ":20"})
    monkeypatch.setattr("tools.bot_desktop.runtime.touch_activity", lambda: None)
    monkeypatch.setattr("tools.bot_desktop.runtime.state_dir", lambda: tmp_path)
    monkeypatch.setattr("tools.bot_desktop.browser.env_for_agent", lambda env: env)


def test_screen_spawns_swap_a_deep_tmpdir_and_own_the_runtime_dir(monkeypatch, tmp_path):
    """Spawns dedicated to the screen (desktop_env) get the isolation layer: a deep scratch TMPDIR
    overflows Chromium's 108-byte singleton-socket sun_path, and the host's XDG_RUNTIME_DIR lets
    wl_display_connect(NULL)'s implicit wayland-0 fallback reach the real desktop's compositor."""
    _screen_spawns(monkeypatch, tmp_path)
    from tools.bot_desktop.runtime import desktop_env

    env = desktop_env(
        {
            "DISPLAY": ":0",
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_SESSION_TYPE": "wayland",
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "TMPDIR": "/tmp/" + "s" * 60,
        }
    )
    assert env["DISPLAY"] == ":20"
    assert "WAYLAND_DISPLAY" not in env
    assert env["XDG_SESSION_TYPE"] == "x11"
    assert len(env["TMPDIR"]) <= 50  # the 108-byte sun_path budget is respected again
    runtime_dir = tmp_path / "runtime-dir"
    assert env["XDG_RUNTIME_DIR"] == str(runtime_dir)
    assert runtime_dir.is_dir()
    assert runtime_dir.stat().st_mode & 0o777 == 0o700


def test_a_short_tmpdir_is_kept_but_the_runtime_dir_is_still_owned(monkeypatch, tmp_path):
    _screen_spawns(monkeypatch, tmp_path)
    from tools.bot_desktop.runtime import desktop_env

    env = desktop_env({"TMPDIR": "/tmp"})
    assert env["TMPDIR"] == "/tmp"
    assert env["XDG_RUNTIME_DIR"] == str(tmp_path / "runtime-dir")


def test_an_uncreatable_runtime_dir_is_dropped_not_kept(monkeypatch, tmp_path):
    _screen_spawns(monkeypatch, tmp_path)
    from tools.bot_desktop.runtime import desktop_env

    blocker = tmp_path / "file"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setattr(
        "tools.bot_desktop.runtime.state_dir", lambda: blocker / "under"
    )
    env = desktop_env({"XDG_RUNTIME_DIR": "/run/user/1000"})
    # The host desktop's socket must not survive a failed runtime-dir creation
    assert "XDG_RUNTIME_DIR" not in env
