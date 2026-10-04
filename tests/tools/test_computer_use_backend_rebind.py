"""A cached computer_use backend is bound to the display it spawned on; the identity helpers notice a change."""

from __future__ import annotations

from tools.computer_use import cua_backend


def test_backend_display_identity_tracks_the_display_a_spawn_would_get():
    before = cua_backend.desktop_identity({"HOME": "/x"})  # no screen yet
    after = cua_backend.desktop_identity({"HOME": "/x", "DISPLAY": ":37"})  # Bot Desktop came up
    assert before == "" and after == ":37"
    assert cua_backend.backend_display_stale(before, after)
    assert not cua_backend.backend_display_stale(after, cua_backend.desktop_identity({"DISPLAY": ":37"}))


def _native_wayland_env(**over):
    env = {"DISPLAY": ":42", "WAYLAND_DISPLAY": "wayland-0", "XDG_RUNTIME_DIR": "/run/user/1000",
           "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus", "AT_SPI_BUS_ADDRESS": "unix:path=/tmp/at-spi"}
    env.update(over)
    return env


def test_native_wayland_identity_covers_session_endpoint_changes(monkeypatch):
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"native_wayland": True})
    monkeypatch.setattr(cua_backend.sys, "platform", "linux")
    base = _native_wayland_env()
    # Same DISPLAY, a restarted compositor session: not the same desktop, so not the same identity.
    rebound = _native_wayland_env(WAYLAND_DISPLAY="wayland-7", DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus2")
    assert cua_backend.desktop_identity(base) != cua_backend.desktop_identity(rebound)
    assert cua_backend.backend_display_stale(cua_backend.desktop_identity(base), cua_backend.desktop_identity(rebound))
    # A rebound AT-SPI or runtime dir is equally a different session.
    assert cua_backend.desktop_identity(base) != cua_backend.desktop_identity(_native_wayland_env(
        AT_SPI_BUS_ADDRESS="unix:path=/tmp/at-spi-2", XDG_RUNTIME_DIR="/run/user/1001"))
    # Unrelated environment keys must not spuriously retire the backend; an A→B→A round trip comes back equal.
    assert cua_backend.desktop_identity(base) == cua_backend.desktop_identity(
        _native_wayland_env(PATH="/other", HOME="/h", TERM="xterm"))
    # A spawn env that already carries the bridge marker (the child-env builder's output) is also scoped.
    assert cua_backend.desktop_identity(_native_wayland_env(CUA_DRIVER_RS_ENABLE_WAYLAND="1")) != ":42"


def test_x11_identity_stays_display_only_without_the_wayland_bridge(monkeypatch):
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"native_wayland": True})
    monkeypatch.setattr(cua_backend.sys, "platform", "linux")
    # A Bot Desktop pops WAYLAND_DISPLAY before the spawn: the legacy X11 value, even with the config on.
    assert cua_backend.desktop_identity({"DISPLAY": ":42", "XDG_RUNTIME_DIR": "/run/user/1000"}) == ":42"
    # No Wayland socket in the env: empty DISPLAY keeps returning the empty identity.
    assert cua_backend.desktop_identity({"HOME": "/x"}) == ""
    # Config off (the default) and no bridge marker: a Wayland env still gets the DISPLAY-only value.
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {})
    assert cua_backend.desktop_identity(_native_wayland_env()) == ":42"
