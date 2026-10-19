"""The display WebSocket can run in a process with an empty terminal registry."""

import pytest

from tools.bot_desktop import runtime, sandbox_host


def test_rfb_relay_reattaches_to_the_screen_markers_container(monkeypatch):
    """A live marker, not a previously primed status() call, identifies the RFB owner."""
    owner = object()
    marker = {"backend": "DockerEnvironment", "container": "screen-container"}
    calls = []
    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: marker)
    monkeypatch.setattr(sandbox_host, "marker_sandbox_alive", lambda value: value is marker)
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: calls.append(create) or (owner if create else None))
    monkeypatch.setattr(runtime, "_profile_name", lambda: "wian")
    monkeypatch.setattr(sandbox_host, "open_rfb_stream", lambda env, profile: (env, profile))

    assert runtime.open_rfb_stream() == (owner, "wian")
    assert calls == [False, True]


def test_rfb_relay_does_not_make_a_new_container_when_screen_owner_is_gone(monkeypatch):
    marker = {"backend": "DockerEnvironment", "container": "removed-container"}
    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: marker)
    monkeypatch.setattr(sandbox_host, "marker_sandbox_alive", lambda value: False)
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: None if not create else pytest.fail("created a new container"))

    with pytest.raises(RuntimeError, match="sandbox hosting this screen is not running"):
        runtime.open_rfb_stream()
