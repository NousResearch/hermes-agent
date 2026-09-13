"""display.install runs its worker inside the caller's profile scope."""

from __future__ import annotations

import threading

import pytest


def test_install_worker_keeps_the_requested_profile_scope(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from tools.bot_desktop import install, runtime
    import tui_gateway.server as server

    named = tmp_path / "profiles" / "named"
    named.mkdir(parents=True)
    monkeypatch.setattr(server, "_profile_home", lambda name: str(named) if name == "named" else None)
    monkeypatch.setattr(runtime, "is_supported_host", lambda: True)
    monkeypatch.setattr(runtime, "install_command", lambda: "sudo apt-get install -y x")
    seen = {}
    done = threading.Event()

    def fake_install(*, ask_password, on_line, timeout_seconds=900.0):
        seen["home"] = str(get_hermes_home())
        done.set()
        return 0

    monkeypatch.setattr(install, "install_packages", fake_install)
    monkeypatch.setattr(server, "_broadcast_global_event", lambda *a, **k: None)
    resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "display.install", "params": {"profile": "named"}})
    assert resp["result"]["started"], resp
    assert done.wait(5)
    assert seen["home"] == str(named)


@pytest.fixture
def _fresh_lease():
    from tools.bot_desktop import lease
    lease._reset_for_tests()
    yield lease
    lease._reset_for_tests()


def _call(server, method, params):
    return server.handle_request({"jsonrpc": "2.0", "id": 7, "method": method, "params": params})


def test_thumbnail_is_suppressed_while_a_human_holds_the_lease(monkeypatch, _fresh_lease):
    """The Desktop polls thumbnails on a timer; while a human drives the screen that grab would ship
    whatever they are typing to every connected client, so it must not touch the framebuffer at all."""
    import tui_gateway.server as server
    from tools.bot_desktop import thumbnail

    grabs = []
    monkeypatch.setattr(thumbnail, "thumbnail_data_url", lambda: grabs.append(1) or "data:image/jpeg;base64,SECRET")
    _fresh_lease.acquire("viewer-1")
    result = _call(server, "display.thumbnail", {})["result"]
    assert result["data_url"] is None and result["suppressed"] == "human_has_control"
    assert grabs == [], "the framebuffer was grabbed while a human held the lease"
    _fresh_lease.release("viewer-1")
    assert _call(server, "display.thumbnail", {})["result"]["data_url"].endswith("SECRET")


def test_release_without_viewer_id_cannot_yank_another_viewers_lease(_fresh_lease):
    """lease.release(None) skips the holder check, so a client that lost its viewer id (or a bare RPC)
    must be refused unless it forces; a matching viewer id and force keep working."""
    import tui_gateway.server as server

    _fresh_lease.acquire("viewer-1")
    refused = _call(server, "display.lease.release", {})
    assert refused["error"]["data"]["code"] == "viewer_mismatch"
    assert _fresh_lease.get().holder == _fresh_lease.HUMAN
    assert _call(server, "display.lease.release", {"viewer_id": "viewer-1"})["result"]["lease"]["holder"] == _fresh_lease.AGENT
    _fresh_lease.acquire("viewer-2")
    assert _call(server, "display.lease.release", {"force": True})["result"]["lease"]["holder"] == _fresh_lease.AGENT
