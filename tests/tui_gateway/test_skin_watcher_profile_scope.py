"""The shared gateway's skin watcher repaints each attached TUI with ITS profile's skin.

Main ran one gateway process per profile, so ``hermes -p work --tui`` only ever saw the work home's
``display.skin``. A multiplexed gateway has one watcher thread (no profile override) for every peer:
it must not push the launch home's skin onto a work-profile TUI, and a work-profile skin switch made
outside the TUI (``hermes -p work config set display.skin …``) must still repaint that TUI.
"""
from __future__ import annotations

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _Peer:
    def __init__(self, skin_home=None):
        self.frames: list[dict] = []
        if skin_home is not None:
            self.skin_home = skin_home

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        pass


def _skins(peer):
    return [f["params"]["payload"]["name"] for f in peer.frames if f["params"]["type"] == "skin.changed"]


def _write_skin(home, name):
    (home / "config.yaml").write_text(f"display:\n  skin: {name}\n", encoding="utf-8")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from tui_gateway import server
    default, work = tmp_path / "default", tmp_path / "default" / "profiles" / "work"
    work.mkdir(parents=True)
    _write_skin(default, "default")
    _write_skin(work, "ares")
    monkeypatch.setattr(server, "_hermes_home", default)
    monkeypatch.setattr(server, "_last_skin_sig", None, raising=False)
    monkeypatch.setattr(server, "_skin_sig_by_home", {}, raising=False)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    yield server, default, work
    server._live_transports.clear()


def _under(home, fn):
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_watcher_never_pushes_the_launch_home_skin_onto_a_profile_tui(homes):
    server, _default, work = homes
    peer = _Peer(skin_home=str(work))
    server.register_live_transport(peer)
    # The first attach seeds the baseline inside the work connection's scope (ws.handle_ws).
    _under(work, server._note_skin_broadcast)

    server._broadcast_skin_if_changed()  # the watcher thread: no override

    assert _skins(peer) == []


def test_profile_skin_switch_repaints_only_that_profiles_tui(homes):
    server, default, work = homes
    work_peer, default_peer = _Peer(skin_home=str(work)), _Peer()
    server.register_live_transport(work_peer)
    server.register_live_transport(default_peer)
    server._note_skin_broadcast()
    server._broadcast_skin_if_changed()  # seeds the work home's baseline without a repaint

    _write_skin(work, "mono")
    server._broadcast_skin_if_changed()
    assert _skins(work_peer) == ["mono"]
    assert _skins(default_peer) == []

    _write_skin(default, "slate")
    server._broadcast_skin_if_changed()
    assert _skins(default_peer) == ["slate"]
    assert _skins(work_peer) == ["mono"]


def test_owner_skin_rpc_baseline_is_per_profile(homes):
    """``/skin`` on the work owner records the work baseline; the launch home's stays put."""
    server, _default, work = homes
    peer = _Peer(skin_home=str(work))
    server.register_live_transport(peer)
    server._note_skin_broadcast()
    _under(work, server._note_skin_broadcast)
    _write_skin(work, "mono")
    _under(work, server._note_skin_broadcast)  # the owner broadcast mono itself

    server._broadcast_skin_if_changed()

    assert _skins(peer) == []
