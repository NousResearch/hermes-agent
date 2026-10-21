"""The standalone stdio entry is never a second state.db writer beside a gateway."""
from __future__ import annotations

import io

import pytest

from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership
from tui_gateway import entry


def _stub_startup(monkeypatch, opened):
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry.server, "_stdio_is_rpc_channel", False, raising=False)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_start_backend_heartbeat_refresher", lambda: opened.append("state.db"))
    monkeypatch.setattr(entry.server, "_schedule_startup_orphan_sweep", lambda: None)
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *_a: False)
    monkeypatch.setattr(entry, "_log_exit", lambda reason: opened.append(reason))
    monkeypatch.setattr(entry, "write_json", lambda payload: True)


def test_entry_refuses_an_owned_home_and_owns_a_free_one(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    opened: list[str] = []
    _stub_startup(monkeypatch, opened)
    gateway = ProfileOwnership()
    gateway.reserve([home])
    try:
        monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))
        with pytest.raises(SystemExit) as refused:
            entry.main()
        assert refused.value.code == 1
        assert "state.db" not in opened and "owned by a running gateway" in opened[-1]
    finally:
        gateway.release(home)

    # A free home: the entry holds the gateway's own reservation while it serves, so a
    # gateway starting on that home is refused instead of becoming a second writer.
    contender = []

    def dispatch(req):
        with pytest.raises(OwnershipConflict):
            ProfileOwnership().reserve([home])
        contender.append("refused")
        return {"jsonrpc": "2.0", "id": req["id"], "result": {}}

    monkeypatch.setattr(entry, "dispatch", dispatch)
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\n'))
    entry.main()
    assert contender == ["refused"]
    after = ProfileOwnership()
    after.reserve([home])  # released at exit
    after.release(home)
