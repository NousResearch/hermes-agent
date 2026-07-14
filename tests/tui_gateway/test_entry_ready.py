"""Behavior tests for the stdio gateway.ready contract."""

import io

from hermes_cli import config as config_mod
from tui_gateway import entry


def test_stdio_gateway_ready_advertises_session_profile_capability(monkeypatch):
    """The stdio transport advertises the same optional RPC capability as WS."""
    sent = []
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(config_mod, "read_raw_config", lambda: {})
    monkeypatch.setattr(entry, "resolve_skin", lambda: {"name": "test"})
    monkeypatch.setattr(entry, "write_json", lambda frame: sent.append(frame) or True)
    monkeypatch.setattr(entry, "_log_exit", lambda _reason: None)
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))

    entry.main()

    assert sent[0]["params"]["type"] == "gateway.ready"
    assert sent[0]["params"]["payload"]["capabilities"]["session_profiles"] is True
