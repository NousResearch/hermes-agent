"""The gateway.ready handshake carries this gateway's device identity.

Clients keep the FIRST ready frame (their local gateway, connected at boot)
as their own device name for cross-device sender attribution (channels
Phase 2b); the field must exist and never break the handshake.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def ws_module():
    with patch.dict(sys.modules, {
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib

        mod = importlib.import_module("tui_gateway.ws")
        yield mod


def test_ready_payload_includes_device_name(ws_module, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(ws_module.server, "resolve_skin", lambda: {"name": "noir"})
    monkeypatch.setattr(hermes_constants, "get_device_name", lambda: "ko-mac")

    payload = ws_module._ready_payload()

    assert payload["skin"] == {"name": "noir"}
    assert payload["device_name"] == "ko-mac"


def test_ready_payload_survives_resolver_failure(ws_module, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(ws_module.server, "resolve_skin", lambda: {})

    def _boom():
        raise RuntimeError("resolver down")

    monkeypatch.setattr(hermes_constants, "get_device_name", _boom)

    payload = ws_module._ready_payload()

    assert payload["device_name"] == ""
