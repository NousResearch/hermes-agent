"""Consent return routing must respect an isolated desktop's missing protocol handler."""

import pytest

from tools.connectors.gateway import client
from tools.connectors.gateway.config import ConnectorConfig


@pytest.mark.parametrize("dev,expected", [(False, "hermes-desktop"), (True, "hermes-desktop-dev")])
def test_normal_desktop_keeps_its_consent_return(monkeypatch, dev, expected):
    monkeypatch.setattr(client, "session_platform", lambda: "desktop")
    monkeypatch.setattr(client, "load_config", lambda: ConnectorConfig())
    if dev:
        monkeypatch.setenv("HERMES_DESKTOP_DEV_SERVER", "http://127.0.0.1:5194")
    else:
        monkeypatch.delenv("HERMES_DESKTOP_DEV_SERVER", raising=False)
    assert client.return_to_args(op="fixture-op") == {"return_to": expected, "op": "fixture-op"}


def test_rehearsal_consent_cannot_open_another_apps_protocol_handler(monkeypatch, tmp_path):
    # Read the actual profile config, rather than bypassing the consuming loader.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("tools:\n  connectors:\n    browser_return: false\n", encoding="utf-8")
    monkeypatch.setattr(client, "session_platform", lambda: "desktop")
    monkeypatch.setenv("HERMES_DESKTOP_DEV_SERVER", "http://127.0.0.1:5194")
    assert client.return_to_args(op="fixture-op") == {}


def test_other_clients_keep_vendor_done_page(monkeypatch):
    monkeypatch.setattr(client, "session_platform", lambda: "web")
    assert client.return_to_args(op="fixture-op") == {}
