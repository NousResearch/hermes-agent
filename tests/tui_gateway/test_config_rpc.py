"""Tests for JSON-safe TUI config RPC responses."""

from __future__ import annotations

import json

import tui_gateway.server as server


def test_full_config_serializes_yaml_timestamp(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(
        """\
plugins:
  entries:
    delegation-guard:
      capabilities_consent:
        granted_at: 2026-08-17 14:50:10+00:00
"""
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_cfg_mtime", None)
    monkeypatch.setattr(server, "_cfg_path", None)

    response = server._methods["config.get"](1, {"key": "full"})

    assert response["result"]["config"]["plugins"]["entries"]["delegation-guard"][
        "capabilities_consent"
    ]["granted_at"] == "2026-08-17T14:50:10+00:00"
    json.dumps(response)
