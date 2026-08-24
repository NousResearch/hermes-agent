"""E2E tests for the per-profile MCP lifecycle RPCs (mcp.servers.*).

These drive the real registered gateway handlers against a real temp
``HERMES_HOME`` with named profile dirs — no mocks of the config/mcp layer — and
assert that every write lands in the RIGHT profile's ``config.yaml`` / ``.env``
and NEVER leaks into the launch (default) profile.

Covered: add + list + set_api_key + remove, profile isolation, and the
duplicate/not-found error envelopes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import tui_gateway.server as server


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    """A temp HERMES_HOME root with two named profiles: 'work' and 'other'.

    Pointing HERMES_HOME at a dir outside ~/.hermes makes it the profile ROOT
    (get_default_hermes_root's Docker/custom branch), so named profiles live at
    ``<root>/profiles/<name>/`` and the launch/default profile is ``<root>``.
    """
    root = tmp_path / "hermes_home"
    (root / "profiles" / "work").mkdir(parents=True)
    (root / "profiles" / "other").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    # Make sure no stale process-wide home override leaks in from another test.
    from hermes_constants import get_hermes_home_override

    assert get_hermes_home_override() is None
    return root


def _call(method, params=None):
    handler = server._methods[method]
    return handler(1, params or {})


def _result(resp):
    assert "error" not in resp, resp.get("error")
    return resp["result"]


def _read_yaml(path: Path) -> dict:
    """Read a config.yaml directly for assertions (test-side, not the guarded loader)."""
    import yaml

    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_add_lands_in_named_profile_only(hermes_root):
    root = hermes_root
    resp = _call(
        "mcp.servers.add",
        {
            "profile": "work",
            "name": "weather",
            "config": {"url": "https://mcp.example.com/weather"},
        },
    )
    result = _result(resp)
    assert result["ok"] is True
    assert result["server"]["transport"] == "http"
    assert result["server"]["url"] == "https://mcp.example.com/weather"

    work_cfg = _read_yaml(root / "profiles" / "work" / "config.yaml")
    assert "weather" in work_cfg.get("mcp_servers", {})
    assert work_cfg["mcp_servers"]["weather"]["url"] == "https://mcp.example.com/weather"

    # The launch/default profile and the sibling profile stay untouched.
    default_cfg = _read_yaml(root / "config.yaml")
    assert "weather" not in default_cfg.get("mcp_servers", {})
    other_cfg = _read_yaml(root / "profiles" / "other" / "config.yaml")
    assert "weather" not in other_cfg.get("mcp_servers", {})


def test_list_reflects_the_scoped_profile(hermes_root):
    _result(
        _call(
            "mcp.servers.add",
            {"profile": "work", "name": "svc-a", "config": {"command": "svc-a-bin"}},
        )
    )
    _result(
        _call(
            "mcp.servers.add",
            {"profile": "other", "name": "svc-b", "config": {"command": "svc-b-bin"}},
        )
    )

    work_names = [s["name"] for s in _result(_call("mcp.servers.list", {"profile": "work"}))["servers"]]
    other_names = [s["name"] for s in _result(_call("mcp.servers.list", {"profile": "other"}))["servers"]]

    assert work_names == ["svc-a"]
    assert other_names == ["svc-b"]

    # stdio transport surfaced correctly.
    work_server = _result(_call("mcp.servers.list", {"profile": "work"}))["servers"][0]
    assert work_server["transport"] == "stdio"
    assert work_server["command"] == "svc-a-bin"


def test_set_api_key_writes_env_and_header_to_right_profile(hermes_root):
    root = hermes_root
    _result(
        _call(
            "mcp.servers.add",
            {
                "profile": "work",
                "name": "gizmo",
                "config": {"url": "https://mcp.example.com/gizmo"},
            },
        )
    )

    resp = _result(
        _call(
            "mcp.servers.set_api_key",
            {"profile": "work", "name": "gizmo", "value": "sk-secret-123"},
        )
    )
    assert resp["ok"] is True
    env_var = resp["env_var"]
    assert env_var == "MCP_GIZMO_API_KEY"

    # The secret is in the work profile's .env — and NOT the default profile's.
    work_env = (root / "profiles" / "work" / ".env").read_text(encoding="utf-8")
    assert "MCP_GIZMO_API_KEY=sk-secret-123" in work_env
    assert not (root / ".env").exists() or "sk-secret-123" not in (root / ".env").read_text(
        encoding="utf-8"
    )

    # config.yaml stores only the interpolation template, never the raw secret.
    work_cfg = _read_yaml(root / "profiles" / "work" / "config.yaml")
    headers = work_cfg["mcp_servers"]["gizmo"]["headers"]
    assert headers["Authorization"] == "Bearer ${MCP_GIZMO_API_KEY}"
    assert "sk-secret-123" not in str(work_cfg)


def test_set_api_key_stdio_references_env_block(hermes_root):
    root = hermes_root
    _result(
        _call(
            "mcp.servers.add",
            {"profile": "work", "name": "localtool", "config": {"command": "localtool-bin"}},
        )
    )
    resp = _result(
        _call(
            "mcp.servers.set_api_key",
            {
                "profile": "work",
                "name": "localtool",
                "env_var": "LOCALTOOL_TOKEN",
                "value": "tok-xyz",
            },
        )
    )
    assert resp["env_var"] == "LOCALTOOL_TOKEN"

    work_cfg = _read_yaml(root / "profiles" / "work" / "config.yaml")
    env_block = work_cfg["mcp_servers"]["localtool"]["env"]
    assert env_block["LOCALTOOL_TOKEN"] == "${LOCALTOOL_TOKEN}"
    work_env = (root / "profiles" / "work" / ".env").read_text(encoding="utf-8")
    assert "LOCALTOOL_TOKEN=tok-xyz" in work_env


def test_remove_scoped_to_profile(hermes_root):
    root = hermes_root
    _result(
        _call(
            "mcp.servers.add",
            {"profile": "work", "name": "temp", "config": {"command": "temp-bin"}},
        )
    )
    # Same-named server in a different profile must be unaffected by the remove.
    _result(
        _call(
            "mcp.servers.add",
            {"profile": "other", "name": "temp", "config": {"command": "temp-bin"}},
        )
    )

    resp = _result(_call("mcp.servers.remove", {"profile": "work", "name": "temp"}))
    assert resp["removed"] is True

    assert "temp" not in _read_yaml(root / "profiles" / "work" / "config.yaml").get("mcp_servers", {})
    # The 'other' profile still has its server.
    assert "temp" in _read_yaml(root / "profiles" / "other" / "config.yaml").get("mcp_servers", {})


def test_add_duplicate_and_missing_errors(hermes_root):
    _result(
        _call(
            "mcp.servers.add",
            {"profile": "work", "name": "dup", "config": {"command": "dup-bin"}},
        )
    )
    dup = _call(
        "mcp.servers.add",
        {"profile": "work", "name": "dup", "config": {"command": "dup-bin"}},
    )
    assert "error" in dup
    assert dup["error"]["code"] == 4090

    missing = _call("mcp.servers.remove", {"profile": "work", "name": "nope"})
    assert "error" in missing
    assert missing["error"]["code"] == 4064

    bad_profile = _call(
        "mcp.servers.add",
        {"profile": "ghost", "name": "x", "config": {"command": "x"}},
    )
    assert "error" in bad_profile
    assert bad_profile["error"]["code"] == 4064


def test_add_requires_transport(hermes_root):
    resp = _call("mcp.servers.add", {"profile": "work", "name": "empty", "config": {}})
    assert "error" in resp
    assert resp["error"]["code"] == 4063


def test_default_profile_add_when_profile_omitted(hermes_root):
    root = hermes_root
    _result(
        _call(
            "mcp.servers.add",
            {"name": "rootsvc", "config": {"command": "rootsvc-bin"}},
        )
    )
    # Omitted profile → launch/default profile == HERMES_HOME root config.yaml.
    default_cfg = _read_yaml(root / "config.yaml")
    assert "rootsvc" in default_cfg.get("mcp_servers", {})
    # ...and NOT in a named profile.
    assert "rootsvc" not in _read_yaml(root / "profiles" / "work" / "config.yaml").get(
        "mcp_servers", {}
    )
