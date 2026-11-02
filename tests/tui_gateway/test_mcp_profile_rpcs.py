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
from types import SimpleNamespace

import pytest

from tui_gateway import server


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
    import hermes_yaml as yaml

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


def test_status_is_profile_scoped_and_credential_safe(hermes_root):
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

    payload = _result(_call("mcp.servers.status", {"profile": "work"}))

    assert payload["checked_at"] > 0
    assert payload["servers"] == [
        {
            "name": "svc-a",
            "transport": "stdio",
            "tools": 0,
            "connected": False,
            "disabled": False,
            "status": "configured",
            "source": "config",
            "plugin": None,
        }
    ]
    assert "error" not in str(payload)


def test_status_does_not_mix_launch_runtime_into_another_profile(hermes_root):
    from tools import mcp_tool

    _result(
        _call(
            "mcp.servers.add",
            {"profile": "work", "name": "shared", "config": {"command": "work-bin"}},
        )
    )
    launch_server = SimpleNamespace(
        session=object(),
        _registered_tool_names=["launch_secret_tool"],
        _tools=[],
        _sampling=None,
    )
    with mcp_tool._lock:
        saved_servers = dict(mcp_tool._servers)
        saved_scopes = dict(mcp_tool._server_scope_keys)
        mcp_tool._servers["shared"] = launch_server
        mcp_tool._server_scope_keys.pop("shared", None)

    try:
        payload = _result(_call("mcp.servers.status", {"profile": "work"}))
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._servers.update(saved_servers)
            mcp_tool._server_scope_keys.clear()
            mcp_tool._server_scope_keys.update(saved_scopes)

    assert payload["servers"][0]["status"] == "configured"
    assert payload["servers"][0]["tools"] == 0


def test_status_includes_named_profile_runtime_in_multiplex(hermes_root):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_constants import (
        hermes_home_key,
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools import mcp_tool

    _result(
        _call(
            "mcp.servers.add",
            {"profile": "work", "name": "shared", "config": {"command": "work-bin"}},
        )
    )
    work_token = set_hermes_home_override(hermes_root / "profiles" / "work")
    try:
        work_scope = hermes_home_key()
    finally:
        reset_hermes_home_override(work_token)

    work_server = SimpleNamespace(
        session=object(),
        _registered_tool_names=["work_tool"],
        _tools=[],
        _sampling=None,
    )
    previous_multiplex = is_multiplex_active()
    with mcp_tool._lock:
        saved_servers = dict(mcp_tool._servers)
        saved_scopes = dict(mcp_tool._server_scope_keys)
        mcp_tool._servers["shared"] = work_server  # type: ignore[assignment]
        mcp_tool._server_scope_keys["shared"] = work_scope

    set_multiplex_active(True)
    try:
        payload = _result(_call("mcp.servers.status", {"profile": "work"}))
    finally:
        set_multiplex_active(previous_multiplex)
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._servers.update(saved_servers)
            mcp_tool._server_scope_keys.clear()
            mcp_tool._server_scope_keys.update(saved_scopes)

    assert payload["servers"][0]["status"] == "connected"
    assert payload["servers"][0]["tools"] == 1


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


# A former bundled-catalog id: no CLI preset carries it.
FORMER_CATALOG_ID = "linear"


@pytest.fixture
def no_catalog(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "hermes_cli.mcp_catalog", None)


def _saved_server(root: Path, profile: str, name: str) -> dict:
    path = root / "profiles" / profile / "config.yaml" if profile else root / "config.yaml"
    return (_read_yaml(path).get("mcp_servers") or {}).get(name) or {}


def test_mcp_catalog_is_an_empty_list_that_matches_its_contract(hermes_root, no_catalog):
    """Older Bot Mode builds and catalog plugins still call mcp.catalog; it answers empty."""
    from tui_gateway.contracts.tools_mcp_plugins import McpCatalogResult

    result = _result(_call("mcp.catalog", {"profile": "work"}))
    assert result == {"servers": []}
    McpCatalogResult.model_validate(result)


@pytest.mark.parametrize("params", [
    {"profile": "work", "name": FORMER_CATALOG_ID, "preset": FORMER_CATALOG_ID},  # old Bot Mode add
    {"name": FORMER_CATALOG_ID, "preset": FORMER_CATALOG_ID},  # old SDK completeMcpOAuth({catalogPreset})
])
def test_a_former_catalog_id_without_a_transport_returns_4063_and_writes_nothing(hermes_root, no_catalog, params):
    resp = _call("mcp.servers.add", params)

    assert resp["error"]["code"] == 4063
    assert "catalog" not in resp["error"]["message"]
    assert not _saved_server(hermes_root, "work", FORMER_CATALOG_ID)
    assert not _saved_server(hermes_root, "", FORMER_CATALOG_ID)


def test_a_former_catalog_id_beside_an_explicit_url_adds_the_server(hermes_root, no_catalog):
    result = _result(
        _call(
            "mcp.servers.add",
            {
                "profile": "work",
                "name": FORMER_CATALOG_ID,
                "preset": FORMER_CATALOG_ID,
                "config": {"url": "https://override.example/mcp"},
            },
        )
    )

    assert result["server"]["url"] == "https://override.example/mcp"
    assert _saved_server(hermes_root, "work", FORMER_CATALOG_ID)["url"] == "https://override.example/mcp"
    assert not _saved_server(hermes_root, "", FORMER_CATALOG_ID)


def test_unknown_preset_returns_4063_and_writes_nothing(hermes_root):
    resp = _call(
        "mcp.servers.add",
        {"profile": "work", "name": "unknown", "preset": "not-a-preset"},
    )

    assert resp["error"]["code"] == 4063
    assert not _saved_server(hermes_root, "work", "unknown")
    assert "unknown" not in (_read_yaml(hermes_root / "config.yaml").get("mcp_servers") or {})


def test_cli_preset_still_fills_transport(hermes_root, no_catalog):
    from hermes_cli import mcp_config

    preset_name, expected = next(iter(mcp_config._MCP_PRESETS.items()))
    result = _result(
        _call(
            "mcp.servers.add",
            {"profile": "work", "name": "cli-preset", "preset": preset_name},
        )
    )

    saved = _saved_server(hermes_root, "work", "cli-preset")
    assert result["server"]["command"] == expected["command"]
    assert saved["command"] == expected["command"]
    assert saved.get("args") == list(expected.get("args") or [])


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


def test_test_resolves_env_refs_from_requested_profile_secret_scope(hermes_root, monkeypatch):
    """``mcp.servers.test`` for a secondary must expand its ``${VAR}`` header from THAT profile's
    secret scope, not the launch process's ``os.environ`` (the default profile's value) — the
    Desktop MCP setup "Test connection" otherwise reports green against the wrong credential.
    ``os.environ`` is never mutated by the scope."""
    from hermes_cli import mcp_config

    work = hermes_root / "profiles" / "work"
    (work / ".env").write_text("ALPHA_ONLY_TOKEN=work-token\n", encoding="utf-8")
    (work / "config.yaml").write_text(
        "mcp_servers:\n  srv:\n    url: http://x/mcp\n"
        "    headers:\n      Authorization: Bearer ${ALPHA_ONLY_TOKEN}\n", encoding="utf-8")
    monkeypatch.setenv("ALPHA_ONLY_TOKEN", "default-process-token")

    resolved = {}

    def fake_probe(name, config, connect_timeout=30, details=None):
        resolved.update(mcp_config._resolve_mcp_server_config(config).get("headers", {}))
        return [("tool-a", "desc")]

    monkeypatch.setattr(mcp_config, "_probe_single_server", fake_probe)
    result = _result(_call("mcp.servers.test", {"profile": "work", "name": "srv"}))

    assert result["ok"] is True
    assert resolved["Authorization"] == "Bearer work-token"
    assert os.environ["ALPHA_ONLY_TOKEN"] == "default-process-token"
