"""profiles.describe / profiles.configure use the MCP ``enabled`` key the runtime reads.

Every runtime resolver (``enabled_mcp_server_names``, coding_context, oneshot, the gateway's
tool-name resolver) keys an MCP server's on/off state off ``mcp_servers.<name>.enabled``. The
profile editor used to read and write a separate ``disabled`` key, so a server toggled off in
the editor stayed live at runtime and a server with ``enabled: false`` showed as on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tui_gateway.server as server


@pytest.fixture
def profile_dir(tmp_path, monkeypatch) -> Path:
    """A temp HERMES_HOME root with one named profile, ``bot``, at ``<root>/profiles/bot``."""
    root = tmp_path / "hermes_home"
    path = root / "profiles" / "bot"
    path.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_constants import get_hermes_home_override

    assert get_hermes_home_override() is None
    return path


def _write_mcp(profile_dir: Path, servers: dict) -> None:
    (profile_dir / "config.yaml").write_text(
        yaml.safe_dump({"mcp_servers": servers}), encoding="utf-8"
    )


def _read_mcp(profile_dir: Path) -> dict:
    return yaml.safe_load((profile_dir / "config.yaml").read_text(encoding="utf-8"))["mcp_servers"]


def _call(method: str, params: dict) -> dict:
    resp = server._methods[method](1, {"name": "bot", **params})
    assert "error" not in resp, resp.get("error")
    return resp["result"]


def test_describe_reports_mcp_enabled_key_and_legacy_disabled(profile_dir):
    _write_mcp(profile_dir, {
        "on": {"command": "on", "enabled": True},
        "off": {"command": "off", "enabled": False},
        "legacy-off": {"command": "legacy", "disabled": True},
        "implicit-on": {"command": "implicit"},
    })

    status = {s["name"]: s["enabled"] for s in _call("profiles.describe", {})["mcp_servers"]}

    assert status == {"on": True, "off": False, "legacy-off": False, "implicit-on": True}


def test_configure_writes_mcp_enabled_key_and_drops_legacy_disabled(profile_dir):
    _write_mcp(profile_dir, {
        "wanted": {"command": "wanted", "enabled": False},
        "unwanted": {"command": "unwanted", "enabled": True, "disabled": True},
    })

    result = _call("profiles.configure", {"enabled_mcp_servers": ["wanted"]})

    assert result["ok"] is True
    assert result["applied"]["mcp_servers"] is True
    assert _read_mcp(profile_dir) == {
        "wanted": {"command": "wanted", "enabled": True},
        "unwanted": {"command": "unwanted", "enabled": False},
    }


def test_configure_toggle_is_seen_by_runtime_resolver(profile_dir):
    """The bug: disabling in the editor wrote ``disabled: true``, which the runtime ignored."""
    from hermes_cli.tools_config import enabled_mcp_server_names

    _write_mcp(profile_dir, {"keep": {"command": "keep"}, "drop": {"command": "drop"}})

    _call("profiles.configure", {"enabled_mcp_servers": ["keep"]})

    assert enabled_mcp_server_names({"mcp_servers": _read_mcp(profile_dir)}) == {"keep"}
    status = {s["name"]: s["enabled"] for s in _call("profiles.describe", {})["mcp_servers"]}
    assert status == {"keep": True, "drop": False}
