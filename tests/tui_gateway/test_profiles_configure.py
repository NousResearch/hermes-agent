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
    return path


def _write_mcp(profile_dir: Path, servers: dict) -> None:
    (profile_dir / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": servers}), encoding="utf-8")


def _read_mcp(profile_dir: Path) -> dict:
    return yaml.safe_load((profile_dir / "config.yaml").read_text(encoding="utf-8"))["mcp_servers"]


def _call(method: str, params: dict) -> dict:
    resp = server._methods[method](1, {"name": "bot", **params})
    assert "error" not in resp, resp.get("error")
    return resp["result"]


def _described() -> dict:
    return {s["name"]: s["enabled"] for s in _call("profiles.describe", {})["mcp_servers"]}


def test_describe_reports_mcp_enabled_key_and_legacy_disabled(profile_dir):
    _write_mcp(profile_dir, {
        "on": {"command": "on", "enabled": True},
        "off": {"command": "off", "enabled": False},
        "legacy-off": {"command": "legacy", "disabled": True},
        "implicit-on": {"command": "implicit"},
    })

    assert _described() == {"on": True, "off": False, "legacy-off": False, "implicit-on": True}


def test_configure_toggle_is_what_the_runtime_resolver_and_describe_see(profile_dir):
    from hermes_cli.tools_config import enabled_mcp_server_names

    _write_mcp(profile_dir, {
        "keep": {"command": "keep", "enabled": False},
        "drop": {"command": "drop"},
        "legacy": {"command": "legacy", "disabled": True},
    })

    _call("profiles.configure", {"enabled_mcp_servers": ["keep", "legacy"]})

    on_disk = _read_mcp(profile_dir)
    assert not any("disabled" in entry for entry in on_disk.values())
    assert enabled_mcp_server_names({"mcp_servers": on_disk}) == {"keep", "legacy"}
    assert _described() == {"keep": True, "drop": False, "legacy": True}
