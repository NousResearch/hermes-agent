"""Every reader of ``mcp_servers.<name>.enabled`` gives one answer per value.

The MCP client, the toolset resolver, the profile editor, the catalog and the list/status
surfaces used to parse the key four different ways, so ``enabled: 0`` was off for the resolver
and on for the client, and ``enabled: "false"`` was on in the server list. The case table is
shared with the desktop (``apps/desktop/src/lib/mcp-enabled-cases.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "apps/desktop/src/lib/mcp-enabled-cases.json").read_text(encoding="utf-8"))


def _readers():
    from hermes_cli.mcp_catalog import server_enabled
    from hermes_cli.tools_config import enabled_mcp_server_names
    from tools.mcp_tool_common import mcp_server_enabled
    from hermes_agent.tui_gateway.mcp_rpc_helpers import summarize_server
    from hermes_agent.tui_gateway.methods_profiles import _mcp_entry_enabled

    return {
        "mcp client": mcp_server_enabled,
        "toolset resolver": lambda cfg: "s" in enabled_mcp_server_names({"mcp_servers": {"s": cfg}}),
        "profile editor": _mcp_entry_enabled,
        "catalog": server_enabled,
        "server list": lambda cfg: summarize_server("s", cfg)["enabled"],
    }


@pytest.mark.parametrize("case", CASES, ids=lambda c: repr({k: v for k, v in c.items() if k != "on"}))
def test_every_reader_agrees_with_the_shared_table(case):
    entry = {k: v for k, v in case.items() if k != "on"}
    answers = {name: bool(read({"command": "x", **entry})) for name, read in _readers().items()}

    assert answers == dict.fromkeys(answers, case["on"])


def _spawned(monkeypatch) -> list:
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as discovery

    spawned: list = []
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(discovery._loop, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(discovery, "_run_discovery_pass", lambda servers: spawned.extend(sorted(servers)))
    monkeypatch.setattr(discovery, "_acquire_discovery_lock_with_retry", lambda: None)
    monkeypatch.setattr(discovery, "_register_lazy_from_cache", lambda s: (s, 0, 0))
    for state in ("_servers", "_server_connecting", "_lazy_server_configs", "_server_scope_keys"):
        monkeypatch.setattr(core, state, type(getattr(core, state))())
    discovery.discover_mcp_tools()
    return spawned


def _picker(name):
    from hermes_cli.mcp_picker import _enable_disable

    _enable_disable(name, enable=True)


def _route(name):
    import asyncio

    from hermes_cli.web_models import MCPEnabledToggle
    from hermes_cli.web_routers.mcp import set_mcp_server_enabled

    asyncio.run(set_mcp_server_enabled(name, MCPEnabledToggle(enabled=True)))


def _connector(name):
    from tools.connectors.mcp import _CatalogBackend

    _CatalogBackend().enable(name)


def _config_set(name):
    from hermes_cli.config import set_config_value

    set_config_value(f"mcp_servers.{name}.enabled", "true")


@pytest.mark.parametrize("enable", [_picker, _route, _connector, _config_set],
                         ids=["picker", "PUT route", "connector", "config set"])
def test_enable_clears_a_legacy_disabled_so_the_server_spawns(enable, tmp_path, monkeypatch):
    """The reader lets a legacy `disabled: true` win over `enabled`, so an enable that only set
    `enabled` reported success and left the server off. Every enable writer drops the key."""
    import hermes_yaml as yaml
    from hermes_cli.config import DEFAULT_CONFIG, load_config
    from tools.mcp_tool_common import mcp_server_enabled

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "mcp_servers": {"s": {"command": "true", "disabled": True}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _spawned(monkeypatch) == []

    enable("s")

    entry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["mcp_servers"]["s"]
    assert "disabled" not in entry and entry["enabled"] is True
    assert mcp_server_enabled(load_config()["mcp_servers"]["s"]) is True
    assert _spawned(monkeypatch) == ["s"]


@pytest.mark.parametrize("case", [c for c in CASES if set(c) == {"disabled", "on"}], ids=lambda c: repr(c["disabled"]))
def test_v46_turns_off_exactly_what_the_reader_reads_as_off(case, tmp_path, monkeypatch):
    """v46 runs on every pass (stamp-independent) because it resolves `disabled` as the reader
    does; a value the two parse differently (a float) would flip a server on a re-run."""
    import hermes_yaml as yaml
    from hermes_cli.config_migrations import run_migrations

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "_config_version": 45, "mcp_servers": {"s": {"command": "x", "disabled": case["disabled"]}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    run_migrations(45, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
    entry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["mcp_servers"]["s"]

    assert (entry.get("enabled") is False) is (not case["on"])


@pytest.mark.parametrize("key", ["mcp_servers.s.example.enabled", "mcp_servers.s\\.example.enabled"],
                         ids=["unescaped", "escaped"])
def test_config_set_enable_clears_disabled_on_a_dotted_server_name(key, tmp_path, monkeypatch):
    """`config set` resolves an existing dotted server name greedily; the cleanup must reach the
    same entry, or the server stays off after a printed success."""
    import hermes_yaml as yaml
    from hermes_cli.config import DEFAULT_CONFIG, load_config, set_config_value
    from tools.mcp_tool_common import mcp_server_enabled

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "mcp_servers": {"s.example": {"command": "true", "disabled": True}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    set_config_value(key, "true")
    entry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["mcp_servers"]["s.example"]
    assert "disabled" not in entry and entry["enabled"] is True
    assert mcp_server_enabled(load_config()["mcp_servers"]["s.example"]) is True
