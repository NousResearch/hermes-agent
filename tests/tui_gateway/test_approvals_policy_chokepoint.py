"""Tests for TUI approvals-policy chokepoint and raw-writer bypass guard (#104697 P1-B)."""

import pytest
import yaml

from tui_gateway import server, transport
from tui_gateway.methods_config_set import _tui_policy_write
from hermes_cli.config import _LOAD_CONFIG_CACHE, _RAW_CONFIG_CACHE


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolated real config store for TUI policy tests, with a bound RPC transport
    (the live gateway dispatch loop binds one; agent kernel subprocesses never do)."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    config_file = home / "config.yaml"
    config_file.write_text("approvals:\n  mode: smart\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None
    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()
    token = transport.bind_transport(server._stdio_transport)
    yield config_file
    transport.reset_transport(token)
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None
    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()


def test_tui_raw_write_config_key_bypass_refused(isolated_home):
    """_write_config_key must refuse to mutate security-sensitive keys directly,
    leaving config.yaml bytes unchanged (#104697 P1-B)."""
    initial_bytes = isolated_home.read_bytes()

    with pytest.raises(ValueError, match="Cannot mutate security-sensitive key"):
        server._write_config_key("approvals.mode", "off")

    assert isolated_home.read_bytes() == initial_bytes
    cfg = yaml.safe_load(isolated_home.read_text(encoding="utf-8"))
    assert (cfg.get("approvals") or {}).get("mode") == "smart"


def test_tui_policy_write_without_transport_refused(isolated_home):
    """The funnel requires the live gateway RPC transport: a process that never
    bound one (agent kernel child importing this module directly) cannot mint
    the grant (#104697 round-5 review, Pro F1/F3)."""
    token = transport.current_transport()
    # simulate the kernel child: no transport bound in this context
    reset_token = transport.bind_transport(None)
    try:
        with pytest.raises(RuntimeError, match="requires the live gateway RPC transport"):
            _tui_policy_write("approvals.mode", "off")
        cfg = yaml.safe_load(isolated_home.read_text(encoding="utf-8"))
        assert (cfg.get("approvals") or {}).get("mode") == "smart"
    finally:
        transport.reset_transport(reset_token)


def test_tui_sanctioned_policy_write_persists(isolated_home):
    """_tui_policy_write funnels through set_config_value with a valid stamp,
    bound transport, and sanctioned frame, persisting the change to disk (#104697 P1-B)."""
    _tui_policy_write("approvals.mode", "off")

    cfg = yaml.safe_load(isolated_home.read_text(encoding="utf-8"))
    assert (cfg.get("approvals") or {}).get("mode") == "off"


def test_tui_rpc_config_set_approvals_mode_persists(isolated_home):
    """TUI RPC config.set for approvals.mode goes through _tui_policy_write
    and persists to disk."""
    res = server._methods["config.set"](1, {"key": "approvals.mode", "value": "off"})
    assert res.get("result", {}).get("value") == "off"

    cfg = yaml.safe_load(isolated_home.read_text(encoding="utf-8"))
    assert (cfg.get("approvals") or {}).get("mode") == "off"


def test_tui_rpc_config_set_yolo_global_persists(isolated_home):
    """TUI RPC config.set for yolo with global scope flips approvals.mode
    through _tui_policy_write and persists to disk."""
    res = server._methods["config.set"](1, {"key": "yolo", "value": "1", "scope": "global"})
    assert res.get("result", {}).get("value") == "1"

    cfg = yaml.safe_load(isolated_home.read_text(encoding="utf-8"))
    assert (cfg.get("approvals") or {}).get("mode") == "off"
