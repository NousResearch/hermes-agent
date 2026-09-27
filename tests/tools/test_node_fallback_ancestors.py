"""G2 guard rail: sub-profiles must resolve the ROOT install's managed Node
runtime. Evidence: under HERMES_HOME=<root>/profiles/bobby, bare ``npx``
crashed every stdio MCP server with WinError 2 because the managed tree
exists only at the root home (2026-09-25, reproduced end-to-end)."""
from tools.mcp_tool_config import _node_fallback


def _seed_root_tree(root):
    node = root / "node"
    node.mkdir(parents=True, exist_ok=True)
    (node / "npx.cmd").write_text("@echo off\r\n", encoding="utf-8")
    (node / "node.exe").write_bytes(b"MZ")
    return node


def test_fallback_finds_root_ancestor_tree_from_sub_profile(tmp_path, monkeypatch):
    root = tmp_path / "root"
    bobby = root / "profiles" / "bobby"
    bobby.mkdir(parents=True)
    npx = _seed_root_tree(root)
    monkeypatch.setenv("HERMES_HOME", str(bobby))

    resolved = _node_fallback("npx", windows=True)

    assert resolved == str(npx / "npx.cmd")


def test_fallback_prefers_active_profile_own_tree(tmp_path, monkeypatch):
    root = tmp_path / "root"
    bobby = root / "profiles" / "bobby"
    (bobby / "node").mkdir(parents=True)
    _seed_root_tree(root)
    own = bobby / "node" / "npx.cmd"
    own.write_text("@echo off\r\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(bobby))

    resolved = _node_fallback("npx", windows=True)

    assert resolved == str(own)


def test_fallback_returns_command_unchanged_when_no_tree(tmp_path, monkeypatch):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(lonely))

    assert _node_fallback("npx", windows=True) == "npx"
