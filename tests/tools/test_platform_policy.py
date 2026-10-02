"""Canonical capability policy: selection stays separate from availability and authorization."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tools import platform_policy as policy
from tools.toolset_selection import LEGACY_TOOLSET_MAP, resolve_toolset_selection

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("platform", [*policy.PLATFORM_DEFAULT_TOOLSETS, "acp", "gui", "plugin"])
@pytest.mark.parametrize("selection", [[], "[]"])
def test_explicit_empty_bypasses_all_additions(platform, selection, monkeypatch):
    def unexpected():
        raise AssertionError("An empty selection must not discover plugins or probe credentials")

    monkeypatch.setattr(policy, "get_plugin_toolset_keys", unexpected)
    monkeypatch.setattr(policy, "_homeassistant_credentials_present", unexpected)
    cfg = {
        "platform_toolsets": {platform: selection},
        "mcp_servers": {"alpha": {}},
        "context": {"engine": "plugin"},
        "toolsets": ["kanban"],
    }
    assert policy.get_platform_tools(cfg, platform, xai_credentials_present=unexpected) == set()


@pytest.mark.parametrize("name", ["debugging", "coding", "hermes-cli", "hermes-discord",
                                *LEGACY_TOOLSET_MAP, "not-a-toolset"])
@pytest.mark.parametrize("disable", [False, True])
def test_shared_expansion_matches_model_selection(name, disable):
    from model_tools import _apply_toolset_selection

    selected = {"terminal", "web_search", "read_file", "discord_list_channels", "unrelated"}
    expected = set(selected)
    from toolsets import validate_toolset, resolve_toolset, get_toolset, bundle_non_core_tools

    if validate_toolset(name):
        resolved = (sorted(bundle_non_core_tools(name))
                    if disable and (name.startswith("hermes-") or get_toolset(name).get("posture"))
                    else resolve_toolset(name))
    else:
        resolved = LEGACY_TOOLSET_MAP.get(name)
    assert resolve_toolset_selection(name, disable=disable) == resolved
    if resolved is not None:
        (expected.difference_update if disable else expected.update)(resolved)
    _apply_toolset_selection(selected, [name], quiet_mode=True, disable=disable)
    assert selected == expected


def test_static_recovery_survives_runtime_registry_additions(monkeypatch):
    import toolsets
    from tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(name="runtime_only", toolset="terminal",
                 schema={"name": "runtime_only", "parameters": {"type": "object", "properties": {}}},
                 handler=lambda args, **kw: "{}")
    monkeypatch.setattr("tools.registry.registry", reg)
    monkeypatch.setattr(policy, "get_plugin_toolset_keys", lambda: set())
    monkeypatch.setattr(policy, "_homeassistant_credentials_present", lambda: False)
    selected = policy.get_platform_tools({}, "cli")
    assert "terminal" in selected
    assert "runtime_only" in resolve_toolset_selection("terminal")
    assert "runtime_only" not in toolsets.resolve_toolset("terminal", include_registry=False)


def test_disabled_composite_prunes_tools_but_keeps_mcp_and_runtime_only_keys(monkeypatch):
    monkeypatch.setattr(policy, "get_plugin_toolset_keys", lambda: set())
    cfg = {
        "platform_toolsets": {"cli": ["terminal", "file", "web", "memory", "alpha"]},
        "mcp_servers": {"alpha": {}, "beta": {}},
        "context": {"engine": "plugin"},
        "agent": {"disabled_toolsets": "['debugging']"},
    }
    enabled = policy.get_platform_tools(cfg, "cli")
    assert not {"terminal", "file", "web"} & enabled
    assert {"memory", "alpha", "context_engine"} <= enabled
    assert "beta" not in enabled


def test_fresh_process_selection_has_no_cli_or_model_pipeline_dependency():
    code = """
import importlib.abc, sys
from types import ModuleType
# Published metadata is an input seam; plugin lifecycle can import application
# code and is independently verified by plugin regressions.
lifecycle = ModuleType("plugin_runtime.lifecycle")
lifecycle.get_plugin_toolset_keys_nowait = lambda: set()
lifecycle.get_portable_mcp_server_names_nowait = lambda: set()
sys.modules["plugin_runtime.lifecycle"] = lifecycle
class BlockCli(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "model_tools" or fullname == "cli" or fullname.startswith("hermes_cli"):
            raise AssertionError("Runtime policy imported " + fullname)
sys.meta_path.insert(0, BlockCli())
from tools.platform_policy import get_platform_tools
assert get_platform_tools({"platform_toolsets": {"cli": []}}, "cli") == set()
selected = get_platform_tools({
    "platform_toolsets": {"cli": ["terminal", "memory", "no_mcp"]},
    "agent": {"disabled_toolsets": ["terminal"]},
}, "cli")
assert selected == {"memory"}, selected
"""
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_policy_direction_and_retired_scope_path():
    assert not (ROOT / "hermes_cli/toolset_scope.py").exists()
    for name in ("platform_policy", "toolset_scope", "toolset_selection"):
        tree = ast.parse((ROOT / "tools" / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            assert not any(mod == "model_tools" or mod == "cli" or mod.startswith("hermes_cli")
                           for mod in modules), (name, node.lineno, modules)
