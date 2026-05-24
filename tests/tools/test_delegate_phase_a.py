"""Tests for delegate_tool Phase A — agent_id, registry, desktop, MCP."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from tools.delegate_tool import (
    _load_agent_registry,
    _load_subagent_profile,
    _desktop_allowed,
    _should_inherit_mcp_toolsets_for_profile,
    _resolve_effective_toolsets,
    _resolve_effective_blocked_tools,
    _check_mcp_server_availability,
    GLOBAL_SUBAGENT_BLOCKED_TOOLSETS,
    GLOBAL_SUBAGENT_BLOCKED_TOOLS,
    DELEGATE_BLOCKED_TOOLS,
)


# ── Test registry data ──────────────────────────────────────────────────

_KIMI_CONFIG = {
    "id": "kimi",
    "type": "researcher",
    "capabilities": ["web_search", "file_reading", "information_synthesis"],
    "subagent_profile": {
        "model": "default",
        "model_ref": "legacy_research",
        "toolsets": ["file", "web"],
        "blocked_tools": ["write_file", "patch", "delegate_task", "terminal"],
        "permission_mode": "read_only",
        "isolation": "readonly",
        "allow_background": False,
        "required_mcp_servers": [],
    },
}

_CLAUDE_CONFIG = {
    "id": "claude",
    "type": "file_executor",
    "capabilities": ["file_modification", "script_execution", "git_operations"],
    "subagent_profile": {
        "model": "default",
        "model_ref": "claude_opus",
        "toolsets": ["file", "terminal"],
        "blocked_tools": ["delegate_task", "send_message", "memory"],
        "permission_mode": "ask",
        "isolation": "shared",
        "allow_background": False,
        "required_mcp_servers": [],
    },
}

_DESKTOP_CONFIG = {
    "id": "desktop-agent",
    "type": "desktop",
    "capabilities": ["desktop_control"],
    "subagent_profile": {
        "model": "default",
        "toolsets": ["desktop", "file"],
        "blocked_tools": ["terminal", "send_message", "memory", "delegate_task", "write_file"],
        "permission_mode": "ask",
        "isolation": "shared",
        "allow_background": False,
        "required_mcp_servers": [],
    },
}

_MCP_CONFIG = {
    "id": "mcp-agent",
    "type": "custom",
    "capabilities": ["file_reading"],
    "subagent_profile": {
        "model": "default",
        "toolsets": ["file", "mcp-github"],
        "blocked_tools": ["delegate_task"],
        "permission_mode": "ask",
        "isolation": "shared",
        "allow_background": False,
        "required_mcp_servers": ["github"],
        "inherit_mcp_toolsets": False,
    },
}

_PARENT_TOOLSETS = {"terminal", "file", "web", "browser", "desktop", "mcp-github", "mcp-slack"}


def _write_registry(hermes_home: Path, agents: dict = None) -> Path:
    """Write a test agent-registry.json to the hermetic HERMES_HOME."""
    config_dir = hermes_home / "config"
    config_dir.mkdir(exist_ok=True)
    registry = {
        "schema_version": "1.0",
        "agents": agents or {
            "kimi": _KIMI_CONFIG,
            "claude": _CLAUDE_CONFIG,
            "desktop-agent": _DESKTOP_CONFIG,
            "mcp-agent": _MCP_CONFIG,
        },
        "routing_rules": {
            "web_research": "kimi",
            "file_modification": "claude",
            "file_reading_analysis": "kimi",
        },
    }
    path = config_dir / "agent-registry.json"
    path.write_text(json.dumps(registry))
    return path


@pytest.fixture
def hermes_registry(_hermetic_environment):
    """Hermetic env with a test registry written to HERMES_HOME/config/."""
    hermes_home = Path(os.environ["HERMES_HOME"])
    _write_registry(hermes_home)
    yield hermes_home


# ── Registry / Profile Tests ────────────────────────────────────────────

class TestLoadSubagentProfile:
    """_load_subagent_profile resolves agent profiles from registry."""

    def test_load_kimi_profile(self, hermes_registry):
        config, profile = _load_subagent_profile("kimi")
        assert config["id"] == "kimi"
        assert profile["isolation"] == "readonly"
        assert "web" in profile["toolsets"]

    def test_load_claude_profile(self, hermes_registry):
        config, profile = _load_subagent_profile("claude")
        assert config["id"] == "claude"
        assert config["model_ref"] == "claude_opus"
        assert profile["model_ref"] == "claude_opus"
        assert "file" in profile["toolsets"]
        assert "terminal" in profile["toolsets"]

    def test_load_profile_resolves_legacy_alias(self, hermes_registry):
        hermes_home = Path(os.environ["HERMES_HOME"])
        _write_registry(hermes_home, agents={
            "hermes-internal": {
                "id": "hermes-internal",
                "display_name": "Hermes 技术翻译官",
                "aliases": ["技术翻译官", "nesta"],
                "capabilities": ["technical_decomposition"],
                "subagent_profile": {
                    "model": "default",
                    "toolsets": ["file"],
                    "blocked_tools": [],
                    "permission_mode": "read_only",
                    "isolation": "readonly",
                },
            },
        })

        config, profile = _load_subagent_profile("nesta")

        assert config["id"] == "hermes-internal"
        assert profile["isolation"] == "readonly"

    def test_nonexistent_agent_raises(self, hermes_registry):
        with pytest.raises(ValueError, match="not found"):
            _load_subagent_profile("nonexistent-agent-12345")

    def test_missing_subagent_profile_raises(self, hermes_registry):
        hermes_home = Path(os.environ["HERMES_HOME"])
        # Overwrite with an agent that has no subagent_profile
        _write_registry(hermes_home, agents={
            "no-profile": {"id": "no-profile", "capabilities": ["test"]},
        })
        with pytest.raises(ValueError, match="no subagent_profile"):
            _load_subagent_profile("no-profile")


# ── Desktop Security Tests ──────────────────────────────────────────────

class TestDesktopSecurity:

    def test_desktop_allowed_both_conditions(self):
        assert _desktop_allowed(_DESKTOP_CONFIG, _DESKTOP_CONFIG["subagent_profile"]) is True

    def test_desktop_not_allowed_missing_capability(self):
        config = {"id": "x", "capabilities": ["web_search"]}
        profile = {"toolsets": ["desktop", "file"]}
        assert _desktop_allowed(config, profile) is False

    def test_desktop_not_allowed_missing_in_profile(self):
        config = {"id": "x", "capabilities": ["desktop_control"]}
        profile = {"toolsets": ["file"]}
        assert _desktop_allowed(config, profile) is False

    def test_desktop_stripped_for_kimi(self):
        toolsets = _resolve_effective_toolsets(
            agent_config=_KIMI_CONFIG,
            profile=_KIMI_CONFIG["subagent_profile"],
            requested_toolsets=["file", "web", "desktop"],
            parent_toolsets=_PARENT_TOOLSETS,
        )
        assert "desktop" not in toolsets

    def test_desktop_present_for_desktop_capable(self):
        toolsets = _resolve_effective_toolsets(
            agent_config=_DESKTOP_CONFIG,
            profile=_DESKTOP_CONFIG["subagent_profile"],
            requested_toolsets=None,
            parent_toolsets=_PARENT_TOOLSETS,
        )
        assert "desktop" in toolsets

    def test_desktop_capable_blocks_dangerous_tools(self):
        profile = _DESKTOP_CONFIG["subagent_profile"]
        blocked = set(profile["blocked_tools"])
        assert "terminal" in blocked
        assert "send_message" in blocked
        assert "memory" in blocked
        assert "delegate_task" in blocked


# ── Toolset Narrowing ───────────────────────────────────────────────────

class TestEffectiveToolsets:

    def test_profile_only(self):
        toolsets = _resolve_effective_toolsets(
            agent_config=_KIMI_CONFIG,
            profile=_KIMI_CONFIG["subagent_profile"],
            requested_toolsets=None,
            parent_toolsets=_PARENT_TOOLSETS,
        )
        assert "file" in toolsets
        assert "web" in toolsets
        assert "terminal" not in toolsets

    def test_intersection_with_requested(self):
        toolsets = _resolve_effective_toolsets(
            agent_config=_KIMI_CONFIG,
            profile=_KIMI_CONFIG["subagent_profile"],
            requested_toolsets=["web"],
            parent_toolsets=_PARENT_TOOLSETS,
        )
        assert toolsets == ["web"]

    def test_parent_limits(self):
        limited = {"file"}
        toolsets = _resolve_effective_toolsets(
            agent_config=_CLAUDE_CONFIG,
            profile=_CLAUDE_CONFIG["subagent_profile"],
            requested_toolsets=None,
            parent_toolsets=limited,
        )
        assert toolsets == ["file"]
        assert "terminal" not in toolsets

    def test_requested_not_in_profile_is_excluded(self):
        toolsets = _resolve_effective_toolsets(
            agent_config=_KIMI_CONFIG,
            profile=_KIMI_CONFIG["subagent_profile"],
            requested_toolsets=["terminal"],
            parent_toolsets=_PARENT_TOOLSETS,
        )
        assert "terminal" not in toolsets


# ── MCP Inheritance ─────────────────────────────────────────────────────

class TestMCPInheritance:

    def test_should_not_inherit_by_default(self):
        assert _should_inherit_mcp_toolsets_for_profile(_KIMI_CONFIG["subagent_profile"]) is False

    def test_should_inherit_when_true(self):
        profile = {"inherit_mcp_toolsets": True, "toolsets": []}
        assert _should_inherit_mcp_toolsets_for_profile(profile) is True

    def test_should_not_inherit_when_false_string(self):
        profile = {"inherit_mcp_toolsets": "true", "toolsets": []}
        assert _should_inherit_mcp_toolsets_for_profile(profile) is False

    def test_agent_id_no_mcp_inheritance_default(self):
        toolsets = _resolve_effective_toolsets(
            agent_config=_KIMI_CONFIG,
            profile=_KIMI_CONFIG["subagent_profile"],
            requested_toolsets=None,
            parent_toolsets=_PARENT_TOOLSETS,
        )
        assert "mcp-github" not in toolsets
        assert "mcp-slack" not in toolsets

    def test_mcp_explicit_in_profile(self):
        toolsets = _resolve_effective_toolsets(
            agent_config=_MCP_CONFIG,
            profile=_MCP_CONFIG["subagent_profile"],
            requested_toolsets=None,
            parent_toolsets=_PARENT_TOOLSETS,
        )
        assert "mcp-github" in toolsets


# ── MCP Server Availability ─────────────────────────────────────────────

class TestMCPServerAvailability:

    def test_no_warning_when_empty(self):
        warnings = []
        _check_mcp_server_availability({"required_mcp_servers": []}, warnings)
        assert warnings == []

    def test_warning_when_unavailable(self):
        warnings = []
        _check_mcp_server_availability(
            {"required_mcp_servers": ["nonexistent-server-xyz"]},
            warnings,
        )
        assert len(warnings) == 1
        assert "not available" in warnings[0]


# ── Blocked Tools ───────────────────────────────────────────────────────

class TestEffectiveBlockedTools:

    def test_blocked_includes_delegate_blocked(self):
        result = _resolve_effective_blocked_tools(profile=_KIMI_CONFIG["subagent_profile"])
        for t in DELEGATE_BLOCKED_TOOLS:
            assert t in result

    def test_blocked_includes_global_blocked(self):
        result = _resolve_effective_blocked_tools(profile=_KIMI_CONFIG["subagent_profile"])
        for t in GLOBAL_SUBAGENT_BLOCKED_TOOLS:
            assert t in result

    def test_blocked_includes_profile_blocked(self):
        result = _resolve_effective_blocked_tools(profile=_KIMI_CONFIG["subagent_profile"])
        assert "write_file" in result
        assert "patch" in result
        assert "terminal" in result

    def test_requested_blocked_can_only_add(self):
        result_without = _resolve_effective_blocked_tools(profile=_KIMI_CONFIG["subagent_profile"])
        result_with = _resolve_effective_blocked_tools(
            profile=_KIMI_CONFIG["subagent_profile"],
            requested_blocked_tools=["browser"],
        )
        assert "browser" in result_with
        assert result_with.issuperset(result_without)


# ── agent_id / role Mutual Exclusion ────────────────────────────────────

class TestAgentIdProfileLoading:

    def test_agent_id_profile_loaded(self, hermes_registry):
        config, profile = _load_subagent_profile("kimi")
        assert config["id"] == "kimi"
        assert "subagent_profile" in config
