"""Behavioral contracts for built-in toolset safety boundaries.

These tests intentionally guard high-level invariants rather than exact full
snapshots. Toolsets can grow, but unsafe surfaces must not accidentally gain
local file/system execution or desktop-only project controls.
"""

from __future__ import annotations

import model_tools  # noqa: F401  # import triggers built-in tool self-registration
from tools.registry import registry
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS, resolve_toolset

LOCAL_SYSTEM_TOOLS = {
    "terminal", "process", "read_terminal", "close_terminal",
    "read_file", "write_file", "patch", "search_files",
    "execute_code", "delegate_task", "cronjob",
}

DESKTOP_PROJECT_TOOLS = {"project_list", "project_create", "project_switch"}


def test_webhook_toolset_stays_local_system_safe():
    """Webhook prompts may include untrusted third-party text.

    They must not get local file, shell, code-execution, delegation, cron, or
    desktop-project tools by accident.
    """
    tools = set(resolve_toolset("hermes-webhook"))
    assert tools
    assert tools.isdisjoint(LOCAL_SYSTEM_TOOLS)
    assert tools.isdisjoint(DESKTOP_PROJECT_TOOLS)


def test_safe_toolset_stays_without_local_system_access():
    """The named safe posture may include web/media helpers but no local system tools."""
    tools = set(resolve_toolset("safe"))
    assert tools
    assert tools.isdisjoint(LOCAL_SYSTEM_TOOLS)
    assert tools.isdisjoint(DESKTOP_PROJECT_TOOLS)


def test_desktop_project_tools_are_not_in_default_core_bundle():
    """Project tools only make sense when a GUI can follow workspace changes."""
    assert set(_HERMES_CORE_TOOLS).isdisjoint(DESKTOP_PROJECT_TOOLS)
    for platform in ("hermes-cli", "hermes-cron", "hermes-telegram", "hermes-discord"):
        assert set(resolve_toolset(platform)).isdisjoint(DESKTOP_PROJECT_TOOLS)


def test_static_toolsets_reference_registered_builtin_tools():
    """Every built-in static tool reference should resolve to a registered tool."""
    registered = set(registry.get_all_tool_names())
    referenced = set()
    for name in TOOLSETS:
        referenced.update(resolve_toolset(name))
    assert referenced - registered == set()


def test_all_resolved_toolsets_are_deduplicated_and_sorted():
    """Resolution order should be stable for prompt/tool-schema cacheability."""
    for name in TOOLSETS:
        resolved = resolve_toolset(name)
        assert resolved == sorted(set(resolved)), name
