"""Tests for the smart-home tools feature branch.

Verifies that the five smart-home tools (Tesla, Tapo, LG TV, Sonos, Rheem)
self-register via the tools.registry auto-discovery mechanism and that their
toolsets are wired into toolsets.py correctly.
"""
import pytest

SMART_HOME_TOOLS = ["tesla", "tapo", "lgtv", "sonos", "rheem"]


@pytest.fixture(scope="module")
def loaded_registry():
    from tools.registry import discover_builtin_tools, registry
    discover_builtin_tools()
    return registry


@pytest.mark.parametrize("tool_name", SMART_HOME_TOOLS)
def test_smart_home_tool_is_registered(loaded_registry, tool_name):
    """Each smart-home tool self-registers and is discoverable by name."""
    all_names = loaded_registry.get_all_tool_names()
    assert tool_name in all_names, (
        f"{tool_name} not found in registry after auto-discovery; "
        f"tools/{tool_name}_tool.py should call registry.register()"
    )


@pytest.mark.parametrize("tool_name", SMART_HOME_TOOLS)
def test_smart_home_tool_has_schema(loaded_registry, tool_name):
    """Each smart-home tool exposes a usable schema with a name + description."""
    schema = loaded_registry.get_schema(tool_name)
    assert schema is not None, f"{tool_name} has no schema"
    assert schema.get("name") == tool_name
    assert schema.get("description"), f"{tool_name} schema missing description"


@pytest.mark.parametrize("tool_name", SMART_HOME_TOOLS)
def test_smart_home_toolset_registered(tool_name):
    """Each smart-home toolset is declared in toolsets.py TOOLSETS."""
    from toolsets import TOOLSETS
    assert tool_name in TOOLSETS, f"{tool_name} toolset missing from TOOLSETS"
    entry = TOOLSETS[tool_name]
    assert tool_name in entry.get("tools", []), (
        f"toolset '{tool_name}' should list tool '{tool_name}'"
    )
    assert entry.get("description"), f"toolset '{tool_name}' missing description"


@pytest.mark.parametrize("tool_name", SMART_HOME_TOOLS)
def test_smart_home_in_core_tools(tool_name):
    """Each smart-home tool is included in the default core toolset."""
    from toolsets import _HERMES_CORE_TOOLS
    assert tool_name in _HERMES_CORE_TOOLS, (
        f"{tool_name} missing from _HERMES_CORE_TOOLS"
    )
