"""Regression tests for suppressing the built-in file-memory writer.

The built-in ``memory`` tool writes to MEMORY.md / USER.md. External memory
providers (Hindsight, etc.) are additive and should remain usable when the
built-in writer is hidden from the model.
"""

from types import SimpleNamespace

import yaml


def _tool_names(tool_defs):
    return {tool.get("function", {}).get("name") for tool in tool_defs}


def _write_config(memory_config):
    from hermes_cli.config import get_config_path

    path = get_config_path()
    path.write_text(yaml.safe_dump({"memory": memory_config}), encoding="utf-8")
    return path


def _clear_tool_availability_caches():
    import model_tools
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()


def test_builtin_memory_writer_available_by_default():
    """Back-compat: existing configs keep exposing the built-in writer."""
    from model_tools import get_tool_definitions

    _clear_tool_availability_caches()
    tools = get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=True)

    assert "memory" in _tool_names(tools)


def test_builtin_memory_writer_can_be_disabled_without_disabling_memory_toolset():
    """``memory.builtin_writer_enabled: false`` hides only the built-in writer."""
    from model_tools import get_tool_definitions
    from tools.memory_tool import check_memory_requirements

    _write_config({"builtin_writer_enabled": False})
    _clear_tool_availability_caches()

    assert check_memory_requirements() is False
    tools = get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=True)

    assert "memory" not in _tool_names(tools)


def test_memory_provider_tools_still_inject_when_builtin_writer_disabled():
    """Provider tools are keyed by the memory toolset, not by the built-in writer."""
    import importlib
    import pytest

    memory_manager = importlib.import_module("agent.memory_manager")
    inject_memory_provider_tools = getattr(memory_manager, "inject_memory_provider_tools", None)
    if inject_memory_provider_tools is None:
        pytest.skip("this live branch injects provider tools inline in agent_init")

    class DummyMemoryManager:
        def get_all_tool_schemas(self):
            return [
                {
                    "name": "hindsight_recall",
                    "description": "Recall memories from Hindsight.",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "hindsight_retain",
                    "description": "Retain memories in Hindsight.",
                    "parameters": {"type": "object", "properties": {}},
                },
            ]

    agent = SimpleNamespace(
        enabled_toolsets=["memory"],
        tools=[],
        valid_tool_names=set(),
        _memory_manager=DummyMemoryManager(),
    )

    added = inject_memory_provider_tools(agent)

    assert added == 2
    assert "memory" not in agent.valid_tool_names
    assert {"hindsight_recall", "hindsight_retain"} <= agent.valid_tool_names


def test_builtin_prompt_memory_remains_injected_when_writer_tool_absent():
    """Read-injected MEMORY.md / USER.md bootloader is independent of writer tool."""
    from agent.system_prompt import build_system_prompt_parts
    from hermes_constants import get_hermes_home
    from tools.memory_tool import ENTRY_DELIMITER, MemoryStore

    memories_dir = get_hermes_home() / "memories"
    (memories_dir / "MEMORY.md").write_text(
        ENTRY_DELIMITER.join(["Compact bootloader guardrail"]),
        encoding="utf-8",
    )
    (memories_dir / "USER.md").write_text(
        ENTRY_DELIMITER.join(["User compact profile fact"]),
        encoding="utf-8",
    )
    store = MemoryStore()
    store.load_from_disk()

    agent = SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=set(),
        provider="test-provider",
        model="test-model",
        platform="cli",
        _tool_use_enforcement=False,
        _environment_probe=False,
        _memory_store=store,
        _memory_enabled=True,
        _user_profile_enabled=True,
        _memory_manager=None,
        pass_session_id=False,
        session_id="test-session",
    )

    volatile = build_system_prompt_parts(agent)["volatile"]

    assert "MEMORY (your personal notes)" in volatile
    assert "Compact bootloader guardrail" in volatile
    assert "USER PROFILE (who the user is)" in volatile
    assert "User compact profile fact" in volatile
