"""Schema-shape tests for the built-in memory tool.

The memory tool previously used ``allOf: [{if: ..., then: {required: ...}}]``
at the top level of ``parameters`` to hint per-action required fields.  That
form was:

  1. Ignored by every provider (Chat Completions doesn't honour ``if/then``
     on function schemas), so it never actually enforced anything.
  2. **Rejected outright by strict backends** — OpenAI's Codex endpoint
     (``chatgpt.com/backend-api/codex``, gpt-5.x) returns
     ``Invalid schema for function 'memory': schema must have type 'object'
     and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'not' at the top level``.

We now rely on the runtime handler (``memory_tool()`` in ``tools/memory_tool.py``)
to validate required fields per action and return actionable error messages.
These tests guard the schema against regressing back to a shape strict
backends reject.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.prompt_builder import MEMORY_GUIDANCE
from agent.system_prompt import build_system_prompt_parts
from tools.memory_tool import MEMORY_SCHEMA, check_memory_requirements


_FORBIDDEN_TOP_LEVEL_KEYS = ("allOf", "anyOf", "oneOf", "enum", "not")


def test_memory_schema_has_no_forbidden_top_level_combinators():
    """OpenAI's Codex backend rejects these at the top level of parameters."""
    params = MEMORY_SCHEMA["parameters"]
    for key in _FORBIDDEN_TOP_LEVEL_KEYS:
        assert key not in params, (
            f"top-level {key!r} in memory tool parameters will break the "
            "Codex backend (chatgpt.com/backend-api/codex). Per-action "
            "required-field checks belong in the runtime handler, not the schema."
        )


def test_memory_schema_is_well_formed():
    params = MEMORY_SCHEMA["parameters"]
    assert params["type"] == "object"
    # Only ``target`` is universally required: ``action`` belongs to the
    # single-op shape and is omitted when the batch ``operations`` array is used.
    assert params["required"] == ["target"]
    # Nested ``enum`` on property values is fine — only top-level is forbidden.
    assert params["properties"]["action"]["enum"] == ["add", "replace", "remove"]
    assert params["properties"]["target"]["enum"] == ["memory", "user"]
    # Batch shape is exposed and its items reuse the same actions.
    assert params["properties"]["operations"]["type"] == "array"
    assert params["properties"]["operations"]["items"]["properties"]["action"]["enum"] == ["add", "replace", "remove"]


def test_memory_schema_is_json_serializable():
    json.dumps(MEMORY_SCHEMA)


def test_memory_tool_hidden_when_built_in_memory_disabled(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"memory_enabled": False, "user_profile_enabled": False}},
    )

    assert check_memory_requirements() is False


def test_memory_tool_available_when_memory_enabled(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"memory_enabled": True, "user_profile_enabled": False}},
    )

    assert check_memory_requirements() is True


def test_memory_tool_available_when_user_profile_enabled(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"memory_enabled": False, "user_profile_enabled": True}},
    )

    assert check_memory_requirements() is True


def test_memory_tool_registry_gate_updates_immediately(monkeypatch):
    from model_tools import get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    state = {"memory_enabled": True, "user_profile_enabled": False}
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": dict(state)},
    )
    invalidate_check_fn_cache()

    enabled = get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=False)
    assert {tool["function"]["name"] for tool in enabled} == {"memory"}

    state["memory_enabled"] = False
    disabled = get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=False)
    assert "memory" not in {tool["function"]["name"] for tool in disabled}


def test_memory_guidance_follows_registry_filtered_tool_names(monkeypatch):
    from model_tools import get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    state = {"memory_enabled": True, "user_profile_enabled": False}
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": dict(state)},
    )
    invalidate_check_fn_cache()

    def stable_prompt_for_registry_tools():
        tools = get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=False)
        agent = SimpleNamespace(
            load_soul_identity=False,
            skip_context_files=False,
            valid_tool_names={tool["function"]["name"] for tool in tools},
            _task_completion_guidance=False,
            _parallel_tool_call_guidance=False,
            _tool_use_enforcement=False,
            _environment_probe=False,
            _kanban_worker_guidance="",
            _memory_store=None,
            _memory_manager=None,
            model="",
            provider="",
            platform="",
            pass_session_id=False,
            session_id="",
        )
        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
        ):
            return build_system_prompt_parts(agent)["stable"]

    assert MEMORY_GUIDANCE in stable_prompt_for_registry_tools()

    state["memory_enabled"] = False
    assert MEMORY_GUIDANCE not in stable_prompt_for_registry_tools()
