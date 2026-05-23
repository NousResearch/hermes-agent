"""OpenClaw opt-in toolsets: registration and schema smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_tools import get_tool_definitions
from toolsets import TOOLSETS, resolve_toolset
from tools.registry import discover_builtin_tools, registry


OPENCLAW_TOOLSETS = ("harness", "openclaw", "vrchat")

EXPECTED_TOOLS = {
    "harness": {
        "harness_status",
        "voice_bridge_status",
        "voice_bridge_turn",
        "vrc_relay_status",
        "vrc_relay_auto_osc",
    },
    "openclaw": {"channel_readiness_check"},
    "vrchat": {
        "vrchat_chatbox",
        "vrchat_avatar_catalog",
        "vrchat_avatar_safe_parameters",
        "vrchat_autonomy_status",
        "vrchat_autonomy_heartbeat",
        "vrchat_autonomy_build_decision_request",
        "vrchat_autonomy_validate_decision",
        "vrchat_autonomy_enqueue_observation",
        "vrchat_autonomy_profile_status",
        "vrchat_autonomy_profile_tick",
        "vrchat_autonomy_heartbeat_tick",
        "vrchat_autonomy_run_turn",
        "vrchat_autonomy_loop_tick",
        "vrchat_autonomy_plan_turn",
        "vrchat_autonomy_prepare_profile",
        "vrchat_autonomy_conversation_dry_run",
        "vrchat_autonomy_wait_ready",
        "vrchat_autonomy_wait_then_tick",
        "vrchat_autonomy_runtime_doctor",
        "vrchat_neuro_status",
        "vrchat_neuro_build_messages",
        "vrchat_neuro_handle_action",
        "vrchat_observation_ingest",
        "vrchat_observation_from_osc",
        "vrchat_observation_queue_status",
        "vrchat_autonomy_prepare_private_smoke",
        "vrchat_autonomy_wait_then_private_smoke",
        "vrchat_autonomy_private_smoke",
        "vrchat_autonomy_preflight_bundle",
        "vrchat_autonomy_completion_audit",
    },
}


@pytest.fixture(autouse=True)
def _fresh_registry():
    discover_builtin_tools()
    yield
    from model_tools import _clear_tool_defs_cache

    _clear_tool_defs_cache()


def test_toolsets_defined_in_toolsets_py():
    for name in OPENCLAW_TOOLSETS:
        assert name in TOOLSETS
        resolved = set(resolve_toolset(name))
        assert EXPECTED_TOOLS[name].issubset(resolved), f"{name}: {resolved}"


def test_get_tool_definitions_includes_openclaw_when_enabled(monkeypatch):
    monkeypatch.setattr("tools.harness_tools._check_harness_running", lambda: True)
    monkeypatch.setattr("tools.openclaw.harness_client.is_harness_running", lambda: True)
    defs = get_tool_definitions(
        enabled_toolsets=list(OPENCLAW_TOOLSETS),
        quiet_mode=True,
    )
    names = {item["function"]["name"] for item in defs}
    for toolset, required in EXPECTED_TOOLS.items():
        for tool_name in required:
            assert tool_name in names, f"{toolset}/{tool_name} missing from schemas"

    for tool_name in ("channel_readiness_check", "harness_status", "vrchat_avatar_catalog"):
        schema = next(d for d in defs if d["function"]["name"] == tool_name)
        assert schema["type"] == "function"
        assert "description" in schema["function"]
        params = schema["function"].get("parameters", {})
        assert params.get("type") == "object"


def test_default_core_toolset_excludes_harness():
    """Harness tools must stay opt-in (prompt caching policy)."""
    defs = get_tool_definitions(enabled_toolsets=None, disabled_toolsets=None, quiet_mode=True)
    names = {item["function"]["name"] for item in defs}
    assert "harness_status" not in names
    assert "channel_readiness_check" not in names


def test_registry_handlers_return_json_strings(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from tools.channel_readiness_tool import channel_readiness_check

    raw = channel_readiness_check(config_path=str(tmp_path / "missing.json"))
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    assert parsed.get("success") is False
