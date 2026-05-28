from __future__ import annotations

import inspect
import json
from pathlib import Path

from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_platform_tools
from model_tools import get_tool_definitions, handle_function_call
from tools import wisdom_tool
from tools.registry import registry
from toolsets import resolve_toolset
from wisdom.config import load_wisdom_config
from wisdom.db import WisdomDB


WISDOM_TOOL_NAMES = [
    "wisdom_status",
    "wisdom_capture",
    "wisdom_search",
    "wisdom_original",
    "wisdom_interpret",
    "wisdom_apply",
    "wisdom_review",
    "wisdom_related",
    "wisdom_accept",
    "wisdom_dismiss",
    "wisdom_archive",
    "wisdom_inbox",
    "wisdom_set_enabled",
]


def test_wisdom_tools_are_registered_with_model_facing_schemas():
    for name in WISDOM_TOOL_NAMES:
        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.toolset == "wisdom"
        schema = entry.schema
        assert schema["name"] == name
        assert schema["description"].strip()
        assert schema["parameters"]["type"] == "object"
        for forbidden in ("allOf", "anyOf", "oneOf", "enum", "not"):
            assert forbidden not in schema["parameters"]


def test_wisdom_tool_schemas_validate_required_inputs():
    required = {
        "wisdom_capture": ["text"],
        "wisdom_search": ["query"],
        "wisdom_original": ["capture_id"],
        "wisdom_interpret": ["capture_id"],
        "wisdom_apply": ["capture_id"],
        "wisdom_related": ["capture_id"],
        "wisdom_accept": ["capture_id"],
        "wisdom_dismiss": ["capture_id"],
        "wisdom_archive": ["capture_id"],
        "wisdom_set_enabled": ["enabled"],
    }
    for name, expected in required.items():
        assert registry.get_entry(name).schema["parameters"]["required"] == expected


def test_tool_descriptions_include_natural_language_trigger_coverage():
    descriptions = {name: registry.get_entry(name).schema["description"].lower() for name in WISDOM_TOOL_NAMES}

    assert all(word in descriptions["wisdom_capture"] for word in ("remember", "save", "capture", "note"))
    assert all(word in descriptions["wisdom_search"] for word in ("find", "search", "recall", "retrieve"))
    assert all(word in descriptions["wisdom_original"] for word in ("exact", "original", "verbatim"))
    assert all(word in descriptions["wisdom_apply"] for word in ("apply", "turn", "client language", "checklist", "rule"))
    assert all(word in descriptions["wisdom_review"] for word in ("review", "high-potential", "unapplied"))
    assert all(word in descriptions["wisdom_related"] for word in ("related", "connect", "something like this"))
    assert all(word in descriptions["wisdom_accept"] for word in ("accept", "keep", "compounding"))
    assert all(word in descriptions["wisdom_dismiss"] for word in ("dismiss", "not useful", "noise"))
    assert all(word in descriptions["wisdom_archive"] for word in ("archive", "hide"))


def test_wisdom_toolset_is_available_to_default_cli_and_telegram_toolsets():
    assert set(WISDOM_TOOL_NAMES).issubset(resolve_toolset("wisdom"))
    assert set(WISDOM_TOOL_NAMES).issubset(resolve_toolset("hermes-cli"))
    assert set(WISDOM_TOOL_NAMES).issubset(resolve_toolset("hermes-telegram"))
    assert "terminal" in resolve_toolset("hermes-telegram")

    assert any(toolset == "wisdom" for toolset, _label, _description in CONFIGURABLE_TOOLSETS)
    assert "wisdom" in _get_platform_tools({}, "cli", include_default_mcp_servers=False)
    assert "wisdom" in _get_platform_tools({}, "telegram", include_default_mcp_servers=False)
    assert "wisdom" not in _get_platform_tools(
        {"agent": {"disabled_toolsets": ["wisdom"]}},
        "telegram",
        include_default_mcp_servers=False,
    )


def test_model_tool_definitions_expose_wisdom_schemas():
    definitions = get_tool_definitions(enabled_toolsets=["wisdom"], quiet_mode=True)
    names = {definition["function"]["name"] for definition in definitions}
    assert set(WISDOM_TOOL_NAMES) == names

    search_description = next(
        definition["function"]["description"]
        for definition in definitions
        if definition["function"]["name"] == "wisdom_search"
    )
    assert "find that idea about peace of mind" in search_description.lower()


def test_wisdom_tools_run_through_hermes_dispatcher_with_temp_db(
    isolated_env_db: Path,
    wisdom_home: Path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_WISDOM_ENABLED", "true")
    monkeypatch.setenv("HERMES_WISDOM_CAPTURE_MODE", "explicit")
    old_productivity_db = wisdom_home / "productivity" / "productivity.db"

    original_text = "Remember this: clients don't buy alpha, they buy peace of mind."
    captured = _call_tool(
        "wisdom_capture",
        {"text": original_text, "category": "business", "source_type": "thought"},
    )
    assert captured["ok"] is True
    capture_id = captured["capture"]["id"]
    assert captured["original_saved_exactly"] is True
    assert captured["capture"]["category"] == "business"

    searched = _call_tool("wisdom_search", {"query": "peace of mind", "limit": 5})
    assert searched["ok"] is True
    assert searched["count"] == 1
    assert searched["captures"][0]["id"] == capture_id

    exact = _call_tool("wisdom_original", {"capture_id": capture_id})
    assert exact["ok"] is True
    assert exact["original_text"] == original_text
    assert exact["exact_original"] is True

    interpretation = _call_tool("wisdom_interpret", {"capture_id": capture_id})
    assert interpretation["ok"] is True
    assert interpretation["interpretation"]["method"] == "deterministic"
    assert interpretation["interpretation"]["counterpoint"]

    applied = _call_tool(
        "wisdom_apply",
        {"capture_id": capture_id, "application_type": "client_language", "context": "x10x"},
    )
    assert applied["ok"] is True
    assert applied["external_actions_created"] is False
    assert [app["application_type"] for app in applied["applications"]] == ["client_language"]

    reviewed = _call_tool("wisdom_review", {"category": "business", "period": "recent", "limit": 5})
    assert reviewed["ok"] is True
    assert reviewed["counts"]["business"] == 1
    assert reviewed["recent"][0]["id"] == capture_id
    assert reviewed["mode"] == "needs_review"

    related_text = "Remember this: peace of mind is the windshield clients pay for."
    second = _call_tool(
        "wisdom_capture",
        {"text": related_text, "category": "business", "source_type": "thought"},
    )
    second_id = second["capture"]["id"]

    related_result = _call_tool("wisdom_related", {"capture_id": capture_id, "limit": 5})
    assert related_result["ok"] is True
    assert related_result["embeddings_used"] is False
    assert any(item["capture"]["id"] == second_id for item in related_result["related"])

    accepted = _call_tool("wisdom_accept", {"capture_id": second_id})
    assert accepted["ok"] is True
    assert accepted["capture"]["review_status"] == "accepted"

    dismissed = _call_tool("wisdom_dismiss", {"capture_id": second_id})
    assert dismissed["ok"] is True
    assert dismissed["capture"]["review_status"] == "dismissed"

    archived = _call_tool("wisdom_archive", {"capture_id": capture_id})
    assert archived["ok"] is True
    assert archived["status"] == "archived"

    assert isolated_env_db.exists()
    assert not old_productivity_db.exists()


def test_secret_like_native_capture_is_blocked_not_stored(isolated_env_db: Path, monkeypatch):
    monkeypatch.setenv("HERMES_WISDOM_ENABLED", "true")
    monkeypatch.setenv("HERMES_WISDOM_CAPTURE_MODE", "explicit")

    blocked = _call_tool(
        "wisdom_capture",
        {"text": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"},
    )
    assert blocked["ok"] is False
    assert blocked["status"] == "blocked_secret"

    db = WisdomDB(load_wisdom_config().db_path)
    try:
        db.init()
        assert db.counts()["captures"] == 0
    finally:
        db.close()


def test_wisdom_tool_errors_are_json_and_do_not_crash(monkeypatch, tmp_path):
    bad_path = tmp_path / "not-a-db"
    bad_path.mkdir()
    monkeypatch.setenv("HERMES_WISDOM_DB_PATH", str(bad_path))

    result = _call_tool("wisdom_status", {})
    assert result["ok"] is False
    assert result["error"] == "Wisdom tool failed safely."


def test_wisdom_tool_module_is_thin_and_has_no_external_side_effect_surfaces():
    source = inspect.getsource(wisdom_tool)
    forbidden = ("sqlite3", "SELECT ", "INSERT ", "telegram", "send_message", "productivity.db")
    for token in forbidden:
        assert token not in source


def _call_tool(name: str, args: dict) -> dict:
    raw = handle_function_call(name, args, task_id="wisdom-test", skip_pre_tool_call_hook=True)
    return json.loads(raw)
