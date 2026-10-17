"""Eager tools (configured and usage-learned) and name inference for unnamed tool_call entries."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.registry import registry
from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig, assemble_tool_defs, load_config
from tools import tool_search_usage
from tools.tool_search_usage import learned_eager_tools
from tools.tool_search_validation import normalize_tool_call_entries


def _td(name, properties=None, required=None):
    params = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": name, "parameters": params}}


_WORKFLOW_PROPS = {"action": {"type": "string", "enum": ["get", "queue", "complete"]},
                   "task_id": {"type": "string"}, "goal": {"type": "string"}}


@pytest.fixture
def plugin_tools():
    names = ("probe_workflow", "probe_lights", "probe_other")
    schemas = {
        "probe_workflow": _td("probe_workflow", _WORKFLOW_PROPS, ["action"]),
        "probe_lights": _td("probe_lights", {"entity_id": {"type": "string"}, "brightness": {"type": "integer"}},
                            ["entity_id"]),
        "probe_other": _td("probe_other", {"query": {"type": "string"}}, ["query"]),
    }
    for name in names:
        registry.register(name=name, handler=lambda args, **kw: "{}", schema=schemas[name]["function"],
                          toolset="probe-plugin")
    yield [schemas[n] for n in names]
    for name in names:
        registry.deregister(name)


def _visible(defs, raw_config, monkeypatch, counts):
    monkeypatch.setattr(tool_search_usage, "recent_tool_session_counts", lambda: counts)
    assembled = assemble_tool_defs([_td("terminal")] + defs, context_length=200_000,
                                   config=ToolSearchConfig.from_raw({"enabled": "on", **raw_config}))
    return {td["function"]["name"] for td in assembled.tool_defs}


def test_learned_eager_tools_are_shown_directly(plugin_tools, monkeypatch):
    names = _visible(plugin_tools, {}, monkeypatch, {"probe_workflow": 40, "probe_lights": 12, "probe_other": 3})
    assert {"probe_workflow", "probe_lights"} <= names
    assert "probe_other" not in names
    assert BRIDGE_TOOL_NAMES <= names


def test_configured_eager_and_learning_off(plugin_tools, monkeypatch):
    names = _visible(plugin_tools, {"eager": ["probe_other"], "eager_learned_min_sessions": 0},
                     monkeypatch, {"probe_workflow": 40})
    assert "probe_other" in names
    assert "probe_workflow" not in names


def test_learned_eager_caps_count_and_schema_size():
    big = _td("probe_big", {f"p{i}": {"type": "string", "description": "x" * 200} for i in range(40)})
    defs = [_td("a"), _td("b"), _td("c"), big]
    counts = {"a": 30, "b": 20, "c": 15, "probe_big": 99}
    assert learned_eager_tools(defs, min_sessions=10, max_tools=2, counts=counts) == {"a", "b"}


def _write_history(home, rows):
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "state.db")
    conn.execute("CREATE TABLE messages (session_id TEXT, role TEXT, tool_name TEXT, timestamp REAL)")
    conn.executemany("INSERT INTO messages VALUES (?, 'tool', ?, ?)", rows)
    conn.commit()
    conn.close()


def _counts_in(home):
    token = set_hermes_home_override(home)
    try:
        return tool_search_usage.recent_tool_session_counts()
    finally:
        reset_hermes_home_override(token)


def test_session_counts_come_from_state_db_and_are_cached(tmp_path, monkeypatch):
    now = time.time()
    _write_history(tmp_path, [
        ("s1", "probe_workflow", now), ("s1", "probe_workflow", now), ("s2", "probe_workflow", now),
        ("s3", "probe_workflow", now - 30 * 86400)])
    monkeypatch.setattr(tool_search_usage, "_memo", {})
    assert _counts_in(tmp_path) == {"probe_workflow": 2}
    cache = tmp_path / "cache" / "tool_usage_counts.json"
    assert json.loads(cache.read_text())["counts"] == {"probe_workflow": 2}


def test_session_counts_stay_per_profile(tmp_path, monkeypatch):
    """One process serves several profiles: a warm memo for one never answers for another."""
    now = time.time()
    _write_history(tmp_path / "work", [("s1", "probe_workflow", now)])
    _write_history(tmp_path / "home", [("s1", "probe_lights", now), ("s2", "probe_lights", now)])
    monkeypatch.setattr(tool_search_usage, "_memo", {})
    assert _counts_in(tmp_path / "work") == {"probe_workflow": 1}
    assert _counts_in(tmp_path / "home") == {"probe_lights": 2}
    assert _counts_in(tmp_path / "work") == {"probe_workflow": 1}


def test_eager_keys_are_read_from_config_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "tools:\n  tool_search:\n    eager: [probe_other]\n"
        "    eager_learned_min_sessions: 3\n    eager_learned_max: 2\n", encoding="utf-8")
    token = set_hermes_home_override(tmp_path)
    try:
        config = load_config()
    finally:
        reset_hermes_home_override(token)
    assert config.eager_tools == {"probe_other"}
    assert (config.eager_learned_min_sessions, config.eager_learned_max) == (3, 2)


def test_unnamed_entry_gets_the_only_fitting_tool(plugin_tools):
    entries, err = normalize_tool_call_entries(
        {"calls": [{"arguments": {"action": "queue", "goal": "long goal text"}}]})
    assert err is None
    assert entries == [{"name": "probe_workflow", "arguments": {"action": "queue", "goal": "long goal text"}}]


def test_unnamed_entry_without_a_unique_fit_explains_the_shape(plugin_tools):
    _, err = normalize_tool_call_entries({"calls": [{"arguments": {"action": "nope"}}]})
    assert "requires a 'name'" in err and '"name":"<tool>"' in err
