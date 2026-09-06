from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _entry(name="remote-demo"):
    from tools.mcp_skills_protocol import SkillEntry
    body = f"---\nname: {name}\ndescription: Remote demo\n---\n\n# Demo\n".encode()
    uri = f"skill://fixture/{name}/SKILL.md"
    return SkillEntry.model_validate({
        "uri": uri,
        "frontmatter": {"name": name, "description": "Remote demo"},
        "resources": [
            {"uri": uri, "digest": "sha256:" + hashlib.sha256(body).hexdigest(), "size": len(body)},
            {"uri": uri.replace("SKILL.md", "references/run.md"),
             "digest": "sha256:" + hashlib.sha256(b"Run B guidance").hexdigest(), "size": 14},
        ],
    })


@pytest.fixture()
def held(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from tools import mcp_skills_registry as registry
    from tools import mcp_tool
    registry.clear_runtime_state()
    entry = _entry()
    registry.publish_live_catalog(home, "fixture", "config-a", [entry])
    record = registry.pin_session(home, "session-a")[0]
    tool_name = "mcp__fixture__read_resource"
    monkeypatch.setitem(mcp_tool._mcp_tool_server_names, tool_name, "fixture")
    yield home, record, tool_name
    registry.clear_runtime_state()


def test_catalog_resource_read_is_ordinary_and_does_not_activate(held):
    _home, record, tool_name = held
    from tools.mcp_skills_consent import enforce_remote_skill_gate
    from tools.mcp_skills_registry import active_skills
    raw = enforce_remote_skill_gate(tool_name, {"uri": record["uri"]}, session_id="session-a")
    assert raw is None
    assert active_skills(_home, "session-a") == []


def test_same_origin_generic_read_allowed_only_after_activation(held):
    home, record, tool_name = held
    from tools.mcp_skills_consent import enforce_remote_skill_gate
    from tools.mcp_skills_registry import mark_active
    mark_active(record, home, "session-a")
    assert enforce_remote_skill_gate(tool_name, {"uri": record["uri"]}, session_id="session-a") is None


def test_cross_origin_read_requires_uncached_per_call_consent(held, monkeypatch):
    home, record, _tool_name = held
    from tools import mcp_tool
    from tools.mcp_skills_consent import enforce_remote_skill_gate
    from tools.mcp_skills_registry import mark_active
    import tools.approval_prompt as approval_prompt

    mark_active(record, home, "session-a")
    monkeypatch.setitem(mcp_tool._mcp_tool_server_names, "mcp__other__read_resource", "other")
    answers = iter(["accept", "deny"])
    calls = []
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent",
                        lambda *args, **kwargs: (calls.append((args, kwargs)) or next(answers)))
    args = {"uri": "resource://other/data"}
    assert enforce_remote_skill_gate("mcp__other__read_resource", args, session_id="session-a") is None
    raw = enforce_remote_skill_gate("mcp__other__read_resource", args, session_id="session-a")
    assert raw is not None
    denied = json.loads(raw)
    assert "BLOCKED" in denied["error"]
    assert len(calls) == 2


def test_host_execution_consent_is_manifest_and_session_bound(held, monkeypatch):
    home, record, _tool_name = held
    from tools.mcp_skills_consent import enforce_remote_skill_gate
    from tools.mcp_skills_registry import mark_active
    import tools.approval_prompt as approval_prompt

    mark_active(record, home, "session-a")
    prompts = []
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent",
                        lambda *args, **kwargs: (prompts.append(args[0]) or "accept"))
    assert enforce_remote_skill_gate("execute_code", {"code": "print('ok')"}, session_id="session-a") is None
    assert enforce_remote_skill_gate("terminal", {"command": "printf ok"}, session_id="session-a") is None
    assert len(prompts) == 1
    assert record["manifest_fingerprint"] in prompts[0]

    # A distinct session has no active remote origin and therefore cannot inherit
    # either the marker or consent from session-a.
    assert enforce_remote_skill_gate("terminal", {"command": "printf ok"}, session_id="session-b") is None


def test_real_dispatch_gate_blocks_before_registry_execution(held, monkeypatch):
    home, record, _tool_name = held
    from tools.mcp_skills_registry import mark_active
    import model_tools
    import tools.approval_prompt as approval_prompt

    mark_active(record, home, "session-a")
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", lambda *a, **k: "deny")
    called = []
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *a, **k: called.append(a) or "{}")
    result = json.loads(model_tools.handle_function_call(
        "browser_exec", {"code": "print('no')"}, session_id="session-a", task_id="session-a"))
    assert "not approved" in result["error"]
    assert called == []


def test_local_inline_shell_uses_remote_content_bound_gate(held, monkeypatch, tmp_path):
    home, record, _tool_name = held
    from agent.skill_preprocessing import preprocess_skill_content
    from tools.mcp_skills_registry import mark_active
    import tools.approval_prompt as approval_prompt

    mark_active(record, home, "session-a")
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", lambda *a, **k: "deny")
    sentinel = tmp_path / "must-not-exist"
    rendered = preprocess_skill_content(
        f"!`touch {sentinel}`", tmp_path, "session-a", {"inline_shell": True})
    assert "inline-shell blocked" in rendered
    assert not sentinel.exists()


def test_real_skill_view_dispatch_uses_session_not_turn_task_for_inline_gate(held, monkeypatch, tmp_path):
    home, record, _tool_name = held
    from agent import skill_preprocessing
    from tools.mcp_skills_registry import mark_active
    import model_tools
    import tools.approval_prompt as approval_prompt

    local = home / "skills" / "general" / "local-inline"
    local.mkdir(parents=True)
    sentinel = tmp_path / "dispatch-must-not-exist"
    (local / "SKILL.md").write_text(
        "---\nname: local-inline\ndescription: local\n---\n\n"
        f"!`touch {sentinel}`\n", encoding="utf-8")
    mark_active(record, home, "durable-session")
    monkeypatch.setattr(skill_preprocessing, "load_skills_config", lambda: {"inline_shell": True})
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", lambda *a, **k: "deny")

    result = json.loads(model_tools.handle_function_call(
        "skill_view", {"name": "local-inline"},
        task_id="distinct-turn-task", session_id="durable-session"))
    assert result["success"] is True
    assert "inline-shell blocked" in result["content"]
    assert not sentinel.exists()


def test_native_skill_view_cross_origin_is_per_call_even_after_b_is_active(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from tools import mcp_skills_registry as registry
    from tools import mcp_skills_view as remote_view
    from tools.mcp_skills_view import serve_remote_skill
    import tools.approval_prompt as approval_prompt

    registry.clear_runtime_state()
    a, b = _entry("skill-a"), _entry("skill-b")
    registry.publish_live_catalog(home, "fixture", "config-a", [a, b])
    records = registry.pin_session(home, "session-cross")
    by_name = {row["frontmatter"]["name"]: row for row in records}
    registry.mark_active(by_name["skill-a"], home, "session-cross")
    prompts = []
    answers = iter(["accept", "deny", "deny"])
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent",
                        lambda *args, **kwargs: (prompts.append(args[0]) or next(answers)))
    def fetched(record, resource, *_args):
        raw = (f"---\nname: {record['frontmatter']['name']}\n"
               "description: Remote demo\n---\n\n# Demo\n").encode()
        if resource["uri"].endswith("references/run.md"):
            raw = b"Run B guidance"
        return raw, "text/markdown", True, tmp_path / "cache", {
            "scope": "single_resource_text", "verdict": "safe", "allowed": True}
    monkeypatch.setattr(remote_view, "get_verified_resource", fetched)

    b_name = registry.qualified_name(by_name["skill-b"])
    first = json.loads(serve_remote_skill(
        b_name, file_path=None, task_id=None, session_id="session-cross",
        materialize=False, destination=None))
    assert first["success"] is True
    assert registry.is_active(by_name["skill-b"], home, "session-cross")

    second = json.loads(serve_remote_skill(
        b_name, file_path="references/run.md", task_id=None, session_id="session-cross",
        materialize=False, destination=None))
    assert "cross-origin native" in second["error"]
    assert len(prompts) == 2

    # Union-of-active-manifests is not authority for the generic route either.
    from tools.mcp_skills_consent import enforce_remote_skill_gate
    from tools import mcp_tool
    tool_name = "mcp__fixture__read_resource"
    monkeypatch.setitem(mcp_tool._mcp_tool_server_names, tool_name, "fixture")
    generic = enforce_remote_skill_gate(
        tool_name, {"uri": by_name["skill-b"]["uri"]}, session_id="session-cross")
    assert generic is not None
    assert len(prompts) == 3
    registry.clear_runtime_state()


def test_execution_consent_is_not_persisted_or_forgeable_by_native_file_tools(held, monkeypatch):
    home, record, _tool_name = held
    from tools import file_tools_write_guards as guards
    from tools import mcp_skills_registry as registry

    registry.mark_active(record, home, "session-a")
    registry.record_execution_consent(record, home, "session-a")
    assert registry.has_execution_consent(record, home, "session-a")
    registry.clear_runtime_state()
    assert registry.active_skills(home, "session-a")
    assert not registry.has_execution_consent(record, home, "session-a")

    monkeypatch.setattr(guards, "_real_hermes_home_cached", str(home.resolve()))
    monkeypatch.setattr(guards, "_real_hermes_home_loaded", True)
    state_file = home / "cache" / "mcp-skills" / "active-origins" / "forged.json"
    blocked = guards._check_sensitive_path(str(state_file), "session-a")
    assert blocked and "Hermes-managed MCP skill state" in blocked
