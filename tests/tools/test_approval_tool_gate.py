"""``approvals.scope: all`` — every tool call reaches a present human, through every dispatch seam.

Repro on the pattern-only gate: a terminal read of a protected path was refused while the model
read the same file with ``read_file`` and nobody was asked. These drive the REAL dispatcher
(``model_tools.handle_function_call``, the execute_code RPC dispatch, the ``tool_call`` bridge)
against a temp HERMES_HOME config; only the human is a stand-in (the CLI approval callback).
"""
import json

import pytest

import hermes_cli.config as hc
from model_tools import handle_function_call
from tools import approval
from tools.approval_context import (
    reset_current_session_key, reset_hermes_interactive_context, set_current_session_key,
    set_hermes_interactive_context,
)
from tools.terminal_tool import set_approval_callback


class _Human:
    """CLI approval callback that answers from a script and records every prompt."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.prompts: list[tuple[str, str]] = []
        self.titles: list[str | None] = []

    def __call__(self, command, description, *, title=None, **_kwargs):
        self.prompts.append((command, description))
        self.titles.append(title)
        return self.answers.pop(0) if self.answers else "deny"


@pytest.fixture
def gate_env(tmp_path, monkeypatch, request):
    """Temp HERMES_HOME with ``approvals.scope``, a workspace, a file outside it, an
    interactive CLI context and a fresh approval session."""
    scope = getattr(request, "param", "all")
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "approvals:\n"
        "  mode: manual\n"
        f"  scope: {scope}\n"
        "  deny:\n    - \"echo forbidden*\"\n"
        "command_allowlist: []\n"
        "tools:\n  tool_search:\n    defer: [read_file]\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside-secret\n")
    (workspace / "notes.txt").write_text("workspace-notes\n")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    monkeypatch.chdir(workspace)
    hc._LOAD_CONFIG_CACHE.clear()
    session_token = set_current_session_key(f"gate-{request.node.name}")
    interactive_token = set_hermes_interactive_context(True)
    yield {"workspace": workspace, "outside": outside}
    set_approval_callback(None)
    approval.clear_session(f"gate-{request.node.name}")
    reset_hermes_interactive_context(interactive_token)
    reset_current_session_key(session_token)
    hc._LOAD_CONFIG_CACHE.clear()


def _call(name: str, args: dict, **kwargs) -> dict:
    return json.loads(handle_function_call(name, args, task_id="gate-task", **kwargs))


def test_read_outside_workspace_asks_and_deny_blocks(gate_env):
    human = _Human("deny")
    set_approval_callback(human)
    result = _call("read_file", {"path": str(gate_env["outside"] / "secret.txt")})
    assert "error" in result and "outside-secret" not in json.dumps(result)
    assert len(human.prompts) == 1
    assert str(gate_env["outside"] / "secret.txt") in human.prompts[0][0]


def test_workspace_session_grant_does_not_open_other_paths(gate_env):
    human = _Human("session", "deny")
    set_approval_callback(human)
    first = _call("write_file", {"path": "a.txt", "content": "one"})
    assert "error" not in first
    second = _call("write_file", {"path": "sub/b.txt", "content": "two"})
    assert "error" not in second
    assert (gate_env["workspace"] / "sub" / "b.txt").read_text() == "two"
    assert len(human.prompts) == 1  # the session grant covered the second workspace write

    escaped = _call("write_file", {"path": str(gate_env["outside"] / "c.txt"), "content": "x"})
    assert "error" in escaped
    assert not (gate_env["outside"] / "c.txt").exists()
    assert len(human.prompts) == 2


def test_tool_call_bridge_is_gated_under_the_real_tool(gate_env):
    human = _Human("deny")
    set_approval_callback(human)
    result = _call(
        "tool_call",
        {"calls": [{"name": "read_file", "arguments": {"path": str(gate_env["outside"] / "secret.txt")}}]},
        enabled_toolsets=["file"],
    )
    assert "outside-secret" not in json.dumps(result)
    assert len(human.prompts) == 1 and human.prompts[0][0].startswith("read_file ")


def test_execute_code_inner_tool_call_is_gated(gate_env):
    from tools.code_execution_rpc import _default_dispatch

    human = _Human("deny")
    set_approval_callback(human)
    raw = _default_dispatch("gate-task")("read_file", {"path": str(gate_env["outside"] / "secret.txt")})
    assert "outside-secret" not in raw
    assert len(human.prompts) == 1


def test_terminal_asks_per_command_and_session_grant_is_exact(gate_env):
    human = _Human("session", "once")
    set_approval_callback(human)
    assert not _call("terminal", {"command": "echo first"}).get("error")
    assert not _call("terminal", {"command": "echo first"}).get("error")
    assert len(human.prompts) == 1  # exact repeat rides the session grant
    assert not _call("terminal", {"command": "echo second"}).get("error")
    assert len(human.prompts) == 2


def test_dangerous_command_is_asked_once(gate_env):
    target = gate_env["workspace"] / "scratch"
    target.mkdir()
    command = f"rm -rf {target}"
    assert approval.detect_dangerous_command(command)[0]
    human = _Human("once")
    set_approval_callback(human)
    result = _call("terminal", {"command": command})
    assert not result.get("error"), result
    assert not target.exists()
    assert len(human.prompts) == 1


def test_deny_rule_blocks_before_the_human_is_asked(gate_env):
    human = _Human("once")
    set_approval_callback(human)
    result = _call("terminal", {"command": "echo forbidden thing"})
    assert "user-defined deny rule" in json.dumps(result)
    assert human.prompts == []


def test_decision_is_recorded_off_the_model_visible_result(gate_env):
    from tools.approval_tool_gate import pop_record

    human = _Human("session")
    set_approval_callback(human)
    first = _call("terminal", {"command": "echo recorded"}, tool_call_id="call-1")
    second = _call("terminal", {"command": "echo recorded"}, tool_call_id="call-2")
    assert "approval" not in first and "approval" not in second
    assert pop_record("terminal", "call-1") == {
        "decision": "approved", "by": "user", "choice": "session", "scope": "this exact command only"}
    assert pop_record("terminal", "call-2")["by"] == "session_grant"
    assert len(human.prompts) == 1


def test_patch_prompt_leads_with_the_changed_lines(gate_env):
    human = _Human("deny")
    set_approval_callback(human)
    _call("patch", {"path": "notes.txt", "old_string": "workspace-notes", "new_string": "1|yeni deger"})
    rows = human.prompts[0][0].splitlines()
    assert "notes.txt" in rows[0] and "line 1" in rows[0] and "1 line changed: -1 +1" in rows[0]
    assert rows[1:3] == ["-workspace-notes", "+1|yeni deger"]
    assert (gate_env["workspace"] / "notes.txt").read_text() == "workspace-notes\n"


def test_patch_header_counts_what_the_short_view_hides(gate_env):
    old = "".join(f"old {i}\n" for i in range(20))
    (gate_env["workspace"] / "long.txt").write_text(old)
    human = _Human("deny")
    set_approval_callback(human)
    _call("patch", {"path": "long.txt", "old_string": old, "new_string": old.replace("old", "new")})
    assert "20 lines changed: -20 +20" in human.prompts[0][0].splitlines()[0]


def test_patch_preview_and_tool_keep_crlf_alike(gate_env):
    target = gate_env["workspace"] / "win.txt"
    target.write_bytes(b"alpha\r\nbeta\r\n")
    human = _Human("once")
    set_approval_callback(human)
    result = _call("patch", {"path": "win.txt", "old_string": "beta", "new_string": "gamma"})
    assert result.get("success"), result
    assert "CRLF line endings kept" in human.prompts[0][0].splitlines()[0]
    assert target.read_bytes() == b"alpha\r\ngamma\r\n"


def test_patch_preview_says_when_the_patch_will_fail(gate_env):
    human = _Human("deny")
    set_approval_callback(human)
    _call("patch", {"path": "notes.txt", "old_string": "not in the file", "new_string": "x"})
    shown = human.prompts[0][0]
    assert "will fail" in shown.splitlines()[0]
    assert "+x" not in shown and "--- a/" not in shown


def test_write_file_prompt_shows_the_content(gate_env):
    human = _Human("deny")
    set_approval_callback(human)
    _call("write_file", {"path": "fresh.txt", "content": "line one\nline two\n"})
    rows = human.prompts[0][0].splitlines()
    assert "fresh.txt" in rows[0] and "new or empty file" in rows[0]
    assert rows[1] == "line one"


def test_write_file_grant_does_not_cover_patch(gate_env):
    human = _Human("session", "deny")
    set_approval_callback(human)
    assert "error" not in _call("write_file", {"path": "a.txt", "content": "one"})
    _call("patch", {"path": "notes.txt", "old_string": "workspace-notes", "new_string": "changed"})
    assert len(human.prompts) == 2
    assert (gate_env["workspace"] / "notes.txt").read_text() == "workspace-notes\n"


def test_dangerous_match_is_flagged_and_titled_apart_from_a_plain_call(gate_env):
    target = gate_env["workspace"] / "scratch"
    target.mkdir()
    human = _Human("deny", "deny", "deny")
    set_approval_callback(human)
    _call("terminal", {"command": f"rm -rf {target}"})
    _call("terminal", {"command": "echo plain"})
    _call("read_file", {"path": str(gate_env["outside"] / "secret.txt")})
    assert "dangerous pattern" in human.prompts[0][1]
    assert "dangerous pattern" not in human.prompts[1][1]
    assert len(set(human.titles)) == 3 and "read_file" in human.titles[2]


def test_delegated_child_tool_calls_are_gated(gate_env):
    from agent.delegation_context import delegated_child_context

    human = _Human("deny")
    set_approval_callback(human)
    with delegated_child_context("child-session"):
        result = _call("write_file", {"path": "child.txt", "content": "x"})
    assert "error" in result and not (gate_env["workspace"] / "child.txt").exists()
    assert len(human.prompts) == 1


def test_background_delegation_runs_synchronously_while_every_call_is_asked(gate_env, monkeypatch):
    """A background child would outlive the turn that hosts its prompts; under scope: all it runs
    inside the turn instead, so its approval requests reach the user."""
    from tools import delegate_tool_dispatch as dispatch

    ran = []
    monkeypatch.setattr(dispatch, "_execute_and_aggregate", lambda batch: ran.append(batch) or {"results": []})
    set_approval_callback(_Human())
    result = json.loads(dispatch._dispatch_background(object()))
    assert len(ran) == 1 and "SYNCHRONOUSLY" in result["note"] and "approvals.scope" in result["note"]


@pytest.mark.parametrize("gate_env", ["dangerous"], indirect=True)
def test_default_scope_keeps_background_delegation(gate_env):
    from tools.approval_tool_gate import asks_every_call

    set_approval_callback(_Human())
    assert asks_every_call() is False


def test_unreachable_human_is_a_definitive_denial_not_a_pending_prompt(gate_env, monkeypatch):
    """A background subagent outlives the parent's turn: the gateway approval callback is gone.
    The call must be refused at once with a do-not-retry reason, and nothing parked."""
    from tools.approval_context import get_current_session_key

    token = set_hermes_interactive_context(False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    try:
        set_approval_callback(lambda *a, **k: pytest.fail("no surface: nobody may be prompted"))
        first = _call("terminal", {"command": "mkdir -p sub/b"})
        again = _call("terminal", {"command": "mkdir -p sub/b"})
    finally:
        reset_hermes_interactive_context(token)
    for result in (first, again):
        assert "cannot be reached" in json.dumps(result) and "Do NOT retry" in json.dumps(result)
    assert not (gate_env["workspace"] / "sub").exists()
    assert get_current_session_key() not in approval._pending


@pytest.mark.parametrize("gate_env", ["dangerous"], indirect=True)
def test_default_scope_keeps_pattern_only_behaviour(gate_env):
    set_approval_callback(lambda *a, **k: pytest.fail("pattern-only scope must not prompt for a read"))
    result = _call("read_file", {"path": str(gate_env["outside"] / "secret.txt")})
    assert "outside-secret" in json.dumps(result)


def test_no_human_present_is_unchanged(gate_env):
    token = set_hermes_interactive_context(False)
    try:
        set_approval_callback(lambda *a, **k: pytest.fail("no human context must not prompt"))
        result = _call("read_file", {"path": str(gate_env["outside"] / "secret.txt")})
    finally:
        reset_hermes_interactive_context(token)
    assert "outside-secret" in json.dumps(result)
