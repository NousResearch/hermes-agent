"""execute_code approval in a gateway-run local session (T2).

Since the one-gateway runtime a classic CLI, TUI or Desktop turn runs in the gateway, so the
whole-script ``execute_code`` approval fired on every call under ``approvals.mode: manual``.
``main``'s interactive CLI never asked: it relied on the per-command ``terminal()`` guards inside
the script. These tests drive a real kernel cell with a local session's frozen launch policy and
the session's own gateway notify callback (what the viewer's approval prompt hangs off).
"""

from __future__ import annotations

import contextlib

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from gateway.session_policy import _ACTIVE_POLICY, SURFACES, LocalSessionPolicy
from tools import approval as A
from tools import approval_context, approval_smart
from tools.approval_detection import detect_dangerous_command
from tests.tools import _child_env_fixtures
from tests.tools._child_env_fixtures import run_code

# Pytest registers an imported fixture under this module's attribute name, so bind it as ``child_env``.
child_env = _child_env_fixtures.child_env

SESSION_KEY = "t2-local-session"
SIDE_KEY = "bg_0a1b2c"


def _policy(source: str, cwd: str) -> LocalSessionPolicy:
    return LocalSessionPolicy(source=source, platform={**SURFACES, "kanban": "cli"}[source], cwd=cwd, model="m",
                              toolsets=(), config_json="{}", request_json="{}", terminal_json="{}")


def _viewer_session(attached: str = "approver"):
    """The live session; ``attached`` is who ``session.resume`` recorded as subscribed: a viewer holding
    ``session:approve``, a read-only one, or nobody."""
    from types import SimpleNamespace
    from gateway.session_authority import LiveSession
    live = LiveSession(None, SESSION_KEY)
    capabilities = {"approver": {"session:read", "session:approve"}, "read-only": {"session:read"}}.get(attached)
    if capabilities is not None:
        live.subscribers["viewer"] = SimpleNamespace(capabilities=frozenset(capabilities))
    return live


@pytest.fixture
def turn(monkeypatch, tmp_path):
    """Bind one gateway turn: the gateway's process-wide ask marker, manual mode, the session's
    vars and (for a local session) its frozen policy. Yields a starter and the captured prompts."""
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    prompts: list[dict] = []
    answers: dict[str, str] = {}

    def notify(approval_data):
        prompts.append(approval_data)
        with A._lock:
            entry = A._gateway_queues[SESSION_KEY][-1]
            entry.result = answers.get(approval_data.get("pattern_key"), answers.get("*", "deny"))
            entry.event.set()

    cleanups = []

    def start(*, platform: str = "local", source: str | None = None, viewer: bool = True,
              attached: str = "approver", **set_answers):
        """``viewer``: the turn is one the session's viewer submitted (``execute_admission`` binds it);
        ``False`` is a plugin injection or another producer admission into the session. ``attached``: see
        ``_viewer_session``."""
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")  # start_gateway() sets it process-wide
        for name in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION"):
            monkeypatch.delenv(name, raising=False)
        answers.update(set_answers)
        vars_tokens = set_session_vars(platform=platform, session_key=SESSION_KEY, session_id=SESSION_KEY)
        key_token = approval_context.set_current_session_key(SESSION_KEY)
        policy_token = _ACTIVE_POLICY.set(_policy(source, str(tmp_path)) if source else None)
        if viewer:
            from gateway.session_ingress import viewer_turn
            viewer_token = viewer_turn.set(_viewer_session(attached))
            cleanups.append(lambda: viewer_turn.reset(viewer_token))
        A.register_gateway_notify(SESSION_KEY, notify)
        cleanups.append(lambda: (A.unregister_gateway_notify(SESSION_KEY), _ACTIVE_POLICY.reset(policy_token),
                                 approval_context.reset_current_session_key(key_token),
                                 clear_session_vars(vars_tokens)))
        return prompts

    with A._lock:
        A._session_approved.pop(SESSION_KEY, None)
        A._permanent_approved.discard("execute_code")
    try:
        yield start
    finally:
        for cleanup in reversed(cleanups):
            cleanup()
        with A._lock:
            for key in (SESSION_KEY, SIDE_KEY):
                A._session_approved.pop(key, None)
                A._pending.pop(key, None)
            A._gateway_queues.pop(SESSION_KEY, None)


def _rm_script(victim) -> tuple[str, str]:
    command = f"rm -rf {victim}"
    assert detect_dangerous_command(command)[0], "probe command must trip a per-command guard"
    return command, ("import json\nfrom hermes_tools import terminal\n"
                     f"print(json.dumps(terminal({command!r})))")


LOCAL_SOURCES = pytest.mark.parametrize("source", ["cli", "tui", "gui", "acp"])


@pytest.mark.platforms("posix")
@LOCAL_SOURCES
def test_terminal_inside_script_reaches_local_session_prompt(child_env, turn, source):
    """The safety T2 relies on: a dangerous terminal() the script issues raises the per-command
    approval on the local session's own prompt, and a denial keeps it from running."""
    victim = child_env / "victim"
    victim.mkdir()
    command, code = _rm_script(victim)
    prompts = turn(source=source, execute_code="once")
    result = run_code(code, enabled_tools=("terminal",))
    terminal_prompts = [p for p in prompts if p.get("pattern_key") != "execute_code"]
    assert [p["command"] for p in terminal_prompts] == [command], prompts
    assert victim.is_dir(), result


@pytest.mark.platforms("posix")
@LOCAL_SOURCES
def test_local_session_skips_whole_script_prompt(child_env, turn, source):
    victim = child_env / "victim"
    victim.mkdir()
    command, code = _rm_script(victim)
    prompts = turn(source=source)  # every prompt is denied
    run_code(code, enabled_tools=("terminal",))
    assert [p.get("pattern_key") for p in prompts if p.get("pattern_key") == "execute_code"] == []
    assert [p["command"] for p in prompts] == [command]
    assert victim.is_dir()


# The approval-specific refusals the gate returns (tools/approval.py _EXECUTE_CODE_GATE / _pending_result).
_DENIED_SCRIPT = ("BLOCKED: execute_code script denied by user. The user has NOT consented to running this code. "
                  "Do NOT retry, do NOT rephrase the script, and do NOT attempt the same outcome via a different tool."
                  " Stop the current workflow and wait for the user to respond before taking any further destructive "
                  "or irreversible action."
                  )


def _unanswered_script(code: str) -> str:
    """No notifier: the script parks as a pending approval nobody can answer, and never starts."""
    return (f"⚠️ {A._EXECUTE_CODE_DESCRIPTION}. Asking the user for approval.\n\n**Code:**\n```python\n{code}\n```\n\n"
            "STOP: do NOT re-run, rephrase, or re-issue this code — each variant sends the user ANOTHER approval "
            "card. Wait for the user's decision; if this turn must end, report that approval is pending.")


def _run_unguarded_rm(victim) -> tuple[str, dict]:
    """A real execute_code call whose script deletes ``victim`` with subprocess, which no terminal()
    guard sees: only the whole-script gate stands in front of it."""
    import json
    from unittest.mock import patch
    from tools.code_execution_tool import execute_code
    code = f"import subprocess\nsubprocess.run(['rm', '-rf', {str(victim)!r}], check=False)\nprint('ran')"
    with patch("tools.code_execution_tool._load_config", return_value={"mode": "project"}):
        return code, json.loads(execute_code(code=code, task_id="child-env-test", enabled_tools=["terminal"], reset=True))


@contextlib.contextmanager
def _side_task(cwd):
    """/bg and /btw (gateway/session_tui_side.py ``_side_scope``): the session's policy, their own task key and
    no notifier."""
    tokens = set_session_vars(session_key=SIDE_KEY, session_id=SIDE_KEY, cwd=str(cwd))
    key_token = approval_context.set_current_session_key(SIDE_KEY)
    try:
        yield
    finally:
        approval_context.reset_current_session_key(key_token)
        clear_session_vars(tokens)


@pytest.mark.platforms("posix")
@LOCAL_SOURCES
def test_injected_turn_keeps_whole_script_gate(child_env, turn, source):
    """A plugin ``inject_message`` turn (or any other producer admission) into a local session is not
    one its viewer submitted: a script whose subprocess call no terminal() guard sees is approved as a
    whole, and with nobody approving it never runs."""
    victim = child_env / "victim"
    victim.mkdir()
    prompts = turn(source=source, viewer=False)  # every prompt is denied
    _code, result = _run_unguarded_rm(victim)
    assert result.get("error") == _DENIED_SCRIPT, result
    assert "ran" not in result.get("output", "") and victim.is_dir()
    assert [p["pattern_key"] for p in prompts] == ["execute_code"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("source", ["cli", "gui"])
@pytest.mark.parametrize("attached", ["nobody", "read-only"])
def test_viewer_turn_with_no_approver_attached_keeps_whole_script_gate(child_env, turn, source, attached):
    """The turn's notifier is registered for every session turn, viewer attached or not; the exemption also
    needs an attached viewer that can answer approvals. Otherwise the script is approved as a whole and,
    with nobody approving, never runs."""
    victim = child_env / "victim"
    victim.mkdir()
    prompts = turn(source=source, attached=attached)  # every prompt is denied
    _code, result = _run_unguarded_rm(victim)
    assert result.get("error") == _DENIED_SCRIPT, result
    assert "ran" not in result.get("output", "") and victim.is_dir()
    assert [p["pattern_key"] for p in prompts] == ["execute_code"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("source", ["cli", "acp"])
@pytest.mark.parametrize("notifier", [True, False], ids=["during-parent-turn", "after-parent-turn"])
def test_delegated_child_keeps_whole_script_gate(child_env, turn, notifier, source):
    """Sync and async delegate_task children carry the parent's launch policy but are not the
    session's foreground turn: the script is approved as a whole, and with no notifier it is refused."""
    from agent.delegation_context import delegated_child_context
    victim = child_env / "victim"
    victim.mkdir()
    prompts = turn(source=source)  # every prompt is denied
    if not notifier:
        A.unregister_gateway_notify(SESSION_KEY)
    with delegated_child_context("child-session"):
        code, result = _run_unguarded_rm(victim)
    assert result.get("error") == (_DENIED_SCRIPT if notifier else _unanswered_script(code)), result
    assert "ran" not in result.get("output", "") and victim.is_dir()
    assert [p["pattern_key"] for p in prompts] == (["execute_code"] if notifier else [])


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("source", ["tui", "acp"])
def test_side_task_keeps_whole_script_gate(child_env, turn, source):
    """/bg and /btw run under the session's policy with their own task identity and no notifier:
    nobody can answer, so the script is refused."""
    victim = child_env / "victim"
    victim.mkdir()
    prompts = turn(source=source, execute_code="once")
    with _side_task(child_env):
        code, result = _run_unguarded_rm(victim)
    assert result.get("error") == _unanswered_script(code), result
    assert "ran" not in result.get("output", "") and victim.is_dir()
    assert prompts == []


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case,source", [("delegated-child", "cli"), ("side-task", "tui")])
def test_no_notifier_refuses_terminal_inside_permitted_script(child_env, turn, case, source):
    """With the whole-script gate permitted independently (execute_code approved for the key), a dangerous
    terminal() the script issues with no notifier is still refused by the command guard and never runs."""
    from agent.delegation_context import delegated_child_context
    victim = child_env / "victim"
    victim.mkdir()
    command, code = _rm_script(victim)
    prompts = turn(source=source)
    A.unregister_gateway_notify(SESSION_KEY)  # the parent turn has ended
    key = SESSION_KEY if case == "delegated-child" else SIDE_KEY
    A.approve_session(key, "execute_code")
    with delegated_child_context("child-session") if case == "delegated-child" else _side_task(child_env):
        out = run_code(code, enabled_tools=("terminal",))  # the script itself ran
    assert out == {"output": "", "exit_code": -1, "error": "", "status": "pending_approval", "approval_pending": True,
                   "command": command, "description": "recursive delete", "pattern_key": "recursive delete",
                   "smart_denied": False, "allow_permanent": True}
    assert A._pending[key]["command"] == command
    assert prompts == [] and victim.is_dir()


@pytest.mark.parametrize("platform,source", [("telegram", None), ("discord", None),
                                             ("local", "kanban"), ("local", None)])
def test_other_sessions_keep_whole_script_prompt(turn, platform, source):
    """Messaging chats, the kanban worker source and a local route without a frozen policy
    still approve the script as a whole."""
    prompts = turn(platform=platform, source=source)
    res = A.check_execute_code_guard("import os", "local")
    assert res["approved"] is False
    assert [p["pattern_key"] for p in prompts] == ["execute_code"]


def test_local_session_unattended_contexts_still_resolve_first(turn, monkeypatch):
    """A finite (-q / -z) turn of a CLI-sourced session still resolves from single_query_mode."""
    for module in (A, approval_context):
        monkeypatch.setattr(module, "_is_single_query_approval_context", lambda: True)
        monkeypatch.setattr(module, "_get_single_query_approval_mode", lambda: "deny", raising=False)
    prompts = turn(source="cli")
    res = A.check_execute_code_guard("import os", "local")
    assert res["approved"] is False and res["outcome"] == "blocked"
    assert prompts == []


def test_local_session_smart_mode_skips_script_assessment(turn, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    assessed = []
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda c, d: assessed.append(c) or "deny")
    prompts = turn(source="tui")
    assert A.check_execute_code_guard("import os", "local")["approved"] is True
    assert assessed == [] and prompts == []
