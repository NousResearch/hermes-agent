"""One key per session-cwd record: env creation and the command path must agree.

Live incident (2026-09-28, accountant profile): the terminal env was created in the profile's
configured ``terminal.cwd`` while the command inside it resolved a relative path against a stale
record left by an earlier session on the same chat key.

Two lookups, two identifiers: env creation read ``get_session_cwd(task_id)`` (the gateway's
``task_id`` is the rotating session id) while the command path (``_resolve_command_cwd``) and every
writer used the session key — the stable chat lane. A record written under the chat key therefore
placed the command, and never the env snapshot.
"""

import pytest

import tools.terminal_tool as tt

CHAT_KEY = "agent:accountant:telegram:dm:42"
SESSION_ID = "20260928_111149_4e1da518"
CONFIGURED = "/home/ec2-user/Developer/finances"


@pytest.fixture(autouse=True)
def _clean_terminal_state(monkeypatch):
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.setattr(
        tt,
        "_get_env_config",
        lambda: {"env_type": "local", "cwd": CONFIGURED, "timeout": 60, "lifetime_seconds": 3600},
    )


@pytest.fixture
def chat_lane():
    """The gateway's real session binding: a stable chat key plus a session id as ``task_id``."""
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(session_key=CHAT_KEY, session_id=SESSION_ID, platform="telegram")
    try:
        yield
    finally:
        clear_session_vars(tokens)


def _plan_cwd() -> str:
    plan = tt._plan_execution(
        "pwd", task_id=SESSION_ID, timeout=None, background=False, _host_local=False
    )
    return plan.cwd


def test_env_creation_resolves_the_sessions_own_record(chat_lane):
    """A recorded ``cd`` places the env where the command will actually run (same key)."""
    tt.record_session_cwd(CHAT_KEY, "/session/dir")

    assert _plan_cwd() == "/session/dir"
    assert tt._resolve_command_cwd(
        workdir=None, default_cwd=CONFIGURED, session_key=CHAT_KEY
    ) == "/session/dir"


def test_session_record_outranks_a_task_scoped_one(chat_lane):
    """The session's own ``cd`` wins over a record written under a task id.

    The reverse is the incident: a non-session identifier decided where the env was created
    while the command ran somewhere else.
    """
    tt.record_session_cwd(SESSION_ID, "/task/scoped/dir")
    tt.record_session_cwd(CHAT_KEY, "/session/dir")

    assert _plan_cwd() == "/session/dir"
