"""execute_code must not leak another request's session id via env (#69820).

`_scrub_child_env` sources from the process-global os.environ, whose
HERMES_SESSION_* mirror is last-writer-wins across concurrent gateway turns.
When a session var is passthrough-allowed (so scrubbing keeps it), the child
must carry THIS task's ContextVar value, not the stale global — the same
leak `tools/environments/local.py` already guards for terminal subprocesses.

These test `_inject_session_context_env(..., present_only=True)`, the shared
reconciliation execute_code now applies after `_scrub_child_env`.
"""
import contextlib
import os

import pytest

import gateway.session_context as sc
from gateway.session_context import _VAR_MAP
from tools.environments.local import _inject_session_context_env

SESSION_VARS = list(_VAR_MAP.keys())


@pytest.fixture(autouse=True)
def _clean_session_state():
    saved_env = {k: os.environ.get(k) for k in SESSION_VARS}
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    saved_engaged = sc._session_context_engaged
    for var in _VAR_MAP.values():
        var.set(sc._UNSET)
    sc._session_context_engaged = False
    try:
        yield
    finally:
        for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
            var.set(val)
        sc._session_context_engaged = saved_engaged
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_present_only_corrects_passthrough_value_to_this_task(monkeypatch):
    """The reported bug: request A's child env must carry request A's session
    id, even though the process-global mirror holds a concurrent request B's."""
    sc._session_context_engaged = True
    sc._SESSION_ID.set("request-A")  # this task's ContextVar
    # Stale process-global mirror written by concurrent request B, and kept in
    # the child env because HERMES_SESSION_ID was passthrough-allowed.
    child_env = {"HERMES_SESSION_ID": "request-B-FOREIGN"}

    _inject_session_context_env(child_env, present_only=True)

    assert child_env["HERMES_SESSION_ID"] == "request-A"


def test_present_only_does_not_add_scrubbed_vars(monkeypatch):
    """present_only corrects values of vars already present; it must not
    re-introduce a session var that scrubbing filtered out by default."""
    sc._session_context_engaged = True
    sc._SESSION_ID.set("request-A")
    child_env = {"PATH": "/usr/bin"}  # HERMES_SESSION_ID scrubbed out

    _inject_session_context_env(child_env, present_only=True)

    assert "HERMES_SESSION_ID" not in child_env
    assert child_env == {"PATH": "/usr/bin"}


def test_present_only_strips_foreign_global_when_unset_and_engaged():
    """This task never bound a session id, but a concurrent host is engaged and
    a passthrough kept a foreign global — it must be stripped, not inherited."""
    sc._session_context_engaged = True
    # _SESSION_ID stays _UNSET for this task.
    child_env = {"HERMES_SESSION_ID": "request-B-FOREIGN"}

    _inject_session_context_env(child_env, present_only=True)

    assert "HERMES_SESSION_ID" not in child_env


def test_present_only_preserves_value_when_not_engaged():
    """Pure CLI/cron (never engaged): no concurrency to leak across, so a
    passthrough-kept value is left as-is."""
    sc._session_context_engaged = False
    child_env = {"HERMES_SESSION_ID": "cli-session"}

    _inject_session_context_env(child_env, present_only=True)

    assert child_env["HERMES_SESSION_ID"] == "cli-session"


def test_default_mode_still_adds_bound_vars():
    """Without present_only (the terminal path), a bound var is added even if
    absent — regression guard that the new param defaults off."""
    sc._session_context_engaged = True
    sc._SESSION_ID.set("request-A")
    child_env: dict = {}

    _inject_session_context_env(child_env)  # default present_only=False

    assert child_env["HERMES_SESSION_ID"] == "request-A"


def test_execute_code_scrub_then_reconcile_corrects_foreign_id():
    """Mirror execute_code's exact sequence — scrub the process env, then
    reconcile — and assert a passthrough-allowed foreign session id is
    corrected to this task's id while unrelated vars are untouched."""
    from tools.code_execution_tool import _scrub_child_env

    sc._session_context_engaged = True
    sc._SESSION_ID.set("request-A")

    # Passthrough allows HERMES_SESSION_ID, so scrubbing keeps the (foreign)
    # global; PATH is an ordinary inherited var.
    source = {
        "HERMES_SESSION_ID": "request-B-FOREIGN",
        "PATH": "/usr/bin",
    }
    child_env = _scrub_child_env(
        source, is_passthrough=lambda name: name == "HERMES_SESSION_ID"
    )
    assert child_env.get("HERMES_SESSION_ID") == "request-B-FOREIGN"  # pre-fix state

    _inject_session_context_env(child_env, present_only=True)  # the fix
    assert child_env["HERMES_SESSION_ID"] == "request-A"
    assert child_env.get("PATH") == "/usr/bin"
