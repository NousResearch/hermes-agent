"""browser_exec code is approved like an execute_code script (#157913).

The harness runs the model's ``code`` as Python on this host whichever browser it drives, so
messaging/API/ask sessions approve the whole script once before anything starts. Unattended
contexts (cron, -q, unattended platforms) are not asked: scheduled browser runs are a documented
feature. The harness spawn is replaced by a recorder; no code is run.
"""
import json
import subprocess

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tools import approval as A
from tools import approval_context
from tools import browser_use_cli as B

SESSION = "browser-exec-approval"
CODE = "import subprocess; subprocess.run(['true'])"


def _exec() -> dict:
    out = B.browser_exec(CODE)
    assert isinstance(out, str)  # no screenshot, so never the multimodal envelope
    return json.loads(out)


@pytest.fixture
def harness(monkeypatch):
    """Every browser_exec call that reaches the harness spawn, by code."""
    ran = []
    monkeypatch.setattr(B, "_find_cli", lambda: ["python", "-m", "browser_harness.run"])
    monkeypatch.setattr(B, "_route_exec_current", lambda env, *a, **k: (env, None, False, 0))
    for name in ("_reload_if_republished", "_own_harness_daemon", "_own_harness_daemon_until_owned",
                 "_attach_vault_supervisor", "_record_driven_tab"):
        monkeypatch.setattr(B, name, lambda *a, **k: None)

    def spawn(cmd, code, env, timeout):
        ran.append(code)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(B, "_run_cli_killing_process_group", spawn)
    return ran


@pytest.fixture
def manual(monkeypatch):
    for var in ("HERMES_GATEWAY_SESSION", "HERMES_INTERACTIVE", "HERMES_CRON_SESSION", "HERMES_EXEC_ASK",
                "HERMES_YOLO_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    token = approval_context.set_current_session_key(SESSION)
    A.clear_session(SESSION)
    yield
    A.clear_session(SESSION)
    with A._lock:
        A._gateway_queues.pop(SESSION, None)
        A._gateway_notify_cbs.pop(SESSION, None)
    approval_context.reset_current_session_key(token)


@pytest.fixture
def gateway_user(manual, monkeypatch):
    """A messaging session whose user answers each prompt with the next queued answer."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    answers, prompts = [], []

    def notify(data):
        prompts.append(data)
        with A._lock:
            entry = A._gateway_queues[SESSION][-1]
            entry.result = answers.pop(0)
            entry.event.set()

    with A._lock:
        A._gateway_notify_cbs[SESSION] = notify
    return answers, prompts


def test_a_denied_script_never_reaches_the_harness(gateway_user, harness):
    answers, prompts = gateway_user
    answers.append("deny")
    result = _exec()
    assert "error" in result and "browser_exec" in result["error"]
    assert harness == []
    assert len(prompts) == 1 and CODE in json.dumps(prompts[0])


def test_an_approved_script_runs_and_approval_is_one_shot(gateway_user, harness):
    answers, prompts = gateway_user
    answers.extend(["once", "deny"])
    assert _exec()["success"] is True
    assert harness == [CODE]
    _exec()
    assert harness == [CODE] and len(prompts) == 2


def test_an_execute_code_session_grant_does_not_cover_browser_exec(gateway_user, harness):
    answers, prompts = gateway_user
    A.approve_session(SESSION, "execute_code")
    answers.append("deny")
    _exec()
    assert harness == [] and len(prompts) == 1


def test_an_interactive_cli_session_is_not_asked_per_script(manual, harness, monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    assert _exec()["success"] is True
    assert harness == [CODE]


def test_a_cron_run_keeps_its_browser_even_when_cron_mode_denies(manual, harness, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "deny")
    tokens = set_session_vars(cron_session="1")
    try:
        result = _exec()
    finally:
        clear_session_vars(tokens)
    assert result["success"] is True and harness == [CODE]
