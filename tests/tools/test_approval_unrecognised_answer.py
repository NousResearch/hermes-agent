"""Only an explicit grant runs a flagged command: an answer outside once/session/always is a deny.

Every approval surface hands the gate a string (a CLI or ACP callback's return value, an
``approval.respond`` RPC ``choice``, a native card's button id). A misspelled, capitalised or
foreign label ("Deny", "reject", "no", "") must block the command, never run it.
"""
import threading

import pytest

from tools import approval as mod
from tools import approval_context

SESSION_KEY = "agent:main:test:dm:unrecognised-answer"
UNRECOGNISED = ["Deny", "reject", "no", ""]


def _clean_presence(monkeypatch):
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    for name in ("HERMES_CRON_SESSION", "HERMES_YOLO_MODE", "HERMES_INTERACTIVE",
                 "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual", "timeout": 60})


@pytest.fixture
def cli_session(monkeypatch):
    _clean_presence(monkeypatch)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION_KEY)
    yield
    mod._session_approved.clear()


@pytest.fixture
def gateway_session(monkeypatch):
    _clean_presence(monkeypatch)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION_KEY)
    yield
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()


def _assert_blocked(result):
    assert result["approved"] is False, f"an unrecognised answer ran the command: {result}"
    assert not mod.is_approved(SESSION_KEY, result.get("pattern_key") or "")


@pytest.mark.parametrize("answer", UNRECOGNISED)
def test_cli_callback_answer_outside_the_grants_blocks_the_command(cli_session, answer):
    result = mod.check_all_command_guards("rm -rf .git", "local",
                                          approval_callback=lambda *_a, **_k: answer)
    _assert_blocked(result)


@pytest.mark.parametrize("answer", UNRECOGNISED)
def test_gateway_answer_outside_the_grants_blocks_the_command(gateway_session, answer):
    notified = threading.Event()
    mod.register_gateway_notify(SESSION_KEY, lambda _data: notified.set())
    holder = {}
    worker = threading.Thread(
        target=lambda: holder.__setitem__("result", mod.check_all_command_guards("rm -rf .git", "local")))
    worker.start()
    assert notified.wait(timeout=10), "approval was never enqueued"
    assert mod.resolve_gateway_approval(SESSION_KEY, answer) == 1
    worker.join(timeout=10)
    assert not worker.is_alive()
    _assert_blocked(holder["result"])
