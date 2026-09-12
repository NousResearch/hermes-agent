"""The approval vocabulary is a contract, not a comment: only the documented words grant.

``prompt_dangerous_approval`` documents ``once | session | always | deny | timeout``, but the three
decision points in ``tools/approval.py`` used to grant on anything that was not exactly a refusal.
So an answer the gate did not recognise — ``None`` from a callback whose UI went away, ``""`` from a
cancelled prompt, ``"no"``/``"approve"`` from a third-party callback, plugin transport or platform
button written to the obvious rather than the documented word — APPROVED the dangerous command.

These tests pin the contract in both directions on all three answer channels (CLI approval callback,
gateway ``resolve_gateway_approval``, selected plugin transport): every documented granting word
still grants, ``deny``/``timeout`` refuse, and everything outside the vocabulary refuses too.
Silence and confusion are not consent.
"""
import os
import threading
import time

import pytest

from tools import approval as approval_module
from tools import approval_context
from tools.approval_detection import detect_dangerous_command

# The documented vocabulary. Only the first three may ever produce approved=True.
GRANTING = ["once", "session", "always"]
REFUSING = ["deny", "timeout"]

#: Answers no surface is allowed to turn into consent. ``"approve"``/``"allow"``/``"yes"`` are the
#: words a callback author reaches for instead of the documented ones; ``""``/``None`` are what a
#: cancelled or vanished prompt produces; the rest are junk that must never read as a grant.
OUT_OF_VOCABULARY = ["", "   ", "no", "n", "y", "yes", "approve", "allow", "ok", "cancelled",
                     "null", "None", "granted", "once please", "deny\nonce", "\x00", "0", "1",
                     None, 0, 1, True, [], {"choice": "once"}]

DANGEROUS = "rm -rf /tmp/hermes-approval-vocabulary-test"


def _reset_approval_state():
    approval_module._session_approved.clear()
    approval_module._permanent_approved.clear()
    approval_module._gateway_queues.clear()
    approval_module._gateway_notify_cbs.clear()
    approval_module._pending.clear()


@pytest.fixture(autouse=True)
def _manual_approvals(monkeypatch, request):
    """A dangerous command with no cached approval, in manual mode, so every test reaches a human.

    Each test gets its own session key: a ``session``/``always`` grant is remembered, and a cached
    approval would short-circuit the next test before it ever asked.
    """
    session_key = f"test:vocab:{request.node.name}"
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"mode": "manual", "timeout": 5})
    _reset_approval_state()
    try:
        yield session_key
    finally:
        _reset_approval_state()


def _guard(**kwargs) -> dict:
    return approval_module.check_all_command_guards(DANGEROUS, "local", **kwargs)


# ── Channel 1: the interactive CLI approval callback ────────────────────────


@pytest.fixture
def _cli(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)


@pytest.mark.usefixtures("_cli")
@pytest.mark.parametrize("answer", GRANTING)
def test_cli_callback_granting_word_still_grants(answer):
    """The fix must not break the gate it protects: the documented words keep working."""
    assert _guard(approval_callback=lambda *a, **k: answer)["approved"] is True


@pytest.mark.usefixtures("_cli")
@pytest.mark.parametrize("answer", GRANTING)
def test_cli_callback_granting_word_survives_whitespace_and_case(answer):
    assert _guard(approval_callback=lambda *a, **k: f"  {answer.upper()}  ")["approved"] is True


@pytest.mark.usefixtures("_cli")
@pytest.mark.parametrize("answer", REFUSING)
def test_cli_callback_refusing_word_refuses(answer):
    assert _guard(approval_callback=lambda *a, **k: answer)["approved"] is False


@pytest.mark.usefixtures("_cli")
@pytest.mark.parametrize("answer", OUT_OF_VOCABULARY)
def test_cli_callback_outside_the_vocabulary_refuses(answer):
    result = _guard(approval_callback=lambda *a, **k: answer)
    assert result["approved"] is False, f"{answer!r} must not approve {DANGEROUS!r}"
    assert result.get("user_consent") is not True


@pytest.mark.usefixtures("_cli")
def test_a_callback_that_raises_refuses():
    def boom(*a, **k):
        raise RuntimeError("the approval UI died")

    assert _guard(approval_callback=boom)["approved"] is False


@pytest.mark.usefixtures("_cli")
@pytest.mark.parametrize("answer", OUT_OF_VOCABULARY)
def test_an_unrecognised_answer_is_not_remembered(answer):
    """A refused answer must leave no cached approval behind: the next command has to ask again."""
    _guard(approval_callback=lambda *a, **k: answer)
    assert _guard(approval_callback=lambda *a, **k: "deny")["approved"] is False
    pattern_key = detect_dangerous_command(DANGEROUS)[1]
    assert approval_module.is_approved(os.environ["HERMES_SESSION_KEY"], pattern_key) is False


# ── Channel 2: the gateway queue (/approve, a button tap, an RPC client) ────


@pytest.fixture
def _gateway(monkeypatch):
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")


def _answer_through_the_gateway(session_key: str, choice) -> dict:
    """Run the gate on a worker thread and answer its pending prompt with *choice*."""
    notified: list = []
    approval_module.register_gateway_notify(session_key, notified.append)
    holder: dict = {}
    worker = threading.Thread(target=lambda: holder.update(result=_guard()))
    worker.start()
    try:
        for _ in range(500):  # the waiter enqueues before notify returns
            if approval_module._gateway_queues.get(session_key):
                break
            time.sleep(0.01)
        assert approval_module.resolve_gateway_approval(session_key, choice) == 1, (
            "no waiter to answer — the gate never asked")
        worker.join(timeout=10)
    finally:
        approval_module.register_gateway_notify(session_key, None)
    assert notified, "a dangerous command with a gateway listener MUST raise a prompt"
    assert "result" in holder, "the approval wait never returned"
    return holder["result"]


@pytest.mark.usefixtures("_gateway")
@pytest.mark.parametrize("choice", GRANTING)
def test_gateway_granting_word_still_grants(_manual_approvals, choice):
    assert _answer_through_the_gateway(_manual_approvals, choice)["approved"] is True


@pytest.mark.usefixtures("_gateway")
@pytest.mark.parametrize("choice", REFUSING)
def test_gateway_refusing_word_refuses(_manual_approvals, choice):
    assert _answer_through_the_gateway(_manual_approvals, choice)["approved"] is False


@pytest.mark.usefixtures("_gateway")
@pytest.mark.parametrize("choice", OUT_OF_VOCABULARY)
def test_gateway_answer_outside_the_vocabulary_refuses(_manual_approvals, choice):
    result = _answer_through_the_gateway(_manual_approvals, choice)
    assert result["approved"] is False, f"{choice!r} must not approve {DANGEROUS!r}"
    assert result.get("user_consent") is not True


# ── Channel 3: an operator-selected plugin approval transport ───────────────


def _fake_transport(monkeypatch, choice):
    """Stand in for a selected transport that answered *choice* (no plugin machinery needed)."""
    monkeypatch.setattr(approval_module, "_present_with_selected_transport",
                        lambda **kwargs: {"selected": True, "choice": choice,
                                          "failure": None, "fallback": None, "name": "fake"})


@pytest.mark.usefixtures("_cli")
@pytest.mark.parametrize("choice", GRANTING)
def test_transport_granting_word_still_grants(monkeypatch, choice):
    _fake_transport(monkeypatch, choice)
    assert _guard()["approved"] is True


@pytest.mark.usefixtures("_cli")
@pytest.mark.parametrize("choice", REFUSING)
def test_transport_refusing_word_refuses(monkeypatch, choice):
    _fake_transport(monkeypatch, choice)
    assert _guard()["approved"] is False


# ``None`` is excluded: it is the transport's documented "no transport selected" signal, which falls
# through to the built-in surfaces rather than deciding anything.
@pytest.mark.usefixtures("_cli")
@pytest.mark.parametrize("choice", [a for a in OUT_OF_VOCABULARY if a is not None])
def test_transport_answer_outside_the_vocabulary_refuses(monkeypatch, choice):
    _fake_transport(monkeypatch, choice)
    result = _guard(approval_callback=lambda *a, **k: "once")  # must never be consulted
    assert result["approved"] is False, f"{choice!r} must not approve {DANGEROUS!r}"
