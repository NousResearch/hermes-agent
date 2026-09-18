"""Smart approval must never auto-approve gateway-lifecycle commands.

Regression tests for the "gateway self-destruct" hole: under
``approvals.mode: smart`` the auxiliary-LLM guardian assessed commands like
``hermes gateway stop`` against its generic risk rubric (recursive deletes,
fork bombs, disk wipes), found them harmless, and auto-approved them — with
zero human involvement.  ``hermes gateway stop`` runs ``launchctl bootout``,
which UNLOADS the launchd job, so KeepAlive never respawns the gateway: one
auto-approved command took the gateway down until a human manually started
it again.  The same hole existed for ``execute_code`` scripts embedding the
lifecycle command via ``subprocess``/``os.system``, which bypass the
terminal() guard entirely.

Invariants under test (both guard entry points):

  1. A flagged gateway-lifecycle command is NEVER passed to the guardian LLM
     — the guardian is not consulted at all.
  2. The command reaches the human approval surface instead (gateway prompt
     path), and a human "once"/"session" approval still executes it.
  3. Ordinary dangerous commands in smart mode still go through the guardian
     (the exemption is surgical, not a mode-wide regression).
  4. The exempt key set stays in lockstep with the dangerous-pattern
     registry, so a future pattern rename cannot silently re-open the hole.

See PR #96555.
"""

from __future__ import annotations

import pytest

from tools import approval as A
from tools import approval_context
from tools import approval_detection
from tools import approval_smart
from tools.approval_smart import SMART_APPROVAL_HUMAN_ONLY_DESCRIPTIONS


@pytest.fixture
def smart_gateway_session(monkeypatch):
    """A clean gateway smart-mode session where the guardian would APPROVE.

    Forces the guardian verdict to "approve" so that any command which is
    (incorrectly) handed to the guardian is auto-approved — exactly the
    pre-fix hole.  Commands that correctly bypass the guardian proceed to
    the human approval queue instead.
    """
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )

    calls: list[str] = []

    def _recording_smart_approve(command, description):
        calls.append(command)
        return "approve"  # the pre-fix hole: guardian waves it through

    monkeypatch.setattr(approval_smart, "_smart_approve", _recording_smart_approve)

    session_key = "gw-lifecycle-smart-test"
    token = approval_context.set_current_session_key(session_key)
    with A._lock:
        A._gateway_queues.pop(session_key, None)
        A._gateway_notify_cbs.pop(session_key, None)
        A._session_approved.pop(session_key, None)
        A._permanent_approved.discard("execute_code")
    try:
        yield session_key, calls
    finally:
        approval_context.reset_current_session_key(token)
        with A._lock:
            A._gateway_queues.pop(session_key, None)
            A._gateway_notify_cbs.pop(session_key, None)
            A._session_approved.pop(session_key, None)


def _register_resolver(session_key: str, result):
    """Notify callback resolving the newest queued approval with *result*."""

    def cb(_approval_data):
        with A._lock:
            entries = A._gateway_queues.get(session_key, [])
            if entries:
                entries[-1].result = result
                entries[-1].event.set()

    with A._lock:
        A._gateway_notify_cbs[session_key] = cb


# ---------------------------------------------------------------------------
# 0. The exempt set tracks the live pattern registry (review point: "set and
#    patterns must not drift apart").
# ---------------------------------------------------------------------------

def test_human_only_set_is_covered_by_the_pattern_registry():
    """Every human-only key must still exist as a live detection description.

    The guardian exemption is keyed on the *description* strings used by
    ``DANGEROUS_PATTERNS_COMPILED``.  A rename there would leave a stale name
    in this set and silently re-open the auto-approval hole, so pin the
    relationship instead of the list.
    """
    live = {description for _, description in approval_detection.DANGEROUS_PATTERNS}
    live.add(approval_detection._GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION)

    stale = SMART_APPROVAL_HUMAN_ONLY_DESCRIPTIONS - live
    assert not stale, (
        "SMART_APPROVAL_HUMAN_ONLY_DESCRIPTIONS names patterns that no longer "
        f"exist in the detection registry: {sorted(stale)}"
    )


def test_lifecycle_keys_are_detected_as_gateway_lifecycle():
    """The registry entries behind the exemption must actually flag as dangerous."""
    for command in ("hermes gateway stop", "hermes update"):
        dangerous, pattern_key, _ = approval_detection.detect_dangerous_command(command)
        assert dangerous is True
        assert pattern_key in SMART_APPROVAL_HUMAN_ONLY_DESCRIPTIONS


# ---------------------------------------------------------------------------
# 1. Guardian is never consulted for gateway-lifecycle commands
# ---------------------------------------------------------------------------

LIFECYCLE_COMMANDS = [
    "hermes gateway stop",
    "hermes gateway restart",
    "hermes -p davidhcox8-gmail-com gateway stop",
    "launchctl bootout gui/501/ai.hermes.gateway",
    "launchctl kickstart -k gui/501/ai.hermes.gateway",
]


@pytest.mark.parametrize("command", LIFECYCLE_COMMANDS)
def test_gateway_lifecycle_never_reaches_guardian(smart_gateway_session, command):
    session_key, guardian_calls = smart_gateway_session
    _register_resolver(session_key, "once")

    result = A.check_all_command_guards(command, "local")

    # The guardian would have APPROVED (the old hole). It must never run.
    assert guardian_calls == [], (
        f"smart-approval guardian was consulted for {command!r} — an APPROVE "
        "verdict there auto-approves gateway self-termination with no human"
    )
    # And it must have reached the human surface: the resolver answered
    # "once", so the command is approved via explicit human consent.
    assert result["approved"] is True
    assert result.get("user_approved") is True
    assert result.get("smart_approved") is not True


# ---------------------------------------------------------------------------
# 2. Human denial still blocks (the prompt is real, not a rubber stamp)
# ---------------------------------------------------------------------------

def test_gateway_lifecycle_human_deny_blocks(smart_gateway_session):
    session_key, guardian_calls = smart_gateway_session
    _register_resolver(session_key, "deny")

    result = A.check_all_command_guards("hermes gateway stop", "local")

    assert guardian_calls == []
    assert result["approved"] is False
    assert "NOT consented" in result["message"]


# ---------------------------------------------------------------------------
# 3. The exemption is surgical: ordinary dangerous commands still use the
#    guardian in smart mode.
# ---------------------------------------------------------------------------

def test_ordinary_dangerous_command_still_uses_guardian(smart_gateway_session):
    session_key, guardian_calls = smart_gateway_session

    result = A.check_all_command_guards("rm -rf /tmp/build-artifacts", "local")

    assert len(guardian_calls) == 1
    assert result["approved"] is True
    assert result.get("smart_approved") is True


# ---------------------------------------------------------------------------
# 4. execute_code scripts embedding lifecycle commands skip the guardian too
# ---------------------------------------------------------------------------

def test_execute_code_lifecycle_skips_guardian(smart_gateway_session):
    session_key, guardian_calls = smart_gateway_session
    _register_resolver(session_key, "once")

    code = (
        "import subprocess\n"
        'subprocess.run(["launchctl", "bootout", '
        '"gui/501/ai.hermes.gateway"], check=True)\n'
    )
    result = A.check_execute_code_guard(code, "local")

    assert guardian_calls == [], (
        "smart-approval guardian was consulted for an execute_code script "
        "embedding a gateway-lifecycle command"
    )
    assert result["approved"] is True
    assert result.get("smart_approved") is not True


def test_execute_code_plain_script_still_uses_guardian(smart_gateway_session):
    session_key, guardian_calls = smart_gateway_session

    result = A.check_execute_code_guard("print('harmless')", "local")

    assert len(guardian_calls) == 1
    assert result["approved"] is True
    assert result.get("smart_approved") is True


# ---------------------------------------------------------------------------
# 5. Fail-closed: a detector error must route to the human prompt, never
#    back to the guardian (Copilot review on PR #96555).
# ---------------------------------------------------------------------------

def test_execute_code_detector_error_fails_closed(smart_gateway_session, monkeypatch):
    session_key, guardian_calls = smart_gateway_session
    _register_resolver(session_key, "once")

    def _raising_detector(_code):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(
        "cron.lifecycle_guard.contains_gateway_lifecycle_command",
        _raising_detector,
    )

    result = A.check_execute_code_guard("print('anything')", "local")

    # Failing open would hand the script to the guardian (the pre-fix hole,
    # reintroduced). Fail-closed means: guardian never consulted, human
    # prompt decides.
    assert guardian_calls == [], (
        "detector error routed the script to the guardian instead of the "
        "human prompt — that is fail-open and reintroduces the outage class"
    )
    assert result["approved"] is True
    assert result.get("user_approved") is True
