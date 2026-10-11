"""A fail-closed smart-approval escalate must say why (#135802).

The guardian call can fail (404, timeout) or produce an empty / unrecognized answer; every
one of those fail-closes to ``escalate`` exactly like a genuine ESCALATE verdict. These tests
pin the reason channel — ``_ReasonedVerdict.reason`` → ``_smart_gate``'s third return value →
the gateway approval payload, the pending card, and the CLI prompt note — so a human can tell
"fix the judge" from "the judge wants you" instead of reading WARNING logs.
"""

import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import pytest

import tools.approval as A
from tools.approval_prompt import prompt_dangerous_approval
from tools.approval_smart import _ReasonedVerdict, _smart_approve


def _response(answer: str, finish_reason=None):
    """Mock LLM response carrying a one-word guardian answer."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = answer
    response.choices[0].finish_reason = finish_reason
    return response


class NotFoundError(Exception):
    """Stand-in for the provider SDK's 404 error class (the #135802 production failure)."""


class TestReasonedVerdictCarriesFailureReasons(unittest.TestCase):
    """_smart_approve's fail-closed escalates carry the reason; genuine verdicts do not."""

    @patch("agent.auxiliary_client.call_llm")
    def test_failed_call_carries_reason(self, mock_call_llm):
        """The #135802 production case: the judge model 404s for days and every flagged
        command escalates through the exception branch."""
        mock_call_llm.side_effect = NotFoundError("404 model z-ai/glm-5.2:free not found")
        verdict = _smart_approve("rm -rf /", "recursive delete")
        assert verdict == "escalate"
        assert verdict.reason is not None
        assert "NotFoundError" in verdict.reason
        assert "404" in verdict.reason

    @patch("agent.auxiliary_client.call_llm")
    def test_empty_answer_carries_reason(self, mock_call_llm):
        """A 200 body with empty content (finish_reason=length, #117428) is an
        infrastructure failure, not a verdict."""
        mock_call_llm.return_value = _response("", finish_reason="length")
        verdict = _smart_approve("rm -rf /", "recursive delete")
        assert verdict == "escalate"
        assert verdict.reason is not None
        assert "empty answer" in verdict.reason
        assert "length" in verdict.reason

    @patch("agent.auxiliary_client.call_llm")
    def test_unrecognized_answer_carries_reason(self, mock_call_llm):
        """A chatty non-verdict answer is not a guardian judgment either."""
        mock_call_llm.return_value = _response("I think this is probably fine")
        verdict = _smart_approve("rm -rf /", "recursive delete")
        assert verdict == "escalate"
        assert verdict.reason is not None
        assert "recognized verdict" in verdict.reason

    @patch("agent.auxiliary_client.call_llm")
    def test_genuine_verdicts_carry_no_reason(self, mock_call_llm):
        """A verdict the guardian actually chose (ESCALATE included) must not be labeled
        as an infrastructure failure."""
        for answer, expected in (("ESCALATE", "escalate"), ("APPROVE", "approve"), ("DENY", "deny")):
            with self.subTest(answer=answer):
                mock_call_llm.return_value = _response(answer)
                verdict = _smart_approve("rm -rf /", "recursive delete")
                assert verdict == expected
                assert getattr(verdict, "reason", None) is None


class TestSmartGateReasonContract(unittest.TestCase):
    """_smart_gate returns the escalate reason as its third element; a plain-str verdict
    (the established monkeypatch seam) reads as reason=None."""

    def test_reasoned_escalate_threads_reason(self):
        verdict = _ReasonedVerdict("escalate", "guardian call failed: NotFoundError: 404")
        with patch.object(A, "_smart_verdict", return_value=verdict):
            result, smart_denied, reason = A._smart_gate(
                A._ACTION_GATE, "cmd", "desc", "pk", ["pk"], "gate-reason-session",
                human_present=False,
            )
        assert result is None and smart_denied is False
        assert reason == "guardian call failed: NotFoundError: 404"

    def test_plain_str_verdict_keeps_none_reason(self):
        """Existing tests monkeypatch _smart_verdict/_smart_approve with plain strings —
        that seam must keep working (reason reads as None, not an AttributeError)."""
        with patch.object(A, "_smart_verdict", return_value="escalate"):
            result, smart_denied, reason = A._smart_gate(
                A._ACTION_GATE, "cmd", "desc", "pk", ["pk"], "gate-reason-session",
                human_present=False,
            )
        assert result is None and smart_denied is False
        assert reason is None


def _pending(spec, reason):
    A.submit_pending("pending-reason-session", {})
    try:
        return A._pending_result(
            spec, "pending-reason-session", command="cmd", description="desc",
            pattern_key="pk", pattern_keys=["pk"], body=None, smart_denied=False,
            smart_escalate_reason=reason,
        )
    finally:
        with A._lock:
            A._pending.pop("pending-reason-session", None)


class TestPendingResultReason:
    """The pending card (no gateway notifier, no CLI panel) tells both /approve reviewers
    and the agent reading the tool result that the escalate was the fail-closed fallback."""

    def test_pending_result_surfaces_reason(self):
        result = _pending(A._EXECUTE_CODE_GATE, "guardian call failed: NotFoundError: 404")
        assert result["smart_escalate_reason"] == "guardian call failed: NotFoundError: 404"
        assert "guardian is unavailable" in result["message"]
        assert "404" in result["message"]

    def test_pending_result_without_reason_is_unchanged(self):
        result = _pending(A._EXECUTE_CODE_GATE, None)
        assert "smart_escalate_reason" not in result
        assert "guardian is unavailable" not in result["message"]


class TestGatewayApprovalData:
    """The gateway approval payload carries the fail-closed reason for the UI (#135802)."""

    def test_reason_rides_the_gateway_payload(self):
        data = A._gateway_approval_data("cmd", "desc", "pk", ["pk"], False,
                                        "guardian call failed: NotFoundError: 404")
        assert data["smart_escalate_reason"] == "guardian call failed: NotFoundError: 404"
        assert data["allow_permanent"] is True and data["allow_session"] is True

    def test_genuine_escalate_payload_is_unchanged(self):
        data = A._gateway_approval_data("cmd", "desc", "pk", ["pk"], False, None)
        assert "smart_escalate_reason" not in data


class TestGuardianNote:
    """The CLI note distinguishes an unavailable judge from an uncertain one."""

    def test_unavailable_note_carries_reason(self):
        note = A._guardian_note(True, "guardian call failed: NotFoundError: 404")
        assert "unavailable" in note
        assert "404" in note

    def test_uncertain_note_without_reason(self):
        assert "uncertain" in A._guardian_note(True, None)

    def test_no_note_outside_smart_mode(self):
        assert A._guardian_note(False, "guardian call failed: NotFoundError: 404") is None


class TestCliPromptNote:
    """The plain-input CLI prompt prints the guardian note under the command."""

    def test_prompt_renders_guardian_note(self, monkeypatch):
        import tools.approval_prompt as prompt_mod
        monkeypatch.setattr(prompt_mod._ctx, "_get_approval_timeout", lambda: 1)
        monkeypatch.setattr(prompt_mod, "_read_choice", lambda prompt, timeout: "d")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            choice = prompt_dangerous_approval(
                "echo hi", "test flag", guardian_note="⚠️ Guardian check unavailable — fail-closed to you: NotFoundError: 404")
        assert choice == "deny"
        assert "Guardian check unavailable" in buffer.getvalue()
        assert "NotFoundError: 404" in buffer.getvalue()

    def test_prompt_without_note_has_no_guardian_line(self, monkeypatch):
        import tools.approval_prompt as prompt_mod
        monkeypatch.setattr(prompt_mod._ctx, "_get_approval_timeout", lambda: 1)
        monkeypatch.setattr(prompt_mod, "_read_choice", lambda prompt, timeout: "d")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            prompt_dangerous_approval("echo hi", "test flag")
        assert "Guardian" not in buffer.getvalue()
