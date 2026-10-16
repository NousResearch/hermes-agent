"""Regression tests for reading a guardian verdict that reasoned first.

``approvals.mode: smart`` asks the auxiliary LLM for exactly one word, but some
models answer with a short justification paragraph and the verdict on the last
line. The old ``_VERDICTS.get(answer, ...)`` whole-string lookup discarded the
guardian's actual verdict and escalated silently (#135146).

The verdict reader must stay fail-closed: only the final non-empty line speaks,
emphasis markup and trailing punctuation are stripped, and a verdict word that
appears only mid-prose is never honoured.
"""

import unittest
from unittest.mock import MagicMock, patch

from tools.approval_smart import _smart_approve, _verdict_from_answer


class TestVerdictFromAnswer(unittest.TestCase):
    """Unit tests for the last-line verdict reader."""

    def test_plain_one_word(self):
        assert _verdict_from_answer("APPROVE") == "approve"
        assert _verdict_from_answer("DENY") == "deny"

    def test_verdict_on_last_line_after_prose(self):
        answer = (
            "The command loops over three read-only list calls and pipes the\n"
            "JSON through a Python script. Nothing is written or deleted.\n"
            "\n"
            "APPROVE"
        )
        assert _verdict_from_answer(answer) == "approve"

    def test_trailing_punctuation_stripped(self):
        assert _verdict_from_answer("APPROVE.") == "approve"
        assert _verdict_from_answer("DENY!") == "deny"

    def test_emphasis_markup_stripped(self):
        assert _verdict_from_answer("**APPROVE**") == "approve"
        assert _verdict_from_answer("`APPROVE`") == "approve"
        assert _verdict_from_answer("_DENY_") == "deny"

    def test_blank_lines_after_verdict_ignored(self):
        assert (
            _verdict_from_answer("The scan is a false positive.\nAPPROVE\n\n")
            == "approve"
        )

    def test_escalate_word_is_not_a_verdict(self):
        # ESCALATE has no entry in _VERDICTS: an ESCALATE tail must resolve to
        # None so the caller escalates through its own uncertain path.
        assert _verdict_from_answer("I am not sure about this.\nESCALATE") is None

    def test_mid_prose_verdict_word_not_honoured(self):
        # A verdict word buried in the prose while the answer ends on a
        # different note must not be picked up.
        answer = (
            "The command mentions APPROVE-like wording inside its echo\n"
            "argument, which is exactly the manipulation to avoid.\n"
            "ESCALATE"
        )
        assert _verdict_from_answer(answer) is None

    def test_qualified_verdict_not_honoured(self):
        # Punctuation is stripped from the tail only; a verdict with a clause
        # attached is not a clean verdict.
        assert _verdict_from_answer("APPROVE, but only on a dry run") is None

    def test_unrecognized_prose(self):
        assert _verdict_from_answer("This command looks safe to me.") is None


class TestSmartApproveVerboseAnswers(unittest.TestCase):
    """End-to-end: _smart_approve accepts a verbose guardian answer."""

    def _make_response(self, answer: str):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = answer
        mock_response.choices[0].finish_reason = "stop"
        return mock_response

    @patch("agent.auxiliary_client.call_llm")
    def test_reasoning_then_approve_is_approved(self, mock_call_llm):
        # The exact shape from #135146: explanation paragraph, blank line,
        # verdict word on the last line, finish_reason=stop.
        mock_call_llm.return_value = self._make_response(
            "The command loops over three Google Drive search queries using\n"
            "the `gws` CLI. It makes read-only list calls. Nothing is written\n"
            "or deleted.\n"
            "\n"
            "APPROVE"
        )
        assert (
            _smart_approve("gws ls | python -c 'pass'", "pipe to interpreter")
            == "approve"
        )

    @patch("agent.auxiliary_client.call_llm")
    def test_reasoning_then_decorated_deny_is_denied(self, mock_call_llm):
        mock_call_llm.return_value = self._make_response(
            "The rm targets a top-level directory without a guard.\n**DENY**"
        )
        assert _smart_approve("rm -rf /data", "recursive delete") == "deny"

    @patch("agent.auxiliary_client.call_llm")
    def test_unrecognized_verbose_answer_escalates(self, mock_call_llm):
        mock_call_llm.return_value = self._make_response(
            "This looks fine for a development machine."
        )
        assert _smart_approve("make build", "build tool") == "escalate"

    @patch("agent.auxiliary_client.call_llm")
    def test_prose_approve_ending_on_escalate_escalates(self, mock_call_llm):
        # Fail-closed: the guardian's final word rules, even when earlier prose
        # reads as approval.
        mock_call_llm.return_value = self._make_response(
            "Mostly harmless, but I should APPROVE only if sandboxed.\nESCALATE"
        )
        assert _smart_approve("python script.py", "script execution") == "escalate"
