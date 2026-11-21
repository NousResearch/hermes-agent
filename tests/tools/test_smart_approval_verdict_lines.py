"""Regression tests for reading a guardian verdict wrapped in text (#135146, #148382).

``approvals.mode: smart`` asks the auxiliary LLM for exactly one word, but some
models answer with a short justification paragraph and the verdict on the last
line. The old ``_VERDICTS.get(answer, ...)`` whole-string lookup discarded the
guardian's actual verdict and escalated silently (#135146).

The verdict reader must stay fail-closed: only the final non-empty line speaks,
emphasis markup and trailing punctuation are stripped, and a verdict word that
appears only mid-prose is never honoured.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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
    def test_interior_conflicting_verdict_escalates(self, mock_call_llm):
        # The first line must not grant approval when a bare DENY follows,
        # even if the final line is just the guardian's explanation.
        mock_call_llm.return_value = self._make_response(
            "APPROVE\nDENY\nThe command deletes /data."
        )
        assert _smart_approve("rm -rf /data", "recursive delete") == "escalate"

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


class TestVerdictOnFirstOrLastLine(unittest.TestCase):
    """The verdict may sit on the first line (#148382) or the last (#135146), never in between,
    and an answer whose first and last line disagree has no verdict."""

    def test_verdict_first_then_justification(self):
        assert _verdict_from_answer("APPROVE\n\nThe script only prints a sum.") == "approve"
        assert _verdict_from_answer("**DENY**\nThis wipes the home directory") == "deny"

    def test_justification_cut_off_at_max_tokens(self):
        # max_tokens=16 truncates the justification mid-sentence; the first line still rules.
        assert _verdict_from_answer("APPROVE\n\nThe command only reads the") == "approve"

    def test_first_and_last_line_disagree(self):
        assert _verdict_from_answer("APPROVE\n\nOn reflection this deletes data.\nDENY") is None
        assert _verdict_from_answer("DENY\n\nIt is only a read.\n**APPROVE**") is None

    def test_escalate_on_either_end_vetoes_the_other(self):
        assert _verdict_from_answer("ESCALATE\n\nIt looks benign, so\nAPPROVE") is None
        assert _verdict_from_answer("APPROVE\n\nBut the echo text is suspicious.\nESCALATE") is None

    def test_same_verdict_on_both_ends(self):
        assert _verdict_from_answer("APPROVE\nRead-only listing.\nAPPROVE") == "approve"

    def test_negated_verdict_on_last_line_is_not_a_verdict(self):
        # Only a line that is the bare word counts: a sentence ending in the word does not.
        assert _verdict_from_answer("This overwrites /etc/hosts.\nI cannot APPROVE") is None
        assert _verdict_from_answer("Verdict: do not APPROVE.") is None

    def test_verdict_word_only_in_the_middle(self):
        assert _verdict_from_answer("The command is fine.\nAPPROVE\nIt only lists files.") is None

    def test_blank_answer(self):
        assert _verdict_from_answer("  \n\n ") is None


class TestSmartApproveFirstOrLastLine(unittest.TestCase):
    """End-to-end through _smart_approve with a stubbed guardian."""

    def _make_response(self, answer: str):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = answer
        mock_response.choices[0].finish_reason = "length"
        return mock_response

    @patch("agent.auxiliary_client.call_llm")
    def test_truncated_verdict_first_answer_is_approved(self, mock_call_llm):
        mock_call_llm.return_value = self._make_response("APPROVE\n\nThis script only performs a")
        assert _smart_approve("python -c 'print(1+1)'", "script execution via -c flag") == "approve"

    @patch("agent.auxiliary_client.call_llm")
    def test_contradicting_answer_escalates(self, mock_call_llm):
        mock_call_llm.return_value = self._make_response("APPROVE\n\nIt removes the data dir.\nDENY")
        assert _smart_approve("rm -rf ./data", "recursive delete") == "escalate"


def _guardian_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.parametrize("reply,expected", [
    # Exact one-word replies (existing contract).
    ("APPROVE", "approve"),
    ("DENY", "deny"),
    # The prompt asks for ESCALATE but the classifier only grants approve/deny;
    # an ESCALATE word is not an exact key and must keep escalating.
    ("ESCALATE", "escalate"),
    # #148382: a chatty guardian appends a justification after the verdict
    # (often truncated at max_tokens=16). The first line is an explicit
    # verdict, so the reply must not escalate.
    ("APPROVE\n\nThis script only performs a simple arithmetic computation", "approve"),
    ("DENY\n\nThis deletes every file in the directory.", "deny"),
    # Markdown-wrapped bare verdicts count as explicit.
    ("**APPROVE**", "approve"),
    ("**DENY**", "deny"),
    # Anything hedged or annotated on the verdict line still escalates.
    ("**DENY** - this deletes files", "escalate"),
    ("APPROVE, but it depends on the environment", "escalate"),
    ("I think APPROVE is right", "escalate"),
    ("APPROVED", "escalate"),
    # Empty reply: infrastructure failure, escalates.
    ("", "escalate"),
])
def test_verdict_reply_parsing(reply, expected):
    with patch("agent.auxiliary_client.call_llm", return_value=_guardian_response(reply)):
        result = _smart_approve("python -c \"print('hello')\"", "script execution via -c flag")

    assert result == expected


def test_trailing_text_after_verdict_is_logged(caplog):
    reply = "APPROVE\n\nThis script only performs a simple arithmetic computation"
    with patch("agent.auxiliary_client.call_llm", return_value=_guardian_response(reply)), \
            caplog.at_level("INFO", logger="tools.approval"):
        _smart_approve("python -c \"print('hello')\"", "script execution via -c flag")

    assert any("first line" in r.message for r in caplog.records)
