"""Tests for the pattern brief the smart-approval guardian reads.

Some dangerous-command pattern keys name the trigger, not the target: ``rm -f /tmp/scratch.md``
is flagged as ``delete in root path``, and a guardian reading that key alone treats routine
cleanup as a system-root deletion. The brief under test states what the rule really matches.

Invariants under test:

  1. A key with a brief reaches the guardian's user message carrying that brief.
  2. The key itself is untouched on every other surface: the tuple that the allowlist, the
     decision journal, standing approvals and the operator prompt all read keeps the bare key.
  3. A key without a brief leaves the guardian prompt byte-identical to its previous format.
"""

import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from tools import approval_smart
from tools.approval_detection import detect_dangerous_command, pattern_judge_brief
from tools.approval_smart import _smart_approve, _smart_verdict


def _make_response(answer: str):
    """Build a mock LLM response with the given one-word answer."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = answer
    return mock_response


def _user_message(mock_call_llm) -> str:
    """Extract the user-message text passed to call_llm."""
    call_args = mock_call_llm.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages", [])
    return messages[1]["content"]


class TestPatternJudgeBrief(unittest.TestCase):
    def test_key_without_a_brief_returns_none(self):
        assert pattern_judge_brief("some future pattern") is None

    def test_missing_key_returns_none(self):
        assert pattern_judge_brief(None) is None
        assert pattern_judge_brief("") is None

    def test_over_broad_key_explains_its_real_scope(self):
        brief = pattern_judge_brief("delete in root path")
        assert brief is not None
        assert "absolute" in brief

    def test_execute_code_key_is_covered(self):
        assert "child process" in (pattern_judge_brief("execute_code") or "")

    @mock.patch("tempfile.gettempdir", return_value="/tmp")
    def test_detection_tuple_keeps_the_bare_key(self, _mock_tmp):
        """The brief must never reach the detection tuple: the allowlist, the decision journal
        and standing approvals all match on the key, and a renamed key would silently stop
        matching them."""
        dangerous, key, description = detect_dangerous_command("rm -f /tmp/hermes-scratch.md")
        assert dangerous is True
        assert key == "delete in root path"
        assert description == key


class TestGuardianPromptCarriesTheBrief(unittest.TestCase):
    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_brief_reaches_the_guardian_user_message(self, mock_call_llm, mock_cfg):
        mock_call_llm.return_value = _make_response("APPROVE")
        mock_cfg.return_value = {"mode": "smart"}

        _smart_approve(
            "rm -f /tmp/hermes-scratch.md", "delete in root path",
            pattern_judge_brief("delete in root path"),
        )

        content = _user_message(mock_call_llm)
        assert "What that pattern matches: " in content
        assert "absolute" in content
        # The key stays visible: it is what the operator reads in the prompt and the logs.
        assert "delete in root path" in content

    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_key_without_a_brief_leaves_the_prompt_unchanged(self, mock_call_llm, mock_cfg):
        mock_call_llm.return_value = _make_response("APPROVE")
        mock_cfg.return_value = {"mode": "smart"}

        _smart_approve("rm -rf /etc/nginx", "recursive delete")

        content = _user_message(mock_call_llm)
        assert content.startswith("The following command was flagged as: recursive delete\n\n")
        assert "What that pattern matches" not in content


class TestVerdictPathResolvesTheBrief(unittest.TestCase):
    @patch("tools.approval_context._fire_approval_hook")
    @patch.object(approval_smart, "_smart_approve", return_value="approve")
    def test_smart_verdict_forwards_the_brief_for_its_key(self, mock_approve, _mock_hook):
        verdict = _smart_verdict(
            "rm -f /tmp/hermes-scratch.md", "delete in root path", "delete in root path",
            ["delete in root path"], "session",
        )

        assert verdict == "approve"
        assert mock_approve.call_args.args[2] == pattern_judge_brief("delete in root path")

    @patch("tools.approval_context._fire_approval_hook")
    @patch.object(approval_smart, "_smart_approve", return_value="approve")
    def test_smart_verdict_sends_no_brief_for_an_unlisted_key(self, mock_approve, _mock_hook):
        _smart_verdict(
            "rm -rf /etc/nginx", "recursive delete", "recursive delete",
            ["recursive delete"], "session",
        )

        assert mock_approve.call_args.args[2] is None


if __name__ == "__main__":
    unittest.main()
