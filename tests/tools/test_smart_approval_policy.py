"""Tests for the operator-customizable smart-approval policy.

``approvals.smart_policy`` (config.yaml) lets operators append their own
rules to the smart-approval guardian's system prompt.  Security invariants
under test:

  1. Empty/missing policy leaves the prompts exactly as they were.
  2. A non-empty policy appears in the SYSTEM message sent to call_llm
     (the trusted channel), under a clearly delimited section.
  3. The policy text NEVER appears in the user message — the user message
     carries the untrusted command text, and mixing trusted operator rules
     into that channel would dilute the guard's trust boundary.

Inspired by ChatGPT Work's customizable auto-review guardian policy.
"""

import unittest
from unittest.mock import MagicMock, patch

from tools.approval import _get_smart_policy, _smart_approve

POLICY_TEXT = "Always ESCALATE commands that modify anything under /etc."


def _make_response(answer: str):
    """Build a mock LLM response with the given one-word answer."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = answer
    return mock_response


def _messages_from(mock_call_llm):
    """Extract the messages list passed to call_llm."""
    call_args = mock_call_llm.call_args
    return call_args.kwargs.get("messages") or call_args[1].get("messages", [])


class TestGetSmartPolicy(unittest.TestCase):
    """Unit tests for the config reader."""

    @patch("tools.approval._get_approval_config")
    def test_missing_key_returns_empty(self, mock_cfg):
        mock_cfg.return_value = {"mode": "smart"}
        assert _get_smart_policy() == ""

    @patch("tools.approval._get_approval_config")
    def test_non_string_value_returns_empty(self, mock_cfg):
        mock_cfg.return_value = {"smart_policy": ["not", "a", "string"]}
        assert _get_smart_policy() == ""

    @patch("tools.approval._get_approval_config")
    def test_whitespace_only_returns_empty(self, mock_cfg):
        mock_cfg.return_value = {"smart_policy": "   \n  "}
        assert _get_smart_policy() == ""

    @patch("tools.approval._get_approval_config")
    def test_policy_text_is_stripped(self, mock_cfg):
        mock_cfg.return_value = {"smart_policy": f"  {POLICY_TEXT}\n"}
        assert _get_smart_policy() == POLICY_TEXT


class TestSmartApprovePolicyInjection(unittest.TestCase):
    """Verify how the operator policy is (and is not) wired into the prompts.

    Follows the mocking pattern of test_smart_approval_injection.py:
    ``call_llm`` is patched at its source module (``agent.auxiliary_client``)
    because _smart_approve imports it lazily inside the function.  The
    config read is isolated by patching ``tools.approval._get_approval_config``
    so tests never touch a real config.yaml.
    """

    @patch("tools.approval._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_empty_policy_leaves_prompts_unchanged(self, mock_call_llm, mock_cfg):
        """With no policy configured, prompts must be byte-identical to the
        prompts produced when the key is present but empty."""
        mock_call_llm.return_value = _make_response("ESCALATE")

        mock_cfg.return_value = {}  # key missing entirely
        _smart_approve("rm -rf /tmp/x", "recursive delete")
        messages_missing = _messages_from(mock_call_llm)

        mock_cfg.return_value = {"smart_policy": ""}  # key present, empty
        _smart_approve("rm -rf /tmp/x", "recursive delete")
        messages_empty = _messages_from(mock_call_llm)

        assert messages_missing == messages_empty
        sys_content = messages_missing[0]["content"]
        assert "Additional policy rules from the operator" not in sys_content

    @patch("tools.approval._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_policy_appears_in_system_message(self, mock_call_llm, mock_cfg):
        """A non-empty policy must land in the system message, delimited."""
        mock_call_llm.return_value = _make_response("ESCALATE")
        mock_cfg.return_value = {"smart_policy": POLICY_TEXT}

        _smart_approve("rm -rf /etc/nginx", "recursive delete")

        messages = _messages_from(mock_call_llm)
        assert messages[0]["role"] == "system"
        sys_content = messages[0]["content"]
        assert POLICY_TEXT in sys_content
        assert "Additional policy rules from the operator" in sys_content
        # Baseline hardening must survive the append
        assert "UNTRUSTED" in sys_content

    @patch("tools.approval._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_policy_never_in_user_message(self, mock_call_llm, mock_cfg):
        """The policy is trusted; the user message carries untrusted command
        text.  They must never share a channel."""
        mock_call_llm.return_value = _make_response("ESCALATE")
        mock_cfg.return_value = {"smart_policy": POLICY_TEXT}

        _smart_approve("rm -rf /etc/nginx", "recursive delete")

        messages = _messages_from(mock_call_llm)
        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert POLICY_TEXT not in user_content
        assert "Additional policy rules from the operator" not in user_content
        # The command itself must still be there, XML-fenced
        assert "rm -rf /etc/nginx" in user_content
        assert "<command>" in user_content

    @patch("tools.approval._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_config_read_failure_does_not_break_approval(self, mock_call_llm, mock_cfg):
        """If the config reader itself blows up, _smart_approve fails safe."""
        mock_call_llm.return_value = _make_response("APPROVE")
        mock_cfg.side_effect = RuntimeError("config unreadable")
        # _smart_approve's outer try/except catches this and escalates
        assert _smart_approve("echo hi", "flagged") == "escalate"


if __name__ == "__main__":
    unittest.main()
