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

from tools.approval_smart import _get_smart_policy, _smart_approve

POLICY_TEXT = "Always ESCALATE commands that modify anything under /etc."


def _make_response(answer: str, finish_reason: str = "stop"):
    """Build a mock LLM response with the given one-word answer."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = answer
    mock_response.choices[0].finish_reason = finish_reason
    return mock_response


def _messages_from(mock_call_llm):
    """Extract the messages list passed to call_llm."""
    call_args = mock_call_llm.call_args
    return call_args.kwargs.get("messages") or call_args[1].get("messages", [])


class TestGetSmartPolicy(unittest.TestCase):
    """Unit tests for the config reader."""

    @patch("tools.approval_context._get_approval_config")
    def test_missing_key_returns_empty(self, mock_cfg):
        mock_cfg.return_value = {"mode": "smart"}
        assert _get_smart_policy() == ""


    @patch("tools.approval_context._get_approval_config")
    def test_policy_text_is_stripped(self, mock_cfg):
        mock_cfg.return_value = {"smart_policy": f"  {POLICY_TEXT}\n"}
        assert _get_smart_policy() == POLICY_TEXT


class TestSmartApprovePolicyInjection(unittest.TestCase):
    """Verify how the operator policy is (and is not) wired into the prompts.

    Follows the mocking pattern of test_smart_approval_injection.py:
    ``call_llm`` is patched at its source module (``agent.auxiliary_client``)
    because _smart_approve imports it lazily inside the function.  The
    config read is isolated by patching ``tools.approval_context._get_approval_config``
    so tests never touch a real config.yaml.
    """

    @patch("tools.approval_context._get_approval_config")
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


    @patch("tools.approval_context._get_approval_config")
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

    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_config_read_failure_does_not_break_approval(self, mock_call_llm, mock_cfg):
        """If the config reader itself blows up, _smart_approve fails safe."""
        mock_call_llm.return_value = _make_response("APPROVE")
        mock_cfg.side_effect = RuntimeError("config unreadable")
        # _smart_approve's outer try/except catches this and escalates
        assert _smart_approve("echo hi", "flagged") == "escalate"


    @patch("agent.auxiliary_client._get_task_timeout")
    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_smart_approve_passes_explicit_timeout(
        self, mock_call_llm, mock_cfg, mock_task_timeout
    ):
        """Regression for #82846: the guardian call must pass an explicit
        timeout instead of relying on the default resolution inside call_llm
        (a defeated internal timeout silently froze agent turns in
        production). The explicit value must equal what
        auxiliary.approval.timeout resolves to."""
        mock_call_llm.return_value = _make_response("APPROVE")
        mock_cfg.return_value = {"mode": "smart"}
        mock_task_timeout.return_value = 42.0

        assert _smart_approve("echo hi", "flagged") == "approve"
        _, kwargs = mock_call_llm.call_args
        assert kwargs.get("timeout") == 42.0


    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_smart_approve_failure_logs_warning_and_escalates(
        self, mock_call_llm, mock_cfg
    ):
        """A failed/blocked guardian call must surface as a WARNING with the
        elapsed time (not a silent DEBUG) — #82846's hang was invisible
        precisely because nothing logged at the failure point."""
        mock_call_llm.side_effect = TimeoutError("stalled provider")
        mock_cfg.return_value = {"mode": "smart"}

        with patch("tools.approval_smart.logger") as mock_logger:
            assert _smart_approve("echo hi", "flagged") == "escalate"

        assert mock_logger.warning.called
        args, _ = mock_logger.warning.call_args
        assert "Smart approvals: LLM call failed" in args[0]
        assert "TimeoutError" in str(args)


    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_truncated_answer_retries_with_larger_budget(self, mock_call_llm, mock_cfg):
        mock_cfg.return_value = {"mode": "smart"}
        mock_call_llm.side_effect = [
            _make_response("", "length"),
            _make_response("APPROVE"),
        ]

        assert _smart_approve("echo hi", "flagged") == "approve"
        assert mock_call_llm.call_count == 2
        first_kwargs = mock_call_llm.call_args_list[0].kwargs
        second_kwargs = mock_call_llm.call_args_list[1].kwargs
        assert first_kwargs["max_tokens"] == 16
        assert second_kwargs["max_tokens"] > first_kwargs["max_tokens"]
        assert first_kwargs["task"] == second_kwargs["task"] == "approval"
        assert first_kwargs["temperature"] == second_kwargs["temperature"] == 0
        assert second_kwargs["messages"] is first_kwargs["messages"]
        assert 0 < second_kwargs["timeout"] <= first_kwargs["timeout"]


    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_truncated_twice_escalates(self, mock_call_llm, mock_cfg):
        mock_cfg.return_value = {"mode": "smart"}
        mock_call_llm.side_effect = [
            _make_response("", "length"),
            _make_response("", "length"),
        ]

        with self.assertLogs("tools.approval", level="WARNING") as logs:
            assert _smart_approve("echo hi", "flagged") == "escalate"
        assert mock_call_llm.call_count == 2
        assert "empty answer" in logs.output[-1]
        assert "finish_reason=length" in logs.output[-1]
        assert "escalating" in logs.output[-1]


    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_completed_unrecognized_answer_is_not_retried(self, mock_call_llm, mock_cfg):
        mock_cfg.return_value = {"mode": "smart"}
        for answer in ("I cannot determine this", "APPROVE."):
            with self.subTest(answer=answer):
                mock_call_llm.reset_mock()
                mock_call_llm.side_effect = [
                    _make_response(answer, "stop"),
                    _make_response("APPROVE", "stop"),
                ]
                with self.assertLogs("tools.approval", level="WARNING") as logs:
                    assert _smart_approve("echo hi", "flagged") == "escalate"
                assert mock_call_llm.call_count == 1
                assert "an unrecognized answer" in logs.output[-1]
                assert "escalating" in logs.output[-1]
                assert answer not in logs.output[-1]


    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_empty_answer_without_length_is_not_retried(self, mock_call_llm, mock_cfg):
        mock_cfg.return_value = {"mode": "smart"}
        for finish_reason in ("stop", None):
            with self.subTest(finish_reason=finish_reason):
                mock_call_llm.reset_mock()
                mock_call_llm.side_effect = [
                    _make_response("", finish_reason),
                    _make_response("APPROVE", "stop"),
                ]
                assert _smart_approve("echo hi", "flagged") == "escalate"
                assert mock_call_llm.call_count == 1


    @patch("agent.auxiliary_client._get_task_timeout", return_value=10.0)
    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_exhausted_budget_skips_the_retry(self, mock_call_llm, mock_cfg, mock_timeout):
        mock_cfg.return_value = {"mode": "smart"}
        mock_call_llm.side_effect = [
            _make_response("", "length"),
            _make_response("APPROVE"),
        ]
        with patch("tools.approval_smart.time.monotonic", side_effect=[0.0, 11.0, 11.0]):
            with self.assertLogs("tools.approval", level="WARNING") as logs:
                assert _smart_approve("echo hi", "flagged") == "escalate"
        assert mock_call_llm.call_count == 1
        assert "budget exhausted" in logs.output[-1]
        assert "escalating" in logs.output[-1]


    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_explicit_escalate_is_not_retried(self, mock_call_llm, mock_cfg):
        mock_cfg.return_value = {"mode": "smart"}
        mock_call_llm.return_value = _make_response("ESCALATE")

        assert _smart_approve("echo hi", "flagged") == "escalate"
        assert mock_call_llm.call_count == 1


    @patch("tools.approval_context._get_approval_config")
    @patch("agent.auxiliary_client.call_llm")
    def test_timeout_is_not_retried(self, mock_call_llm, mock_cfg):
        mock_cfg.return_value = {"mode": "smart"}
        mock_call_llm.side_effect = TimeoutError("stalled provider")

        assert _smart_approve("echo hi", "flagged") == "escalate"
        assert mock_call_llm.call_count == 1


if __name__ == "__main__":
    unittest.main()
