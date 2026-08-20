"""Tests for _looks_like_error_output in tools/delegate_tool.py.

This function classifies tool-result blobs as errors or successes so the
parent agent's tool_trace accurately reflects sub-agent outcomes.  A
misclassification causes the orchestrator to report false success counts
back to the user.

Bug (fixed): json.loads(content) raised JSONDecodeError: Extra data when
Hermes' tool-loop-warning injector appended '\\n\\n[Tool loop warning: ...]'
after the closing brace of an error JSON blob.  The exception was swallowed;
the fallback first-line check looks for plain 'error:'/'failed:' prefixes, not
'{"error": "..."}' JSON, so it returned False — marking the error as success.

Impact: from the 3rd consecutive failure onward (when same_tool_failure_warning
fires), every sub-agent error was relabelled status:ok in the parent
tool_trace.  Reproduced in production: 16 create_calendar_event calls all
failed (circuit breaker tripped after the first 3), but the orchestrator
reported "14 of 16 events successfully created".

Fix: json.JSONDecoder().raw_decode(head) — parses the first JSON value and
stops, ignoring trailing non-JSON text such as the loop-warning appendix.
"""

import pytest

from tools.delegate_tool import _looks_like_error_output


class TestLooksLikeErrorOutputBaseline:
    """Baseline coverage: core cases that must always hold."""

    def test_pure_error_json(self):
        """Canonical error blob — the most common shape."""
        content = '{"error": "MCP call failed: Tool execution failed: not found"}'
        assert _looks_like_error_output(content) is True

    def test_pure_success_json(self):
        """Normal success response must not be flagged as error."""
        content = '{"status": "ok", "result": {"created": 1}}'
        assert _looks_like_error_output(content) is False

    def test_error_colon_prefix(self):
        """Plain-text stderr with 'error:' prefix."""
        assert _looks_like_error_output("error: connection refused") is True

    def test_failed_colon_prefix(self):
        """Plain-text stderr with 'failed:' prefix."""
        assert _looks_like_error_output("failed: timeout after 30s") is True

    def test_plain_text_no_error(self):
        """Arbitrary plain-text output that is not an error."""
        assert _looks_like_error_output("2 calendar events created") is False

    def test_empty_string(self):
        """Empty content should not raise and should return False."""
        assert _looks_like_error_output("") is False

    def test_status_failed(self):
        """JSON with status:'failed' should be classified as error."""
        content = '{"status": "failed", "code": 503}'
        assert _looks_like_error_output(content) is True

    def test_status_timeout(self):
        """JSON with status:'timeout' should be classified as error."""
        content = '{"status": "timeout", "elapsed_ms": 30000}'
        assert _looks_like_error_output(content) is True


class TestLooksLikeErrorOutputLoopWarningRegression:
    """Regression tests for the json.loads / loop-warning-appendix bug.

    Before the fix, these inputs returned False (incorrectly classified as
    success).  Each test name references the production scenario it covers.
    """

    def test_error_json_with_loop_warning_appended(self):
        """Regression: error JSON followed by [Tool loop warning: ...] appendix
        must still be classified as error.

        Was: json.loads() threw 'Extra data', exception swallowed, fallback
        first-line check missed {"error": "..."} (looks for 'error:' prefix,
        not JSON), returned False — causing sub-agent tool_trace to report
        failures as 'ok' starting at call #3.
        """
        content = (
            '{"error": "MCP call failed: MCP error -32603: Tool execution failed: '
            'Calendar not found: 4c646201-472c-4377-b4c8-10f4455c6ecf"}'
            "\n\n[Tool loop warning: same_tool_failure_warning; count=3; "
            "same arguments repeated]"
        )
        assert _looks_like_error_output(content) is True

    def test_circuit_breaker_error_with_loop_warning(self):
        """Same root cause, different error shape (circuit breaker).

        Reproduces the production #20 hallucination: 16 create_calendar_event
        calls all failed after the circuit breaker tripped, but the
        orchestrator said '14 of 16 events were successfully created'.
        """
        content = (
            '{"error": "MCP server \'fastmail\' is unreachable after '
            '3 consecutive failures"}'
            "\n\n[Tool loop warning: same_tool_failure_warning; count=5]"
        )
        assert _looks_like_error_output(content) is True

    def test_status_failed_with_loop_warning_appended(self):
        """status:'failed' blob plus loop-warning appendix — still an error."""
        content = (
            '{"status": "failed", "message": "rate limit exceeded"}'
            "\n\n[Tool loop warning: same_tool_failure_warning; count=4]"
        )
        assert _looks_like_error_output(content) is True

    def test_success_json_with_loop_warning_not_flagged(self):
        """A success blob with loop-warning appendix must NOT be an error.

        Guards against overcorrection: the fix should only affect the JSON
        parsing path, not flip successes to errors.
        """
        content = (
            '{"status": "ok", "events_created": 1}'
            "\n\n[Tool loop warning: same_tool_failure_warning; count=3]"
        )
        assert _looks_like_error_output(content) is False
