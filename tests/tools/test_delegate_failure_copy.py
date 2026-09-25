"""Subagent failure lines gloss the classified `failure_reason` instead of echoing the child's raw
exception, and always name the next step (/agents). Contract tests, not snapshots.
"""

from tools.delegate_tool_progress import describe_subagent_failure, format_subagent_failure_line


def test_classified_reason_replaces_raw_exception_text():
    line = format_subagent_failure_line(
        "research pricing", "failed", error="Error code: 429 - {'error': 'rate limited'}",
        duration_seconds=12, failure_reason="rate_limit")
    assert "rate-limited" in line
    assert "Error code" not in line and "429" not in line
    assert "/agents" in line


def test_unclassified_failure_keeps_the_cleaned_error_as_cause():
    assert describe_subagent_failure(None, "Traceback...\nKeyError: 'x'") == "KeyError: 'x'"
    line = format_subagent_failure_line("g", "failed", error="KeyError: 'x'")
    assert "KeyError: 'x'" in line and "/agents" in line


def test_timeout_line_names_outcome_and_the_timeout_setting_without_duplicating_duration():
    raw = "Subagent timed out after 600s with 4 API call(s) completed — likely stuck on a slow API call."
    line = format_subagent_failure_line("scan the repo", "timeout", error=raw, duration_seconds=600)
    assert "delegation.child_timeout_seconds" in line
    assert line.count("600") == 0 and "10 min" in line
    assert "API call" not in line
