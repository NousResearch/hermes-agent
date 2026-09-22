"""Contract tests for the configurable execute_code stdout cap.

The cap is the same user-facing ``tool_output.max_bytes`` knob terminal output honors;
when unset, behavior must be the historical 50 KB default (MAX_STDOUT_BYTES). Regression
coverage for the local patch documented in ~/hermes-projects/hermes-local-patches/."""

from tools import code_execution_tool as cet
from tools import tool_output_limits


def test_configured_cap_flows_through_truncation(monkeypatch):
    monkeypatch.setattr(tool_output_limits, "get_max_bytes", lambda: 1_000)
    text, meta = cet._truncate_stdout_text("x" * 5_000)
    assert cet._resolve_stdout_cap() == 1_000
    assert meta["stdout_truncated"] is True
    assert meta["stdout_bytes_captured"] == 1_000
    assert meta["stdout_bytes_omitted"] == 4_000
    assert meta["stdout_bytes_total"] == 5_000


def test_unset_config_keeps_50kb_default(monkeypatch):
    tool_output_limits._reset_tool_output_limits_cache()
    monkeypatch.setattr(tool_output_limits, "get_max_bytes",
                        lambda: tool_output_limits.DEFAULT_MAX_BYTES)
    assert tool_output_limits.DEFAULT_MAX_BYTES == cet.MAX_STDOUT_BYTES  # default unchanged
    text, meta = cet._truncate_stdout_text("x" * 60_000)
    assert meta["stdout_bytes_captured"] == 50_000
    assert meta["stdout_bytes_omitted"] == 10_000
