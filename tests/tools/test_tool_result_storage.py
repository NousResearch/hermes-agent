"""Tests for tools/tool_result_storage.py -- 3-layer tool result persistence."""

import pytest
from unittest.mock import MagicMock, patch

from tools.tool_result_storage import (
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    HEREDOC_MARKER,
    MAX_TURN_BUDGET_CHARS,
    PERSISTED_OUTPUT_TAG,
    PERSISTED_OUTPUT_CLOSING_TAG,
    PREVIEW_SIZE_CHARS,
    STORAGE_DIR,
    _build_persisted_message,
    _extract_raw_output,
    _heredoc_marker,
    _write_to_sandbox,
    enforce_turn_budget,
    generate_preview,
    maybe_persist_tool_result,
)


# ── generate_preview ──────────────────────────────────────────────────

class TestGeneratePreview:
    def test_short_content_unchanged(self):
        text = "short result"
        preview, has_more = generate_preview(text)
        assert preview == text
        assert has_more is False

    def test_long_content_truncated(self):
        text = "x" * 5000
        preview, has_more = generate_preview(text, max_chars=2000)
        assert len(preview) <= 2000
        assert has_more is True

    def test_truncates_at_newline_boundary(self):
        # 1500 chars + newline + 600 chars  (past halfway)
        text = "a" * 1500 + "\n" + "b" * 600
        preview, has_more = generate_preview(text, max_chars=2000)
        assert preview == "a" * 1500 + "\n"
        assert has_more is True

    def test_ignores_early_newline(self):
        # Newline at position 100, well before halfway of 2000
        text = "a" * 100 + "\n" + "b" * 3000
        preview, has_more = generate_preview(text, max_chars=2000)
        assert len(preview) == 2000
        assert has_more is True

    def test_empty_content(self):
        preview, has_more = generate_preview("")
        assert preview == ""
        assert has_more is False

    def test_exact_boundary(self):
        text = "x" * PREVIEW_SIZE_CHARS
        preview, has_more = generate_preview(text)
        assert preview == text
        assert has_more is False


# ── _extract_raw_output ────────────────────────────────────────────────

class TestExtractRawOutput:
    def test_extracts_output_from_terminal_json(self):
        import json
        content = json.dumps({"output": "hello world\nline2", "exit_code": 0, "error": None})
        assert _extract_raw_output(content) == "hello world\nline2"

    def test_passes_through_non_json(self):
        assert _extract_raw_output("plain text output") == "plain text output"

    def test_passes_through_json_without_output_key(self):
        import json
        content = json.dumps({"result": "something", "status": "ok"})
        assert _extract_raw_output(content) == content

    def test_extracts_large_output(self):
        import json
        big = "x\n" * 30_000
        content = json.dumps({"output": big, "exit_code": 0, "error": None})
        assert _extract_raw_output(content) == big


# ── _heredoc_marker ───────────────────────────────────────────────────

class TestHeredocMarker:
    def test_default_marker_when_no_collision(self):
        assert _heredoc_marker("normal content") == HEREDOC_MARKER

    def test_uuid_marker_on_collision(self):
        content = f"some text with {HEREDOC_MARKER} embedded"
        marker = _heredoc_marker(content)
        assert marker != HEREDOC_MARKER
        assert marker.startswith("HERMES_PERSIST_")
        assert marker not in content


# ── _write_to_sandbox ─────────────────────────────────────────────────

class TestWriteToSandbox:
    def test_success(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        result = _write_to_sandbox("hello world", "/tmp/hermes-results/abc.txt", env)
        assert result is True
        env.execute.assert_called_once()
        cmd = env.execute.call_args[0][0]
        assert "mkdir -p" in cmd
        assert "hello world" in cmd
        assert HEREDOC_MARKER in cmd

    def test_failure_returns_false(self):
        env = MagicMock()
        env.execute.return_value = {"output": "error", "returncode": 1}
        result = _write_to_sandbox("content", "/tmp/hermes-results/abc.txt", env)
        assert result is False

    def test_heredoc_collision_uses_uuid_marker(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = f"text with {HEREDOC_MARKER} inside"
        _write_to_sandbox(content, "/tmp/hermes-results/abc.txt", env)
        cmd = env.execute.call_args[0][0]
        # The default marker should NOT be used as the delimiter
        lines = cmd.split("\n")
        # The first and last lines contain the actual delimiter
        assert HEREDOC_MARKER not in lines[0].split("<<")[1]

    def test_timeout_passed(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        _write_to_sandbox("content", "/tmp/hermes-results/abc.txt", env)
        assert env.execute.call_args[1]["timeout"] == 30


# ── _build_persisted_message ──────────────────────────────────────────

class TestBuildPersistedMessage:
    def test_structure(self):
        msg = _build_persisted_message(
            preview="first 100 chars...",
            has_more=True,
            original_size=50_000,
            file_path="/tmp/hermes-results/test123.txt",
        )
        assert msg.startswith(PERSISTED_OUTPUT_TAG)
        assert msg.endswith(PERSISTED_OUTPUT_CLOSING_TAG)
        assert "50,000 characters" in msg
        assert "/tmp/hermes-results/test123.txt" in msg
        assert "read_file" in msg
        assert "first 100 chars..." in msg
        assert "..." in msg  # has_more indicator

    def test_no_ellipsis_when_complete(self):
        msg = _build_persisted_message(
            preview="complete content",
            has_more=False,
            original_size=16,
            file_path="/tmp/hermes-results/x.txt",
        )
        # Should not have the trailing "..." indicator before closing tag
        lines = msg.strip().split("\n")
        assert lines[-2] != "..."

    def test_large_size_shows_mb(self):
        msg = _build_persisted_message(
            preview="x",
            has_more=True,
            original_size=2_000_000,
            file_path="/tmp/hermes-results/big.txt",
        )
        assert "MB" in msg


# ── maybe_persist_tool_result ─────────────────────────────────────────

class TestMaybePersistToolResult:
    def test_below_threshold_returns_unchanged(self):
        content = "small result"
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_123",
            env=None,
            threshold=50_000,
        )
        assert result == content

    def test_above_threshold_with_env_persists(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_456",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        assert "tc_456.txt" in result
        assert len(result) < len(content)
        env.execute.assert_called_once()

    def test_persists_raw_output_not_json_wrapper(self):
        """When content is JSON with 'output' key, file should contain raw output."""
        import json
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        raw = "line1\nline2\n" * 5_000
        content = json.dumps({"output": raw, "exit_code": 0, "error": None})
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_json",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        # The heredoc written to sandbox should contain raw text, not JSON
        cmd = env.execute.call_args[0][0]
        assert "line1\nline2\n" in cmd
        assert '"exit_code"' not in cmd

    def test_above_threshold_no_env_truncates_inline(self):
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_789",
            env=None,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG not in result
        assert "Truncated" in result
        assert len(result) < len(content)

    def test_env_write_failure_falls_back_to_truncation(self):
        env = MagicMock()
        env.execute.return_value = {"output": "disk full", "returncode": 1}
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_fail",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG not in result
        assert "Truncated" in result

    def test_env_execute_exception_falls_back(self):
        env = MagicMock()
        env.execute.side_effect = RuntimeError("connection lost")
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_exc",
            env=env,
            threshold=30_000,
        )
        assert "Truncated" in result

    def test_read_file_never_persisted(self):
        """read_file has threshold=inf, should never be persisted."""
        env = MagicMock()
        content = "x" * 200_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="read_file",
            tool_use_id="tc_rf",
            env=env,
            threshold=float("inf"),
        )
        assert result == content
        env.execute.assert_not_called()

    def test_uses_registry_threshold_when_not_provided(self):
        """When threshold=None, looks up from registry."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "x" * 60_000

        mock_registry = MagicMock()
        mock_registry.get_max_result_size.return_value = 30_000

        with patch("tools.registry.registry", mock_registry):
            result = maybe_persist_tool_result(
                content=content,
                tool_name="terminal",
                tool_use_id="tc_reg",
                env=env,
                threshold=None,
            )
        # Should have persisted since 60K > 30K
        assert PERSISTED_OUTPUT_TAG in result or "Truncated" in result

    def test_unicode_content_survives(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "日本語テスト " * 10_000  # ~60K chars of unicode
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_uni",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        # Preview should contain unicode
        assert "日本語テスト" in result

    def test_empty_content_returns_unchanged(self):
        result = maybe_persist_tool_result(
            content="",
            tool_name="terminal",
            tool_use_id="tc_empty",
            env=None,
            threshold=30_000,
        )
        assert result == ""

    def test_whitespace_only_below_threshold(self):
        content = " " * 100
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_ws",
            env=None,
            threshold=30_000,
        )
        assert result == content

    def test_file_path_uses_tool_use_id(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="unique_id_abc",
            env=env,
            threshold=30_000,
        )
        assert "unique_id_abc.txt" in result

    def test_preview_included_in_persisted_output(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        # Create content with a distinctive start
        content = "DISTINCTIVE_START_MARKER" + "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_prev",
            env=env,
            threshold=30_000,
        )
        assert "DISTINCTIVE_START_MARKER" in result

    def test_threshold_zero_forces_persist(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "even short content"
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_zero",
            env=env,
            threshold=0,
        )
        # Any non-empty content with threshold=0 should be persisted
        assert PERSISTED_OUTPUT_TAG in result


# ── enforce_turn_budget ───────────────────────────────────────────────

class TestEnforceTurnBudget:
    def test_under_budget_no_changes(self):
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": "small"},
            {"role": "tool", "tool_call_id": "t2", "content": "also small"},
        ]
        result = enforce_turn_budget(msgs, env=None, budget=200_000)
        assert result[0]["content"] == "small"
        assert result[1]["content"] == "also small"

    def test_over_budget_largest_persisted_first(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": "a" * 80_000},
            {"role": "tool", "tool_call_id": "t2", "content": "b" * 130_000},
        ]
        # Total 210K > 200K budget
        enforce_turn_budget(msgs, env=env, budget=200_000)
        # The larger one (130K) should be persisted first
        assert PERSISTED_OUTPUT_TAG in msgs[1]["content"]

    def test_already_persisted_results_skipped(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "tool_call_id": "t1",
             "content": f"{PERSISTED_OUTPUT_TAG}\nalready persisted\n{PERSISTED_OUTPUT_CLOSING_TAG}"},
            {"role": "tool", "tool_call_id": "t2", "content": "x" * 250_000},
        ]
        enforce_turn_budget(msgs, env=env, budget=200_000)
        # t1 should be untouched (already persisted)
        assert msgs[0]["content"].startswith(PERSISTED_OUTPUT_TAG)
        # t2 should be persisted
        assert PERSISTED_OUTPUT_TAG in msgs[1]["content"]

    def test_medium_result_regression(self):
        """6 results of 42K chars each (252K total) — each under 50K default
        threshold but aggregate exceeds 200K budget. L3 should persist."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 42_000}
            for i in range(6)
        ]
        enforce_turn_budget(msgs, env=env, budget=200_000)
        # At least some results should be persisted to get under 200K
        persisted_count = sum(
            1 for m in msgs if PERSISTED_OUTPUT_TAG in m["content"]
        )
        assert persisted_count >= 2  # Need to shed at least ~52K

    def test_no_env_falls_back_to_truncation(self):
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": "x" * 250_000},
        ]
        enforce_turn_budget(msgs, env=None, budget=200_000)
        # Should be truncated (no sandbox available)
        assert "Truncated" in msgs[0]["content"] or PERSISTED_OUTPUT_TAG in msgs[0]["content"]

    def test_returns_same_list(self):
        msgs = [{"role": "tool", "tool_call_id": "t1", "content": "ok"}]
        result = enforce_turn_budget(msgs, env=None, budget=200_000)
        assert result is msgs

    def test_empty_messages(self):
        result = enforce_turn_budget([], env=None, budget=200_000)
        assert result == []


# ── Per-tool threshold integration ────────────────────────────────────

class TestPerToolThresholds:
    """Verify registry wiring for per-tool thresholds."""

    def test_registry_has_get_max_result_size(self):
        from tools.registry import registry
        assert hasattr(registry, "get_max_result_size")

    def test_default_threshold(self):
        from tools.registry import registry
        # Unknown tool should return the default
        val = registry.get_max_result_size("nonexistent_tool_xyz")
        assert val == DEFAULT_MAX_RESULT_SIZE_CHARS

    def test_terminal_threshold(self):
        from tools.registry import registry
        # Trigger import of terminal_tool to register the tool
        try:
            import tools.terminal_tool  # noqa: F401
            val = registry.get_max_result_size("terminal")
            assert val == 30_000
        except ImportError:
            pytest.skip("terminal_tool not importable in test env")

    def test_read_file_never_persisted(self):
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("read_file")
            assert val == float("inf")
        except ImportError:
            pytest.skip("file_tools not importable in test env")

    def test_search_files_threshold(self):
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("search_files")
            assert val == 20_000
        except ImportError:
            pytest.skip("file_tools not importable in test env")
