"""Tests for tools/tool_result_storage.py -- 3-layer tool result persistence."""

import pytest
from unittest.mock import MagicMock, patch

from tools.budget_config import (
    DEFAULT_RESULT_SIZE_CHARS,
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
)
from tools.tool_result_storage import (
    HEREDOC_MARKER,
    PERSISTED_OUTPUT_TAG,
    PERSISTED_OUTPUT_CLOSING_TAG,
    STORAGE_DIR,
    _build_persisted_message,
    _heredoc_marker,
    _resolve_storage_dir,
    _safe_result_filename,
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
        text = "x" * DEFAULT_PREVIEW_SIZE_CHARS
        preview, has_more = generate_preview(text)
        assert preview == text
        assert has_more is False


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
        # Content travels through stdin, NOT inside the command string —
        # otherwise large content would hit Linux's 128 KB MAX_ARG_STRLEN
        # ceiling on `bash -c <cmd>` (#22906).
        assert "hello world" not in cmd
        assert env.execute.call_args[1]["stdin_data"] == "hello world"

    def test_failure_returns_false(self):
        env = MagicMock()
        env.execute.return_value = {"output": "error", "returncode": 1}
        result = _write_to_sandbox("content", "/tmp/hermes-results/abc.txt", env)
        assert result is False

    def test_large_content_via_stdin(self):
        """Regression: 200 KB content exceeds Linux MAX_ARG_STRLEN (128 KB).
        It must travel via stdin, never inside the command string."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        big = "x" * 200_000
        _write_to_sandbox(big, "/tmp/hermes-results/big.txt", env)
        cmd = env.execute.call_args[0][0]
        assert len(cmd) < 1_000  # cmd is just `mkdir -p X && cat > Y`
        assert env.execute.call_args[1]["stdin_data"] == big

    def test_timeout_passed(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        _write_to_sandbox("content", "/tmp/hermes-results/abc.txt", env)
        assert env.execute.call_args[1]["timeout"] == 30

    def test_uses_parent_dir_of_remote_path(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        remote_path = "/data/data/com.termux/files/usr/tmp/hermes-results/abc.txt"
        _write_to_sandbox("content", remote_path, env)
        cmd = env.execute.call_args[0][0]
        assert "mkdir -p /data/data/com.termux/files/usr/tmp/hermes-results" in cmd

    def test_path_with_spaces_is_quoted(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        remote_path = "/tmp/hermes results/abc file.txt"
        _write_to_sandbox("content", remote_path, env)
        cmd = env.execute.call_args[0][0]
        assert "'/tmp/hermes results'" in cmd
        assert "'/tmp/hermes results/abc file.txt'" in cmd

    def test_shell_metacharacters_neutralized(self):
        """Paths with shell metacharacters must be quoted to prevent injection."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        malicious_path = "/tmp/hermes-results/$(whoami).txt"
        _write_to_sandbox("content", malicious_path, env)
        cmd = env.execute.call_args[0][0]
        # The $() must not appear unquoted — shlex.quote wraps it
        assert "'/tmp/hermes-results/$(whoami).txt'" in cmd

    def test_semicolon_injection_neutralized(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        malicious_path = "/tmp/x; rm -rf /; echo .txt"
        _write_to_sandbox("content", malicious_path, env)
        cmd = env.execute.call_args[0][0]
        # The semicolons must be inside quotes, not acting as command separators
        assert "'/tmp/x; rm -rf /; echo .txt'" in cmd


class TestResolveStorageDir:
    def test_defaults_to_storage_dir_without_env(self):
        assert _resolve_storage_dir(None) == STORAGE_DIR

    def test_uses_env_temp_dir_when_available(self):
        env = MagicMock()
        env.get_temp_dir.return_value = "/data/data/com.termux/files/usr/tmp"
        assert _resolve_storage_dir(env) == "/data/data/com.termux/files/usr/tmp/hermes-results"


class TestSafeResultFilename:
    def test_preserves_normal_tool_call_id(self):
        assert _safe_result_filename("tc_456") == "tc_456.txt"

    def test_replaces_path_and_shell_metacharacters(self):
        filename = _safe_result_filename("../outside/$(whoami);x")
        assert filename.startswith("outside_whoami_x_")
        assert filename.endswith(".txt")
        assert "/" not in filename
        assert "$" not in filename
        assert ";" not in filename


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

    def test_persists_full_content_as_is(self):
        """Content is persisted verbatim — no JSON extraction."""
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
        # Content is delivered through stdin (no longer embedded in the
        # command string — see test_large_content_via_stdin for why).
        assert env.execute.call_args[1]["stdin_data"] == content

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

    def test_tool_use_id_cannot_escape_storage_dir(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        env.get_temp_dir.return_value = ""
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="../outside/$(whoami);x",
            env=env,
            threshold=30_000,
        )
        cmd = env.execute.call_args[0][0]
        target = cmd.split("cat > ", 1)[1].split(" <<", 1)[0]

        assert "Full output saved to: /tmp/hermes-results/outside_whoami_x_" in result
        assert "/tmp/hermes-results/../" not in result
        assert target.startswith("/tmp/hermes-results/outside_whoami_x_")
        assert "/../" not in target
        assert "$(whoami)" not in target
        assert ";" not in target

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

    def test_env_temp_dir_changes_persisted_path(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        env.get_temp_dir.return_value = "/data/data/com.termux/files/usr/tmp"
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_termux",
            env=env,
            threshold=30_000,
        )
        assert "/data/data/com.termux/files/usr/tmp/hermes-results/tc_termux.txt" in result
        cmd = env.execute.call_args[0][0]
        assert "mkdir -p /data/data/com.termux/files/usr/tmp/hermes-results" in cmd

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
        result = enforce_turn_budget(msgs, env=None, config=BudgetConfig(turn_budget=200_000))
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
        enforce_turn_budget(msgs, env=env, config=BudgetConfig(turn_budget=200_000))
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
        enforce_turn_budget(msgs, env=env, config=BudgetConfig(turn_budget=200_000))
        # t1 should be untouched (already persisted)
        assert msgs[0]["content"].startswith(PERSISTED_OUTPUT_TAG)
        # t2 should be persisted
        assert PERSISTED_OUTPUT_TAG in msgs[1]["content"]

    def test_medium_result_regression(self):
        """6 results of 42K chars each (252K total) — each under 100K default
        threshold but aggregate exceeds 200K budget. L3 should persist."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 42_000}
            for i in range(6)
        ]
        enforce_turn_budget(msgs, env=env, config=BudgetConfig(turn_budget=200_000))
        # At least some results should be persisted to get under 200K
        persisted_count = sum(
            1 for m in msgs if PERSISTED_OUTPUT_TAG in m["content"]
        )
        assert persisted_count >= 2  # Need to shed at least ~52K

    def test_no_env_falls_back_to_truncation(self):
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": "x" * 250_000},
        ]
        enforce_turn_budget(msgs, env=None, config=BudgetConfig(turn_budget=200_000))
        # Should be truncated (no sandbox available)
        assert "Truncated" in msgs[0]["content"] or PERSISTED_OUTPUT_TAG in msgs[0]["content"]

    def test_returns_same_list(self):
        msgs = [{"role": "tool", "tool_call_id": "t1", "content": "ok"}]
        result = enforce_turn_budget(msgs, env=None, config=BudgetConfig(turn_budget=200_000))
        assert result is msgs

    def test_empty_messages(self):
        result = enforce_turn_budget([], env=None, config=BudgetConfig(turn_budget=200_000))
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
        assert val == DEFAULT_RESULT_SIZE_CHARS

    def test_terminal_threshold(self):
        from tools.registry import registry
        # Trigger import of terminal_tool to register the tool
        try:
            import tools.terminal_tool  # noqa: F401
            val = registry.get_max_result_size("terminal")
            assert val == 100_000
        except ImportError:
            pytest.skip("terminal_tool not importable in test env")

    def test_read_file_result_size_cap(self):
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("read_file")
            assert val == 100_000
        except ImportError:
            pytest.skip("file_tools not importable in test env")

    def test_read_file_registry_cap_is_100k(self):
        """Regression test: read_file must have a 100_000 char registry cap (Layer 2 safety net)."""
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("read_file")
            assert val == 100_000, (
                f"read_file registry cap must be 100_000, got {val!r}. "
                "float('inf') is not allowed — it disables the Layer 2 result-size guard."
            )
        except ImportError:
            pytest.skip("file_tools not importable in test env")

    def test_search_files_threshold(self):
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("search_files")
            assert val == 100_000
        except ImportError:
            pytest.skip("file_tools not importable in test env")


# ── _resolve_storage_dir edge cases ───────────────────────────────────

class TestResolveStorageDirEdgeCases:
    def test_env_get_temp_dir_raises(self):
        """When env.get_temp_dir() raises, falls back to STORAGE_DIR."""
        env = MagicMock()
        env.get_temp_dir.side_effect = OSError("permission denied")
        assert _resolve_storage_dir(env) == STORAGE_DIR

    def test_env_get_temp_dir_returns_empty(self):
        """When env.get_temp_dir() returns empty string, falls back to STORAGE_DIR."""
        env = MagicMock()
        env.get_temp_dir.return_value = ""
        assert _resolve_storage_dir(env) == STORAGE_DIR

    def test_env_get_temp_dir_returns_none(self):
        """When env.get_temp_dir() returns None, falls back to STORAGE_DIR."""
        env = MagicMock()
        env.get_temp_dir.return_value = None
        assert _resolve_storage_dir(env) == STORAGE_DIR

    def test_env_has_no_get_temp_dir(self):
        """When env has no get_temp_dir attribute, falls back to STORAGE_DIR."""
        env = MagicMock()
        del env.get_temp_dir
        assert _resolve_storage_dir(env) == STORAGE_DIR

    def test_env_temp_dir_trailing_slash_stripped(self):
        """Trailing slashes are stripped from the temp dir."""
        env = MagicMock()
        env.get_temp_dir.return_value = "/var/tmp///"
        result = _resolve_storage_dir(env)
        assert result == "/var/tmp/hermes-results"

    def test_env_temp_dir_root_only(self):
        """When temp dir is just '/', the rstrip-or-slash fallback yields
        a double-slash path — cosmetic, not a bug."""
        env = MagicMock()
        env.get_temp_dir.return_value = "/"
        result = _resolve_storage_dir(env)
        assert result.endswith("/hermes-results")


# ── enforce_turn_budget edge cases ─────────────────────────────────────

class TestEnforceTurnBudgetEdgeCases:
    def test_missing_tool_call_id_uses_budget_prefix(self):
        """Messages without tool_call_id get a budget_ prefix."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "content": "x" * 250_000},
        ]
        enforce_turn_budget(msgs, env=env, config=BudgetConfig(turn_budget=200_000))
        # Should be persisted with a budget_ prefix filename
        assert PERSISTED_OUTPUT_TAG in msgs[0]["content"]
        assert "budget_0" in msgs[0]["content"]

    def test_no_content_key_uses_empty_string(self):
        """Missing content remains safe when another result exceeds the budget."""
        persisted = (
            f"{PERSISTED_OUTPUT_TAG}\n{'x' * 200}\n{PERSISTED_OUTPUT_CLOSING_TAG}"
        )
        msgs = [
            {"role": "tool", "tool_call_id": "t1"},
            {"role": "tool", "tool_call_id": "t2", "content": persisted},
        ]
        result = enforce_turn_budget(
            msgs, env=None, config=BudgetConfig(turn_budget=100)
        )
        assert result is msgs
        assert "content" not in result[0]
        assert result[1]["content"] == persisted

    def test_all_already_persisted(self):
        """Over-budget persisted results are not persisted a second time."""
        persisted = (
            f"{PERSISTED_OUTPUT_TAG}\n{'x' * 200}\n{PERSISTED_OUTPUT_CLOSING_TAG}"
        )
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": persisted},
        ]
        result = enforce_turn_budget(msgs, env=None, config=BudgetConfig(turn_budget=100))
        assert result is msgs
        assert result[0]["content"] == persisted

    def test_budget_enforcement_logs_persisted(self):
        """Verify the logging path for budget enforcement is exercised."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": "x" * 250_000},
        ]
        with patch("tools.tool_result_storage.logger") as mock_logger:
            enforce_turn_budget(msgs, env=env, config=BudgetConfig(turn_budget=200_000))
            mock_logger.info.assert_any_call(
                "Budget enforcement: persisted tool result %s (%d chars)",
                "t1",
                250_000,
            )
