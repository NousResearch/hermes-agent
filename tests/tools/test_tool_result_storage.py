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
    cleanup_spillover_cache,
    enforce_turn_budget,
    generate_preview,
    get_spillover_dir,
    maybe_persist_tool_result,
)


# ── generate_preview ──────────────────────────────────────────────────

class TestGeneratePreview:
    def test_short_content_unchanged(self):
        text = "short result"
        preview, has_more = generate_preview(text)
        assert preview == text
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

    def test_persists_full_content_as_is(self):
        """Content is persisted verbatim — no JSON extraction."""
        import json
        env = MagicMock()
        # Readability probe fails -> falls back to the in-sandbox write.
        env.execute.side_effect = [
            {"output": "", "returncode": 1},
            {"output": "", "returncode": 0},
        ]
        env.get_temp_dir.return_value = ""
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


    def test_tool_use_id_cannot_escape_storage_dir(self):
        env = MagicMock()
        # Readability probe fails -> in-sandbox write is the reference path.
        env.execute.side_effect = [
            {"output": "", "returncode": 1},
            {"output": "", "returncode": 0},
        ]
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


    def test_empty_messages(self):
        result = enforce_turn_budget([], env=None, config=BudgetConfig(turn_budget=200_000))
        assert result == []


# ── Per-tool threshold integration ────────────────────────────────────

class TestPerToolThresholds:
    """Verify registry wiring for per-tool thresholds."""

    def test_registry_has_get_max_result_size(self):
        from tools.registry import registry
        assert hasattr(registry, "get_max_result_size")


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


# ── Host-side spillover ($HERMES_HOME/cache/spillover) ────────────────

class TestSpillover:
    @pytest.fixture(autouse=True)
    def _isolated_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        # Reset the once-per-process prune flag so each test is independent.
        import tools.tool_result_storage as trs
        monkeypatch.setattr(trs, "_spillover_pruned_once", False)
        yield

    def test_env_none_persists_to_spillover(self):
        """No active sandbox env (MCP-only / cron session) must persist
        host-side instead of inline-truncating — the guglielmo bundle bug."""
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="tool_call",
            tool_use_id="tc_mcp_1",
            env=None,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        assert "could not be saved" not in result
        spill_file = get_spillover_dir() / "tc_mcp_1.txt"
        assert spill_file.exists()
        assert spill_file.read_text(encoding="utf-8") == content
        assert str(spill_file) in result

    def test_local_env_persists_to_spillover_not_sandbox(self):
        """LocalEnvironment routes host-side: no env.execute() shell-out."""
        from tools.environments.local import LocalEnvironment

        env = MagicMock(spec=LocalEnvironment)
        content = "y" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_local_1",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        assert (get_spillover_dir() / "tc_local_1.txt").exists()
        env.execute.assert_not_called()

    def test_remote_env_probe_success_references_mounted_path(self):
        """Remote env: host-side write is canonical; when the sandbox can read
        the mounted/synced spillover path, the reference uses it and no
        in-sandbox copy is written."""
        env = MagicMock()  # not a LocalEnvironment
        env.execute.return_value = {"output": "", "returncode": 0}  # probe OK
        content = "z" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_remote_1",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        # Canonical host copy always exists now.
        assert (get_spillover_dir() / "tc_remote_1.txt").exists()
        # Only the readability probe ran — no cat-into-sandbox call.
        assert env.execute.call_count == 1
        assert "test -r" in env.execute.call_args[0][0]

    def test_remote_env_probe_failure_falls_back_to_sandbox_write(self):
        """Persistent containers without the spillover mount still get a
        readable in-sandbox copy."""
        env = MagicMock()
        env.execute.side_effect = [
            {"output": "", "returncode": 1},  # probe: not readable
            {"output": "", "returncode": 0},  # cat > sandbox path
        ]
        env.get_temp_dir.return_value = "/tmp"
        content = "z" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_remote_2",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        assert "/tmp/hermes-results/tc_remote_2.txt" in result
        assert env.execute.call_count == 2
        # Host canonical copy exists regardless.
        assert (get_spillover_dir() / "tc_remote_2.txt").exists()

    def test_spillover_write_failure_falls_back_to_inline(self, monkeypatch):
        import tools.tool_result_storage as trs
        monkeypatch.setattr(trs, "_write_to_spillover", lambda *a, **k: None)
        content = "w" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="tool_call",
            tool_use_id="tc_fail_1",
            env=None,
            threshold=30_000,
        )
        assert "could not be saved" in result
        assert PERSISTED_OUTPUT_TAG not in result

    def test_cleanup_spillover_cache_removes_old_keeps_new(self):
        import os
        import time as _time

        spill_dir = get_spillover_dir()
        spill_dir.mkdir(parents=True, exist_ok=True)
        old = spill_dir / "old.txt"
        new = spill_dir / "new.txt"
        old.write_text("old")
        new.write_text("new")
        stale = _time.time() - (48 * 3600)
        os.utime(old, (stale, stale))

        removed = cleanup_spillover_cache(max_age_hours=24)

        assert removed == 1
        assert not old.exists()
        assert new.exists()

    def test_cleanup_missing_dir_returns_zero(self):
        assert cleanup_spillover_cache() == 0

    def test_first_spill_prunes_expired_files(self):
        """The once-per-process prune fires on the first host-side spill."""
        import os
        import time as _time

        spill_dir = get_spillover_dir()
        spill_dir.mkdir(parents=True, exist_ok=True)
        old = spill_dir / "ancient.txt"
        old.write_text("ancient")
        stale = _time.time() - (48 * 3600)
        os.utime(old, (stale, stale))

        maybe_persist_tool_result(
            content="v" * 60_000,
            tool_name="tool_call",
            tool_use_id="tc_prune_1",
            env=None,
            threshold=30_000,
        )

        assert not old.exists()
        assert (spill_dir / "tc_prune_1.txt").exists()


# ── recovery hint in the persisted preview ────────────────────────────

class TestRecoveryHint:
    def test_preview_teaches_recovery_not_refetch(self):
        msg = _build_persisted_message(
            preview="preview text",
            has_more=True,
            original_size=60_000,
            file_path="/tmp/hermes-results/r.txt",
        )
        assert "Recovery:" in msg
        assert "execute_code" in msg
        assert "re-request" in msg
        # Structure preserved: tag, size, path, read_file guidance all intact.
        assert msg.startswith(PERSISTED_OUTPUT_TAG)
        assert msg.endswith(PERSISTED_OUTPUT_CLOSING_TAG)
        assert "read_file" in msg
