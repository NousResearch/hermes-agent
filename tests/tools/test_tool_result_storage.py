"""Tests for tools/tool_result_storage.py -- 3-layer tool result persistence."""

import os
import subprocess
import sys

import pytest
from unittest.mock import MagicMock, patch

from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
)
from tools.tool_result_storage import (
    PERSISTED_OUTPUT_TAG,
    PERSISTED_OUTPUT_CLOSING_TAG,
    STORAGE_DIR,
    _build_persisted_message,
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

# ── _write_to_sandbox ─────────────────────────────────────────────────

class TestWriteToSandbox:
    def test_success(self, tmp_path):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        remote = str(tmp_path / "shared" / "hermes-results" / "abc.txt")
        if sys.platform != "win32":
            # Run the real commands so the owner-only modes are observed, not
            # pattern-matched.
            def _bash(cmd, timeout=None, stdin_data=None, **_kw):
                r = subprocess.run(["bash", "-c", cmd], input=stdin_data or "",
                                   capture_output=True, text=True, timeout=timeout)
                return {"output": r.stdout + r.stderr, "returncode": r.returncode}
            env.execute.side_effect = _bash
        result = _write_to_sandbox("hello world", remote, env)
        assert result is True
        cmd = env.execute.call_args_list[0][0][0]
        if sys.platform != "win32":
            # The storage dir sits under shared temp and holds tool output that
            # can contain secrets: dir and archive are owner-only.
            assert os.stat(os.path.dirname(remote)).st_mode & 0o777 == 0o700
            assert os.stat(remote).st_mode & 0o777 == 0o600
            with open(remote, encoding="utf-8") as fh:
                assert fh.read() == "hello world"
        # Content travels through stdin, NOT inside the command string —
        # otherwise large content would hit Linux's 128 KB MAX_ARG_STRLEN
        # ceiling on `bash -c <cmd>` (#22906).
        assert "hello world" not in cmd
        assert env.execute.call_args_list[0][1]["stdin_data"] == "hello world"

    def test_large_content_via_stdin(self):
        """Regression: 200 KB content exceeds Linux MAX_ARG_STRLEN (128 KB).
        It must travel via stdin, never inside the command string."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        big = "x" * 200_000
        _write_to_sandbox(big, "/tmp/hermes-results/big.txt", env)
        cmd = env.execute.call_args_list[0][0][0]
        assert len(cmd) < 1_000  # cmd is just `mkdir -p X && cat > Y`
        assert env.execute.call_args_list[0][1]["stdin_data"] == big

    def test_path_with_spaces_is_quoted(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        remote_path = "/tmp/hermes results/abc file.txt"
        _write_to_sandbox("content", remote_path, env)
        cmd = env.execute.call_args_list[0][0][0]
        assert "'/tmp/hermes results'" in cmd
        assert "'/tmp/hermes results/abc file.txt'" in cmd

    def test_shell_metacharacters_neutralized(self):
        """Paths with shell metacharacters must be quoted to prevent injection."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        malicious_path = "/tmp/hermes-results/$(whoami).txt"
        _write_to_sandbox("content", malicious_path, env)
        cmd = env.execute.call_args_list[0][0][0]
        # The $() must not appear unquoted — shlex.quote wraps it
        assert "'/tmp/hermes-results/$(whoami).txt'" in cmd

    def test_semicolon_injection_neutralized(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        malicious_path = "/tmp/x; rm -rf /; echo .txt"
        _write_to_sandbox("content", malicious_path, env)
        cmd = env.execute.call_args_list[0][0][0]
        # The semicolons must be inside quotes, not acting as command separators
        assert "'/tmp/x; rm -rf /; echo .txt'" in cmd

    @pytest.mark.parametrize(
        "stdin_mode, probed, ok",
        [
            ("pipe", 512, False),      # short write: bytes lost
            ("pipe", 171, False),      # pipe backends must be exact
            ("heredoc", 171, False),   # heredoc stdin is byte-exact too; a +1 is a loss, not the old newline
            ("host", 3, False),        # host spillover: os.stat says only 3 bytes landed
            ("host", None, True),      # host spillover: real write, real stat
        ],
    )
    def test_size_probe_decides_lossless(self, stdin_mode, probed, ok):
        """An archive that is not byte-exact is discarded — never
        referenced to the model (port of lobehub/lobehub#18258). Multibyte content pins the
        comparison to UTF-8 bytes (170 here, 130 chars), on both the sandbox and host paths."""
        import os

        from tools.tool_result_storage import _write_to_spillover

        content = "héllo wörld ✓" * 10
        assert len(content.encode("utf-8")) == 170 != len(content)
        if stdin_mode == "host":
            filename = "tc_verify_host.txt"
            real_stat = os.stat

            def fake_stat(p, *a, **kw):
                if probed is not None and str(p).endswith(filename):
                    return type("S", (), {"st_size": probed})()
                return real_stat(p, *a, **kw)

            with patch("tools.tool_result_storage.os.stat", side_effect=fake_stat):
                path = _write_to_spillover(content, filename)
            assert (path is not None) is ok
            assert (get_spillover_dir() / filename).exists() is ok
            if path is not None:
                with open(path, encoding="utf-8") as fh:
                    assert fh.read() == content
            return
        env = MagicMock()
        env._stdin_mode = stdin_mode
        env.execute.side_effect = [
            {"output": "", "returncode": 0},          # write
            {"output": f"{probed}\n", "returncode": 0},  # wc -c
            {"output": "", "returncode": 0},          # rm -f cleanup (mismatch only)
        ]
        assert _write_to_sandbox(content, "/tmp/hermes-results/p.txt", env) is ok
        if not ok:
            rm_cmd = env.execute.call_args_list[2][0][0]
            assert rm_cmd.startswith("rm -f ") and "/tmp/hermes-results/p.txt" in rm_cmd
        else:
            assert env.execute.call_count == 2

    def test_unprobeable_backend_is_best_effort_success(self):
        """No wc / probe crash must not discard a likely-good archive."""
        env = MagicMock()
        env.execute.side_effect = [
            {"output": "", "returncode": 0},
            RuntimeError("exec transport gone"),
        ]
        assert _write_to_sandbox("data", "/tmp/hermes-results/np.txt", env) is True

class TestResolveStorageDir:
    def test_defaults_to_storage_dir_without_env(self):
        assert _resolve_storage_dir(None) == STORAGE_DIR

    def test_uses_env_temp_dir_when_available(self):
        env = MagicMock()
        env.get_temp_dir.return_value = "/var/host/tmp"
        assert _resolve_storage_dir(env) == "/var/host/tmp/hermes-results"

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
        assert "/tmp/hermes-results/test123.txt" in msg
        assert "read_file" in msg
        assert "first 100 chars..." in msg
        assert "..." in msg  # has_more indicator

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
        # Readability probe fails -> falls back to the in-sandbox write,
        # whose size probe returns unparseable output (best-effort success).
        env.execute.side_effect = [
            {"output": "", "returncode": 1},
            {"output": "", "returncode": 0},
            {"output": "", "returncode": 1},  # wc -c size probe: no answer
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
        assert env.execute.call_args_list[1][1]["stdin_data"] == content

    def test_tool_use_id_cannot_escape_storage_dir(self):
        env = MagicMock()
        # Readability probe fails -> in-sandbox write is the reference path.
        env.execute.side_effect = [
            {"output": "", "returncode": 1},
            {"output": "", "returncode": 0},
            {"output": "", "returncode": 1},  # wc -c size probe: no answer
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
        cmd = env.execute.call_args_list[1][0][0]
        target = cmd.split("cat > ", 1)[1].split(" <<", 1)[0]

        from tools.tool_result_storage import STORAGE_DIR

        assert f"Full output saved to: {STORAGE_DIR}/outside_whoami_x_" in result
        assert f"{STORAGE_DIR}/../" not in result
        assert target.startswith(f"{STORAGE_DIR}/outside_whoami_x_")
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

    def test_read_file_registry_cap_is_finite(self):
        """read_file must keep a finite registry cap (Layer 2 safety net)."""
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("read_file")
            # float('inf') disables the Layer 2 result-size guard.
            assert 0 < val < float("inf"), val
        except ImportError:
            pytest.skip("file_tools not importable in test env")

    def test_search_files_threshold(self):
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("search_files")
            assert 0 < val < float("inf"), val
        except ImportError:
            pytest.skip("file_tools not importable in test env")

# ── Host-side spillover ($HERMES_HOME/cache/spillover) ────────────────

class TestSpillover:
    @pytest.fixture(autouse=True)
    def _isolated_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        # Reset the once-per-process prune flag so each test is independent.
        import tools.tool_result_storage as trs
        monkeypatch.setattr(trs, "_spillover_pruned_homes", set())
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
            {"output": "60000\n", "returncode": 0},  # wc -c verification
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
        assert env.execute.call_count == 3
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
        old.write_text("old", encoding="utf-8")
        new.write_text("new", encoding="utf-8")
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
        old.write_text("ancient", encoding="utf-8")
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
