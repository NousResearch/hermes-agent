#!/usr/bin/env python3
"""Unit tests for command_compressors.py"""

import pytest
from tools.command_compressors import (
    DEFAULT_COMPRESSORS,
    GitStatusCompressor,
    GitDiffCompressor,
    PytestCompressor,
    CargoTestCompressor,
    LsCompressor,
    DockerPsCompressor,
    RuffCheckCompressor,
    GoTestCompressor,
    DockerLogsCompressor,
    NpmTestCompressor,
    _normalize_cmd,
    _has_flag,
    CompoundCommandSplitter,
)


# -----------------------------------------------------------------------
# Git Status
# -----------------------------------------------------------------------

class TestGitStatusCompressor:
    def test_clean_repo(self):
        c = GitStatusCompressor()
        stdout = "On branch main\nnothing to commit, working tree clean"
        assert c.can_compress("git status", stdout, "")
        result = c.compress("git status", stdout, "", 0)
        assert "main" in result
        assert "clean" in result

    def test_with_staged_changes(self):
        c = GitStatusCompressor()
        stdout = (
            "On branch feature/login\n"
            "Changes to be committed:\n"
            "  new file:   src/auth.py\n"
            "  modified:   src/config.py\n"
            "Changes not staged for commit:\n"
            "  modified:   README.md\n"
            "Untracked files:\n"
            "  test/new.py\n"
        )
        result = c.compress("git status", stdout, "", 0)
        assert "feature/login" in result
        assert "1 new" in result
        assert "1 modified" in result or "?1 untracked" in result

    def test_detached_head(self):
        c = GitStatusCompressor()
        stdout = "HEAD detached at abc1234\nnothing to commit"
        result = c.compress("git status", stdout, "", 0)
        assert "detached" in result

    def test_non_git_command(self):
        c = GitStatusCompressor()
        assert not c.can_compress("git log", "", "")

    def test_can_compress_flags(self):
        c = GitStatusCompressor()
        assert c.can_compress("git status -sb", "", "")
        assert c.can_compress("git status --short", "", "")
        assert not c.can_compress("git log", "", "")


# -----------------------------------------------------------------------
# Git Diff
# -----------------------------------------------------------------------

class TestGitDiffCompressor:
    def test_no_changes(self):
        c = GitDiffCompressor()
        assert c.can_compress("git diff", "", "")
        result = c.compress("git diff", "", "", 0)
        assert "no changes" in result.lower()

    def test_stat_summary(self):
        c = GitDiffCompressor()
        # diff --stat style output
        stdout = (
            " src/model.py | 10 +++++------\n"
            " src/main.py  |  2 +-\n"
            " 2 files changed, 6 insertions(+), 6 deletions(-)"
        )
        result = c.compress("git diff", stdout, "", 0)
        assert "2 files" in result
        assert "6" in result

    def test_full_diff(self):
        c = GitDiffCompressor()
        stdout = (
            "diff --git a/src/main.py b/src/main.py\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,3 +1,4 @@\n"
            " def hello():\n"
            "-    print('hi')\n"
            "+    print('hello world')\n"
            "+    return True\n"
        )
        result = c.compress("git diff", stdout, "", 0)
        assert "src/main.py" in result

    def test_cached_flag(self):
        c = GitDiffCompressor()
        stdout = "1 file changed, 2 insertions(+)"
        result = c.compress("git diff --cached", stdout, "", 0)
        assert "staged" in result


# -----------------------------------------------------------------------
# Pytest
# -----------------------------------------------------------------------

class TestPytestCompressor:
    def test_all_passed(self):
        c = PytestCompressor()
        stdout = (
            "collected 10 items\n"
            "tests/test_a.py::test_one PASSED\n"
            "tests/test_a.py::test_two PASSED\n"
            "===== 2 passed in 0.45s =====\n"
        )
        assert c.can_compress("pytest -v", stdout, "")
        result = c.compress("pytest -v", stdout, "", 0)
        assert "passed" in result
        assert "FAILED" not in result
        assert "✓" in result

    def test_some_failed(self):
        c = PytestCompressor()
        stdout = (
            "tests/test_main.py::test_fail FAILED\n"
            "tests/test_main.py::test_ok PASSED\n"
            "===== 1 failed, 1 passed in 0.12s =====\n"
        )
        result = c.compress("pytest", stdout, "", 1)
        assert "failed" in result
        assert "1" in result

    def test_empty_output(self):
        c = PytestCompressor()
        result = c.compress("pytest", "", "", 0)
        assert result  # Must return something, not crash

    def test_with_skipped(self):
        c = PytestCompressor()
        stdout = "===== 5 passed, 2 skipped in 1.00s ====="
        result = c.compress("pytest", stdout, "", 0)
        assert "passed" in result
        assert "skipped" in result

    def test_python_m_pytest(self):
        c = PytestCompressor()
        assert c.can_compress("python -m pytest", "", "")
        assert c.can_compress("py.test", "", "")


# -----------------------------------------------------------------------
# Cargo Test
# -----------------------------------------------------------------------

class TestCargoTestCompressor:
    def test_all_pass(self):
        c = CargoTestCompressor()
        stdout = (
            "running 12 tests\n"
            "test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; run time: 1.23s\n"
        )
        assert c.can_compress("cargo test", stdout, "")
        result = c.compress("cargo test", stdout, "", 0)
        # Must contain pass count and duration (from test result summary)
        assert "passed" in result
        assert "1.23" in result

    def test_with_failures(self):
        c = CargoTestCompressor()
        stdout = (
            "test result: FAILED. 3 passed; 1 failed; 0 ignored; run time = 2.10s\n"
        )
        result = c.compress("cargo test", stdout, "", 101)
        # Output must indicate failure — "✗" symbol appears for FAILED result
        assert "✗" in result

    def test_compile_error(self):
        c = CargoTestCompressor()
        stdout = "error[E0308]: mismatched types\n  --> src/main.rs:5:12"
        result = c.compress("cargo build", stdout, "", 1)
        assert "compile error" in result

    def test_cargo_check(self):
        c = CargoTestCompressor()
        assert c.can_compress("cargo check", "", "")


# -----------------------------------------------------------------------
# LS
# -----------------------------------------------------------------------

class TestLsCompressor:
    def test_empty_dir(self):
        c = LsCompressor()
        assert c.can_compress("ls", "", "")
        assert c.can_compress("ls -la", "", "")
        assert c.can_compress("tree", "", "")
        result = c.compress("ls", "", "", 0)
        assert "empty" in result

    def test_normal_ls(self):
        c = LsCompressor()
        stdout = "total 32\ndrwxr-xr-x  5 user staff  160 Jan 19 10:00 .\ndrwxr-xr-x  2 user staff   64 Jan 19 09:00 src\ndrwxr-xr-x  3 user staff   96 Jan 19 10:00 tests"
        result = c.compress("ls -la", stdout, "", 0)
        assert "ls" in result
        assert "entries" in result

    def test_tree_output(self):
        c = LsCompressor()
        stdout = "+- src\n  +- main.py\n+- tests\n  +- test_main.py"
        result = c.compress("tree", stdout, "", 0)
        assert "tree" in result
        assert "dirs" in result or "files" in result


# -----------------------------------------------------------------------
# Docker PS
# -----------------------------------------------------------------------

class TestDockerPsCompressor:
    def test_no_containers(self):
        c = DockerPsCompressor()
        stdout = "CONTAINER ID   IMAGE   COMMAND   CREATED   STATUS   PORTS   NAMES"
        result = c.compress("docker ps", stdout, "", 0)
        assert "no containers" in result.lower()

    def test_running_containers(self):
        c = DockerPsCompressor()
        stdout = (
            "CONTAINER ID   IMAGE       STATUS        PORTS     NAMES\n"
            "abc123         nginx       Up 2 hours    80/tcp    web\n"
            "def456         postgres    Up 5 hours    5432/tcp  db\n"
        )
        result = c.compress("docker ps", stdout, "", 0)
        assert "docker" in result
        assert "running" in result or "2 total" in result

    def test_mixed_status(self):
        c = DockerPsCompressor()
        stdout = (
            "CONTAINER ID   IMAGE   STATUS        NAMES\n"
            "abc123         nginx   Up 2 hours    web\n"
            "def456         redis   Exited (0)   cache\n"
        )
        result = c.compress("docker ps -a", stdout, "", 0)
        assert "exited" in result or "running" in result


# -----------------------------------------------------------------------
# Ruff Check
# -----------------------------------------------------------------------

class TestRuffCheckCompressor:
    def test_no_violations(self):
        c = RuffCheckCompressor()
        stdout = ""
        result = c.compress("ruff check .", stdout, "", 0)
        assert "no violations" in result.lower()
        assert "✓" in result

    def test_with_errors(self):
        c = RuffCheckCompressor()
        stdout = (
            "src/main.py:10:5: E302 expected 2 blank lines\n"
            "Found 1 error (1 error, 0 warning)\n"
        )
        result = c.compress("ruff check .", stdout, "", 1)
        assert "violations" in result
        assert "error" in result


# -----------------------------------------------------------------------
# Go Test
# -----------------------------------------------------------------------

class TestGoTestCompressor:
    def test_all_pass(self):
        c = GoTestCompressor()
        stdout = "ok  \tpackage/path\t0.500s\n"
        assert c.can_compress("go test", stdout, "")
        result = c.compress("go test ./...", stdout, "", 0)
        assert "passed" in result
        assert "✓" in result

    def test_with_failure(self):
        c = GoTestCompressor()
        stdout = "FAIL\tpackage/path\t0.100s\n"
        result = c.compress("go test", stdout, "", 1)
        assert "fail" in result.lower()
        assert "✗" in result

    def test_verbose_output(self):
        c = GoTestCompressor()
        stdout = (
            "=== RUN   TestMain\n"
            "--- PASS: TestMain (0.00s)\n"
            "PASS\n"
        )
        result = c.compress("go test -v", stdout, "", 0)
        assert "passed" in result or "PASS" in result


# -----------------------------------------------------------------------
# Docker Logs
# -----------------------------------------------------------------------

class TestDockerLogsCompressor:
    def test_empty_logs(self):
        c = DockerLogsCompressor()
        result = c.compress("docker logs abc123", "", "", 0)
        assert "empty" in result.lower()

    def test_short_logs(self):
        c = DockerLogsCompressor()
        stdout = "Starting server\nListening on port 8080\n"
        result = c.compress("docker logs abc123", stdout, "", 0)
        # Short logs pass through
        assert "Starting server" in result

    def test_long_logs_with_errors(self):
        c = DockerLogsCompressor()
        stdout = "\n".join([f"Log line {i}" for i in range(50)])
        stdout += "\n[ERROR] Connection refused"
        result = c.compress("docker logs abc123", stdout, "", 0)
        assert "lines" in result
        assert "errors" in result


# -----------------------------------------------------------------------
# NPM Test
# -----------------------------------------------------------------------

class TestNpmTestCompressor:
    def test_all_pass(self):
        c = NpmTestCompressor()
        stdout = "Test Suites: 1 passed, 1 total\nTests:       5 passed, 5 total"
        assert c.can_compress("npm test", stdout, "")
        result = c.compress("npm test", stdout, "", 0)
        assert "passed" in result
        assert "✓" in result

    def test_with_failures(self):
        c = NpmTestCompressor()
        stdout = "Test Suites: 1 failed, 1 passed, 2 total\nTests:       3 failed, 5 passed, 8 total"
        result = c.compress("npm test", stdout, "", 1)
        assert "failed" in result


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

class TestNormalizeCmd:
    def test_basic(self):
        assert _normalize_cmd("git status") == "git"
        assert _normalize_cmd("cargo test") == "cargo"

    def test_with_flags(self):
        assert _normalize_cmd("git status -sb") == "git"

    def test_with_path(self):
        assert _normalize_cmd("/usr/bin/cargo test") == "cargo"

    def test_sudo(self):
        assert _normalize_cmd("sudo cargo test") == "cargo"


class TestHasFlag:
    def test_dash_dash(self):
        assert _has_flag("git status --short", "--short")

    def test_dash_s(self):
        assert _has_flag("git status -s", "-s")

    def test_no_match(self):
        assert not _has_flag("git log", "status")


# -----------------------------------------------------------------------
# CompressorRegistry integration
# -----------------------------------------------------------------------

class TestCompressorRegistry:
    def test_git_status_via_registry(self):
        stdout = "On branch main\nnothing to commit"
        result = DEFAULT_COMPRESSORS.compress("git status", stdout, "", 0)
        assert result is not None
        assert "main" in result

    def test_pytest_via_registry(self):
        stdout = "===== 3 passed in 0.50s ====="
        result = DEFAULT_COMPRESSORS.compress("pytest", stdout, "", 0)
        assert result is not None
        assert "passed" in result

    def test_unknown_command_returns_none(self):
        result = DEFAULT_COMPRESSORS.compress("curl https://example.com", "<html>...", "", 0)
        # Default compressor returns None, signalling no compression
        assert result is None

    def test_stats_tracking(self):
        DEFAULT_COMPRESSORS.reset_stats()
        stdout = "===== 5 passed in 1.00s ====="
        result = DEFAULT_COMPRESSORS.compress("pytest", stdout, "", 0)
        stats = DEFAULT_COMPRESSORS.get_stats()
        assert "pytest" in stats or "python -m pytest" in stats


# -----------------------------------------------------------------------
# Compound Command Splitter
# -----------------------------------------------------------------------

class TestCompoundCommandSplitter:
    """Test the compound command splitter."""

    def test_simple_command(self):
        """Single command should return as-is."""
        splitter = CompoundCommandSplitter()
        assert splitter.split("git status") == ["git status"]
        assert splitter.split("cargo test --lib") == ["cargo test --lib"]

    def test_and_operator(self):
        """&& should split commands."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cargo fmt && cargo test")
        assert result == ["cargo fmt", "cargo test"]

    def test_or_operator(self):
        """|| should split commands."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cmd1 || cmd2")
        assert result == ["cmd1", "cmd2"]

    def test_semicolon(self):
        """Semicolon should split commands."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cmd1 ; cmd2")
        assert result == ["cmd1", "cmd2"]

    def test_pipe_preserved(self):
        """Pipe should NOT split (stays in same segment)."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cargo test 2>&1 | tail -20")
        assert result == ["cargo test 2>&1 | tail -20"]

    def test_compound_with_pipe(self):
        """Compound command with pipe."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cargo fmt --all && cargo test 2>&1 | tail -20")
        assert result == ["cargo fmt --all", "cargo test 2>&1 | tail -20"]

    def test_multiple_and(self):
        """Multiple && operators."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cmd1 && cmd2 && cmd3")
        assert result == ["cmd1", "cmd2", "cmd3"]

    def test_mixed_operators(self):
        """Mixed && and ||."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cmd1 && cmd2 || cmd3")
        assert result == ["cmd1", "cmd2", "cmd3"]

    def test_empty_command(self):
        """Empty command returns empty list."""
        splitter = CompoundCommandSplitter()
        assert splitter.split("") == []
        assert splitter.split("   ") == []

    def test_quoted_args(self):
        """Quoted arguments should not be split."""
        splitter = CompoundCommandSplitter()
        # Quotes should preserve the argument
        result = splitter.split('echo "hello && world"')
        # The quoted string is one argument
        assert len(result) == 1

    def test_redirect_preserved(self):
        """Redirects stay with their command."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cmd1 > output.txt && cmd2")
        assert "cmd1" in result[0]
        assert ">" in result[0] or "output.txt" in result[0]

    def test_compress_segments_single_delegates(self):
        """Single segment returns None, delegating to registry."""
        splitter = CompoundCommandSplitter()
        stdout = "test output"
        result = splitter.compress_segments(DEFAULT_COMPRESSORS, "git status", stdout, "", 0)
        # Single segment returns None, letting registry handle it
        assert result is None

    def test_compress_segments_compound(self):
        """Compound command returns None (cannot compress without per-segment output)."""
        splitter = CompoundCommandSplitter()
        stdout = "cargo fmt output\ncargo test output"
        result = splitter.compress_segments(DEFAULT_COMPRESSORS, "cargo fmt && cargo test", stdout, "", 0)
        # Should return None, signaling fallback to raw output
        assert result is None

    def test_subshell_not_split(self):
        """Commands with subshells are not split."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("echo done && (git status | mail test)")
        assert len(result) == 1
        assert result[0] == "echo done && (git status | mail test)"

    def test_command_substitution_not_split(self):
        """Commands with $() are not split."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("echo $(date) && git status")
        assert len(result) == 1

    def test_backtick_substitution_not_split(self):
        """Commands with backticks are not split."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("echo `date` && git status")
        assert len(result) == 1

    def test_backslash_escaped_quote(self):
        """Escaped quotes in arguments."""
        splitter = CompoundCommandSplitter()
        # Should not crash, should handle escape
        result = splitter.split(r'echo "hello \"world\""')
        assert len(result) == 1

    def test_backslash_at_end(self):
        """Backslash at end of text should not crash."""
        splitter = CompoundCommandSplitter()
        # This tests the fix for the backslash-at-EOF bug
        # Note: Can't use raw string with trailing backslash in Python
        result = splitter.split('echo "test' + chr(92))  # "test\
        # Should return something without crashing
        assert isinstance(result, list)

    def test_pipe_chain_preserved(self):
        """Multiple pipes in a chain stay together."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cat file | grep foo | tail -5")
        assert result == ["cat file | grep foo | tail -5"]

    def test_complex_compound(self):
        """Complex compound command with multiple operators."""
        splitter = CompoundCommandSplitter()
        result = splitter.split("cmd1 && cmd2 || cmd3 ; cmd4")
        assert result == ["cmd1", "cmd2", "cmd3", "cmd4"]
