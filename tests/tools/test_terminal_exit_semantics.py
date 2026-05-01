"""Tests for terminal command exit code semantic interpretation."""

import pytest

from tools.terminal_tool import _interpret_exit_code


class TestInterpretExitCode:
    """Test _interpret_exit_code returns correct notes for known command semantics."""

    # ---- exit code 0 always returns None ----

    def test_success_returns_none(self):
        assert _interpret_exit_code("grep foo bar", 0) is None
        assert _interpret_exit_code("diff a b", 0) is None
        assert _interpret_exit_code("test -f /etc/passwd", 0) is None

    # ---- grep / rg family: exit 1 = no matches ----

    @pytest.mark.parametrize("cmd", [
        "grep 'pattern' file.txt",
        "egrep 'pattern' file.txt",
        "fgrep 'pattern' file.txt",
        "rg 'foo' .",
        "ag 'foo' .",
        "ack 'foo' .",
    ])
    def test_grep_family_no_matches(self, cmd):
        result = _interpret_exit_code(cmd, 1)
        assert result is not None
        assert "no matches" in result.lower()

    def test_grep_real_error_no_note(self):
        """grep exit 2+ is a real error — should return None."""
        assert _interpret_exit_code("grep 'foo' bar", 2) is None
        assert _interpret_exit_code("rg 'foo' .", 2) is None

    # ---- diff: exit 1 = files differ ----

    def test_diff_files_differ(self):
        result = _interpret_exit_code("diff file1 file2", 1)
        assert result is not None
        assert "differ" in result.lower()

    def test_colordiff_files_differ(self):
        result = _interpret_exit_code("colordiff file1 file2", 1)
        assert result is not None
        assert "differ" in result.lower()

    def test_diff_real_error_no_note(self):
        assert _interpret_exit_code("diff a b", 2) is None

    # ---- test / [: exit 1 = condition false ----

    def test_test_condition_false(self):
        result = _interpret_exit_code("test -f /nonexistent", 1)
        assert result is not None
        assert "false" in result.lower()

    def test_bracket_condition_false(self):
        result = _interpret_exit_code("[ -f /nonexistent ]", 1)
        assert result is not None
        assert "false" in result.lower()

    # ---- find: exit 1 = partial success ----

    def test_find_partial_success(self):
        result = _interpret_exit_code("find . -name '*.py'", 1)
        assert result is not None
        assert "inaccessible" in result.lower()

    # ---- curl: various informational codes ----

    def test_curl_timeout(self):
        result = _interpret_exit_code("curl https://example.com", 28)
        assert result is not None
        assert "timed out" in result.lower()

    def test_curl_connection_refused(self):
        result = _interpret_exit_code("curl http://localhost:99999", 7)
        assert result is not None
        assert "connect" in result.lower()

    # ---- git: exit 1 is context-dependent ----

    def test_git_diff_exit_1(self):
        result = _interpret_exit_code("git diff HEAD~1", 1)
        assert result is not None
        assert "normal" in result.lower()

    # ---- pipeline / chain handling ----

    def test_pipeline_last_command(self):
        """In a pipeline, the last command determines the exit code."""
        result = _interpret_exit_code("ls -la | grep 'pattern'", 1)
        assert result is not None
        assert "no matches" in result.lower()

    def test_and_chain_last_command(self):
        result = _interpret_exit_code("cd /tmp && grep foo bar", 1)
        assert result is not None
        assert "no matches" in result.lower()

    def test_semicolon_chain_last_command(self):
        result = _interpret_exit_code("cat file; diff a b", 1)
        assert result is not None
        assert "differ" in result.lower()

    def test_or_chain_last_command(self):
        result = _interpret_exit_code("false || grep foo bar", 1)
        assert result is not None
        assert "no matches" in result.lower()

    # ---- full paths ----

    def test_full_path_command(self):
        result = _interpret_exit_code("/usr/bin/grep 'foo' bar", 1)
        assert result is not None
        assert "no matches" in result.lower()

    # ---- env var prefix ----

    def test_env_var_prefix_stripped(self):
        result = _interpret_exit_code("LANG=C grep 'foo' bar", 1)
        assert result is not None
        assert "no matches" in result.lower()

    def test_multiple_env_vars(self):
        result = _interpret_exit_code("FOO=1 BAR=2 grep 'foo' bar", 1)
        assert result is not None
        assert "no matches" in result.lower()

    # ---- unknown commands return None ----

    @pytest.mark.parametrize("cmd", [
        "python3 script.py",
        "rm -rf /tmp/test",
        "npm test",
        "make build",
        "cargo build",
    ])
    def test_unknown_commands_return_none(self, cmd):
        assert _interpret_exit_code(cmd, 1) is None

    # ---- edge cases ----

    def test_empty_command(self):
        assert _interpret_exit_code("", 1) is None

    def test_only_env_vars(self):
        """Command with only env var assignments, no actual command."""
        assert _interpret_exit_code("FOO=bar", 1) is None
