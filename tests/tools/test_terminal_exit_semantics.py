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

    # ---- curl: genuine failures are NOT tagged benign ----
    # curl 6/7/22/28 (DNS / connect / HTTP-error / timeout) are real failures,
    # not "context". They must return None so the classifiers flag them — the
    # failure reason is already in curl's stderr (merged into output). If they
    # carried exit_code_meaning, a curl that could not connect would render as
    # success and dodge the guardrail.

    @pytest.mark.parametrize("code", [6, 7, 22, 28])
    def test_curl_failure_codes_not_benign(self, code):
        assert _interpret_exit_code("curl https://example.com", code) is None

    # ---- git: exit 1 is ambiguous, so never tagged benign ----
    # git exit 1 is benign for `git diff` (files differ) but a genuine failure
    # for push-rejected / merge|rebase conflicts / `commit` with nothing staged.
    # A by-base-command table can't disambiguate, and masking a failed push is
    # worse than over-tagging `git diff`, so git returns None for all of them.

    def test_git_exit_1_not_benign(self):
        assert _interpret_exit_code("git diff HEAD~1", 1) is None
        assert _interpret_exit_code("git push origin main", 1) is None

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

    # ---- interrupt (130) has no command-semantic note ----

    def test_interrupt_130_has_no_meaning(self):
        """A user interrupt (130) is not a command-semantic benign code, so
        _interpret_exit_code returns None for it. The benign handling for 130
        lives in the failure classifiers (explicit INTERRUPT_EXIT_CODE check),
        not here — this test locks that contract so the two layers stay split."""
        assert _interpret_exit_code("grep 'foo' bar", 130) is None
        assert _interpret_exit_code("python3 script.py", 130) is None
