"""Tests for terminal safety-filter false-positive fixes.

Regression tests for https://github.com/NousResearch/hermes-agent/issues/20064

The ``_foreground_background_guidance`` function previously used a naive
word-boundary regex that fired on *any* occurrence of ``nohup``/``disown``/
``setsid`` — including inside quoted arguments, commit messages, Python -c
snippets, and PR body text.

After the fix, quoted string content is stripped before the regex is applied,
so only actual shell-level token usage triggers the safety filter.
"""

import pytest
from tools.terminal_tool import _foreground_background_guidance, _strip_quoted_strings


# ---------------------------------------------------------------------------
# _strip_quoted_strings unit tests
# ---------------------------------------------------------------------------

class TestStripQuotedStrings:
    """Verify the helper removes quoted content without touching unquoted tokens."""

    def test_removes_double_quoted_content(self):
        result = _strip_quoted_strings('git commit -m "fix: replace setsid call"')
        assert "setsid" not in result
        assert "git commit -m" in result

    def test_removes_single_quoted_content(self):
        result = _strip_quoted_strings("echo 'nohup is a unix command'")
        assert "nohup" not in result
        assert "echo" in result

    def test_preserves_unquoted_tokens(self):
        result = _strip_quoted_strings("nohup ./server.sh")
        assert "nohup" in result

    def test_preserves_unquoted_setsid(self):
        result = _strip_quoted_strings("setsid my_daemon --foreground")
        assert "setsid" in result

    def test_removes_escaped_quote_content(self):
        # Escaped quote inside a double-quoted string should still be consumed
        result = _strip_quoted_strings(r'"he said \"nohup\" here"')
        assert "nohup" not in result

    def test_empty_string_safe(self):
        assert _strip_quoted_strings("") == ""

    def test_no_quotes_unchanged(self):
        cmd = "echo hello world"
        assert _strip_quoted_strings(cmd) == cmd

    def test_multiple_quoted_segments(self):
        result = _strip_quoted_strings(
            'gh pr create --title "fix setsid" --body "removed nohup call"'
        )
        assert "setsid" not in result
        assert "nohup" not in result
        assert "gh pr create --title" in result


# ---------------------------------------------------------------------------
# _foreground_background_guidance — false-positive regression tests
# ---------------------------------------------------------------------------

class TestForegroundBackgroundGuidanceFalsePositives:
    """Commands that SHOULD NOT trigger the safety filter after the fix."""

    def test_git_commit_message_with_setsid(self):
        """setsid inside a commit message body must not be flagged."""
        assert _foreground_background_guidance(
            'git commit -m "fix: replace preexec_fn=os.setsid with process_group=0"'
        ) is None

    def test_python_c_with_setsid_string(self):
        """setsid inside a Python -c string literal must not be flagged."""
        assert _foreground_background_guidance(
            "python3 -c \"import os; x = 'preexec_fn=os.setsid'\""
        ) is None

    def test_gh_pr_create_body_with_nohup(self):
        """nohup in a gh pr body argument must not be flagged."""
        assert _foreground_background_guidance(
            "gh pr create --body \"We removed the nohup call in favour of background=true\""
        ) is None

    def test_echo_nohup_explanation(self):
        """echo with a quoted nohup explanation must not be flagged."""
        assert _foreground_background_guidance(
            "echo 'The nohup command prevents SIGHUP from killing the process'"
        ) is None

    def test_grep_for_disown(self):
        """grep targeting the word disown in a quoted pattern must not be flagged."""
        assert _foreground_background_guidance(
            "grep -r 'disown' ./docs/"
        ) is None

    def test_sed_replacing_setsid(self):
        """sed with setsid in quoted replacement string must not be flagged."""
        assert _foreground_background_guidance(
            "sed -i 's/os.setsid/process_group=0/g' agent/runner.py"
        ) is None


# ---------------------------------------------------------------------------
# _foreground_background_guidance — true-positive tests (must still fire)
# ---------------------------------------------------------------------------

class TestForegroundBackgroundGuidanceTruePositives:
    """Commands that SHOULD still trigger the safety filter."""

    def test_bare_nohup_command(self):
        """Actual nohup usage as the first token must still be flagged."""
        result = _foreground_background_guidance("nohup ./server.sh &")
        assert result is not None
        assert "background=true" in result

    def test_bare_setsid_command(self):
        """Actual setsid as a shell command must still be flagged."""
        result = _foreground_background_guidance("setsid my_daemon --foreground")
        assert result is not None
        assert "background=true" in result

    def test_nohup_with_redirect(self):
        """nohup with output redirection (common pattern) must still be flagged."""
        result = _foreground_background_guidance(
            "nohup pnpm dev > /tmp/server.log 2>&1 &"
        )
        assert result is not None
        assert "nohup" in result.lower()

    def test_bare_disown_command(self):
        """disown as a shell built-in must still be flagged."""
        result = _foreground_background_guidance("disown %1")
        assert result is not None
        assert "background=true" in result

    def test_pnpm_dev_long_lived(self):
        """Long-lived dev server pattern must still be flagged."""
        result = _foreground_background_guidance("pnpm dev")
        assert result is not None
        assert "background=true" in result

    def test_trailing_amp(self):
        """Trailing & backgrounding must still be flagged."""
        result = _foreground_background_guidance("python server.py &")
        assert result is not None
        assert "background=true" in result
