"""Regression tests for _rewrite_compound_background.

Context: bash parses ``A && B &`` as ``(A && B) &`` — it forks a subshell
for the compound and backgrounds the subshell. Inside the subshell, B
runs foreground, so the subshell waits for B. When B never exits on its
own (HTTP servers, ``yes > /dev/null``, etc.), the subshell is stuck in
``wait4`` forever and leaks as an orphan process. Pre-fix, we saw this
pattern leak processes across the fleet (vela, sal, combiagent).

The rewriter fixes this by wrapping the tail in a brace group —
``A && { B & }`` — so B runs as a simple backgrounded command inside
the current shell. No subshell fork, no wait.
"""

import shutil
import subprocess

import pytest

from tools.terminal_tool_sudo import _rewrite_compound_background as rewrite


class TestRewrites:
    """Commands that trigger the subshell-wait bug MUST be rewritten."""

    def test_simple_and_background(self):
        assert rewrite("A && B &") == "A && { B & }"

    def test_or_background(self):
        assert rewrite("A || B &") == "A || { B & }"


    def test_multiple_rewrites_in_one_script(self):
        cmd = "A && B &\nfalse || C &"
        assert rewrite(cmd) == "A && { B & }\nfalse || { C & }"


class TestPreserved:
    """Commands that DON'T have the bug MUST pass through unchanged."""

    def test_simple_background(self):
        # No compound — just background a single command. Works fine as-is.
        assert rewrite("sleep 5 &") == "sleep 5 &"

    def test_plain_server_background(self):
        assert rewrite("python3 -m http.server 0 &") == "python3 -m http.server 0 &"


    def test_whitespace_only(self):
        assert rewrite("   \n\t") == "   \n\t"


class TestRedirectsNotConfused:
    """``&>``, ``2>&1``, ``>&2`` must not be mistaken for background ``&``."""

    def test_amp_gt_redirect_alone(self):
        assert rewrite("echo hi &>/dev/null") == "echo hi &>/dev/null"


    def test_gt_amp_inside_compound(self):
        cmd = "A && B 2>&1 &"
        assert rewrite(cmd) == "A && { B 2>&1 & }"


class TestQuotingAndParens:
    """Shell metacharacters inside quotes/parens must not be parsed as operators."""

    def test_and_and_inside_single_quotes(self):
        cmd = "echo 'A && B &'"
        assert rewrite(cmd) == "echo 'A && B &'"


    def test_backslash_escaped_ampersand(self):
        # Escaped & is not a background operator.
        cmd = r"echo A \&\& B"
        assert rewrite(cmd) == cmd

    def test_comment_line_not_rewritten(self):
        cmd = "# A && B &\nC"
        assert rewrite(cmd) == "# A && B &\nC"


class TestIdempotence:
    """Running the rewriter twice should be a no-op on its own output."""

    def test_already_rewritten(self):
        once = rewrite("A && B &")
        twice = rewrite(once)
        assert once == twice
        assert twice == "A && { B & }"

    def test_multiline_idempotent(self):
        once = rewrite("cd /tmp && server &\nsleep 1")
        assert rewrite(once) == once


class TestEdgeCases:
    def test_only_chain_op_no_second_command(self):
        # Malformed input: bash would error, we shouldn't crash or rewrite.
        cmd = "A && &"
        # Don't assert a specific output; just don't raise.
        rewrite(cmd)


    def test_tabs_between_tokens(self):
        assert rewrite("A\t&&\tB\t&") == "A\t&&\t{ B\t& }"


class TestTrailingStatementSeparator:
    """A statement after the backgrounded compound on the SAME line.

    In ``A && B & C`` the trailing ``&`` is both the background operator and
    the separator between the compound and ``C``. The rewrite consumes that
    ``&`` into the brace group; without restoring a separator the result is
    ``A && { B & } C`` — a bash syntax error (a brace group must be terminated
    by ``;``, ``&``, ``|``, a newline, or ``)``/``}`` before the next command).
    That mangles a valid command into one that fails entirely.
    """

    def test_trailing_command_gets_separator(self):
        assert rewrite("echo hi && sleep 5 & echo done") == (
            "echo hi && { sleep 5 & } ; echo done"
        )

    def test_trailing_chain_gets_separator(self):
        assert rewrite("a && b & c && d") == "a && { b & } ; c && d"

    def test_redirect_then_trailing_command(self):
        assert rewrite("echo hi && sleep 5 &>/dev/null & echo done") == (
            "echo hi && { sleep 5 &>/dev/null & } ; echo done"
        )

    def test_existing_semicolon_separator_untouched(self):
        # An explicit `;` already separates the group; don't add a second one.
        assert rewrite("a && b &; c") == "a && { b & }; c"

    def test_newline_separator_untouched(self):
        # A newline already terminates the brace group — no `;` needed.
        assert rewrite("a && b &\necho next") == "a && { b & }\necho next"

    def test_pipe_after_group_untouched(self):
        # `{ ...; } | cmd` is valid; the pipe is its own terminator.
        assert rewrite("a && b & | cat") == "a && { b & } | cat"

    def test_redirect_prefix_on_trailing_command_gets_separator(self):
        # `&>` after the group is a redirect for the NEXT command, not a
        # terminator: `{ b & } &>/dev/null c` is a syntax error.
        assert rewrite("a && b & &>/dev/null c") == "a && { b & } ; &>/dev/null c"

    def test_case_arm_terminator_untouched(self):
        # `;;` already terminates the arm; adding `;` would leave an empty
        # command between `;` and `;;`, which bash rejects.
        assert rewrite("case $x in p) b && c & ;; esac") == "case $x in p) b && { c & } ;; esac"

    def test_separator_is_idempotent(self):
        once = rewrite("echo hi && sleep 5 & echo done")
        assert rewrite(once) == once

    def test_second_background_then_trailing(self):
        assert rewrite("echo a && sleep 5 & echo b & echo c") == (
            "echo a && { sleep 5 & } ; echo b & echo c"
        )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
class TestRewriteIsValidBash:
    """The rewrite must always produce syntactically valid bash.

    This is the crux of the trailing-statement bug: a mangled command fails
    with a confusing syntax error and neither half runs. ``bash -n`` parses
    without executing, so it catches the corruption directly.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "echo hi && sleep 5 & echo done",
            "a && b & c && d",
            "echo hi && sleep 5 &>/dev/null & echo done",
            "echo a && sleep 5 & echo b & echo c",
            "A && B &",
            "A && B &; C",
            "A && B &\nC",
            "cd /tmp && python3 -m http.server 0 &>/dev/null & curl localhost",
            "a && b & &>/dev/null c",
            "case $x in p) b && c & ;; esac",
            "A && B & echo x\nC && D & echo y && E & echo z",
        ],
    )
    def test_rewrite_parses(self, command):
        rewritten = rewrite(command)
        result = subprocess.run(
            ["bash", "-n", "-c", rewritten],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"rewrite produced invalid bash: {rewritten!r}\n{result.stderr}"
        )

    def test_trailing_statement_actually_runs(self):
        # End-to-end: the command after the backgrounded compound must run.
        rewritten = rewrite("echo first && true & echo SECOND_RAN")
        result = subprocess.run(
            ["bash", "-c", rewritten], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "SECOND_RAN" in result.stdout
