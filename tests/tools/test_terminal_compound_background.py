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

from tools.terminal_tool import _rewrite_compound_background as rewrite


class TestRewrites:
    """Commands that trigger the subshell-wait bug MUST be rewritten."""

    def test_simple_and_background(self):
        assert rewrite("A && B &") == "A && { B & }"

    def test_or_background(self):
        assert rewrite("A || B &") == "A || { B & }"

    def test_chained_and(self):
        assert rewrite("A && B && C &") == "A && B && { C & }"

    def test_chained_or(self):
        assert rewrite("A || B || C &") == "A || B || { C & }"

    def test_mixed_and_or(self):
        assert rewrite("A && B || C &") == "A && B || { C & }"

    def test_realistic_server_start(self):
        # The exact shape observed in the vela incident.
        cmd = (
            "cd /home/exedev && python3 -m http.server 8000 &>/dev/null &\n"
            "sleep 1\n"
            'curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/'
        )
        expected = (
            "cd /home/exedev && { python3 -m http.server 8000 &>/dev/null & }\n"
            "sleep 1\n"
            'curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/'
        )
        assert rewrite(cmd) == expected

    def test_newline_resets_chain_state(self):
        # A && newline starts a new statement; B & on its own line is simple.
        cmd = "A && B\nC &"
        assert rewrite(cmd) == "A && B\nC &"

    def test_semicolon_resets_chain_state(self):
        cmd = "A && B; C &"
        assert rewrite(cmd) == "A && B; C &"

    def test_pipe_resets_chain_state(self):
        cmd = "A && B | C &"
        assert rewrite(cmd) == "A && B | C &"

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

    def test_semicolon_sequence(self):
        assert rewrite("cd /tmp; start-server &") == "cd /tmp; start-server &"

    def test_no_trailing_ampersand(self):
        assert rewrite("A && B") == "A && B"

    def test_no_chain_at_all(self):
        assert rewrite("echo hello") == "echo hello"

    def test_empty_string(self):
        assert rewrite("") == ""

    def test_whitespace_only(self):
        assert rewrite("   \n\t") == "   \n\t"


class TestRedirectsNotConfused:
    """``&>``, ``2>&1``, ``>&2`` must not be mistaken for background ``&``."""

    def test_amp_gt_redirect_alone(self):
        assert rewrite("echo hi &>/dev/null") == "echo hi &>/dev/null"

    def test_fd_to_fd_redirect(self):
        assert rewrite("cmd 2>&1") == "cmd 2>&1"

    def test_fd_redirect_with_trailing_bg(self):
        # 2>&1 is redirect; trailing & is simple bg (no compound).
        assert rewrite("cmd 2>&1 &") == "cmd 2>&1 &"

    def test_amp_gt_inside_compound_background(self):
        # &> should be preserved; the trailing & still needs wrapping.
        cmd = "A && B &>/dev/null &"
        assert rewrite(cmd) == "A && { B &>/dev/null & }"

    def test_gt_amp_inside_compound(self):
        cmd = "A && B 2>&1 &"
        assert rewrite(cmd) == "A && { B 2>&1 & }"


class TestQuotingAndParens:
    """Shell metacharacters inside quotes/parens must not be parsed as operators."""

    def test_and_and_inside_single_quotes(self):
        cmd = "echo 'A && B &'"
        assert rewrite(cmd) == "echo 'A && B &'"

    def test_and_and_inside_double_quotes(self):
        cmd = 'echo "A && B &"'
        assert rewrite(cmd) == 'echo "A && B &"'

    def test_parenthesised_subshell_left_alone(self):
        # `(A && B) &` has the same bug class but isn't the common agent
        # pattern. Leave for a follow-up; do not rewrite and do not
        # misrewrite content inside the parens.
        assert rewrite("(A && B) &") == "(A && B) &"

    def test_command_substitution_not_rewritten(self):
        # $(A && B) is command substitution; the `&&` inside is a compound
        # expression in the subshell, unrelated to the outer `&`.
        cmd = 'echo "$(A && B)" &'
        assert rewrite(cmd) == 'echo "$(A && B)" &'

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

    def test_only_trailing_ampersand(self):
        assert rewrite("&") == "&"

    def test_leading_whitespace(self):
        assert rewrite("   A && B &") == "   A && { B & }"

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
