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

Two gaps found in that fix (issue #68915):

1. ``A && B & C`` rewrote to ``A && { B & } C``, which is a bash syntax
   error (a brace *group* needs a statement terminator before the next
   command on the same line). Fixed by inserting ``;`` after ``}`` when
   something else follows.
2. ``(A && B) &`` passed through unchanged — the explicit ``(...)``
   spelling has the identical subshell-wait bug (parens always fork,
   ``&`` or not) but the tokenizer deliberately skipped parenthesised
   content. Fixed by ``_rewrite_parenthesized_background``, which
   rewrites the *inside* of the parens so its own tail backgrounds
   itself (``(A && { B & })``), leaving the explicit subshell in place
   for any `cd`/env isolation it provides.
"""


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


class TestTrailingCommandAfterBackground:
    """`A && B & C` must stay valid bash — a brace group needs a `;`
    before the next command on the same line."""

    def test_simple_trailing_command(self):
        assert rewrite("A && B & C") == "A && { B & }; C"

    def test_realistic_server_start_with_trailing_command(self):
        cmd = "cd /app && node server.js & echo started"
        assert rewrite(cmd) == "cd /app && { node server.js & }; echo started"

    def test_trailing_command_already_valid_not_double_separated(self):
        # Newline, `;`, `&&`, `||`, `|` after `}` are already valid — don't
        # insert a redundant `;`.
        assert rewrite("A && B &\nC") == "A && { B & }\nC"
        assert rewrite("A && B &;C") == "A && { B & };C"
        assert rewrite("A && B & && C") == "A && { B & } && C"
        assert rewrite("A && B & || C") == "A && { B & } || C"
        assert rewrite("A && B & | C") == "A && { B & } | C"

    def test_trailing_command_end_of_string_no_separator(self):
        assert rewrite("A && B &") == "A && { B & }"

    def test_bash_accepts_the_rewritten_syntax(self):
        shutil = pytest.importorskip("shutil")
        if shutil.which("bash") is None:
            pytest.skip("bash not available")
        rewritten = rewrite("cd /app && node server.js & echo started")
        result = subprocess.run(
            ["bash", "-n", "-c", rewritten],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestParenthesisedSubshellBackground:
    """`(A && B) &` has the same subshell-wait bug as `A && B &` — parens
    always fork a subshell, so the subshell waits on B before it (and the
    pipe it inherited) can exit."""

    def test_simple_parenthesised_chain(self):
        assert rewrite("(A && B) &") == "(A && { B & }) &"

    def test_parenthesised_or(self):
        assert rewrite("(A || B) &") == "(A || { B & }) &"

    def test_parenthesised_single_command(self):
        # No chain operator inside — the whole body is the tail.
        assert rewrite("(cmd) &") == "({ cmd & }) &"

    def test_realistic_server_start_in_parens(self):
        cmd = "(cd /app && node server.js) &"
        assert rewrite(cmd) == "(cd /app && { node server.js & }) &"

    def test_already_backgrounded_tail_left_alone(self):
        # The subshell already backgrounds B itself — it won't wait on
        # it either way, so there's nothing to fix.
        assert rewrite("(A && B &) &") == "(A && B &) &"

    def test_parens_without_trailing_background_untouched(self):
        # No outer `&` — the subshell isn't backgrounded, so it's a
        # normal synchronous subshell with no wait4-forever risk.
        assert rewrite("(A && B)") == "(A && B)"

    def test_parens_followed_by_and_and_not_confused(self):
        assert rewrite("(A && B) && C") == "(A && B) && C"

    def test_command_substitution_still_not_rewritten(self):
        cmd = 'echo "$(A && B)" &'
        assert rewrite(cmd) == 'echo "$(A && B)" &'

    def test_bash_accepts_the_rewritten_syntax(self):
        shutil = pytest.importorskip("shutil")
        if shutil.which("bash") is None:
            pytest.skip("bash not available")
        for cmd in [
            "(A && B) &",
            "(cmd) &",
            "(cd /app && node server.js) &",
        ]:
            rewritten = rewrite(cmd)
            result = subprocess.run(
                ["bash", "-n", "-c", rewritten],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (cmd, rewritten, result.stderr)


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
