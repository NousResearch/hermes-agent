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

One gap found in that fix, and one attempted extension reverted (issue
#68915):

1. ``A && B & C`` rewrote to ``A && { B & } C``, a bash syntax error (a
   brace *group* needs a statement terminator before the next command on
   the same line). An initial fix inserted ``;`` after ``}`` — but that
   changes scheduling: originally the whole ``A && B`` compound is
   backgrounded as one job, so ``C`` runs immediately; ``A && { B & };C``
   instead runs ``A`` in the foreground first, delaying ``C`` (measured
   0.01s -> 0.24s in adversarial review). The actual fix is to **skip**
   rewriting that statement entirely, leaving the original (leaky but
   behaviorally correct) bash alone.
2. An attempted extension rewrote the parenthesised spelling of the same
   bug, ``(A && B) &``, by textually rewriting the inside of the parens.
   Reverted: it mis-rewrote ``(( 1+1 )) &`` (bash arithmetic evaluation,
   not two nested subshells) into ``({ ( 1+1 ) & }) &``, which runs
   ``1+1`` as a *command* — a silent semantic corruption, not just a
   syntax error — and would equally mishandle heredocs, backtick
   substitutions, and reserved-word compounds (`while`, `case`, `for`)
   if extended further. See ``TestParenthesisedSubshellUntouched`` below:
   `(...)&` is a deliberate non-goal, not an oversight — the top-level
   shell doesn't block on it either way, and the terminal tool's own
   pipe-inheritance hang risk is handled generically elsewhere
   (``tools/environments/base.py``, issue #8340). What's left unfixed is
   the resource-hygiene cost of a leaked subshell, accepted rather than
   chased with more textual rewriting that can't be made safe in
   general.
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


class TestTrailingCommandAfterBackgroundIsSkipped:
    """`A && B & C` must stay valid bash AND keep its original scheduling
    (C runs immediately, not after A). The only way to guarantee both is
    to not rewrite this shape at all — leave it exactly as written."""

    def test_simple_trailing_command_untouched(self):
        assert rewrite("A && B & C") == "A && B & C"

    def test_realistic_server_start_with_trailing_command_untouched(self):
        cmd = "cd /app && node server.js & echo started"
        assert rewrite(cmd) == cmd

    def test_multiline_only_the_trailing_statement_is_skipped(self):
        # The first line has a same-line trailing command -> skip. The
        # second line ends cleanly -> still rewritten.
        cmd = "A && B & C\nfalse || D &"
        assert rewrite(cmd) == "A && B & C\nfalse || { D & }"

    def test_trailing_command_already_valid_shapes_still_rewritten(self):
        # Newline, `;`, `&&`, `||`, `|` after `&` are already valid
        # continuations -> these ARE rewritten (no scheduling change,
        # since there's no bare trailing command to reorder against).
        assert rewrite("A && B &\nC") == "A && { B & }\nC"
        assert rewrite("A && B &;C") == "A && { B & };C"
        assert rewrite("A && B & && C") == "A && { B & } && C"
        assert rewrite("A && B & || C") == "A && { B & } || C"
        assert rewrite("A && B & | C") == "A && { B & } | C"

    def test_trailing_command_end_of_string_still_rewritten(self):
        assert rewrite("A && B &") == "A && { B & }"

    def test_bash_accepts_the_untouched_original(self):
        shutil = pytest.importorskip("shutil")
        if shutil.which("bash") is None:
            pytest.skip("bash not available")
        cmd = "cd /app && node server.js & echo started"
        rewritten = rewrite(cmd)
        assert rewritten == cmd
        result = subprocess.run(
            ["bash", "-n", "-c", rewritten], capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr


class TestParenthesisedSubshellUntouched:
    """`(...) &` is a deliberate non-goal (see module docstring): textual
    rewriting inside an arbitrary subshell body cannot be made safe in
    general. All of these MUST pass through byte-for-byte unchanged."""

    def test_simple_parenthesised_chain(self):
        assert rewrite("(A && B) &") == "(A && B) &"

    def test_parenthesised_or(self):
        assert rewrite("(A || B) &") == "(A || B) &"

    def test_parenthesised_single_command(self):
        assert rewrite("(cmd) &") == "(cmd) &"

    def test_realistic_server_start_in_parens(self):
        cmd = "(cd /app && node server.js) &"
        assert rewrite(cmd) == cmd

    def test_arithmetic_context_not_mangled(self):
        # CRITICAL regression: `((...))` is arithmetic evaluation, not
        # two nested subshells. A textual parens-matcher can't tell the
        # difference; a prior version of this rewrite turned this into
        # `({ ( 1+1 ) & }) &`, which runs `1+1` as a *command* (bash:
        # "1+1: command not found") instead of evaluating it.
        assert rewrite("(( 1+1 )) &") == "(( 1+1 )) &"

    def test_subshell_with_loop_not_split(self):
        # A textual rewrite would try to background only the loop's
        # tail, splitting a reserved-word compound and producing a
        # syntax error.
        cmd = "(while false; do :; done) &"
        assert rewrite(cmd) == cmd

    def test_subshell_with_heredoc_not_split(self):
        cmd = "(cat <<'EOF'\nhello\nEOF\n) &"
        assert rewrite(cmd) == cmd

    def test_subshell_with_backtick_substitution_not_split(self):
        cmd = "(echo `date`) &"
        assert rewrite(cmd) == cmd

    def test_already_backgrounded_tail_untouched(self):
        assert rewrite("(A && B &) &") == "(A && B &) &"

    def test_parens_without_trailing_background_untouched(self):
        assert rewrite("(A && B)") == "(A && B)"

    def test_parens_followed_by_and_and_not_confused(self):
        assert rewrite("(A && B) && C") == "(A && B) && C"

    def test_command_substitution_not_rewritten(self):
        cmd = 'echo "$(A && B)" &'
        assert rewrite(cmd) == cmd


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
