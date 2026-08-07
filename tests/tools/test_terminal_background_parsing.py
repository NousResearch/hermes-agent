"""Tests for the background-operator parsing helpers in terminal_tool.

Covers:
  _heredoc_declarations       — parse heredoc openers on a single line
  _strip_heredoc_bodies       — blank out heredoc content before & scanning
  _shell_background_operator_positions — locate job-control & in a command
  _contains_shell_background_operator  — boolean wrapper
  _is_final_shell_background_operator  — is the & the last real token?
  _normalize_final_shell_background_operator — strip trailing & cleanly
"""

import pytest

from tools.terminal_tool import (
    _contains_shell_background_operator,
    _heredoc_declarations,
    _is_final_shell_background_operator,
    _normalize_final_shell_background_operator,
    _shell_background_operator_positions,
    _strip_heredoc_bodies,
)


# ---------------------------------------------------------------------------
# _heredoc_declarations
# ---------------------------------------------------------------------------

class TestHeredocDeclarations:
    def test_basic_heredoc(self):
        assert _heredoc_declarations("cat << EOF") == [("EOF", False)]

    def test_dash_heredoc_strips_tabs(self):
        assert _heredoc_declarations("cat <<- EOF") == [("EOF", True)]

    def test_quoted_delimiter_single(self):
        assert _heredoc_declarations("cat << 'EOF'") == [("EOF", False)]

    def test_quoted_delimiter_double(self):
        assert _heredoc_declarations('cat << "EOF"') == [("EOF", False)]

    def test_multiple_heredocs_on_one_line(self):
        result = _heredoc_declarations("cmd <<A && cmd2 <<B")
        assert result == [("A", False), ("B", False)]

    def test_no_heredoc(self):
        assert _heredoc_declarations("echo hello") == []

    def test_herestring_not_matched(self):
        # <<< is a herestring, not a heredoc — must not be parsed as one.
        assert _heredoc_declarations("cat <<< 'foo'") == []

    def test_amp_before_heredoc_not_confused(self):
        # Bit-op or redirect before << must not suppress heredoc detection.
        assert _heredoc_declarations("cmd 2>&1 << EOF") == [("EOF", False)]

    def test_delimiter_in_comment_ignored(self):
        # After a # comment, << is not a heredoc.
        assert _heredoc_declarations("echo hi # << EOF") == []

    def test_delimiter_inside_single_quotes_ignored(self):
        assert _heredoc_declarations("echo '<< EOF'") == []


# ---------------------------------------------------------------------------
# _strip_heredoc_bodies
# ---------------------------------------------------------------------------

class TestStripHeredocBodies:
    def test_single_heredoc_body_blanked(self):
        cmd = "cat << EOF\nhello & world\nEOF"
        result = _strip_heredoc_bodies(cmd)
        # Body line should be replaced with same-length whitespace; delimiter
        # line and opener line must be preserved.
        assert "hello & world" not in result
        assert len(result) == len(cmd)

    def test_amp_in_heredoc_body_not_visible(self):
        # The key regression: C++ bitwise-AND inside a heredoc body was being
        # picked up as a job-control operator before this fix.
        cmd = "g++ - << 'EOF'\nint r = a & b;\nEOF"
        stripped = _strip_heredoc_bodies(cmd)
        assert "&" not in stripped.split("\n")[1]

    def test_command_after_heredoc_preserved(self):
        cmd = "cat << EOF\nsome data\nEOF\necho done &"
        stripped = _strip_heredoc_bodies(cmd)
        assert "echo done &" in stripped

    def test_no_heredoc_unchanged(self):
        cmd = "echo hello && sleep 1 &"
        assert _strip_heredoc_bodies(cmd) == cmd

    def test_tab_stripped_heredoc(self):
        cmd = "cat <<- EOF\n\thello & world\n\tEOF"
        result = _strip_heredoc_bodies(cmd)
        assert "hello & world" not in result

    def test_length_preserved(self):
        cmd = "cmd <<EOF\nline one\nline two\nEOF\necho end"
        assert len(_strip_heredoc_bodies(cmd)) == len(cmd)


# ---------------------------------------------------------------------------
# _shell_background_operator_positions
# ---------------------------------------------------------------------------

class TestShellBackgroundOperatorPositions:
    def test_simple_trailing_amp(self):
        positions = _shell_background_operator_positions("sleep 5 &")
        assert len(positions) == 1

    def test_no_amp(self):
        assert _shell_background_operator_positions("echo hello") == []

    def test_double_amp_not_counted(self):
        assert _shell_background_operator_positions("A && B") == []

    def test_amp_gt_redirect_not_counted(self):
        assert _shell_background_operator_positions("cmd &>/dev/null") == []

    def test_fd_redirect_2_gt_amp_not_counted(self):
        assert _shell_background_operator_positions("cmd 2>&1") == []

    def test_amp_inside_single_quotes_not_counted(self):
        assert _shell_background_operator_positions("echo 'a & b'") == []

    def test_amp_inside_double_quotes_not_counted(self):
        assert _shell_background_operator_positions('echo "a & b"') == []

    def test_amp_after_backslash_not_counted(self):
        assert _shell_background_operator_positions(r"echo a \& b") == []

    def test_amp_in_comment_not_counted(self):
        assert _shell_background_operator_positions("echo hi # start &") == []

    def test_amp_in_heredoc_body_not_counted(self):
        # The original bug: C++ bitwise-AND inside a heredoc was detected.
        cmd = "g++ - << 'EOF'\nint r = a & b;\nEOF"
        assert _shell_background_operator_positions(cmd) == []

    def test_internal_amp_detected(self):
        # Internal & before more commands — two operators.
        positions = _shell_background_operator_positions("A & B &")
        assert len(positions) == 2

    def test_single_final_amp_position_correct(self):
        cmd = "sleep 5 &"
        positions = _shell_background_operator_positions(cmd)
        assert len(positions) == 1
        assert cmd[positions[0]] == "&"

    def test_amp_with_trailing_comment(self):
        # `cmd & # comment` — the & is a real background operator.
        positions = _shell_background_operator_positions("cmd & # comment")
        assert len(positions) == 1

    # --- Regressions flagged in code review ---

    def test_pipe_stderr_operator_not_counted(self):
        # |& is bash's pipe-stderr operator (pipes stdout+stderr to next cmd).
        # It is not a background operator and must not be flagged.
        assert _shell_background_operator_positions("make |& tee build.log") == []

    def test_trailing_pipe_stderr_not_counted(self):
        # A trailing |& must not be confused with a job-control &.
        # Normalizing it would produce `cmd |` — invalid syntax.
        assert _shell_background_operator_positions("cmd |&") == []

    def test_arithmetic_bitwise_and_not_counted(self):
        # $((5&3)) is arithmetic; the & is not a job-control operator.
        assert _shell_background_operator_positions("x=$((5&3))") == []

    def test_arithmetic_bitwise_and_in_expression_not_counted(self):
        assert _shell_background_operator_positions("echo $((a & b))") == []

    def test_case_fallthrough_semicolon_amp_not_counted(self):
        # ;& is the case-statement fallthrough operator, not backgrounding.
        cmd = "case $x in\n  a) echo a;&\n  b) echo b;;\nesac"
        assert _shell_background_operator_positions(cmd) == []

    def test_case_test_next_semicolon_amp_not_counted(self):
        # ;;& tests the next pattern; also not backgrounding.
        cmd = "case $x in\n  a) echo a;;&\n  b) echo b;;\nesac"
        assert _shell_background_operator_positions(cmd) == []

    def test_parallel_jobs_with_wait(self):
        # `job1 & job2 & wait` is a legitimate parallel pattern.
        # All three & are real background operators.
        positions = _shell_background_operator_positions("job1 & job2 & wait")
        assert len(positions) == 2


# ---------------------------------------------------------------------------
# _contains_shell_background_operator
# ---------------------------------------------------------------------------

class TestContainsShellBackgroundOperator:
    def test_true_for_trailing_amp(self):
        assert _contains_shell_background_operator("sleep 5 &") is True

    def test_false_for_double_amp(self):
        assert _contains_shell_background_operator("A && B") is False

    def test_false_for_redirect(self):
        assert _contains_shell_background_operator("cmd &>/dev/null") is False

    def test_false_for_no_amp(self):
        assert _contains_shell_background_operator("echo hello") is False

    def test_false_for_heredoc_body_amp(self):
        cmd = "cat << EOF\na & b\nEOF"
        assert _contains_shell_background_operator(cmd) is False


# ---------------------------------------------------------------------------
# _is_final_shell_background_operator
# ---------------------------------------------------------------------------

class TestIsFinalShellBackgroundOperator:
    def _pos(self, cmd: str) -> int:
        """Return first & position from the positions list."""
        positions = _shell_background_operator_positions(cmd)
        assert positions, f"No & found in: {cmd!r}"
        return positions[0]

    def test_trailing_amp_is_final(self):
        cmd = "sleep 5 &"
        assert _is_final_shell_background_operator(cmd, self._pos(cmd)) is True

    def test_amp_with_trailing_whitespace_is_final(self):
        cmd = "sleep 5 &   "
        assert _is_final_shell_background_operator(cmd, self._pos(cmd)) is True

    def test_amp_with_trailing_comment_is_final(self):
        cmd = "sleep 5 & # start the server"
        assert _is_final_shell_background_operator(cmd, self._pos(cmd)) is True

    def test_internal_amp_is_not_final(self):
        cmd = "A & B"
        pos = _shell_background_operator_positions(cmd)[0]
        assert _is_final_shell_background_operator(cmd, pos) is False

    def test_first_of_two_amps_is_not_final(self):
        cmd = "A & B &"
        positions = _shell_background_operator_positions(cmd)
        assert _is_final_shell_background_operator(cmd, positions[0]) is False
        assert _is_final_shell_background_operator(cmd, positions[1]) is True


# ---------------------------------------------------------------------------
# _normalize_final_shell_background_operator
# ---------------------------------------------------------------------------

class TestNormalizeFinalShellBackgroundOperator:
    def _pos(self, cmd: str) -> int:
        positions = _shell_background_operator_positions(cmd)
        assert positions
        return positions[-1]

    def test_simple_trailing_amp_removed(self):
        cmd = "sleep 5 &"
        result = _normalize_final_shell_background_operator(cmd, self._pos(cmd))
        assert result == "sleep 5"

    def test_amp_with_trailing_whitespace(self):
        cmd = "sleep 5 &   "
        result = _normalize_final_shell_background_operator(cmd, self._pos(cmd))
        assert "&" not in result
        assert result.strip() == "sleep 5"

    def test_amp_with_trailing_comment_preserved(self):
        cmd = "sleep 5 & # background"
        result = _normalize_final_shell_background_operator(cmd, self._pos(cmd))
        assert "&" not in result.split("#")[0]
        assert "# background" in result

    def test_line_continuation_before_amp_removed(self):
        # `cmd \\\n&` — the backslash-newline before & should be cleaned up.
        cmd = "sleep 5 \\\n&"
        result = _normalize_final_shell_background_operator(cmd, self._pos(cmd))
        assert "&" not in result
        assert "\\\n" not in result

    def test_multiline_command_only_last_amp_removed(self):
        cmd = "cd /app \\\n  && python server.py &"
        result = _normalize_final_shell_background_operator(cmd, self._pos(cmd))
        assert result.endswith("python server.py")
        assert "&&" in result  # the && is preserved

    def test_result_does_not_have_trailing_whitespace(self):
        cmd = "sleep 5 &"
        result = _normalize_final_shell_background_operator(cmd, self._pos(cmd))
        assert result == result.rstrip()
