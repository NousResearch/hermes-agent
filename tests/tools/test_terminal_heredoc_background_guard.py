"""Regression tests for conservative heredoc-aware background-'&' detection.

Context: ``_foreground_background_guidance`` blocks a foreground command that
looks like it backgrounds a process with ``&`` (so the agent is nudged toward
``terminal(background=true)``). Before scanning, it calls ``_strip_quotes`` to
blank out quoted content so an ``&`` *inside a string* isn't mistaken for the
shell background operator.

Bug (#63788): ``_strip_quotes`` documented that it strips "heredoc-style
inline text" but had no heredoc handling, so a foreground command carrying a
heredoc whose BODY contains a spaced ``&`` was wrongly rejected. Real-world
triggers:

- ``osascript <<'EOF' ... set x to "a" & b ... EOF``  (AppleScript concat)
- ``python3 <<'EOF' ... z = a & b ... EOF``            (Python bitwise-and)

The fix masks heredoc bodies via ``tools.shell_heredoc`` — conservatively.
The guard may ignore ampersands only in quoted heredoc bodies sent to known
non-shell interpreters. Unknown, expandable (unquoted delimiter), compound,
nested, or shell-consumed bodies stay visible so process-management guidance
cannot be bypassed: a false positive on exotic syntax is acceptable, hiding a
real background operator is not.
"""

from tools.shell_heredoc import strip_inert_heredoc_bodies
from tools.terminal_tool import (
    _foreground_background_guidance as guidance,
    _strip_quotes,
)

# Build commands without a literal '&' in this source where convenient, so the
# test file itself never trips a naive scanner. AMP is just an ampersand.
AMP = chr(38)
NL = chr(10)


class TestInertQuotedHeredocPayloadAllowed:
    """A spaced '&' inside a quoted, inert heredoc body is payload."""

    def test_applescript_string_concat(self):
        cmd = (
            "osascript <<'EOF'" + NL
            + 'set out to "count " ' + AMP + " (count of items)" + NL
            + "EOF"
        )
        assert guidance(cmd) is None

    def test_python_bitwise_and(self):
        cmd = "python3 <<'EOF'" + NL + "z = a " + AMP + " b" + NL + "print(z)" + NL + "EOF"
        assert guidance(cmd) is None

    def test_cat_literal_ui_text_in_body(self):
        cmd = "cat <<'EOF'" + NL + "About FaceTime " + AMP + " Privacy" + NL + "EOF"
        assert guidance(cmd) is None

    def test_double_quoted_delimiter(self):
        cmd = 'cat <<"EOF"' + NL + "foo " + AMP + " bar" + NL + "EOF"
        assert guidance(cmd) is None

    def test_dash_delimiter_tab_indented_close(self):
        cmd = "cat <<-'EOF'" + NL + "\tfoo " + AMP + " bar" + NL + "\tEOF"
        assert guidance(cmd) is None

    def test_quoted_delimiter_with_punctuation(self):
        cmd = (
            "python3 - <<'END.X'" + NL
            + "mode = current " + AMP + " mask" + NL
            + "END.X"
        )
        assert guidance(cmd) is None

    def test_multiple_quoted_heredocs_on_one_opener(self):
        cmd = (
            "python3 - <<'A' 3<<'B'" + NL
            + "one " + AMP + " two" + NL
            + "A" + NL
            + "three " + AMP + " four" + NL
            + "B"
        )
        assert guidance(cmd) is None

    def test_env_prefix_and_interpreter_path(self):
        cmd = (
            "FOO=1 env /usr/bin/python3.12 - <<'PY'" + NL
            + "x = a " + AMP + " b" + NL
            + "PY"
        )
        assert guidance(cmd) is None


class TestUnsafeHeredocPayloadRemainsVisible:
    """Bodies that can execute (or can't be proven inert) stay scanned.

    These are deliberate false positives: the model quotes the delimiter or
    splits the command, rather than the guard risking a bypass.
    """

    def test_unquoted_delimiter_is_still_scanned(self):
        # Unquoted bodies undergo shell expansion — $(...) inside would run.
        cmd = "cat <<EOF" + NL + "foo " + AMP + " bar" + NL + "EOF"
        assert guidance(cmd) is not None

    def test_unquoted_command_substitution_is_still_scanned(self):
        cmd = (
            "cat <<EOF" + NL
            + "$(nohup sleep 10 >/dev/null 2>" + AMP + "1 " + AMP + ")" + NL
            + "EOF"
        )
        assert guidance(cmd) is not None

    def test_shell_interpreter_payload_is_still_scanned(self):
        cmd = (
            "bash <<'EOF'" + NL
            + "nohup sleep 10 >/dev/null 2>" + AMP + "1 " + AMP + NL
            + "EOF"
        )
        assert guidance(cmd) is not None

    def test_python_elsewhere_does_not_authorize_bash_heredoc(self):
        cmd = (
            "python3 -c 'pass'; bash <<'EOF'" + NL
            + "nohup sleep 10 " + AMP + NL
            + "EOF"
        )
        assert guidance(cmd) is not None

    def test_pipeline_python_does_not_authorize_bash_heredoc(self):
        cmd = (
            "bash <<'EOF' | python3" + NL
            + "nohup sleep 10 " + AMP + NL
            + "EOF"
        )
        assert guidance(cmd) is not None

    def test_nested_substitution_does_not_authorize_heredoc(self):
        cmd = (
            "python3 -c $(bash <<'SH'" + NL
            + "nohup sleep 100 >/dev/null 2>" + AMP + "1 " + AMP + NL
            + "printf pass" + NL
            + "SH" + NL
            + ")"
        )
        assert guidance(cmd) is not None


class TestInactiveMarkersCannotHideShellTail:
    """Fake '<<' markers must not swallow a later real background operator."""

    def test_marker_in_comment_does_not_hide_background_command(self):
        cmd = ": # <<EOF" + NL + "nohup sleep 10 " + AMP
        assert guidance(cmd) is not None

    def test_marker_in_multiline_quote_does_not_hide_background_command(self):
        cmd = "printf '<<EOF" + NL + "literal'" + NL + "sleep 100 " + AMP
        assert guidance(cmd) is not None

    def test_here_string_does_not_hide_background_command(self):
        cmd = "cat <<<EOF" + NL + "nohup sleep 10 " + AMP
        assert guidance(cmd) is not None

    def test_unterminated_heredoc_keeps_everything_visible(self):
        cmd = "python3 - <<'PY'" + NL + "sleep 100 " + AMP
        assert guidance(cmd) is not None

    def test_line_continuation_keeps_opener_background_visible(self):
        cmd = (
            "python3 - <<'PY' \\" + NL
            + "  >/dev/null " + AMP + NL
            + "print('ok')" + NL
            + "PY"
        )
        assert guidance(cmd) is not None


class TestRealBackgroundingStillBlocked:
    """A genuine shell-level '&' must still be caught after the fix."""

    def test_trailing_background(self):
        assert guidance("python3 server.py " + AMP) is not None

    def test_inline_background(self):
        assert guidance("sleep 100 " + AMP + " echo done") is not None

    def test_background_on_heredoc_opener(self):
        cmd = "python3 - <<'PY' " + AMP + NL + "print('ok')" + NL + "PY"
        assert guidance(cmd) is not None

    def test_background_after_heredoc(self):
        # A real backgrounding '&' AFTER the closing delimiter is still caught.
        cmd = (
            "python3 - <<'PY'" + NL
            + "print('ok')" + NL
            + "PY" + NL
            + "long_running " + AMP
        )
        assert guidance(cmd) is not None

    def test_background_after_cat_heredoc_redirect(self):
        cmd = (
            "cat <<'EOF' > f.txt" + NL + "payload" + NL + "EOF" + NL
            + "long_running " + AMP
        )
        assert guidance(cmd) is not None


class TestStripHelpers:
    """Direct unit checks on the masking helpers."""

    def test_inert_body_removed_shell_tail_preserved(self):
        cmd = (
            "python3 - <<'PY'" + NL
            + "x = left " + AMP + " right" + NL
            + "PY" + NL
            + "sleep 10 " + AMP
        )
        stripped = _strip_quotes(cmd)
        assert "x = left " + AMP + " right" not in stripped
        assert "sleep 10 " + AMP in stripped

    def test_masking_preserves_line_structure(self):
        cmd = (
            "python3 - <<'PY'" + NL
            + "a " + AMP + " b" + NL
            + "c " + AMP + " d" + NL
            + "PY" + NL
            + "echo done"
        )
        assert strip_inert_heredoc_bodies(cmd).count(NL) == cmd.count(NL)

    def test_multiple_sequential_heredocs(self):
        cmd = (
            "cat <<'A'" + NL + "one " + AMP + " two" + NL + "A" + NL
            + "cat <<'B'" + NL + "three " + AMP + " four" + NL + "B"
        )
        stripped = strip_inert_heredoc_bodies(cmd)
        assert (" " + AMP + " ") not in stripped

    def test_ambiguous_input_returned_unchanged(self):
        # Unparseable '<<' token → fail closed, identical string back.
        cmd = "cat <<" + NL + "text " + AMP + " more"
        assert strip_inert_heredoc_bodies(cmd) == cmd

    def test_delimiter_requires_exact_terminator_line(self):
        # Indented terminator doesn't close a normal heredoc; the heredoc is
        # unterminated → everything stays visible.
        cmd = (
            "python3 - <<'PY'" + NL
            + "payload " + AMP + " text" + NL
            + "\tPY"
        )
        assert strip_inert_heredoc_bodies(cmd) == cmd

    def test_dash_heredoc_space_indented_line_is_body(self):
        # For <<- only TABS are stripped before terminator comparison. A
        # space-indented " PY" line does NOT close the heredoc (verified
        # against real bash), so the following "sleep 7 &" line is still BODY
        # text — inert data that cat prints — and masking it is correct.
        cmd = (
            "cat <<-'PY'" + NL
            + "payload" + NL
            + " PY" + NL
            + "sleep 7 " + AMP + NL
            + "PY" + NL
            + "echo after"
        )
        stripped = strip_inert_heredoc_bodies(cmd)
        assert "sleep 7 " + AMP not in stripped
        assert "echo after" in stripped
