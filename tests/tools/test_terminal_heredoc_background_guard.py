"""Regression tests for heredoc-aware background-'&' detection.

Context: ``_foreground_background_guidance`` blocks a foreground command that
looks like it backgrounds a process with ``&`` (so the agent is nudged toward
``terminal(background=true)``). Before scanning, it calls ``_strip_quotes`` to
blank out quoted content so an ``&`` *inside a string* isn't mistaken for the
shell background operator.

Bug: ``_strip_quotes`` documented that it strips "heredoc-style inline text"
but only stripped single/double/backtick quotes — it had no heredoc handling.
So a foreground command carrying a heredoc whose BODY contains a spaced ``&``
was wrongly rejected. Real-world triggers:

- ``osascript <<'EOF' ... set x to "a" & b ... EOF``  (AppleScript concat)
- ``python3 <<'EOF' ... z = a & b ... EOF``            (Python bitwise-and)
- a heredoc body containing literal UI text like ``FaceTime & Privacy``

The fix strips heredoc bodies (``<<EOF``, ``<<-EOF``, ``<<'EOF'``, ``<<"EOF"``)
before the ``&`` scan, so payload ampersands are ignored while a *real*
backgrounding ``&`` at the shell level is still caught.
"""

from tools.terminal_tool import (
    _foreground_background_guidance as guidance,
    _strip_quotes,
)

# Build commands without a literal '&' in this source where convenient, so the
# test file itself never trips a naive scanner. AMP is just an ampersand.
AMP = chr(38)
NL = chr(10)


class TestHeredocBodyAmpersandAllowed:
    """A spaced '&' inside a heredoc body is payload, not backgrounding."""

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

    def test_unquoted_delimiter(self):
        cmd = "cat <<EOF" + NL + "foo " + AMP + " bar" + NL + "EOF"
        assert guidance(cmd) is None

    def test_double_quoted_delimiter(self):
        cmd = 'cat <<"EOF"' + NL + "foo " + AMP + " bar" + NL + "EOF"
        assert guidance(cmd) is None

    def test_dash_delimiter_indented_close(self):
        cmd = "cat <<-EOF" + NL + "\tfoo " + AMP + " bar" + NL + "\tEOF"
        assert guidance(cmd) is None

    def test_literal_ui_text_in_body(self):
        cmd = "cat <<'EOF'" + NL + "About FaceTime " + AMP + " Privacy" + NL + "EOF"
        assert guidance(cmd) is None


class TestRealBackgroundingStillBlocked:
    """A genuine shell-level '&' must still be caught after the fix."""

    def test_trailing_background(self):
        assert guidance("python3 server.py " + AMP) is not None

    def test_inline_background(self):
        assert guidance("sleep 100 " + AMP + " echo done") is not None

    def test_background_after_heredoc(self):
        # A heredoc that itself is backgrounded — the trailing '&' after the
        # closing delimiter is real backgrounding and must still be flagged.
        cmd = "cat <<'EOF' > f.txt" + NL + "payload" + NL + "EOF" + NL + "long_running " + AMP
        assert guidance(cmd) is not None


class TestStripQuotesHeredoc:
    """Direct unit checks on the helper."""

    def test_heredoc_body_removed(self):
        cmd = "osascript <<'EOF'" + NL + 'x ' + AMP + " y" + NL + "EOF"
        stripped = _strip_quotes(cmd)
        # The bare spaced ampersand from the body must not survive.
        assert (" " + AMP + " ") not in stripped

    def test_multiple_heredocs(self):
        cmd = (
            "cat <<'A'" + NL + "one " + AMP + " two" + NL + "A" + NL
            + "cat <<'B'" + NL + "three " + AMP + " four" + NL + "B"
        )
        stripped = _strip_quotes(cmd)
        assert (" " + AMP + " ") not in stripped
