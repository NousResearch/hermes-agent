"""Tests for the H4 bash-quoting-error hint detector.

When bash itself fails to parse a command (typically an apostrophe inside a
single-quoted JSON body in a curl call), it exits 2 with a recognizable
message. The terminal tool surfaces an actionable hint pointing at the new
`http` tool so the model stops retrying the same broken quoting.
"""

from tools.terminal_tool import (
    _BASH_QUOTING_ERROR_MARKERS,
    _detect_bash_quoting_error,
)


class TestDetectBashQuotingError:
    def test_unexpected_eof_returns_hint(self):
        # The exact string we observed in /home/rkt2/.hermes-paperclip/logs.
        out = "/usr/bin/bash: eval: line 3: unexpected EOF while looking for matching ''"
        hint = _detect_bash_quoting_error(out, returncode=2)
        assert hint is not None
        assert "http" in hint  # points at the new tool
        assert "apostrophe" in hint.lower() or "quote" in hint.lower()

    def test_eof_variant_without_unexpected_returns_hint(self):
        out = "bash: line 1: EOF while looking for matching '\""
        hint = _detect_bash_quoting_error(out, returncode=2)
        assert hint is not None

    def test_syntax_error_near_unexpected_token_returns_hint(self):
        out = "bash: -c: line 0: syntax error near unexpected token `)`"
        hint = _detect_bash_quoting_error(out, returncode=2)
        assert hint is not None

    def test_nonzero_exit_without_marker_returns_none(self):
        # `false`, `grep` no-match, etc. — exit 1 or other codes. Never trigger.
        assert _detect_bash_quoting_error("permission denied", returncode=1) is None
        assert _detect_bash_quoting_error("not found", returncode=127) is None

    def test_exit_zero_with_marker_in_output_returns_none(self):
        # If a real command happens to print the marker text in stdout but
        # exits 0, we must NOT fire the hint — the command succeeded.
        out = "echo unexpected EOF while looking for matching"
        assert _detect_bash_quoting_error(out, returncode=0) is None

    def test_empty_output_returns_none(self):
        assert _detect_bash_quoting_error("", returncode=2) is None

    def test_exit_2_without_marker_returns_none(self):
        # bash exit 2 is also used for `usage` errors in many programs.
        # Without the parser-error marker we don't claim quoting was the cause.
        assert _detect_bash_quoting_error("usage: foo [-x]", returncode=2) is None

    def test_hint_advertises_http_tool_first(self):
        out = "bash: -c: line 0: unexpected EOF while looking for matching '"
        hint = _detect_bash_quoting_error(out, returncode=2)
        assert hint is not None
        # The whole point of this hint is to redirect the model to `http`.
        assert "http" in hint.lower()
        # And it should explicitly tell the model NOT to retry blindly.
        assert "do not just retry" in hint.lower() or "do not retry" in hint.lower()

    def test_all_documented_markers_actually_fire(self):
        """If we list a marker in the constant, the detector must match on it."""
        for marker in _BASH_QUOTING_ERROR_MARKERS:
            out = f"bash: -c: line 0: {marker} something"
            hint = _detect_bash_quoting_error(out, returncode=2)
            assert hint is not None, f"marker {marker!r} did not trigger detector"
