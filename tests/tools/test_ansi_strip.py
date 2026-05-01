"""Comprehensive tests for ANSI escape sequence stripping (ECMA-48).

The strip_ansi function in tools/ansi_strip.py is the source-level fix for
ANSI codes leaking into the model's context via terminal/execute_code output.
It must strip ALL terminal escape sequences while preserving legitimate text.
"""

from tools.ansi_strip import strip_ansi


class TestStripAnsiBasicSGR:
    """Select Graphic Rendition — the most common ANSI sequences."""

    def test_reset(self):
        assert strip_ansi("\x1b[0m") == ""

    def test_color(self):
        assert strip_ansi("\x1b[31;1m") == ""

    def test_truecolor_semicolon(self):
        assert strip_ansi("\x1b[38;2;255;0;0m") == ""

    def test_truecolor_colon_separated(self):
        """Modern terminals use colon-separated SGR params."""
        assert strip_ansi("\x1b[38:2:255:0:0m") == ""
        assert strip_ansi("\x1b[48:2:0:255:0m") == ""


class TestStripAnsiCSIPrivateMode:
    """CSI sequences with ? prefix (DEC private modes)."""

    def test_cursor_show_hide(self):
        assert strip_ansi("\x1b[?25h") == ""
        assert strip_ansi("\x1b[?25l") == ""

    def test_alt_screen(self):
        assert strip_ansi("\x1b[?1049h") == ""
        assert strip_ansi("\x1b[?1049l") == ""

    def test_bracketed_paste(self):
        assert strip_ansi("\x1b[?2004h") == ""


class TestStripAnsiCSIIntermediate:
    """CSI sequences with intermediate bytes (space, etc.)."""

    def test_cursor_shape(self):
        assert strip_ansi("\x1b[0 q") == ""
        assert strip_ansi("\x1b[2 q") == ""
        assert strip_ansi("\x1b[6 q") == ""


class TestStripAnsiOSC:
    """Operating System Command sequences."""

    def test_bel_terminator(self):
        assert strip_ansi("\x1b]0;title\x07") == ""

    def test_st_terminator(self):
        assert strip_ansi("\x1b]0;title\x1b\\") == ""

    def test_hyperlink_preserves_text(self):
        assert strip_ansi(
            "\x1b]8;;https://example.com\x1b\\click\x1b]8;;\x1b\\"
        ) == "click"


class TestStripAnsiDECPrivate:
    """DEC private / Fp escape sequences."""

    def test_save_restore_cursor(self):
        assert strip_ansi("\x1b7") == ""
        assert strip_ansi("\x1b8") == ""

    def test_keypad_modes(self):
        assert strip_ansi("\x1b=") == ""
        assert strip_ansi("\x1b>") == ""


class TestStripAnsiFe:
    """Fe (C1 as 7-bit) escape sequences."""

    def test_reverse_index(self):
        assert strip_ansi("\x1bM") == ""

    def test_reset_terminal(self):
        assert strip_ansi("\x1bc") == ""

    def test_index_and_newline(self):
        assert strip_ansi("\x1bD") == ""
        assert strip_ansi("\x1bE") == ""


class TestStripAnsiNF:
    """nF (character set selection) sequences."""

    def test_charset_selection(self):
        assert strip_ansi("\x1b(A") == ""
        assert strip_ansi("\x1b(B") == ""
        assert strip_ansi("\x1b(0") == ""


class TestStripAnsiDCS:
    """Device Control String sequences."""

    def test_dcs(self):
        assert strip_ansi("\x1bP+q\x1b\\") == ""


class TestStripAnsi8BitC1:
    """8-bit C1 control characters."""

    def test_8bit_csi(self):
        assert strip_ansi("\x9b31m") == ""
        assert strip_ansi("\x9b38;2;255;0;0m") == ""

    def test_8bit_standalone(self):
        assert strip_ansi("\x9c") == ""
        assert strip_ansi("\x9d") == ""
        assert strip_ansi("\x90") == ""


class TestStripAnsiRealWorld:
    """Real-world contamination scenarios from bug reports."""

    def test_colored_shebang(self):
        """The original reported bug: shebang corrupted by color codes."""
        assert strip_ansi(
            "\x1b[32m#!/usr/bin/env python3\x1b[0m\nprint('hello')"
        ) == "#!/usr/bin/env python3\nprint('hello')"

    def test_stacked_sgr(self):
        assert strip_ansi(
            "\x1b[1m\x1b[31m\x1b[42mhello\x1b[0m"
        ) == "hello"

    def test_ansi_mid_code(self):
        assert strip_ansi(
            "def foo(\x1b[33m):\x1b[0m\n    return 42"
        ) == "def foo():\n    return 42"


class TestStripAnsiPassthrough:
    """Clean content must pass through unmodified."""

    def test_plain_text(self):
        assert strip_ansi("normal text") == "normal text"

    def test_empty(self):
        assert strip_ansi("") == ""

    def test_none(self):
        assert strip_ansi(None) is None

    def test_whitespace_preserved(self):
        assert strip_ansi("line1\nline2\ttab") == "line1\nline2\ttab"

    def test_unicode_safe(self):
        assert strip_ansi("emoji 🎉 and ñ café") == "emoji 🎉 and ñ café"

    def test_backslash_in_code(self):
        code = "path = 'C:\\\\Users\\\\test'"
        assert strip_ansi(code) == code

    def test_square_brackets_in_code(self):
        """Array indexing must not be confused with CSI."""
        code = "arr[0] = arr[31]"
        assert strip_ansi(code) == code
