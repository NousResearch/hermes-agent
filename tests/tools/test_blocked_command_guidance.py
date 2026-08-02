"""Tests for blocked-command recovery guidance (parser-limit + backgrounding)."""

import pytest

from tools.approval import _hardline_block_result, _PARSER_LIMIT_DESCRIPTION, _MALFORMED_EXEC_DESCRIPTION
from tools.terminal_tool import _foreground_background_guidance


class TestParserLimitRecovery:
    def test_parser_limit_block_has_recovery_recipe(self):
        r = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION)
        assert r["approved"] is False
        assert "RECOVERY" in r["message"]
        assert "write_file" in r["message"]
        assert "bash /path/script.sh" in r["message"]

    def test_malformed_exec_block_has_recovery_recipe(self):
        r = _hardline_block_result(_MALFORMED_EXEC_DESCRIPTION)
        assert "RECOVERY" in r["message"]

    def test_real_hardline_blocks_unchanged(self):
        r = _hardline_block_result("recursive delete of root filesystem")
        assert "RECOVERY" not in r["message"]
        assert "unconditional blocklist" in r["message"]


class TestBackgroundGuidanceRecipes:
    def test_ampersand_block_names_exact_call_shape(self):
        msg = _foreground_background_guidance("python3 server.py &")
        assert msg is not None
        assert "WITHOUT the '&'" in msg
        assert "background=true" in msg

    def test_nohup_block_names_exact_call_shape(self):
        msg = _foreground_background_guidance("nohup ./worker.sh > /dev/null 2>&1")
        assert msg is not None
        assert "WITHOUT the wrapper" in msg
        assert "notify_on_complete=true" in msg

    def test_plain_command_unaffected(self):
        assert _foreground_background_guidance("echo hello") is None

    def test_quoted_ampersand_not_flagged(self):
        assert _foreground_background_guidance('git commit -m "a & b"') is None
