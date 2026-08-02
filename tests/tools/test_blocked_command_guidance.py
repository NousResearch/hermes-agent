"""Tests for blocked-command recovery guidance (parser-limit + backgrounding)."""

import pytest

from tools.approval import _hardline_block_result, _PARSER_LIMIT_DESCRIPTION, _MALFORMED_EXEC_DESCRIPTION
from tools.terminal_tool import _foreground_background_guidance


class TestParserLimitRecovery:
    def test_parser_limit_block_saves_payload_and_names_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        cmd = "python3 -c '" + "x = 1; " * 900 + "'"
        r = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, cmd)
        assert r["approved"] is False
        assert "RECOVERY" in r["message"]
        assert "blocked-scripts" in r["message"]
        import re as _re
        m = _re.search(r"saved to (\S+\.sh)", r["message"])
        assert m, r["message"]
        from pathlib import Path
        saved = Path(m.group(1))
        assert saved.exists()
        body = saved.read_text()
        assert cmd in body
        assert body.startswith("#!/bin/bash")
        assert f"bash {saved}" in r["message"]

    def test_save_failure_falls_back_to_manual_recipe(self, monkeypatch):
        import tools.approval as ap
        monkeypatch.setattr(ap, "_save_blocked_payload", lambda c: None)
        r = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, "python3 -c 'x'")
        assert "write_file" in r["message"]
        assert "bash /path/script.sh" in r["message"]

    def test_no_command_falls_back_to_manual_recipe(self):
        r = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION)
        assert "RECOVERY" in r["message"]
        assert "write_file" in r["message"]

    def test_malformed_exec_block_has_recovery_recipe(self):
        r = _hardline_block_result(_MALFORMED_EXEC_DESCRIPTION)
        assert "RECOVERY" in r["message"]

    def test_real_hardline_blocks_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        r = _hardline_block_result("recursive delete of root filesystem", "rm -rf --no-preserve-root /")
        assert "RECOVERY" not in r["message"]
        assert "unconditional blocklist" in r["message"]
        # And nothing was saved for a genuine hardline block.
        assert not (tmp_path / ".hermes" / "cache" / "blocked-scripts").exists()

    def test_old_saved_payloads_cleaned(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        import os
        d = tmp_path / ".hermes" / "cache" / "blocked-scripts"
        d.mkdir(parents=True)
        stale = d / "blocked-1-dead.sh"
        stale.write_text("old")
        os.utime(stale, (1, 1))
        _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, "python3 -c 'y'")
        assert not stale.exists()


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
