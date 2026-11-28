"""A guard refusal in write_file/patch must reach the logs (#136835).

Before the fix, every pre-write guard (sensitive path, binary document,
protected instruction, approval, cross-profile, and the read_file-display-text
guard) returned ``tool_error`` to the caller silently: the tool call reported
failure while agent.log/errors.log got no line, so a refused write was
invisible after the fact and had to be reproduced by hand.
"""

import json
import logging

import pytest

from tools.file_tools import patch_tool, write_file_tool


# read_file's display format is "LINE_NUM|CONTENT"; >=2 consecutive numbered
# lines trips _looks_like_read_file_line_numbered_content.
_READ_FILE_DISPLAY = "1|alpha\n2|beta\n3|gamma\n"


def _refusal_payload(result: str) -> dict:
    payload = json.loads(result)
    assert payload.get("error"), f"expected a guard refusal, got: {payload}"
    return payload


class TestWriteFileRefusalLogging:
    def test_read_file_display_text_refusal_is_logged(self, caplog, tmp_path):
        target = str(tmp_path / "SKILL.md")
        with caplog.at_level(logging.WARNING, logger="tools.file_tools"):
            result = write_file_tool(target, _READ_FILE_DISPLAY)
        _refusal_payload(result)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                    and "write_file refused" in r.getMessage()]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "write_file refused" in message
        assert target in message
        assert "read_file display text" in message

    def test_sensitive_path_refusal_is_logged(self, caplog):
        target = "/etc/shadow"
        with caplog.at_level(logging.WARNING, logger="tools.file_tools"):
            result = write_file_tool(target, "harmless\n")
        _refusal_payload(result)
        assert any("write_file refused" in r.getMessage() and target in r.getMessage()
                   for r in caplog.records), [r.getMessage() for r in caplog.records]

    def test_successful_write_emits_no_refusal_log(self, caplog, tmp_path):
        target = str(tmp_path / "real.txt")
        with caplog.at_level(logging.WARNING, logger="tools.file_tools"):
            result = write_file_tool(target, "legitimate body\n")
        assert not json.loads(result).get("error")
        assert not any("write_file refused" in r.getMessage() for r in caplog.records)


class TestPatchRefusalLogging:
    def test_protected_instruction_patch_refusal_is_logged(self, caplog, tmp_path):
        # AGENTS.md is a protected instruction file: patch is always gated.
        target = str(tmp_path / "AGENTS.md")
        with caplog.at_level(logging.WARNING, logger="tools.file_tools"):
            result = patch_tool(
                mode="replace", path=target,
                old_string="nothing", new_string="changed")
        _refusal_payload(result)
        assert any("patch refused" in r.getMessage() and target in r.getMessage()
                   for r in caplog.records)

    def test_sensitive_path_patch_refusal_is_logged(self, caplog):
        target = "/etc/shadow"
        with caplog.at_level(logging.WARNING, logger="tools.file_tools"):
            result = patch_tool(
                mode="replace", path=target,
                old_string="a", new_string="b")
        _refusal_payload(result)
        assert any("patch refused" in r.getMessage() and target in r.getMessage()
                   for r in caplog.records)
