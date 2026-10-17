"""Persisted previews keep a result's metadata visible (#126444)."""

import json

from tools.tool_result_storage import _preview_with_metadata


def test_terminal_style_result_keeps_exit_code_and_tail():
    payload = {
        "output": "x" * 5000 + "\nTAIL_MARKER",
        "exit_code": 1,
        "error": "command failed",
        "hint": "check the command",
    }
    preview, has_more = _preview_with_metadata(json.dumps(payload), max_chars=1500)
    assert '"exit_code": 1' in preview
    assert '"hint": "check the command"' in preview
    assert '"error": "command failed"' in preview
    assert "TAIL_MARKER" in preview
    assert has_more is True


def test_metadata_only_result_still_reachable():
    payload = {"output": "y" * 4000, "exit_code": 0, "error": None}
    preview, _ = _preview_with_metadata(json.dumps(payload), max_chars=1500)
    assert '"exit_code": 0' in preview


def test_plain_text_falls_back_to_head_preview():
    text = "a" * 5000
    preview, has_more = _preview_with_metadata(text, max_chars=1500)
    assert preview == text[:1500]
    assert has_more is True


def test_small_result_falls_back_to_head_preview():
    payload = {"output": "short", "exit_code": 0}
    preview, has_more = _preview_with_metadata(json.dumps(payload), max_chars=1500)
    assert preview == json.dumps(payload)
    assert has_more is False
