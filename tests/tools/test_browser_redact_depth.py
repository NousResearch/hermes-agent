"""Page-controlled JSON must not crash browser output redaction."""

import json

from agent.redact import REDACT_WALK_DEPTH_MARKER, REDACT_WALK_MAX_DEPTH
from tools.browser_cdp_tool import _redact_cdp_output
from tools.browser_tool_snapshot import _redact_browser_output

SECRET = "sk-" + "a" * 30


def _nest(value, levels: int):
    for _ in range(levels):
        value = [value]
    return value


def test_shallow_secret_is_still_redacted():
    redacted = _redact_browser_output({"note": SECRET})
    blob = json.dumps(redacted)
    assert SECRET not in blob
    assert REDACT_WALK_DEPTH_MARKER not in blob


def test_browser_output_replaces_a_pathological_tree():
    redacted = _redact_browser_output(_nest(SECRET, REDACT_WALK_MAX_DEPTH + 20))
    blob = json.dumps(redacted)
    assert SECRET not in blob
    assert REDACT_WALK_DEPTH_MARKER in blob


def test_cdp_output_replaces_a_pathological_tree():
    redacted = _redact_cdp_output(_nest({"body": SECRET}, REDACT_WALK_MAX_DEPTH + 20))
    blob = json.dumps(redacted)
    assert SECRET not in blob
    assert REDACT_WALK_DEPTH_MARKER in blob


def test_cdp_binary_leaf_stays_byte_identical():
    payload = "gAAAA" + "A" * 40
    redacted = _redact_cdp_output(
        {"data": payload},
        always_paths=(("data",),),
    )
    assert redacted["data"] == payload
