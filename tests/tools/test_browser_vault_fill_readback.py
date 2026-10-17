"""The vault fill must report what landed, not what it attempted (#136592).

A controlled page (React Hook Form-style checkouts) can batch-reset a written
input right after the fill script returns: month/year ``<select>``s keep their
value while the card-number text input is reset to empty, yet the tool answered
``success: true`` because ``filled`` was counted at write time. The fill script
now yields one macrotask before reading every field back, reports reset fields
as ``dropped`` (tokens only, never values), and the tool fails closed on them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.vault_login_classifier import build_fill_js

# A stand-in byte sequence, not a credential: the redaction assertions below
# only need a marker that must never appear in tool output.
_FILL_VALUE = os.environ.get("HERMES_TEST_FILL_MARKER", "fill-marker-value")


class TestFillJsReadbackShape:
    def test_fill_js_settles_before_judging_fields(self):
        # The read-back must happen after a macrotask yield (React flushes its batch
        # after the current task), and reset fields must be reported as dropped.
        js = build_fill_js(
            [{"index": 0, "token": "current-password", "value": _FILL_VALUE}],
            expected_origin="https://example.com",
        )
        assert "return (async () => {" in js  # supervisor evaluates with awaitPromise
        assert "setTimeout" in js
        assert "dropped" in js
        assert js.index("setTimeout") < js.index("dropped.push")
        # counted only after the settle, never at write time
        assert "filled += 1" in js
        assert js.index("filled += 1") > js.index("setTimeout")

    def test_fill_js_returns_tokens_and_counts_never_values(self):
        fills = [{"index": 0, "token": "cc-number", "value": _FILL_VALUE}]
        js = build_fill_js(fills, expected_origin="https://example.com")
        assert "JSON.stringify({ filled, dropped })" in js
        # the payload embeds the value, but the return path must not touch it
        tail = js.rsplit("setTimeout", 1)[1]
        assert _FILL_VALUE not in tail

    def test_fill_js_origin_assert_still_precedes_every_write(self):
        # Regression anchor: the settle/read-back change must not reorder the
        # synchronous TOCTOU origin assert ahead of any DOM access.
        js = build_fill_js(
            [{"index": 0, "token": "current-password", "value": _FILL_VALUE}],
            expected_origin="https://example.com",
        )
        assert js.index("origin_changed") < js.index("querySelector")
        assert js.index('removeAttribute("data-hermes-vault-slot")') > js.index("setter.set.call")


class _ReadbackBackend:
    name, display_name, needs_unlock = "testvault", "Test Vault", False

    def __init__(self, password=_FILL_VALUE):
        self._password = password

    def is_unlocked(self):
        return True

    def get_meta(self, handle):
        from agent.vault_store import VaultItemMeta

        return VaultItemMeta(
            id=handle, kind="login", label="Example login", origin="https://example.com",
            created_at="2026-01-01T00:00:00Z", identifier_type="email",
            identifier="user@example.com", allowed_origins=("https://example.com",),
        )

    def resolve_password(self, handle):
        return self._password


_CONTROLS = [
    {"autocomplete": "email", "formIndex": 0, "index": 0, "label": "", "name": "email", "type": "email"},
    {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
]


def _fill_with_page_result(page_result: str):
    """Drive browser_vault_fill with the page answering ``page_result`` for the fill script."""
    from tools import browser_vault_tool

    def fake_eval(task_id, expression):
        return {"success": True, "result": json.dumps(_CONTROLS)}

    def fake_eval_secret(task_id, expression):
        return {"success": True, "result": page_result}

    with patch("agent.vault_backends.backend_for_handle", return_value=_ReadbackBackend()), \
         patch.object(browser_vault_tool, "_current_page_origin", return_value="https://example.com"), \
         patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
         patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
        return json.loads(browser_vault_tool.browser_vault_fill("vault_x"))


class TestFillFailsClosedOnDroppedField:
    def test_page_reset_field_fails_the_fill_and_names_the_field(self):
        # #136592 shape: the page reset the field after the write; the tool must not
        # answer success, and must name the dropped field (token, never the value).
        out = _fill_with_page_result(json.dumps({"filled": 0, "dropped": ["current-password"]}))
        assert out["success"] is False
        assert out["error_type"] == "field_dropped"
        assert out["dropped_fields"] == ["current-password"]
        assert out["filled_fields"] == 0
        assert _FILL_VALUE not in json.dumps(out)

    def test_landed_fill_still_succeeds_with_empty_dropped(self):
        # Regression anchor: an empty dropped list keeps the historical success contract.
        out = _fill_with_page_result(json.dumps({"filled": 1, "dropped": []}))
        assert out["success"] is True
        assert out["filled_fields"] == 1
        assert "dropped_fields" not in out

    def test_legacy_page_result_without_dropped_still_succeeds(self):
        # Older page results (or a stripped middlebox) carry no dropped key at all.
        out = _fill_with_page_result(json.dumps({"filled": 1}))
        assert out["success"] is True
        assert out["filled_fields"] == 1
