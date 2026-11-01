"""Local aliases keep password and saved authenticator-code fills exact-origin."""
import json
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore


@pytest.mark.parametrize("origin,permitted", [
    ("https://www.example.com", True),
    ("https://example.com", True),
    ("https://accounts.example.com", False),
    ("https://www.example.com.evil.test", False),
    ("http://www.example.com", False),
    ("https://www.example.com:8443", False),
])
@pytest.mark.parametrize("action", ["fill", "enter_code"])
def test_local_alias_browser_fill_is_exact(tmp_path, origin, permitted, action):
    from tools import browser_vault_tool as tool

    store = VaultStore(tmp_path / "vault")
    meta = store.add_item("login", "Example", {
        "identifier_type": "email", "identifier": "person@example.com",
        "password": "synthetic-password", "otp_secret": "JBSWY3DPEHPK3PXP",
    }, origin="https://example.com")
    store.add_origin(meta.id, "https://www.example.com")
    controls = [{"autocomplete": "current-password" if action == "fill" else "one-time-code",
                 "formIndex": 0, "index": 0, "label": "", "name": "otp",
                 "type": "password" if action == "fill" else "text"}]
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(tool, "_focus_bound_origin", return_value=None), \
         patch.object(tool, "_current_page_origin", return_value=origin), \
         patch.object(tool, "_eval_js", return_value={"success": True, "result": json.dumps(controls)}), \
         patch.object(tool, "_eval_js_secret", return_value={"success": True, "result": json.dumps({"filled": 1})}) as fill:
        raw = getattr(tool, f"browser_vault_{action}")(meta.id)
    result = json.loads(raw)
    assert result["success"] is permitted
    assert "synthetic-password" not in raw
    assert "JBSWY3DPEHPK3PXP" not in raw
    if permitted:
        assert result["origin"] == origin
        assert origin in fill.call_args.args[1]
    else:
        assert result["error_type"] == "origin_mismatch"
        assert "add-origin" in result["error"]
        assert "approval" in result["error"]
        fill.assert_not_called()
