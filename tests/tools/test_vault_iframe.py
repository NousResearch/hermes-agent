"""Regression for #120382: exact relying-party and child-frame origin checks."""
import json
from unittest.mock import patch

from agent.vault_store import VaultStore
from tools import browser_vault_tool as vault


def test_cross_origin_login_requires_explicit_pair_consent_and_keeps_password_blind(tmp_path):
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item("login", "site", {"identifier": "a@b.test", "password": "iframe-only-canary",
                                             "identifier_type": "email"}, origin="https://site.test")
    controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 0,
                 "name": "password", "type": "password"}]
    expressions = []

    def frame_eval(task, frame, expression):
        assert frame == "child"
        if expression == "location.href":
            return {"success": True, "result": "https://identity.test/login"}
        if "iframe-only-canary" in expression:
            expressions.append(expression)
            return {"success": True, "result": {"filled": 1}}
        return {"success": True, "result": controls}

    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(vault, "_focus_login_frame", return_value="child"), \
         patch.object(vault, "_frame_eval", side_effect=frame_eval), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value="deny") as consent:
        denied = json.loads(vault.browser_vault_fill(meta.id))
        assert denied["error_type"] == "frame_origin_declined"
        assert not expressions
        consent.return_value = "accept"
        raw = vault.browser_vault_fill(meta.id)
    assert json.loads(raw)["filled_fields"] == 1
    assert "iframe-only-canary" not in raw
    assert len(expressions) == 1
    assert "https://site.test" in expressions[0]
    assert "https://identity.test" in expressions[0]
    assert "location.ancestorOrigins" in expressions[0]


def test_cross_origin_login_refuses_unknown_child_origin(tmp_path):
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item("login", "site", {"identifier": "a@b.test", "password": "canary",
                                             "identifier_type": "email"}, origin="https://site.test")
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(vault, "_focus_login_frame", return_value="child"), \
         patch.object(vault, "_frame_eval", return_value={"success": False}), \
         patch("tools.approval_prompt.request_elicitation_consent") as consent:
        result = json.loads(vault.browser_vault_fill(meta.id))
    assert result["error_type"] == "frame_origin_unknown"
    consent.assert_not_called()
