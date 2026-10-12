"""Vault-fill regression using synthetic credentials and mocked browser calls."""
import json
from unittest.mock import patch

import pytest

from agent import redact
from agent.vault_store import VaultStore
from tools import browser_vault_tool


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    redact.clear_vault_redaction_values()
    yield
    redact.clear_vault_redaction_values()


def test_payment_fill_preserves_amounts_dates_and_names(isolated_registry, tmp_path):
    store = VaultStore(base_dir=tmp_path / "vault")
    card = {"card_number": "4111111111111111", "cvc": "683",
            "exp_month": "10", "exp_year": "2029",
            "cardholder_name": "D", "billing_postal_code": "10001"}
    item = store.add_item(kind="payment", label="Test card", origin="https://shop.test", secret=card)
    controls = [{"autocomplete": token, "index": index, "type": "text"}
                for index, token in enumerate(("cc-number", "cc-csc", "cc-exp-month", "cc-exp-year", "cc-name"))]

    def inspect(task_id, expression):
        if "location.href" in expression:
            return {"success": True, "result": "https://shop.test/checkout"}
        return {"success": True, "result": json.dumps(controls)}

    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(browser_vault_tool, "_focus_bound_origin", return_value=None) as focus, \
         patch.object(browser_vault_tool, "_eval_js", side_effect=inspect), \
         patch.object(browser_vault_tool, "_eval_js_secret", return_value={"success": True, "result": json.dumps({"filled": 5})}), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value="accept"):
        result = json.loads(browser_vault_tool.browser_vault_fill(item.id))
    focus.assert_called_once()
    assert result["success"]
    ordinary = "USD 12; 1200 requests; 2026-12-02; year 2029; ID 12001; Example Customer; 1683; $683.245"
    ordinary += "; " + "; ".join(card[field] for field in ("exp_month", "exp_year", "billing_postal_code", "cardholder_name"))
    assert redact.redact_registered_vault_values(ordinary) == ordinary
    assert card["card_number"] not in redact.redact_registered_vault_values("DOM value=" + card["card_number"])
    assert redact.redact_registered_vault_values(card["cvc"]) == card["cvc"]
    assert card["cvc"] not in redact.redact_registered_vault_values("CVC=" + card["cvc"])
    for field in ("card_number", "cvc"):
        assert card[field] not in json.dumps(result)


def test_short_actual_password_remains_protected_inside_text(isolated_registry):
    redact.register_vault_redaction_value("XY")
    assert "XY" not in redact.redact_registered_vault_values("value=AXYZ")


def test_real_otp_fill_registers_context_only(isolated_registry):
    controls = [{"autocomplete": "one-time-code", "type": "text", "index": 0}]
    with patch.object(browser_vault_tool, "_focus_bound_origin"), \
         patch.object(browser_vault_tool, "_current_page_origin", return_value="https://login.test"), \
         patch.object(browser_vault_tool, "_eval_js", return_value={"success": True, "result": json.dumps(controls)}), \
         patch.object(browser_vault_tool, "_eval_js_secret", return_value={"success": True, "result": json.dumps({"filled": 1})}), \
         patch("agent.vault_backends.unlock.get_code_prompt_callback", return_value=lambda *args: "743821"), \
         patch("agent.vault_backends.unlock.can_prompt_here", return_value=True):
        result = json.loads(browser_vault_tool.browser_vault_enter_code())
    assert result["success"]
    assert "743821" not in json.dumps(result)
    assert redact.redact_registered_vault_values("743821 requests") == "743821 requests"
    assert "743821" not in redact.redact_registered_vault_values("OTP:743821")


@pytest.mark.parametrize("raise_error", [False, True])
def test_otp_fill_errors_are_locally_scrubbed(isolated_registry, raise_error):
    controls = [{"autocomplete": "one-time-code", "type": "text", "index": 0}]
    failure = RuntimeError("failed value 743821") if raise_error else None
    with patch.object(browser_vault_tool, "_focus_bound_origin"), \
         patch.object(browser_vault_tool, "_current_page_origin", return_value="https://login.test"), \
         patch.object(browser_vault_tool, "_eval_js", return_value={"success": True, "result": json.dumps(controls)}), \
         patch.object(browser_vault_tool, "_eval_js_secret", side_effect=failure, return_value={"success": False, "error": "failed value 743821"}), \
         patch("agent.vault_backends.unlock.get_code_prompt_callback", return_value=lambda *args: "743821"), \
         patch("agent.vault_backends.unlock.can_prompt_here", return_value=True):
        raw = browser_vault_tool.browser_vault_enter_code()
    assert not json.loads(raw)["success"]
    assert "743821" not in raw
