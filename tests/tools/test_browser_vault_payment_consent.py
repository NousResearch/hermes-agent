"""Payment consent is requested only for a card-number target on the bound page."""

import json
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore
from tools.registry import registry


_CARD = {"card_number": "4111111111111111", "cardholder_name": "A User", "exp_month": "7",
         "exp_year": "2029", "cvc": "123", "billing_postal_code": "94110"}
_CARD_CONTROLS = [
    {"autocomplete": "cc-number", "index": 0, "type": "text"},
    {"autocomplete": "cc-name", "index": 1, "type": "text"},
]


def _fill(tmp_path, *, page_origin="https://shop.test", controls=_CARD_CONTROLS,
          after_consent=None, decision="accept"):
    from tools import browser_vault_tool as vault

    store = VaultStore(base_dir=tmp_path / "vault")
    card = store.add_item(kind="payment", label="Synthetic card", origin="https://shop.test",
                          secret=_CARD)
    events, nonces, writes = [], [], []
    current_controls = [controls]

    def inspect(task_id, expression):
        events.append("inspect")
        nonces.append(expression.split("const nonce = ", 1)[1].split(";", 1)[0])
        return {"success": True, "result": json.dumps(current_controls[0])}

    def consent(*args, **kwargs):
        events.append("prompt")
        if after_consent is not None:
            current_controls[0] = after_consent
        return decision

    def write(task_id, expression):
        writes.append(expression)
        return {"success": True, "result": json.dumps({"filled": 1})}

    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(vault, "_focus_bound_origin", return_value=None), \
         patch.object(vault, "_current_page_origin", return_value=page_origin), \
         patch.object(vault, "_eval_js", side_effect=inspect), \
         patch.object(vault, "_eval_js_secret", side_effect=write), \
         patch("tools.approval_prompt.request_elicitation_consent", side_effect=consent):
        result = json.loads(registry.dispatch("browser_vault_fill", {"handle": card.id}, task_id="card-test"))
    return result, events, nonces, writes


@pytest.mark.parametrize("controls,origin", [
    ([], "https://shop.test"),  # Only cross-origin processor-frame inputs exist.
    ([{"autocomplete": "cc-name", "index": 0, "type": "text"},
      {"autocomplete": "postal-code", "index": 1, "type": "text"}], "https://shop.test"),
    (_CARD_CONTROLS, "https://other.test"),
], ids=["processor-frame-only", "billing-fields-only", "wrong-origin"])
def test_card_without_a_card_number_on_the_bound_page_never_asks(tmp_path, controls, origin):
    result, events, _, writes = _fill(tmp_path, page_origin=origin, controls=controls)
    assert result["error_type"] == ("origin_mismatch" if origin != "https://shop.test" else "no_payment_fields")
    assert "prompt" not in events and writes == []


@pytest.mark.parametrize("after_consent,decision,expected", [
    (None, "decline", "payment_declined"),
    (None, "accept", "filled"),
    ([], "accept", "no_payment_fields"),
], ids=["declined", "accepted", "form-replaced"])
def test_card_consent_follows_inspection_and_only_post_consent_targets_can_be_written(
        tmp_path, after_consent, decision, expected):
    from agent import redact
    try:
        result, events, nonces, writes = _fill(tmp_path, after_consent=after_consent, decision=decision)
        if expected == "filled":
            assert result["success"] is True
            assert events == ["inspect", "prompt", "inspect"]
            assert len(nonces) == 2 and nonces[0] != nonces[1]
            assert len(writes) == 1 and nonces[1] in writes[0] and nonces[0] not in writes[0]
        else:
            assert result["error_type"] == expected
            assert events == (["inspect", "prompt"] if decision == "decline" else
                              ["inspect", "prompt", "inspect"])
            assert writes == []
    finally:
        redact.clear_vault_redaction_values()
