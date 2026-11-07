"""A card fill asks for consent only when the bound page can take the card.

The page is inspected before the prompt (no card number control: refused without asking) and again after
it; only the post-consent inspection's controls may be written. The page is a stand-in for the CDP
supervisor and the card is a dummy value in a temporary local vault.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore, normalize_origin

_CARD = {"card_number": "4111111111111111", "cardholder_name": "Test User", "exp_month": "07",
         "exp_year": "2029", "cvc": "123", "billing_postal_code": "94107"}
_CARD_CONTROLS = [
    {"autocomplete": "cc-number", "index": 0, "type": "text"},
    {"autocomplete": "cc-name", "index": 1, "type": "text"},
]
_BILLING_ONLY = [
    {"autocomplete": "cc-name", "index": 0, "type": "text"},
    {"autocomplete": "postal-code", "index": 1, "type": "text"},
]


class _FakePage:
    """One open tab: answers origin reads and input inspection (recording each inspection's nonce), and
    records every secret-bearing fill script as a write."""

    def __init__(self, href: str, controls: list[dict]):
        self.href, self.controls = href, controls
        self.events: list[str] = []
        self.nonces: list[str] = []
        self.writes: list[str] = []

    def focus_page(self, origin: str, accept=None) -> dict:
        return {"ok": bool(origin) and normalize_origin(self.href) == origin, "url": self.href}

    def evaluate_runtime(self, expression: str) -> dict:
        if expression.strip() == "window.location.href":
            return {"ok": True, "result": self.href}
        if "expectedOrigin" in expression:  # the fill script (build_fill_js)
            self.writes.append(expression)
            return {"ok": True, "result": json.dumps({"filled": 1})}
        self.events.append("inspect")
        self.nonces.append(expression.split("const nonce = ", 1)[1].split(";", 1)[0])
        return {"ok": True, "result": json.dumps(self.controls)}


@pytest.fixture()
def card(tmp_path):
    from agent import redact
    from agent.vault_backends.local import LocalLoginBackend

    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item(kind="payment", label="Test card", origin="https://shop.example", secret=_CARD)
    try:
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch("agent.vault_backends.backend_for_handle", return_value=LocalLoginBackend()):
            yield meta
    finally:
        redact.clear_vault_redaction_values()


def _fill(meta, page: _FakePage, *, decision: str = "accept", controls_after_consent=None) -> dict:
    from tools import browser_vault_tool

    def consent(*args, **kwargs):
        page.events.append("prompt")
        if controls_after_consent is not None:
            page.controls = controls_after_consent
        return decision

    with patch.object(browser_vault_tool, "_ensure_supervisor", return_value=page), \
         patch("tools.approval_prompt.request_elicitation_consent", side_effect=consent):
        return json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))


@pytest.mark.parametrize("controls", [[], _BILLING_ONLY], ids=["processor-frame-only", "billing-fields-only"])
def test_a_page_without_a_card_number_field_is_refused_without_asking(card, controls):
    page = _FakePage("https://shop.example/checkout", controls)
    out = _fill(card, page)
    assert out["success"] is False and out["error_type"] == "no_payment_fields"
    assert "prompt" not in page.events and page.writes == []


def test_a_page_on_another_origin_is_refused_before_inspection_or_asking(card):
    page = _FakePage("https://elsewhere.example/checkout", _CARD_CONTROLS)
    out = _fill(card, page)
    assert out["error_type"] == "origin_mismatch"
    assert page.events == [] and page.writes == []


def test_a_declined_card_is_not_written(card):
    page = _FakePage("https://shop.example/checkout", _CARD_CONTROLS)
    out = _fill(card, page, decision="decline")
    assert out["error_type"] == "payment_declined"
    assert page.events == ["inspect", "prompt"] and page.writes == []


def test_an_approved_card_is_written_to_the_post_consent_inspection_only(card):
    page = _FakePage("https://shop.example/checkout", _CARD_CONTROLS)
    out = _fill(card, page)
    assert out["success"] is True and out["filled_fields"] == 1
    assert page.events == ["inspect", "prompt", "inspect"]
    first, second = page.nonces
    assert first != second
    [write] = page.writes
    assert second in write and first not in write


def test_a_card_form_removed_while_the_prompt_was_open_is_not_written(card):
    page = _FakePage("https://shop.example/checkout", _CARD_CONTROLS)
    out = _fill(card, page, controls_after_consent=[])
    assert out["error_type"] == "no_payment_fields"
    assert page.events == ["inspect", "prompt", "inspect"] and page.writes == []
