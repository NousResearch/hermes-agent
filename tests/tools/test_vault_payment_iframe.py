"""Payment fill into hosted card-field iframes (Stripe Elements & co.), #132541.

Card number / CVC on most checkouts live in a provider iframe (js.stripe.com) that the top-level fill
script cannot reach. The vault writes into such an iframe only when it is a known provider AND embedded
directly in the page on the item's bound origin, re-checked inside the iframe at write time.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from agent import redact
from agent.vault_store import VaultStore
from tools import browser_vault_tool as vault
from tools.browser_supervisor import CDPSupervisor
from tools.browser_supervisor_frames import FrameInfo

_CARD = {"card_number": "4111111111111111", "cardholder_name": "A User", "exp_month": "7", "exp_year": "2029",
         "cvc": "987", "billing_postal_code": "94110"}
PAGE = "https://shop.test"
STRIPE = "https://js.stripe.com"

# What Stripe's Payment Element iframe exposes (observed live on a Smorefood checkout).
STRIPE_CONTROLS = [
    {"autocomplete": "cc-number", "name": "number", "index": 0, "type": "text"},
    {"autocomplete": "cc-exp", "name": "expiry", "index": 1, "type": "text"},
    {"autocomplete": "cc-csc", "name": "cvc", "index": 2, "type": "text"},
    {"autocomplete": "billing country", "name": "country", "index": 3, "type": "select"},
    {"autocomplete": "billing postal-code", "name": "postalCode", "index": 4, "type": "text"},
]


@pytest.fixture()
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


@pytest.fixture(autouse=True)
def _clear_redaction():
    yield
    redact.clear_vault_redaction_values()


def _run_fill(store, *, page_controls, frames, frame_controls, consent="accept"):
    """Drive browser_vault_fill with a fake page + provider iframes. Returns (raw_result, writes)."""
    meta = store.add_item(kind="payment", label="Visa", origin=PAGE, secret=_CARD)
    writes = []

    def fake_eval(task_id, expression):
        if "location.href" in expression:
            return {"success": True, "result": f"{PAGE}/checkout"}
        return {"success": True, "result": json.dumps(page_controls)}

    def fake_frame_eval(task_id, frame_id, expression):
        assert _CARD["card_number"] not in expression  # inspection only on this path
        return {"success": True, "result": json.dumps(frame_controls[frame_id])}

    def fake_eval_secret(task_id, expression, frame_id=None):
        writes.append((frame_id, expression))
        return {"success": True, "result": json.dumps({"filled": expression.count('"index"')})}

    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(vault, "_focus_bound_origin", return_value=PAGE), \
         patch.object(vault, "_eval_js", side_effect=fake_eval), \
         patch.object(vault, "_payment_frames", return_value=frames), \
         patch.object(vault, "_frame_eval", side_effect=fake_frame_eval), \
         patch.object(vault, "_eval_js_secret", side_effect=fake_eval_secret), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value=consent):
        raw = vault.browser_vault_fill(meta.id)
    return raw, writes


def test_card_lands_in_stripe_frame_when_page_has_no_card_fields(store):
    raw, writes = _run_fill(store, page_controls=[{"autocomplete": "email", "index": 0, "type": "email"}],
                            frames=[("stripe-frame", STRIPE)], frame_controls={"stripe-frame": STRIPE_CONTROLS})
    out = json.loads(raw)
    assert out["success"] is True
    assert out["fields"] == ["cc-csc", "cc-exp", "cc-number", "postal-code"]
    assert out["frames"] == [STRIPE]
    assert _CARD["card_number"] not in raw and _CARD["cvc"] not in raw
    assert [w[0] for w in writes] == ["stripe-frame"]
    expression = writes[0][1]
    assert _CARD["card_number"] in expression and "07/29" in expression
    # bound to the provider origin AND re-checks, inside the iframe, that the merchant page is its only ancestor
    assert json.dumps(STRIPE) in expression and "location.ancestorOrigins" in expression
    assert json.dumps(PAGE) in expression


def test_declined_consent_writes_nothing(store):
    raw, writes = _run_fill(store, page_controls=[], frames=[("f", STRIPE)],
                            frame_controls={"f": STRIPE_CONTROLS}, consent="decline")
    assert json.loads(raw)["error_type"] == "payment_declined"
    assert writes == []


def test_tokens_taken_by_the_frame_are_not_also_written_to_the_page(store):
    page = [{"label": "Name on card", "index": 0, "type": "text"},
            {"autocomplete": "postal-code", "index": 1, "type": "text"},
            {"autocomplete": "cc-exp-month", "index": 2, "type": "select"}]
    raw, writes = _run_fill(store, page_controls=page, frames=[("f", STRIPE)],
                            frame_controls={"f": STRIPE_CONTROLS})
    page_writes = [e for fid, e in writes if fid is None]
    assert len(page_writes) == 1
    assert '"cc-name"' in page_writes[0]
    assert '"postal-code"' not in page_writes[0] and '"cc-exp-month"' not in page_writes[0]
    assert json.loads(raw)["fields"] == ["cc-csc", "cc-exp", "cc-name", "cc-number", "postal-code"]


def test_frame_refusal_at_write_time_is_reported(store):
    meta = store.add_item(kind="payment", label="Visa", origin=PAGE, secret=_CARD)
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(vault, "_focus_bound_origin", return_value=PAGE), \
         patch.object(vault, "_eval_js", return_value={"success": True, "result": "[]"}), \
         patch.object(vault, "_payment_frames", return_value=[("f", STRIPE)]), \
         patch.object(vault, "_frame_eval",
                      return_value={"success": True, "result": json.dumps(STRIPE_CONTROLS)}), \
         patch.object(vault, "_eval_js_secret", return_value={
             "success": True, "result": json.dumps({"refused": "origin_changed", "found": "https://evil.test"})}), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value="accept"):
        out = json.loads(vault.browser_vault_fill(meta.id))
    assert out["success"] is False and out["error_type"] == "origin_changed"
    assert "Nothing was written" in out["error"]


def test_login_items_never_look_into_frames(store):
    meta = store.add_item(kind="login", label="x", origin=PAGE,
                          secret={"identifier_type": "email", "identifier": "a@b.test", "password": "pw-canary"})
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(vault, "_focus_bound_origin", return_value=PAGE), \
         patch.object(vault, "_eval_js", return_value={"success": True, "result": "[]"}), \
         patch.object(vault, "_payment_frames") as frames:
        vault.browser_vault_fill(meta.id)
    frames.assert_not_called()


# ── _payment_frames: which iframes qualify ──────────────────────────────────


class _FakeSupervisor:
    def __init__(self, frames, contexts):
        self.frames, self.contexts = frames, contexts

    def page_child_oopifs(self):
        return {"ok": True, "frames": self.frames}

    def evaluate_runtime(self, expression, frame_id=None):
        return {"ok": True, "result": json.dumps(self.contexts[frame_id])}


@pytest.mark.parametrize("ctx, url, qualifies", [
    ({"origin": STRIPE, "ancestors": [PAGE]}, "https://js.stripe.com/v3/elements-inner", True),
    ({"origin": STRIPE, "ancestors": [PAGE]}, "", True),  # url not yet known: the in-frame read decides
    ({"origin": STRIPE, "ancestors": ["https://other.test"]}, "", False),  # embedded by another site
    ({"origin": STRIPE, "ancestors": ["https://ads.test", PAGE]}, "", False),  # nested, not a direct child
    ({"origin": "https://evil.test", "ancestors": [PAGE]}, "", False),  # not a card-field provider
    ({"origin": STRIPE, "ancestors": [PAGE]}, "https://evil.test/frame", False),  # url says not a provider
])
def test_payment_frames_requires_known_provider_directly_in_bound_page(ctx, url, qualifies):
    sup = _FakeSupervisor([{"frame_id": "f", "url": url}], {"f": ctx})
    with patch.object(vault, "_ensure_supervisor", return_value=sup):
        assert vault._payment_frames("t", PAGE) == ([("f", STRIPE)] if qualifies else [])


def test_payment_frames_without_supervisor_is_empty():
    with patch.object(vault, "_ensure_supervisor", return_value=None):
        assert vault._payment_frames("t", PAGE) == []


def test_payment_tab_probe_matches_provider_iframes():
    probe = vault._TAB_PROBES["payment"]
    assert 'iframe[src^="https://js.stripe.com/"]' in probe


# ── supervisor: DOM-owned OOPIFs only, eval routed to the frame session ─────


def _supervisor_with(cdp):
    supervisor = CDPSupervisor("test", "ws://localhost")
    supervisor._loop = type("Loop", (), {"is_running": lambda self: True})()
    supervisor._active = True
    supervisor._page_session_id = "page-session"
    supervisor._cdp = cdp
    return supervisor


def test_page_child_oopifs_ignores_frames_owned_by_other_tabs():
    async def cdp(method, params=None, *, session_id=None, timeout=10):
        assert session_id == "page-session"
        if method == "DOM.getDocument":
            return {"result": {"root": {"nodeId": 1}}}
        if method == "DOM.querySelectorAll":
            return {"result": {"nodeIds": [2, 3]}}
        if method == "DOM.describeNode":
            return {"result": {"node": {2: {"frameId": "mine"}, 3: {"frameId": "same-process"}}[params["nodeId"]]}}
        return {"result": {}}

    sup = _supervisor_with(cdp)
    sup._frames["mine"] = FrameInfo("mine", "https://js.stripe.com/v3/x", "", None, True, "mine-session")
    sup._frames["same-process"] = FrameInfo("same-process", "", "", None, False, None)
    sup._frames["other-tab"] = FrameInfo("other-tab", "https://js.stripe.com/v3/x", "", None, True, "other-session")
    with patch("tools.browser_supervisor._schedule", side_effect=lambda coro, loop, **kw: asyncio.run(coro)):
        result = sup.page_child_oopifs()
    assert result == {"ok": True, "frames": [{"frame_id": "mine", "url": "https://js.stripe.com/v3/x"}]}


def test_evaluate_runtime_frame_id_uses_the_iframe_session():
    seen = []

    async def cdp(method, params=None, *, session_id=None, timeout=10):
        seen.append(session_id)
        return {"result": {"result": {"type": "number", "value": 1}}}

    sup = _supervisor_with(cdp)
    sup._frames["f"] = FrameInfo("f", "", "", None, True, "frame-session")
    with patch("tools.browser_supervisor._schedule", side_effect=lambda coro, loop, **kw: asyncio.run(coro)):
        assert sup.evaluate_runtime("1", frame_id="f")["ok"]
        assert sup.evaluate_runtime("1", frame_id="gone") == {"ok": False, "error": "iframe session is no longer attached"}
    assert seen == ["frame-session"]
