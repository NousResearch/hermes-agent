"""browser_vault on a plugin browser with no CDP endpoint (Camofox): the registered
``BrowserPageBackend`` holding the task's page carries every vault script, and the secret path never
lets a page or backend error echo the credential."""

import json
from unittest.mock import patch

import pytest

from agent.browser_page_backend import BrowserPageBackend
from agent.vault_store import VaultStore

SECRET = "s3cret-pw"
CONTROLS = [
    {"autocomplete": "email", "formIndex": 0, "index": 0, "label": "", "name": "email", "type": "email"},
    {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
]


class FakePage(BrowserPageBackend):
    name = "fakepage"

    def __init__(self, url):
        self.url, self.calls, self.fail_secret = url, [], False

    def has_page(self, task_id):
        return True

    def evaluate(self, task_id, expression, *, secret):
        self.calls.append((expression, secret))
        if secret and self.fail_secret:
            raise RuntimeError(f"page.evaluate failed: {expression}")
        if "location.href" in expression:
            return self.url
        if SECRET in expression:
            return json.dumps({"filled": 1})
        return json.dumps(CONTROLS)


@pytest.fixture()
def page(tmp_path, monkeypatch):
    from agent import browser_page_registry
    from tools import browser_vault_tool

    def cdp_used(*args, **kwargs):
        raise AssertionError("CDP supervisor used while a page backend holds the task's page")

    monkeypatch.setattr(browser_vault_tool, "_ensure_supervisor", cdp_used)
    fake = FakePage("https://example.com/login")
    browser_page_registry.register_provider(fake)
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item(kind="login", label="Example", origin="https://example.com",
                          secret={"identifier_type": "email", "identifier": "a@example.com", "password": SECRET})
    with patch("agent.vault_store.get_vault_store", return_value=store):
        yield fake, meta
    browser_page_registry._reset_for_tests()


def test_fill_runs_on_the_page_backend_and_refuses_another_origin(page):
    from tools import browser_vault_tool

    fake, meta = page
    raw = browser_vault_tool.browser_vault_fill(meta.id)
    out = json.loads(raw)
    assert out["success"] is True and out["filled_fields"] == 1 and SECRET not in raw
    secret_calls = [expr for expr, secret in fake.calls if SECRET in expr]
    assert len(secret_calls) == 1 and all(secret for expr, secret in fake.calls if SECRET in expr)

    fake.calls.clear()
    fake.url = "https://evil.example/login"
    out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
    assert out["error_type"] == "origin_mismatch"
    assert not any(SECRET in expr for expr, _ in fake.calls)


def test_secret_failure_never_echoes_the_script(page):
    from tools import browser_vault_tool

    fake, meta = page
    fake.fail_secret = True
    raw = browser_vault_tool.browser_vault_fill(meta.id)
    out = json.loads(raw)
    assert out["success"] is False and SECRET not in raw


CARD = {"card_number": "4111111111111111", "cardholder_name": "Test User", "exp_month": "07",
        "exp_year": "2029", "cvc": "123"}
CARD_CONTROLS = [
    {"autocomplete": "cc-number", "index": 0, "type": "text"},
    {"autocomplete": "cc-name", "index": 1, "type": "text"},
    {"autocomplete": "cc-exp-month", "index": 2, "type": "text"},
    {"autocomplete": "cc-exp-year", "index": 3, "type": "text"},
    {"autocomplete": "cc-csc", "index": 4, "type": "text"},
]
CARD_PROBE = "input[autocomplete^=cc-]"


class FakeCardPage(BrowserPageBackend):
    """A plugin page holding a checkout; ``card_form=False`` makes the payment tab probe answer false."""

    name = "fakepage"

    def __init__(self, url, card_form=True):
        self.url, self.card_form, self.calls = url, card_form, []

    def has_page(self, task_id):
        return True

    def evaluate(self, task_id, expression, *, secret):
        self.calls.append((expression, secret))
        if "location.href" in expression:
            return self.url
        if CARD["card_number"] in expression:
            return json.dumps({"filled": len(CARD_CONTROLS)})
        if CARD_PROBE in expression:
            return self.card_form
        return json.dumps(CARD_CONTROLS)


@pytest.fixture()
def card_page(tmp_path, monkeypatch):
    from agent import browser_page_registry
    from tools import browser_vault_tool

    def cdp_used(*args, **kwargs):
        raise AssertionError("CDP supervisor used while a page backend holds the task's page")

    monkeypatch.setattr(browser_vault_tool, "_ensure_supervisor", cdp_used)
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item(kind="payment", label="Card", secret=CARD, origin_match="any_site")
    with patch("agent.vault_store.get_vault_store", return_value=store):
        yield meta
    browser_page_registry._reset_for_tests()


def test_any_site_card_fills_the_page_backend_current_page_after_consent(card_page):
    """A card saved for any site works on a plugin browser (no CDP) like an exact-origin item does: the
    prompt names the page's origin and the one secret script runs on that page, pinned to that origin."""
    from agent import browser_page_registry
    from tools import browser_vault_tool

    fake = FakeCardPage("https://shop.example/checkout")
    browser_page_registry.register_provider(fake)
    with patch("tools.approval_prompt.request_elicitation_consent", return_value="accept") as consent:
        raw = browser_vault_tool.browser_vault_fill(card_page.id)
    out = json.loads(raw)
    assert out["success"] is True and out["origin"] == "https://shop.example"
    [call] = consent.call_args_list
    assert call.args[0] == "Fill payment card 'Card' on https://shop.example"
    secret_calls = [expr for expr, secret in fake.calls if CARD["card_number"] in expr]
    assert len(secret_calls) == 1 and '"https://shop.example"' in secret_calls[0]
    assert all(secret for expr, secret in fake.calls if CARD["card_number"] in expr)
    assert CARD["card_number"] not in raw and CARD["cvc"] not in raw


def test_any_site_card_on_a_page_backend_without_a_card_form_is_refused_before_asking(card_page):
    from agent import browser_page_registry
    from tools import browser_vault_tool

    fake = FakeCardPage("https://shop.example/checkout", card_form=False)
    browser_page_registry.register_provider(fake)
    with patch("tools.approval_prompt.request_elicitation_consent", return_value="accept") as consent:
        out = json.loads(browser_vault_tool.browser_vault_fill(card_page.id))
    assert out["error_type"] == "no_card_form_on_current_tab"
    consent.assert_not_called()
    assert not any(CARD["card_number"] in expr for expr, _ in fake.calls)
