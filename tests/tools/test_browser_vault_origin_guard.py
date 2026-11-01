"""Every secret-bearing browser_vault fill is bound to the item's saved origins.

One behaviour per fill path (password, one-time code, payment card, address, save-then-fill): on a page
whose origin is not one the item was saved for, the tool refuses BEFORE the backend resolves any secret
value and before anything is written into the page; on a saved origin the fill goes ahead and the
in-page write is pinned to that origin.

The one explicit exception is a payment card the user saved for any site: it is filled only on the https
page whose exact origin the user confirms for that fill, and the in-page write is pinned to that origin.

The page is a stand-in for the CDP supervisor (current URL + inspected controls; it records every
secret-bearing evaluation). Secrets are dummy values in a temporary local vault.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.vault_backends.local import LocalLoginBackend
from agent.vault_store import VaultStore, normalize_origin

_PASSWORD = "dummy-password-7"
_OTP_SEED = "JBSWY3DPEHPK3PXP"  # base32 test seed, not a real account
_CARD = {"card_number": "4111111111111111", "cardholder_name": "Test User", "exp_month": "07",
         "exp_year": "2029", "cvc": "123", "billing_postal_code": "94107"}
_ADDRESS = {"address_line1": "1 Test Way", "city": "Testville", "state": "CA", "postal_code": "94107",
            "country": "US"}

_LOGIN_CONTROLS = [
    {"autocomplete": "username", "formIndex": 0, "index": 0, "label": "", "name": "user", "type": "text"},
    {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
]
_OTP_CONTROLS = [{"autocomplete": "one-time-code", "formIndex": 0, "index": 0, "label": "Authentication code",
                  "name": "otp", "type": "text"}]
_CARD_CONTROLS = [
    {"autocomplete": "cc-number", "index": 0, "type": "text"},
    {"autocomplete": "cc-exp", "index": 1, "type": "text"},
    {"autocomplete": "cc-csc", "index": 2, "type": "text"},
]
_ADDRESS_CONTROLS = [
    {"autocomplete": "address-line1", "index": 0, "type": "text"},
    {"autocomplete": "postal-code", "index": 1, "type": "text"},
]


def _payment_probe() -> str:
    from tools.browser_vault_tool import _TAB_PROBES

    return _TAB_PROBES["payment"]


_PAYMENT_PROBE = _payment_probe()


class _TwoTabBrowser:
    """A supervisor over several open tabs, like the real one: its page session is bound to one tab (the
    current one), ``focus_page`` re-binds it the way ``CDPSupervisor.focus_page`` does (``origin=""`` = any
    http(s) tab, first one whose ``accept`` probe is truthy), and every evaluation runs on the bound tab."""

    def __init__(self, *tabs: _FakePage):
        self.tabs, self.current, self.focused = list(tabs), tabs[0], []

    def focus_page(self, origin: str, accept=None) -> dict:
        for tab in self.tabs:
            if not tab.href.startswith(("http://", "https://")):
                continue
            if origin and normalize_origin(tab.href) != origin:
                continue
            if accept and not tab.evaluate_runtime(accept).get("result"):
                continue
            self.current = tab
            self.focused.append(tab.href)
            return {"ok": True, "url": tab.href}
        return {"ok": False, "error": "no open page"}

    def evaluate_runtime(self, expression: str) -> dict:
        return self.current.evaluate_runtime(expression)


class _FakePage:
    """The supervisor's view of one open tab: answers origin reads and input inspection, and records every
    secret-bearing fill script as a write."""

    def __init__(self, href: str, controls: list[dict], card_form: bool = True):
        self.href, self.controls, self.writes, self.card_form = href, controls, [], card_form

    def focus_page(self, origin: str, accept=None) -> dict:
        return {"ok": bool(origin) and normalize_origin(self.href) == origin, "url": self.href}

    def evaluate_runtime(self, expression: str) -> dict:
        if expression.strip() == "window.location.href":
            return {"ok": True, "result": self.href}
        if expression == _PAYMENT_PROBE:  # "does this tab hold a card form?"
            return {"ok": True, "result": self.card_form}
        if "expectedOrigin" in expression:  # the fill script (build_fill_js)
            self.writes.append(expression)
            return {"ok": True, "result": json.dumps({"filled": 1})}
        return {"ok": True, "result": json.dumps(self.controls)}  # input inspection


class _RecordingBackend(LocalLoginBackend):
    """The real local backend, recording which secret kinds were resolved."""

    def __init__(self):
        self.resolved: list[str] = []

    def resolve_password(self, handle):
        self.resolved.append("password")
        return super().resolve_password(handle)

    def resolve_otp(self, handle):
        self.resolved.append("otp")
        return super().resolve_otp(handle)

    def resolve_secret(self, handle):
        self.resolved.append("secret")
        return super().resolve_secret(handle)


@pytest.fixture()
def vault(tmp_path):
    from agent import redact
    from agent.vault_backends import unlock as unlock_mod

    store = VaultStore(base_dir=tmp_path / "vault")
    backend = _RecordingBackend()
    asked: list[str] = []
    unlock_mod.set_code_prompt_callback(lambda site, hint: asked.append(site) or "000000")
    try:
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch("agent.vault_backends.backend_for_handle", return_value=backend), \
             patch("agent.vault_backends.unlock.can_prompt_here", return_value=True):
            yield store, backend, asked
    finally:
        unlock_mod.set_code_prompt_callback(None)
        unlock_mod.set_save_login_prompt_callback(None)
        redact.clear_vault_redaction_values()


def _on_page(page: _FakePage):
    from tools import browser_vault_tool

    return patch.object(browser_vault_tool, "_ensure_supervisor", return_value=page)


def _login(store, origin="https://bank.example", otp=True):
    secret = {"identifier_type": "username", "identifier": "tester", "password": _PASSWORD}
    if otp:
        secret["otp_secret"] = _OTP_SEED
    return store.add_item("login", "Bank", secret, origin=origin)


def _refused_before_resolution(out: dict, backend: _RecordingBackend, page: _FakePage):
    assert out["success"] is False
    assert out["error_type"] == "origin_mismatch"
    assert backend.resolved == []
    assert page.writes == []


class TestOneTimeCode:
    def test_saved_code_is_not_minted_for_a_page_on_another_origin(self, vault):
        from tools import browser_vault_tool

        store, backend, asked = vault
        meta = _login(store)
        page = _FakePage("https://elsewhere.example/verify", _OTP_CONTROLS)
        with _on_page(page):
            out = json.loads(browser_vault_tool.browser_vault_enter_code(meta.id, task_id="t"))
        _refused_before_resolution(out, backend, page)
        assert asked == []  # a refusal, not a silent switch to asking the user

    def test_saved_code_is_minted_on_the_saved_origin_and_pinned_to_it(self, vault):
        from tools import browser_vault_tool

        store, backend, asked = vault
        meta = _login(store)
        page = _FakePage("https://bank.example/2fa", _OTP_CONTROLS)
        with _on_page(page):
            out = json.loads(browser_vault_tool.browser_vault_enter_code(meta.id, task_id="t"))
        assert out["success"] is True and out["source"] == "local" and out["origin"] == "https://bank.example"
        assert backend.resolved == ["otp"] and asked == []
        [write] = page.writes
        assert '"https://bank.example"' in write  # the in-page origin assert names the saved origin

    def test_a_code_without_a_saved_item_is_still_asked_of_the_user(self, vault):
        from tools import browser_vault_tool

        _, backend, asked = vault
        page = _FakePage("https://elsewhere.example/verify", _OTP_CONTROLS)
        with _on_page(page):
            out = json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))
        assert out["success"] is True and out["source"] == "user"
        assert asked == ["elsewhere.example"] and backend.resolved == []


class TestFill:
    def test_password_is_not_resolved_for_a_page_on_another_origin(self, vault):
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = _login(store, otp=False)
        page = _FakePage("https://elsewhere.example/login", _LOGIN_CONTROLS)
        with _on_page(page):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        _refused_before_resolution(out, backend, page)

    def test_card_is_refused_before_the_user_is_asked_to_confirm(self, vault):
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = store.add_item(kind="payment", label="Card", origin="https://shop.example", secret=_CARD)
        page = _FakePage("https://elsewhere.example/checkout", _CARD_CONTROLS)
        with _on_page(page), \
             patch("tools.approval_prompt.request_elicitation_consent", return_value="accept") as consent:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        _refused_before_resolution(out, backend, page)
        consent.assert_not_called()

    def test_address_is_not_resolved_for_a_page_on_another_origin(self, vault):
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = store.add_item(kind="address", label="Home", origin="https://shop.example", secret=_ADDRESS)
        page = _FakePage("https://elsewhere.example/shipping", _ADDRESS_CONTROLS)
        with _on_page(page):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        _refused_before_resolution(out, backend, page)

    def test_save_login_fill_is_refused_when_the_page_left_the_saved_origin(self, vault):
        """save_login stores the login for the page it read, then fills through the same guard: a page that
        navigated away while the user was typing gets nothing."""
        from agent.vault_backends import unlock as unlock_mod
        from tools import browser_vault_tool

        store, backend, _ = vault
        page = _FakePage("https://bank.example/login", _LOGIN_CONTROLS)

        def prompt(origin, host):
            page.href = "https://elsewhere.example/login"
            return {"identifier": "tester", "password": _PASSWORD}

        unlock_mod.set_save_login_prompt_callback(prompt)
        with _on_page(page):
            out = json.loads(browser_vault_tool.browser_vault_save_login(task_id="t"))
        assert [m.origin for m in store.list_items()] == ["https://bank.example"]
        _refused_before_resolution(out["fill"], backend, page)


def _any_site_card(store):
    return store.add_item(kind="payment", label="Card", secret=_CARD, origin_match="any_site")


def _consent(answer="accept", page=None, navigate_to=None):
    """The card prompt, answering ``answer``; ``navigate_to`` moves the page while the user is asked."""
    def ask(message, description, **kwargs):
        if navigate_to:
            page.href = navigate_to
        return answer

    return patch("tools.approval_prompt.request_elicitation_consent", side_effect=ask)


class TestAnySiteCard:
    def test_any_site_card_fills_only_the_confirmed_page(self, vault):
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = _any_site_card(store)
        page = _FakePage("https://shop.example/checkout", _CARD_CONTROLS)
        with _on_page(page), _consent() as consent:
            raw = browser_vault_tool.browser_vault_fill(meta.id, task_id="t")
        out = json.loads(raw)
        assert out["success"] is True and out["origin"] == "https://shop.example"
        [call] = consent.call_args_list
        assert call.args[0] == "Fill payment card 'Card' on https://shop.example"
        assert call.kwargs["surface"] == "vault-payment"
        [write] = page.writes
        assert '"https://shop.example"' in write  # the in-page assert names the confirmed origin
        assert backend.resolved == ["secret"]
        assert _CARD["card_number"] not in raw and _CARD["cvc"] not in raw


    def test_any_site_card_never_switches_to_a_background_tab_with_a_card_form(self, vault):
        """The agent is on the shop's checkout (its card fields live in a hosted iframe, so the top document
        has none); a background tab on another https site holds a card input. The card is refused on the
        current tab before the user is asked: no tab switch, no prompt, nothing resolved or written."""
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = _any_site_card(store)
        shop = _FakePage("https://legit-shop.example/checkout", [], card_form=False)
        evil = _FakePage("https://evil.example/pay", _CARD_CONTROLS)
        browser = _TwoTabBrowser(shop, evil)
        with _on_page(browser), _consent() as consent:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        assert out["success"] is False and out["error_type"] == "no_card_form_on_current_tab"
        assert "legit-shop.example" in out["error"] and "evil.example" not in out["error"]
        consent.assert_not_called()
        assert backend.resolved == [] and shop.writes == [] and evil.writes == []
        assert browser.focused == [] and browser.current is shop

    def test_any_site_card_with_no_card_form_on_the_only_tab_is_refused_before_asking(self, vault):
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = _any_site_card(store)
        page = _FakePage("https://shop.example/cart", [], card_form=False)
        with _on_page(page), _consent() as consent:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        assert out["success"] is False and out["error_type"] == "no_card_form_on_current_tab"
        consent.assert_not_called()
        assert backend.resolved == [] and page.writes == []

    def test_any_site_card_handle_does_not_pass_the_one_time_code_guard(self, vault):
        """Only the card fill (which asks the user) may take the any-site path: enter_code with a card
        handle is refused like any origin-less item, before anything is resolved, asked or written."""
        from tools import browser_vault_tool

        store, backend, asked = vault
        meta = _any_site_card(store)
        page = _FakePage("https://shop.example/verify", _OTP_CONTROLS)
        with _on_page(page):
            out = json.loads(browser_vault_tool.browser_vault_enter_code(meta.id, task_id="t"))
        assert out["success"] is False and out["error_type"] == "no_origin"
        assert backend.resolved == [] and page.writes == [] and asked == []

    @pytest.mark.parametrize("probe_kind", ["otp", "login", "address"])
    def test_any_site_path_is_for_the_card_fill_only(self, vault, probe_kind):
        from tools import browser_vault_tool

        store, _, _ = vault
        meta = _any_site_card(store)
        page = _FakePage("https://shop.example/checkout", _CARD_CONTROLS)
        with _on_page(page):
            out = browser_vault_tool._bound_page_origin("t", meta, probe_kind)
        assert isinstance(out, dict) and out["error_type"] == "no_origin"
        with _on_page(page):
            assert browser_vault_tool._bound_page_origin("t", meta, "payment") == "https://shop.example"

    def test_any_site_card_declined_resolves_and_writes_nothing(self, vault):
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = _any_site_card(store)
        page = _FakePage("https://shop.example/checkout", _CARD_CONTROLS)
        with _on_page(page), _consent("decline"):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        assert out["success"] is False and out["error_type"] == "payment_declined"
        assert backend.resolved == [] and page.writes == []

    @pytest.mark.parametrize("env", ["HERMES_SINGLE_QUERY_SESSION", "HERMES_CRON_SESSION"])
    def test_any_site_card_is_refused_when_nobody_can_confirm(self, vault, monkeypatch, env):
        """Real consent routing, no mock: a -q or cron session has nobody to ask, so the fill is refused."""
        from tools import browser_vault_tool

        monkeypatch.setenv(env, "1")
        store, backend, _ = vault
        meta = _any_site_card(store)
        page = _FakePage("https://shop.example/checkout", _CARD_CONTROLS)
        with _on_page(page):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        assert out["success"] is False and out["error_type"] == "payment_declined"
        assert backend.resolved == [] and page.writes == []

    @pytest.mark.parametrize("href", ["http://shop.example/checkout", "chrome://settings",
                                      "file:///x/checkout.html"])
    def test_any_site_card_refuses_a_non_https_page_before_asking(self, vault, href):
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = _any_site_card(store)
        page = _FakePage(href, _CARD_CONTROLS)
        with _on_page(page), _consent() as consent:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        assert out["success"] is False
        if not href.startswith("file:"):
            assert out["error_type"] == "insecure_origin"
        consent.assert_not_called()
        assert backend.resolved == [] and page.writes == []

    def test_any_site_card_allows_a_loopback_http_checkout(self, vault):
        from tools import browser_vault_tool

        store, _, _ = vault
        meta = _any_site_card(store)
        page = _FakePage("http://127.0.0.1:8765/checkout", _CARD_CONTROLS)
        with _on_page(page), _consent() as consent:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        assert out["success"] is True
        assert consent.call_args.args[0] == "Fill payment card 'Card' on http://127.0.0.1:8765"
        [write] = page.writes
        assert '"http://127.0.0.1:8765"' in write

    def test_navigation_after_confirm_keeps_the_confirmed_origin_pinned(self, vault):
        """The page moves while the user is asked: the write still carries the origin the user confirmed,
        so the in-page assert refuses on the new page."""
        from tools import browser_vault_tool

        store, _, _ = vault
        meta = _any_site_card(store)
        page = _FakePage("https://shop.example/checkout", _CARD_CONTROLS)
        with _on_page(page), _consent(page=page, navigate_to="https://evil.example/pay"):
            browser_vault_tool.browser_vault_fill(meta.id, task_id="t")
        for write in page.writes:
            assert '"https://shop.example"' in write and "evil.example" not in write

    def test_any_site_card_without_a_supervisor_is_refused_before_asking(self, vault):
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = _any_site_card(store)
        with patch.object(browser_vault_tool, "_ensure_supervisor", return_value=None), _consent() as consent:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        assert out["success"] is False and out["error_type"] == "supervisor_required"
        consent.assert_not_called()
        assert backend.resolved == []

    def test_card_saved_without_a_site_is_still_refused(self, vault):
        """A blank origin is not an opt-in: such a card stays unfillable and nobody is asked."""
        from tools import browser_vault_tool

        store, backend, _ = vault
        meta = store.add_item(kind="payment", label="Card", secret=_CARD, origin=None)
        page = _FakePage("https://shop.example/checkout", _CARD_CONTROLS)
        with _on_page(page), _consent() as consent:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
        assert out["success"] is False and out["error_type"] == "no_origin"
        consent.assert_not_called()
        assert backend.resolved == [] and page.writes == []
