"""browser_vault action=check is a dry run of fill.

It answers "would fill act on this page?" with the lookup, tab choice and exact-origin test fill
itself uses, and it does so without resolving a secret, prompting the user, entering the backend's
fill context or writing to the page.

The page is a stand-in for the CDP supervisor (current URL, a password-box count, and a record of
every script that is not one of check's two reads). Secrets are dummy values in a temporary local vault.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from agent.vault_backends.local import LocalLoginBackend
from agent.vault_store import VaultStore, normalize_origin

_PASSWORD = "dummy-password-7"
_CARD = {"card_number": "4111111111111111", "cardholder_name": "Test User", "exp_month": "07",
         "exp_year": "2029", "cvc": "123", "billing_postal_code": "94107"}


class _FakePage:
    """One open tab as the supervisor sees it. ``other`` collects every script check has no business
    running (inspection stamps, fill, read-back)."""

    def __init__(self, href: str, password_fields: int = 1, form: bool = True):
        self.href, self.password_fields, self.form = href, password_fields, form
        self.other: list[str] = []

    def focus_page(self, origin: str, accept=None) -> dict:
        on_origin = bool(origin) and normalize_origin(self.href) == origin
        return {"ok": on_origin and self.form, "url": self.href, "session_id": "S1"}

    def current_page_session(self) -> str:
        return "S1"

    def evaluate_runtime(self, expression: str, *, session_id=None, **kwargs) -> dict:
        if expression.strip() == "window.location.href":
            return {"ok": True, "result": self.href}
        if "window.location.pathname" in expression:  # check's read of the checked page
            path = self.href.split("://", 1)[1].partition("/")[2].split("?")[0].split("#")[0]
            return {"ok": True, "result": json.dumps({
                "origin": normalize_origin(self.href), "path": "/" + path, "form": self.form,
                "password_fields": self.password_fields if self.form else 0})}
        self.other.append(expression)
        return {"ok": True, "result": "[]"}


class _RecordingBackend(LocalLoginBackend):
    """The real local backend, recording every secret resolution and fill-context entry."""

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

    @contextmanager
    def fill_context(self):
        self.resolved.append("fill_context")
        yield


class _LockedManager:
    """A password manager that is locked: any metadata read or unlock would be a prompt."""

    name, display_name, needs_unlock = "manager", "Test Manager", True

    def is_unlocked(self) -> bool:
        return False

    def get_meta(self, handle):
        raise AssertionError("a locked manager's items are not read")

    def unlock(self, master):
        raise AssertionError("check never unlocks")


@pytest.fixture()
def vault(tmp_path):
    from agent import redact

    store = VaultStore(base_dir=tmp_path / "vault")
    backend = _RecordingBackend()
    try:
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch("agent.vault_backends.backend_for_handle", return_value=backend):
            yield store, backend
    finally:
        redact.clear_vault_redaction_values()


def _login(store, origin="https://app.example.test"):
    return store.add_item("login", "App", {"identifier_type": "username", "identifier": "tester",
                                          "password": _PASSWORD}, origin=origin)


def _check(page: _FakePage, handle: str) -> tuple[dict, str]:
    """Run check against ``page``; any secret-bearing script fails the test."""
    from tools import browser_vault_tool

    with patch.object(browser_vault_tool, "_ensure_supervisor", return_value=page), \
         patch.object(browser_vault_tool, "_eval_js_secret", side_effect=AssertionError("secret script")):
        raw = browser_vault_tool.browser_vault_check(handle, task_id="t")
    return json.loads(raw), raw


def test_would_fill_on_the_saved_origin_with_a_password_box(vault):
    store, _ = vault
    meta = _login(store)
    out, _ = _check(_FakePage("https://app.example.test/login?next=%2Fhome#top"), meta.id)
    assert out["success"] is True and out["would_fill"] is True
    assert out["item_origins"] == ["https://app.example.test"]
    assert out["page_origin"] == "https://app.example.test"
    assert out["page"] == "https://app.example.test/login"  # no query, no fragment
    assert out["form_found"] is True and out["password_fields"] == 1
    assert "refusal" not in out and out["reason"]


def test_origin_mismatch_reports_the_refusal_fill_would_give(vault):
    from tools import browser_vault_tool

    store, _ = vault
    meta = _login(store)
    page = _FakePage("https://elsewhere.example.test/login")
    out, _ = _check(page, meta.id)
    assert out["success"] is True and out["would_fill"] is False
    assert out["refusal"] == "origin_mismatch"
    assert "https://elsewhere.example.test" in out["reason"] and "https://app.example.test" in out["reason"]
    with patch.object(browser_vault_tool, "_ensure_supervisor", return_value=page):
        filled = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
    assert filled["error_type"] == out["refusal"] and filled["error"] == out["reason"]


def test_saved_origin_without_a_password_box_would_not_fill(vault):
    store, _ = vault
    meta = _login(store)
    out, _ = _check(_FakePage("https://app.example.test/dashboard", form=False), meta.id)
    assert out["success"] is True and out["would_fill"] is False
    assert out["refusal"] == "no_form" and out["password_fields"] == 0
    assert "no password box" in out["reason"]


def test_unknown_handle_is_reported_without_touching_the_page(vault):
    page = _FakePage("https://app.example.test/login")
    out, _ = _check(page, "no-such-handle")
    assert out["success"] is False and out["would_fill"] is False
    assert "no-such-handle" in out["error"] and "action=list" in out["error"]
    assert page.other == []


def test_locked_manager_is_reported_and_never_unlocked():
    from agent.vault_backends import unlock as unlock_mod
    from tools import browser_vault_tool

    prompted: list[str] = []
    unlock_mod.set_unlock_prompt_callback(lambda name, display: prompted.append(name) or "master")
    try:
        with patch("agent.vault_backends.backend_for_handle", return_value=_LockedManager()), \
             patch.object(browser_vault_tool, "browser_vault_unlock",
                          side_effect=AssertionError("check never unlocks")):
            out, _ = _check(_FakePage("https://app.example.test/login"), "manager:item")
    finally:
        unlock_mod.set_unlock_prompt_callback(None)
    assert out["success"] is False and out["would_fill"] is False
    assert out["error_type"] == "unlock_required" and "Test Manager is locked" in out["error"]
    assert prompted == []


def test_check_resolves_no_secret_and_writes_nothing(vault):
    store, backend = vault
    meta = _login(store)
    page = _FakePage("https://app.example.test/login")
    out, raw = _check(page, meta.id)
    assert out["would_fill"] is True
    assert backend.resolved == []  # no password, code or card read; no fill context entered
    assert page.other == []  # no inspection stamps, no fill script, no read-back
    assert _PASSWORD not in raw


def test_card_check_does_not_ask_the_user_to_confirm(vault):
    store, backend = vault
    meta = store.add_item(kind="payment", label="Card", origin="https://shop.example.test", secret=_CARD)
    with patch("tools.approval_prompt.request_elicitation_consent", return_value="accept") as consent:
        out, raw = _check(_FakePage("https://shop.example.test/checkout"), meta.id)
    assert out["would_fill"] is True and "confirms the card" in out["reason"]
    consent.assert_not_called()
    assert backend.resolved == []
    assert not any(value in raw for value in (_CARD["card_number"], _CARD["cvc"]))


def test_check_is_offered_and_dispatched(vault):
    from tools import browser_vault_tool

    store, _ = vault
    meta = _login(store)
    assert "check" in browser_vault_tool.BROWSER_VAULT_SCHEMA["parameters"]["properties"]["action"]["enum"]
    page = _FakePage("https://app.example.test/login")
    with patch.object(browser_vault_tool, "_ensure_supervisor", return_value=page), \
         patch.object(browser_vault_tool, "_fenced_page_op", side_effect=lambda task_id, fn: fn()):
        out = json.loads(browser_vault_tool._handle_vault({"action": "check", "handle": meta.id}, task_id="t"))
    assert out["would_fill"] is True
