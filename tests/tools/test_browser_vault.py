"""Tests for the vault-backed password-blind browser autofill feature.

Covers:
- VaultStore: encrypt/decrypt round-trip, file perms, identifier-as-metadata
  (login secret payload is password-only)
- login-control classifier: scoring + new-password/one-time-code exclusion,
  password-only fill selection
- origin-binding refusal (pre-check + in-script TOCTOU assert)
- fail-closed secret eval (no argv fallback)
- vault-value redaction registry (browser_cdp read-back regression)
- tool gating: check_fn False when the vault is empty
"""

from __future__ import annotations

import json
import re
import os
import stat
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.vault_login_classifier import (
    ClassifiedLoginControl,
    LoginControl,
    build_fill_js,
    classify_login_control,
    select_password_fill,
)
from agent.vault_store import (
    VaultError,
    VaultStore,
    normalize_origin,
    scrub_secret_from_text,
)


@pytest.fixture()
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


def _add_login(store, origin="https://example.com", password="s3cret-pw"):
    return store.add_item(
        kind="login",
        label="Example login",
        origin=origin,
        secret={
            "identifier_type": "email",
            "identifier": "user@example.com",
            "password": password,
            "origin": origin,
        },
    )


# ---------------------------------------------------------------------------
# VaultStore
# ---------------------------------------------------------------------------

class TestVaultStore:
    def test_roundtrip_encrypt_decrypt(self, store):
        meta = _add_login(store)
        secret = store.resolve_secret(meta.id)
        # Design: login secret payload is password-only; identifier is metadata.
        assert secret == {"password": "s3cret-pw"}
        assert meta.identifier == "user@example.com"
        assert meta.identifier_type == "email"

    def test_vault_file_never_contains_password(self, store, tmp_path):
        _add_login(store)
        blob = (tmp_path / "vault" / "vault.json.enc").read_bytes()
        assert b"s3cret-pw" not in blob

    def test_file_permissions_0600(self, store, tmp_path):
        _add_login(store)
        for name in ("vault.json.enc", "vault.key"):
            mode = stat.S_IMODE(os.stat(tmp_path / "vault" / name).st_mode)
            assert mode == 0o600, f"{name} has mode {oct(mode)}"

    def test_listing_is_password_free(self, store):
        meta = _add_login(store)
        items = store.list_items()
        assert len(items) == 1
        dumped = json.dumps(items[0].to_dict())
        assert "s3cret-pw" not in dumped
        assert "password" not in dumped
        # Identifier IS visible metadata now.
        assert items[0].identifier == "user@example.com"
        assert items[0].identifier_type == "email"
        assert items[0].id == meta.id
        assert items[0].origin == "https://example.com"

    def test_remove_item(self, store):
        meta = _add_login(store)
        assert store.remove_item(meta.id) is True
        assert store.remove_item(meta.id) is False
        assert store.list_items() == []

    def test_login_requires_origin(self, store):
        with pytest.raises(VaultError):
            store.add_item(
                kind="login",
                label="x",
                secret={
                    "identifier_type": "email",
                    "identifier": "a@b.c",
                    "password": "p",
                },
            )

    def test_all_kinds_supported(self, store):
        store.add_item(kind="payment", label="Card", origin="https://shop.test", secret=_CARD)
        store.add_item(kind="address", label="Home", secret=_ADDRESS)
        kinds = {m.kind for m in store.list_items()}
        assert kinds == {"payment", "address"}

    def test_checkout_kinds_keep_only_canonical_fields(self, store):
        """The fill maps canonical names → autocomplete tokens; a stray ad-hoc key would be stored (secret!)
        yet unfillable, and a missing required field would make the item dead on every checkout."""
        meta = store.add_item(kind="payment", label="Card", secret={**_CARD, "note": "personal"})
        assert "note" not in store.resolve_secret(meta.id)
        with pytest.raises(VaultError, match="cvc"):
            store.add_item(kind="payment", label="Card", secret={k: v for k, v in _CARD.items() if k != "cvc"})

    def test_unknown_kind_rejected(self, store):
        with pytest.raises(VaultError):
            store.add_item(kind="totp", label="x", secret={})

    def test_has_items(self, store):
        assert store.has_items() is False
        _add_login(store)
        assert store.has_items() is True

    def test_normalize_origin(self):
        assert normalize_origin("https://Example.com:443/login?x=1") == "https://example.com"
        assert normalize_origin("http://localhost:8931/") == "http://localhost:8931"
        assert normalize_origin("http://site.test:80") == "http://site.test"
        with pytest.raises(VaultError):
            normalize_origin("example.com")

    def test_scrub_secret_from_text(self):
        secret = {"password": "hunter22x", "identifier": "me@x.io"}
        out = scrub_secret_from_text("boom hunter22x at me@x.io", secret)
        assert "hunter22x" not in out
        assert "me@x.io" not in out


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

_CARD = {"card_number": "4111111111111111", "cardholder_name": "A User", "exp_month": "7", "exp_year": "2029",
         "cvc": "123", "billing_postal_code": "94110"}
_ADDRESS = {"address_line1": "1 Main St", "city": "Springfield", "postal_code": "12345", "country": "US"}


def _ctrl(**kw):
    base = dict(autocomplete="", form_index=0, index=0, label="", name="", type="text")
    base.update(kw)
    return LoginControl(**base)


class TestClassifier:
    def test_autocomplete_exact_match_maps_token(self):
        for token in ("username", "email", "tel", "current-password"):
            res = classify_login_control(_ctrl(autocomplete=token))
            assert res is not None and res.token == token

    def test_new_password_autocomplete_excluded(self):
        assert classify_login_control(
            _ctrl(autocomplete="new-password", type="password")
        ) is None

    def test_one_time_code_excluded(self):
        assert classify_login_control(_ctrl(autocomplete="one-time-code")) is None

    def test_label_new_password_excluded(self):
        for label in ("New password", "Confirm Password", "create-password", "Repeat  password"):
            assert classify_login_control(_ctrl(type="password", label=label)) is None, label

    def test_password_type_maps_current_password(self):
        res = classify_login_control(_ctrl(type="password"))
        assert res.token == "current-password"

    def test_email_tel_types_map_tokens(self):
        assert classify_login_control(_ctrl(type="email")).token == "email"
        res = classify_login_control(_ctrl(type="tel"))
        assert res.token == "tel"

    def test_label_heuristics(self):
        assert classify_login_control(_ctrl(label="E-mail address")).token == "email"
        assert classify_login_control(_ctrl(name="mobile_number")).token == "tel"
        res = classify_login_control(_ctrl(label="Username or account"))
        assert res.token == "username"

    def test_unmatched_returns_none(self):
        assert classify_login_control(_ctrl(label="Search the docs")) is None

    def test_select_password_fill_picks_best_password(self):
        user = ClassifiedLoginControl(_ctrl(index=0, form_index=0, autocomplete="username"), 100, "username")
        pw_heur = ClassifiedLoginControl(_ctrl(index=1, form_index=0, type="password"), 90, "current-password")
        pw_exact = ClassifiedLoginControl(_ctrl(index=3, form_index=0, autocomplete="current-password"), 100, "current-password")
        fills = select_password_fill([user, pw_heur, pw_exact], "p")
        # Password only — the identifier field is never filled by the vault.
        assert [(f["index"], f["token"]) for f in fills] == [(3, "current-password")]

    def test_select_password_fill_requires_password_field(self):
        user = ClassifiedLoginControl(_ctrl(index=0, autocomplete="username"), 100, "username")
        assert select_password_fill([user], "p") == []

    def test_select_password_fill_single_field_only(self):
        pw1 = ClassifiedLoginControl(_ctrl(index=1, type="password"), 90, "current-password")
        pw2 = ClassifiedLoginControl(_ctrl(index=2, type="password"), 90, "current-password")
        fills = select_password_fill([pw1, pw2], "p")
        assert len(fills) == 1 and fills[0]["index"] == 1


    def test_build_fill_js_leaves_no_dom_marker_and_binds_target_to_inspection(self):
        # P1-1: no persistent selector for filled controls. The fill targets the input by the
        # <nonce>:<index> stamp ITS OWN inspection wrote (a bare index is re-resolved by position,
        # and a second inspection in between would re-stamp — either way the password could land in
        # another field); stamps carry no secret and the fill script strips every one before returning.
        js = build_fill_js(
            [{"index": 0, "token": "current-password", "value": "x"}],
            expected_origin="https://example.com",
        )
        assert "vaultSecret" not in js
        assert "data-vault-secret" not in js
        assert "elements[f.index]" not in js
        assert "[data-hermes-vault-slot=" in js and "nonce + ':' + f.index" in js
        assert 'f.token === "current-password" && el.type !== "password"' in js  # a password fill never lands in a text box
        # unmasking moved to build_fill_verify_js: the fill cannot know what the page kept
        assert 'removeAttribute("data-hermes-vault-slot")' not in js

    def test_build_fill_js_carries_the_value_in_the_input_event(self):
        # A framework-controlled input (React/Vue/MUI) reverts a value written without an `input`
        # event that carries `data`: React reads `event.data` and treats an event without it as
        # irrelevant. Without this the field reads filled in the DOM and empty to the page.
        js = build_fill_js(
            [{"index": 0, "token": "current-password", "value": "x"}],
            expected_origin="https://example.com",
        )
        assert 'new InputEvent("input", { bubbles: true, composed: true, data: f.value, inputType: "insertText" })' in js

    # The read-back and focus scripts are executed against a synthetic DOM in
    # tests/agent/test_vault_fill_scripts.py.

    def test_build_fill_js_asserts_origin_before_any_write(self):
        # P1-2: the origin assert must run inside the SAME script, before
        # any element write.
        js = build_fill_js(
            [{"index": 0, "token": "current-password", "value": "x"}],
            expected_origin="https://example.com",
        )
        assert '"https://example.com"' in js
        assert "window.location.origin" in js
        assert "origin_changed" in js
        assert js.index("origin_changed") < js.index("querySelectorAll")


# ---------------------------------------------------------------------------
# Browser tool: origin binding + gating
# ---------------------------------------------------------------------------

class TestBrowserVaultTools:
    def test_check_fn_follows_the_browser_not_the_item_count(self, tmp_path):
        """The vault tools ride with the browser toolset: an empty vault must still expose
        browser_vault_save_login (that is how the first login gets saved), and no browser means no tools."""
        from tools import browser_vault_tool

        empty = VaultStore(base_dir=tmp_path / "empty-vault")
        with patch("agent.vault_store.get_vault_store", return_value=empty), \
             patch("tools.browser_session_cdp.get_cdp_override_raw", return_value=""):
            with patch("tools.browser_use_cli.browser_exec_available", return_value=True):
                assert browser_vault_tool._check_vault_available() is True
            with patch("tools.browser_use_cli.browser_exec_available", return_value=False):
                assert browser_vault_tool._check_vault_available() is False
        # A configured CDP endpoint is a browser too: the vault rides along even without browser-harness
        with patch("tools.browser_use_cli.browser_exec_available", return_value=False), \
             patch("tools.browser_session_cdp.get_cdp_override_raw", return_value="ws://127.0.0.1:9222/devtools/browser/x"):
            assert browser_vault_tool._check_vault_available() is True

    def test_list_returns_identifier_never_password(self, store):
        from tools import browser_vault_tool

        _add_login(store)
        with patch("agent.vault_store.get_vault_store", return_value=store):
            out = json.loads(browser_vault_tool.browser_vault_list())
        assert out["success"] is True
        assert out["items"][0]["handle"].startswith("vault_")
        # Design change: identifier is agent-visible metadata.
        assert out["items"][0]["identifier"] == "user@example.com"
        assert out["items"][0]["identifier_type"] == "email"
        assert "s3cret-pw" not in json.dumps(out)

    def test_fill_refused_on_origin_mismatch(self, store):
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_current_page_origin", return_value="https://evil.com"):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
        assert out["success"] is False
        assert out["error_type"] == "origin_mismatch"
        assert "s3cret-pw" not in json.dumps(out)

    @staticmethod
    def _manager_meta():
        from agent.vault_store import VaultItemMeta

        return VaultItemMeta(
            id="op:multi", kind="login", label="Amazon", origin="https://amazon.co.uk",
            created_at="2026-01-01T00:00:00Z", identifier_type="username", identifier="jane@example.com",
            allowed_origins=("https://amazon.co.uk", "https://www.amazon.co.uk",
                             "https://eu.account.amazon.com"))

    def test_fill_allowed_on_every_saved_origin_of_a_multi_website_item(self):
        """A manager item with several saved websites fills on each of them
        (exact match only), and the in-page origin assert pins the origin
        actually being filled — not just the first saved one."""
        from tools import browser_vault_tool

        meta = self._manager_meta()

        class _ManagerBackend:
            name, display_name, needs_unlock = "onepassword", "1Password", False

            def is_unlocked(self):
                return True

            def get_meta(self, handle):
                return meta if handle == meta.id else None

            def fill_context(self):
                return nullcontext()

            def resolve_password(self, handle):
                return "s3cret-pw"

        controls = [
            {"autocomplete": "email", "formIndex": 0, "index": 0, "label": "", "name": "email", "type": "email"},
            {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
        ]
        secret_exprs = []

        def fake_eval(task_id, expression):
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            secret_exprs.append(expression)
            if "reverted" in expression:  # read-back pass (build_fill_verify_js)
                return {"success": True, "result": json.dumps({"landed": 1, "reverted": 0, "missing": []})}
            return {"success": True, "result": json.dumps({"filled": 1})}

        with patch("agent.vault_backends.backend_for_handle", return_value=_ManagerBackend()), \
             patch.object(browser_vault_tool, "_current_page_origin", return_value="https://www.amazon.co.uk"), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
            raw = browser_vault_tool.browser_vault_fill("op:multi", with_identifier=False)
        out = json.loads(raw)
        assert out["success"] is True
        assert out["origin"] == "https://www.amazon.co.uk"
        # the synchronous in-page check is bound to the origin actually matched
        assert "https://www.amazon.co.uk" in secret_exprs[0]
        assert "s3cret-pw" not in raw
        assert out["filled_fields"] == 1

    def test_fill_still_refused_on_origin_not_saved_on_the_item(self):
        """Multi-website items widen nothing: an unsaved origin — even a sibling
        subdomain — is still refused (fail-closed regression)."""
        from tools import browser_vault_tool

        meta = self._manager_meta()

        class _ManagerBackend:
            name, display_name, needs_unlock = "onepassword", "1Password", False

            def is_unlocked(self):
                return True

            def get_meta(self, handle):
                return meta if handle == meta.id else None

            def resolve_password(self, handle):
                return "s3cret-pw"

        with patch("agent.vault_backends.backend_for_handle", return_value=_ManagerBackend()), \
             patch.object(browser_vault_tool, "_current_page_origin", return_value="https://payments.amazon.co.uk"):
            out = json.loads(browser_vault_tool.browser_vault_fill("op:multi"))
        assert out["success"] is False
        assert out["error_type"] == "origin_mismatch"
        assert "s3cret-pw" not in json.dumps(out)

    def test_fill_unknown_handle(self, store):
        from tools import browser_vault_tool

        with patch("agent.vault_store.get_vault_store", return_value=store):
            out = json.loads(browser_vault_tool.browser_vault_fill("vault_nope"))
        assert out["success"] is False

    def test_fill_success_returns_counts_only(self, store):
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [
            {"autocomplete": "email", "formIndex": 0, "index": 0, "label": "", "name": "email", "type": "email"},
            {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        secret_exprs = []

        def fake_eval_secret(task_id, expression):
            secret_exprs.append(expression)
            if "reverted" in expression:  # read-back pass (build_fill_verify_js)
                return {"success": True, "result": json.dumps({"landed": 1, "reverted": 0, "missing": []})}
            return {"success": True, "result": json.dumps({"filled": 1})}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
            raw = browser_vault_tool.browser_vault_fill(meta.id, with_identifier=False)
        out = json.loads(raw)
        # Password-only fill: exactly one field.
        out.pop("next", None)  # workflow hint, not data
        assert out == {
            "success": True,
            "filled_fields": 1,
            "backend": "local",
            "kind": "login",
            "origin": "https://example.com",
        }
        assert "s3cret-pw" not in raw
        # The secret expression only ever goes through the secret eval path,
        # and it targets only the password field (index 1).
        # the fill, then the read-back that proves the page kept the value
        assert "s3cret-pw" in secret_exprs[0]
        assert secret_exprs[1:] and all("reverted" in e for e in secret_exprs[1:])
        assert '"index": 0' not in secret_exprs[0]
        assert "user@example.com" not in secret_exprs[0]

    def _trusted_harness(self, controls, *, focus_result, verify_result, typed, secret_exprs):
        """Patch set for the trusted-typing path: a supervisor that records keystrokes."""
        from tools import browser_vault_tool

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            secret_exprs.append(expression)
            if "ok: true" in expression:  # the in-page origin assert (build_origin_check_js)
                return {"success": True, "result": json.dumps({"ok": True})}          # in-page origin assert
            if 'refused: "focus_lost"' in expression:
                return {"success": True, "result": json.dumps(focus_result)}          # focus assert
            if "reverted" in expression:
                return {"success": True, "result": json.dumps(verify_result)}         # read-back
            if "unstamped" in expression:
                return {"success": True, "result": json.dumps({"unstamped": True})}   # cleanup
            return {"success": True, "result": json.dumps({"filled": 1})}

        class _Sup:
            def current_page_session(self):
                return "S1"

            def insert_text(self, text, **kwargs):
                typed.append(text)
                return {"ok": True}

            def focus_page(self, origin, accept=None):
                return {"ok": True, "url": "https://example.com/login"}

        return [
            patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval),
            patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret),
            patch.object(browser_vault_tool, "_ensure_supervisor", return_value=_Sup()),
            patch.object(browser_vault_tool, "_page_backend", return_value=None),
        ]

    def test_trusted_typing_types_the_secret_once_when_focus_holds(self, store):
        """A controlled input accepts a TRUSTED keystroke: the password goes in through the browser,
        not through a JS value write, and the read-back is what reports the fill."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 1,
                     "label": "", "name": "pw", "type": "password"}]
        typed, secret_exprs = [], []
        patches = self._trusted_harness(controls, focus_result={"focused": 1},
                                        verify_result={"landed": 1, "reverted": 0, "missing": []},
                                        typed=typed, secret_exprs=secret_exprs)
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patches[0], patches[1], patches[2], patches[3]:
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is True and out["filled_fields"] == 1
        assert typed == ["s3cret-pw"]  # one keystroke per text field, and it is the secret
        assert "s3cret-pw" not in raw

    def test_trusted_typing_aborts_on_focus_drift_and_types_nothing(self, store):
        """Input.insertText hits whatever holds focus. If the target does not hold it, the secret is
        never typed — the fill refuses instead of gambling on the focused element."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 1,
                     "label": "", "name": "pw", "type": "password"}]
        typed, secret_exprs = [], []
        patches = self._trusted_harness(controls,
                                        focus_result={"refused": "focus_lost", "tag": "INPUT", "found": "BODY"},
                                        verify_result={"landed": 1, "reverted": 0, "missing": []},
                                        typed=typed, secret_exprs=secret_exprs)
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patches[0], patches[1], patches[2], patches[3]:
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is False and out["error_type"] == "focus_lost"
        assert typed == []             # not one keystroke reached the page
        assert "s3cret-pw" not in raw

    def test_trusted_typing_reports_a_reverted_value_as_failure(self, store):
        """A trusted keystroke the page still rejects must not be reported as a fill."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 1,
                     "label": "", "name": "pw", "type": "password"}]
        typed, secret_exprs = [], []
        patches = self._trusted_harness(controls, focus_result={"focused": 1},
                                        verify_result={"landed": 0, "reverted": 1, "missing": []},
                                        typed=typed, secret_exprs=secret_exprs)
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patches[0], patches[1], patches[2], patches[3]:
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is False and out["error_type"] == "fill_reverted"
        # The bulk insert, then one retype key by key: still rejected, so the fill aborts
        assert typed[0] == "s3cret-pw"
        assert typed[1:] == list("s3cret-pw")

    def _per_key_harness(self, store, sup, *, focus_answers, verify_answers):
        """A one-password page whose bulk insert reads back wrong, so the fill retypes key by key.
        ``focus_answers`` / ``verify_answers`` are consumed one per check (the last repeats); a callable
        focus answer gets the supervisor and the focus script."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 1,
                     "label": "", "name": "pw", "type": "password"}]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def take(answers):
            return answers.pop(0) if len(answers) > 1 else answers[0]

        def fake_eval_secret(task_id, expression):
            if "ok: true" in expression:  # the in-page origin assert (build_origin_check_js)
                return {"success": True, "result": json.dumps({"ok": True})}
            if 'refused: "focus_lost"' in expression:
                answer = take(focus_answers)
                return {"success": True, "result": json.dumps(answer(sup, expression) if callable(answer) else answer)}
            if "reverted" in expression:
                return {"success": True, "result": json.dumps(take(verify_answers))}
            if "cleared" in expression:
                return {"success": True, "result": json.dumps({"cleared": 1})}
            return {"success": True, "result": json.dumps({"unstamped": True})}

        patches = (patch("agent.vault_store.get_vault_store", return_value=store),
                   patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval),
                   patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret),
                   patch.object(browser_vault_tool, "_ensure_supervisor", return_value=sup),
                   patch.object(browser_vault_tool, "_page_backend", return_value=None),
                   patch.object(browser_vault_tool, "_TYPE_PER_KEY_DELAY_S", 0))
        return meta, patches

    class _SessionSup:
        """A supervisor whose page session another tool can move. ``insert_text`` behaves like the
        real one: a ``session_id`` that is no longer current is refused."""

        def __init__(self):
            self.sid = "S1"
            self.typed: list[tuple[str, str]] = []

        def current_page_session(self):
            return self.sid

        def insert_text(self, text, *, session_id, **kwargs):
            if session_id != self.sid:
                return {"ok": False, "error": "page session changed"}
            self.typed.append((text, self.sid))
            return {"ok": True}

        def focus_page(self, origin, accept=None):
            return {"ok": True, "url": "https://example.com/login"}

    def test_focus_is_reasserted_before_every_retyped_key(self, store):
        """On the key-by-key retype the page can move focus after the first character; the next
        character must not follow it into another input."""
        from tools import browser_vault_tool

        sup = self._SessionSup()
        lost = {"refused": "focus_lost", "tag": "INPUT", "found": "INPUT"}
        meta, patches = self._per_key_harness(
            store, sup,
            focus_answers=[{"focused": 1}, {"focused": 1}, lost],
            verify_answers=[{"landed": 0, "reverted": 1, "missing": []}])
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
        assert out["success"] is False and out["error_type"] == "focus_lost"
        # the bulk insert, then the first retyped key; focus was gone before the second
        assert [t for t, _ in sup.typed] == ["s3cret-pw", "s"]

    def test_keystrokes_never_follow_the_supervisor_to_another_session(self, store):
        """The focus check validated the page on session S1; when another tool retargets the
        supervisor before the keystroke, the secret is not typed into the new session."""
        from tools import browser_vault_tool

        sup = self._SessionSup()

        def focused_then_retargeted(s, _expression):
            s.sid = "S2"  # e.g. focus_page from a concurrent browser call
            return {"focused": 1}

        meta, patches = self._per_key_harness(
            store, sup, focus_answers=[focused_then_retargeted, {"focused": 1}],
            verify_answers=[{"landed": 1, "reverted": 0, "missing": []}])
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
        assert out["success"] is False
        assert [sid for _, sid in sup.typed if sid == "S2"] == []

    def test_every_secret_script_is_pinned_to_the_session_the_fill_started_on(self, store):
        """The read-back carries the expected values and runs while the page may move on: it, the
        write and every focus/clear check go to the page session the fill validated, never to
        whatever page the supervisor points at by then."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 1,
                     "label": "", "name": "pw", "type": "password"}]
        sessions = []

        class _Sup:  # the JS write path: no trusted typing
            sid = "S1"

            def current_page_session(self):
                return self.sid

            def focus_page(self, origin, accept=None):
                return {"ok": True, "url": "https://example.com/login"}

            def evaluate_runtime(self, expression, *, session_id=None, **kwargs):
                if "location.href" in expression:
                    return {"ok": True, "result": "https://example.com/login"}
                if "s3cret-pw" not in expression:
                    return {"ok": True, "result": json.dumps(controls)}
                sessions.append(session_id)
                if session_id is not None and session_id != self.sid:
                    return {"ok": False, "error": "the page session changed since it was validated"}
                if "reverted" in expression:
                    return {"ok": True, "result": json.dumps({"landed": 1, "reverted": 0, "missing": []})}
                self.sid = "S2"  # another tool retargets the supervisor right after the write
                return {"ok": True, "result": json.dumps({"filled": 1})}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_ensure_supervisor", return_value=_Sup()), \
             patch.object(browser_vault_tool, "_page_backend", return_value=None), \
             patch.object(browser_vault_tool, "_bot_desktop_browser_session", return_value=False):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t", with_identifier=False))
        assert out["success"] is False
        assert sessions and all(s == "S1" for s in sessions)

    def test_a_value_reverted_after_the_first_good_read_is_not_a_fill(self, store):
        """The read-back is a claim the page KEPT the value: a framework that accepts it for one
        frame and reverts it on the next commit must fail the fill, not pass on the first sample."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 1,
                     "label": "", "name": "pw", "type": "password"}]
        samples = [{"landed": 1, "reverted": 0, "missing": []}]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            if "reverted" in expression:
                answer = samples[0]
                samples[0] = {"landed": 0, "reverted": 1, "missing": []}  # reverted from now on
                return {"success": True, "result": json.dumps(answer)}
            return {"success": True, "result": json.dumps({"filled": 1})}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, with_identifier=False))
        assert out["success"] is False and out["error_type"] == "fill_reverted"

    def test_a_revert_lasting_a_frame_or_two_is_seen(self):
        """The read-back samples; a revert shorter than its sampling gap passes unseen. A framework
        that reverts for a frame or two (20 ms) in every 200 ms must not read as a kept value. The
        clock is simulated: each sample reads the page at the read-back's own sleep schedule."""
        from tools import browser_vault_tool

        clock = {"now": 0.0}
        ms = lambda: round(clock["now"] * 1000)

        def page(_task_id, _expression):
            kept = not 10 <= ms() % 200 < 30  # reverted 10-30 ms into every 200 ms
            return {"success": True, "result": json.dumps(
                {"landed": int(kept), "reverted": int(not kept), "missing": []})}

        def sleep(seconds):
            clock["now"] += seconds

        with patch.object(browser_vault_tool, "_eval_js_secret", side_effect=page), \
             patch.object(browser_vault_tool, "_eval_js", return_value={"success": True}), \
             patch.object(browser_vault_tool.time, "sleep", side_effect=sleep), \
             patch.object(browser_vault_tool.time, "monotonic", side_effect=lambda: clock["now"]):
            out = browser_vault_tool._read_back("t", [{"index": 1, "value": "x"}], "n", "https://example.com")
        assert out["success"] is False and out["error_type"] == "fill_reverted"

    def test_a_value_that_settles_after_a_mask_rewrite_is_a_fill(self, store):
        """The other half of stability: a mask that writes twice reads wrong first, then holds."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 1,
                     "label": "", "name": "pw", "type": "password"}]
        reads = []

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            if "reverted" in expression:
                reads.append(1)
                ok = len(reads) > 1
                return {"success": True, "result": json.dumps(
                    {"landed": int(ok), "reverted": int(not ok), "missing": []})}
            return {"success": True, "result": json.dumps({"filled": 1})}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, with_identifier=False))
        assert out["success"] is True and out["filled_fields"] == 1

    @pytest.mark.parametrize("failing", ["fill", "read-back"])
    def test_a_transport_error_quoting_the_script_never_carries_the_secret_out(self, store, caplog, failing):
        """Every secret-bearing script (the write AND the read-back, which carries the expected
        values) goes over the supervisor. A transport that raises with the expression in its message
        must not surface it in the tool result or the logs."""
        from tools import browser_vault_tool
        from tools.registry import registry

        meta = _add_login(store, origin="https://example.com")
        controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 1,
                     "label": "", "name": "pw", "type": "password"}]

        class _Sup:  # no insert_text: the JS write path
            def current_page_session(self):
                return "S1"

            def focus_page(self, origin, accept=None):
                return {"ok": True, "url": "https://example.com/login"}

            def evaluate_runtime(self, expression, *, session_id=None):
                if "location.href" in expression:
                    return {"ok": True, "result": "https://example.com/login"}
                if "s3cret-pw" not in expression:
                    return {"ok": True, "result": json.dumps(controls)}
                if ("reverted" in expression) == (failing == "read-back"):
                    raise RuntimeError(f"socket closed while sending {expression}")
                return {"ok": True, "result": json.dumps({"filled": 1})}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_ensure_supervisor", return_value=_Sup()), \
             patch.object(browser_vault_tool, "_page_backend", return_value=None), \
             patch.object(browser_vault_tool, "_bot_desktop_browser_session", return_value=False), \
             patch.object(browser_vault_tool, "_fenced_page_op", lambda tid, fn: fn()), \
             caplog.at_level("DEBUG"):
            raw = registry.dispatch("browser_vault", {"action": "fill", "handle": meta.id}, task_id="t")
        assert json.loads(raw)["success"] is False
        assert "s3cret-pw" not in raw
        assert "s3cret-pw" not in caplog.text

    def test_identifier_falls_back_to_a_single_unlabelled_text_box(self, store):
        """A framework login form can expose a phone box with no autocomplete, name, id, label or
        placeholder — nothing to classify. When the password's form holds exactly one other text
        box, that is the identifier; when it holds several, the fill writes none of them."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [
            {"formIndex": 0, "index": 0, "label": "", "name": "", "type": "text", "autocomplete": ""},
            {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        secret_exprs = []

        def fake_eval_secret(task_id, expression):
            secret_exprs.append(expression)
            if "reverted" in expression:
                return {"success": True, "result": json.dumps(
                    {"landed": expression.count('"index"'), "reverted": 0, "missing": []})}
            if "cleared" in expression:
                return {"success": True, "result": json.dumps({"cleared": 1})}
            return {"success": True, "result": json.dumps({"filled": expression.count('"index"')})}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is True and out["filled_fields"] == 2
        assert '"index": 0' in secret_exprs[0]      # the unlabelled box was included
        assert "user@example.com" in secret_exprs[0]

    def test_fill_writes_the_identifier_by_default(self, store):
        """A login fill covers both fields of a login. The identifier goes through the same trusted
        path as the password: a normal input tool writes the value into the DOM without the
        framework accepting it, and a controlled form then re-renders the field empty."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [
            {"autocomplete": "email", "formIndex": 0, "index": 0, "label": "", "name": "email", "type": "email"},
            {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
        ]
        typed = []

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            if "ok: true" in expression:  # the in-page origin assert (build_origin_check_js)
                return {"success": True, "result": json.dumps({"ok": True})}
            if 'refused: "focus_lost"' in expression:
                match = re.search(r'"index":\s*(\d+)', expression)
                index = int(match.group(1)) if match else -1
                return {"success": True, "result": json.dumps({"focused": index})}
            if "reverted" in expression:
                return {"success": True, "result": json.dumps(
                    {"landed": expression.count('"index"'), "reverted": 0, "missing": []})}
            if "cleared" in expression:
                return {"success": True, "result": json.dumps({"cleared": 1})}
            return {"success": True, "result": json.dumps({"filled": expression.count('"index"')})}

        class _Sup:
            def current_page_session(self):
                return "S1"

            def insert_text(self, text, **kwargs):
                typed.append(text)
                return {"ok": True}

            def focus_page(self, origin, accept=None):
                return {"ok": True, "url": "https://example.com/login"}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret), \
             patch.object(browser_vault_tool, "_ensure_supervisor", return_value=_Sup()), \
             patch.object(browser_vault_tool, "_page_backend", return_value=None):
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is True
        # The password is filled first (it anchors the form lookup), the identifier second
        assert typed == ["s3cret-pw", "user@example.com"]
        assert out["filled_fields"] == 2
        assert "s3cret-pw" not in raw

    def test_a_masked_field_is_retyped_key_by_key(self, store):
        """A bulk insert loses characters in a masked field (a phone number, a date). The fill reads
        the mismatch, clears the field, retypes it one character at a time, and only then reports
        success — the value that survives is the value that was verified."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 1,
                     "label": "", "name": "pw", "type": "password"}]
        typed = []
        verifies = []

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            if "ok: true" in expression:  # the in-page origin assert (build_origin_check_js)
                return {"success": True, "result": json.dumps({"ok": True})}
            if 'refused: "focus_lost"' in expression:
                return {"success": True, "result": json.dumps({"focused": 1})}
            if "reverted" in expression:
                verifies.append(expression)
                first = len(verifies) == 1
                return {"success": True, "result": json.dumps(
                    {"landed": 0, "reverted": 1, "missing": []} if first
                    else {"landed": 1, "reverted": 0, "missing": []})}
            if "cleared" in expression:
                return {"success": True, "result": json.dumps({"cleared": 1})}
            return {"success": True, "result": json.dumps({"filled": 1})}

        class _Sup:
            def current_page_session(self):
                return "S1"

            def insert_text(self, text, **kwargs):
                typed.append(text)
                return {"ok": True}

            def focus_page(self, origin, accept=None):
                return {"ok": True, "url": "https://example.com/login"}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret), \
             patch.object(browser_vault_tool, "_ensure_supervisor", return_value=_Sup()), \
             patch.object(browser_vault_tool, "_page_backend", return_value=None):
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is True
        assert typed[0] == "s3cret-pw"              # the bulk insert
        assert typed[1:] == list("s3cret-pw")       # the retype, one key at a time
        assert out["filled_fields"] == 1

    def test_fill_toctou_navigation_writes_nothing(self, store):
        """P1-2 schedule regression: inspection passes on the allowed origin,
        the page navigates before the fill script runs, the in-script origin
        assert refuses, and zero credential bytes are written."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [
            {"autocomplete": "current-password", "formIndex": 0, "index": 0, "label": "", "name": "pw", "type": "password"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                # Pre-check sees the allowed origin.
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            # The evaluated script itself must carry the origin assert.
            assert "window.location.origin" in expression
            assert '"https://example.com"' in expression
            # Simulate the page having navigated cross-origin by the time
            # the fill script executes: the script's own assert fires.
            return {
                "success": True,
                "result": json.dumps(
                    {"refused": "origin_changed", "found": "https://evil.com"}
                ),
            }

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is False
        assert out["error_type"] == "origin_changed"
        assert out.get("filled_fields", 0) == 0
        assert "s3cret-pw" not in raw

    def test_secret_eval_fails_closed_without_supervisor(self, store):
        """P1-1: the secret-bearing eval NEVER falls back to the argv path."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [
            {"autocomplete": "current-password", "formIndex": 0, "index": 0, "label": "", "name": "pw", "type": "password"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        # No supervisor registered and the task's session exposes no CDP endpoint → the fill refuses.
        # There is no other path the password could take (argv, a subprocess eval): nothing is attached
        # and nothing evaluates the secret-bearing script.
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch("tools.browser_session.session_for_task", return_value={}), \
             patch("tools.browser_supervisor.SUPERVISOR_REGISTRY") as reg:
            reg.get.return_value = None
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is False
        assert out["error_type"] == "supervisor_required"
        reg.get_or_start.assert_not_called()
        assert "s3cret-pw" not in json.dumps([str(c) for c in reg.mock_calls]) and "s3cret-pw" not in raw


    def test_vault_canary_redacted_from_browser_cdp_results(self, store):
        """P1-1 regression: a filled, non-token-shaped canary password must be
        unrecoverable through a model-facing browser_cdp-style result."""
        from agent import redact
        from agent.redact import redact_sensitive_text
        from tools import browser_vault_tool
        from tools.browser_cdp_tool import _redact_cdp_output

        canary = "plain sentence nobody would flag 7"
        meta = _add_login(store, origin="https://example.com", password=canary)
        controls = [
            {"autocomplete": "current-password", "formIndex": 0, "index": 0, "label": "", "name": "pw", "type": "password"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            if "reverted" in expression:  # read-back pass (build_fill_verify_js)
                return {"success": True, "result": json.dumps({"landed": 1, "reverted": 0, "missing": []})}
            return {"success": True, "result": json.dumps({"filled": 1})}

        try:
            with patch("agent.vault_store.get_vault_store", return_value=store), \
                 patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
                 patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
                out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
            assert out["success"] is True

            # Simulate a browser_cdp Runtime.evaluate sibling read echoing
            # the canary back (e.g. reading the input's value from the DOM).
            cdp_result = {
                "result": {"type": "string", "value": canary},
                "description": f"input value is {canary}",
            }
            scrubbed = _redact_cdp_output(cdp_result)
            assert canary not in json.dumps(scrubbed)
            assert "«redacted-vault-secret»" in json.dumps(scrubbed, ensure_ascii=False)

            # And the generic browser-result scrub catches it too, even with
            # user-level redaction preferences irrelevant (unconditional).
            assert canary not in redact_sensitive_text(f"page text: {canary}")
        finally:
            redact.clear_vault_redaction_values()

    def test_payment_fill_requires_confirmation_then_fills_card_fields(self, store):
        """A card is written only after the user confirms (a prompt injection reaching a checkout must not be
        able to spend); the secret eval then targets the classified card controls and the result carries
        the field tokens but never a value."""
        from tools import browser_vault_tool
        from agent import redact

        meta = store.add_item(kind="payment", label="Visa", origin="https://shop.test", secret=_CARD)
        controls = [
            {"autocomplete": "cc-number", "index": 0, "type": "text"},
            {"label": "Expiry (MM/YY)", "index": 1, "type": "text"},
            {"label": "CVC", "index": 2, "type": "text"},
            {"autocomplete": "email", "index": 3, "type": "email"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://shop.test/checkout"}
            return {"success": True, "result": json.dumps(controls)}

        secret_exprs = []

        def fake_eval_secret(task_id, expression):
            secret_exprs.append(expression)
            if "reverted" in expression:  # read-back pass
                return {"success": True, "result": json.dumps({"landed": 3, "reverted": 0, "missing": []})}
            return {"success": True, "result": json.dumps({"filled": 3})}

        try:
            with patch("agent.vault_store.get_vault_store", return_value=store), \
                 patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
                 patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret), \
                 patch("tools.approval_prompt.request_elicitation_consent", return_value="decline"):
                declined = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
            assert declined["success"] is False and declined["error_type"] == "payment_declined"
            assert secret_exprs == []

            with patch("agent.vault_store.get_vault_store", return_value=store), \
                 patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
                 patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret), \
                 patch("tools.approval_prompt.request_elicitation_consent", return_value="accept"):
                raw = browser_vault_tool.browser_vault_fill(meta.id)
            out = json.loads(raw)
            assert out["success"] is True and out["fields"] == ["cc-csc", "cc-exp", "cc-number"]
            assert _CARD["card_number"] not in raw and _CARD["cvc"] not in raw
            # the fill, then the read-back that proves the page kept it
            assert _CARD["card_number"] in secret_exprs[0] and "07/29" in secret_exprs[0]
            assert secret_exprs[1:] and all("reverted" in e for e in secret_exprs[1:])
            assert '"index": 3' not in secret_exprs[0]  # the email box is never a card target
            assert _CARD["card_number"] not in redact.redact_sensitive_text(f"dom says {_CARD['card_number']}")
        finally:
            redact.clear_vault_redaction_values()


class TestVaultHardening:
    """Read-deny, backup perms-tightening, canonical dir securing.

    Mirrors the browser-profile snapshot hardening (f1d05c): the vault dir
    holds key + ciphertext side by side, so it gets the same treatment.
    """

    def test_read_block_vault_dir_and_contents(self, tmp_path, monkeypatch):
        import agent.file_safety as fs

        home = tmp_path / "hermes_home"
        vault = home / "vault"
        vault.mkdir(parents=True)
        (vault / "vault.key").write_text("k", encoding="utf-8")
        (vault / "vault.json.enc").write_text("blob", encoding="utf-8")
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)

        for target in (vault, vault / "vault.key", vault / "vault.json.enc"):
            err = fs.get_read_block_error(str(target))
            assert err is not None, f"expected read deny for {target}"

    def test_read_block_leaves_sibling_dirs_alone(self, tmp_path, monkeypatch):
        import agent.file_safety as fs

        home = tmp_path / "hermes_home"
        other = home / "vaults-notes"
        other.mkdir(parents=True)
        f = other / "notes.txt"
        f.write_text("hi", encoding="utf-8")
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
        assert fs.get_read_block_error(str(f)) is None

    def test_backup_secret_names_include_vault_files(self):
        from hermes_cli.backup import _SECRET_FILE_NAMES

        assert "vault.key" in _SECRET_FILE_NAMES
        assert "vault.json.enc" in _SECRET_FILE_NAMES

    def test_vault_dir_is_owner_only(self, store, tmp_path):
        old_umask = os.umask(0o022)
        try:
            _add_login(store)
        finally:
            os.umask(old_umask)
        mode = stat.S_IMODE(os.stat(tmp_path / "vault").st_mode)
        assert not mode & 0o077, oct(mode)




def test_every_registered_tool_schema_declares_openai_style_parameters():
    """The registry emits ``parameters`` (OpenAI function shape) and every provider adapter converts from
    it; a schema that spells it ``input_schema`` (Anthropic shape) ships with NO parameters, so the model is
    told the tool takes nothing and calls browser_vault_fill without a handle."""
    import model_tools
    from tools.registry import registry

    missing = [entry.name for entry in registry.get_all_entries()
               if "parameters" not in entry.schema or "input_schema" in entry.schema]
    assert not missing, missing


class TestSaveLoginPrompt:
    """browser_vault_save_login: the surface prompt supplies the login, the tool stores it bound to the page
    origin and fills. The password must never come back in the tool result."""

    def test_saves_to_page_origin_and_never_echoes_the_password(self, store, monkeypatch):
        from agent.vault_backends import unlock as unlock_mod
        from tools import browser_vault_tool

        seen = {}

        def prompt(origin, site):
            seen["origin"], seen["site"] = origin, site
            return {"identifier": "tek@acme.test", "password": "hunter2-very-secret"}

        unlock_mod.set_save_login_prompt_callback(prompt)
        monkeypatch.setattr(browser_vault_tool, "_current_page_origin", lambda task_id: "https://acme.test")
        monkeypatch.setattr(browser_vault_tool, "browser_vault_fill",
                            lambda handle, task_id=None: json.dumps({"success": True, "filled_fields": 1}))
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch("agent.vault_backends.unlock.can_prompt_here", return_value=True):
            out = json.loads(browser_vault_tool.browser_vault_save_login(task_id="t1"))
        unlock_mod.set_save_login_prompt_callback(None)

        assert out["success"] is True and out["identifier"] == "tek@acme.test"
        assert "hunter2" not in json.dumps(out)
        assert seen == {"origin": "https://acme.test", "site": "acme.test"}
        [meta] = store.list_items()
        assert meta.origin == "https://acme.test" and meta.identifier == "tek@acme.test"

    def test_declined_or_headless_stores_nothing(self, store, monkeypatch):
        from agent.vault_backends import unlock as unlock_mod
        from tools import browser_vault_tool

        monkeypatch.setattr(browser_vault_tool, "_current_page_origin", lambda task_id: "https://acme.test")
        with patch("agent.vault_store.get_vault_store", return_value=store):
            unlock_mod.set_save_login_prompt_callback(lambda origin, site: None)
            with patch("agent.vault_backends.unlock.can_prompt_here", return_value=True):
                declined = json.loads(browser_vault_tool.browser_vault_save_login())
            with patch("agent.vault_backends.unlock.can_prompt_here", return_value=False):
                headless = json.loads(browser_vault_tool.browser_vault_save_login())
            unlock_mod.set_save_login_prompt_callback(None)
        assert declined["error_type"] == "save_declined"
        assert headless["error_type"] == "prompt_unavailable"
        assert store.list_items() == []


class TestManagerAutoDetection:
    def test_installed_manager_is_a_source_without_config_and_config_can_opt_out(self):
        from agent.vault_backends import base

        with patch.object(base, "is_installed", return_value=True):
            with patch.object(base, "_cfg", return_value={}):
                assert {b.name for b in base.enabled_backends()} == {"local", "onepassword", "bitwarden"}
            with patch.object(base, "_cfg", return_value={"bitwarden": {"enabled": False}}):
                assert {b.name for b in base.enabled_backends()} == {"local", "onepassword"}
        with patch.object(base, "is_installed", return_value=False), patch.object(base, "_cfg", return_value={}):
            assert [b.name for b in base.enabled_backends()] == ["local"]


def test_every_vault_tool_is_in_the_browser_toolset():
    """toolsets.py is a hand-maintained list; a tool registered here but missing there is invisible to the model
    (live: browser_vault_save_login was registered, tested, and never offered)."""
    import toolsets
    from tools import browser_vault_tool
    from tools.registry import registry

    registered = {e.name for e in registry.get_all_entries() if e.name.startswith("browser_vault_")}
    assert registered <= set(toolsets.TOOLSETS["browser"]["tools"]), registered - set(toolsets.TOOLSETS["browser"]["tools"])


class TestTwoFactor:
    def test_totp_matches_rfc6238_vector_and_seed_normalisation(self):
        from agent.vault_store import VaultError, normalize_otp_secret, totp_now

        seed = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"  # "12345678901234567890"
        assert totp_now(seed, digits=8, at=59) == "94287082"
        assert totp_now(seed, at=1111111109) == "081804"
        assert normalize_otp_secret("otpauth://totp/GitHub:tek?secret=jbsw y3dp ehpk3pxp&issuer=GitHub") == "JBSWY3DPEHPK3PXP"
        with pytest.raises(VaultError):
            normalize_otp_secret("not base32!")
        # Non-default otpauth parameters are kept and honoured (RFC 6238 SHA-256 / 8-digit vector at T=59).
        stored = normalize_otp_secret(f"otpauth://totp/x?secret={'GEZDGNBVGY3TQOJQ' * 2}&digits=8&period=30&algorithm=SHA256")
        assert stored.endswith("|8|30|SHA256")
        sha256_seed = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZA"  # "1234567890" * 3.2 -> RFC 32-byte seed
        assert totp_now(sha256_seed + "|8|30|SHA256", at=59) == "46119246"
        assert totp_now("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ|6|60|SHA1", at=119) == totp_now("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", period=60, at=119)
        with pytest.raises(VaultError):
            normalize_otp_secret("otpauth://hotp/x?secret=JBSWY3DPEHPK3PXP&counter=1")

    def test_saved_authenticator_key_mints_codes_without_asking(self, store, monkeypatch):
        """The whole point: with a seed on the login, enter_code never prompts and the code never comes back."""
        from agent.vault_backends import unlock as unlock_mod
        from tools import browser_vault_tool

        meta = store.add_item("login", "gh", {"identifier_type": "username", "identifier": "tek", "password": "pw",
                                              "otp_secret": "JBSWY3DPEHPK3PXP"}, origin="https://github.com")
        assert store.get_meta(meta.id).has_otp is True
        asked = []
        unlock_mod.set_code_prompt_callback(lambda site, hint: asked.append(site) or "000000")
        controls = [{"index": 0, "type": "text", "name": "otp", "label": "Authentication code", "autocomplete": "one-time-code"}]
        seen = {}

        def fake_eval(task_id, expr):
            return {"success": True, "result": json.dumps(controls) if "querySelectorAll" in expr else "https://github.com/sessions/two-factor"}

        def fake_secret(task_id, expr):
            seen["expr"] = expr
            if "reverted" in expr:  # read-back pass (build_fill_verify_js)
                return {"success": True, "result": json.dumps({"landed": 1, "reverted": 0, "missing": []})}
            return {"success": True, "result": json.dumps({"filled": 1})}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_focus_bound_origin", lambda *a, **k: None), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_secret):
            raw = browser_vault_tool.browser_vault_enter_code(meta.id, task_id="t")
        unlock_mod.set_code_prompt_callback(None)
        out = json.loads(raw)
        assert out["success"] and out["source"] == "local" and asked == []
        code = re.search(r'"value": "(\d{6})"', seen["expr"]).group(1)
        assert code not in raw  # the code went to the page, not to the model

    def test_without_a_key_the_user_is_asked_and_split_boxes_get_one_digit_each(self, store):
        from agent.vault_backends import unlock as unlock_mod
        from tools import browser_vault_tool

        unlock_mod.set_code_prompt_callback(lambda site, hint: "246 810")
        boxes = [{"index": i, "type": "tel", "name": f"digit{i}", "label": "", "autocomplete": "one-time-code",
                  "formIndex": 0, "maxLength": 1} for i in range(6)]
        seen = {}
        fake_eval = lambda t, e: {"success": True, "result": json.dumps(boxes) if "querySelectorAll" in e else "https://acme.test/2fa"}

        def fake_secret(t, e):
            seen["expr"] = e
            return {"success": True, "result": json.dumps({"filled": 6})}

        with patch("agent.vault_backends.unlock.can_prompt_here", return_value=True), \
             patch.object(browser_vault_tool, "_focus_bound_origin", lambda *a, **k: None), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_secret):
            out = json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))
            unlock_mod.set_code_prompt_callback(lambda site, hint: "")
            declined = json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))
        unlock_mod.set_code_prompt_callback(None)
        assert out["success"] and out["source"] == "user" and out["filled_fields"] == 6
        assert re.findall(r'"value": "(\d)"', seen["expr"]) == list("246810")
        assert declined["error_type"] == "code_declined"

    def test_several_code_like_inputs_that_are_not_a_digit_widget_get_one_field(self):
        """Reviewer case: a page with 4+ code-ish inputs (promo code, zip code, a real OTP box...) must never
        get a digit sprayed across them. Only an unmistakable maxlength=1 same-form adjacent group splits."""
        from agent.vault_login_classifier import ClassifiedLoginControl, LoginControl, build_otp_fills

        def ctl(i, form=0, maxlen=None, score=70):
            return ClassifiedLoginControl(LoginControl("", form, i, "", f"code{i}", "text", maxlen), score, "one-time-code")

        scattered = [ctl(0), ctl(3), ctl(7), ctl(9, form=1), ctl(12, score=100)]
        assert build_otp_fills(scattered, "246810") == [{"index": 12, "token": "one-time-code", "value": "246810"}]
        # maxlength=1 but different forms / non-adjacent: still one field
        assert len(build_otp_fills([ctl(i, form=i % 2, maxlen=1) for i in range(6)], "246810")) == 1
        assert len(build_otp_fills([ctl(i * 2, maxlen=1) for i in range(6)], "246810")) == 1
        # five boxes for a six-digit code: one field
        assert len(build_otp_fills([ctl(i, maxlen=1) for i in range(5)], "246810")) == 1
        # the real widget
        assert [f["value"] for f in build_otp_fills([ctl(i + 4, maxlen=1) for i in range(6)], "246810")] == list("246810")

    def test_no_code_field_points_at_passkey_or_device_approval(self):
        from tools import browser_vault_tool

        fake_eval = lambda t, e: {"success": True, "result": json.dumps([{"index": 0, "type": "text", "name": "q", "label": "Search", "autocomplete": ""}]) if "querySelectorAll" in e else "https://acme.test/approve"}
        with patch.object(browser_vault_tool, "_focus_bound_origin", lambda *a, **k: None), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval):
            out = json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))
        assert out["error_type"] == "no_code_field"