"""Phase 1 vault kind ``identity`` (SSN / tax / passport) — origin-bound fill.

Focused RED/GREEN contract for issue #107704. Rides the existing vault; does
not change login/payment/address semantics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.vault_login_classifier import LoginControl  # noqa: E402
from agent.vault_store import VaultError, VaultStore  # noqa: E402


ORIGIN = "https://gov.test"
SSN_A = "111-22-3333"
SSN_B = "999-88-7777"


@pytest.fixture()
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


@pytest.fixture(autouse=True)
def _reset_identity_freeze():
    try:
        from agent.vault_identity_freeze import clear_identity_freeze
    except ImportError:
        yield
        return
    clear_identity_freeze()
    yield
    clear_identity_freeze()


def _enable_identity():
    return patch("agent.vault_store.identity_vault_enabled", return_value=True)


def _add_identity(store, *, origin=ORIGIN, ssn=SSN_A, **extra):
    secret = {"ssn": ssn, **extra}
    with _enable_identity():
        return store.add_item(kind="identity", label="Tax ID", origin=origin, secret=secret)


def _ctrl(**kw):
    base = dict(autocomplete="", form_index=0, index=0, label="", name="", type="text")
    base.update(kw)
    return LoginControl(**base)


def _ssn_controls():
    return [
        {"autocomplete": "ssn", "index": 0, "type": "text", "label": "SSN", "name": "ssn"},
        {"autocomplete": "email", "index": 1, "type": "email", "label": "Email", "name": "email"},
    ]


def _page_eval(origin=ORIGIN, controls=None):
    controls = controls if controls is not None else _ssn_controls()

    def fake_eval(task_id, expression):
        if "location.href" in expression:
            return {"success": True, "result": f"{origin}/form"}
        return {"success": True, "result": json.dumps(controls)}

    return fake_eval


# ---------------------------------------------------------------------------
# 1–2. Kind / store
# ---------------------------------------------------------------------------

class TestIdentityKindAndStore:
    def test_identity_kind_add_enabled_and_disabled_refuses(self, store):
        from agent.vault_store import IDENTITY_FIELDS, VAULT_KINDS

        assert "identity" in VAULT_KINDS
        assert set(IDENTITY_FIELDS) >= {"ssn", "tax_id", "itin", "ein", "national_id", "passport_number"}

        with pytest.raises(VaultError) as disabled:
            store.add_item(kind="identity", label="Tax ID", origin=ORIGIN, secret={"ssn": SSN_A})
        assert "identity" in str(disabled.value).lower()

        meta = _add_identity(store)
        with _enable_identity():
            secret = store.resolve_secret(meta.id)
        assert secret == {"ssn": SSN_A}
        assert meta.origin == ORIGIN
        assert meta.kind == "identity"

    def test_missing_origin_or_empty_payload_refuses(self, store):
        with _enable_identity():
            with pytest.raises(VaultError):
                store.add_item(kind="identity", label="Tax ID", secret={"ssn": SSN_A})
            with pytest.raises(VaultError):
                store.add_item(kind="identity", label="Tax ID", origin=ORIGIN, secret={})
            with pytest.raises(VaultError):
                store.add_item(
                    kind="identity",
                    label="Tax ID",
                    origin=ORIGIN,
                    secret={"pdf": "bytes", "image": "x.png", "file": "/tmp/scan.pdf"},
                )
            meta = store.add_item(
                kind="identity",
                label="Tax ID",
                origin=ORIGIN,
                secret={"ssn": SSN_A, "pdf": "do-not-store", "image": "x.png"},
            )
            assert "pdf" not in store.resolve_secret(meta.id)
            assert "image" not in store.resolve_secret(meta.id)


# ---------------------------------------------------------------------------
# 3. Classifier
# ---------------------------------------------------------------------------

class TestIdentityClassifier:
    def test_ssn_label_and_autocomplete_not_password_or_cc(self):
        from agent.vault_login_classifier import classify_identity_control

        label = classify_identity_control(_ctrl(label="Social Security Number"))
        assert label is not None and label.token == "ssn" and label.score == 70

        auto = classify_identity_control(_ctrl(autocomplete="ssn"))
        assert auto is not None and auto.token == "ssn" and auto.score == 100

        assert classify_identity_control(_ctrl(type="password")) is None
        assert classify_identity_control(_ctrl(autocomplete="cc-number")) is None
        assert classify_identity_control(_ctrl(type="email")) is None
        assert classify_identity_control(_ctrl(autocomplete="address-line1")) is None


# ---------------------------------------------------------------------------
# 4–6, 10. Fill confirm / redaction / canary
# ---------------------------------------------------------------------------

def _fill(store, meta, *, consent="accept", eval_secret=None, origin=ORIGIN):
    from tools import browser_vault_tool

    secret_exprs = []

    def fake_secret(task_id, expression):
        secret_exprs.append(expression)
        if eval_secret is not None:
            return eval_secret(task_id, expression)
        return {"success": True, "result": json.dumps({"filled": 1})}

    with _enable_identity(), \
         patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(browser_vault_tool, "_focus_bound_origin", return_value=origin), \
         patch.object(browser_vault_tool, "_eval_js", side_effect=_page_eval(origin)), \
         patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_secret), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value=consent) as consent_fn:
        raw = browser_vault_tool.browser_vault_fill(meta.id)
    return json.loads(raw), raw, secret_exprs, consent_fn


class TestIdentityFill:
    def test_decline_is_identity_declined_and_does_not_eval_secret(self, store):
        meta = _add_identity(store)
        out, raw, secret_exprs, _ = _fill(store, meta, consent="decline")
        assert out["success"] is False
        assert out["error_type"] == "identity_declined"
        assert secret_exprs == []
        assert SSN_A not in raw

    def test_accept_fills_tokens_not_values_and_registers_redaction(self, store):
        from agent import redact

        meta = _add_identity(store)
        try:
            out, raw, secret_exprs, consent_fn = _fill(store, meta, consent="accept")
            assert out["success"] is True
            assert out["kind"] == "identity"
            assert out["origin"] == ORIGIN
            assert out["filled_fields"] == 1
            assert "ssn" in out["fields"]
            assert SSN_A not in raw
            assert SSN_A not in json.dumps(out)
            assert len(secret_exprs) == 1
            assert SSN_A in secret_exprs[0]
            kwargs = consent_fn.call_args
            title = kwargs[0][0] if kwargs.args else ""
            assert "SSN" in title and ORIGIN in title
            assert kwargs.kwargs.get("surface") == "vault-identity"
            assert SSN_A not in redact.redact_sensitive_text(f"dom says {SSN_A}")
        finally:
            redact.clear_vault_redaction_values()

    def test_headless_non_accept_refuses(self, store):
        meta = _add_identity(store)
        out, raw, secret_exprs, _ = _fill(store, meta, consent="cancel")
        assert out["success"] is False
        assert out["error_type"] == "identity_declined"
        assert secret_exprs == []
        assert SSN_A not in raw

    def test_disabled_config_refuses_fill(self, store):
        from tools import browser_vault_tool

        meta = _add_identity(store)
        with patch("agent.vault_store.identity_vault_enabled", return_value=False), \
             patch("agent.vault_store.get_vault_store", return_value=store), \
             patch("tools.approval_prompt.request_elicitation_consent") as consent, \
             patch.object(browser_vault_tool, "_eval_js_secret") as secret_eval:
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
        assert out["success"] is False
        assert out["error_type"] == "identity_disabled"
        consent.assert_not_called()
        secret_eval.assert_not_called()

    def test_canary_two_ssns_results_indistinguishable(self, store):
        from agent import redact

        try:
            meta_a = _add_identity(store, ssn=SSN_A)
            out_a, raw_a, _, _ = _fill(store, meta_a)
            meta_b = _add_identity(store, ssn=SSN_B)
            out_b, raw_b, _, _ = _fill(store, meta_b)
            for raw, ssn in ((raw_a, SSN_A), (raw_b, SSN_B), (raw_a, SSN_B), (raw_b, SSN_A)):
                assert ssn not in raw
            keep = ("filled_fields", "origin", "kind", "fields")
            assert {k: out_a[k] for k in keep} == {k: out_b[k] for k in keep}
            rest_a = {k: v for k, v in out_a.items() if k not in keep}
            rest_b = {k: v for k, v in out_b.items() if k not in keep}
            assert rest_a == rest_b
        finally:
            redact.clear_vault_redaction_values()


# ---------------------------------------------------------------------------
# 7–9. Vision freeze
# ---------------------------------------------------------------------------

class TestIdentityVisionFreeze:
    def _successful_fill(self, store):
        from agent import redact

        meta = _add_identity(store)
        out, _, _, _ = _fill(store, meta, consent="accept")
        assert out["success"] is True
        return meta

    def test_browser_vision_frozen_on_same_origin_no_screenshot(self, store):
        from tools import browser_tool

        self._successful_fill(store)
        with patch("tools.browser_vault_tool._current_page_origin", return_value=ORIGIN), \
             patch.object(browser_tool, "_capture_vision_screenshot") as capture, \
             patch.object(browser_tool, "_is_camofox_mode", return_value=False):
            raw = browser_tool.browser_vision("what is on the page?", task_id="t")
        out = json.loads(raw) if isinstance(raw, str) else raw
        assert out["success"] is False
        assert out.get("error_type")
        assert "screenshot" not in json.dumps(out).lower() or "screenshot" in (out.get("error") or "").lower()
        assert not out.get("analysis")
        assert not out.get("screenshot_path")
        capture.assert_not_called()

    def test_cdp_evaluate_input_value_frozen(self, store):
        from tools import browser_cdp_tool

        self._successful_fill(store)
        with patch("tools.browser_vault_tool._current_page_origin", return_value=ORIGIN), \
             patch.object(browser_cdp_tool, "_run_async") as run, \
             patch.object(browser_cdp_tool, "_resolve_cdp_endpoint", return_value="ws://localhost:1"):
            raw = browser_cdp_tool.browser_cdp(
                method="Runtime.evaluate",
                params={"expression": "document.querySelector('input').value"},
                task_id="t",
            )
        out = json.loads(raw)
        assert out["success"] is False
        assert out.get("error_type")
        run.assert_not_called()

    def test_origin_change_clears_freeze(self, store):
        from tools import browser_cdp_tool, browser_tool

        self._successful_fill(store)
        other = "https://other.test"
        with patch("tools.browser_vault_tool._current_page_origin", return_value=other), \
             patch.object(browser_tool, "_is_camofox_mode", return_value=False), \
             patch.object(browser_tool, "_blocked_private_page_content", return_value=None), \
             patch.object(
                 browser_tool,
                 "_capture_vision_screenshot",
                 return_value=({"success": True, "data": {}}, Path("/tmp/ok.png"), None),
             ), \
             patch("tools.vision_tools._should_use_native_vision_fast_path", return_value=False), \
             patch.object(browser_tool, "_vision") as vision:
            vision._lightpanda_vision_preroute.return_value = (False, None, Path("/tmp/ok.png"))
            vision._analyze_screenshot_with_aux_llm.return_value = "ok page"
            raw = browser_tool.browser_vision("what", task_id="t")
        out = json.loads(raw) if isinstance(raw, str) else raw
        assert out.get("error_type") not in {"identity_vision_frozen", "identity_frozen"}
        assert out.get("success") is not False or "frozen" not in str(out).lower()

        with patch("tools.browser_vault_tool._current_page_origin", return_value=other), \
             patch.object(browser_cdp_tool, "_browser_cdp_private_guard", return_value=None), \
             patch.object(browser_cdp_tool, "_resolve_cdp_endpoint", return_value="ws://localhost:1"), \
             patch.object(browser_cdp_tool, "_run_async", return_value={"result": {"type": "string", "value": "x"}}):
            cdp = json.loads(browser_cdp_tool.browser_cdp(
                method="Runtime.evaluate",
                params={"expression": "document.querySelector('input').value"},
                task_id="t",
            ))
        assert cdp.get("success") is True
        assert cdp.get("error_type") not in {"identity_vision_frozen", "identity_frozen"}
