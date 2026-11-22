"""Every vault operation that inspected the page removes its own inspection stamps on every exit.

Each operation stamps under its own ``data-hermes-vault-slot-<nonce>`` attribute, so nothing else
ever removes them; a stamp left behind lets a later read address the control the vault targeted.
"""

from __future__ import annotations

import json
import re
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore


@pytest.fixture()
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


def _add_login(store, origin="https://example.com"):
    return store.add_item(kind="login", label="Example login", origin=origin,
                          secret={"identifier_type": "email", "identifier": "user@example.com",
                                  "password": "s3cret-pw", "origin": origin})


_CODE_BOX = {"index": 0, "type": "text", "name": "otp", "label": "Code", "autocomplete": "one-time-code"}
_SEARCH_BOX = {"index": 0, "type": "search", "name": "q", "label": "Search", "autocomplete": "off"}
_USER_BOX = {"autocomplete": "username", "formIndex": 0, "index": 0, "label": "", "name": "user", "type": "text"}
_PW_BOX = {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw",
           "type": "password"}


@pytest.mark.parametrize("path, controls, expected", [
    ("enter_code", [_SEARCH_BOX], "no_code_field"),
    ("enter_code", [_CODE_BOX], "prompt_unavailable"),
    ("enter_code", [_CODE_BOX], "code_declined"),
    ("fill", [_SEARCH_BOX], "no_form_field"),
    ("fill", [_USER_BOX], "no_fillable_field"),
    ("fill", [_PW_BOX], "origin_changed"),
    ("fill", [_USER_BOX], "unstamp_failed"),
])
def test_every_exit_after_the_inspection_removes_its_stamps(store, path, controls, expected):
    """Each operation stamps with its own attribute name, so nothing else ever removes them: a
    cancelled code, a page with no code or login field, a session that cannot ask, and a refused
    write must all leave the page without the stamps. A failed removal does not replace the result."""
    from agent.vault_backends import unlock as unlock_mod
    from tools import browser_vault_tool

    page = {"stamped": False, "nonce": None}

    def run(expression):
        if 'setAttribute("data-hermes-vault-slot-"' in expression:
            page["stamped"] = True
            page["nonce"] = re.search(r'const nonce = "(\w+)"', expression).group(1)
            return {"success": True, "result": json.dumps(controls)}
        if "removeAttribute(slot)" in expression and "unstamped" in expression:
            if expected == "unstamp_failed":
                return {"success": False, "error": "the page went away"}
            if f'"data-hermes-vault-slot-{page["nonce"]}"' in expression:  # this inspection's own stamps
                page["stamped"] = False
            return {"success": True, "result": {"unstamped": True}}
        if "location.href" in expression:
            return {"success": True, "result": "https://example.com/login"}
        if "filled }" in expression and expected == "origin_changed":
            return {"success": True, "result": {"refused": "origin_changed", "found": "https://evil.test"}}
        if "filled }" in expression:
            return {"success": True, "result": {"filled": 1}}
        if "reverted" in expression:
            return {"success": True, "result": {"landed": 1, "reverted": 0, "missing": []}}
        return {"success": True, "result": False}

    meta = _add_login(store, origin="https://example.com")
    unlock_mod.set_code_prompt_callback(lambda site, hint: "")
    try:
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch("agent.vault_backends.unlock.can_prompt_here", return_value=expected != "prompt_unavailable"), \
             patch.object(browser_vault_tool, "_focus_bound_origin", lambda *a, **k: None), \
             patch.object(browser_vault_tool, "_FILL_VERIFY_STABLE_S", 0), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=lambda t, e: run(e)), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=lambda t, e: run(e)):
            if path == "enter_code":
                out = json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))
            else:
                out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t", with_identifier=False))
    finally:
        unlock_mod.set_code_prompt_callback(None)
    assert out["success"] is False, out
    if expected == "unstamp_failed":  # the removal failed: the fill's own refusal is still the result
        assert out["error"].startswith("No fillable login field"), out
        return
    if path == "enter_code" or expected == "origin_changed":
        assert out["error_type"] == expected
    assert page["stamped"] is False


def test_an_operation_that_raises_after_the_inspection_removes_its_stamps(store):
    """A plugin manager can fail while the password is resolved, after the page was stamped: the
    error propagates to the tool's ``backend_unavailable`` refusal, and the stamps go on the way out."""
    from agent.vault_backends.local import LocalLoginBackend
    from agent.vault_backends.plugin import PluginBackendError
    from tools import browser_vault_tool

    page = {"stamped": False, "nonce": None}

    def run(expression):
        if 'setAttribute("data-hermes-vault-slot-"' in expression:
            page["stamped"] = True
            page["nonce"] = re.search(r'const nonce = "(\w+)"', expression).group(1)
            return {"success": True, "result": json.dumps([_USER_BOX, _PW_BOX])}
        if "removeAttribute(slot)" in expression and f'"data-hermes-vault-slot-{page["nonce"]}"' in expression:
            page["stamped"] = False
            return {"success": True, "result": {"unstamped": True}}
        if "location.href" in expression:
            return {"success": True, "result": "https://example.com/login"}
        return {"success": True, "result": False}

    meta = _add_login(store, origin="https://example.com")
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(LocalLoginBackend, "resolve_password", side_effect=PluginBackendError("manager failed")), \
         patch.object(browser_vault_tool, "_focus_bound_origin", lambda *a, **k: None), \
         patch.object(browser_vault_tool, "_eval_js", side_effect=lambda t, e: run(e)), \
         patch.object(browser_vault_tool, "_eval_js_secret", side_effect=lambda t, e: run(e)):
        out = json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))
    assert out["error_type"] == "backend_unavailable", out
    assert page["nonce"] is not None and page["stamped"] is False
