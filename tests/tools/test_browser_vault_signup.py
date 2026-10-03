"""Signup-only vault fill regression and security contracts."""
import json
import shutil
import subprocess
from unittest.mock import patch

import pytest

from agent.vault_login_classifier import LoginControl, build_fill_js, classify_login_control, select_signup_fills
from agent.vault_store import VaultStore
from tools import browser_vault_tool as tool


def control(index, autocomplete="", name="", form=0, label="", kind="password"):
    return {"index": index, "autocomplete": autocomplete, "name": name,
            "label": label, "formIndex": form, "type": kind}


@pytest.mark.parametrize("fields,expected", [
    ([control(0, "new-password"), control(1, "new-password", name="Confirm password")], [0, 1]),
    ([control(0, "new-password"), control(1, "new-password")], []),
    ([control(0, name="Create password"), control(1, name="Confirm password")], [0, 1]),
    ([control(0, "new-password")], [0]),
    ([control(0, "current-password"), control(1, "new-password"), control(2, name="Repeat password")], []),
    ([control(0, name="Password")], []),
    ([control(0, "current-password")], []),
    ([control(0, "new-password"), control(1, name="Confirm password", form=1)], [0]),
    ([control(0, "new-password"), control(1, "new-password", form=1)], []),
])
def test_signup_selection(fields, expected):
    chosen = select_signup_fills([LoginControl.from_dict(x) for x in fields], "fixture-only-password")
    assert [f["index"] for f in chosen] == expected
    assert all(f["value"] == "fixture-only-password" for f in chosen)


def test_login_classifier_still_excludes_new_password():
    assert classify_login_control(LoginControl.from_dict(control(0, "new-password"))) is None
    assert classify_login_control(LoginControl.from_dict(control(0, name="Confirm password"))) is None


def test_signup_registry_dispatch_origin_and_secret_redaction(tmp_path):
    from agent import redact
    from tools.registry import registry
    store = VaultStore(base_dir=tmp_path / "vault")
    password = "fixture-only-password"
    meta = store.add_item("login", "Example", {"identifier_type": "email", "identifier": "a@example.test",
                                                "password": password}, origin="https://example.test")
    fields = [control(0, "new-password"), control(1, name="Confirm password")]
    expressions = []

    def eval_page(_task, expr):
        return {"success": True, "result": "https://example.test/signup" if "location.href" in expr else json.dumps(fields)}

    def eval_secret(_task, expr):
        expressions.append(expr)
        return {"success": True, "result": json.dumps({"filled": 2})}

    try:
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(tool, "_current_page_origin", return_value="https://example.test"), \
             patch.object(tool, "_focus_bound_origin", return_value="https://example.test") as focus, \
             patch.object(tool, "_eval_js", side_effect=eval_page), \
             patch.object(tool, "_eval_js_secret", side_effect=eval_secret), \
             patch.object(tool, "_fenced_page_op", side_effect=lambda tid, fn: fn()):
            result = json.loads(tool._handle_vault_fill({"handle": meta.id, "mode": "signup"}, task_id="test"))
            refused = json.loads(tool._handle_vault_fill({"handle": meta.id, "mode": "signup", "password": password}, task_id="test"))
        assert result["success"] and result["filled_fields"] == 2
        focus.assert_called_once_with("test", "https://example.test", "signup")
        assert password not in json.dumps(result)
        assert '"token": "current-password"' not in expressions[0]
        assert '"https://example.test"' in expressions[0] and "window.location.origin" in expressions[0]
        assert password not in redact.redact_sensitive_text(password)
        assert refused["error_type"] == "invalid_arguments" and len(expressions) == 1
    finally:
        redact.clear_vault_redaction_values()


def test_signup_fill_script_origin_and_field_guard():
    """Execute the real generated JS: origin and current-password checks precede writes."""
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    fills = [{"index": 0, "token": "new-password", "value": "fixture-only-password"},
             {"index": 1, "token": "confirm-password", "value": "fixture-only-password"}]
    expression = build_fill_js(fills, expected_origin="https://example.test", nonce="nonce")
    harness = '''
const form = {};
const slots = [
  {type: "password", name: "new_password", autocomplete: "new-password", form},
  {type: "password", name: "confirm_password", autocomplete: "new-password", form}
].map((props) => ({...props, tagName: "INPUT", value: "", focus() {},
  getAttribute(key) { return key === "autocomplete" ? this.autocomplete : null; },
  dispatchEvent() {}, removeAttribute() {}}));
globalThis.window = {location: {origin: process.argv[1]}};
globalThis.document = {querySelector(selector) {
  return slots[Number(selector.match(/nonce:(\\d+)/)[1])];
}, querySelectorAll() {return slots;}};
globalThis.HTMLInputElement = function() {};
globalThis.InputEvent = class {};
globalThis.Event = class {};
const result = EXPRESSION;
console.log(JSON.stringify({result: JSON.parse(result), values: slots.map(s => s.value)}));
'''.replace("EXPRESSION", expression)
    def execute(origin):
        run = subprocess.run(["node", "-e", harness, origin], capture_output=True, text=True, check=True)
        return json.loads(run.stdout)
    assert execute("https://evil.test")["values"] == ["", ""]
    success = execute("https://example.test")
    assert success["values"] == ["fixture-only-password"] * 2
    assert success["result"]["filled"] == 2
    # A change-password form is not a signup form, even if it has new-password inputs.
    change = harness.replace('globalThis.window =',
                             'slots.push({...slots[0], autocomplete: "current-password"});\nglobalThis.window =')
    result = subprocess.run(["node", "-e", change, "https://example.test"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["values"][:2] == ["", ""]
    # Only aria-labelledby supplies the signup intent; the synchronous fill guard must
    # recognize the same evidence the inspection classifier used.
    aria = harness.replace('name: "new_password", autocomplete: "new-password"', 'name: "", autocomplete: ""')
    aria = aria.replace('name: "confirm_password", autocomplete: "new-password"', 'name: "", autocomplete: ""')
    aria = aria.replace('return key === "autocomplete" ? this.autocomplete : null;',
                        'return key === "autocomplete" ? this.autocomplete : '
                        '(key === "aria-labelledby" ? (this === slots[0] ? "new-label" : "confirm-label") : null);')
    aria = aria.replace('document = {querySelector(selector) {',
                        'document = {getElementById(id) { return {textContent: id === "new-label" ? '
                        '"Create password" : "Confirm password"}; }, querySelector(selector) {')
    result = subprocess.run(["node", "-e", aria, "https://example.test"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["values"] == ["fixture-only-password"] * 2
    # A new-password control replaced by a current-password control at write time is refused.
    altered = harness.replace('autocomplete: "new-password"', 'autocomplete: "current-password"', 1)
    run = subprocess.run(["node", "-e", altered, "https://example.test"], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)["values"] == ["", ""]
    heuristic = harness.replace('autocomplete: "new-password"', 'autocomplete: ""')
    run = subprocess.run(["node", "-e", heuristic, "https://example.test"], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)["values"] == ["fixture-only-password"] * 2
    mutated = harness.replace('dispatchEvent() {}, removeAttribute() {}',
                              'dispatchEvent() { if (this === slots[0]) { slots[1].type = "text"; slots[1].autocomplete = "current-password"; } }, removeAttribute() {}')
    run = subprocess.run(["node", "-e", mutated, "https://example.test"], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)["values"] == ["fixture-only-password", ""]
    moved = harness.replace('focus() {},', 'focus() { if (this === slots[1]) this.form = {}; },')
    run = subprocess.run(["node", "-e", moved, "https://example.test"], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)["values"] == ["fixture-only-password", ""]


def test_formless_password_controls_are_not_a_pair():
    fields = [control(0, "new-password"), control(1, name="Confirm password")]
    for field in fields:
        field["formIndex"] = None
    assert select_signup_fills([LoginControl.from_dict(x) for x in fields], "fixture-only-password") == []


def test_signup_mismatch_and_supervisor_refusal(tmp_path):
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item("login", "Example", {"identifier_type": "email", "identifier": "a@example.test",
                                                "password": "fixture-only-password"}, origin="https://example.test")
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(tool, "_current_page_origin", return_value="https://evil.test"):
        assert json.loads(tool.browser_vault_fill(meta.id, mode="signup"))["error_type"] == "origin_mismatch"
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(tool, "_current_page_origin", return_value="https://example.test"), \
         patch.object(tool, "_eval_js", return_value={"success": True, "result": json.dumps([control(0, "new-password")])}), \
         patch.object(tool, "_ensure_supervisor", return_value=None):
        result = json.loads(tool.browser_vault_fill(meta.id, mode="signup"))
        assert result["error_type"] == "supervisor_required"
        assert "fixture-only-password" not in json.dumps(result)
