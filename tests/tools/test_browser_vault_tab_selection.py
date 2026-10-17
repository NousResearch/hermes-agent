"""Regression coverage for origin-bound OTP tab selection (no real secrets)."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import browser_vault_tool as tool
import pytest


@pytest.mark.parametrize("meta, origin, expected", [
    (None, "https://www.marriott.com", "invalid_handle"),
    (SimpleNamespace(kind="payment"), "https://www.marriott.com", "invalid_handle"),
    (SimpleNamespace(kind="login", origin=None, allowed_origins=()), "https://www.marriott.com", "no_origin"),
    (SimpleNamespace(kind="login", origin="https://www.marriott.com", allowed_origins=()), "https://www.ihg.com", "origin_mismatch"),
])
def test_invalid_or_mismatched_handle_never_resolves_or_prompts(meta, origin, expected):
    backend = Mock()
    backend.get_meta.return_value = meta
    with patch("agent.vault_backends.backend_for_handle", return_value=backend), \
         patch.object(tool, "_focus_bound_origin", return_value=None), \
         patch.object(tool, "_current_page_origin", return_value=origin), \
         patch.object(tool, "_eval_js") as inspect, \
         patch("agent.vault_backends.unlock.get_code_prompt_callback") as prompt:
        result = json.loads(tool.browser_vault_enter_code("test_handle"))
    assert result["error_type"] == expected
    backend.resolve_otp.assert_not_called()
    inspect.assert_not_called()
    prompt.assert_not_called()


def test_enter_code_selects_saved_login_origins_before_inspection():
    backend = Mock()
    backend.get_meta.return_value = SimpleNamespace(
        kind="login", origin="https://www.marriott.com",
        allowed_origins=("https://marriott.com", "https://www.marriott.com"))
    with patch("agent.vault_backends.backend_for_handle", return_value=backend), \
         patch.object(tool, "_focus_bound_origin", side_effect=[None, "https://www.marriott.com"]) as focus, \
         patch.object(tool, "_current_page_origin", return_value="https://www.marriott.com"), \
         patch.object(tool, "_eval_js", return_value={"success": True, "result": "[]"}):
        result = json.loads(tool.browser_vault_enter_code("test_handle"))
    assert [c.args[1] for c in focus.call_args_list] == ["https://marriott.com", "https://www.marriott.com"]
    assert result["error_type"] == "no_code_field"
    backend.resolve_otp.assert_not_called()


def test_otp_probe_ignores_hidden_and_non_otp_controls():
    import shutil
    import subprocess
    import pytest
    node = shutil.which("node")
    if not node:
        pytest.skip("node required to exercise the real tab predicate")
    # In-memory DOM descriptors only: no HTML page, input values or credential collection.
    setup = r'''
const controls = JSON.parse(process.argv[1]);
const elements = controls.map(c => ({
  type: c.type || "text", name: c.name || "", id: "", autocomplete: c.autocomplete || "",
  disabled: !!c.disabled, readOnly: !!c.readOnly, labels: [{textContent: c.label || ""}],
  tagName: "INPUT", form: null, maxLength: 6,
  getAttribute: () => "", getClientRects: () => c.hidden ? [] : [{}],
  setAttribute: () => { throw Error("probe must not mutate DOM"); }
}));
global.document = {querySelectorAll: () => elements, forms: [],
  querySelector: () => elements[0] || null, getElementById: () => null};
global.getComputedStyle = () => ({display: "block", visibility: "visible"});
'''
    cases = [
        ([{"name": "countryCode", "type": "hidden"}], False),
        ([{"name": "otp", "hidden": True}], False),
        ([{"name": "otp", "disabled": True}], False),
        ([{"name": "otp", "readOnly": True}], False),
        ([{"name": "postalCode"}], False),
        ([{"name": "promo-code"}], False),
        ([{"name": "input-text-Verification Code", "type": "number"}], True),
        ([{"label": "Authentication code", "type": "password"}], True),
        ([{"autocomplete": "section-login one-time-code", "type": "tel"}], True),
    ]
    for controls, expected in cases:
        script = setup + "console.log(JSON.stringify(" + tool._TAB_PROBES["otp"] + "));"
        result = subprocess.run([node, "-e", script, json.dumps(controls)], capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) is expected, controls
