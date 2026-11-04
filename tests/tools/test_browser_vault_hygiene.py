"""Vault privacy around the current page: list scoped to the page's site, card numbers masked in any
grouping, the CVC masked only while its page lives, and captures refused while a card is filled."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore

_CARD = {"card_number": "4242424242424242", "cardholder_name": "A User", "exp_month": "7", "exp_year": "2029",
         "cvc": "123", "billing_postal_code": "94110"}
_CONTROLS = [{"autocomplete": "cc-number", "index": 0, "type": "text"},
             {"label": "CVC", "index": 1, "type": "text"}]


@pytest.fixture()
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


def _login(store, origin):
    return store.add_item(kind="login", label=origin, origin=origin,
                          secret={"identifier_type": "email", "identifier": f"me@{origin[8:]}", "password": "pw-1"})


class _Page:
    """The browser as the vault sees it: open tabs (target id → document, which changes when that tab
    navigates) and the tab the supervisor's page session points at. It serves both the supervisor's
    ``evaluate_runtime`` and the captured CDP channel (``SUPERVISOR_REGISTRY.capture``)."""

    page_session_id = "page"

    def __init__(self):
        self.tabs = {"checkout": "1000.5"}
        self.current = "checkout"

    def evaluate_runtime(self, expression, **_):
        return {"ok": True, "result": self.tabs[self.current]}

    def call(self, method, params=None, *, session_id=None, timeout=None):
        if method == "Target.getTargetInfo":
            return {"result": {"targetInfo": {"targetId": self.current}}}
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"targetId": t, "type": "page"} for t in self.tabs]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "tab:" + params["targetId"]}}
        if method == "Runtime.evaluate":
            return {"result": {"result": {"type": "string", "value": self.tabs[session_id.removeprefix("tab:")]}}}
        return {"result": {}}


@pytest.fixture()
def filled_card(store):
    """A card filled into https://shop.test for task ``t``; yields the fake page."""
    from agent import redact, redact_cards
    from tools import browser_vault_card_guard, browser_vault_tool

    page = _Page()
    meta = store.add_item(kind="payment", label="Card", origin="https://shop.test", secret=_CARD)

    def fake_eval(task_id, expression):
        if "location.href" in expression:
            return {"success": True, "result": "https://shop.test/checkout"}
        return {"success": True, "result": json.dumps(_CONTROLS)}

    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch("tools.browser_supervisor.SUPERVISOR_REGISTRY.get", return_value=page), \
         patch("tools.browser_supervisor.SUPERVISOR_REGISTRY.capture", return_value=page), \
         patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
         patch.object(browser_vault_tool, "_focus_bound_origin", return_value=None), \
         patch.object(browser_vault_tool, "_eval_js_secret",
                      return_value={"success": True, "result": json.dumps({"filled": 2})}), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value="accept"):
        assert json.loads(browser_vault_tool.browser_vault_fill(meta.id, task_id="t"))["success"] is True
        try:
            yield page
        finally:
            browser_vault_card_guard.clear("t")
            redact.clear_vault_redaction_values()
            redact_cards.clear_card_redactions()


def test_list_names_only_the_current_sites_items(store):
    from tools import browser_vault_tool

    _login(store, "https://github.com")
    _login(store, "https://shop.example.org")
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(browser_vault_tool, "_read_page_origin", return_value="https://github.com"):
        scoped = json.loads(browser_vault_tool._handle_vault({"action": "list"}, task_id="t"))
        everything = json.loads(browser_vault_tool._handle_vault({"action": "list", "all_sites": True}, task_id="t"))
    assert [i["origin"] for i in scoped["items"]] == ["https://github.com"]
    assert scoped["site"] == "github.com" and scoped["other_sites_items"] == 1
    assert "shop.example.org" not in json.dumps(scoped)
    assert {i["origin"] for i in everything["items"]} == {"https://github.com", "https://shop.example.org"}


@pytest.mark.parametrize("shown", ["4242 4242 4242 4242", "4242-4242-4242-4242", "4242 42424242 4242",
                                   "4242.4242.4242.4242"])
def test_card_number_masked_in_any_grouping_the_page_prints(filled_card, shown):
    from agent.redact import redact_sensitive_text

    out = redact_sensitive_text(f"value={shown};")
    assert "4242 4242" not in out and "42424242" not in out and out.count("4242") == 1, out
    assert "••••4242" in out


def test_cvc_masked_whole_number_only_until_the_page_navigates(filled_card):
    from agent.redact import redact_sensitive_text
    from tools.browser_vault_card_guard import capture_refusal

    assert redact_sensitive_text("cvc 123.") == "cvc «redacted-cvc»."
    assert redact_sensitive_text("order 41234, 1230 items, 2029-07") == "order 41234, 1230 items, 2029-07"
    assert capture_refusal("t")  # same document: still filled
    filled_card.tabs["checkout"] = "2000.25"  # submit loaded the confirmation page
    assert capture_refusal("t") is None
    assert redact_sensitive_text("cvc 123.") == "cvc 123."


def test_screenshot_and_vision_refused_while_card_is_filled(filled_card, tmp_path):
    from tools.browser_cdp_tool import browser_cdp
    from tools.vision_tools import _handle_vision_analyze
    import asyncio

    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    refused = json.loads(browser_cdp("Page.captureScreenshot", task_id="t"))
    assert refused.get("error_type") == "card_filled", refused
    seen = json.loads(asyncio.run(_handle_vision_analyze({"image_url": str(shot), "question": "?"}, task_id="t")))
    assert seen["error_type"] == "card_filled"
    assert json.loads(browser_cdp("Page.captureScreenshot", task_id="other")).get("error_type") != "card_filled"
    filled_card.tabs["checkout"] = "3000.75"
    assert json.loads(browser_cdp("Page.captureScreenshot", task_id="t")).get("error_type") != "card_filled"


@pytest.mark.parametrize("spelling", ["FILE://{}", "File://{}", "  {}  ", "\tfile://{}\n"])
def test_vision_refusal_judges_the_file_the_reader_opens(filled_card, tmp_path, spelling):
    """Any spelling the image reader accepts for a fresh capture (scheme case, surrounding whitespace)
    must hit the same refusal as the plain path."""
    from tools.browser_vault_card_guard import image_refusal

    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert image_refusal("t", spelling.format(shot)), spelling


def test_card_protection_follows_the_filled_tab_not_the_supervisors_focus(filled_card):
    """A later fill or focus moves the supervisor to another tab; the checkout tab still holds the card
    and can be captured by its target id, so its protection lasts until that tab navigates or closes."""
    from tools.browser_vault_card_guard import capture_refusal

    filled_card.tabs["login"] = "5000.0"
    filled_card.current = "login"
    assert capture_refusal("t"), "focusing another tab must not end the checkout tab's protection"
    filled_card.tabs["checkout"] = "6000.0"  # the checkout tab itself navigated
    assert capture_refusal("t") is None


def test_cvc_mask_ends_when_a_plain_cdp_read_follows_navigation(filled_card):
    """Page.navigate then Runtime.evaluate: no capture call in between, yet the CVC mask must end with
    the checkout page instead of masking every standalone 123 afterwards."""
    from agent.redact import redact_sensitive_text
    from tools.browser_cdp_tool import browser_cdp

    assert redact_sensitive_text("cvc 123.") == "cvc «redacted-cvc»."
    filled_card.tabs["checkout"] = "7000.0"
    browser_cdp("Runtime.evaluate", {"expression": "document.title"}, task_id="t")
    assert redact_sensitive_text("cvc 123.") == "cvc 123."


def test_list_withholds_other_sites_when_the_open_page_cannot_be_read(store):
    """A failed page read is not "no page open": listing every site's logins then would undo the scoping."""
    from tools import browser_vault_tool

    _login(store, "https://github.com")
    _login(store, "https://shop.example.org")
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch("tools.browser_supervisor.SUPERVISOR_REGISTRY.get", return_value=_Page()), \
         patch.object(browser_vault_tool, "_eval_js", return_value={"success": False, "error": "timeout"}), \
         patch.object(browser_vault_tool, "_focus_bound_origin", return_value=None):
        out = json.loads(browser_vault_tool._handle_vault({"action": "list"}, task_id="t"))
    assert out["items"] == [] and out["total"] == 2 and "query" in out["hint"]
    assert "github.com" not in json.dumps(out) and "example.org" not in json.dumps(out)


# Stands in for the harness's run.py: helpers imported into the exec globals, capture_screenshot calling
# cdp through its own module, the model's code exec'd after Hermes's preambles.
_FAKE_HARNESS = """\
import sys, types
helpers = types.ModuleType("browser_harness.helpers")
package = types.ModuleType("browser_harness")
package.helpers = helpers
sys.modules.update({"browser_harness": package, "browser_harness.helpers": helpers})
def cdp(method, **params):
    return {"data": ""}
def capture_screenshot(path):
    helpers.cdp("Page.captureScreenshot", format="png")
    open(path, "wb").write(b"card pixels")
    return path
helpers.cdp, helpers.capture_screenshot = cdp, capture_screenshot
exec(sys.stdin.read(), {"cdp": cdp, "capture_screenshot": capture_screenshot})
"""


def test_browser_exec_refuses_captures_of_a_filled_card_and_keeps_other_images(tmp_path, monkeypatch):
    """While a card is filled the call's screenshot is refused before an image exists, so images the call
    or an earlier one wrote (product downloads, saved files) stay in the workspace."""
    from tools import browser_session, browser_supervisor, browser_use_cli

    class _Registry:
        def get_or_start(self, task_id, cdp_url, **kw):
            return None

        def get(self, task_id):
            return None

    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", _Registry())
    monkeypatch.setattr(browser_session, "get_session_info",
                        lambda key: {"cdp_url": "ws://127.0.0.1:47000/devtools/browser/x", "features": {"local": True}})
    monkeypatch.setattr("hermes_cli.config_effective.load_user_config_effective", dict)
    monkeypatch.setattr("tools.browser_vault_card_guard.capture_refusal", lambda task_id: "card filled")
    harness = tmp_path / "harness.py"
    harness.write_text(_FAKE_HARNESS, encoding="utf-8")
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: [sys.executable, str(harness)])
    workspace = browser_use_cli._workspace_dir("t")
    earlier = os.path.join(workspace, "earlier.png")
    with open(earlier, "wb") as f:
        f.write(b"product")

    code = ("import os\nws = os.environ['BH_AGENT_WORKSPACE']\n"
            "open(os.path.join(ws, 'product.png'), 'wb').write(b'product')\n"
            "print(os.path.join(ws, 'product.png'))\n"
            "print(capture_screenshot(os.path.join(ws, 'shot.png')))\n")
    result = json.loads(browser_use_cli.browser_exec(code, task_id="t"))

    assert os.path.exists(earlier) and os.path.exists(os.path.join(workspace, "product.png"))
    assert "card filled" in result.get("stderr", ""), result
    assert not os.path.exists(os.path.join(workspace, "shot.png"))
    assert "screenshot_path" not in result
