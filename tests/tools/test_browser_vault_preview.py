"""browser_vault on the Desktop in-app browser (``target='preview'``).

The preview callback is the external Desktop boundary (a ``preview.act`` server request); the registry,
vault encryption, backend selection, classifier, origin guard and secret-egress redaction are real.
"""

import json
from types import SimpleNamespace

import pytest

from agent.redact import clear_vault_redaction_values
from agent.vault_store import get_vault_store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_agent.tools import browser_vault_tool
from hermes_agent.tools.registry import registry

ORIGIN = "https://login.example.test"
CONTROLS = [{"index": 0, "formIndex": 0, "type": "password", "autocomplete": "current-password", "name": "password"}]


class PreviewWindow:
    """The window showing the chat: pins its active tab on ``open`` and answers that target only."""

    def __init__(self, origin=ORIGIN):
        self.origin, self.calls = origin, []

    def __call__(self, payload):
        assert payload["action"] == "vault"
        request = payload["vault"]
        self.calls.append(request)
        if request["operation"] == "open":
            return json.dumps({"success": True, "target": "pinned-tab"})
        if request.get("target") != "pinned-tab":
            return json.dumps({"success": False, "error": "unknown target"})
        if request["operation"] == "close":
            return json.dumps({"success": True})
        expression = request["expression"]
        if "location.href" in expression:
            result = self.origin + "/login"
        elif "data-hermes-vault-slot\", nonce" in expression:
            result = json.dumps(CONTROLS)
        else:
            result = json.dumps({"filled": 1})
        return json.dumps({"success": True, "result": result})

    def ops(self):
        return [c["operation"] for c in self.calls]


def _managed_browser_used(*_args, **_kwargs):
    pytest.fail("a preview vault operation reached the managed browser")


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(browser_vault_tool, "_fenced_page_op", _managed_browser_used)
    monkeypatch.setattr(browser_vault_tool, "_ensure_supervisor", _managed_browser_used)
    token = set_hermes_home_override(tmp_path)
    (tmp_path / "config.yaml").write_text(json.dumps({"vault": {"onepassword": {"enabled": False},
                                                                  "bitwarden": {"enabled": False}}}))
    try:
        yield tmp_path
    finally:
        clear_vault_redaction_values()
        reset_hermes_home_override(token)


def _login(password, origin=ORIGIN):
    return get_vault_store().add_item("login", "Synthetic", {"identifier": "test-user", "identifier_type": "username",
                                                             "password": password}, origin=origin)


def test_preview_fill_writes_through_the_pinned_tab_and_redacts_echoes(home):
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    from hermes_agent.tools.drive_preview_tool import drive_preview_tool
    from hermes_agent.tools.read_preview_tool import read_preview_tool

    password = 'synthetic-quote"-slash\\-secret'
    meta = _login(password)
    window = PreviewWindow()
    raw = INLINE_TOOL_EXECUTORS["browser_vault"](SimpleNamespace(drive_preview_callback=window), {
        "action": "fill", "handle": meta.id, "target": "preview"}, InlineToolContext(effective_task_id="t"))
    result = json.loads(raw)
    assert result["success"] is True and result["filled_fields"] == 1 and result["origin"] == ORIGIN, result
    assert window.ops() == ["open", "evaluate", "evaluate", "evaluate", "close"]
    assert json.dumps(password) in window.calls[-2]["expression"]
    assert password not in raw
    # A page can echo its input into its text, an element label or an error.
    echoed = json.dumps({"success": True, "text": password, "elements": [{"label": password}], password: 1})
    for read in (read_preview_tool(callback=lambda **_kw: echoed), drive_preview_tool(action="elements",
                                                                                       callback=lambda _p: echoed)):
        assert password not in read and "redacted-vault-secret" in read


@pytest.mark.parametrize("failure", ["wrong_origin", "no_window", "bad_target"])
def test_preview_refusals_never_resolve_the_secret_or_fall_back(home, monkeypatch, failure):
    from agent.vault_backends.local import LocalLoginBackend

    meta = _login("synthetic-refusal-secret")
    monkeypatch.setattr(LocalLoginBackend, "resolve_password", _managed_browser_used)
    window = PreviewWindow(origin="https://wrong.example.test")
    result = json.loads(registry.dispatch("browser_vault", {
        "action": "fill", "handle": meta.id, "target": "elsewhere" if failure == "bad_target" else "preview",
    }, preview_callback=None if failure == "no_window" else window))
    assert result["success"] is False
    assert result["error_type"] == {"wrong_origin": "origin_mismatch", "no_window": "preview_unavailable",
                                    "bad_target": "invalid_target"}[failure]
    assert window.ops() == (["open", "evaluate", "close"] if failure == "wrong_origin" else [])


def test_saved_code_is_origin_bound_on_the_preview(home, monkeypatch):
    from agent.vault_backends.local import LocalLoginBackend

    meta = _login("synthetic-otp-login")
    monkeypatch.setattr(LocalLoginBackend, "resolve_otp", _managed_browser_used)
    window = PreviewWindow(origin="https://wrong.example.test")
    result = json.loads(registry.dispatch("browser_vault", {"action": "enter_code", "handle": meta.id,
                                                             "target": "preview"}, preview_callback=window))
    assert result["error_type"] == "origin_mismatch"
    assert window.ops() == ["open", "evaluate", "close"]


@pytest.mark.parametrize("surface", ["preview", "preview_deferred", "managed_page", "plugin_page", "no_preview", "explicit"])
def test_omitted_target_picks_the_preview_only_without_a_managed_page(home, monkeypatch, surface):
    """Desktop sessions with drive_preview (eager or behind tool_call) and no managed browser page fill
    the in-app browser; a session that opened a page with browser_exec, OR whose live page is owned by a
    plugin backend (e.g. Camofox), keeps filling that page."""
    from agent import tool_executor
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    from hermes_agent.tools import browser_session

    managed = []
    monkeypatch.setattr(browser_vault_tool, "_fenced_page_op", lambda tid, _op: managed.append(tid) or "{}")
    monkeypatch.setattr(tool_executor, "_tool_search_scoped_names", lambda _agent: frozenset({"drive_preview"}))
    monkeypatch.setattr(browser_session, "session_for_task",
                        lambda tid: {"cdp_url": "ws://x"} if surface == "managed_page" else {})
    # A plugin backend holding the task's live page must win over an omitted target, exactly like a
    # managed CDP page: browser_vault's own backend selection already picks it (#150283 P1-4).
    import agent.browser_page_registry as page_registry
    monkeypatch.setattr(page_registry, "backend_for_task",
                        lambda tid: object() if surface == "plugin_page" else None)
    names = {"preview": {"drive_preview"}, "preview_deferred": {"tool_call"}, "managed_page": {"drive_preview"},
             "plugin_page": {"drive_preview"},
             "no_preview": set(), "explicit": {"drive_preview"}}[surface]
    window = PreviewWindow()
    agent = SimpleNamespace(drive_preview_callback=window, valid_tool_names=names)
    args = {"action": "fill", "handle": _login("synthetic-default").id}
    if surface == "explicit":
        args["target"] = "browser"
    INLINE_TOOL_EXECUTORS["browser_vault"](agent, args, InlineToolContext(effective_task_id="t"))
    to_preview = surface in ("preview", "preview_deferred")
    assert (window.ops()[:1] == ["open"]) is to_preview
    assert managed == ([] if to_preview else ["t"])
