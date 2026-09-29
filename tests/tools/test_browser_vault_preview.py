"""The preview target shares the real scoped vault, never the managed browser.

The callback is the external Desktop boundary; the registry, vault encryption,
backend selection, classifier and secret-egress redaction are real imports.
"""

import json
from types import SimpleNamespace

import pytest

from agent.redact import clear_vault_redaction_values
from agent.vault_store import get_vault_store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import browser_vault_tool
from tools.registry import registry


class PreviewClient:
    def __init__(self, origin="https://login.example.test"):
        self.origin = origin
        self.calls = []
        self.evaluations = 0

    def __call__(self, payload):
        assert payload["action"] == "vault"
        request = payload["vault"]
        self.calls.append(request)
        if request["operation"] == "open":
            return json.dumps({"success": True, "target": "window-owned-binding"})
        assert request["target"] == "window-owned-binding"
        if request["operation"] == "close":
            return json.dumps({"success": True})
        self.evaluations += 1
        results = [
            self.origin + "/login",
            [{"index": 0, "formIndex": 0, "type": "password",
              "autocomplete": "current-password", "name": "password"}],
            {"filled": 1},
        ]
        return json.dumps({"success": True, "result": results[self.evaluations - 1]})


def _no_managed_browser(*_args, **_kwargs):
    pytest.fail("A preview credential operation reached the unrelated managed browser")


def _login(home, password):
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        "vault:\n  onepassword:\n    enabled: false\n  bitwarden:\n    enabled: false\n",
        encoding="utf-8",
    )
    return get_vault_store().add_item(
        "login", "Synthetic preview login",
        {"identifier": "test-user", "identifier_type": "username", "password": password},
        origin="https://login.example.test",
    )


def test_preview_dispatch_keeps_home_and_page_binding_through_a_b_a(tmp_path, monkeypatch):
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    from tools.drive_preview_tool import drive_preview_tool
    from tools.read_preview_tool import read_preview_tool

    monkeypatch.setattr(browser_vault_tool, "_fenced_page_op", _no_managed_browser)
    homes = {name: tmp_path / name for name in ("a", "b")}
    handles = {}
    for name in ("a", "b", "a"):
        token = set_hermes_home_override(homes[name])
        password = f'synthetic-{name}-quote"-slash\\-secret'
        try:
            if name not in handles:
                handles[name] = _login(homes[name], password).id
            client = PreviewClient()
            raw = INLINE_TOOL_EXECUTORS["browser_vault_fill"](SimpleNamespace(drive_preview_callback=client), {
                "handle": handles[name], "target": "preview",
            }, InlineToolContext(effective_task_id="same-task-name"))
            result = json.loads(raw)
            assert result["success"] is True, result
            assert result["filled_fields"] == 1
            assert result["origin"] == client.origin
            assert [call["operation"] for call in client.calls] == [
                "open", "evaluate", "evaluate", "evaluate", "close",
            ]
            assert json.dumps(password) in client.calls[-2]["expression"]
            assert password not in raw

            # A page can echo its input into text, values, labels or errors.
            echoed = json.dumps({"success": True, "text": password, "elements": [{"label": password}]})
            reads = [
                read_preview_tool(callback=lambda **_kw: echoed),
                drive_preview_tool(action="elements", callback=lambda _p: echoed),
            ]
            for read in reads:
                assert password not in json.dumps(json.loads(read), ensure_ascii=False)
                assert "redacted-vault-secret" in read
        finally:
            clear_vault_redaction_values()
            reset_hermes_home_override(token)


@pytest.mark.parametrize("failure", ["wrong_origin", "no_window", "unsupported_target"])
def test_preview_refusals_never_resolve_credentials_or_fall_back(tmp_path, monkeypatch, failure):
    from agent.vault_backends.local import LocalLoginBackend

    monkeypatch.setattr(browser_vault_tool, "_fenced_page_op", _no_managed_browser)
    monkeypatch.setattr(LocalLoginBackend, "resolve_password", _no_managed_browser)
    token = set_hermes_home_override(tmp_path)
    try:
        meta = _login(tmp_path, "synthetic-refusal-secret")
        client = PreviewClient(origin="https://wrong.example.test")
        result = json.loads(registry.dispatch("browser_vault_fill", {
            "handle": meta.id, "target": "invalid" if failure == "unsupported_target" else "preview",
        }, preview_callback=None if failure == "no_window" else client))
        assert result["success"] is False
        assert result["error_type"] == {
            "wrong_origin": "origin_mismatch", "no_window": "preview_unavailable",
            "unsupported_target": "invalid_target",
        }[failure]
        if failure == "wrong_origin":
            assert [call["operation"] for call in client.calls] == ["open", "evaluate", "close"]
        else:
            assert client.calls == []
    finally:
        reset_hermes_home_override(token)


def test_saved_otp_is_origin_bound_before_inspection_or_secret_resolution(tmp_path, monkeypatch):
    from agent.vault_backends.local import LocalLoginBackend

    monkeypatch.setattr(browser_vault_tool, "_fenced_page_op", _no_managed_browser)
    monkeypatch.setattr(LocalLoginBackend, "resolve_otp", _no_managed_browser)
    token = set_hermes_home_override(tmp_path)
    try:
        meta = _login(tmp_path, "synthetic-otp-login")
        client = PreviewClient(origin="https://wrong.example.test")
        result = json.loads(registry.dispatch("browser_vault_enter_code", {
            "handle": meta.id, "target": "preview",
        }, preview_callback=client))
        assert result["success"] is False
        assert result["error_type"] == "origin_mismatch"
        assert [call["operation"] for call in client.calls] == ["open", "evaluate", "close"]
    finally:
        reset_hermes_home_override(token)


def test_preview_vault_schemas_follow_each_session_not_the_shared_gateway(monkeypatch):
    import model_tools
    from tools import browser_tool_install, browser_use_cli, desktop_ui

    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setattr(desktop_ui, "_emit", lambda *_args: None)
    monkeypatch.setattr(browser_tool_install, "check_browser_requirements", lambda: False)
    monkeypatch.setattr(browser_use_cli, "is_browser_use_cli_mode", lambda: False)
    vault_names = {"browser_vault_list", "browser_vault_unlock", "browser_vault_fill",
                   "browser_vault_save_login", "browser_vault_enter_code"}
    # One process, alternating Desktop/remote/Desktop selections; no managed
    # runtime and no Desktop env flag. Use the real resolver and registrations.
    for surface in ("desktop_ui", None, "desktop_ui"):
        selected = ["browser"] + ([surface] if surface else [])
        definitions = model_tools._compute_tool_definitions(selected, quiet_mode=True, skip_tool_search_assembly=True)
        by_name = {d["function"]["name"]: d["function"] for d in definitions}
        assert vault_names.intersection(by_name) == (vault_names if surface else set())
    assert "drive_preview" in by_name["browser_vault_fill"]["description"]
    # The same headless session retains the normal vault tools when its managed
    # browser prerequisite is satisfied.
    monkeypatch.setattr(browser_use_cli, "is_browser_use_cli_mode", lambda: True)
    definitions = model_tools._compute_tool_definitions(["browser"], quiet_mode=True, skip_tool_search_assembly=True)
    assert vault_names <= {d["function"]["name"] for d in definitions}
