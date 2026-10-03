"""Credential destination policies through native plugin discovery and tool dispatch."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from hermes_yaml import safe_dump

from agent.vault_backends.base import LoginBackend
from agent.vault_backends.local import LocalLoginBackend
from agent.vault_store import VaultItemMeta


@pytest.mark.parametrize("origins, primary, destination, expected", [
    ((), "https://example.test", "https://example.test", True),
    ((), "https://example.test", "https://login.example.test", False),
    (("https://other.test",), "https://example.test", "https://other.test", True),
    (("https://other.test",), "https://example.test", "https://example.test", False),
    ((), None, "https://example.test", False),
    (("",), None, "", False),
])
def test_inherited_policy_accepts_only_explicit_nonempty_origins(
    origins: tuple[str, ...], primary: str | None, destination: str, expected: bool,
) -> None:
    meta = VaultItemMeta("vault_test", "login", "Test", primary, "", allowed_origins=origins)
    assert LocalLoginBackend().matches_origin(meta, destination) is expected


_SOURCE = '''
from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultItemMeta

class Backend(LoginBackend):
    name, display_name, prefix = "policytest", "Policy Test", "policytest:"
    calls = []
    reads = []

    @classmethod
    def is_available(cls, config):
        return True

    def __init__(self, config):
        self.config = config

    def list_items(self):
        return [self.get_meta("policytest:item")]

    def get_meta(self, handle):
        return VaultItemMeta(
            handle, self.config["kind"], "Test",
            None if self.config["empty"] else "https://example.test", "",
            allowed_origins=("",) if self.config["empty"] == "blank" else (),
        )

    def matches_origin(self, meta, origin):
        self.calls.append(origin)
        policy = self.config["policy"]
        if policy == "error":
            raise RuntimeError("PRIVATE-POLICY-DETAIL")
        if policy == "integer":
            return 1
        if policy == "string":
            return "true"
        if policy == "none":
            return None
        if policy == "default":
            return super().matches_origin(meta, origin)
        if policy == "deny":
            return False
        return origin in ("https://example.test", "https://login.example.test")

    def resolve_password(self, handle):
        self.reads.append(handle)
        return "synthetic-password"

    def resolve_otp(self, handle):
        self.reads.append(handle)
        return "0" * 6

    def resolve_secret(self, handle):
        self.reads.append(handle)
        return {"card_number": "4111111111111111", "address_line1": "1 Test St"}

def register(ctx):
    ctx.register_login_backend(Backend)
'''


@pytest.mark.parametrize("tool_name", ["browser_vault_fill", "browser_vault_enter_code"])
@pytest.mark.parametrize("policy, kind, empty, destination, focus, authorized", [
    ("default", "login", False, "https://example.test", True, True),
    ("default", "login", False, "https://login.example.test", True, False),
    ("related", "login", False, "https://login.example.test", True, True),
    ("related", "login", False, "https://login.example.test", False, True),
    ("related", "login", False, "https://unrelated.test", True, False),
    ("related", "login", True, "https://login.example.test", True, False),
    pytest.param("related", "login", "blank", "https://login.example.test", True, False, id="blank-saved-origin"),
    ("error", "login", False, "https://example.test", True, False),
    ("integer", "login", False, "https://example.test", True, False),
    ("string", "login", False, "https://example.test", True, False),
    ("none", "login", False, "https://example.test", True, False),
    ("deny", "login", False, "https://example.test", True, False),
    ("related", "payment", False, "https://login.example.test", True, False),
    ("related", "address", False, "https://login.example.test", True, False),
    ("error", "payment", False, "https://example.test", True, True),
    ("error", "address", False, "https://example.test", True, True),
])
def test_native_tab_policy_and_fill_precheck_share_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    policy: str, kind: str, empty: bool | str, destination: str, focus: bool, authorized: bool,
    tool_name: str,
) -> None:
    from agent import redact, vault_login_classifier
    from agent.vault_backends.base import backend_for_handle
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from tools import browser_supervisor, browser_vault_tool
    from tools.registry import registry

    otp = tool_name == "browser_vault_enter_code"
    authorized = authorized and (not otp or kind == "login")
    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    plugin = home / "plugins" / "origin-policy-test"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: origin-policy-test\nversion: 1.0.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_SOURCE, encoding="utf-8")
    (home / "config.yaml").write_text(safe_dump({
        "plugins": {"enabled": ["origin-policy-test"]},
        "vault": {"onepassword": {"enabled": False}, "bitwarden": {"enabled": False},
                  "policytest": {"enabled": True, "policy": policy, "kind": kind, "empty": empty}},
    }), encoding="utf-8")
    discover_plugins()
    backend = backend_for_handle("policytest:item")
    assert isinstance(backend, LoginBackend)

    supervisor = browser_supervisor.CDPSupervisor("policy-test", "ws://synthetic.invalid")
    supervisor._loop = Mock(is_running=lambda: True)
    # Run the real focus coroutine; only the CDP transport is synthetic.
    monkeypatch.setattr(browser_supervisor, "_schedule", lambda coro, loop, **kw: asyncio.run(coro))
    supervisor._enable_page_domains = AsyncMock()
    supervisor._install_dialog_bridge = AsyncMock()
    attached: list[str] = []

    async def cdp(method: str, params: dict | None = None, **kwargs: object) -> dict:
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [
                {"targetId": "decoy", "type": "page", "url": "https://decoy.test/login"},
                {"targetId": "destination", "type": "page", "url": destination + ":443/login?next=home"},
            ] if focus else []}}
        if method == "Target.attachToTarget":
            assert params is not None
            attached.append(params["targetId"])
            return {"result": {"sessionId": params["targetId"]}}
        if method == "Runtime.evaluate":
            return {"result": {"result": {"value": True}}}
        raise AssertionError(method)

    monkeypatch.setattr(supervisor, "_cdp", cdp)
    monkeypatch.setattr(browser_vault_tool, "_ensure_supervisor", lambda task_id: supervisor)
    monkeypatch.setattr(browser_vault_tool, "_current_page_origin", lambda task_id: destination)
    monkeypatch.setattr(browser_vault_tool, "_confirm_payment_fill", lambda *args: True)
    inspect = Mock(wraps=vault_login_classifier.build_inspection_js)
    fill = Mock(wraps=vault_login_classifier.build_fill_js)
    monkeypatch.setattr(vault_login_classifier, "build_inspection_js", inspect)
    monkeypatch.setattr(vault_login_classifier, "build_fill_js", fill)
    controls = [{"index": 0, "type": "password", "autocomplete": "current-password"},
                {"index": 1, "type": "text", "autocomplete": "cc-number"},
                {"index": 2, "type": "text", "autocomplete": "address-line1"},
                {"index": 3, "type": "text", "autocomplete": "one-time-code"}]
    monkeypatch.setattr(browser_vault_tool, "_eval_js", lambda *args: {"success": True, "result": controls})
    secret_eval = Mock(return_value={"success": True, "result": {"filled": 1}})
    monkeypatch.setattr(browser_vault_tool, "_eval_js_secret", secret_eval)
    try:
        raw = registry.dispatch(tool_name, {"handle": "policytest:item"}, task_id="policy-test")
        assert isinstance(raw, str)
        out = json.loads(raw)
        assert out["success"] is authorized
        assert attached == (["destination"] if authorized and focus else [])
        assert getattr(backend, "reads") == (["policytest:item"] if authorized else [])
        if authorized:
            assert out["origin"] == destination
            assert fill.call_args.kwargs == {"expected_origin": destination, "nonce": inspect.call_args.args[0]}
            assert inspect.call_args.args[0]
            secret_eval.assert_called_once()
        else:
            assert out["error_type"] == ("invalid_login" if otp and kind != "login" else "origin_mismatch")
            inspect.assert_not_called()
            secret_eval.assert_not_called()
        if kind != "login" or empty:
            assert getattr(backend, "calls") == []
        else:
            assert destination in getattr(backend, "calls")
        assert "0" * 6 not in raw
        assert "synthetic-password" not in raw
        assert "PRIVATE-POLICY-DETAIL" not in raw + caplog.text
        assert not any(record.exc_info for record in caplog.records)
    finally:
        get_plugin_manager().unload("origin-policy-test")
        redact.clear_vault_redaction_values()
