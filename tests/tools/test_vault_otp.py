"""OTP errors stay visible and opaque; explicit manual entry preserves authorization."""
from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from agent import redact, vault_backends
from agent.vault_backends.base import LoginBackend
from agent.vault_backends import unlock
from agent.vault_store import VaultItemMeta
from tools import browser_vault_tool
from tools.registry import registry


@pytest.fixture
def otp_path(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, Mock]:
    backend = Mock(spec=LoginBackend)
    backend.name = "fixture"
    backend.needs_unlock = False
    backend.get_meta.return_value = VaultItemMeta("fixture:item", "login", "Fixture", "https://example.test", "")
    backend.matches_origin.return_value = True
    backend.resolve_otp.return_value = "0" * 6
    prompt = Mock(return_value="0" * 6)
    transport = Mock(return_value={"success": True, "result": {"filled": 1}})
    monkeypatch.setattr(vault_backends, "backend_for_handle", lambda handle: backend)
    monkeypatch.setattr(browser_vault_tool, "_focus_bound_origin", lambda *args, **kwargs: None)
    monkeypatch.setattr(browser_vault_tool, "_current_page_origin", lambda task: "https://example.test")
    monkeypatch.setattr(browser_vault_tool, "_eval_js", lambda *args: {"success": True, "result": [
        {"index": 0, "type": "text", "autocomplete": "one-time-code"},
    ]})
    monkeypatch.setattr(browser_vault_tool, "_eval_js_secret", transport)
    monkeypatch.setattr(unlock, "can_prompt_here", lambda: True)
    monkeypatch.setattr(unlock, "get_code_prompt_callback", lambda: prompt)
    monkeypatch.setattr(redact, "register_vault_redaction_value", Mock())
    return backend, prompt, transport


@pytest.mark.parametrize("failure, expected", [
    ("unknown", "invalid_login"), ("missing", "invalid_login"),
    ("metadata", "metadata_unavailable"), ("resolve", "code_unavailable"),
    ("transport", "code_fill_failed"), ("transport_result", "code_fill_failed"),
])
def test_otp_failures_neither_leak_nor_silently_prompt(
    otp_path: tuple[Mock, Mock, Mock], monkeypatch: pytest.MonkeyPatch,
    failure: str, expected: str,
) -> None:
    backend, prompt, transport = otp_path
    private = "PRIVATE-OTP-DETAIL " + "0" * 6
    match failure:
        case "unknown":
            monkeypatch.setattr(vault_backends, "backend_for_handle", lambda handle: None)
        case "missing":
            backend.get_meta.return_value = None
        case "metadata":
            backend.get_meta.side_effect = RuntimeError(private)
        case "resolve":
            backend.resolve_otp.side_effect = RuntimeError(private)
        case "transport":
            transport.side_effect = RuntimeError(private)
        case "transport_result":
            transport.return_value = {"success": False, "error": private}
    raw = registry.dispatch("browser_vault_enter_code", {"handle": "fixture:item"}, task_id="otp-test")
    assert isinstance(raw, str)
    assert json.loads(raw)["error_type"] == expected
    assert "PRIVATE-OTP-DETAIL" not in raw and "0" * 6 not in raw
    prompt.assert_not_called()
    if failure in ("unknown", "missing", "metadata"):
        backend.resolve_otp.assert_not_called()
    if failure not in ("transport", "transport_result"):
        transport.assert_not_called()


@pytest.mark.parametrize("handle", ["", "fixture:item"])
@pytest.mark.parametrize("manual", [False, True])
def test_manual_code_fallback_names_the_authorized_site(
    otp_path: tuple[Mock, Mock, Mock], handle: str, manual: bool,
) -> None:
    backend, prompt, transport = otp_path
    backend.resolve_otp.return_value = None
    raw = registry.dispatch("browser_vault_enter_code", {"handle": handle, "manual": manual}, task_id="otp-test")
    assert isinstance(raw, str)
    assert json.loads(raw)["success"] and json.loads(raw)["source"] == "user"
    prompt.assert_called_once_with("example.test", "")
    transport.assert_called_once()
    assert "0" * 6 not in raw
    if handle:
        backend.get_meta.assert_called_once_with(handle)
        backend.matches_origin.assert_called_once()
    else:
        backend.get_meta.assert_not_called()
    if handle and not manual:
        backend.resolve_otp.assert_called_once_with(handle)
    else:
        backend.resolve_otp.assert_not_called()


@pytest.mark.parametrize("denial", ["", "origin", "metadata", "missing"])
def test_explicit_manual_retry_preserves_the_saved_login_checks(
    otp_path: tuple[Mock, Mock, Mock], denial: str,
) -> None:
    backend, prompt, transport = otp_path
    backend.resolve_otp.side_effect = RuntimeError("PRIVATE-OTP-DETAIL")
    args = {"handle": "fixture:item"}
    first = registry.dispatch("browser_vault_enter_code", args, task_id="otp-test")
    assert isinstance(first, str)
    assert json.loads(first)["error_type"] == "code_unavailable"
    assert "manual=true" in json.loads(first)["next"]
    prompt.assert_not_called()
    match denial:
        case "origin":
            backend.matches_origin.return_value = False
        case "metadata":
            backend.get_meta.side_effect = RuntimeError("PRIVATE-OTP-DETAIL")
        case "missing":
            backend.get_meta.return_value = None
    raw = registry.dispatch("browser_vault_enter_code", {**args, "manual": True}, task_id="otp-test")
    assert isinstance(raw, str)
    assert "PRIVATE-OTP-DETAIL" not in raw and "0" * 6 not in raw
    assert json.loads(raw)["success"] is (not denial)
    assert backend.get_meta.call_count == 2
    backend.resolve_otp.assert_called_once_with(args["handle"])
    if denial:
        prompt.assert_not_called()
        transport.assert_not_called()
    else:
        prompt.assert_called_once_with("example.test", "")
        transport.assert_called_once()
        assert json.loads(raw)["source"] == "user"
