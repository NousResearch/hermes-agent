"""Browser output forces redaction only with explicit profile-local vault opt-in."""

import json
from pathlib import Path
import sys

import pytest
import yaml


@pytest.fixture
def browser_output(monkeypatch):
    """Real dispatch/config/redactors and CLI subprocess; fake only browser I/O."""
    from tools import browser_use_cli, browser_tool, browser_tool_session
    from tools import browser_cdp_tool, browser_supervisor
    from tools.registry import registry

    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: [sys.executable, "-c", "import sys; exec(sys.stdin.read())"])
    monkeypatch.setattr(browser_use_cli, "_route_backend", lambda *args: None)
    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, "get", lambda *args: None)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_blocked_private_page_content", lambda *args: None)
    monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "ws://example.test")
    monkeypatch.setattr(browser_cdp_tool, "_browser_cdp_private_guard", lambda **kw: None)

    def read(boundary, text):
        if boundary in ("typed-result", "typed-args"):
            from agent.display import redact_tool_args_for_display
            if boundary == "typed-args":
                args = {"text": text}
                result = redact_tool_args_for_display("browser_type", args)
                assert args == {"text": text}  # do not mutate the actual input
                assert redact_tool_args_for_display("other_tool", args) == args
                return result["text"]
            def type_command(task, command, args, **kwargs):
                assert args == ["@password", text]  # browser gets the original bytes
                return {"success": True}
            monkeypatch.setattr(browser_tool_session, "_run_browser_command", type_command)
            result = json.loads(registry.dispatch("browser_type", {"ref": "@password", "text": text}))
            assert result["success"] is True
            return result["typed"]
        if boundary.startswith("exec-"):
            code = f"import sys; print({text!r}); print({text!r}, file=sys.stderr)"
            result = json.loads(registry.dispatch("browser_exec", {"code": code}))
            assert result["success"] is True
            return result["output"] if boundary == "exec-stdout" else result["stderr"]
        if boundary == "stored-snapshot":
            from tools.browser_tool_snapshot import _store_full_snapshot
            stored = _store_full_snapshot(text)
            assert stored is not None
            return Path(stored).read_text(encoding="utf-8")
        if boundary == "snapshot":
            from tools.browser_tool_snapshot import _redact_browser_output
            # Exercise recursive strings, keys and tuple/list containers unchanged.
            value = {text: [text, (text, 1, None)]}
            result = _redact_browser_output(value)
            key = next(iter(result))
            assert result[key] == [key, (key, 1, None)]
            return key
        if boundary == "dialog":
            from tools.browser_supervisor_dialogs import PendingDialog
            return PendingDialog(id="d1", type="prompt", message=text, default_prompt=text,
                                 opened_at=1.0, cdp_session_id="s1").to_dict()["message"]
        if boundary == "cdp":
            async def call(*args):
                return {text: [text]}
            monkeypatch.setattr(browser_cdp_tool, "_cdp_call", call)
            result = json.loads(registry.dispatch("browser_cdp", {"method": "Runtime.evaluate"}))
            assert result["success"] is True
            key = next(iter(result["result"]))
            assert result["result"][key] == [key]  # keys and string leaves share the policy
            return key

        def command(task, name, *args, **kwargs):
            data = {"result": json.dumps(text), "messages": [{"text": text}], "errors": [{"message": text}]}
            return {"success": True, "data": data}
        monkeypatch.setattr(browser_tool_session, "_run_browser_command", command)
        args = {"expression": "document.title"} if boundary == "eval" else {}
        result = json.loads(registry.dispatch("browser_console", args))
        assert result["success"] is True
        if boundary == "eval":
            return result["result"]
        assert result["js_errors"][0]["message"] == result["console_messages"][0]["text"]
        return result["console_messages"][0]["text"]

    return read


@pytest.mark.parametrize("boundary", ["exec-stdout", "exec-stderr", "snapshot", "stored-snapshot", "console", "eval", "cdp", "dialog", "typed-result", "typed-args"])
@pytest.mark.parametrize("serialization", ["repr", "json"])
@pytest.mark.parametrize("vault_enabled,redact_secrets", [(False, False), (False, True), (True, False), (True, True)])
def test_browser_output_obeys_profile_redaction_policy(
    tmp_path, monkeypatch, browser_output, boundary, serialization, vault_enabled, redact_secrets,
):
    from agent import secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    # A->B->A catches both import-time and process-environment vault policy leaks.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    for home, enabled, redact in ((a, vault_enabled, redact_secrets), (b, False, False)):
        home.mkdir()
        (home / "config.yaml").write_text(yaml.safe_dump({
            "vault": {"enabled": enabled}, "security": {"redact_secrets": redact},
            "browser": {"eval_policy": "allow"},
        }))
    monkeypatch.setenv("HERMES_HOME", str(a))
    secret_scope.set_multiplex_active(True)
    scope = secret_scope.set_secret_scope({})
    value = {"password": "Synthetic-direct-fixture"}
    text = repr(value) if serialization == "repr" else json.dumps(value)
    try:
        for home in (a, b, a):
            token = set_hermes_home_override(home)
            try:
                result = browser_output(boundary, text)
                if home == a and (vault_enabled or redact_secrets):
                    assert value["password"] not in result
                    assert "password" in result
                else:
                    assert result == text + ("\n" if boundary == "exec-stdout" else "")
            finally:
                reset_hermes_home_override(token)
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.set_multiplex_active(False)


@pytest.mark.parametrize("boundary", ["exec-stdout", "exec-stderr", "snapshot", "stored-snapshot", "console", "eval", "cdp", "dialog", "typed-result", "typed-args"])
@pytest.mark.parametrize("serialization", ["scalar", "repr", "json"])
def test_registered_values_follow_runtime_vault_opt_in(tmp_path, browser_output, boundary, serialization):
    from agent import redact, secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    home = tmp_path / "profile"
    home.mkdir()
    config = home / "config.yaml"
    token = set_hermes_home_override(home)
    scope = secret_scope.set_secret_scope({})
    secret = "Synthetic-previously-injected"
    value = {"password": secret}
    text = secret if serialization == "scalar" else repr(value) if serialization == "repr" else json.dumps(value)
    suffix = "\n" if boundary == "exec-stdout" else ""
    try:
        config.write_text("vault:\n  enabled: true\nsecurity:\n  redact_secrets: false\nbrowser:\n  eval_policy: allow\n")
        redact.register_vault_redaction_value(secret)
        for enabled in (True, False, True):
            config.write_text(yaml.safe_dump({
                "vault": {"enabled": enabled}, "security": {"redact_secrets": False},
                "browser": {"eval_policy": "allow"},
            }))
            expected = text.replace(secret, "«redacted-vault-secret»") if enabled else text
            assert redact.redact_registered_vault_values(text) == expected
            assert redact.redact_sensitive_text(text) == expected
            result = browser_output(boundary, text)
            if enabled:
                assert secret not in result
            else:
                assert result == text + suffix
    finally:
        redact.clear_vault_redaction_values()
        secret_scope.reset_secret_scope(scope)
        reset_hermes_home_override(token)
