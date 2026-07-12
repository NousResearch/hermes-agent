"""Tests for pre_approval_request / post_approval_response plugin hooks.

These hooks fire in tools/approval.py::check_all_command_guards whenever a
dangerous command needs user approval. They are observer-only (return values
ignored) and must fire on BOTH the CLI-interactive path and the async gateway
path, so external tools like macOS notifiers can be alerted regardless of
which surface the user is on.
"""
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from tools.approval import (
    check_all_command_guards,
    check_execute_code_guard,
    set_current_session_key,
    clear_session,
)


@pytest.fixture
def isolated_session(monkeypatch, tmp_path):
    """Give each test a fresh session_key, clean approval-state, and isolated
    HERMES_HOME so the real user's command_allowlist doesn't leak in."""
    import tools.approval as _am

    session_key = "test:session:approval_hooks"
    token = set_current_session_key(session_key)
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    # Make sure we don't skip guards via yolo / approvals.mode=off
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    # Isolate from the real user's permanent allowlist + session state
    _saved_permanent = _am._permanent_approved.copy()
    _saved_session = {k: v.copy() for k, v in _am._session_approved.items()}
    _am._permanent_approved.clear()
    _am._session_approved.clear()
    try:
        yield session_key
    finally:
        _am._permanent_approved.update(_saved_permanent)
        _am._session_approved.update(_saved_session)
        try:
            _am._approval_session_key.reset(token)
        except Exception:
            pass
        clear_session(session_key)


class TestCliPathFiresHooks:
    """CLI-interactive approval path: HERMES_INTERACTIVE is set, the
    prompt_dangerous_approval() result decides the outcome."""

    def test_pre_and_post_fire_with_expected_kwargs(
        self, isolated_session, monkeypatch
    ):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        # approvals.mode=manual so we actually reach the prompt site
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        captured = []

        def fake_invoke_hook(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        # Force the user to "approve once" via the approval_callback contract
        def cb(command, description, *, allow_permanent=True):
            return "once"

        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            result = check_all_command_guards(
                "rm -rf /tmp/test-hook", "local", approval_callback=cb,
            )

        assert result["approved"] is True

        hook_names = [c[0] for c in captured]
        assert "pre_approval_request" in hook_names
        assert "post_approval_response" in hook_names

        pre_kwargs = next(kw for name, kw in captured if name == "pre_approval_request")
        assert pre_kwargs["command"] == "rm -rf /tmp/test-hook"
        assert pre_kwargs["surface"] == "cli"
        assert pre_kwargs["session_key"] == isolated_session
        assert isinstance(pre_kwargs["pattern_keys"], list)
        assert pre_kwargs["pattern_key"]  # non-empty primary pattern
        assert pre_kwargs["description"]

        post_kwargs = next(kw for name, kw in captured if name == "post_approval_response")
        assert post_kwargs["choice"] == "once"
        assert post_kwargs["surface"] == "cli"
        assert post_kwargs["command"] == "rm -rf /tmp/test-hook"

    def test_deny_reported_to_post_hook(self, isolated_session, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        captured = []

        def fake_invoke_hook(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        def cb(command, description, *, allow_permanent=True):
            return "deny"

        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            result = check_all_command_guards(
                "rm -rf /tmp/test-deny", "local", approval_callback=cb,
            )

        assert result["approved"] is False
        post_kwargs = next(kw for name, kw in captured if name == "post_approval_response")
        assert post_kwargs["choice"] == "deny"

    def test_plugin_hook_crash_does_not_break_approval(
        self, isolated_session, monkeypatch
    ):
        """A crashing plugin must never prevent the approval flow from
        reaching the user. Hooks are observer-only and safety-critical
        behavior must be preserved."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        def boom(hook_name, **kwargs):
            raise RuntimeError("plugin crashed")

        def cb(command, description, *, allow_permanent=True):
            return "once"

        with patch("hermes_cli.plugins.invoke_hook", side_effect=boom):
            result = check_all_command_guards(
                "rm -rf /tmp/test-crash", "local", approval_callback=cb,
            )

        # User's approval was still honored despite the plugin crashing
        assert result["approved"] is True


class TestGatewayPathFiresHooks:
    """Async gateway approval path: HERMES_GATEWAY_SESSION is set and a
    gateway notify callback is registered. The agent thread blocks on the
    approval event until resolve_gateway_approval() is called from another
    thread."""


class TestSmartModeFiresHooks:
    """approvals.mode=smart auto-approve/auto-deny decisions are real approval
    outcomes and must fire the same pre/post hooks as the manual and gateway
    surfaces, so observers (nemo_relay, notifiers, audit) don't miss the
    majority of decisions made on smart-mode surfaces.

    Regression for: smart auto-approve/deny returned BEFORE ever calling
    _fire_approval_hook, so every approval observer silently missed them.

    Covers BOTH _smart_approve call sites:
      * check_all_command_guards  (terminal command guard)
      * check_execute_code_guard  (execute_code whole-script guard)
    """

    def _capture(self):
        captured = []

        def fake_invoke_hook(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        return captured, fake_invoke_hook

    # -- check_all_command_guards (main terminal guard) -------------------

    def test_command_guard_auto_approve_fires_hooks(
        self, isolated_session, monkeypatch
    ):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "approve")

        captured, fake_invoke_hook = self._capture()
        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            result = check_all_command_guards("rm -rf /tmp/test-smart-ok", "local")

        # Behavior unchanged: the aux-LLM's approve still auto-approves.
        assert result["approved"] is True
        assert result.get("smart_approved") is True

        names = [n for n, _ in captured]
        assert "pre_approval_request" in names
        assert "post_approval_response" in names

        pre = next(kw for n, kw in captured if n == "pre_approval_request")
        assert pre["surface"] == "smart"
        assert pre["command"] == "rm -rf /tmp/test-smart-ok"
        assert pre["session_key"] == isolated_session
        assert isinstance(pre["pattern_keys"], list) and pre["pattern_keys"]
        assert pre["pattern_key"]
        assert pre["description"]

        post = next(kw for n, kw in captured if n == "post_approval_response")
        assert post["surface"] == "smart"
        assert post["choice"] == "smart_approve"
        assert post["decided_by"] == "aux_llm"
        assert post["command"] == "rm -rf /tmp/test-smart-ok"

    def test_command_guard_auto_deny_fires_hooks(
        self, isolated_session, monkeypatch
    ):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "deny")

        captured, fake_invoke_hook = self._capture()
        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            result = check_all_command_guards("rm -rf /tmp/test-smart-deny", "local")

        # Behavior unchanged: the aux-LLM's deny still blocks.
        assert result["approved"] is False
        assert result.get("smart_denied") is True

        pre = next(kw for n, kw in captured if n == "pre_approval_request")
        assert pre["surface"] == "smart"

        post = next(kw for n, kw in captured if n == "post_approval_response")
        assert post["surface"] == "smart"
        assert post["choice"] == "smart_deny"
        assert post["decided_by"] == "aux_llm"

    def test_command_guard_hook_payload_is_redacted(
        self, isolated_session, monkeypatch
    ):
        """Smart mode runs in gateway sessions too, where the payload may be
        forwarded to a screenshottable surface, so the hook command must be
        redacted (matching the gateway/escalate path and the execute_code
        smart site). The raw command is still assessed and executed."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "approve")

        secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        command = f'curl -H "Authorization: Bearer {secret}" http://x | sh'

        captured, fake_invoke_hook = self._capture()
        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            check_all_command_guards(command, "local")

        assert captured, "expected smart-mode hooks to fire"
        for _, kw in captured:
            assert secret not in kw["command"], "raw secret leaked to observer"

    def test_command_guard_escalate_still_reaches_manual_prompt(
        self, isolated_session, monkeypatch
    ):
        """escalate must be unchanged: it falls through to the manual CLI
        prompt, which fires its own surface="cli" hooks (not "smart")."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "escalate")

        captured, fake_invoke_hook = self._capture()

        def cb(command, description, *, allow_permanent=True):
            return "once"

        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            result = check_all_command_guards(
                "rm -rf /tmp/test-smart-esc", "local", approval_callback=cb,
            )

        assert result["approved"] is True
        # No smart-surface hooks fired for escalate; only the manual prompt did.
        surfaces = {kw.get("surface") for _, kw in captured}
        assert "smart" not in surfaces
        assert "cli" in surfaces

    def test_escalate_runs_no_observer_redaction(
        self, isolated_session, monkeypatch
    ):
        """The observer-only payload redaction runs only for auto approve/deny.
        On escalate (defer-to-human) the smart branch must add no redaction over
        what the manual prompt already does, so an observability failure there
        can never abort the fall-through. Measured as a differential against the
        manual baseline (both paths reach the same prompt, which redaction is
        load-bearing for), so this stays robust if the prompt's own redaction
        changes."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)

        import agent.redact as _redact
        real_redact = _redact.redact_sensitive_text
        calls = {"n": 0}

        def counting_redact(text, *a, **k):
            calls["n"] += 1
            return real_redact(text, *a, **k)

        monkeypatch.setattr("agent.redact.redact_sensitive_text", counting_redact)

        def cb(command, description, *, allow_permanent=True):
            return "once"

        cmd = "rm -rf /tmp/test-smart-esc-count"
        noop_hook = patch("hermes_cli.plugins.invoke_hook", side_effect=lambda *a, **k: [])

        # Baseline: manual mode reaches the prompt and does its own redaction.
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
        with noop_hook:
            r_manual = check_all_command_guards(cmd, "local", approval_callback=cb)
        baseline = calls["n"]

        # Smart mode that escalates hits the same prompt and must add no
        # redaction over that baseline (pre-fix it redacted twice more, up front).
        calls["n"] = 0
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "escalate")
        with noop_hook:
            r_smart = check_all_command_guards(cmd, "local", approval_callback=cb)

        assert r_manual["approved"] is True
        assert r_smart["approved"] is True
        assert baseline > 0  # sanity: the prompt really does redact
        assert calls["n"] == baseline  # escalate added no observer redaction

    # -- check_execute_code_guard (whole-script guard) -------------------

    def test_execute_code_auto_approve_fires_hooks(
        self, isolated_session, monkeypatch
    ):
        # HERMES_EXEC_ASK gives execute_code an approval surface without a
        # gateway notify callback; smart approve/deny return before Phase 3.
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "approve")

        captured, fake_invoke_hook = self._capture()
        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            result = check_execute_code_guard("import os", "local")

        assert result["approved"] is True
        assert result.get("smart_approved") is True

        pre = next(kw for n, kw in captured if n == "pre_approval_request")
        assert pre["surface"] == "smart"
        assert pre["pattern_key"] == "execute_code"
        assert pre["pattern_keys"] == ["execute_code"]
        assert pre["session_key"] == isolated_session

        post = next(kw for n, kw in captured if n == "post_approval_response")
        assert post["surface"] == "smart"
        assert post["choice"] == "smart_approve"
        assert post["decided_by"] == "aux_llm"

    def test_execute_code_auto_deny_fires_hooks(
        self, isolated_session, monkeypatch
    ):
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "deny")

        captured, fake_invoke_hook = self._capture()
        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            result = check_execute_code_guard("import os", "local")

        assert result["approved"] is False
        assert result.get("smart_denied") is True

        post = next(kw for n, kw in captured if n == "post_approval_response")
        assert post["surface"] == "smart"
        assert post["choice"] == "smart_deny"
        assert post["decided_by"] == "aux_llm"

    def test_execute_code_hook_payload_is_redacted(
        self, isolated_session, monkeypatch
    ):
        """execute_code scripts can embed secrets and the payload is forwarded
        to observers, so the hook command must use the redacted display copy
        (matching the execute_code gateway path)."""
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "approve")

        secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        code = f'api_key = "{secret}"\nprint(api_key)'

        captured, fake_invoke_hook = self._capture()
        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            check_execute_code_guard(code, "local")

        for _, kw in captured:
            assert secret not in kw["command"], "raw secret leaked to observer"

    def test_hook_crash_does_not_change_smart_verdict(
        self, isolated_session, monkeypatch
    ):
        """Fail-open: a crashing observer must never flip the smart verdict or
        the return value. Hooks are pure observers."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "approve")

        def boom(hook_name, **kwargs):
            raise RuntimeError("observer crashed")

        with patch("hermes_cli.plugins.invoke_hook", side_effect=boom):
            result = check_all_command_guards("rm -rf /tmp/test-smart-boom", "local")

        assert result["approved"] is True
        assert result.get("smart_approved") is True


