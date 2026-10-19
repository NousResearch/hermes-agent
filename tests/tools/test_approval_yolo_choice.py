"""The YOLO approval choice (#98139): approve this command, then stop asking this session.

Split out of :mod:`tests.tools.test_approval`, which is already over its line budget: the choice is
a distinct contract (a mid-task trust grant, not a per-pattern allowlist entry).
"""

from unittest.mock import patch as mock_patch


class TestApprovalYoloChoice:
    """Mid-task trust: the prompt's YOLO choice approves the pending command and turns the gate off
    for the rest of the session (#98139)."""

    SESSION_KEY = "test-yolo-session"

    def _interactive_env(self):
        # A real session key, like the CLI/gateway export: the gate resolves its key from it and
        # the yolo bypass is keyed off the same value.
        return mock_patch.dict(
            "os.environ",
            {"HERMES_INTERACTIVE": "1", "HERMES_SESSION_KEY": self.SESSION_KEY},
            clear=False,
        )

    def test_yolo_choice_approves_command_and_stops_asking(self):
        from unittest.mock import patch as _patch
        from tools import approval as mod

        mod._session_approved.clear()
        mod._permanent_approved.clear()
        session_key = self.SESSION_KEY
        mod.disable_session_yolo(session_key)
        assert mod.is_session_yolo_enabled(session_key) is False

        cfg = {"approvals": {"mode": "manual"}}
        with self._interactive_env():
            with _patch("hermes_cli.config.load_config_readonly", return_value=cfg):
                result = mod.check_all_command_guards(
                    "rm -rf /var/data", "local",
                    approval_callback=lambda *a, **kw: "yolo",
                )

        # 1. the command that was pending when the user picked YOLO runs
        assert result["approved"] is True
        # 2. the session-wide bypass is armed, so the next command never prompts
        assert mod.is_session_yolo_enabled(session_key) is True
        # 3. no pattern was allowlisted: the wider scope replaces the narrower grant
        assert mod._session_approved.get(session_key, set()) == set()

        def _unexpected_prompt(*a, **kw):
            raise AssertionError("a command must not prompt once session YOLO is armed")

        with self._interactive_env():
            with _patch("hermes_cli.config.load_config_readonly", return_value=cfg):
                second = mod.check_all_command_guards(
                    "rm -rf /var/data2", "local", approval_callback=_unexpected_prompt)
        assert second["approved"] is True
        mod.disable_session_yolo(session_key)

    def test_smart_deny_override_yolo_choice_stays_one_operation(self):
        """A smart-DENY owner override is one operation: answering it with 'yolo' runs that command
        but must not arm the session-wide bypass (#98139)."""
        from unittest.mock import patch as _patch
        from tools import approval as mod

        session_key = self.SESSION_KEY
        mod.disable_session_yolo(session_key)
        with _patch.object(mod, "_smart_gate", return_value=(None, True)), \
             _patch.object(mod, "prompt_dangerous_approval", return_value="yolo"):
            decision = mod._human_decision(
                mod._COMMAND_GATE, command="rm -rf /var/data", description="recursive delete",
                pattern_key="rm -rf", pattern_keys=["rm -rf"],
                session_key=session_key, approval_callback=None,
                is_cli=True, is_gateway=False, is_ask=False, smart=True,
            )
        assert decision["approved"] is True
        assert mod.is_session_yolo_enabled(session_key) is False
