"""A refused terminal policy must stop the subprocess HOME builder."""
import pytest


def test_refused_policy_cannot_fall_back_to_launcher_home(monkeypatch):
    from tools.environments.local import build_subprocess_env
    from tools.terminal_scope import (
        TerminalPolicyRefusal, TerminalPolicyUnavailable, set_terminal_scope, reset_terminal_scope,
    )

    monkeypatch.setenv("TERMINAL_HOME_MODE", "real")
    token = set_terminal_scope(TerminalPolicyRefusal("broken profile"))
    try:
        with pytest.raises(TerminalPolicyUnavailable, match="broken profile"):
            build_subprocess_env(scrub_secrets=False)
    finally:
        reset_terminal_scope(token)
