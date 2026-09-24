"""A profile gate is a PLATFORM prefix plus a gate-shaped suffix; an operator's own variable that
merely contains ``_ALLOWED_`` is script data and must survive the routed-child strip (#119539)."""

import os

from tools.environments.local import build_subprocess_env
from tools.environments.local_env_policy import is_profile_gate_env


def test_operator_allowlist_shaped_vars_are_not_gates_but_platform_gates_are():
    # Same suffix shape on both sides — only the owner prefix decides.
    assert not is_profile_gate_env("DEMO_ALLOWED_SENDER")
    assert not is_profile_gate_env("ACCESS_ALLOWED_CALLBACK_ACE_TYPE")
    assert not is_profile_gate_env("HERMES_MEDIA_ALLOW_DIRS")
    for gate in ("DISCORD_ALLOWED_CHANNELS", "TELEGRAM_GROUP_ALLOWED_CHATS", "GATEWAY_ALLOW_ALL_USERS",
                 "WHATSAPP_GROUP_ALLOW_FROM", "QQ_GROUP_ALLOWED_USERS", "IRC_ALLOW_ALL_USERS"):
        assert is_profile_gate_env(gate), gate


def test_routed_no_agent_script_env_keeps_operator_allowlist_and_drops_platform_gate(tmp_path, monkeypatch):
    """The cron ``no_agent`` spawn seam: a child built for ANOTHER profile drops Hermes gates but
    keeps the operator's script data, whatever its name looks like."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch = tmp_path / ".hermes"
    other = launch / "profiles" / "other"
    other.mkdir(parents=True)
    (other / ".env").write_text("OTHER_MARKER=1\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("DEMO_ALLOWED_SENDER", "someone@example.com")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111")

    ht = set_hermes_home_override(str(other))
    st = set_secret_scope(build_profile_secret_scope(other), profile_home=str(other))
    try:
        env = build_subprocess_env(strip_launch_profile=True)
    finally:
        reset_secret_scope(st)
        reset_hermes_home_override(ht)
    assert env.get("DEMO_ALLOWED_SENDER") == "someone@example.com"
    assert "DISCORD_ALLOWED_CHANNELS" not in env
    assert os.environ["DISCORD_ALLOWED_CHANNELS"] == "111"  # the strip never mutates the process env
