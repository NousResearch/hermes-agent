"""A child spawned FOR another profile never inherits this process's authorization gates (#113270).

``DISCORD_ALLOWED_CHANNELS`` / ``TELEGRAM_GROUP_ALLOWED_CHATS`` / ``GATEWAY_ALLOW_ALL_USERS``-class names
decide WHO may talk to the agent. They are not credentials (no secret scrub sees them), a unit-file
``Environment=`` or operator export never appears in the launch ``.env`` (no name list sees them), and
the target profile's own ``.env`` rarely defines them (its dotenv load never overwrites them) — so a
``hermes -p B`` child built from profile A's process enforced A's channel list as its own.
"""

import json
import os
import subprocess
import sys

import pytest

from tools.environments.local import served_profile_child_env

# Gates injected outside any dotenv — the shape a name-list scrub cannot see. Mix of allowlist,
# deny list, allow-all opt-in, bot policy, channel scoping and a plugin adapter's gate.
_GATES = {
    "DISCORD_ALLOWED_CHANNELS": "111", "DISCORD_ALLOWED_ROLES": "r1", "DISCORD_IGNORED_CHANNELS": "222",
    "DISCORD_ALLOW_BOTS": "all", "TELEGRAM_GROUP_ALLOWED_CHATS": "-100", "SLACK_ALLOWED_CHANNELS": "C1",
    "WHATSAPP_GROUP_ALLOW_FROM": "+1555", "GATEWAY_ALLOW_ALL_USERS": "true",
}
_OPERATOR_ENV = {"DEPLOYMENT_ALLOWED_REGIONS": "us-east-1,eu-west-1"}
_PROBE = "import json,os;print(json.dumps({k:os.environ.get(k) for k in %r}))" % sorted(_GATES)


def _seen_by_child(env: dict) -> set[str]:
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=60)
    return {k for k, v in json.loads(out.stdout.strip().splitlines()[-1]).items() if v is not None}


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Launch home A (the process's HERMES_HOME) and routed home B; gates in the process env only."""
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    (a / ".env").write_text("A_MARKER=a\n", encoding="utf-8")
    (b / ".env").write_text("B_MARKER=b\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(a))
    for key, value in _GATES.items():
        monkeypatch.setenv(key, value)
    return a, b


def test_routed_child_drops_gates_and_same_home_child_keeps_them(homes):
    """The seam every served-profile spawn funnels through: routed target → no gate reaches the child
    (observed from INSIDE a real child); same-home target → an operator export is preserved."""
    a, b = homes
    assert _seen_by_child(served_profile_child_env(base=os.environ, target_home=b)) == set()
    assert _seen_by_child(served_profile_child_env(base=os.environ, target_home=a)) == set(_GATES)


def test_routed_child_keeps_unrelated_operator_allowlist_env(homes, monkeypatch):
    """Only documented platform gates are profile-scoped; operator allowlists survive routing."""
    _, b = homes
    for key, value in _OPERATOR_ENV.items():
        monkeypatch.setenv(key, value)

    routed = served_profile_child_env(base=os.environ, target_home=b)
    assert {key: routed[key] for key in _OPERATOR_ENV} == _OPERATOR_ENV


def test_update_recovery_child_env_drops_gates_only_for_other_profiles(homes):
    """The updater relaunches EVERY profile from one environment; the per-profile child env must
    strip gates for a foreign profile and keep them for the profile the updater itself runs as."""
    from hermes_cli import update_restart_recovery as recovery

    routed = recovery._child_environment("b")
    same = recovery._child_environment("default")
    assert not any(key in routed for key in _GATES)
    assert all(same[key] == value for key, value in _GATES.items())
    # The recovery marker and the gateway-owner scrub are unchanged on both.
    for env in (routed, same):
        assert env[recovery._RECOVERY_ENV] == "1"
        assert not any(marker in env for marker in recovery._GATEWAY_MARKERS)
