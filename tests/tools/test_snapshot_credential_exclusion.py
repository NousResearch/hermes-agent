"""Snapshot credential exclusion must strip user/custom creds but keep infra.

Action 180-A: the persisted terminal session snapshot (``hermes-snap-*.sh``,
a login-shell ``export -p`` dump) must not carry credentials. The fix reuses the
existing subprocess-spawn exclusion sources (provider blocklist + internal-secret
predicate) plus an explicit snapshot-specific set and a custom-provider API-key
glob, WITHOUT a broad KEY|SECRET|TOKEN pattern that would wrongly strip the AWS
credential chain (intentionally inheritable in the local terminal).
"""

import pytest

from tools.environments.local import (
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _is_hermes_internal_secret,
)
from tools.environments.local import LocalEnvironment


@pytest.fixture
def env():
    inst = LocalEnvironment.__new__(LocalEnvironment)
    return inst


def test_custom_api_key_glob_matches_known_and_gstudio(env):
    glob = env._snapshot_credential_exclusion_glob()
    assert glob.match("HERMES_CUSTOM_GSTUDIO_API_KEY")
    assert glob.match("HERMES_CUSTOM_B_AI4_API_KEY")
    assert glob.match("HERMES_CUSTOM_AGENTROUTER_API_KEY")
    # not a custom provider key
    assert not glob.match("OPENAI_API_KEY")
    assert not glob.match("Github_personal_TOKEN")


def test_credential_exclusion_names_include_specific_set(env):
    names = env._snapshot_credential_exclusion_names()
    assert "Github_personal_TOKEN" in names
    assert "HERMES_DESKTOP_PASSWORD_STORE" in names
    assert "ANTHROPIC_AUTH_TOKEN" in names


def test_aws_chain_not_stripped(env):
    """Critical: the broad KEY|SECRET|TOKEN pattern must NOT be used; the AWS
    inheritable chain must remain in the snapshot."""
    names = env._snapshot_credential_exclusion_names()
    glob = env._snapshot_credential_exclusion_glob()
    aws = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]
    assert all(a not in names for a in aws)
    assert all(not glob.match(a) for a in aws)


def test_eight_leaking_creds_covered(env, monkeypatch):
    """Reproduces the Action 180 finding: the 8 creds seen in hermes-snap-*.sh
    must all be excluded by the combined policy (blocklist + specific + glob)."""
    inst = env
    cred_names = inst._snapshot_credential_exclusion_names()
    cred_glob = inst._snapshot_credential_exclusion_glob()
    infra = {"SSH_AUTH_SOCK", "TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "HERMES_SESSION_KEY"}

    # Simulate the init_session merge against a fake env containing the 8.
    fake_env = {
        "HERMES_CUSTOM_AGENTROUTER_API_KEY": "x",
        "HERMES_CUSTOM_B_AI_API_KEY": "x",
        "HERMES_CUSTOM_B_AI2_API_KEY": "x",
        "HERMES_CUSTOM_B_AI3_API_KEY": "x",
        "HERMES_CUSTOM_B_AI4_API_KEY": "x",
        "HERMES_CUSTOM_GMI_API_KEY": "x",
        "HERMES_CUSTOM_GSTUDIO_API_KEY": "x",
        "Github_personal_TOKEN": "x",
        "AWS_ACCESS_KEY_ID": "x",
        "AWS_SECRET_ACCESS_KEY": "x",
        "AWS_SESSION_TOKEN": "x",
        "SSH_AUTH_SOCK": "/tmp/ssh",
        "TERMINAL_DOCKER_SHARED_CONTAINER_KEY": "x",
        "HERMES_SESSION_KEY": "x",
        "PATH": "/usr/bin",
    }
    monkeypatch.setattr("os.environ", fake_env)

    stripped = set()
    for n in fake_env:
        if n in infra:
            continue
        if n in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_hermes_internal_secret(n):
            stripped.add(n)
            continue
        if cred_glob.match(n) or n in cred_names:
            stripped.add(n)

    eight = [
        "HERMES_CUSTOM_AGENTROUTER_API_KEY",
        "HERMES_CUSTOM_B_AI_API_KEY",
        "HERMES_CUSTOM_B_AI2_API_KEY",
        "HERMES_CUSTOM_B_AI3_API_KEY",
        "HERMES_CUSTOM_B_AI4_API_KEY",
        "HERMES_CUSTOM_GMI_API_KEY",
        "HERMES_CUSTOM_GSTUDIO_API_KEY",
        "Github_personal_TOKEN",
    ]
    assert all(c in stripped for c in eight), stripped
    # AWS must survive
    aws_chain = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]
    assert all(a not in stripped for a in aws_chain)
    # infra must survive
    assert all(i not in stripped for i in infra)
