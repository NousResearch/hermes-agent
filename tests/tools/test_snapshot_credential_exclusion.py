"""Snapshot credential exclusion must strip user/custom creds but keep infra.

Action 180-A: the persisted terminal session snapshot (``hermes-snap-*.sh``,
a login-shell ``export -p`` dump) must not carry credentials. The fix reuses the
existing subprocess-spawn exclusion sources (provider blocklist + internal-secret
predicate) plus an explicit snapshot-specific set and a custom-provider API-key
glob, WITHOUT a broad KEY|SECRET|TOKEN pattern that would wrongly strip the AWS
credential chain (intentionally inheritable in the local terminal).

Tests exercise the PRODUCTION exclusion path (``_snapshot_credential_exclusions``)
so the declared "same exclusion sources" invariant cannot silently diverge.
"""

import pytest

from tools.environments.local import _is_hermes_internal_secret
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
    exclusions = env._snapshot_credential_exclusions()
    aws = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]
    assert all(a not in exclusions for a in aws)


def test_dynamic_auxiliary_family_covered(env, monkeypatch):
    """AUXILIARY_<TASK>_API_KEY / _BASE_URL must be stripped by the production
    path (dynamic-family contract)."""
    fake_env = {
        "AUXILIARY_VISION_API_KEY": "x",
        "AUXILIARY_WEB_EXTRACT_API_KEY": "x",
        "AUXILIARY_APPROVAL_API_KEY": "x",
        "AUXILIARY_MY_PLUGIN_TASK_API_KEY": "x",
        "AUXILIARY_VISION_BASE_URL": "x",
        "AUXILIARY_COMPRESSION_BASE_URL": "x",
        # non-secret AUXILIARY_* must survive
        "AUXILIARY_VISION_PROVIDER": "openai",
        "AUXILIARY_VISION_MODEL": "gpt-4",
    }
    exclusions = env._snapshot_credential_exclusions(live_env=fake_env)
    assert "AUXILIARY_VISION_API_KEY" in exclusions
    assert "AUXILIARY_WEB_EXTRACT_API_KEY" in exclusions
    assert "AUXILIARY_APPROVAL_API_KEY" in exclusions
    assert "AUXILIARY_MY_PLUGIN_TASK_API_KEY" in exclusions
    assert "AUXILIARY_VISION_BASE_URL" in exclusions
    assert "AUXILIARY_COMPRESSION_BASE_URL" in exclusions
    assert "AUXILIARY_VISION_PROVIDER" not in exclusions
    assert "AUXILIARY_VISION_MODEL" not in exclusions


def test_dynamic_gateway_relay_family_covered(env, monkeypatch):
    """GATEWAY_RELAY_*_{SECRET,KEY,TOKEN} must be stripped by the production
    path (dynamic-family contract)."""
    fake_env = {
        "GATEWAY_RELAY_SECRET": "x",
        "GATEWAY_RELAY_DELIVERY_KEY": "x",
        "GATEWAY_RELAY_SESSION_TOKEN": "x",
        # non-secret GATEWAY_RELAY_* must survive
        "GATEWAY_RELAY_URL": "https://example.com",
        "GATEWAY_RELAY_PLATFORMS": "telegram",
    }
    exclusions = env._snapshot_credential_exclusions(live_env=fake_env)
    assert "GATEWAY_RELAY_SECRET" in exclusions
    assert "GATEWAY_RELAY_DELIVERY_KEY" in exclusions
    assert "GATEWAY_RELAY_SESSION_TOKEN" in exclusions
    assert "GATEWAY_RELAY_URL" not in exclusions
    assert "GATEWAY_RELAY_PLATFORMS" not in exclusions


def test_eight_leaking_creds_covered_through_production_path(env):
    """Reproduces the Action 180 finding: the 8 creds seen in hermes-snap-*.sh
    must all be excluded by the PRODUCTION exclusion path (not a test-side
    reimplementation)."""
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
    exclusions = env._snapshot_credential_exclusions(live_env=fake_env)

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
    assert all(c in exclusions for c in eight), exclusions
    # AWS must survive
    aws_chain = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]
    assert all(a not in exclusions for a in aws_chain)
    # infra must survive
    infra = {"SSH_AUTH_SOCK", "TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "HERMES_SESSION_KEY"}
    assert all(i not in exclusions for i in infra)


def test_wrap_command_applies_credential_exclusions():
    """The post-command snapshot re-dump in _wrap_command must apply the SAME
    credential-exclusion policy as init_session (closes the two-writer root
    cause from #62336)."""
    from tools.environments.base import BaseEnvironment

    class TestEnv(BaseEnvironment):
        def __init__(self):
            self._snapshot_path = "/tmp/test-snap.sh"
            self._snapshot_ready = True
            self._profile_scoped_passthrough = False
            self._snapshot_passthrough_names = set()
            self._cwd_marker = "HERMES_CWD_MARKER_TEST"

        def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
            raise NotImplementedError

        def cleanup(self):
            pass

    env = TestEnv()
    wrapped = env._wrap_command("echo hello", "/tmp")

    # The wrapped command must contain the credential exclusion names
    # (they appear in the unset list inside _export_dump_excluding_session_vars)
    assert "Github_personal_TOKEN" in wrapped
    assert "HERMES_DESKTOP_PASSWORD_STORE" in wrapped
    assert "ANTHROPIC_AUTH_TOKEN" in wrapped

    # AWS chain must NOT be in the unset list
    # (they would appear as quoted names in the unset command)
    assert "'AWS_ACCESS_KEY_ID'" not in wrapped
    assert "'AWS_SECRET_ACCESS_KEY'" not in wrapped
    assert "'AWS_SESSION_TOKEN'" not in wrapped


def test_is_hermes_internal_secret_direct():
    """Direct test of the _is_hermes_internal_secret predicate for
    representative cases."""
    assert _is_hermes_internal_secret("AUXILIARY_VISION_API_KEY")
    assert _is_hermes_internal_secret("AUXILIARY_WEB_EXTRACT_API_KEY")
    assert _is_hermes_internal_secret("AUXILIARY_APPROVAL_API_KEY")
    assert _is_hermes_internal_secret("AUXILIARY_MY_PLUGIN_TASK_API_KEY")
    assert _is_hermes_internal_secret("AUXILIARY_VISION_BASE_URL")
    assert _is_hermes_internal_secret("AUXILIARY_COMPRESSION_BASE_URL")
    assert _is_hermes_internal_secret("GATEWAY_RELAY_SECRET")
    assert _is_hermes_internal_secret("GATEWAY_RELAY_DELIVERY_KEY")
    assert _is_hermes_internal_secret("GATEWAY_RELAY_SESSION_TOKEN")
    # non-secret suffixes must NOT match
    assert not _is_hermes_internal_secret("AUXILIARY_VISION_PROVIDER")
    assert not _is_hermes_internal_secret("AUXILIARY_VISION_MODEL")
    assert not _is_hermes_internal_secret("GATEWAY_RELAY_URL")
    assert not _is_hermes_internal_secret("GATEWAY_RELAY_PLATFORMS")
    assert not _is_hermes_internal_secret("GATEWAY_RELAY_ID")
    assert not _is_hermes_internal_secret("PATH")
    assert not _is_hermes_internal_secret("MY_APP_KEY")
