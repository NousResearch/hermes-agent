"""Tests for tools.environments.local.build_subprocess_env — the single
factory for child-process environments (profile-home + secret-scrub owner).
"""

import os
import subprocess
import sys

import pytest

from tools.environments.local import build_subprocess_env


# ---------------------------------------------------------------------------
# Unit: scrub path delegates to _sanitize_subprocess_env semantics
# ---------------------------------------------------------------------------

def test_scrub_on_strips_provider_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    env = build_subprocess_env()
    assert "ANTHROPIC_API_KEY" not in env


def test_scrub_on_strips_dynamic_internal_secret(monkeypatch):
    monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "sk-aux")
    monkeypatch.setenv("GATEWAY_RELAY_FOO_TOKEN", "tok")
    env = build_subprocess_env()
    assert "AUXILIARY_VISION_API_KEY" not in env
    assert "GATEWAY_RELAY_FOO_TOKEN" not in env


def test_scrub_on_strips_venv_markers(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/some/venv")
    env = build_subprocess_env()
    assert "VIRTUAL_ENV" not in env


def test_scrub_on_forwards_extra_like_sanitize_extra_env(monkeypatch):
    env = build_subprocess_env(extra={"MY_HARMLESS_VAR": "1"})
    assert env.get("MY_HARMLESS_VAR") == "1"
    # extra still goes through the blocklist on the scrub path
    env2 = build_subprocess_env(extra={"ANTHROPIC_API_KEY": "sk"})
    assert "ANTHROPIC_API_KEY" not in env2


def test_scrub_on_matches_sanitize_exactly(monkeypatch):
    """build_subprocess_env(scrub_secrets=True) must equal
    _sanitize_subprocess_env(os.environ.copy()) — single owner, zero drift."""
    from tools.environments.local import _sanitize_subprocess_env

    monkeypatch.setenv("OPENAI_API_KEEP_TEST", "x")
    assert build_subprocess_env() == _sanitize_subprocess_env(os.environ.copy())


# ---------------------------------------------------------------------------
# Unit: no-scrub path preserves content exactly
# ---------------------------------------------------------------------------

def test_no_scrub_no_home_is_exact_environ_copy(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    assert env == os.environ.copy()
    assert env is not os.environ  # detached copy


def test_no_scrub_explicit_base_preserved(monkeypatch):
    base = {"PATH": "/bin", "ANTHROPIC_API_KEY": "sk"}
    env = build_subprocess_env(base, scrub_secrets=False, inherit_profile_home=False)
    assert env == base
    assert env is not base


def test_extra_wins_last_on_no_scrub_path():
    base = {"HERMES_HOME": "/old"}
    env = build_subprocess_env(
        base, scrub_secrets=False, inherit_profile_home=False,
        extra={"HERMES_HOME": "/new"},
    )
    assert env["HERMES_HOME"] == "/new"


def test_no_scrub_inherit_profile_home_bridges_context_override(tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(str(tmp_path))
    try:
        env = build_subprocess_env(
            {"PATH": "/bin"}, scrub_secrets=False, inherit_profile_home=True
        )
    finally:
        reset_hermes_home_override(token)
    assert env["HERMES_HOME"] == str(tmp_path)


# ---------------------------------------------------------------------------
# E2E: real subprocess sees the factory's contract
# ---------------------------------------------------------------------------

def test_e2e_child_sees_hermes_home_and_no_planted_secret(tmp_path, monkeypatch):
    """A real child spawned with a factory-built env must see HERMES_HOME
    propagated and (with scrub on) a planted provider-style key absent."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-FAKE-planted")
    monkeypatch.setenv("AUXILIARY_FAKE_API_KEY", "sk-FAKE-aux")

    env = build_subprocess_env()  # scrub on (default)

    code = (
        "import os, json; "
        "print(json.dumps({'home': os.environ.get('HERMES_HOME'), "
        "'k1': 'ANTHROPIC_API_KEY' in os.environ, "
        "'k2': 'AUXILIARY_FAKE_API_KEY' in os.environ}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    import json

    result = json.loads(out.stdout)
    assert result["home"] == str(hermes_home)
    assert result["k1"] is False
    assert result["k2"] is False


def test_e2e_no_scrub_child_keeps_planted_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-FAKE-planted")
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    out = subprocess.run(
        [sys.executable, "-c",
         "import os; print(os.environ.get('ANTHROPIC_API_KEY', ''))"],
        env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    assert out.stdout.strip() == "sk-FAKE-planted"
