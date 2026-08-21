"""Tests for tools.environments.local.build_subprocess_env — the single
factory for child-process environments (profile-home + secret-scrub owner).
"""

import json
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


def test_scrub_on_forwards_extra_like_sanitize_extra_env(monkeypatch):
    env = build_subprocess_env(extra={"MY_HARMLESS_VAR": "1"})
    assert env.get("MY_HARMLESS_VAR") == "1"
    # extra still goes through the blocklist on the scrub path
    env2 = build_subprocess_env(extra={"ANTHROPIC_API_KEY": "sk"})
    assert "ANTHROPIC_API_KEY" not in env2


# ---------------------------------------------------------------------------
# Unit: no-scrub path preserves content exactly
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# E2E: provenance-aware value scrub (#77164) — externally-applied secrets
# under arbitrary (non-credential-shaped) names must not reach children;
# explicitly registered env_passthrough vars still win.
# ---------------------------------------------------------------------------

def _seed_applied_secrets(home, values):
    """Populate the per-home applied-secrets snapshot the way the real code
    keys it (resolved home path) and return the key for cleanup."""
    from hermes_cli import env_loader

    home_key = str(home.resolve())
    env_loader._SECRET_SOURCE_VALUES_BY_HOME[home_key] = dict(values)
    return home_key


def test_e2e_provenance_scrub_strips_arbitrarily_named_applied_secret(tmp_path, monkeypatch):
    """A secret applied by an external source under a non-credential-shaped
    name (DATABASE_URL, FOO, a 1Password-style item key) must not reach a
    real spawned child — name-shape predicates alone miss all three."""
    from hermes_cli import env_loader

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    applied = {
        "DATABASE_URL": "postgres://svc:«redacted:db-pass»@db.internal:5432/app",
        "FOO": "«redacted:op-item-value»",
        "ONEPASS_ITEM_KEY": "«redacted:1password-item»",
    }
    home_key = _seed_applied_secrets(hermes_home, applied)
    try:
        for name, value in applied.items():
            monkeypatch.setenv(name, value)

        env = build_subprocess_env()  # scrub on (default)

        code = (
            "import os, json; print(json.dumps({"
            "'k1': 'DATABASE_URL' in os.environ, "
            "'k2': 'FOO' in os.environ, "
            "'k3': 'ONEPASS_ITEM_KEY' in os.environ}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env, capture_output=True, text=True, timeout=60, check=True,
        )
        result = json.loads(out.stdout)
        assert result["k1"] is False, "DATABASE_URL leaked to child"
        assert result["k2"] is False, "FOO leaked to child"
        assert result["k3"] is False, "1Password item key leaked to child"
    finally:
        env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(home_key, None)


def test_e2e_provenance_scrub_strips_secret_under_renamed_key(tmp_path, monkeypatch):
    """#77164: the scrub is VALUE-based. A child env carrying the applied
    secret value under a DIFFERENT name than the snapshot (e.g. a provider
    that renames DATABASE_URL to DB_URI before forking) must still be
    stripped — name-shape predicates would miss it entirely."""
    from hermes_cli import env_loader

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    secret_value = "«redacted:secret-db-value»"
    # Snapshot has the value under DATABASE_URL...
    home_key = _seed_applied_secrets(hermes_home, {"DATABASE_URL": secret_value})
    try:
        # ...but the child env carries it under a renamed, non-credential-shaped key.
        monkeypatch.setenv("DB_URI", secret_value)

        env = build_subprocess_env()  # scrub on (default)

        code = (
            "import os, json; print(json.dumps({"
            "'renamed': 'DB_URI' in os.environ}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env, capture_output=True, text=True, timeout=60, check=True,
        )
        result = json.loads(out.stdout)
        assert result["renamed"] is False, "renamed-key secret leaked to child"
    finally:
        env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(home_key, None)


def test_e2e_provenance_scrub_keeps_legitimate_vars(tmp_path, monkeypatch):
    """Non-secret vars that legitimately reach children (unrelated vars)
    must not be stripped by the provenance scrub."""
    from hermes_cli import env_loader

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    home_key = _seed_applied_secrets(
        hermes_home, {"DATABASE_URL": "«redacted:secret-db-value»"}
    )
    try:
        monkeypatch.setenv("DATABASE_URL", "«redacted:secret-db-value»")
        monkeypatch.setenv("MY_UNRELATED_VAR", "just-a-value")

        env = build_subprocess_env()

        code = (
            "import os, json; print(json.dumps({"
            "'legit': os.environ.get('MY_UNRELATED_VAR'), "
            "'leak': 'DATABASE_URL' in os.environ}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env, capture_output=True, text=True, timeout=60, check=True,
        )
        result = json.loads(out.stdout)
        assert result["legit"] == "just-a-value"
        assert result["leak"] is False
    finally:
        env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(home_key, None)


def test_e2e_provenance_scrub_passthrough_wins(tmp_path, monkeypatch):
    """#77164: an explicitly registered env_passthrough var still receives
    its value in a real child even when that value also appears in the
    applied-secrets snapshot — the passthrough contract outranks the scrub."""
    from hermes_cli import env_loader
    from tools.env_passthrough import clear_env_passthrough, register_env_passthrough

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    db_url = "postgres://svc:«redacted:db-pass»@db.internal:5432/app"
    home_key = _seed_applied_secrets(hermes_home, {"DATABASE_URL": db_url})
    try:
        monkeypatch.setenv("DATABASE_URL", db_url)
        register_env_passthrough(["DATABASE_URL"])
        try:
            env = build_subprocess_env()  # scrub on (default)
            out = subprocess.run(
                [sys.executable, "-c",
                 "import os; print(os.environ.get('DATABASE_URL', 'MISSING'))"],
                env=env, capture_output=True, text=True, timeout=60, check=True,
            )
            assert out.stdout.strip() == db_url
        finally:
            clear_env_passthrough()
    finally:
        env_loader._SECRET_SOURCE_VALUES_BY_HOME.pop(home_key, None)
