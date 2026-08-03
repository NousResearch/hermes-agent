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
# E2E regression (#93082): cron/no_agent children keep bare `hermes` on PATH
# ---------------------------------------------------------------------------


def test_e2e_scrubbed_env_resolves_bare_hermes_under_minimal_parent_path(monkeypatch):
    """Regression for #92998/#93082: a gateway launched by systemd/cron with a
    minimal PATH (no hermes console-script dir) must still hand cron job
    children an env whose PATH resolves bare ``hermes``.

    Exercises the REAL factory and the REAL bin-dir resolver — no mocks of the
    helpers. cron/scheduler._run_job_script builds its child env via exactly
    this call (``build_subprocess_env()`` with scrub on).
    """
    import shutil

    from tools.environments import local as local_mod

    bin_dir = local_mod._resolve_hermes_bin_dir()
    if not bin_dir or not os.path.isfile(
        os.path.join(bin_dir, "hermes.exe" if os.name == "nt" else "hermes")
    ):
        pytest.skip("no real hermes console-script install available")

    minimal_path = os.pathsep.join(["/usr/bin", "/bin"])
    monkeypatch.setenv("PATH", minimal_path)
    assert shutil.which("hermes", path=minimal_path) is None

    env = build_subprocess_env(scrub_secrets=True)

    resolved = shutil.which("hermes", path=env.get("PATH", ""))
    assert resolved is not None, (
        f"bare 'hermes' must resolve from the child PATH {env.get('PATH')!r}"
    )
    assert os.path.dirname(resolved) == bin_dir
    assert env["PATH"].split(os.pathsep)[0] == bin_dir
    env2 = build_subprocess_env(env, scrub_secrets=True)
    assert env2["PATH"].split(os.pathsep).count(bin_dir) == 1


def test_e2e_child_never_sees_bws_token_or_password(tmp_path, monkeypatch):
    """BWS bootstrap tokens and password-shaped values stay out of children."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.abc123.def456:xyz789")
    monkeypatch.setenv("DB_PASSWORD", "db-pass-9f2c1a")

    env = build_subprocess_env()
    code = (
        "import os, json; "
        "print(json.dumps({'bws': 'BWS_ACCESS_TOKEN' in os.environ, "
        "'pw': 'DB_PASSWORD' in os.environ}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    result = json.loads(out.stdout)
    assert result["bws"] is False
    assert result["pw"] is False


def test_hermes_subprocess_env_strips_bws_token_and_password(monkeypatch):
    """Non-terminal child environments also strip these values by default."""
    from tools.environments.local import hermes_subprocess_env

    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.abc123.def456:xyz789")
    monkeypatch.setenv("DB_PASSWORD", "db-pass-9f2c1a")

    env = hermes_subprocess_env()
    assert "BWS_ACCESS_TOKEN" not in env
    assert "DB_PASSWORD" not in env


def test_hermes_subprocess_env_strips_password_with_inherit_credentials(monkeypatch):
    """*_PASSWORD values are stripped unconditionally on the non-terminal
    factory even when the caller opts into credential inheritance — a
    model-driving CLI has no legitimate use for a DB/redis/postgres password."""
    from tools.environments.local import hermes_subprocess_env

    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.abc123.def456:xyz789")
    monkeypatch.setenv("DB_PASSWORD", "db-pass-9f2c1a")

    env = hermes_subprocess_env(inherit_credentials=True)
    assert "BWS_ACCESS_TOKEN" not in env
    assert "DB_PASSWORD" not in env


def test_terminal_path_keeps_passthrough_db_password(monkeypatch):
    """An explicitly registered DB_PASSWORD passthrough (skill
    required_environment_variables or terminal.env_passthrough) must still
    reach the terminal child — the *_PASSWORD strip is passthrough-aware."""
    from tools.env_passthrough import clear_env_passthrough, register_env_passthrough

    monkeypatch.setenv("DB_PASSWORD", "db-pass-9f2c1a")
    register_env_passthrough(["DB_PASSWORD"])
    try:
        env = build_subprocess_env()
        assert env.get("DB_PASSWORD") == "db-pass-9f2c1a"
    finally:
        clear_env_passthrough()


def test_make_run_env_strips_password_by_default(monkeypatch):
    """The LocalEnvironment terminal spawn factory (_make_run_env) must
    strip *_PASSWORD values even when they are not in the static blocklist —
    the exact gap egilewski flagged: _sanitize_subprocess_env protected the
    build_subprocess_env/background path, but the live terminal path merged
    os.environ without the credential-shaped password predicate."""
    from tools.environments.local import _make_run_env

    monkeypatch.setenv("DB_PASSWORD", "db-pass-9f2c1a")
    monkeypatch.setenv("REDIS_PASSWORD", "redis-pass-77aa")
    monkeypatch.setenv("PGPASSWORD", "pg-pass-e11")
    monkeypatch.setenv("MYSQL_PWD", "mysql-pwd-4d2")
    monkeypatch.setenv("PASSWORD", "bare-pass-8c1")
    monkeypatch.setenv("MY_HARMLESS_VAR", "keep-me")
    monkeypatch.setenv("PWD", "C:\\some\\cwd")  # shell cwd var must survive

    env = _make_run_env({})

    assert "DB_PASSWORD" not in env
    assert "REDIS_PASSWORD" not in env
    assert "PGPASSWORD" not in env
    assert "MYSQL_PWD" not in env
    assert "PASSWORD" not in env
    assert env.get("MY_HARMLESS_VAR") == "keep-me"
    assert env.get("PWD") == "C:\\some\\cwd"


def test_make_run_env_keeps_passthrough_db_password(monkeypatch):
    """An explicitly registered DB_PASSWORD passthrough must survive the
    terminal spawn factory — the strip is passthrough-aware here too, so a
    skill-registered command that legitimately needs the value still gets it."""
    from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
    from tools.environments.local import _make_run_env

    monkeypatch.setenv("DB_PASSWORD", "db-pass-9f2c1a")
    register_env_passthrough(["DB_PASSWORD"])
    try:
        env = _make_run_env({})
        assert env.get("DB_PASSWORD") == "db-pass-9f2c1a"
    finally:
        clear_env_passthrough()


@pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.path.isfile("/bin/bash"),
    reason="Requires bash; CI sandbox may strip it.",
)
def test_local_environment_e2e_password_denial_and_passthrough(tmp_path, monkeypatch):
    """End-to-end LocalEnvironment execution: a real terminal child must
    NOT see a *_PASSWORD value by default, and MUST see it once explicitly
    registered as passthrough — proving the _make_run_env filter closes the
    live spawn sink, not just the helper factories."""
    from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
    from tools.environments.local import LocalEnvironment

    monkeypatch.setenv("DB_PASSWORD", "db-pass-e2e-77")

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        denied = env.execute(
            'if [ -n "$DB_PASSWORD" ]; then echo "LEAKED"; else echo "DENIED"; fi'
        )
        assert denied["returncode"] == 0
        assert "DENIED" in denied.get("output", "")

        register_env_passthrough(["DB_PASSWORD"])
        try:
            allowed = env.execute(
                'if [ "$DB_PASSWORD" = "db-pass-e2e-77" ]; then echo "PASSTHROUGH"; else echo "MISSING"; fi'
            )
            assert allowed["returncode"] == 0
            assert "PASSTHROUGH" in allowed.get("output", "")
        finally:
            clear_env_passthrough()
    finally:
        env.cleanup()


def test_bws_token_env_cache_is_scoped_per_profile(tmp_path, monkeypatch):
    """The Bitwarden token-name cache must be keyed by the active Hermes
    home, not process-global — a gateway serving multiple profiles by
    switching the context-local home per turn must match each profile's own
    configured access_token_env name (egilewski P1)."""
    import yaml

    import tools.environments.local as _local_mod

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    (profile_a / "config.yaml").write_text(
        yaml.dump({"secrets": {"bitwarden": {"access_token_env": "ALPHA_BWS_TOKEN"}}}),
        encoding="utf-8",
    )
    (profile_b / "config.yaml").write_text(
        yaml.dump({"secrets": {"bitwarden": {"access_token_env": "BETA_BWS_TOKEN"}}}),
        encoding="utf-8",
    )

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    # Fresh cache: no cross-test pollution.
    monkeypatch.setattr(_local_mod, "_configured_bws_token_env_by_home", {})

    token_a = set_hermes_home_override(str(profile_a))
    try:
        assert _local_mod._get_configured_bws_token_env() == "ALPHA_BWS_TOKEN"
        assert _local_mod._is_hermes_internal_secret("ALPHA_BWS_TOKEN") is True
        assert _local_mod._is_hermes_internal_secret("BETA_BWS_TOKEN") is False
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(str(profile_b))
    try:
        assert _local_mod._get_configured_bws_token_env() == "BETA_BWS_TOKEN"
        assert _local_mod._is_hermes_internal_secret("BETA_BWS_TOKEN") is True
        assert _local_mod._is_hermes_internal_secret("ALPHA_BWS_TOKEN") is False
        # The default name stays internal even in a remapped profile: the
        # process-global os.environ can carry a default profile's token into
        # this profile's turn, and it must not cross the child boundary.
        assert _local_mod._is_hermes_internal_secret("BWS_ACCESS_TOKEN") is True
        # Third-party tokens remain registerable in every profile.
        assert _local_mod._is_hermes_internal_secret("STRIPE_ACCESS_TOKEN") is False
    finally:
        reset_hermes_home_override(token_b)

    # Back on profile A the first resolution is still its own name (cache
    # entries are isolated per home, not overwritten by the last visitor).
    token_a2 = set_hermes_home_override(str(profile_a))
    try:
        assert _local_mod._get_configured_bws_token_env() == "ALPHA_BWS_TOKEN"
    finally:
        reset_hermes_home_override(token_a2)


def test_bws_token_env_remap_non_suffix_stripped(tmp_path, monkeypatch):
    """A Bitwarden access_token_env remapped to a non-suffix name (e.g.
    MY_BWS_TOKEN) is stripped exactly, while third-party *_ACCESS_TOKEN vars
    (e.g. STRIPE_ACCESS_TOKEN) stay passthrough-able — the fix for the
    over-broad *_ACCESS_TOKEN suffix match."""
    import yaml

    import tools.environments.local as _local_mod

    config = {"secrets": {"bitwarden": {"access_token_env": "MY_BWS_TOKEN"}}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Reset the per-home cache so the remap is picked up for this home.
    monkeypatch.setattr(_local_mod, "_configured_bws_token_env_by_home", {})

    monkeypatch.setenv("MY_BWS_TOKEN", "0.remapped.token")
    monkeypatch.setenv("STRIPE_ACCESS_TOKEN", "sk_live_third_party")

    assert _local_mod._is_hermes_internal_secret("MY_BWS_TOKEN") is True
    assert _local_mod._is_hermes_internal_secret("STRIPE_ACCESS_TOKEN") is False

    env = build_subprocess_env()
    assert "MY_BWS_TOKEN" not in env
    # Third-party access token is not Hermes-internal — survives on the
    # terminal path (and is registerable as passthrough, unlike BWS).
    assert env.get("STRIPE_ACCESS_TOKEN") == "sk_live_third_party"

    from tools.env_passthrough import (
        clear_env_passthrough,
        is_env_passthrough,
        register_env_passthrough,
    )

    register_env_passthrough(["STRIPE_ACCESS_TOKEN"])
    try:
        assert is_env_passthrough("STRIPE_ACCESS_TOKEN")
    finally:
        clear_env_passthrough()
