"""Real children observe factory scrubbing, overrides and routed profile homes."""

import json
import os
import subprocess
import sys

import pytest

from tests.tools._child_env_fixtures import child_env, observe_child  # noqa: F401
from tools.environments.local import build_subprocess_env


@pytest.mark.parametrize("scrub,inherit_home", [(True, True), (True, False), (False, True), (False, False)])
def test_factory_child_policy_and_profile_home(child_env, monkeypatch, scrub, inherit_home):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    routed = child_env / "profiles/coder"
    routed.mkdir(parents=True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-parent-secret")
    monkeypatch.setenv("AUXILIARY_FAKE_API_KEY", "fake-auxiliary")
    monkeypatch.setenv("GATEWAY_RELAY_FOO_TOKEN", "fake-relay")
    before = dict(os.environ)
    token = set_hermes_home_override(str(routed))
    try:
        env = build_subprocess_env(scrub_secrets=scrub, inherit_profile_home=inherit_home,
                                   extra={"MY_HARMLESS_VAR": "caller", "ANTHROPIC_API_KEY": "fake-extra"})
        result = observe_child(env, ["HERMES_HOME", "ANTHROPIC_API_KEY", "AUXILIARY_FAKE_API_KEY",
                                     "GATEWAY_RELAY_FOO_TOKEN", "MY_HARMLESS_VAR"])
    finally:
        reset_hermes_home_override(token)
    assert result == {
        "HERMES_HOME": str(routed) if scrub or inherit_home else before["HERMES_HOME"],
        "ANTHROPIC_API_KEY": None if scrub else "fake-extra",
        "AUXILIARY_FAKE_API_KEY": None if scrub else "fake-auxiliary",
        "GATEWAY_RELAY_FOO_TOKEN": None if scrub else "fake-relay", "MY_HARMLESS_VAR": "caller",
    }
    assert dict(os.environ) == before


def test_no_scrub_is_byte_preserving_except_explicit_extra():
    base = {"PATH": "/bin", "PYTHONHOME": "/poison", "VIRTUAL_ENV": "/venv",
            "CONDA_PREFIX": "/conda", "SERVICE_TOKEN": "fake", "HERMES_HOME": "/original"}
    assert build_subprocess_env(base, scrub_secrets=False, inherit_profile_home=False) == base
    assert build_subprocess_env(base, scrub_secrets=False, extra={"HERMES_HOME": "/caller"})["HERMES_HOME"] == "/caller"


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


def test_e2e_child_strips_bws_but_preserves_single_profile_password(tmp_path, monkeypatch):
    """Single-profile children deny Hermes bootstrap auth, not user passwords."""
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
    assert result["pw"] is True


def test_hermes_subprocess_env_strips_bws_but_preserves_password(monkeypatch):
    """Non-terminal single-profile children retain user shell passwords."""
    from tools.environments.local import hermes_subprocess_env

    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.abc123.def456:xyz789")
    monkeypatch.setenv("DB_PASSWORD", "db-pass-9f2c1a")

    env = hermes_subprocess_env()
    assert "BWS_ACCESS_TOKEN" not in env
    assert env["DB_PASSWORD"] == "db-pass-9f2c1a"


def test_hermes_subprocess_env_inherit_credentials_keeps_user_password(monkeypatch):
    """Model-driver compatibility does not erase unrelated single-profile state."""
    from tools.environments.local import hermes_subprocess_env

    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.abc123.def456:xyz789")
    monkeypatch.setenv("DB_PASSWORD", "db-pass-9f2c1a")

    env = hermes_subprocess_env(inherit_credentials=True)
    assert "BWS_ACCESS_TOKEN" not in env
    assert env["DB_PASSWORD"] == "db-pass-9f2c1a"


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


def test_make_run_env_preserves_single_profile_password(monkeypatch):
    from agent.secret_scope import set_multiplex_active
    from tools.environments.local import _make_run_env

    set_multiplex_active(False)
    monkeypatch.setenv("DB_PASSWORD", "single-profile-password")
    result = _make_run_env({})
    assert result["DB_PASSWORD"] == "single-profile-password"


def test_make_run_env_strips_source_passwords_at_profile_boundary(
    tmp_path, monkeypatch
):
    """Password-shaped source values are denied only when authority crosses."""
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.environments.local import _make_run_env

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    password_values = {
        "DB_PASSWORD": "db-pass-9f2c1a",
        "REDIS_PASSWORD": "redis-pass-77aa",
        "PGPASSWORD": "pg-pass-e11",
        "MYSQL_PWD": "mysql-pwd-4d2",
        "PASSWORD": "bare-pass-8c1",
    }
    (source / ".env").write_text(
        "".join(f"{key}={value}\n" for key, value in password_values.items()),
        encoding="utf-8",
    )
    (target / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(source))
    for key, value in password_values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MY_HARMLESS_VAR", "keep-me")
    monkeypatch.setenv("PWD", "C:\\some\\cwd")

    set_multiplex_active(True)
    token = set_hermes_home_override(target)
    try:
        env = _make_run_env({})
    finally:
        reset_hermes_home_override(token)
        set_multiplex_active(False)

    for key in password_values:
        assert key not in env
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
def test_local_environment_e2e_profile_password_denial_and_passthrough(
    tmp_path, monkeypatch
):
    """Real A→B execution denies source password and permits target passthrough."""
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
    from tools.environments.local import LocalEnvironment

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / ".env").write_text("DB_PASSWORD=source-password\n", encoding="utf-8")
    (target / ".env").write_text("DB_PASSWORD=target-password\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv("DB_PASSWORD", "source-password")

    set_multiplex_active(True)
    token = set_hermes_home_override(target)
    secret_token = set_secret_scope({"DB_PASSWORD": "target-password"})
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
                'if [ "$DB_PASSWORD" = "target-password" ]; then echo "PASSTHROUGH"; else echo "MISSING"; fi'
            )
            assert allowed["returncode"] == 0
            assert "PASSTHROUGH" in allowed.get("output", "")
        finally:
            clear_env_passthrough()
    finally:
        env.cleanup()
        reset_secret_scope(secret_token)
        reset_hermes_home_override(token)
        set_multiplex_active(False)


def test_bws_token_env_resolution_tracks_profile_and_config_revision(tmp_path):
    """Bitwarden token-name resolution follows active profile and config edits."""

    import tools.environments.local as _local_mod

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    (profile_a / "config.yaml").write_text(
        json.dumps({"secrets": {"bitwarden": {"access_token_env": "ALPHA_BWS_TOKEN"}}}),
        encoding="utf-8",
    )
    (profile_b / "config.yaml").write_text(
        json.dumps({"secrets": {"bitwarden": {"access_token_env": "BETA_BWS_TOKEN"}}}),
        encoding="utf-8",
    )

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token_a = set_hermes_home_override(str(profile_a))
    try:
        assert _local_mod._get_configured_bws_token_env() == "ALPHA_BWS_TOKEN"
        assert _local_mod._is_hermes_internal_secret("ALPHA_BWS_TOKEN") is True
        assert _local_mod._is_hermes_internal_secret("BETA_BWS_TOKEN") is True
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(str(profile_b))
    try:
        assert _local_mod._get_configured_bws_token_env() == "BETA_BWS_TOKEN"
        assert _local_mod._is_hermes_internal_secret("BETA_BWS_TOKEN") is True
        assert _local_mod._is_hermes_internal_secret("ALPHA_BWS_TOKEN") is True
        # The default name stays internal even in a remapped profile: the
        # process-global os.environ can carry a default profile's token into
        # this profile's turn, and it must not cross the child boundary.
        assert _local_mod._is_hermes_internal_secret("BWS_ACCESS_TOKEN") is True
        # Third-party tokens remain registerable in every profile.
        assert _local_mod._is_hermes_internal_secret("STRIPE_ACCESS_TOKEN") is False
    finally:
        reset_hermes_home_override(token_b)

    # Back on profile A, resolution returns A's own name and then follows a
    # live config revision instead of pinning a shadow cache entry forever.
    token_a2 = set_hermes_home_override(str(profile_a))
    try:
        assert _local_mod._get_configured_bws_token_env() == "ALPHA_BWS_TOKEN"
        refreshed_name = "ALPHA_BWS_TOKEN_REFRESHED"
        (profile_a / "config.yaml").write_text(
            json.dumps(
                {"secrets": {"bitwarden": {"access_token_env": refreshed_name}}}
            ),
            encoding="utf-8",
        )
        assert _local_mod._get_configured_bws_token_env() == refreshed_name
        assert _local_mod._is_hermes_internal_secret(refreshed_name) is True
    finally:
        reset_hermes_home_override(token_a2)


def test_bws_token_env_remap_non_suffix_stripped(tmp_path, monkeypatch):
    """A Bitwarden access_token_env remapped to a non-suffix name (e.g.
    MY_BWS_TOKEN) is stripped exactly, while third-party *_ACCESS_TOKEN vars
    (e.g. STRIPE_ACCESS_TOKEN) stay passthrough-able — the fix for the
    over-broad *_ACCESS_TOKEN suffix match."""

    import tools.environments.local as _local_mod

    config = {"secrets": {"bitwarden": {"access_token_env": "MY_BWS_TOKEN"}}}
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

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


@pytest.mark.parametrize("token_name", ["MY_BWS_TOKEN", "VAULT_BOOTSTRAP_AUTH"])
def test_bws_bootstrap_policy_failure_refuses_child(monkeypatch, token_name):
    import hermes_cli.config as config_mod

    def _raise():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(config_mod, "read_raw_config", _raise)
    monkeypatch.setenv(token_name, "fake-bootstrap")

    with pytest.raises(RuntimeError, match="Bitwarden token policy unavailable"):
        build_subprocess_env()


def test_plugin_strip_registry_failure_refuses_child(monkeypatch):
    import agent.terminal_env_registry as registry

    def _raise():
        raise RuntimeError("plugin registry unavailable")

    monkeypatch.setattr(registry, "plugin_strip_env_keys", _raise)
    with pytest.raises(
        RuntimeError, match="plugin terminal environment policy unavailable"
    ):
        build_subprocess_env()
