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

    # Simulate the service-manager minimal PATH: hermes dir absent.
    minimal_path = os.pathsep.join(["/usr/bin", "/bin"])
    monkeypatch.setenv("PATH", minimal_path)
    assert shutil.which("hermes", path=minimal_path) is None

    env = build_subprocess_env(scrub_secrets=True)  # cron _run_job_script path

    resolved = shutil.which("hermes", path=env.get("PATH", ""))
    assert resolved is not None, (
        f"bare 'hermes' must resolve from the child PATH {env.get('PATH')!r}"
    )
    assert os.path.dirname(resolved) == bin_dir
    assert env["PATH"].split(os.pathsep)[0] == bin_dir
    # Idempotent: running the parent env through the factory again must not
    # duplicate the entry.
    env2 = build_subprocess_env(env, scrub_secrets=True)
    assert env2["PATH"].split(os.pathsep).count(bin_dir) == 1


# ---------------------------------------------------------------------------
# Regression: the SESSION's profile id reaches the child env as HERMES_PROFILE
# (a child had only the CLI/Kanban author fallback, which named the DEFAULT home)
# ---------------------------------------------------------------------------


def _clear_identity_env(monkeypatch):
    for name in ("HERMES_PROFILE", "HERMES_PROFILE_NAME", "HERMES_SESSION_PROFILE"):
        monkeypatch.delenv(name, raising=False)


def test_session_profile_is_exported_as_hermes_profile(monkeypatch):
    """A gateway-served session exports no HERMES_PROFILE: the served profile lives in
    the session ContextVar (the gateway's own HERMES_HOME is the DEFAULT root), so a child
    running ``hermes kanban comment``/``hermes peer dm`` could not name its profile."""
    from gateway.session_context import (
        clear_session_vars, reset_session_vars, set_session_vars)

    _clear_identity_env(monkeypatch)
    tokens = set_session_vars(profile="ops-coder")
    try:
        env = build_subprocess_env()
    finally:
        clear_session_vars(tokens)
        reset_session_vars()  # leave the ContextVars _UNSET for later tests
    assert env["HERMES_PROFILE"] == "ops-coder"


def test_profile_scoped_home_alone_still_names_the_profile(tmp_path, monkeypatch):
    """``hermes -p X <cmd>`` scopes only HERMES_HOME: no env export, no bound session.
    The child must still be able to name X instead of the home-derived fallback."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _clear_identity_env(monkeypatch)
    home = tmp_path / ".hermes" / "profiles" / "ops-coder"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_constants._default_hermes_root_memo", None)

    token = set_hermes_home_override(str(home))
    try:
        env = build_subprocess_env()
    finally:
        reset_hermes_home_override(token)
    assert env["HERMES_PROFILE"] == "ops-coder"


def test_dispatcher_profile_pin_is_preserved_without_a_session(monkeypatch, tmp_path):
    """The kanban dispatcher pins HERMES_PROFILE on the workers it spawns; a worker
    spawning a child with no session bound must keep that pin (no clobbering)."""
    _clear_identity_env(monkeypatch)
    monkeypatch.setenv("HERMES_PROFILE", "platform-coder")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    env = build_subprocess_env()
    assert env["HERMES_PROFILE"] == "platform-coder"


def test_e2e_child_cli_author_names_the_served_session_profile(tmp_path, monkeypatch):
    """Cross-surface: a real child spawned through the factory resolves the CLI author
    for the gateway-served session (``HERMES_HOME`` = the DEFAULT root, no env export)."""
    from gateway.session_context import (
        clear_session_vars, reset_session_vars, set_session_vars)

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _clear_identity_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    tokens = set_session_vars(profile="ops-coder")
    try:
        env = build_subprocess_env()
    finally:
        clear_session_vars(tokens)
        reset_session_vars()
    assert env["HERMES_PROFILE"] == "ops-coder"  # what the child inherits

    code = (
        "from hermes_cli.kanban import _profile_author; "
        "from hermes_cli.profiles import resolve_acting_profile_name; "
        "print(_profile_author(), resolve_acting_profile_name('user'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=120, check=True,
    )
    assert out.stdout.split() == ["ops-coder", "ops-coder"]
