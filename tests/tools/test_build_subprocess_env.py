"""Real children observe factory scrubbing, overrides and routed profile homes."""

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
        "from hermes_cli.profiles import current_profile_name; "
        "print(_profile_author(), current_profile_name('user'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=120, check=True,
    )
    assert out.stdout.split() == ["ops-coder", "ops-coder"]


def test_explicit_target_home_owns_the_identity(tmp_path, monkeypatch):
    """A child handed an EXPLICIT home must not carry the spawning session's identity pin.

    ``current_profile_name`` reads the env pin before the home and a child has no bound
    override, so an exported pin would label a process that acts under another profile
    (``hermes -p X``, a routed profile's gateway service, the host gateway on the default
    root). The pin is dropped, never invented — the child resolves its own home.
    """
    from gateway.session_context import (
        clear_session_vars, reset_session_vars, set_session_vars)
    from tools.environments.local import served_profile_child_env

    _clear_identity_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    foreign = tmp_path / ".hermes" / "profiles" / "platform-coder"
    foreign.mkdir(parents=True)
    own = tmp_path / ".hermes" / "profiles" / "ops-coder"
    own.mkdir(parents=True)

    tokens = set_session_vars(profile="ops-coder")
    try:
        other = served_profile_child_env(target_home=foreign)
        same = served_profile_child_env(target_home=own)
    finally:
        clear_session_vars(tokens)
        reset_session_vars()

    assert other["HERMES_HOME"] == str(foreign)
    assert not other.get("HERMES_PROFILE"), "a foreign home must not inherit the session pin"
    assert same["HERMES_HOME"] == str(own)
    assert same["HERMES_PROFILE"] == "ops-coder"
