"""Real children observe factory scrubbing, overrides and routed profile homes."""

import os
import subprocess
import sys

import pytest

from tests.tools._child_env_fixtures import child_env, observe_child
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
        result = observe_child(env, ["HERMES_HOME", "HERMES_PROFILE", "ANTHROPIC_API_KEY",
                                     "AUXILIARY_FAKE_API_KEY", "GATEWAY_RELAY_FOO_TOKEN",
                                     "MY_HARMLESS_VAR"])
    finally:
        reset_hermes_home_override(token)
    # The bound turn identity is published as a name+home PAIR on BOTH factory branches: a child
    # always receives the home of the profile it acts for, so the pin it carries cannot be
    # contradicted by the sticky re-home a bare CLI child performs (_export_served_profile_env).
    assert result == {
        "HERMES_HOME": str(routed), "HERMES_PROFILE": "coder",
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


def test_session_profile_is_exported_as_hermes_profile(tmp_path, monkeypatch):
    """A gateway-served session exports no HERMES_PROFILE: the served profile lives in
    the session ContextVar (the gateway's own HERMES_HOME is the DEFAULT root), so a child
    running ``hermes kanban comment``/``hermes peer dm`` could not name its profile. The pin
    is published WITH the served profile's home, so the name cannot be contradicted by the
    home the child ends up acting under."""
    from gateway.session_context import (
        clear_session_vars, reset_session_vars, set_session_vars)

    root = tmp_path / ".hermes"
    (root / "profiles" / "ops-coder").mkdir(parents=True)
    _clear_identity_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(root))
    tokens = set_session_vars(profile="ops-coder")
    try:
        env = build_subprocess_env()
    finally:
        clear_session_vars(tokens)
        reset_session_vars()  # leave the ContextVars _UNSET for later tests
    assert env["HERMES_PROFILE"] == "ops-coder"
    assert env["HERMES_HOME"] == str(root / "profiles" / "ops-coder")


def test_default_identity_publishes_no_pin(tmp_path, monkeypatch):
    """An identity with no profile-scoped home to pin (the default root) publishes NOTHING:
    its only home IS the launch root, so a pin would be free to be contradicted by the child's
    sticky re-home while attributing its work to a profile it does not act under."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / ".hermes"
    root.mkdir(parents=True)
    _clear_identity_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(root))

    token = set_hermes_home_override(str(root))
    try:
        env = build_subprocess_env()
    finally:
        reset_hermes_home_override(token)
    assert not env.get("HERMES_PROFILE")


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


def test_e2e_child_cli_names_and_homes_the_served_session_profile(tmp_path, monkeypatch):
    """Cross-surface: a REAL child booted through the CLI entry resolves the CLI author AND the
    home it acts under for the gateway-served session — even while the host's sticky
    ``active_profile`` names a DIFFERENT profile, which is exactly the state that re-homes a bare
    CLI child (``hermes_cli.main`` runs ``_apply_profile_override()`` at import). Booting through
    the real entry is the point: a ``python -c`` child importing ``hermes_cli.kanban`` directly
    never runs that step, so it certifies a child that cannot occur."""
    from gateway.session_context import (
        clear_session_vars, reset_session_vars, set_session_vars)

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _clear_identity_env(monkeypatch)
    root = tmp_path / ".hermes"
    served, sticky = root / "profiles" / "ops-coder", root / "profiles" / "worker_beta"
    for home in (served, sticky):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(f"name: {home.name}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "active_profile").write_text("worker_beta", encoding="utf-8")

    tokens = set_session_vars(profile="ops-coder")
    try:
        env = build_subprocess_env()
    finally:
        clear_session_vars(tokens)
        reset_session_vars()
    assert env["HERMES_PROFILE"] == "ops-coder"  # what the child inherits
    assert env["HERMES_HOME"] == str(served)

    env["HOME"] = str(tmp_path)  # the child resolves <tmp>/.hermes as its own default root
    # The venv's editable install maps ``hermes_cli`` to its own tree; point the child at the
    # tree under test so the assertion is about THIS checkout's CLI entry point.
    from pathlib import Path
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    code = (
        "import hermes_cli.main, os;"  # module-level _apply_profile_override() = the re-home step
        "from hermes_cli.kanban import _profile_author;"
        "from hermes_cli.profiles import current_profile_name;"
        "print(_profile_author(), current_profile_name('user'), os.environ['HERMES_HOME'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=180, check=True,
    )
    assert out.stdout.strip().split() == ["ops-coder", "ops-coder", str(served)]


def test_no_scrub_branch_exports_the_same_identity(tmp_path, monkeypatch):
    """The factory publishes the acting identity on BOTH branches. ``scrub_secrets=False`` is the
    one a dozen production sites use (``hermes_cli/pty_bridge.py``, the secret managers): a child
    spawned there used to keep naming the launch default. An explicit caller override still wins."""
    from gateway.session_context import (
        clear_session_vars, reset_session_vars, set_session_vars)

    root = tmp_path / ".hermes"
    (root / "profiles" / "ops-coder").mkdir(parents=True)
    _clear_identity_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(root))

    tokens = set_session_vars(profile="ops-coder")
    try:
        env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
        explicit = build_subprocess_env(
            scrub_secrets=False, inherit_profile_home=False, extra={"HERMES_PROFILE": "caller"})
    finally:
        clear_session_vars(tokens)
        reset_session_vars()

    assert env["HERMES_PROFILE"] == "ops-coder"
    assert env["HERMES_HOME"] == str(root / "profiles" / "ops-coder")
    assert explicit["HERMES_PROFILE"] == "caller"  # extra is applied last


def test_align_pin_fails_closed_when_the_target_owner_is_unresolvable(monkeypatch):
    """The explicit-home guard must DROP the pin when it cannot prove the target home owns it.
    Failing open would leave a served session's pin outranking the home the child acts under —
    a process reading one profile's configuration while claiming another's identity."""
    import hermes_constants
    from tools.environments.local import _align_pin_with_target_home

    def _unresolvable(*_args, **_kwargs):
        raise RuntimeError("home resolution unavailable")

    monkeypatch.setattr(hermes_constants, "profile_name_for_home", _unresolvable)
    env = {"HERMES_PROFILE": "ops-coder", "HERMES_HOME": "/somewhere"}
    _align_pin_with_target_home(env, "/other/home")
    assert "HERMES_PROFILE" not in env


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
