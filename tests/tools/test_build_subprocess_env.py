"""Real children observe factory scrubbing, overrides and routed profile homes."""

import os

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
