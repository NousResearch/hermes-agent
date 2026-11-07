"""A routed home's declared passthrough names survive ``strip_launch_profile_env``.

The strip removes every name the LAUNCH profile's external secret sources supplied. Under the
gateway/multiplex path the caller overlays the served profile's bound secret scope, which puts
back exactly the names that profile's own sources supply. A single-process worker acting for a
routed home (desktop backend, ``key_cmd`` helper, a worker spawned with an explicit target home)
has no bound scope, so the restore used to be a no-op: a ``terminal.env_passthrough`` name that
lived only in the target's vault vanished from the target's terminal children while the same name
reached them under a gateway-served turn — a false negative for credential verification.
"""

from pathlib import Path

import pytest

import tools.env_passthrough as ep
from hermes_constants import (
    pin_process_hermes_home, reset_hermes_home_override, set_hermes_home_override)
from hermes_cli import env_loader


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    target = root / "profiles" / "alpha"
    target.mkdir(parents=True)
    (root / ".env").write_text("LAUNCH_ONLY_SETTING=1\n")
    (target / ".env").write_text("ROUTED_TEST_MAIL=target-mail\n")
    (target / "config.yaml").write_text(
        "terminal:\n  env_passthrough:\n    - ROUTED_TEST_URL\n    - ROUTED_TEST_MAIL\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    pin_process_hermes_home(str(root))
    ep._config_passthrough.clear()
    # The launch process loaded both routed names from an external source; alpha's own source
    # supplies only the URL. FORGEJO_API_KEY is launch-only and undeclared by alpha.
    monkeypatch.setattr(env_loader, "source_supplied_names",
                        lambda: ("ROUTED_TEST_URL", "ROUTED_TEST_MAIL", "FORGEJO_API_KEY"))
    monkeypatch.setattr(env_loader, "launch_dotenv_keys", lambda: frozenset())
    monkeypatch.setattr(env_loader, "managed_dotenv_keys", lambda: frozenset())
    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", lambda _home: {})

    def _source_values(home):
        return ({"ROUTED_TEST_URL": "target-vault-url"}
                if Path(home).resolve() == target.resolve() else {})

    monkeypatch.setattr(env_loader, "get_secret_source_values", _source_values)
    yield root, target
    pin_process_hermes_home(None)
    ep._config_passthrough.clear()


def _launch_env() -> dict:
    return {
        "PATH": "/usr/bin:/bin",
        "HERMES_HOME": "/somewhere/launch",
        "ROUTED_TEST_URL": "launch-vault-url",
        "ROUTED_TEST_MAIL": "launch-mail",
        "FORGEJO_API_KEY": "launch-forgejo",
    }


def test_routed_child_gets_the_target_profiles_own_passthrough_values(homes):
    from tools.environments.local import strip_launch_profile_env

    _root, target = homes
    env = strip_launch_profile_env(_launch_env(), target_home=target)

    assert env["ROUTED_TEST_URL"] == "target-vault-url"   # vault-only: restored from alpha
    assert env["ROUTED_TEST_MAIL"] == "target-mail"       # .env-defined: alpha's value wins
    assert "FORGEJO_API_KEY" not in env                   # alpha never supplies it
    assert env["PATH"] == "/usr/bin:/bin"


def test_restore_works_without_a_home_override(homes):
    """``served_profile_child_env`` passes ``target_home`` explicitly; the target's own
    ``terminal.env_passthrough`` must still be the list that is honored."""
    from tools.environments.local import strip_launch_profile_env

    _root, target = homes
    token = set_hermes_home_override(None)
    try:
        env = strip_launch_profile_env(_launch_env(), target_home=target)
    finally:
        reset_hermes_home_override(token)

    assert env["ROUTED_TEST_URL"] == "target-vault-url"
    assert env["ROUTED_TEST_MAIL"] == "target-mail"


def test_unscoped_multiplex_task_stays_fail_closed(homes, monkeypatch):
    """Under multiplex semantics no scope means no restore: the profile's files must not be
    read behind an unscoped task's back."""
    from tools.environments.local import strip_launch_profile_env

    _root, target = homes
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    env = strip_launch_profile_env(_launch_env(), target_home=target)

    assert "ROUTED_TEST_URL" not in env
    assert "ROUTED_TEST_MAIL" not in env
    assert "FORGEJO_API_KEY" not in env


def test_launch_profile_is_untouched(homes):
    from tools.environments.local import strip_launch_profile_env

    root, _target = homes
    env = strip_launch_profile_env(_launch_env(), target_home=root)

    assert env["ROUTED_TEST_URL"] == "launch-vault-url"
    assert env["ROUTED_TEST_MAIL"] == "launch-mail"
    assert env["FORGEJO_API_KEY"] == "launch-forgejo"
