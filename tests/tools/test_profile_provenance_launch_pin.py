"""Pinned launch identity must remain the source of child-env provenance."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tools._child_env_fixtures import child_env, observe_terminal  # noqa: F401


@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("multiplex", [False, True])
def test_shared_snapshot_excludes_routed_ownership_after_revocation(child_env, monkeypatch, multiplex):
    from agent import secret_scope as ss
    from hermes_constants import (
        get_routing_process_hermes_home, pin_process_hermes_home,
        process_hermes_home_is_pinned, reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools.environments.local import LocalEnvironment

    launch, target = child_env / "launch", child_env / "target"
    launch.mkdir()
    target.mkdir()
    names = ["ACME_LOGIN", "APPTAINERENV_ACME_LOGIN"]
    (launch / ".env").write_text(
        "APPTAINERENV_ACME_LOGIN=launch-private\n", encoding="utf-8")
    (target / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    for name in names:
        monkeypatch.setenv(name, "launch-private")
    prior_pin = get_routing_process_hermes_home() if process_hermes_home_is_pinned() else None
    prior_mode = ss.is_multiplex_active()
    pin_process_hermes_home(launch)
    ss.set_multiplex_active(multiplex)
    env = None
    try:
        env = LocalEnvironment(cwd=str(child_env), timeout=30)
        assert observe_terminal(env, names) == dict.fromkeys(names, "launch-private")
        home_token = set_hermes_home_override(target)
        secret_token = ss.set_secret_scope(ss.build_profile_secret_scope(target))
        try:
            assert observe_terminal(env, names) == dict.fromkeys(names)
            snapshot = Path(env._snapshot_path)
            assert all(name not in snapshot.read_text(encoding="utf-8") for name in names)
            # Removing the source declaration must not release its old snapshot values.
            (launch / ".env").write_text("", encoding="utf-8")
            assert observe_terminal(env, names) == dict.fromkeys(names)
        finally:
            ss.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
        assert observe_terminal(env, names) == dict.fromkeys(names, "launch-private")
        assert all(name not in snapshot.read_text(encoding="utf-8") for name in names)
    finally:
        if env is not None:
            env.cleanup()
        ss.set_multiplex_active(prior_mode)
        pin_process_hermes_home(prior_pin)


@pytest.mark.parametrize("multiplex", [False, True])
def test_mirrored_launch_home_cannot_reclassify_source_credentials(tmp_path, monkeypatch, multiplex):
    from agent.secret_scope import (
        build_profile_secret_scope, is_multiplex_active, reset_secret_scope,
        set_multiplex_active, set_secret_scope,
    )
    from hermes_constants import (
        get_routing_process_hermes_home, pin_process_hermes_home,
        process_hermes_home_is_pinned, reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools.environments.local import build_subprocess_env

    launch, target = tmp_path / "launch", tmp_path / "target"
    launch.mkdir()
    target.mkdir()
    (launch / ".env").write_text("ACME_LOGIN=launch-private\nSHARED_LOGIN=launch-shared\n", encoding="utf-8")
    (target / ".env").write_text("SHARED_LOGIN=target-shared\n", encoding="utf-8")
    prior_pin = get_routing_process_hermes_home() if process_hermes_home_is_pinned() else None
    prior_mode = is_multiplex_active()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    pin_process_hermes_home(launch)
    set_multiplex_active(multiplex)
    home_token = set_hermes_home_override(target)
    scope_token = set_secret_scope(build_profile_secret_scope(target), profile_home=str(target))
    # Embedding hosts may mirror the active home. This does not transfer launch ownership.
    monkeypatch.setenv("HERMES_HOME", str(target))
    names = ["ACME_LOGIN", "APPTAINERENV_ACME_LOGIN", "SHARED_LOGIN"]
    try:
        env = build_subprocess_env(
            base={**os.environ, "ACME_LOGIN": "launch-private", "APPTAINERENV_ACME_LOGIN": "launch-private", "SHARED_LOGIN": "launch-shared"},
            profile_home=target, enforce_profile_boundary=True,
        )
        child = subprocess.run(
            [sys.executable, "-c", "import json,os,sys; print(json.dumps({k:os.getenv(k) for k in sys.argv[1:]}))", *names],
            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", timeout=10, check=True,
        )
        assert json.loads(child.stdout) == {"ACME_LOGIN": None, "APPTAINERENV_ACME_LOGIN": None, "SHARED_LOGIN": "target-shared"}
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(prior_mode)
        pin_process_hermes_home(prior_pin)


def test_foreground_force_authority_is_only_per_call(monkeypatch):
    from tools.environments.local import _make_run_env
    monkeypatch.setenv("_HERMES_FORCE_OPENAI_API_KEY", "ambient-not-authorized")
    inherited = _make_run_env({})
    explicit = _make_run_env({"_HERMES_FORCE_OPENAI_API_KEY": "explicitly-authorized"})
    assert "OPENAI_API_KEY" not in inherited
    assert "_HERMES_FORCE_OPENAI_API_KEY" not in inherited
    assert explicit["OPENAI_API_KEY"] == "explicitly-authorized"
