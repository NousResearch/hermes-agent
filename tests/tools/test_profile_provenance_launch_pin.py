"""Pinned launch identity must remain the source of child-env provenance."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


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
