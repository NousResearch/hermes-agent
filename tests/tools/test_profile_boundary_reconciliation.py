"""Real-child witnesses for profile provenance and terminal policy composition."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent.secret_scope import build_profile_env_boundary, build_profile_secret_scope

import pytest

from agent.secret_scope import set_multiplex_active, is_multiplex_active
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from tools.environments.local import _make_run_env, _sanitize_subprocess_env


@pytest.fixture
def routed_profiles(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(source))
    was_active = is_multiplex_active()
    set_multiplex_active(True)
    token = set_hermes_home_override(target)
    try:
        yield source, target
    finally:
        reset_hermes_home_override(token)
        set_multiplex_active(was_active)


def _child_values(env, names):
    child = subprocess.run(
        [sys.executable, "-c", "import json,os,sys; print(json.dumps({k:os.getenv(k) for k in sys.argv[1:]}))", *names],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding="utf-8", timeout=10, check=True,
    )
    return json.loads(child.stdout)


@pytest.mark.parametrize("surface", ["foreground", "background"])
def test_child_replaces_profile_owned_carriers(routed_profiles, monkeypatch, surface):
    source, target = routed_profiles
    (source / ".env").write_text("ACME_LOGIN=source-only\nDATABASE_URL=source-db\n", encoding="utf-8")
    (target / ".env").write_text("DATABASE_URL=target-db\n", encoding="utf-8")
    carriers = {
        "ACME_LOGIN": "source-only",
        "APPTAINERENV_ACME_LOGIN": "source-only",
        "SINGULARITYENV_APPTAINERENV_ACME_LOGIN": "source-only",
        "DATABASE_URL": "source-db",
        "APPTAINERENV_DATABASE_URL": "source-db",
        "ORDINARY_SHELL_SETTING": "keep",
    }
    for name, value in carriers.items():
        monkeypatch.setenv(name, value)
    extras = {"_HERMES_FORCE_ACME_LOGIN": "forced-source"}
    env = _make_run_env(extras) if surface == "foreground" else _sanitize_subprocess_env(dict(os.environ), extras)
    values = _child_values(env, carriers)
    assert values == {
        "ACME_LOGIN": None,
        "APPTAINERENV_ACME_LOGIN": None,
        "SINGULARITYENV_APPTAINERENV_ACME_LOGIN": None,
        "DATABASE_URL": "target-db",
        "APPTAINERENV_DATABASE_URL": None,
        "ORDINARY_SHELL_SETTING": "keep",
    }


@pytest.mark.parametrize("surface", ["foreground", "background"])
def test_target_overlay_does_not_regrant_filtered_credentials(routed_profiles, monkeypatch, surface):
    source, target = routed_profiles
    names = ("OPENAI_API_KEY", "AUXILIARY_TEST_API_KEY", "PLUGIN_TEST_CREDENTIAL")
    (source / ".env").write_text("".join(f"{name}=source-secret\n" for name in names), encoding="utf-8")
    (target / ".env").write_text("".join(f"{name}=target-secret\n" for name in names), encoding="utf-8")
    monkeypatch.setattr("tools.environments.local._plugin_terminal_env_strip_keys", lambda: frozenset({names[2]}))
    for name in names:
        monkeypatch.setenv(name, "source-secret")
    env = _make_run_env({}) if surface == "foreground" else _sanitize_subprocess_env(dict(os.environ))
    assert _child_values(env, names) == dict.fromkeys(names)


@pytest.mark.parametrize("owner", ["source", "target"])
def test_unreadable_dotenv_refuses_boundary_but_not_normal_scope(tmp_path, owner):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    # A directory in place of a dotenv is an actual read error, not a missing
    # optional file. Normal scope compatibility must not weaken child boundaries.
    (tmp_path / owner / ".env").mkdir()
    assert build_profile_secret_scope(tmp_path / owner) == {}
    with pytest.raises(RuntimeError, match="dotenv"):
        build_profile_env_boundary(source, target)


@pytest.mark.parametrize("spelling", ["ACME_LOGIN", "APPTAINERENV_ACME_LOGIN"])
def test_dotenv_reload_cannot_forget_inherited_source_ownership(tmp_path, monkeypatch, spelling):
    from hermes_cli.env_loader import load_hermes_dotenv
    from tools.environments.local import build_subprocess_env

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(source))
    # Restore process env even though production dotenv ingestion writes it.
    monkeypatch.setenv(spelling, "before")
    (source / ".env").write_text(f"{spelling}=source-secret\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=source, load_external_secrets=False)
    (source / ".env").write_text("", encoding="utf-8")
    load_hermes_dotenv(hermes_home=source, load_external_secrets=False)
    assert os.environ[spelling] == "source-secret"  # arbitrary names survive reload
    env = build_subprocess_env(
        base={spelling: os.environ[spelling], "ACME_LOGIN": "unwrapped-source"},
        profile_home=target, source_profile_home=source, enforce_profile_boundary=True)
    assert _child_values(env, [spelling, "ACME_LOGIN"]) == dict.fromkeys([spelling, "ACME_LOGIN"])


@pytest.mark.parametrize("mode", ["real", "profile"])
@pytest.mark.parametrize("context_home", ["source", "target"])
def test_explicit_worker_home_agrees_with_target_not_context(tmp_path, monkeypatch, mode, context_home):
    from hermes_constants import get_real_home
    from tools.environments.local import build_subprocess_env

    source, target = tmp_path / "source", tmp_path / "target"
    (source / "home").mkdir(parents=True)
    (target / "home").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv("HOME", str(source / "home"))
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    monkeypatch.setenv("TERMINAL_HOME_MODE", mode)
    token = set_hermes_home_override(source)
    try:
        real_home = get_real_home()
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(tmp_path / context_home)
    try:
        env = build_subprocess_env(
            base=dict(os.environ), profile_home=target, source_profile_home=source,
            enforce_profile_boundary=True)
        from hermes_constants import get_hermes_home_override
        assert get_hermes_home_override() == str(tmp_path / context_home)
    finally:
        reset_hermes_home_override(token)
    assert env["HERMES_HOME"] == str(target)
    assert env["HERMES_REAL_HOME"] == real_home
    assert env["HOME"] == (str(target / "home") if mode == "profile" else real_home)
