"""Target authority must not depend on the launch environment's credential names."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent import secret_scope
from hermes_constants import (
    get_hermes_home_override, get_real_home, reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
from tools.environments.local import LocalEnvironment, _make_run_env, build_subprocess_env


def _profile(home, dotenv, config="{}\n"):
    home.mkdir()
    (home / ".env").write_text(dotenv, encoding="utf-8")
    (home / "config.yaml").write_text(config, encoding="utf-8")


def _child(env, names):
    result = subprocess.run(
        [sys.executable, "-c",
         "import json,os,sys; print(json.dumps({k:os.getenv(k) for k in sys.argv[1:]}))", *names],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding="utf-8", check=True, timeout=10,
    )
    return json.loads(result.stdout)


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    _profile(source, "SOURCE_ONLY=alpha\nSHARED_LOGIN=alpha-shared\n",
             "terminal:\n  env_passthrough: [UNPERMITTED]\n")
    _profile(target, "")
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    clear_env_passthrough()
    home_token = set_hermes_home_override(source)
    scope_token = secret_scope.set_secret_scope(None)
    try:
        yield source, target
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        clear_env_passthrough()


@pytest.mark.parametrize("surface", ["explicit-source-context", "explicit-target-context", "foreground"])
@pytest.mark.parametrize("permitted", [False, True])
@pytest.mark.parametrize("value", ["beta", ""])
@pytest.mark.parametrize("seed", ["absent", "direct", "transport", "force"])
def test_target_only_projection_obeys_target_grant(profiles, monkeypatch, surface, permitted, value, seed):
    source, target = profiles
    denied = ["OPENAI_API_KEY", "AUXILIARY_TEST_API_KEY", "PLUGIN_TEST_CREDENTIAL"]
    (target / ".env").write_text(
        f"TARGET_ONLY={value}\nUNPERMITTED=target-private\nSHARED_LOGIN=beta-shared\n"
        + "".join(f"{key}=target-private\n" for key in denied), encoding="utf-8")
    grants = [*denied, *(["TARGET_ONLY"] if permitted else [])]
    (target / "config.yaml").write_text(
        "terminal:\n  env_passthrough: " + json.dumps(grants) + "\n", encoding="utf-8")
    monkeypatch.setattr("tools.environments.local._plugin_terminal_env_strip_keys",
                        lambda: frozenset({"PLUGIN_TEST_CREDENTIAL"}))
    aliases = ["APPTAINERENV_TARGET_ONLY", "SINGULARITYENV_APPTAINERENV_TARGET_ONLY"]
    base = {"SOURCE_ONLY": "alpha", "SHARED_LOGIN": "alpha-shared", "ORDINARY_SETTING": "keep"}
    extra = {}
    if seed == "direct":
        base.update(TARGET_ONLY="stale-ambient", UNPERMITTED="stale-private")
    elif seed == "transport":
        base.update({name: "stale-ambient" for name in aliases})
    elif seed == "force":
        extra["_HERMES_FORCE_TARGET_ONLY"] = "stale-override"
    names = ["TARGET_ONLY", "UNPERMITTED", "SOURCE_ONLY", "SHARED_LOGIN", "ORDINARY_SETTING", *aliases, *denied]
    for name in names:
        monkeypatch.delenv(name, raising=False)
    token = set_hermes_home_override(source if surface == "explicit-source-context" else target)
    try:
        if surface == "foreground":
            for name, content in base.items():
                monkeypatch.setenv(name, content)
            env = _make_run_env(extra)
        else:
            env = build_subprocess_env(base=base, extra=extra, profile_home=target,
                                       source_profile_home=source, enforce_profile_boundary=True)
        assert get_hermes_home_override() == str(source if surface == "explicit-source-context" else target)
    finally:
        reset_hermes_home_override(token)
    expected = dict.fromkeys(names)
    expected.update(TARGET_ONLY=value if permitted else None, SHARED_LOGIN="beta-shared", ORDINARY_SETTING="keep")
    assert _child(env, names) == expected


@pytest.mark.parametrize("multiplex", [False, True])
@pytest.mark.parametrize("value", ["beta", "", None])
def test_forwarded_shared_value_uses_explicit_boundary_without_secret_scope(profiles, monkeypatch, multiplex, value):
    source, target = profiles
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", multiplex)
    (target / ".env").write_text("" if value is None else f"SHARED_LOGIN={value}\n", encoding="utf-8")
    register_env_passthrough(["SHARED_LOGIN"])
    env = build_subprocess_env(base={"SHARED_LOGIN": "alpha-shared"}, profile_home=target,
                               source_profile_home=source, enforce_profile_boundary=True)
    assert _child(env, ["SHARED_LOGIN"]) == {"SHARED_LOGIN": value}
    assert secret_scope.current_secret_scope() is None
    if multiplex:
        with pytest.raises(secret_scope.UnscopedSecretError):
            secret_scope.get_secret("SHARED_LOGIN")


def test_forwarded_target_only_value_does_not_survive_shared_snapshot(profiles, tmp_path):
    source, target = profiles
    other = tmp_path / "other"
    _profile(other, "")
    (target / ".env").write_text("TARGET_ONLY=beta\n", encoding="utf-8")
    (target / "config.yaml").write_text("terminal:\n  env_passthrough: [TARGET_ONLY]\n", encoding="utf-8")
    environment = LocalEnvironment(cwd=str(tmp_path))
    try:
        environment.init_session()
        assert environment._snapshot_ready
        for home, expected in [(target, "VALUE:beta"), (other, "VALUE:UNSET")]:
            token = set_hermes_home_override(home)
            try:
                result = environment.execute('printf "VALUE:%s" "${TARGET_ONLY-UNSET}"')
            finally:
                reset_hermes_home_override(token)
            assert result["returncode"] == 0
            assert expected in result["output"]
    finally:
        environment.cleanup()


@pytest.mark.parametrize("context", ["source", "target"])
@pytest.mark.parametrize("mode", ["real", "profile"])
def test_explicit_worker_identity_and_context_are_independent(profiles, monkeypatch, context, mode):
    source, target = profiles
    (source / "home").mkdir()
    (target / "home").mkdir()
    monkeypatch.setenv("HOME", str(source / "home"))
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    monkeypatch.setenv("TERMINAL_HOME_MODE", mode)
    real_home = get_real_home()
    caller = source if context == "source" else target
    token = set_hermes_home_override(caller)
    try:
        env = build_subprocess_env(base=dict(os.environ), profile_home=target,
                                   source_profile_home=source, enforce_profile_boundary=True)
        assert get_hermes_home_override() == str(caller)
    finally:
        reset_hermes_home_override(token)
    assert _child(env, ["HERMES_HOME", "HOME", "HERMES_REAL_HOME"]) == {
        "HERMES_HOME": str(target), "HOME": str(target / "home") if mode == "profile" else real_home,
        "HERMES_REAL_HOME": real_home,
    }


def test_config_revocation_removes_target_only_forwarding(profiles, monkeypatch):
    from tools.env_passthrough import is_env_passthrough

    source, target = profiles
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    (target / ".env").write_text("TARGET_ONLY=beta\n", encoding="utf-8")
    config = target / "config.yaml"
    config.write_text("terminal:\n  env_passthrough: [TARGET_ONLY]\n", encoding="utf-8")
    token = set_hermes_home_override(target)
    try:
        assert is_env_passthrough("TARGET_ONLY")
        before = build_subprocess_env(base={"TARGET_ONLY": "beta"}, profile_home=target,
                                      source_profile_home=source, enforce_profile_boundary=True)
        assert _child(before, ["TARGET_ONLY"]) == {"TARGET_ONLY": "beta"}
        config.write_text("{}\n", encoding="utf-8")
        granted = is_env_passthrough("TARGET_ONLY")
        after = build_subprocess_env(base={}, profile_home=target,
                                     source_profile_home=source, enforce_profile_boundary=True)
        assert (granted, _child(after, ["TARGET_ONLY"])) == (False, {"TARGET_ONLY": None})
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("name", [
    "APPTAINERENV_OPENAI_API_KEY", "SINGULARITYENV_APPTAINERENV_AUXILIARY_TEST_API_KEY",
    "APPTAINERENV_PLUGIN_TEST_CREDENTIAL", "_HERMES_FORCE_OPENAI_API_KEY",
])
def test_target_forwarding_cannot_regrant_denied_carriers(profiles, monkeypatch, name):
    source, target = profiles
    (target / ".env").write_text(f"{name}=target-private\n", encoding="utf-8")
    (target / "config.yaml").write_text(f"terminal:\n  env_passthrough: [{name}]\n", encoding="utf-8")
    monkeypatch.setattr("tools.environments.local._plugin_terminal_env_strip_keys",
                        lambda: frozenset({"PLUGIN_TEST_CREDENTIAL"}))
    env = build_subprocess_env(base={}, profile_home=target,
                               source_profile_home=source, enforce_profile_boundary=True)
    names = [name, "OPENAI_API_KEY", "AUXILIARY_TEST_API_KEY", "PLUGIN_TEST_CREDENTIAL"]
    assert _child(env, names) == dict.fromkeys(names)


@pytest.mark.parametrize("declaration", ["profile-force", "plugin-force"])
def test_force_carrier_cannot_promote_profile_or_plugin_authority(profiles, monkeypatch, declaration):
    source, target = profiles
    extra = {}
    if declaration == "profile-force":
        name = "_HERMES_FORCE_OPENAI_API_KEY"
        (source / ".env").write_text(f"{name}=source-private\n", encoding="utf-8")
        (target / ".env").write_text(f"{name}=target-private\n", encoding="utf-8")
    else:
        extra["_HERMES_FORCE_PLUGIN_TEST_CREDENTIAL"] = "caller-private"
    monkeypatch.setattr("tools.environments.local._plugin_terminal_env_strip_keys",
                        lambda: frozenset({"PLUGIN_TEST_CREDENTIAL"}))
    env = build_subprocess_env(base={}, extra=extra, profile_home=target,
                               source_profile_home=source, enforce_profile_boundary=True)
    names = ["OPENAI_API_KEY", "PLUGIN_TEST_CREDENTIAL"]
    assert _child(env, names) == dict.fromkeys(names)


def test_wrapped_source_ownership_excludes_raw_snapshot_value(profiles, monkeypatch, tmp_path):
    source, target = profiles
    (source / ".env").write_text("APPTAINERENV_SOURCE_ONLY=alpha\n", encoding="utf-8")
    monkeypatch.setenv("SOURCE_ONLY", "alpha")
    environment = LocalEnvironment(cwd=str(tmp_path))
    try:
        assert environment._snapshot_ready
        token = set_hermes_home_override(target)
        try:
            result = environment.execute('printf "VALUE:%s" "${SOURCE_ONLY-UNSET}"')
        finally:
            reset_hermes_home_override(token)
        assert result["returncode"] == 0
        assert "VALUE:UNSET" in result["output"]
    finally:
        environment.cleanup()
