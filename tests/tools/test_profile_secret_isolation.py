"""Regression tests for multiplex profile-owned environment isolation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from agent.secret_scope import (
    ProfileEnvBoundary,
    build_profile_env_boundary,
    build_profile_secret_scope,
    get_profile_owned_secret_names,
    set_multiplex_active,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.environments.base import BaseEnvironment
from tools.environments.local import (
    LocalEnvironment,
    _make_run_env,
    build_subprocess_env,
    hermes_subprocess_env,
)


_SOURCE_ONLY = "ACME_LOGIN"
_SHARED = "DATABASE_URL"
_FORCE = "_HERMES_FORCE_ACME_LOGIN"


def _write_profile(home: Path, values: dict[str, str]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


@pytest.fixture
def multiplex_mode():
    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(False)


def test_boundary_removes_source_only_and_replaces_same_name(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SOURCE_ONLY: "alpha", _SHARED: "alpha-db"})
    _write_profile(target, {_SHARED: "beta-db"})

    boundary = build_profile_env_boundary(source, target)
    result = boundary.sanitize(
        {
            _SOURCE_ONLY: "alpha",
            _SHARED: "alpha-db",
            "PATH": "/usr/bin",
        }
    )

    assert _SOURCE_ONLY not in result
    assert result[_SHARED] == "beta-db"
    assert result["PATH"] == "/usr/bin"
    assert boundary.identity == str(target.resolve())


def test_boundary_matches_windows_environment_names_case_insensitively(
    tmp_path, monkeypatch
):
    import agent.secret_scope as secret_scope

    monkeypatch.setattr(secret_scope, "_ENV_KEYS_CASE_INSENSITIVE", True)
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    remove_boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset({"Acme_Login"}),
        target_values={},
    )
    removed = remove_boundary.sanitize(
        {"ACME_LOGIN": "source-value", "Path": "C:/Windows/System32"}
    )
    assert "ACME_LOGIN" not in removed
    assert removed["Path"] == "C:/Windows/System32"

    replace_boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset({"Acme_Login"}),
        target_values={"acme_login": "target-value"},
    )
    replaced = replace_boundary.sanitize({"ACME_LOGIN": "source-value"})
    assert replaced == {"ACME_LOGIN": "target-value"}

    _write_profile(source, {"Path": "C:/bad", "Acme_Login": "source-value"})
    assert get_profile_owned_secret_names(source) == frozenset({"Acme_Login"})


def test_boundary_applies_ownership_to_container_forwarding_aliases(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    remove_boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset({"DATABASE_URL"}),
        target_values={},
    )
    assert remove_boundary.sanitize(
        {"APPTAINERENV_DATABASE_URL": "source-value"}
    ) == {}

    replace_boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset({"DATABASE_URL"}),
        target_values={"DATABASE_URL": "target-value"},
    )
    assert replace_boundary.sanitize(
        {"SINGULARITYENV_DATABASE_URL": "source-value"}
    ) == {"SINGULARITYENV_DATABASE_URL": "target-value"}


def test_make_run_env_applies_boundary_to_arbitrary_names(
    tmp_path, monkeypatch, multiplex_mode
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SOURCE_ONLY: "alpha", _SHARED: "alpha-db"})
    _write_profile(target, {_SHARED: "beta-db"})
    monkeypatch.setenv("HERMES_HOME", str(source))
    token = set_hermes_home_override(target)
    try:
        monkeypatch.setenv(_SOURCE_ONLY, "alpha")
        monkeypatch.setenv(_SHARED, "alpha-db")
        result = _make_run_env({})
    finally:
        reset_hermes_home_override(token)

    assert _SOURCE_ONLY not in result
    assert result[_SHARED] == "beta-db"


def test_nonterminal_model_driver_env_uses_target_profile_boundary(
    tmp_path, monkeypatch, multiplex_mode
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(
        source,
        {"OPENAI_API_KEY": "source-provider", "CUSTOM_CAP": "source-cap"},
    )
    _write_profile(target, {"OPENAI_API_KEY": "target-provider"})
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv("OPENAI_API_KEY", "source-provider")
    monkeypatch.setenv("CUSTOM_CAP", "source-cap")

    token = set_hermes_home_override(target)
    try:
        result = hermes_subprocess_env(inherit_credentials=True)
    finally:
        reset_hermes_home_override(token)

    assert result["OPENAI_API_KEY"] == "target-provider"
    assert "CUSTOM_CAP" not in result


def test_build_subprocess_env_supports_standalone_explicit_boundary(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SOURCE_ONLY: "alpha", _SHARED: "alpha-db"})
    _write_profile(target, {_SHARED: "beta-db"})

    result = build_subprocess_env(
        base={
            "PATH": "/usr/bin",
            _SOURCE_ONLY: "alpha",
            _SHARED: "alpha-db",
        },
        profile_home=target,
        source_profile_home=source,
        enforce_profile_boundary=True,
    )

    assert _SOURCE_ONLY not in result
    assert result[_SHARED] == "beta-db"
    assert result["PATH"].endswith(":/usr/bin")


def test_ambient_force_prefix_cannot_unwrap_in_make_run_env(
    tmp_path, monkeypatch, multiplex_mode
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SOURCE_ONLY: "alpha"})
    _write_profile(target, {})
    monkeypatch.setenv("HERMES_HOME", str(source))
    token = set_hermes_home_override(target)
    try:
        monkeypatch.setenv(_FORCE, "alpha")
        result = _make_run_env({})
        explicit = _make_run_env({_FORCE: "explicit-beta"})
    finally:
        reset_hermes_home_override(token)

    assert _SOURCE_ONLY not in result
    assert _SOURCE_ONLY not in explicit


def test_make_run_env_force_prefix_cannot_override_target_owned_value(
    tmp_path, monkeypatch, multiplex_mode
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SHARED: "alpha-db"})
    _write_profile(target, {_SHARED: "beta-db"})
    monkeypatch.setenv("HERMES_HOME", str(source))
    token = set_hermes_home_override(target)
    try:
        result = _make_run_env({"_HERMES_FORCE_DATABASE_URL": "attacker-value"})
    finally:
        reset_hermes_home_override(token)

    assert result[_SHARED] == "beta-db"


def test_explicit_force_prefix_cannot_bypass_build_subprocess_boundary(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SOURCE_ONLY: "alpha"})
    _write_profile(target, {})

    result = build_subprocess_env(
        base={"PATH": "/usr/bin", _SOURCE_ONLY: "ambient"},
        extra={_FORCE: "forced-source"},
        profile_home=target,
        source_profile_home=source,
        enforce_profile_boundary=True,
    )

    assert _SOURCE_ONLY not in result


def test_force_prefix_cannot_override_target_owned_value(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SHARED: "alpha-db"})
    _write_profile(target, {_SHARED: "beta-db"})

    result = build_subprocess_env(
        base={},
        extra={"_HERMES_FORCE_DATABASE_URL": "attacker-value"},
        profile_home=target,
        source_profile_home=source,
        enforce_profile_boundary=True,
    )

    assert result[_SHARED] == "beta-db"


def test_profile_op_env_values_are_boundary_scoped(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SOURCE_ONLY: "source-env"})
    _write_profile(target, {})
    (source / ".op.env").write_text(f"{_SHARED}=source-op\n", encoding="utf-8")
    (target / ".op.env").write_text(f"{_SHARED}=target-op\n", encoding="utf-8")

    boundary = build_profile_env_boundary(source, target)
    result = boundary.sanitize({_SHARED: "source-op"})

    assert result[_SHARED] == "target-op"


def test_external_secret_source_failure_preserves_normal_scope_fail_open(
    monkeypatch, tmp_path
):
    import hermes_cli.env_loader as env_loader

    def _raise(_home):
        raise RuntimeError("secret backend unavailable")

    monkeypatch.setattr(env_loader, "get_secret_source_values", _raise)
    assert build_profile_secret_scope(tmp_path) == {}


def test_external_secret_source_failure_refuses_boundary(monkeypatch, tmp_path):
    import hermes_cli.env_loader as env_loader

    def _raise(_home):
        raise RuntimeError("secret backend unavailable")

    monkeypatch.setattr(env_loader, "get_secret_source_values", _raise)
    with pytest.raises(RuntimeError, match="secret backend unavailable"):
        build_profile_env_boundary(tmp_path / "source", tmp_path / "target")


def test_multiplex_target_home_resolution_failure_refuses_run(
    monkeypatch, tmp_path, multiplex_mode
):
    source = tmp_path / "source"
    _write_profile(source, {_SOURCE_ONLY: "alpha"})
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv(_SOURCE_ONLY, "alpha")

    def _raise():
        raise RuntimeError("profile context unavailable")

    monkeypatch.setattr("hermes_constants.get_hermes_home_override", _raise)
    with pytest.raises(RuntimeError, match="boundary could not be constructed"):
        _make_run_env({})


class _DummyEnvironment(BaseEnvironment):
    _profile_scoped_passthrough = True

    def _run_bash(self, *args, **kwargs):  # pragma: no cover - not used here
        raise AssertionError("not called")

    def cleanup(self):  # pragma: no cover - not used here
        return None


def _dummy_with_boundary(boundary: ProfileEnvBoundary) -> _DummyEnvironment:
    obj = _DummyEnvironment.__new__(_DummyEnvironment)
    obj._profile_env_boundary = boundary
    obj._snapshot_passthrough_names = set()
    return obj


def test_snapshot_exclusions_include_profile_owned_names(tmp_path, multiplex_mode):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SOURCE_ONLY: "alpha"})
    _write_profile(target, {})
    boundary = build_profile_env_boundary(source, target)
    obj = _dummy_with_boundary(boundary)

    names = obj._snapshot_excluded_passthrough_names()

    assert _SOURCE_ONLY in names


def test_snapshot_exclusion_refresh_fails_closed(monkeypatch, tmp_path, multiplex_mode):
    import agent.secret_scope as secret_scope

    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {})
    _write_profile(target, {})
    obj = _dummy_with_boundary(build_profile_env_boundary(source, target))

    def _raise(*_args, **_kwargs):
        raise RuntimeError("ownership unavailable")

    monkeypatch.setattr(secret_scope, "get_profile_owned_secret_names", _raise)
    with pytest.raises(RuntimeError, match="snapshot exclusions"):
        obj._snapshot_excluded_passthrough_names()


def test_snapshot_refreshes_ownership_added_after_construction(
    tmp_path, monkeypatch, multiplex_mode
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {})
    _write_profile(target, {})
    monkeypatch.setenv("HERMES_HOME", str(source))

    environment = LocalEnvironment(cwd=str(tmp_path))
    try:
        # Ownership appears only after the long-lived environment and its first
        # snapshot already exist (dotenv/external-source hot reload shape).
        _write_profile(source, {_SOURCE_ONLY: "alpha"})
        monkeypatch.setenv(_SOURCE_ONLY, "alpha")

        source_result = environment.execute(
            'printf "A:%s" "${' + _SOURCE_ONLY + '-UNSET}"'
        )
        assert "A:alpha" in source_result["output"]

        token = set_hermes_home_override(target)
        try:
            target_result = environment.execute(
                'printf "B:%s" "${' + _SOURCE_ONLY + '-UNSET}"'
            )
        finally:
            reset_hermes_home_override(token)

        assert "B:UNSET" in target_result["output"]
        assert "alpha" not in target_result["output"]
    finally:
        environment.cleanup()


def test_snapshot_restore_cannot_reintroduce_foreign_value(tmp_path, multiplex_mode):
    source = tmp_path / "source"
    target = tmp_path / "target"
    snapshot = tmp_path / "snapshot.sh"
    _write_profile(source, {_SOURCE_ONLY: "alpha"})
    _write_profile(target, {})
    snapshot.write_text(f"export {_SOURCE_ONLY}=alpha\n", encoding="utf-8")

    boundary = build_profile_env_boundary(source, target)
    obj = _dummy_with_boundary(boundary)
    obj._snapshot_ready = True
    obj._snapshot_path = str(snapshot)
    obj._cwd_marker = "__CWD__"
    obj._session_id = "test"
    obj.cwd = str(tmp_path)
    obj.env = {}
    obj._prefer_nonlogin = False

    script = obj._wrap_command(
        'printf "%s" "${' + _SOURCE_ONLY + '-UNSET}"',
        str(tmp_path),
    )
    env = os.environ.copy()
    env[_SOURCE_ONLY] = "beta-process"
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "beta-process" in completed.stdout
    assert "alpha" not in completed.stdout


def test_kanban_spawn_fails_closed_for_missing_profile(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb

    class _Task:
        id = "task-missing-profile"
        assignee = "missing-profile"

    def _missing(_profile):
        raise FileNotFoundError("missing profile")

    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", _missing)

    with pytest.raises(RuntimeError, match="unresolved profile"):
        kb._default_spawn(cast(Any, _Task()), str(tmp_path))


def test_target_profile_password_still_requires_explicit_passthrough(tmp_path):
    """Provenance replacement cannot bypass generic credential filtering."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {"DB_PASSWORD": "source-value"})
    _write_profile(target, {"DB_PASSWORD": "target-value"})

    result = build_subprocess_env(
        base={"DB_PASSWORD": "source-value", "PATH": "/usr/bin"},
        profile_home=target,
        source_profile_home=source,
        enforce_profile_boundary=True,
    )
    assert "DB_PASSWORD" not in result

    from tools.env_passthrough import clear_env_passthrough, register_env_passthrough

    register_env_passthrough(["DB_PASSWORD"])
    try:
        permitted = build_subprocess_env(
            base={"DB_PASSWORD": "source-value", "PATH": "/usr/bin"},
            profile_home=target,
            source_profile_home=source,
            enforce_profile_boundary=True,
        )
    finally:
        clear_env_passthrough()

    assert permitted["DB_PASSWORD"] == "target-value"
