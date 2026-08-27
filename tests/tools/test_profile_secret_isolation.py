"""Regression tests for multiplex profile-owned environment isolation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from agent.secret_scope import (
    ProfileEnvBoundary,
    _env_name_key as secret_scope_name_key,
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


def test_boundary_revocation_keeps_prior_source_name_tainted(tmp_path, monkeypatch):
    """PA-002: deleting ownership must not legitimize stale ambient state."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {_SOURCE_ONLY: "alpha"})
    _write_profile(target, {})
    monkeypatch.setenv(_SOURCE_ONLY, "stale-alpha")

    first = build_profile_env_boundary(source, target)
    assert _SOURCE_ONLY not in first.sanitize(
        {_SOURCE_ONLY: "stale-alpha", "BENIGN_CONTROL": "preserved"}
    )

    _write_profile(source, {})
    after_revocation = build_profile_env_boundary(source, target).sanitize(
        {_SOURCE_ONLY: "stale-alpha", "BENIGN_CONTROL": "preserved"}
    )

    assert _SOURCE_ONLY not in after_revocation
    assert after_revocation["BENIGN_CONTROL"] == "preserved"


def test_boundary_refuses_failed_dotenv_parse_instead_of_treating_it_as_empty(
    tmp_path,
):
    """PA-009: unknown profile source state must not widen ambient authority."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / ".env").write_bytes(b"ACME_LOGIN=\xff\xfe\n")
    _write_profile(target, {})

    with pytest.raises(RuntimeError, match="dotenv ownership unavailable"):
        build_profile_env_boundary(source, target)


def test_forwarded_path_source_is_removed_when_target_has_no_path_grant(tmp_path):
    """CAR-PATH-A: a forwarding carrier is not a global PATH grant."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {"APPTAINERENV_PATH": "/source/bin"})
    _write_profile(target, {})

    result = build_profile_env_boundary(source, target).sanitize(
        {"PATH": "/usr/bin", "APPTAINERENV_PATH": "/source/bin"}
    )

    assert result["PATH"] == "/usr/bin"
    assert "APPTAINERENV_PATH" not in result


def test_forwarded_path_source_cannot_choose_target_direct_carrier(tmp_path):
    """CAR-PATH-B: target direct PATH must not inherit a source wrapper."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {"APPTAINERENV_PATH": "/source/bin"})
    _write_profile(target, {"PATH": "/target/bin"})

    result = build_profile_env_boundary(source, target).sanitize(
        {"APPTAINERENV_PATH": "/source/bin", "PATH": "/usr/bin"}
    )

    assert result["PATH"] == "/target/bin"
    assert "APPTAINERENV_PATH" not in result


def test_profile_config_passthrough_cache_cannot_authorize_a_sibling(
    tmp_path, monkeypatch, multiplex_mode
):
    """POL-001: profile A's allowlist must not authorize profile B."""
    import tools.env_passthrough as passthrough

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    for home in (profile_a, profile_b):
        home.mkdir()
    (profile_a / "config.yaml").write_text(
        "terminal:\n  env_passthrough:\n    - DB_PASSWORD\n",
        encoding="utf-8",
    )
    (profile_b / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(passthrough, "_config_passthrough", None)

    token_a = set_hermes_home_override(profile_a)
    try:
        assert passthrough.is_env_passthrough("DB_PASSWORD") is True
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(profile_b)
    try:
        assert passthrough.is_env_passthrough("DB_PASSWORD") is False
    finally:
        reset_hermes_home_override(token_b)


def test_unbound_docker_credential_mounts_are_quarantined_only_in_multiplex(
    tmp_path, monkeypatch
):
    from tools import credential_files
    from tools.environments.docker import _credential_file_mounts_for_boundary

    source = tmp_path / "source"
    target = tmp_path / "target"
    boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset(),
        target_values={},
    )
    mount = {"host_path": "/foreign/token", "container_path": "/token"}
    monkeypatch.setattr(credential_files, "get_credential_file_mounts", lambda: [mount])

    assert _credential_file_mounts_for_boundary(boundary) == []
    assert _credential_file_mounts_for_boundary(None) == [mount]


def test_installed_profile_scope_is_immutable_and_generation_bound(tmp_path):
    from agent.secret_scope import current_secret_scope, reset_secret_scope, set_secret_scope

    profile = tmp_path / "profile"
    _write_profile(profile, {"ACME_TOKEN": "one"})
    first = build_profile_secret_scope(profile)
    token = set_secret_scope(first)
    try:
        installed = current_secret_scope()
        assert installed is first
        assert installed is not None
        with pytest.raises(TypeError):
            installed.data["ACME_TOKEN"] = "mutated"  # type: ignore[index]

        _write_profile(profile, {"ACME_TOKEN": "two", "SECOND": "value"})
        second = build_profile_secret_scope(profile)
        assert second.generation != first.generation
        assert installed["ACME_TOKEN"] == "one"
        assert second["ACME_TOKEN"] == "two"
    finally:
        reset_secret_scope(token)


def test_installed_target_scope_refuses_after_source_generation_changes(tmp_path):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {})
    _write_profile(target, {"ACME_TOKEN": "old"})
    token = set_secret_scope(build_profile_secret_scope(target))
    try:
        _write_profile(target, {"ACME_TOKEN": "new"})
        with pytest.raises(RuntimeError, match="scope is stale"):
            build_profile_env_boundary(source, target)
    finally:
        reset_secret_scope(token)


def test_legacy_external_value_projection_is_not_fail_closed_authority(tmp_path):
    from hermes_cli import env_loader

    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {})
    _write_profile(target, {})
    env_loader.reset_secret_source_cache()
    env_loader._SECRET_SOURCE_VALUES_BY_HOME[str(target.resolve())] = {
        "ACME_TOKEN": "legacy-unversioned"
    }
    try:
        snapshot = env_loader.get_external_secret_snapshot(target)
        assert snapshot.status == "degraded"
        assert snapshot.generation == 0
        with pytest.raises(RuntimeError, match="snapshot is degraded"):
            build_profile_env_boundary(source, target)
    finally:
        env_loader.reset_secret_source_cache()


def test_scope_generation_binds_equal_length_value_with_restored_mtime(tmp_path):
    """A value rotation must advance authority even when metadata is unchanged."""
    profile = tmp_path / "profile"
    _write_profile(profile, {"ACME_TOKEN": "one"})
    env_path = profile / ".env"
    original = env_path.stat()
    first = build_profile_secret_scope(profile)

    _write_profile(profile, {"ACME_TOKEN": "two"})
    os.utime(env_path, ns=(original.st_atime_ns, original.st_mtime_ns))
    second = build_profile_secret_scope(profile)

    assert env_path.stat().st_size == original.st_size
    assert env_path.stat().st_mtime_ns == original.st_mtime_ns
    assert second["ACME_TOKEN"] == "two"
    assert second.generation != first.generation


def test_scope_profile_identity_mismatch_refuses_before_child_env(tmp_path, monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    source = tmp_path / "source"
    profile_b = tmp_path / "profile-b"
    profile_c = tmp_path / "profile-c"
    _write_profile(source, {"ACME_TOKEN": "source"})
    _write_profile(profile_b, {"ACME_TOKEN": "b"})
    _write_profile(profile_c, {"ACME_TOKEN": "c"})
    monkeypatch.setenv("HERMES_HOME", str(source))

    set_multiplex_active(True)
    scope_token = set_secret_scope(build_profile_secret_scope(profile_c))
    home_token = set_hermes_home_override(profile_b)
    try:
        with pytest.raises(RuntimeError, match="does not match target profile"):
            build_subprocess_env(base={"PATH": "/usr/bin"})
    finally:
        reset_hermes_home_override(home_token)
        reset_secret_scope(scope_token)
        set_multiplex_active(False)


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
    assert replaced == {"acme_login": "target-value"}

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
    ) == {"DATABASE_URL": "target-value"}


def test_conflicting_target_carriers_fail_closed_deterministically(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset(),
        target_values={
            "DATABASE_URL": "direct-value",
            "APPTAINERENV_DATABASE_URL": "forwarded-value",
        },
    )

    with pytest.raises(RuntimeError, match="conflicting target environment"):
        boundary.sanitize({"PATH": "/usr/bin"})


def test_equal_target_carriers_still_fail_closed_as_duplicate_authority(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset({"DATABASE_URL"}),
        target_values={
            "APPTAINERENV_DATABASE_URL": "target-value",
            "DATABASE_URL": "target-value",
        },
    )

    with pytest.raises(RuntimeError, match="conflicting target environment"):
        boundary.sanitize({"SINGULARITYENV_DATABASE_URL": "source-value"})


def test_nonterminal_inheritance_cannot_reinsert_raw_duplicate_aliases(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset({"OPENAI_API_KEY"}),
        target_values={
            "OPENAI_API_KEY": "target-value",
            "APPTAINERENV_OPENAI_API_KEY": "target-value",
        },
    )

    with pytest.raises(RuntimeError, match="conflicting target environment"):
        hermes_subprocess_env(
            inherit_credentials=True,
            profile_boundary=boundary,
        )


@pytest.mark.parametrize(
    "aliases",
    [
        [
            ("DATABASE_URL", "direct-source"),
            ("APPTAINERENV_DATABASE_URL", "apptainer-source"),
            ("SINGULARITYENV_DATABASE_URL", "singularity-source"),
        ],
        [
            ("SINGULARITYENV_DATABASE_URL", "singularity-source"),
            ("APPTAINERENV_DATABASE_URL", "apptainer-source"),
            ("DATABASE_URL", "direct-source"),
        ],
    ],
)
def test_boundary_removes_or_replaces_all_duplicate_normalized_aliases(
    aliases: list[tuple[str, str]], tmp_path
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    base: dict[str, str] = dict([*aliases, ("PATH", "/usr/bin")])

    remove_boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset({"DATABASE_URL"}),
        target_values={},
    )
    assert remove_boundary.sanitize(base) == {"PATH": "/usr/bin"}

    replace_boundary = ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=frozenset({"DATABASE_URL"}),
        target_values={"DATABASE_URL": "target-value"},
    )
    replaced = replace_boundary.sanitize(base)
    matching = {
        key: value
        for key, value in replaced.items()
        if secret_scope_name_key(key) == "DATABASE_URL"
    }
    assert matching and set(matching.values()) == {"target-value"}
    assert len(matching) == 1
    assert replaced["PATH"] == "/usr/bin"


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


def test_make_run_env_materializes_target_only_passthrough(
    tmp_path, monkeypatch, multiplex_mode
):
    """Foreground local execution must honor the target profile's explicit grant."""
    import tools.env_passthrough as passthrough

    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {})
    _write_profile(target, {"ACME_LOGIN": "target-only"})
    (target / "config.yaml").write_text(
        "terminal:\n  env_passthrough:\n    - ACME_LOGIN\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(passthrough, "_config_passthrough", None)
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.delenv("ACME_LOGIN", raising=False)

    token = set_hermes_home_override(target)
    try:
        result = _make_run_env({"PATH": "/usr/bin"})
    finally:
        reset_hermes_home_override(token)

    assert result["ACME_LOGIN"] == "target-only"
    assert result["HERMES_HOME"] == str(target.resolve())


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


def test_nonterminal_model_driver_receives_target_only_provider_not_other_secrets(
    tmp_path, monkeypatch, multiplex_mode
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {})
    _write_profile(
        target,
        {
            "OPENAI_API_KEY": "target-only-provider",
            "AWS_ACCESS_KEY_ID": "target-only-aws",
            "TELEGRAM_BOT_TOKEN": "target-messaging-token",
            "BROWSERBASE_API_KEY": "target-tool-token",
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)

    token = set_hermes_home_override(target)
    try:
        result = hermes_subprocess_env(inherit_credentials=True)
    finally:
        reset_hermes_home_override(token)

    assert result["OPENAI_API_KEY"] == "target-only-provider"
    assert result["AWS_ACCESS_KEY_ID"] == "target-only-aws"
    assert "TELEGRAM_BOT_TOKEN" not in result
    assert "BROWSERBASE_API_KEY" not in result


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

    monkeypatch.setattr(env_loader, "get_external_secret_snapshot", _raise)
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


def test_same_profile_password_keeps_trusted_shell_compatibility(
    tmp_path, multiplex_mode
):
    profile = tmp_path / "profile"
    _write_profile(profile, {})

    result = build_subprocess_env(
        base={"DB_PASSWORD": "trusted-shell", "PATH": "/usr/bin"},
        profile_home=profile,
        source_profile_home=profile,
        enforce_profile_boundary=True,
    )

    assert result["DB_PASSWORD"] == "trusted-shell"


def test_target_only_passthrough_value_materializes_without_ambient_seed(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {})
    _write_profile(target, {"ACME_LOGIN": "target-only"})
    (target / "config.yaml").write_text(
        "terminal:\n  env_passthrough:\n    - ACME_LOGIN\n",
        encoding="utf-8",
    )

    result = build_subprocess_env(
        base={"PATH": "/usr/bin"},
        profile_home=target,
        source_profile_home=source,
        enforce_profile_boundary=True,
    )

    assert result["ACME_LOGIN"] == "target-only"


def test_target_only_unpermitted_value_stays_absent(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {})
    _write_profile(target, {"ACME_LOGIN": "target-only"})
    (target / "config.yaml").write_text("{}\n", encoding="utf-8")

    result = build_subprocess_env(
        base={"PATH": "/usr/bin"},
        profile_home=target,
        source_profile_home=source,
        enforce_profile_boundary=True,
    )

    assert "ACME_LOGIN" not in result


def test_late_provider_registration_updates_terminal_and_model_policy(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from hermes_cli import auth

    provider = SimpleNamespace(
        api_key_env_vars=("ACME_PROVIDER_API_KEY",),
        auth_type="api_key",
        base_url_env_var=None,
    )
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, "late-acme", provider)

    terminal_env = build_subprocess_env(
        base={"PATH": "/usr/bin", "ACME_PROVIDER_API_KEY": "ambient"}
    )
    assert "ACME_PROVIDER_API_KEY" not in terminal_env

    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_profile(source, {"ACME_PROVIDER_API_KEY": "source"})
    _write_profile(target, {"ACME_PROVIDER_API_KEY": "target"})
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv("ACME_PROVIDER_API_KEY", "source")
    set_multiplex_active(True)
    token = set_hermes_home_override(target)
    try:
        model_env = hermes_subprocess_env(inherit_credentials=True)
    finally:
        reset_hermes_home_override(token)
        set_multiplex_active(False)

    assert model_env["ACME_PROVIDER_API_KEY"] == "target"
