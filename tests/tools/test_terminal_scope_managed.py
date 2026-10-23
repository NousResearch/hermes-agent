"""Managed Scope wins inside the per-turn terminal policy (#138044).

``build_profile_terminal_scope`` rebuilt a routed profile's COMPLETE ``TERMINAL_*`` policy from
the profile's own ``.env`` and ``config.yaml`` only, so an administrator-pinned
``terminal.backend: docker`` was replaced by the built-in default ``local``. Under
``gateway.multiplex_profiles`` that runs on every turn of every surface (messaging gateway, cron,
secondary TUI sessions): the sandbox was silently skipped and commands ran on the host, able to
read other profiles' homes (#90040/#91073 class, #128410 symptom).

Managed scope is documented as the top tier for every surface — ``managed-scope.md``
"Precedence": ``/etc/hermes/config.yaml`` and ``/etc/hermes/.env`` beat the user's files and the
shell environment, leaf-level — and the docs carve out no exception for ``terminal.*``.
"""

from __future__ import annotations

import pytest

from tools.terminal_scope import build_profile_terminal_scope


@pytest.fixture(autouse=True)
def _managed_scope(monkeypatch, tmp_path):
    """A managed dir under the test's tmp_path; ``HERMES_MANAGED_DIR`` is the documented
    bootstrap override and is honored even under pytest (``managed_scope.get_managed_dir``)."""
    from hermes_cli import managed_scope

    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    managed_scope.invalidate_managed_cache()
    yield managed
    managed_scope.invalidate_managed_cache()


def _profile(tmp_path, config_yaml="", dotenv=""):
    home = tmp_path / "profiles" / "routed"
    home.mkdir(parents=True)
    if config_yaml:
        (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    if dotenv:
        (home / ".env").write_text(dotenv, encoding="utf-8")
    return home


def _write_managed(managed, config_yaml="", dotenv=""):
    if config_yaml:
        (managed / "config.yaml").write_text(config_yaml, encoding="utf-8")
    if dotenv:
        (managed / ".env").write_text(dotenv, encoding="utf-8")


def test_managed_config_backend_reaches_the_scope(tmp_path, _managed_scope):
    """The reported bug: managed ``terminal.backend: docker`` was replaced by the default local."""
    _write_managed(_managed_scope, config_yaml=(
        "terminal:\n"
        "  backend: docker\n"
        "  docker_image: nousresearch/hermes-sandbox:desktop\n"
    ))

    scope = build_profile_terminal_scope(_profile(tmp_path))

    assert scope["TERMINAL_ENV"] == "docker"
    assert scope["TERMINAL_DOCKER_IMAGE"] == "nousresearch/hermes-sandbox:desktop"
    # A managed image is the administrator's choice, never the shipped default: it pins, so a
    # persisted container is recreated on mismatch instead of kept.
    assert scope["TERMINAL_DOCKER_IMAGE_PINNED"] == "1"


def test_managed_terminal_beats_the_profile_config(tmp_path, _managed_scope):
    """Tier 1 (managed) wins over tier 2 (the profile's own files), leaf-level."""
    _write_managed(_managed_scope, config_yaml="terminal:\n  backend: docker\n")

    scope = build_profile_terminal_scope(_profile(
        tmp_path,
        config_yaml="terminal:\n  backend: local\n  docker_image: user/image:1\n",
    ))

    assert scope["TERMINAL_ENV"] == "docker"
    # Only the pinned leaf moved: the profile's own docker_image stays user-controlled.
    assert scope["TERMINAL_DOCKER_IMAGE"] == "user/image:1"
    assert scope["TERMINAL_DOCKER_IMAGE_PINNED"] == "1"


def test_managed_env_terminal_wins_over_the_profile_yaml(tmp_path, _managed_scope):
    """The managed layer is applied last, so a managed TERMINAL_* also beats a user YAML key."""
    _write_managed(_managed_scope, dotenv="TERMINAL_ENV=docker\n")

    scope = build_profile_terminal_scope(_profile(
        tmp_path, config_yaml="terminal:\n  backend: local\n"))

    assert scope["TERMINAL_ENV"] == "docker"


def test_managed_env_docker_image_pins(tmp_path, _managed_scope):
    _write_managed(_managed_scope, dotenv="TERMINAL_DOCKER_IMAGE=registry.example/admin:1\n")

    scope = build_profile_terminal_scope(_profile(tmp_path))

    assert scope["TERMINAL_DOCKER_IMAGE"] == "registry.example/admin:1"
    assert scope["TERMINAL_DOCKER_IMAGE_PINNED"] == "1"


def test_managed_terminal_is_leaf_level(tmp_path, _managed_scope):
    """Pinning the backend must not freeze the rest of the profile's terminal.* policy."""
    _write_managed(_managed_scope, config_yaml="terminal:\n  backend: docker\n")

    scope = build_profile_terminal_scope(_profile(
        tmp_path, config_yaml="terminal:\n  timeout: 45\n"))

    assert scope["TERMINAL_ENV"] == "docker"
    assert scope["TERMINAL_TIMEOUT"] == "45"


def test_no_managed_scope_leaves_the_profile_policy_alone(tmp_path, monkeypatch):
    """No managed dir (the common install): the profile's own policy resolves as before."""
    from hermes_cli import managed_scope

    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    managed_scope.invalidate_managed_cache()

    scope = build_profile_terminal_scope(_profile(
        tmp_path, config_yaml="terminal:\n  backend: ssh\n  ssh_host: 10.10.0.103\n"))

    assert scope["TERMINAL_ENV"] == "ssh"
    assert scope["TERMINAL_SSH_HOST"] == "10.10.0.103"
