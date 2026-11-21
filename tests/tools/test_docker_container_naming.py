"""Sandbox container names carry the owning profile.

Multi-profile installs spawn identical-looking containers (``hermes-<rand>``);
the name now correlates to its profile for ops readability. The profile segment
is display-only — reuse, reaping and drift resolve through the ``hermes-profile``
label — so the contract is purely: valid docker name, profile inside, unique
per spawn.
"""

import re

import pytest

from tools.environments import docker as docker_env

# docker create --name: [a-zA-Z0-9][a-zA-Z0-9_.-]*
_DOCKER_NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")


def _name_for(monkeypatch, profile: str) -> str:
    monkeypatch.setattr(docker_env, "home_profile_name", lambda: profile)
    return docker_env._new_container_name()


def test_name_embeds_profile(monkeypatch):
    assert re.fullmatch(r"hermes-testbot-[0-9a-f]{6}", _name_for(monkeypatch, "testbot"))


def test_default_profile_names_cleanly(monkeypatch):
    assert re.fullmatch(r"hermes-default-[0-9a-f]{6}", _name_for(monkeypatch, "default"))


@pytest.mark.parametrize("profile", ["my bot", "o'brien", "a/b", "profilé", "", "x" * 200])
def test_odd_profile_names_stay_docker_valid(monkeypatch, profile):
    """An unsanitized profile segment would make `docker run --name` fail; every
    produced name must satisfy the docker charset regardless of input."""
    assert _DOCKER_NAME_RE.fullmatch(_name_for(monkeypatch, profile))


def test_names_unique_per_spawn(monkeypatch):
    """Two sandboxes of one profile must not collide on --name."""
    monkeypatch.setattr(docker_env, "home_profile_name", lambda: "testbot")
    assert docker_env._new_container_name() != docker_env._new_container_name()
