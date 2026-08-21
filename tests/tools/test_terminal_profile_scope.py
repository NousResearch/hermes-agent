"""Regression tests for context-local terminal backend configuration."""

import asyncio
import os

from tools.terminal_tool import (
    _get_env_config,
    _resolve_container_task_id,
    terminal_config_scope,
)


def test_terminal_config_scope_does_not_mutate_process_environment(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    with terminal_config_scope({"TERMINAL_ENV": "docker", "TERMINAL_DOCKER_IMAGE": "profile-image"}):
        config = _get_env_config()
        assert config["env_type"] == "docker"
        assert config["docker_image"] == "profile-image"
    assert "TERMINAL_ENV" not in os.environ


def test_overlapping_profile_scopes_remain_isolated():
    async def observe(backend, image, entered):
        with terminal_config_scope(
            {"TERMINAL_ENV": backend, "TERMINAL_DOCKER_IMAGE": image}
        ):
            entered.set()
            await asyncio.sleep(0)
            config = _get_env_config()
            return config["env_type"], config["docker_image"]

    async def run():
        first = asyncio.Event()
        second = asyncio.Event()

        async def profile_a():
            result = await observe("docker", "profile-a", first)
            assert second.is_set()
            return result

        async def profile_b():
            await first.wait()
            result = await observe("local", "profile-b", second)
            return result

        return await asyncio.gather(profile_a(), profile_b())

    assert asyncio.run(run()) == [("docker", "profile-a"), ("local", "profile-b")]


def test_profile_scope_namespaces_environment_keys():
    with terminal_config_scope({"TERMINAL_ENV": "docker", "__profile_key": "/profile/a"}):
        a = _resolve_container_task_id("session")
    with terminal_config_scope({"TERMINAL_ENV": "docker", "__profile_key": "/profile/b"}):
        b = _resolve_container_task_id("session")
    assert a != b
