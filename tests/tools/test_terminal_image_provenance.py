"""Sandbox image references are checked where they actually arrive.

The setup wizard no longer prompts for terminal images — they reach the
backends via the TERMINAL_*_IMAGE env vars / config file and via
`register_task_env_overrides()`. Both are code-execution inputs (the backend
pulls and runs the image), so provenance is checked at those two chokepoints,
warn-only.
"""

import json
import logging

import pytest

from tools.terminal_tool import (
    _check_image_provenance,
    _image_registry_host,
    _warned_images,
    image_is_trusted,
    register_task_env_overrides,
)


@pytest.fixture(autouse=True)
def _clear_warn_cache():
    _warned_images.clear()
    yield
    _warned_images.clear()


# ── Registry resolution ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "image, expected",
    [
        # Implicit Docker Hub: no dot/colon in the first segment.
        ("nikolaik/python-nodejs:python3.11-nodejs20", "docker.io"),
        ("python:3.11", "docker.io"),
        ("ubuntu", "docker.io"),
        # Explicit hosts.
        ("ghcr.io/nousresearch/hermes:latest", "ghcr.io"),
        ("gcr.io/project/img:v1", "gcr.io"),
        ("evil.example.com/backdoor:latest", "evil.example.com"),
        ("localhost:5000/dev:latest", "localhost:5000"),
        # Singularity/OCI schemes are stripped before classification.
        ("docker://nikolaik/python-nodejs:python3.11-nodejs20", "docker.io"),
        ("docker://evil.example.com/x:1", "evil.example.com"),
        ("oci://ghcr.io/org/img:1", "ghcr.io"),
        # Not registry references at all.
        ("/home/user/project/Dockerfile", None),
        ("./Dockerfile", None),
        ("", None),
    ],
)
def test_image_registry_host(image, expected):
    assert _image_registry_host(image) == expected


@pytest.mark.parametrize(
    "image, trusted",
    [
        ("nikolaik/python-nodejs:python3.11-nodejs20", True),
        ("python:3.11", True),
        ("ghcr.io/nousresearch/hermes:latest", True),
        ("quay.io/org/img:1", True),
        ("docker://nikolaik/python-nodejs:python3.11-nodejs20", True),
        ("evil.example.com/backdoor:latest", False),
        ("192.168.1.10:5000/img:1", False),
        # Unclassifiable refs (local Dockerfiles, Modal build contexts) pass.
        ("/home/user/project/Dockerfile", True),
    ],
)
def test_image_is_trusted(image, trusted):
    assert image_is_trusted(image) is trusted


def test_operator_extends_trusted_registries_from_config(monkeypatch):
    """``terminal.trusted_image_registries`` in config.yaml.

    The startup bridges JSON-encode list values into the TERMINAL_* env var
    (same as docker_volumes / docker_env), so that is the form this must
    accept.
    """
    assert image_is_trusted("registry.corp.internal/base:1") is False

    monkeypatch.setenv(
        "TERMINAL_TRUSTED_IMAGE_REGISTRIES",
        json.dumps(["registry.corp.internal", "harbor.example.com"]),
    )
    assert image_is_trusted("registry.corp.internal/base:1") is True
    assert image_is_trusted("harbor.example.com/base:1") is True
    assert image_is_trusted("evil.example.com/x:1") is False


def test_trusted_registries_accepts_comma_separated_env(monkeypatch):
    """Hand-exported env (no config bridge) still works."""
    monkeypatch.setenv(
        "TERMINAL_TRUSTED_IMAGE_REGISTRIES", "registry.corp.internal, other.host"
    )
    assert image_is_trusted("registry.corp.internal/base:1") is True
    assert image_is_trusted("other.host/base:1") is True
    assert image_is_trusted("evil.example.com/x:1") is False


def test_trusted_registries_config_key_is_bridged():
    """The config.yaml key must reach terminal_tool through every entry point.

    A key present in DEFAULT_CONFIG but missing from a bridge map silently
    does nothing for that entry point — the bug class documented in
    tests/tools/test_terminal_config_env_sync.py.
    """
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert "trusted_image_registries" in DEFAULT_CONFIG["terminal"]
    assert (
        TERMINAL_CONFIG_ENV_MAP["trusted_image_registries"]
        == "TERMINAL_TRUSTED_IMAGE_REGISTRIES"
    )


def test_config_value_reaches_the_checker(monkeypatch):
    """End-to-end through the documented mechanism: config value → warning gone."""
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP

    configured = {"terminal": {"trusted_image_registries": ["registry.corp.internal"]}}
    # Mirror what the startup bridges do with a list value.
    for cfg_key, env_var in TERMINAL_CONFIG_ENV_MAP.items():
        if cfg_key in configured["terminal"]:
            monkeypatch.setenv(env_var, json.dumps(configured["terminal"][cfg_key]))

    assert image_is_trusted("registry.corp.internal/python:3.11") is True


# ── Warning behaviour ─────────────────────────────────────────────────────────

def test_untrusted_image_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger="tools.terminal_tool"):
        _check_image_provenance("docker_image", "evil.example.com/x:1")
        _check_image_provenance("docker_image", "evil.example.com/x:1")

    warnings = [r for r in caplog.records if "evil.example.com" in r.getMessage()]
    assert len(warnings) == 1, "repeated registrations must not spam the log"


def test_trusted_image_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="tools.terminal_tool"):
        _check_image_provenance("docker_image", "ghcr.io/nousresearch/hermes:latest")
    assert not caplog.records


# ── The live paths ────────────────────────────────────────────────────────────

def test_task_env_override_is_checked(caplog):
    """The infra override registry — one of the paths a prompt check misses."""
    task_id = "test-provenance-override"
    try:
        with caplog.at_level(logging.WARNING, logger="tools.terminal_tool"):
            register_task_env_overrides(
                task_id, {"modal_image": "evil.example.com/backdoor:latest"}
            )
        assert any("evil.example.com" in r.getMessage() for r in caplog.records)
    finally:
        from tools.terminal_tool import clear_task_env_overrides

        clear_task_env_overrides(task_id)


def test_task_env_override_still_registers_the_value():
    """Warn-only: an unrecognized registry must not be silently dropped."""
    from tools.terminal_tool import _task_env_overrides, clear_task_env_overrides

    task_id = "test-provenance-passthrough"
    try:
        register_task_env_overrides(task_id, {"docker_image": "evil.example.com/x:1"})
        assert _task_env_overrides[task_id]["docker_image"] == "evil.example.com/x:1"
    finally:
        clear_task_env_overrides(task_id)


def test_env_var_image_is_checked(monkeypatch, caplog):
    """TERMINAL_*_IMAGE — the path that replaced the removed setup prompts."""
    from tools.terminal_tool import _get_env_config

    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "evil.example.com/backdoor:latest")
    with caplog.at_level(logging.WARNING, logger="tools.terminal_tool"):
        config = _get_env_config()

    assert any("evil.example.com" in r.getMessage() for r in caplog.records)
    # Warn-only: the configured image is still honoured.
    assert config["docker_image"] == "evil.example.com/backdoor:latest"


def test_default_config_is_silent(monkeypatch, caplog):
    for var in (
        "TERMINAL_DOCKER_IMAGE",
        "TERMINAL_SINGULARITY_IMAGE",
        "TERMINAL_MODAL_IMAGE",
        "TERMINAL_DAYTONA_IMAGE",
    ):
        monkeypatch.delenv(var, raising=False)

    from tools.terminal_tool import _get_env_config

    with caplog.at_level(logging.WARNING, logger="tools.terminal_tool"):
        _get_env_config()
    assert not [r for r in caplog.records if "trusted set" in r.getMessage()]
