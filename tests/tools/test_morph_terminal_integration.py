"""Tests for Morph terminal backend integration."""

import importlib
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

tools_pkg = ModuleType("tools")
tools_pkg.__path__ = [str(_repo_root / "tools")]
sys.modules["tools"] = tools_pkg

terminal_module = importlib.import_module("tools.terminal_tool")
file_tools_module = importlib.import_module("tools.file_tools")


def test_get_env_config_sanitizes_host_cwd_for_morph():
    with patch.dict(
        os.environ,
        {
            "TERMINAL_ENV": "morph",
            "TERMINAL_CWD": "/home/tester/project",
        },
        clear=True,
    ):
        config = terminal_module._get_env_config()
        assert config["cwd"] == "/root"
        assert config["morph_image_id"] == "morphvm-minimal"


def test_get_env_config_defaults_cwd_to_root_for_morph():
    with patch.dict(
        os.environ,
        {
            "TERMINAL_ENV": "morph",
        },
        clear=True,
    ):
        config = terminal_module._get_env_config()
        assert config["cwd"] == "/root"
        assert config["morph_image_id"] == "morphvm-minimal"


def test_create_environment_constructs_morph_backend():
    with patch("tools.environments.morph.MorphEnvironment") as mock_env:
        terminal_module._create_environment(
            env_type="morph",
            image="morphvm-minimal",
            cwd="/root",
            timeout=45,
            container_config={
                "container_cpu": 2.5,
                "container_memory": 6144,
                "container_disk": 20480,
                "container_persistent": True,
                "lifetime_seconds": 900,
            },
            task_id="task-morph",
        )

    mock_env.assert_called_once_with(
        image_id="morphvm-minimal",
        cwd="/root",
        timeout=45,
        cpu=2.5,
        memory=6144,
        disk=20480,
        persistent_filesystem=True,
        task_id="task-morph",
        lifetime_seconds=900,
    )


def test_file_tools_use_morph_image_override(monkeypatch):
    fake_env = object()
    fake_file_ops = object()
    create_calls = []

    monkeypatch.setattr(terminal_module, "_active_environments", {}, raising=False)
    monkeypatch.setattr(terminal_module, "_last_activity", {}, raising=False)
    monkeypatch.setattr(terminal_module, "_creation_locks", {}, raising=False)
    monkeypatch.setattr(terminal_module, "_env_lock", threading.Lock(), raising=False)
    monkeypatch.setattr(
        terminal_module, "_creation_locks_lock", threading.Lock(), raising=False
    )
    monkeypatch.setattr(
        terminal_module,
        "_task_env_overrides",
        {"task-morph": {"morph_image_id": "morphvm-override"}},
        raising=False,
    )
    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_module,
        "_get_env_config",
        lambda: {
            "env_type": "morph",
            "morph_image_id": "morphvm-default",
            "cwd": "/root",
            "timeout": 60,
            "lifetime_seconds": 444,
            "container_cpu": 1,
            "container_memory": 5120,
            "container_disk": 51200,
            "container_persistent": True,
            "docker_volumes": [],
        },
    )

    def _fake_create_environment(**kwargs):
        create_calls.append(kwargs)
        return fake_env

    monkeypatch.setattr(terminal_module, "_create_environment", _fake_create_environment)
    monkeypatch.setattr(file_tools_module, "_file_ops_cache", {}, raising=False)
    monkeypatch.setattr(file_tools_module, "ShellFileOperations", lambda env: fake_file_ops)

    result = file_tools_module._get_file_ops("task-morph")

    assert result is fake_file_ops
    assert create_calls[0]["image"] == "morphvm-override"
    assert create_calls[0]["container_config"]["lifetime_seconds"] == 444


def test_check_terminal_requirements_for_morph(monkeypatch):
    monkeypatch.setitem(sys.modules, "morphcloud", ModuleType("morphcloud"))
    monkeypatch.setenv("TERMINAL_ENV", "morph")
    monkeypatch.setenv("MORPH_API_KEY", "morph-key")
    monkeypatch.setenv("TERMINAL_MORPH_IMAGE_ID", "morphvm-minimal")

    assert terminal_module.check_terminal_requirements() is True
