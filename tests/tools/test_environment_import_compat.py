"""Regression tests for mixed-version environment helper imports."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER_CONSUMERS = (
    ("tools.environments.file_sync", ("_file_mtime_key",)),
    ("tools.environments.local", ("_pipe_stdin",)),
    ("tools.environments.docker", ("_popen_bash",)),
    ("tools.environments.ssh", ("_popen_bash",)),
    (
        "tools.environments.modal",
        ("_ThreadedProcessHandle", "_load_json_store", "_save_json_store"),
    ),
    ("tools.environments.daytona", ("_ThreadedProcessHandle",)),
    (
        "tools.environments.singularity",
        ("_load_json_store", "_popen_bash", "_save_json_store"),
    ),
)
_BASE_HELPER_NAMES = tuple(
    sorted(
        {
            helper_name
            for _, helper_names in _HELPER_CONSUMERS
            for helper_name in helper_names
        }
    )
)


@pytest.mark.parametrize(("module_name", "helper_names"), _HELPER_CONSUMERS)
def test_environment_module_import_survives_stale_cached_base(
    module_name, helper_names
):
    """New helper consumers must import beside an older cached base module."""
    script = r'''
import importlib
import sys
import types

module_name = sys.argv[1]
helper_names = sys.argv[2:]
stale_base = types.ModuleType("tools.environments.base")
stale_base.BaseEnvironment = type("BaseEnvironment", (), {})
sys.modules["tools.environments.base"] = stale_base

module = importlib.import_module(module_name)
execution_helpers = importlib.import_module("tools.environments.execution_helpers")
assert module is sys.modules[module_name]

for name in helper_names:
    assert getattr(module, name) is getattr(execution_helpers, name), name
'''

    result = subprocess.run(
        [sys.executable, "-c", script, module_name, *helper_names],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_base_reexports_shared_environment_helpers():
    """Legacy imports from base and new direct imports share one implementation."""
    from tools.environments import base, execution_helpers

    for name in _BASE_HELPER_NAMES:
        assert getattr(base, name) is getattr(execution_helpers, name)


@pytest.mark.parametrize(
    "module_order",
    [
        ("tools.environments.base", "tools.environments.execution_helpers"),
        ("tools.environments.execution_helpers", "tools.environments.base"),
    ],
)
def test_base_and_execution_helpers_import_cleanly_in_either_order(module_order):
    """Either import order must preserve base's shared-helper reexports."""
    script = r'''
import importlib
import sys

first_module, second_module, *helper_names = sys.argv[1:]
importlib.import_module(first_module)
importlib.import_module(second_module)

base = sys.modules["tools.environments.base"]
execution_helpers = sys.modules["tools.environments.execution_helpers"]
for name in helper_names:
    assert getattr(base, name) is getattr(execution_helpers, name), name
'''

    result = subprocess.run(
        [sys.executable, "-c", script, *module_order, *_BASE_HELPER_NAMES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("module_name", "helper_names"),
    _HELPER_CONSUMERS,
)
def test_environment_modules_use_shared_helper_contract(module_name, helper_names):
    """Backend imports must not carry divergent local compatibility copies."""
    module = importlib.import_module(module_name)
    execution_helpers = importlib.import_module("tools.environments.execution_helpers")

    for name in helper_names:
        assert getattr(module, name) is getattr(execution_helpers, name)
