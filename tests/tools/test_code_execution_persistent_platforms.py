import json
from unittest.mock import patch

import pytest

from tools.code_execution_kernel import kernel_registry
from tools.code_execution_tool import execute_code


def _run_persistent_smoke():
    config = {
        "mode": "project",
        "kernel_mode": "session",
        "timeout": 10,
        "max_tool_calls": 10,
        "kernel_idle_seconds": 60,
        "max_live_kernels": 8,
    }
    with patch("tools.code_execution_tool._load_config", return_value=config), patch(
        "tools.approval.check_execute_code_guard", return_value={"approved": True}
    ):
        first = json.loads(
            execute_code("value = 41", execution_session_id="platform-smoke")
        )
        second = json.loads(
            execute_code("value + 1", execution_session_id="platform-smoke")
        )
    return first, second


@pytest.fixture(autouse=True)
def _clean_kernels(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    kernel_registry.close_all()
    yield
    kernel_registry.close_all()


@pytest.mark.windows_only
def test_persistent_kernel_smoke_on_windows():
    first, second = _run_persistent_smoke()

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert second["output"].strip() == "42"
    assert second["kernel_reused"] is True


@pytest.mark.macos_only
def test_persistent_kernel_smoke_on_macos():
    first, second = _run_persistent_smoke()

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert second["output"].strip() == "42"
    assert second["kernel_reused"] is True
