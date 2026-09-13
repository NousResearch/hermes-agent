"""Exercise existing computer-use actions on the actual Windows/macOS CI hosts."""

import json
import sys

import pytest

from tools.computer_use import tool


@pytest.fixture
def backend(monkeypatch):
    tool.reset_backend_for_tests()
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    value = tool._get_backend()
    yield value
    tool.reset_backend_for_tests()


@pytest.mark.parametrize("host_platform", [
    pytest.param("win32", marks=pytest.mark.windows_only),
    pytest.param("darwin", marks=pytest.mark.macos_only),
])
@pytest.mark.parametrize("args, expected_call", [
    ({"action": "capture", "mode": "ax"}, "capture"),
    ({"action": "click", "coordinate": [10, 10]}, "click"),
    ({"action": "type", "text": "test input"}, "type"),
    ({"action": "list_windows"}, "list_windows"),
])
def test_native_computer_use_dispatches_to_backend(backend, args, expected_call, host_platform):
    import tools.computer_use_tool  # noqa: F401 - register the real tool handler
    from tools.registry import registry

    assert sys.platform == host_platform
    result = registry.dispatch("computer_use", args)

    assert "error" not in json.loads(result)
    assert [name for name, _ in backend.calls] == [expected_call]
