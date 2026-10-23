"""action="launch": how a free-form app reaches cua-driver."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use import tool as cu_tool
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.cua_backend_parse import _launch_target_args


@pytest.fixture
def noop_backend(grant_computer_use_approvals):
    cu_tool.reset_backend_for_tests()
    with patch.object(cu_tool, "_new_backend", lambda mode: cu_tool._NoopBackend()):
        yield cu_tool._get_backend()
    cu_tool.reset_backend_for_tests()


def _backend(structured=None) -> CuaDriverBackend:
    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.call_tool.return_value = {"data": "ok", "images": [], "structuredContent": structured,
                                               "isError": False}
    backend._session.supports_capability.return_value = False
    return backend


def test_launch_action_routes_to_backend(noop_backend):
    out = json.loads(cu_tool.handle_computer_use({"action": "launch", "app": "Calculator"}))
    assert out.get("ok") is True
    assert [kw["app"] for name, kw in noop_backend.calls if name == "launch_app"] == ["Calculator"]


def test_launch_action_requires_app(noop_backend):
    out = json.loads(cu_tool.handle_computer_use({"action": "launch"}))
    assert "launch requires `app`" in out["error"]


@pytest.mark.parametrize("bad_action", ["open_app", "start_app", "run_app"])
def test_launch_aliases_name_the_real_action(noop_backend, bad_action):
    out = json.loads(cu_tool.handle_computer_use({"action": bad_action, "app": "Calculator"}))
    assert "did you mean 'launch'" in out["error"]


def test_launch_app_requires_bundle_id_or_name():
    with pytest.raises(ValueError, match="bundle_id or name"):
        _backend().launch_app()


def test_launch_app_sends_the_resolved_field_and_session():
    backend = _backend(structured={"pid": 1234, "name": "Calculator", "windows": [{"window_id": 42}]})
    result = backend.launch_app("Microsoft.WindowsCalculator_8wekyb3d8bbwe!App")
    assert backend._session.call_tool.call_args.args == (
        "launch_app", {"aumid": "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App", "session": backend._session_id})
    assert result.ok and backend._last_app == "Calculator"


@pytest.mark.parametrize("target, platform, expected", [
    ("shell:AppsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App", "win32",
     {"launch_path": "shell:AppsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"}),
    ("Microsoft.WindowsCalculator_8wekyb3d8bbwe!App", "win32", {"aumid": "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"}),
    ("C:\\Windows\\System32\\calc.exe", "win32", {"path": "C:\\Windows\\System32\\calc.exe"}),
    ("calc.exe", "win32", {"name": "calc.exe"}),
    ("com.apple.Safari", "darwin", {"bundle_id": "com.apple.Safari"}),
    ("com.apple.Safari", "linux", {"name": "com.apple.Safari"}),
    ("/usr/bin/gedit", "linux", {"name": "/usr/bin/gedit"}),
    ("Calculator", "win32", {"name": "Calculator"}),
])
def test_launch_target_resolution(target, platform, expected):
    assert _launch_target_args(target, platform) == expected
