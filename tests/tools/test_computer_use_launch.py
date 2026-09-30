"""Launch -> window -> input through the registry and a real MCP subprocess.

Regression for #51132 and the unbound launch target in #121852. The external
driver is a deterministic protocol fixture; Hermes imports, approvals, binary
resolution, stdio transport, parsing, dispatch and target state are real.
"""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest


_DRIVER = '''import json, sys
from pathlib import Path

config = json.loads(Path(CONFIG).read_text())
if sys.argv[1:] == ["manifest"]:
    print(json.dumps({
        "binary_version": "0.28.2",
        "mcp_invocation": {"command": sys.executable, "args": [__file__, "mcp"]},
        "subcommands": [
            {"name": "mcp", "args": [{"name": "--socket"}, {"name": "--grant"}]},
            {"name": "serve", "args": [{"name": flag} for flag in
                ["--socket", "--permission-mode", "--capability-manifest",
                 "--approve-capability-manifest", "--embedded"]]},
            {"name": "stop", "args": [{"name": "--socket"}]},
        ],
    }))
    sys.exit(0)

properties = {key: {} for key in ["session", "name", "bundle_id", "path", "aumid",
    "launch_path", "additional_arguments", "start_minimized"]}
if config["case"] == "unsupported":
    properties.pop("start_minimized")
tools = [{"name": name, "inputSchema": {"type": "object", "properties":
    properties if name == "launch_app" else {}, "additionalProperties": name != "launch_app"}}
    for name in ["start_session", "end_session", "set_config", "set_agent_cursor_enabled",
                 "launch_app", "list_windows", "list_apps", "get_window_state", "type_text"]
    if name != "launch_app" or config["case"] != "missing"]
polls = 0
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"],
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "launch-fixture", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": tools}
    elif method == "tools/call":
        name, args = request["params"]["name"], request["params"].get("arguments", {})
        with Path(config["log"]).open("a") as log:
            log.write(json.dumps({"name": name, "args": args}) + "\\n")
        case, error, data = config["case"], False, {}
        if name == "launch_app":
            assert not (args.keys() - properties.keys()), args
            data = {"pid": 0 if case in ("metadata", "ambiguous") else 42, "name": "SharedName",
                    "windows": [{"window_id": 7}] if case == "returned" else [],
                    "launch_state": "request_sent", "message": "request accepted"}
            if case == "launch_error":
                error, data = True, {"code": "not_found", "message": "executable not found"}
            if case == "unknown":
                error, data = True, {"code": "transport_outcome_unknown", "message": "launch outcome unknown"}
            if case == "wrong_owner":
                data["windows"] = [{"pid": 111, "window_id": 222}]
        elif name == "list_apps":
            data = {"apps": [{"pid": 42, "running": True, "name": "Localized Editor",
                              config["key"]: config["target"]}]}
            if case == "ambiguous":
                data["apps"].append({"pid": 43, "running": True, config["key"]: config["target"]})
        elif name == "list_windows":
            polls += 1
            data = {"windows": [{"pid": 111, "window_id": 222, "app_name": "SharedName",
                                  "z_index": 100, "is_on_screen": True}]}
            if case in ("delayed", "metadata", "pending") and polls >= 2:
                data["windows"].append({"pid": 42, "window_id": 7, "app_name": "SharedName",
                                        "z_index": 1, "is_on_screen": False})
            if case == "discovery_error":
                error, data = True, {"code": "permission_denied", "message": "window discovery denied"}
        elif name == "get_window_state":
            data = {"elements": [{"element_index": 1, "role": "AXTextField",
                                  "label": "Editor", "element_token": "fresh-token"}],
                    "tree_markdown": "[1] AXTextField", "app_name": "SharedName"}
        result = {"content": [{"type": "text", "text": json.dumps(data)}],
                  "structuredContent": data, "isError": error}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
'''


@pytest.fixture
def launch_driver(tmp_path, monkeypatch, grant_computer_use_approvals):
    from hermes_constants import get_hermes_home
    import tools.computer_use_tool  # register the production entry point
    from tools.computer_use import tool
    from tools.registry import registry

    tool.reset_backend_for_tests()
    config, log = tmp_path / "driver.json", tmp_path / "calls.jsonl"
    script = tmp_path / "cua-driver.py"
    script.write_text(f"#!{sys.executable}\n" + _DRIVER.replace("CONFIG", repr(str(config))), encoding="utf-8")
    script.chmod(0o755)
    executable = script
    if sys.platform == "win32":
        executable = tmp_path / "cua-driver.cmd"
        executable.write_text("@" + subprocess.list2cmdline([sys.executable, str(script)]) + " %*\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(executable))
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "cua")
    (get_hermes_home() / "config.yaml").write_text(
        "computer_use:\n  no_overlay: false\n  capture_after_mode: ax\n", encoding="utf-8")

    def configure(case, key, target):
        config.write_text(json.dumps({"case": case, "key": key, "target": target, "log": str(log)}), encoding="utf-8")

    def dispatch(**args):
        result = registry.dispatch("computer_use", args)
        return json.loads(result) if isinstance(result, str) else result

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    yield configure, dispatch, calls
    tool.reset_backend_for_tests()


@pytest.mark.parametrize("case,key,target,capture_after", [
    ("returned", "bundle_id", "com.example.editor", True),
    ("returned", "name", "SharedName", False),
    ("delayed", "name", "SharedName", True),
    ("delayed", "path", r"C:\Program Files\Editor\editor.exe", True),
    ("delayed", "aumid", "Example.Editor_abc!App", True),
    ("metadata", "launch_path", "/opt/My Editor/editor", True),
])
def test_launch_binds_only_the_launched_window(launch_driver, case, key, target, capture_after):
    from tools.computer_use import tool

    configure, dispatch, calls = launch_driver
    configure(case, key, target)
    initial = dispatch(action="capture", mode="ax", pid=111, window_id=222)
    assert "error" not in initial, initial
    backend = tool._get_backend()
    assert backend._snapshot_tokens == {1: "fresh-token"}

    result = dispatch(action="launch_app", **{key: target},
                      additional_arguments=["--literal", "two words", "$(do-not-expand)"],
                      capture_after=capture_after, wait_timeout=3)
    assert result["ok"] and result["meta"]["window_ready"] is True
    assert result["meta"]["pid"] == 42 and result["meta"]["window_id"] == 7
    assert backend._last_target == {"pid": 42, "window_id": 7}
    if not capture_after:
        assert backend._snapshot_tokens == {}
        dispatch(action="capture", mode="ax", pid=42, window_id=7)
    assert dispatch(action="type", text="hello")["ok"]
    trace = calls()
    launch = [call["args"] for call in trace if call["name"] == "launch_app"]
    assert len(launch) == 1
    assert launch[0][key] == target
    assert launch[0]["additional_arguments"] == ["--literal", "two words", "$(do-not-expand)"]
    assert "wait_timeout" not in launch[0]
    assert [(call["args"]["pid"], call["args"]["window_id"]) for call in trace
            if call["name"] in ("get_window_state", "type_text")] == [(111, 222), (42, 7), (42, 7)]
    if case != "returned":
        assert [call for call in trace if call["name"] == "list_windows"]
        assert all(call["args"].get("on_screen_only") is False for call in trace if call["name"] == "list_windows")


@pytest.mark.parametrize("case", ["pending", "discovery_error", "launch_error", "denied", "missing", "unsupported",
                                  "ambiguous", "wrong_owner", "unknown", "invalid_timeout"])
def test_unready_or_refused_launch_cannot_retarget_input(launch_driver, monkeypatch, case):
    from tools.computer_use import tool

    configure, dispatch, calls = launch_driver
    configure(case, "name", "SharedName")
    initial = dispatch(action="capture", mode="ax", pid=111, window_id=222)
    assert "error" not in initial, initial
    backend = tool._get_backend()
    # Expire the discovery budget after one unsuccessful poll without racing
    # wall-clock scheduling; actual MCP calls retain a >= 2s timeout.
    clock = {"now": 0.0}
    monkeypatch.setitem(backend.launch_app.__func__.__globals__, "time", SimpleNamespace(
        monotonic=lambda: clock["now"], sleep=lambda delay: clock.update(now=4.0)))
    before = len(calls())
    if case == "denied":
        tool.set_approval_callback(lambda *args, **kwargs: "deny")
    result = dispatch(action="launch_app", name="SharedName", capture_after=True,
                      wait_timeout=-1 if case == "invalid_timeout" else 3,
                      **({"start_minimized": True} if case == "unsupported" else {}))
    trace = calls()[before:]
    assert not any(call["name"] == "get_window_state" for call in trace)
    launches = [call for call in trace if call["name"] == "launch_app"]
    assert len(launches) == (0 if case in ("denied", "missing", "unsupported", "invalid_timeout") else 1)
    if case in ("pending", "discovery_error", "ambiguous", "wrong_owner", "unknown"):
        if case == "unknown":
            assert result["ok"] is False and result["code"] == "transport_outcome_unknown"
        else:
            assert result["ok"] and result["meta"]["window_ready"] is False
            assert result["code"] == ("window_discovery_failed" if case == "discovery_error" else "window_not_ready")
        assert backend._last_target is None and backend._snapshot_tokens == {}
        tool.set_approval_callback(lambda *args, **kwargs: "once")
        assert dispatch(action="type", text="must not land")["ok"] is False
        assert not any(call["name"] == "type_text" for call in calls()[before:])
        if case == "pending":
            found = dispatch(action="list_windows", on_screen_only=False, pid=result["meta"]["pid"])
            assert [(w["pid"], w["window_id"]) for w in found["windows"]] == [(42, 7)]
            dispatch(action="capture", mode="ax", pid=42, window_id=7)
            assert dispatch(action="type", text="recovered")["ok"]
            assert len([call for call in calls()[before:] if call["name"] == "launch_app"]) == 1
    else:
        assert result.get("ok") is not True
        assert backend._last_target == {"pid": 111, "window_id": 222}
        assert backend._snapshot_tokens == {1: "fresh-token"}
        if case != "denied":
            assert result["code"] == {"launch_error": "not_found", "missing": "launch_app_unsupported",
                                       "unsupported": "launch_parameters_unsupported", "invalid_timeout": "bad_wait_timeout"}[case]
