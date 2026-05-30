import pytest

from tools.openhands_bridge import (
    DEFAULT_SERVER_URL,
    OpenHandsBridge,
    OpenHandsBridgeError,
    OpenHandsSession,
    openhands_server_status,
    read_openhands_server_metadata,
    start_openhands_server,
    stop_openhands_server,
)


class _FakeProcess:
    pid = 424242


def test_openhands_bridge_discovery_without_server_is_not_launch_supported(monkeypatch):
    for name in ("OPENHANDS_AGENT_SERVER_URL", "OPENHANDS_SERVER_URL", "OPENHANDS_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tools.openhands_bridge.read_openhands_server_metadata", lambda **kwargs: {})
    bridge = OpenHandsBridge(server_url="", command="")

    discovery = bridge.discovery()

    assert discovery["launch_supported"] is False
    assert discovery["configured_mode"] in {"missing", "sdk", "cli"}
    assert discovery["setup_warning"]


def test_openhands_bridge_spawn_without_server_fails_cleanly(monkeypatch):
    for name in ("OPENHANDS_AGENT_SERVER_URL", "OPENHANDS_SERVER_URL", "OPENHANDS_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tools.openhands_bridge.read_openhands_server_metadata", lambda **kwargs: {})
    bridge = OpenHandsBridge(server_url="", command="")

    with pytest.raises(OpenHandsBridgeError) as exc:
        bridge.spawn(project_id="OrynWorkspace", prompt="Inspect only.")

    assert "OpenHands" in str(exc.value)


def test_openhands_server_status_missing_cli_reports_install_instruction(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.openhands_bridge.shutil.which", lambda _cmd: None)

    status = openhands_server_status(metadata_path=tmp_path / "server.json")

    assert status["status"] == "missing_cli"
    assert status["ok"] is False
    assert "uv tool install openhands --python 3.12" in status["install_instruction"]


def test_openhands_server_start_records_metadata_and_stop_clears_it(tmp_path, monkeypatch):
    metadata_path = tmp_path / "server.json"
    monkeypatch.setattr("tools.openhands_bridge.shutil.which", lambda _cmd: "/usr/local/bin/openhands")
    monkeypatch.setattr("tools.openhands_bridge.subprocess.Popen", lambda *args, **kwargs: _FakeProcess())
    monkeypatch.setattr("tools.openhands_bridge._pid_running", lambda pid: bool(pid))
    monkeypatch.setattr(
        OpenHandsBridge,
        "runtime_health",
        lambda self, session: {"runtime_health": "ok", "runtime_warning": None, "configured_mode": "server"},
    )

    started = start_openhands_server(
        cwd=str(tmp_path),
        metadata_path=metadata_path,
        wait_seconds=0,
    )
    metadata = read_openhands_server_metadata(metadata_path=metadata_path)

    assert started["status"] == "running"
    assert metadata["pid"] == _FakeProcess.pid
    assert metadata["server_url"].startswith("http://127.0.0.1:")
    assert metadata["argv"][1:3] == ["-m", "openhands.agent_server"]
    assert "--port" in metadata["argv"]
    assert metadata["server_mode"] == "agent_server"

    monkeypatch.setattr("tools.openhands_bridge._pid_running", lambda pid: False)
    stopped = stop_openhands_server(metadata_path=metadata_path)

    assert stopped["status"] == "stopped"
    assert read_openhands_server_metadata(metadata_path=metadata_path) == {}


def test_openhands_bridge_server_methods_normalize_session_and_events():
    class FakeServerBridge(OpenHandsBridge):
        def __init__(self):
            super().__init__(server_url="http://127.0.0.1:3000")
            self.requests = []

        def _request(self, method, path, payload=None, *, timeout=None):
            self.requests.append((method, path, payload))
            if path == "/health":
                return {"status": "ok"}
            if path == "/server_info":
                return {"title": "OpenHands Agent Server"}
            if method == "POST" and path == "/conversations":
                return {
                    "conversation": {
                        "id": "oh-1",
                        "project_id": payload["tags"]["project"],
                        "status": "running",
                        "workspace_path": "/workspace",
                    }
                }
            if method == "POST" and path == "/conversations/oh-1/events":
                return {"ok": True}
            if method == "GET" and path == "/conversations/oh-1":
                return {"id": "oh-1", "project_id": "OrynWorkspace", "status": "finished"}
            if method == "GET" and path == "/conversations/oh-1/agent_final_response":
                return {"final_response": "PHASE19_OPENHANDS_DEV_PLAN_DONE"}
            if method == "GET" and path == "/conversations":
                return {"conversations": [{"id": "oh-1", "project_id": "OrynWorkspace", "status": "finished"}]}
            if method == "GET" and path == "/conversations/oh-1/events/search?limit=2":
                return {"items": [{"message": "line one"}, {"content": "line two"}]}
            if method == "GET" and path == "/conversations/oh-1/events/search?limit=80":
                return {"items": [{"message": "line one"}, {"content": "line two"}]}
            if method == "DELETE" and path == "/conversations/oh-1":
                return {"ok": True}
            raise AssertionError((method, path, payload))

    bridge = FakeServerBridge()

    spawned = bridge.spawn(project_id="OrynWorkspace", prompt="Inspect files.", model="gpt-5.5")
    status = bridge.status("oh-1")
    sessions = bridge.list(project_id="OrynWorkspace")
    tail = bridge.capture_output(spawned, lines=2)
    bridge.kill("oh-1")

    assert spawned.id == "oh-1"
    assert spawned.event_fields()["runtime"] == "openhands"
    assert status.display_status == "completed"
    assert status.summary == "PHASE19_OPENHANDS_DEV_PLAN_DONE"
    assert "PHASE19_OPENHANDS_DEV_PLAN_DONE" in status.output_tail
    assert len(sessions) == 1
    assert tail == "line two\nPHASE19_OPENHANDS_DEV_PLAN_DONE"
    create_request = next(payload for method, path, payload in bridge.requests if method == "POST" and path == "/conversations")
    assert create_request["initial_message"]["content"][0]["text"] == "Inspect files."
    assert create_request["initial_message"]["run"] is True


@pytest.mark.parametrize(
    ("raw_status", "display_status"),
    [
        ("running", "running"),
        ("finished", "completed"),
        ("completed", "completed"),
        ("done", "completed"),
        ("failed", "failed"),
        ("error", "failed"),
        ("killed", "failed"),
    ],
)
def test_openhands_session_status_mapping(raw_status, display_status):
    session = OpenHandsSession.from_payload({"id": "oh-1", "status": raw_status})

    assert session.display_status == display_status
    assert session.event_fields()["runtime"] == "openhands"
    assert session.event_fields()["runtime_session_id"] == "oh-1"
