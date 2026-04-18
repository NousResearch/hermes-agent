import json

import pytest

import tools.codex_app_server_tool as tool


class FakeBridge:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.turns = []

    def start(self, *, timeout):
        self.started = True
        return {"bridge_status": "ready", "timeout": timeout}

    def stop(self):
        self.stopped = True
        return {"bridge_status": "stopped"}

    def get_status(self):
        return {"bridge_status": "ready", "normalized_status": "idle"}

    def get_recent_events(self, limit):
        return [{"raw_method": "codex/example", "limit": limit}]

    def start_turn(self, *, repo_path, prompt, timeout):
        self.turns.append((repo_path, prompt, timeout))
        return {
            "thread_id": "thread-1",
            "turn_id": "turn-1",
            "status": {"normalized_status": "running"},
        }


@pytest.fixture(autouse=True)
def reset_bridge_singleton():
    tool._set_bridge_for_tests(None)
    yield
    tool._set_bridge_for_tests(None)


def test_unknown_action_returns_json_error():
    result = json.loads(tool.codex_app_server_tool(action="bogus"))

    assert "error" in result
    assert "Unknown action" in result["error"]


def test_start_bridge_returns_success_status(monkeypatch):
    fake = FakeBridge()
    monkeypatch.setattr(tool, "CodexAppServerBridge", lambda: fake)

    result = json.loads(tool.codex_app_server_tool(action="start_bridge", timeout_seconds=1.5))

    assert result == {
        "success": True,
        "action": "start_bridge",
        "status": {"bridge_status": "ready", "timeout": 1.5},
    }
    assert fake.started is True


def test_status_without_bridge_is_stopped_shape():
    result = json.loads(tool.codex_app_server_tool(action="status"))

    assert result["success"] is True
    assert result["action"] == "status"
    assert result["status"] == {"bridge_status": "stopped"}
    assert result["methods"] == {
        "start_thread": "thread/start",
        "start_turn": "turn/start",
    }


def test_events_returns_limited_recent_events():
    fake = FakeBridge()
    tool._set_bridge_for_tests(fake)

    result = json.loads(tool.codex_app_server_tool(action="events", limit=7))

    assert result["success"] is True
    assert result["count"] == 1
    assert result["events"] == [{"raw_method": "codex/example", "limit": 7}]


def test_start_turn_validates_required_fields():
    tool._set_bridge_for_tests(FakeBridge())

    missing_repo = json.loads(tool.codex_app_server_tool(action="start_turn", prompt="hi"))
    missing_prompt = json.loads(tool.codex_app_server_tool(action="start_turn", repo_path="/tmp/repo"))

    assert missing_repo["error"] == "repo_path is required for start_turn"
    assert missing_prompt["error"] == "prompt is required for start_turn"


def test_start_turn_requires_started_bridge():
    result = json.loads(
        tool.codex_app_server_tool(
            action="start_turn",
            repo_path="/tmp/repo",
            prompt="hi",
        )
    )

    assert result["error"] == "Bridge is not started. Call start_bridge first."


def test_start_turn_returns_bridge_result():
    fake = FakeBridge()
    tool._set_bridge_for_tests(fake)

    result = json.loads(
        tool.codex_app_server_tool(
            action="start_turn",
            repo_path="/tmp/repo",
            prompt="hi",
            timeout_seconds=2,
        )
    )

    assert result["success"] is True
    assert result["action"] == "start_turn"
    assert result["thread_id"] == "thread-1"
    assert result["turn_id"] == "turn-1"
    assert result["result"]["thread_id"] == "thread-1"
    assert result["result"]["turn_id"] == "turn-1"
    assert fake.turns == [("/tmp/repo", "hi", 2.0)]


def test_start_job_starts_bridge_and_turn_in_one_call(monkeypatch):
    fake = FakeBridge()
    monkeypatch.setattr(tool, "CodexAppServerBridge", lambda: fake)

    result = json.loads(
        tool.codex_app_server_tool(
            action="start_job",
            repo_path="/tmp/repo",
            prompt="do the work",
            timeout_seconds=3,
        )
    )

    assert result["success"] is True
    assert result["action"] == "start_job"
    assert result["thread_id"] == "thread-1"
    assert result["turn_id"] == "turn-1"
    assert result["status"] == {"normalized_status": "running"}
    assert result["bridge_start_status"] == {"bridge_status": "ready", "timeout": 3.0}
    assert result["result"]["thread_id"] == "thread-1"
    assert fake.started is True
    assert fake.turns == [("/tmp/repo", "do the work", 3.0)]


def test_start_job_validates_required_fields():
    missing_repo = json.loads(tool.codex_app_server_tool(action="start_job", prompt="hi"))
    missing_prompt = json.loads(tool.codex_app_server_tool(action="start_job", repo_path="/tmp/repo"))

    assert missing_repo["error"] == "repo_path is required for start_job"
    assert missing_prompt["error"] == "prompt is required for start_job"
