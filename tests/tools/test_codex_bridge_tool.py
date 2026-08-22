import json

import tools.codex_bridge_tool as bridge
from tools.codex_bridge_tool import CodexBridgeManager, CodexBridgeStore


class FakeCodexClient:
    instances = []

    def __init__(self, task_id, task, manager):
        self.task_id = task_id
        self.task = task
        self.manager = manager
        self.requests = []
        self.responses = []
        self.closed = False
        FakeCodexClient.instances.append(self)

    def start(self, *, codex_home=None):
        self.codex_home = codex_home

    def initialize(self):
        return {"userAgent": "fake-codex", "codexHome": "/tmp/codex"}

    def request(self, method, params=None, timeout=30):
        self.requests.append((method, params, timeout))
        if method == "thread/start":
            return {"thread": {"id": "thread-1"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-1", "status": "inProgress"}}
        if method == "turn/steer":
            return {"ok": True, "steered": params}
        if method == "turn/interrupt":
            return {"ok": True, "interrupted": params}
        raise AssertionError(f"unexpected request: {method}")

    def notify(self, method, params=None):
        self.notifications = getattr(self, "notifications", [])
        self.notifications.append((method, params))

    def respond(self, request_id, result):
        self.responses.append((request_id, result))

    def close(self):
        self.closed = True


def make_manager(tmp_path, monkeypatch):
    FakeCodexClient.instances.clear()
    monkeypatch.setattr(bridge, "CodexJsonRpcClient", FakeCodexClient)
    store = CodexBridgeStore(tmp_path / "codex_bridge.db")
    return CodexBridgeManager(store=store)


def test_start_task_uses_app_server_thread_turn_without_mailbox(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)

    result = manager.start_task("Investigate the failing test", cwd=str(tmp_path))

    assert result["success"] is True
    assert result["protocol"] == {"transport": "app-server stdio", "mailbox": False}
    task = result["task"]
    assert task["status"] == "working"
    assert task["codex_thread_id"] == "thread-1"
    assert task["codex_turn_id"] == "turn-1"

    client = FakeCodexClient.instances[0]
    methods = [method for method, _params, _timeout in client.requests]
    assert methods == ["thread/start", "turn/start"]
    thread_params = client.requests[0][1]
    assert thread_params["sandbox"] == "read-only"
    assert thread_params["approvalPolicy"] == "untrusted"
    assert "mailbox" not in json.dumps(client.requests).lower()
    assert "outbox" not in json.dumps(client.requests).lower()
    assert "inbox" not in json.dumps(client.requests).lower()


def test_start_task_records_notify_target(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)

    result = manager.start_task("Analyze tests", cwd=str(tmp_path), notify_target="feishu:chat-1")
    task_id = result["task"]["hermes_task_id"]

    assert result["task"]["notify_target"] == "feishu:chat-1"
    assert result["task"]["notification_status"] == "pending"
    persisted = manager.status(task_id)["task"]
    assert persisted["notify_target"] == "feishu:chat-1"
    assert persisted["notification_status"] == "pending"


def test_server_approval_request_can_be_reported_and_resolved(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)
    started = manager.start_task("Run a safe command", cwd=str(tmp_path))
    task_id = started["task"]["hermes_task_id"]
    client = FakeCodexClient.instances[0]

    manager.handle_server_request(
        task_id,
        client,
        {
            "id": "approval-1",
            "method": "item/commandExecution/requestApproval",
            "params": {"threadId": "thread-1", "turnId": "turn-1", "command": "pwd"},
        },
    )

    status = manager.status(task_id)
    assert status["task"]["status"] == "waiting_for_approval"
    assert status["task"]["pending_requests"][0]["request_id"] == "approval-1"

    response = manager.respond(task_id, "approval-1", decision="decline")
    assert response["success"] is True
    assert client.responses == [("approval-1", {"decision": "decline"})]
    assert manager.status(task_id)["task"]["pending_requests"] == []


def test_request_user_input_response_uses_answers_payload(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)
    started = manager.start_task("Ask for missing context", cwd=str(tmp_path))
    task_id = started["task"]["hermes_task_id"]
    client = FakeCodexClient.instances[0]

    manager.handle_server_request(
        task_id,
        client,
        {
            "id": "input-1",
            "method": "item/tool/requestUserInput",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "questions": [{"id": "q1", "question": "Which file?", "options": None}],
            },
        },
    )

    answers = {"q1": {"answers": ["README.md"]}}
    manager.respond(task_id, "input-1", decision="decline", answers=answers)

    assert client.responses == [("input-1", {"answers": answers})]


def test_steer_and_interrupt_call_codex_turn_methods(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)
    started = manager.start_task("Long running task", cwd=str(tmp_path))
    task_id = started["task"]["hermes_task_id"]
    client = FakeCodexClient.instances[0]

    steer = manager.steer(task_id, "Only analyze; do not edit.")
    interrupt = manager.interrupt(task_id)

    assert steer["success"] is True
    assert interrupt["task"]["status"] == "cancelled"
    assert client.requests[-2][0] == "turn/steer"
    assert client.requests[-2][1]["expectedTurnId"] == "turn-1"
    assert client.requests[-1][0] == "turn/interrupt"


def test_notify_completed_sends_once_for_targeted_completed_task(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)
    started = manager.start_task("Summarize a bug", cwd=str(tmp_path), notify_target="feishu:chat-1")
    task_id = started["task"]["hermes_task_id"]
    deliveries = []

    manager.record_event(
        task_id,
        "turn/completed",
        {"turn": {"id": "turn-1", "status": "completed"}, "message": "Done fixing it."},
    )

    first = manager.notify_completed(notifier=lambda target, message: deliveries.append((target, message)) or {"ok": True})
    second = manager.notify_completed(notifier=lambda target, message: deliveries.append((target, message)) or {"ok": True})

    assert first["processed"] == 1
    assert first["notifications"][0]["notification_status"] == "sent"
    assert first["notifications"][0]["sent"] is True
    assert second["processed"] == 0
    assert len(deliveries) == 1
    assert deliveries[0][0] == "feishu:chat-1"
    assert task_id in deliveries[0][1]
    assert manager.status(task_id)["task"]["notification_status"] == "sent"


def test_notify_completed_marks_no_target_without_sending(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)
    started = manager.start_task("No callback needed", cwd=str(tmp_path))
    task_id = started["task"]["hermes_task_id"]

    manager.record_event(
        task_id,
        "turn/completed",
        {"turn": {"id": "turn-1", "status": "completed"}, "message": "Done."},
    )

    result = manager.notify_completed(notifier=lambda _target, _message: (_ for _ in ()).throw(AssertionError("sent")))

    assert result["processed"] == 1
    assert result["notifications"][0]["notification_status"] == "no_target"
    assert result["notifications"][0]["sent"] is False
    assert manager.status(task_id)["task"]["notification_status"] == "no_target"


def test_notify_completed_dry_run_does_not_send_or_mark(tmp_path, monkeypatch):
    manager = make_manager(tmp_path, monkeypatch)
    started = manager.start_task("Preview callback", cwd=str(tmp_path), notify_target="local")
    task_id = started["task"]["hermes_task_id"]

    manager.record_event(
        task_id,
        "turn/completed",
        {"turn": {"id": "turn-1", "status": "completed"}, "message": "Done."},
    )

    result = manager.notify_completed(
        dry_run=True,
        notifier=lambda _target, _message: (_ for _ in ()).throw(AssertionError("sent")),
    )

    assert result["processed"] == 1
    assert result["notifications"][0]["notification_status"] == "dry_run"
    assert result["notifications"][0]["sent"] is False
    assert manager.status(task_id)["task"]["notification_status"] == "pending"


def test_tool_schema_refuses_danger_full_access():
    props = bridge.CODEX_BRIDGE_SCHEMA["parameters"]["properties"]

    assert "danger-full-access" not in props["sandbox"]["enum"]
    assert "never" not in props["approval_policy"]["enum"]
    assert "notify_completed" in props["action"]["enum"]
    assert "notify_target" in props
