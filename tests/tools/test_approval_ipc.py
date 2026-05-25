import json
import threading
import time

from tools import approval


def _clear_ipc_env(monkeypatch):
    for name in (
        "HERMES_APPROVAL_EVENT_PATH",
        "HERMES_APPROVAL_DECISION_PATH",
        "HERMES_APPROVAL_IPC_TIMEOUT",
        "HERMES_GATEWAY_SESSION",
        "HERMES_EXEC_ASK",
        "HERMES_INTERACTIVE",
        "HERMES_YOLO_MODE",
        "HERMES_SESSION_KEY",
        "HERMES_APPROVAL_SKIP_TIRITH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_gateway_ask_without_callback_writes_ipc_event_and_uses_decision(tmp_path, monkeypatch):
    _clear_ipc_env(monkeypatch)
    event_path = tmp_path / "approval.jsonl"
    decision_path = tmp_path / "decision.json"
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "agent-mesh:testp:ipc-approve")
    monkeypatch.setenv("HERMES_APPROVAL_SKIP_TIRITH", "1")
    monkeypatch.setenv("HERMES_APPROVAL_EVENT_PATH", str(event_path))
    monkeypatch.setenv("HERMES_APPROVAL_DECISION_PATH", str(decision_path))
    monkeypatch.setenv("HERMES_APPROVAL_IPC_TIMEOUT", "2")

    def approve_later():
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if event_path.exists() and event_path.read_text().strip():
                decision_path.write_text(json.dumps({"choice": "once"}), encoding="utf-8")
                return
            time.sleep(0.02)

    thread = threading.Thread(target=approve_later)
    thread.start()
    try:
        result = approval.check_all_command_guards("rm -rf /tmp/test-exec-ask", "local")
    finally:
        thread.join(timeout=1)
        approval.clear_session("agent-mesh:testp:ipc-approve")

    assert result["approved"] is True
    event = json.loads(event_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["type"] == "approval_required"
    assert event["session_key"] == "agent-mesh:testp:ipc-approve"
    assert event["command"] == "rm -rf /tmp/test-exec-ask"
    assert event["pattern_key"] in {"delete in root path", "recursive force remove"}


def test_gateway_ask_without_callback_ipc_deny_blocks(tmp_path, monkeypatch):
    _clear_ipc_env(monkeypatch)
    event_path = tmp_path / "approval.jsonl"
    decision_path = tmp_path / "decision.json"
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "agent-mesh:testp:ipc-deny")
    monkeypatch.setenv("HERMES_APPROVAL_SKIP_TIRITH", "1")
    monkeypatch.setenv("HERMES_APPROVAL_EVENT_PATH", str(event_path))
    monkeypatch.setenv("HERMES_APPROVAL_DECISION_PATH", str(decision_path))
    monkeypatch.setenv("HERMES_APPROVAL_IPC_TIMEOUT", "2")

    def deny_later():
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if event_path.exists() and event_path.read_text().strip():
                decision_path.write_text(json.dumps({"choice": "deny"}), encoding="utf-8")
                return
            time.sleep(0.02)

    thread = threading.Thread(target=deny_later)
    thread.start()
    try:
        result = approval.check_all_command_guards("rm -rf /tmp/test-exec-ask", "local")
    finally:
        thread.join(timeout=1)
        approval.clear_session("agent-mesh:testp:ipc-deny")

    assert result["approved"] is False
    assert "denied" in result["message"]
    assert event_path.exists()
