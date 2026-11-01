"""Normal prompt.submit admission is the only recovery execution path."""

from contextlib import contextmanager

from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry
from tui_gateway import methods_prompt


SESSION = "recovery-path-session"
COMMAND = "echo exact displayed command"


def test_recovery_invokes_named_prompt_submit_and_keeps_old_approval_terminal(monkeypatch):
    approval._approval_recovery_records.clear()
    approval._approval_recovery_used_tokens.clear()
    entry = _ApprovalEntry({
        "request_id": "recovery-path-request",
        "command": COMMAND,
        "command_digest": approval._approval_command_digest(COMMAND),
        "execution_generation": 3,
        "turn_id": "turn-3",
    })
    approval._record_approval_recovery(entry, SESSION, "expired")
    calls = []

    monkeypatch.setattr(methods_prompt, "_sess", lambda params, rid: ({
        "session_key": SESSION,
        "auth_user_id": "principal",
    }, None), raising=False)
    @contextmanager
    def no_db(_session):
        yield None
    monkeypatch.setattr(methods_prompt, "_session_db", no_db, raising=False)
    monkeypatch.setattr(methods_prompt, "_ok", lambda rid, result: {"result": result}, raising=False)
    monkeypatch.setattr(methods_prompt, "_err", lambda rid, code, message: {"error": {"code": code, "message": message}}, raising=False)

    def normal_submit(rid, params):
        calls.append((rid, params))
        return {"result": {"status": "streaming", "user_row_id": 41}}
    monkeypatch.setattr(methods_prompt, "prompt_submit", normal_submit)

    response = methods_prompt.approval_recover("rpc-1", {
        "session_id": SESSION,
        "request_id": entry.data["request_id"],
        "command_digest": entry.data["command_digest"],
        "recovery_token": entry.data["recovery_token"],
        "execution_generation": 3,
        "turn_id": "turn-3",
    })

    assert response["result"]["status"] == "streaming"
    assert len(calls) == 1
    assert calls[0][1]["text"].endswith("\n\n" + COMMAND)
    assert entry.cancelled is None
    assert approval._approval_recovery_records[entry.data["request_id"]]["status"] == "streaming"
