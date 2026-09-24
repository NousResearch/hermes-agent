from tools import approval


def test_persist_choice_records_each_session_approval(monkeypatch):
    recorded = []
    monkeypatch.setattr(approval, "approve_session", lambda *_args: None)
    monkeypatch.setattr(approval, "_record_decision_ledger", lambda **kwargs: recorded.append(kwargs))
    approval._persist_choice("session", "session", [("dangerous:rm", "removes files", False)])
    assert recorded == [{"session_key": "session", "kind": "approval", "text": "session approval: removes files"}]
