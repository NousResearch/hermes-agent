from tools import approval


def test_persist_choice_records_each_session_approval(monkeypatch):
    approved = []
    monkeypatch.setattr(approval, "approve_session", lambda *args: approved.append(args))
    approval._persist_choice("session", "session", ["dangerous:rm"])
    assert approved == [("session", "dangerous:rm")]
