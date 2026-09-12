"""Display projection cannot change the raw result used by a turn receipt or goal."""

import threading
from types import SimpleNamespace

from tui_gateway import server


def test_complete_payload_fences_only_display_copy(monkeypatch):
    raw = "Visible <memory-context>PRIVATE_TURN</memory-context> answer"
    receipts = []
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "render_message", lambda text, _cols: "rendered:" + text)
    monkeypatch.setattr(server, "_clear_inflight_turn", lambda _session: None)
    monkeypatch.setattr(server, "_retire_turn_marker", lambda *_args: None)
    st = SimpleNamespace(result={"final_response": raw}, agent=SimpleNamespace(),
                         terminal_callback=receipts.append, receipt_attempted=False,
                         receipt_committed=False, marker_key=None)
    payload, returned_raw, status = server._complete_turn_payload(
        {"history_lock": threading.RLock()}, st, None, 80)
    assert status == "complete"
    assert "PRIVATE_TURN" not in repr(payload)
    assert payload["text"] == "Visible  answer"
    assert returned_raw == raw
    assert receipts[0]["text"] == raw
    assert st.result["final_response"] == raw
