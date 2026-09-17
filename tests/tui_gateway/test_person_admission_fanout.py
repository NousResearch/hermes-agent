"""Does person.admission fan out to every attached viewer, or only the sidecar?"""

from __future__ import annotations

import threading

from agent.turn_authorization import TurnAuthorization
from tui_gateway import server as srv
from tui_gateway.transport import FanoutTransport


class Sink:
    def __init__(self, name):
        self.name = name
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        return None


def test_person_admission_reaches_every_attached_viewer():
    sidecar, browser = Sink("sidecar"), Sink("browser")
    fanout = FanoutTransport(sidecar, browser)
    session = {
        "history_lock": threading.Lock(),
        "transport": fanout,
        "session_key": "k",
    }
    sid = "fanout-probe"
    srv._sessions[sid] = session
    try:
        holder = TurnAuthorization.from_raw(
            "bearer-value", expires_at=9e9, principal_id="a" * 64,
            admission_id="7" * 32,
        )
        # prompt.submit (the only authority ingress) pins the peer that supplied
        # the authority; admission events go to that peer alone.
        holder._fizko_bind_peer(sidecar)
        srv._emit_person_admission(sid, holder, "started")
        for _ in range(200):
            if browser.frames or sidecar.frames:
                break
            threading.Event().wait(0.01)
        threading.Event().wait(0.3)
    finally:
        srv._sessions.pop(sid, None)

    got = {
        s.name: [f["params"]["payload"]["admission_id"]
                 for f in s.frames if f["params"].get("type") == "person.admission"]
        for s in (sidecar, browser)
    }
    # No bearer material anywhere.
    assert "bearer-value" not in repr(got) + repr(sidecar.frames) + repr(browser.frames)
    assert got["sidecar"] == ["7" * 32]
    assert got["browser"] == [], (
        "the private admission id was delivered to a second attached viewer too: "
        f"{got}"
    )
