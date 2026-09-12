"""Bot Desktop invariants: the byte-level RFB input gate follows the lease, and computer_use refuses
every action (capture included) while a human holds the screen."""

from __future__ import annotations

import json

import pytest

from tools.bot_desktop import lease
from tools.bot_desktop.rfb_filter import RfbClientFilter

_HANDSHAKE = b"RFB 003.008\n" + b"\x01" + b"\x00"
_KEY = b"\x04\x01\x00\x00\x00\x00\x00\x61"          # KeyEvent 'a' down
# QEMU Extended KeyEvent (type 255, sub 0): what noVNC sends once Xvnc advertises the pseudo-encoding.
_QEMU_KEY = bytes([255, 0, 0, 1]) + (0x65).to_bytes(4, "big") + (0x12).to_bytes(4, "big")
_POINTER = b"\x05\x01\x00\x10\x00\x10"              # PointerEvent, button 1
_CUT = b"\x06\x00\x00\x00\x00\x00\x00\x02hi"        # ClientCutText "hi"
_FBUR = b"\x03\x00" + b"\x00" * 8                   # FramebufferUpdateRequest
_SETENC = b"\x02\x00\x00\x02" + b"\x00\x00\x00\x07" + b"\xff\xff\xff\x21"  # SetEncodings x2


@pytest.fixture(autouse=True)
def _fresh_lease():
    lease._reset_for_tests()
    yield
    lease._reset_for_tests()


def test_rfb_filter_forwards_input_only_from_the_lease_holder_across_arbitrary_chunking():
    f = RfbClientFilter(lambda: lease.viewer_may_send_input("v1"))
    head = f.feed(_HANDSHAKE)
    assert head[-1:] == b"\x01", "ClientInit is forced shared so a viewer never kicks the agent's watcher"

    # Agent holds: read-only messages pass, input is dropped, even when split byte by byte.
    stream = _KEY + _FBUR + _POINTER + _SETENC + _CUT + _QEMU_KEY
    out = b"".join(f.feed(stream[i:i + 1]) for i in range(len(stream)))
    assert out == _FBUR + _SETENC

    lease.acquire("v1")
    assert f.feed(_KEY + _POINTER + _QEMU_KEY) == _KEY + _POINTER + _QEMU_KEY

    lease.acquire("v2")  # last writer wins: v1 is evicted from input on the very next message
    assert f.feed(_KEY) == b""
    assert lease.release("v1").holder == lease.HUMAN, "a stale viewer's release must not yank control from v2"
    assert lease.release("v2").holder == lease.AGENT


def test_computer_use_refuses_every_action_while_a_human_holds_the_screen(monkeypatch):
    from tools.computer_use import tool

    calls = []
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": calls.append(session_id) or object())
    lease.acquire("human")
    for action in ("capture", "click", "type", "list_windows"):
        res = json.loads(tool.handle_computer_use({"action": action, "text": "pw"}))
        assert res["code"] == "human_has_control", action
    assert calls == [], "the driver is never touched while the human may be typing a credential"

    # Handoff round trip: the agent asks, the human takes over and hands back, the agent is unblocked.
    asked = json.loads(tool.handle_computer_use({"action": "request_handoff", "reason": "log in"}))
    assert asked["ok"] and asked["state"]["pending_handoff"] == "log in"
    lease.acquire("human", reason="log in")
    assert lease.get().pending_handoff is None
    lease.release("human")
    done = json.loads(tool.handle_computer_use({"action": "wait_for_human", "seconds": 1}))
    assert done["ok"] and done["state"]["holder"] == lease.AGENT
