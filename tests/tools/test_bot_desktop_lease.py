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


def test_lease_authority_is_shared_across_processes(tmp_path):
    """The gateway that streams the screen and the process running the agent are different processes;
    a human takeover in one must refuse actions in the other."""
    import os
    import subprocess
    import sys

    lease.acquire("desktop-viewer")
    probe = ("import sys; sys.path.insert(0, %r)\n"
             "from tools.bot_desktop import lease\n"
             "try:\n    lease.assert_agent_may_act(); print('AGENT')\n"
             "except lease.HumanHasControl:\n    print('HUMAN')\n"
             "lease.release('desktop-viewer')\n") % os.getcwd()
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, encoding="utf-8", timeout=30,
                         stdin=subprocess.DEVNULL, env={**os.environ, "HERMES_HOME": os.environ["HERMES_HOME"]})
    assert out.stdout.strip() == "HUMAN", out.stderr
    assert lease.get().holder == lease.AGENT, "the other process's release is visible here"


def test_takeover_during_an_admitted_action_discards_its_result(monkeypatch):
    """Approval / backend start-up can take seconds; a human who takes over meanwhile must not have
    their keystrokes captured by an action admitted before they did."""
    from tools.computer_use import tool

    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": object())

    def _dispatch_then_takeover(backend, action, args, **_):
        lease.acquire("human")  # a whole take-over / hand-back cycle inside the driver call:
        lease.release("human")  # control is back, but the frame is still the human's turn
        return json.dumps({"ok": True, "action": action, "png_b64": "SECRET"})

    monkeypatch.setattr(tool, "_dispatch", _dispatch_then_takeover)
    res = json.loads(tool.handle_computer_use({"action": "capture"}))
    assert res["code"] == "human_has_control" and "SECRET" not in json.dumps(res)


def test_unreadable_lease_file_fails_closed_and_takeover_keeps_the_agents_reason(tmp_path):
    """Missing file = fresh profile (agent). A file that exists but cannot be parsed must not read as
    "agent holds": a torn write must never let the agent act on a human's screen. Taking over after a
    request keeps the agent's reason so the human still sees WHY while they act."""
    from hermes_constants import hermes_home_key

    home = str(tmp_path)
    assert lease.get(profile_key=home).holder == lease.AGENT
    path = lease._path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ torn", encoding="utf-8")
    assert lease.get(profile_key=home).holder == lease.HUMAN
    lease.release(profile_key=home)  # a successful write repairs it
    assert lease.get(profile_key=home).holder == lease.AGENT

    lease.request_handoff("log in to the bank, 2FA on your phone", profile_key=home)
    held = lease.acquire("desk-1", profile_key=home)
    assert held.pending_handoff is None and held.reason == "log in to the bank, 2FA on your phone"
    assert hermes_home_key(home)  # sanity: the key derivation used by the bridge is available


def test_lease_works_without_fcntl(tmp_path):
    """Windows and fcntl-less hosts: ``computer_use`` imports the lease (via handoff) on EVERY call, so a
    module-level fcntl dependency turns every desktop action into ModuleNotFoundError there. The file
    semantics must still work; only the cross-process lock degrades. Subprocess so the module cache is clean."""
    import os
    import subprocess
    import sys

    probe = ("import sys; sys.modules['fcntl'] = None; sys.path.insert(0, %r)\n"
             "from tools.bot_desktop import lease\n"
             "import tools.computer_use.handoff\n"
             "assert lease.get().holder == lease.AGENT\n"
             "assert lease.acquire('v1').holder == lease.HUMAN\n"
             "assert lease.get().holder == lease.HUMAN\n"
             "assert lease.release('v1').holder == lease.AGENT\n"
             "print('OK')\n") % os.getcwd()
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, encoding="utf-8", timeout=60,
                         stdin=subprocess.DEVNULL, env={**os.environ, "HERMES_HOME": str(tmp_path)})
    assert out.stdout.strip() == "OK", out.stderr
