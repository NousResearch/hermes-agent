"""Tests: tools/group_relay.py — the file store between `hermes group send`
and the Desktop for Desktop-coordinated Group Chats."""

from __future__ import annotations

import json
import os
import time

import pytest

from tools import group_relay as gr


@pytest.fixture
def root(tmp_path):
    return tmp_path / ".hermes"


def test_enqueue_writes_envelope_atomically(root):
    env = gr.enqueue(
        root, room_id="rm-1", room_name="Launchpad", text="  hello  ",
        from_profile="researcher", label="Ada via Discord", thread=None,
    )
    assert len(env["id"]) == 32 and env["text"] == "hello" and env["thread"] is None
    path = gr.relay_root(root) / gr.OUTBOX_DIR / f"{env['id']}.json"
    assert json.loads(path.read_text()) == env
    assert not list((gr.relay_root(root) / gr.OUTBOX_DIR).glob(".env-*"))  # no temp leftovers
    assert gr.pending_count(root) == 1


def test_enqueue_validation(root):
    with pytest.raises(gr.GroupRelayError, match="text"):
        gr.enqueue(root, room_id="r", room_name="n", text="  ", from_profile="p", label="l")
    with pytest.raises(gr.GroupRelayError, match="room"):
        gr.enqueue(root, room_id="", room_name="", text="x", from_profile="p", label="l")
    with pytest.raises(gr.GroupRelayError, match="too long"):
        gr.enqueue(root, room_id="r", room_name="n", text="x" * (gr.MAX_TEXT_CHARS + 1), from_profile="p", label="l")
    with pytest.raises(gr.GroupRelayError, match="label"):
        gr.enqueue(root, room_id="r", room_name="n", text="x", from_profile="p", label="l" * 201)


def test_claim_pending_is_exactly_once_and_ordered(root):
    a = gr.enqueue(root, room_id="r", room_name="n", text="a", from_profile="p", label="l")
    b = gr.enqueue(root, room_id="r", room_name="n", text="b", from_profile="p", label="l")
    first = gr.claim_pending(root)
    assert sorted(e["id"] for e in first) == sorted([a["id"], b["id"]])
    assert gr.claim_pending(root) == []
    assert gr.pending_count(root) == 0
    assert (gr.relay_root(root) / gr.CLAIMED_DIR / f"{a['id']}.json").exists()


def test_claim_expires_stale_envelopes_with_error_line(root):
    env = gr.enqueue(root, room_id="r", room_name="Launchpad", text="a", from_profile="p", label="l")
    path = gr.relay_root(root) / gr.OUTBOX_DIR / f"{env['id']}.json"
    stale = {**env, "created_at": int(time.time()) - 3600}
    path.write_text(json.dumps(stale))
    assert gr.claim_pending(root, ttl_seconds=60) == []
    assert not path.exists()
    lines, _ = gr.read_reply_lines(root, env["id"])
    assert lines[0]["kind"] == "error" and lines[0]["reason"] == "queued_expired"
    assert "Launchpad" in lines[0]["error"]


def test_claim_ttl_zero_never_expires(root):
    env = gr.enqueue(root, room_id="r", room_name="n", text="a", from_profile="p", label="l")
    path = gr.relay_root(root) / gr.OUTBOX_DIR / f"{env['id']}.json"
    path.write_text(json.dumps({**env, "created_at": 1}))
    assert [e["id"] for e in gr.claim_pending(root, ttl_seconds=0)] == [env["id"]]


def test_reply_lines_append_and_incremental_read(root):
    env = gr.enqueue(root, room_id="r", room_name="n", text="a", from_profile="p", label="l")
    gr.append_reply_line(root, env["id"], {"kind": "accepted", "thread": "tmtm1"})
    lines, off = gr.read_reply_lines(root, env["id"])
    assert [l["kind"] for l in lines] == ["accepted"] and lines[0]["thread"] == "tmtm1" and off > 0
    gr.append_reply_line(root, env["id"], {"kind": "reply", "member": "helper", "text": "hi"})
    gr.append_reply_line(root, env["id"], {"kind": "done", "status": "settled", "replies": 1})
    more, off2 = gr.read_reply_lines(root, env["id"], offset=off)
    assert [l["kind"] for l in more] == ["reply", "done"] and off2 > off
    assert gr.read_reply_lines(root, env["id"], offset=off2) == ([], off2)
    assert all(l["id"] == env["id"] and isinstance(l["at"], int) for l in lines + more)


def test_reply_lines_skip_partial_trailing_line(root):
    env = gr.enqueue(root, room_id="r", room_name="n", text="a", from_profile="p", label="l")
    gr.append_reply_line(root, env["id"], {"kind": "accepted", "thread": "t"})
    path = gr.relay_root(root) / gr.REPLIES_DIR / f"{env['id']}.jsonl"
    with open(path, "ab") as fh:
        fh.write(b'{"kind": "reply", "text": "half')  # no newline yet
    lines, off = gr.read_reply_lines(root, env["id"])
    assert [l["kind"] for l in lines] == ["accepted"]
    with open(path, "ab") as fh:
        fh.write(b'"}\n')
    more, _ = gr.read_reply_lines(root, env["id"], offset=off)
    assert more[0]["text"] == "half"


def test_reply_line_validation(root):
    env = gr.enqueue(root, room_id="r", room_name="n", text="a", from_profile="p", label="l")
    with pytest.raises(gr.GroupRelayError, match="kind"):
        gr.append_reply_line(root, env["id"], {"kind": "nope"})
    with pytest.raises(gr.GroupRelayError, match="status"):
        gr.append_reply_line(root, env["id"], {"kind": "done", "status": "weird"})
    with pytest.raises(gr.GroupRelayError, match="text"):
        gr.append_reply_line(root, env["id"], {"kind": "reply", "member": "x", "text": " "})
    with pytest.raises(gr.GroupRelayError, match="invalid group relay id"):
        gr.append_reply_line(root, "../../etc/passwd", {"kind": "accepted"})
    with pytest.raises(gr.GroupRelayError):
        gr.read_reply_lines(root, "not-hex")
    assert gr.read_reply_lines(root, "0" * 32) == ([], 0)


def test_reply_file_mode_is_private(root):
    env = gr.enqueue(root, room_id="r", room_name="n", text="a", from_profile="p", label="l")
    path = gr.append_reply_line(root, env["id"], {"kind": "accepted", "thread": "t"})
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_sweep_and_cleanup_hook(root, monkeypatch):
    env = gr.enqueue(root, room_id="r", room_name="n", text="a", from_profile="p", label="l")
    gr.append_reply_line(root, env["id"], {"kind": "accepted", "thread": "t"})
    old = time.time() - gr.STALE_AFTER_SECONDS - 10
    for path in gr.relay_root(root).rglob("*.json*"):
        os.utime(path, (old, old))
    monkeypatch.setenv("HERMES_HOME", str(root))
    assert gr.cleanup_group_relay_artifacts() == 2
    assert gr.pending_count(root) == 0


def test_gateway_root_resolves_profile_home_to_machine_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes" / "profiles" / "researcher"))
    assert gr.gateway_root() == tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    assert gr.gateway_root() == tmp_path / ".hermes"
