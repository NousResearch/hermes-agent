"""A long streamed turn stays replayable: deltas fold into one catch-up per open block.

Every streamed chunk used to take its own replay-ring entry, so a 2,000-token reply pushed the turn's
``message.start`` (and every tool frame before it) out of the 512-entry window: a client reconnecting
mid-turn got ``truncated`` and had to refetch the transcript while the reply was still streaming.
"""

import pytest

from tui_gateway import event_replay


@pytest.fixture(autouse=True)
def _clean():
    event_replay.reset_replay_state()
    event_replay.configure(coalesce_deltas=True)  # dashboard.replay_coalesce_deltas: true
    yield
    event_replay.configure(coalesce_deltas=False)
    event_replay.reset_replay_state()


def _stamp(sid, etype, payload):
    frame = {"jsonrpc": "2.0", "method": "event", "params": {"type": etype, "session_id": sid, "payload": payload}}
    event_replay._stamp_event(frame)
    return frame["params"]["seq"]


def _stream_reply(sid, words, *, turn_id="turn-1", segment=0):
    offset, text = 0, ""
    for index in range(words):
        chunk = f"w{index} "
        _stamp(sid, "message.delta", {"text": chunk, "turn_id": turn_id, "segment": segment, "offset": offset})
        offset += len(chunk)
        text += chunk
    return text


def test_a_long_streamed_turn_replays_whole_with_one_catch_up_delta_per_block():
    _stamp("s", "message.start", {"turn_id": "turn-1"})
    _stamp("s", "tool.start", {"tool_id": "t1", "name": "terminal"})
    _stamp("s", "tool.complete", {"tool_id": "t1", "name": "terminal"})
    reply = _stream_reply("s", 2000)
    midway = event_replay.latest_seq("s") - 500

    # A reconnect from before the turn: nothing is truncated, and the reply arrives as ONE delta
    # carrying the block's whole text at offset 0 of its segment, stamped with its last chunk's seq.
    assert not event_replay.is_truncated("s", 0)
    replay = event_replay.events_since("s", 0)
    assert [event["type"] for event in replay] == ["message.start", "tool.start", "tool.complete", "message.delta"]
    catch_up = replay[-1]
    assert catch_up["payload"] == {"text": reply, "turn_id": "turn-1", "segment": 0, "offset": 0}
    assert catch_up["seq"] == event_replay.latest_seq("s")

    # A reconnect from mid-reply gets only the unseen tail, at the offset it starts from.
    tail = event_replay.events_since("s", midway)
    assert len(tail) == 1
    seen = sum(len(f"w{index} ") for index in range(midway - 3))
    assert tail[0]["payload"]["offset"] == seen and tail[0]["payload"]["text"] == reply[seen:]

    # The completed record closes the block: a later delta starts a new one, so order is kept.
    _stamp("s", "message.complete", {"text": reply})
    _stamp("s", "message.delta", {"text": "next", "turn_id": "turn-2", "segment": 0, "offset": 0})
    assert [event["type"] for event in event_replay.events_since("s", 0)][-2:] == ["message.complete", "message.delta"]
    assert event_replay.replay_stats()["events"] == 6


def test_an_agent_terminal_run_keeps_only_its_last_four_kib():
    for index in range(400):
        _stamp("s", "agent.terminal.output", {"process_id": "p1", "chunk": f"line {index:04d}\n" * 4})

    [catch_up] = event_replay.events_since("s", 0)
    chunk = catch_up["payload"]["chunk"]
    assert catch_up["payload"]["process_id"] == "p1"
    assert len(chunk.encode()) <= event_replay_tail_bytes() and chunk.endswith("line 0399\n")
    assert not event_replay.is_truncated("s", 0)


def test_the_legacy_default_keeps_one_replay_entry_per_chunk():
    event_replay.configure(coalesce_deltas=False)
    _stamp("s", "message.start", {"turn_id": "turn-1"})
    _stream_reply("s", 3)

    assert [event["payload"].get("text") for event in event_replay.events_since("s", 0)] == [None, "w0 ", "w1 ", "w2 "]


def event_replay_tail_bytes():
    from tui_gateway.event_replay_deltas import TOOL_TAIL_BYTES
    return TOOL_TAIL_BYTES


def test_a_flood_of_empty_chunks_is_held_to_the_ring_byte_cap(monkeypatch):
    # One run is one ring entry, so the 512-entry cap cannot bound it: each chunk's own retained
    # cost must count against the byte caps, or empty chunks would grow a run for free.
    monkeypatch.setattr(event_replay, "_REPLAY_BUFFER_BYTES_MAX", 64 * 1024)
    for _ in range(5000):
        _stamp("s", "message.delta", {"text": "", "turn_id": "turn-1", "segment": 0, "offset": 0})

    stats = event_replay.replay_stats()
    assert 0 < stats["bytes"] <= 64 * 1024
    assert event_replay.is_truncated("s", 0)
