"""Approve binds to the review token the user echoes, not to a hash minted at apply time."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import write_approval as wa
from tools.memory_tool import MemoryStore, apply_memory_pending


def _store():
    store = MemoryStore()
    store.load_from_disk()
    return store


def _stage(subsystem=wa.MEMORY):
    payload = ({"action": "add", "target": "user", "content": "reviewed"} if subsystem == wa.MEMORY
               else {"action": "create", "name": "review-example", "content": "reviewed"})
    return wa.stage_write(subsystem, payload, summary="reviewed", origin="foreground")


@pytest.mark.parametrize("command", ["pending", "approve", "stale-token", "diff"])
def test_review_shows_complete_payload_before_its_token(command):
    store = _store()
    subsystem = wa.SKILLS if command == "diff" else wa.MEMORY
    rec = _stage(subsystem)
    reviewed = wa.payload_sha256(rec["payload"])
    rec["payload"]["content"] = "complete replacement\n" + "detail " * 80 + "END OF PAYLOAD"
    wa._pending_path(subsystem, rec["id"]).write_text(json.dumps(rec), encoding="utf-8")
    if command == "stale-token":
        out = handle_pending_subcommand(subsystem, ["approve", rec["id"], reviewed], memory_store=store)
        assert "Approved 0" in out and "changed since review" in out
        command = "approve"
    args = ["pending"] if command == "pending" else [command, rec["id"]]
    out = handle_pending_subcommand(subsystem, args, memory_store=store)
    start = out.index("{")
    shown, end = json.JSONDecoder().raw_decode(out[start:])
    assert shown == rec["payload"]  # The summary deliberately still describes the old payload.
    assert out.index("review token:") >= start + end
    assert wa.payload_sha256(shown) in out
    assert store.user_entries == []
    assert wa.get_pending(subsystem, rec["id"]) is not None


@pytest.mark.parametrize("fault", ["successor", "cleanup-error", "concurrent"])
def test_successful_write_keeps_its_settlement(monkeypatch, fault):
    store, rec = _store(), _stage()
    path = wa._pending_path(wa.MEMORY, rec["id"])
    entered, release, calls = Event(), Event(), []
    if fault == "cleanup-error":
        unlink = Path.unlink

        def fail_cleanup(target, *args, **kwargs):
            if target == path:
                raise OSError("cleanup failure")
            return unlink(target, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_cleanup)

    def apply(current, *, hold=False):
        calls.append(current["id"])
        if hold:
            entered.set()
            assert release.wait(5)
        result = apply_memory_pending(current["payload"], store)
        if fault == "successor":
            path.write_text(json.dumps({**current, "payload": {**current["payload"], "content": "successor"}}),
                            encoding="utf-8")
        return result["success"], "", result

    consume = lambda callback: wa.apply_pending_record(
        wa.MEMORY, rec["id"], callback, expected_payload_sha256=wa.payload_sha256(rec["payload"]))
    if fault == "concurrent":  # A second approve or a reject during the apply must neither run nor drop it.
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(consume, lambda current: apply(current, hold=True))
            try:
                assert entered.wait(5)
                assert not consume(apply).applied
                out = handle_pending_subcommand(wa.MEMORY, ["reject", rec["id"]])
                assert "busy" in out.lower() and path.exists()
            finally:
                release.set()
            outcome = first.result(timeout=5)
    else:
        outcome = consume(apply)
    assert outcome.applied and outcome.result["success"]
    assert calls == [rec["id"]] and store.user_entries == ["reviewed"]
    assert path.exists() == (fault != "concurrent")
    if fault == "successor":
        assert wa.get_pending(wa.MEMORY, rec["id"])["payload"]["content"] == "successor"
    if fault == "cleanup-error":
        assert "failure" in outcome.message
