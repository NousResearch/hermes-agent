"""Approve binds to the review token the user echoes, not to a hash minted at apply time."""
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import pytest

from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import write_approval as wa
from tools.memory_tool import MemoryStore, apply_memory_pending


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_wa_bind_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _stage(content="reviewed"):
    return wa.stage_write(
        "memory",
        {"action": "add", "target": "user", "content": content},
        summary=content,
        origin="foreground",
    )


@pytest.mark.parametrize("command", ["pending", "approve", "stale-token", "diff"])
def test_review_shows_complete_payload_before_its_token(hermes_home, command):
    store = MemoryStore()
    store.load_from_disk()
    subsystem = wa.SKILLS if command == "diff" else wa.MEMORY
    rec = (_stage() if subsystem == wa.MEMORY else wa.stage_write(
        wa.SKILLS, {"action": "create", "name": "review-example", "content": "reviewed"},
        summary="reviewed", origin="foreground"))
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
    if command == "diff":
        assert out.index("END OF PAYLOAD") < out.index("review token:") < out.index("Apply:")
    assert store.user_entries == []
    assert wa.get_pending(subsystem, rec["id"]) is not None


@pytest.mark.parametrize("fault", ["successor", "unlink-error", "unlock-error", "double-consumer"])
def test_successful_write_keeps_its_settlement(hermes_home, monkeypatch, fault):
    store = MemoryStore()
    store.load_from_disk()
    rec = _stage()
    path = wa._pending_path(wa.MEMORY, rec["id"])
    entered, release, calls = Event(), Event(), []
    if fault == "unlink-error":
        unlink = Path.unlink

        def fail_cleanup(target, *args, **kwargs):
            if target == path:
                raise OSError("cleanup failure")
            return unlink(target, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_cleanup)
    if fault == "unlock-error":
        lock = wa._pending_lock

        @contextmanager
        def fail_unlock(subsystem):
            with lock(subsystem):
                yield
            raise OSError("unlock failure")

        monkeypatch.setattr(wa, "_pending_lock", fail_unlock)

    def apply(current, *, hold=False):
        calls.append(current["id"])
        if hold:
            entered.set()
            assert release.wait(5)
        result = apply_memory_pending(current["payload"], store)
        if fault == "successor":
            successor = {**current, "payload": {**current["payload"], "content": "successor"}}
            path.write_text(json.dumps(successor), encoding="utf-8")
        return result["success"], "", result

    consume = lambda callback: wa.apply_pending_record(
        wa.MEMORY, rec["id"], callback, expected_payload_sha256=wa.payload_sha256(rec["payload"]))
    if fault == "double-consumer":
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(consume, lambda current: apply(current, hold=True))
            try:
                assert entered.wait(5)
                second = consume(apply)
                assert not second.applied
            finally:
                release.set()
            outcome = first.result(timeout=5)
    else:
        outcome = consume(apply)
    assert outcome.applied and outcome.result["success"]
    assert outcome.consumed == (fault in {"unlock-error", "double-consumer"})
    assert calls == [rec["id"]] and store.user_entries == ["reviewed"]
    if fault in {"successor", "unlink-error"}:
        assert path.exists()
    else:
        assert not path.exists()
    if fault == "successor":
        assert wa.get_pending(wa.MEMORY, rec["id"])["payload"]["content"] == "successor"
    if fault.endswith("error"):
        assert "failure" in outcome.message


@pytest.mark.parametrize("state", ["corrupt", "busy", "missing"])
def test_reject_reports_the_actual_pending_state(hermes_home, state):
    rec = _stage()
    path = wa._pending_path(wa.MEMORY, rec["id"])
    if state == "corrupt":
        path.write_text("{broken json", encoding="utf-8")
    if state == "missing":
        path.unlink()
    if state == "busy":
        with wa._pending_lock(wa.MEMORY):
            out = handle_pending_subcommand(wa.MEMORY, ["reject", rec["id"]])
        assert "busy" in out.lower() and "No pending" not in out
        assert path.exists()
    else:
        out = handle_pending_subcommand(wa.MEMORY, ["reject", rec["id"]])
        assert ("Rejected" if state == "corrupt" else "No pending") in out
        assert not path.exists()
