"""Approve binds to the review token the user echoes, not to a hash minted at apply time."""
import json
import os
import shutil
import tempfile

import pytest

from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import write_approval as wa
from tools.memory_tool import MemoryStore


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


def test_approve_without_token_does_not_apply(hermes_home):
    store = MemoryStore()
    store.load_from_disk()
    rec = _stage()
    out = handle_pending_subcommand(wa.MEMORY, ["approve", rec["id"]], memory_store=store)
    assert "Not applied" in out
    assert wa.payload_sha256(rec["payload"]) in out
    assert store.user_entries == []
    assert wa.get_pending(wa.MEMORY, rec["id"]) is not None


def test_stale_review_token_does_not_apply_replacement(hermes_home):
    store = MemoryStore()
    store.load_from_disk()
    rec = _stage("reviewed")
    reviewed = wa.payload_sha256(rec["payload"])
    path = wa._pending_path(wa.MEMORY, rec["id"])
    swapped = json.loads(path.read_text(encoding="utf-8"))
    swapped["payload"]["content"] = "replaced"
    swapped["summary"] = "replaced"
    path.write_text(json.dumps(swapped), encoding="utf-8")

    out = handle_pending_subcommand(wa.MEMORY, ["approve", rec["id"], reviewed], memory_store=store)
    assert "Approved 0" in out
    assert "changed since review" in out
    assert store.user_entries == []
    assert wa.get_pending(wa.MEMORY, rec["id"])["payload"]["content"] == "replaced"


def test_echoed_current_token_applies(hermes_home):
    store = MemoryStore()
    store.load_from_disk()
    rec = _stage("kept")
    token = wa.payload_sha256(rec["payload"])
    out = handle_pending_subcommand(wa.MEMORY, ["approve", rec["id"], token], memory_store=store)
    assert "Approved 1" in out
    assert store.user_entries == ["kept"]
    assert wa.pending_count("memory") == 0


def test_replacement_during_apply_is_reported_as_applied(hermes_home, monkeypatch):
    store = MemoryStore()
    store.load_from_disk()
    rec = _stage("original")
    token = wa.payload_sha256(rec["payload"])
    path = wa._pending_path(wa.MEMORY, rec["id"])

    def swap_then_succeed(payload, memory_store):
        current = json.loads(path.read_text(encoding="utf-8"))
        current["payload"]["content"] = "replacement"
        path.write_text(json.dumps(current), encoding="utf-8")
        memory_store.add("user", payload["content"])
        return {"success": True}

    monkeypatch.setattr("tools.memory_tool.apply_memory_pending", swap_then_succeed)
    out = handle_pending_subcommand(wa.MEMORY, ["approve", rec["id"], token], memory_store=store)
    assert "Approved 1" in out
    assert "Approved 0" not in out
    assert "Failed:" not in out
    assert "newer pending record was kept" in out
    assert store.user_entries == ["original"]
    assert wa.get_pending(wa.MEMORY, rec["id"])["payload"]["content"] == "replacement"
