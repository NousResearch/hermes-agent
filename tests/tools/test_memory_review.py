"""Real pending memory review, using synthetic files in isolated homes only."""
from pathlib import Path

from tools import write_approval as wa
from tools.memory_tool import MemoryStore, ENTRY_DELIMITER
import pytest


def test_read_only_memory_settings_expand_scoped_refs_and_managed_overlay(tmp_path, monkeypatch):
    from tools.memory_review_config import read_memory_review_config
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GENERIC_MEMORY_LIMIT", "52")
    (tmp_path / "config.yaml").write_text('memory:\n  memory_char_limit: "${env:GENERIC_MEMORY_LIMIT}"\n  user_profile_enabled: false\n')
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text('memory:\n  memory_char_limit: 65\n  write_approval: true\n')
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    cfg = read_memory_review_config()["memory"]
    assert cfg["memory_char_limit"] == 65
    assert cfg["write_approval"] is True
    assert cfg["user_profile_enabled"] is False
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "absent"))
    assert read_memory_review_config()["memory"]["memory_char_limit"] == "52"


def test_process_reject_waits_for_inflight_approval(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    import json
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = wa.stage_write(wa.MEMORY, {"action": "add", "content": "Generic entry."},
                            summary="Generic", origin="foreground")
    ready, release = tmp_path / "ready", tmp_path / "release"
    script = '''
import os, pathlib, time
from tools import memory_review as mr
home = pathlib.Path(os.environ['HERMES_HOME'])
review = mr.list_memory_reviews()['batches'][0]
apply = mr.apply_memory_pending
def paused(payload, store):
    if not isinstance(store, mr._PreviewStore):
        (home / 'ready').touch()
        while not (home / 'release').exists(): time.sleep(.01)
    return apply(payload, store)
mr.apply_memory_pending = paused
print(__import__('json').dumps(mr.decide_memory_review(review['id'], 'approve', review['revision'])), flush=True)
'''
    env = {**os.environ, "HOME": str(tmp_path)}
    child = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    import time
    deadline = time.monotonic() + 15
    try:
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        reject = subprocess.Popen([sys.executable, "-c", "import os,pathlib; from tools import write_approval as wa; (pathlib.Path(os.environ['HERMES_HOME']) / 'reject-ready').touch(); print(wa.discard_pending(wa.MEMORY, '" + record['id'] + "'))"],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while not (tmp_path / 'reject-ready').exists() and reject.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / 'reject-ready').exists()
        with pytest.raises(subprocess.TimeoutExpired):
            reject.communicate(timeout=2)
        release.touch()
        out, err = child.communicate(timeout=15)
        rejected, reject_err = reject.communicate(timeout=15)
        assert child.returncode == reject.returncode == 0, (err, reject_err)
        assert json.loads(out)['success']
        assert rejected.strip() == "False"
        assert (tmp_path / "memories/MEMORY.md").read_text() == "Generic entry."
    finally:
        release.touch()
        if child.poll() is None:
            child.kill()
            child.wait()


def test_reentrant_legacy_reject_cannot_succeed_during_approval(tmp_path, monkeypatch):
    from tools import memory_review as mr
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    record = wa.stage_write(wa.MEMORY, {"action": "add", "content": "Generic entry."},
                            summary="Generic", origin="foreground")
    revision = mr.list_memory_reviews()["batches"][0]["revision"]
    apply = mr.apply_memory_pending
    rejected = []
    def reentrant(payload, store):
        if not isinstance(store, mr._PreviewStore):
            rejected.append(handle_pending_subcommand(wa.MEMORY, ["reject", record["id"]]))
        return apply(payload, store)
    monkeypatch.setattr(mr, "apply_memory_pending", reentrant)
    assert mr.decide_memory_review(record["id"], "approve", revision)["success"]
    assert not any(message.startswith("Rejected") for message in rejected)


def test_legacy_approval_does_not_replay_a_concurrently_rejected_record(tmp_path, monkeypatch):
    from hermes_cli import write_approval_commands as commands
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = wa.stage_write(wa.MEMORY, {"action": "add", "content": "Generic entry."},
                            summary="Generic", origin="foreground")
    listing = wa.list_pending
    def reject_after_list(subsystem):
        values = listing(subsystem)
        wa.discard_pending(subsystem, record["id"])
        return values
    monkeypatch.setattr(wa, "list_pending", reject_after_list)
    output = commands.handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=MemoryStore())
    assert "Approved 0" in output
    assert not (tmp_path / "memories/MEMORY.md").exists()


def test_cold_preview_with_config_preserves_entire_home(tmp_path):
    import os
    import subprocess
    import sys
    home = tmp_path / "isolated"
    home.mkdir()
    (home / "config.yaml").write_text("memory:\n  write_approval: true\n  memory_char_limit: 42\n")
    pending = home / "pending/memory"
    pending.mkdir(parents=True)
    (pending / "abcdef12.json").write_text('{"id":"abcdef12","payload":{"action":"add","content":"Generic entry."}}')
    script = '''
import pathlib
home = pathlib.Path(__import__('os').environ['HERMES_HOME'])
def snapshot():
    return {str(p.relative_to(home)): (p.stat().st_mode, p.read_bytes() if p.is_file() else None) for p in home.rglob('*')}
before = snapshot()
from tools.memory_review import list_memory_reviews
result = list_memory_reviews()
assert result['write_approval']
assert result['batches'][0]['can_approve']
assert snapshot() == before, (set(snapshot()) - set(before))
'''
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1"}
    run = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr



@pytest.mark.parametrize("payload", [
    {"action": "remove", "target": "memory", "old_text": "Generic", "matched_entry": "Generic entry."},
    {"action": "replace", "target": "user", "old_text": "Generic", "matched_entry": "Generic entry.", "content": "Generic new entry."},
    {"action": "batch", "target": "memory", "operations": [{"action": "add", "content": "Generic added entry."}, {"action": "remove", "old_text": "Generic entry", "matched_entry": "Generic entry."}]},
])
def test_review_equivalence_and_stale_revision(tmp_path, monkeypatch, payload):
    from tools.memory_review import list_memory_reviews, decide_memory_review
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    store = MemoryStore()
    target = payload["target"]
    store.add(target, "Generic entry.")
    record = wa.stage_write(wa.MEMORY, payload, summary="Generic", origin="foreground")
    batch = list_memory_reviews()["batches"][0]
    assert batch["can_approve"]
    store.add(target, "Generic concurrent entry.")
    assert not decide_memory_review(record["id"], "approve", batch["revision"])["success"]
    fresh = list_memory_reviews()["batches"][0]
    assert decide_memory_review(record["id"], "approve", fresh["revision"])["success"]
    assert store._path_for(target).read_text() == fresh["after"]


def test_invalid_batch_and_legacy_destructive_write_cannot_approve(tmp_path, monkeypatch):
    from tools.memory_review import list_memory_reviews, decide_memory_review
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    store = MemoryStore()
    store.add("memory", "Generic entry.")
    payloads = [
        {"action": "remove", "target": "memory", "old_text": "Generic"},
        {"action": "batch", "target": "memory", "operations": [{"action": "add", "content": "Generic addition."}, {"action": "invalid"}]},
        {"action": "add", "target": "memory", "content": "x" * 3000},
    ]
    for payload in payloads:
        wa.stage_write(wa.MEMORY, payload, summary="Generic invalid", origin="foreground")
    for batch in list_memory_reviews()["batches"]:
        assert not batch["can_approve"] and batch["error"]
        assert batch["after"] == batch["before"]
        assert not decide_memory_review(batch["id"], "approve", batch["revision"])["success"]
        assert decide_memory_review(batch["id"], "reject", batch["revision"])["success"]
    assert store._path_for("memory").read_text() == "Generic entry."
    assert not decide_memory_review("../bad", "reject", "")["success"]


def test_preview_refuses_a_changed_read_snapshot(tmp_path, monkeypatch):
    from tools.memory_review import list_memory_reviews, _PreviewStore
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    MemoryStore().add("memory", "Generic initial entry.")
    wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "Generic addition."},
                   summary="Generic", origin="foreground")
    read = MemoryStore._read_raw_checked
    calls = 0
    def changing_read(path):
        nonlocal calls
        calls += 1
        raw, ok = read(path)
        return ("Generic newer entry.", ok) if calls > 1 else (raw, ok)
    monkeypatch.setattr(_PreviewStore, "_read_raw_checked", staticmethod(changing_read))
    batch = list_memory_reviews()["batches"][0]
    assert not batch["can_approve"]
    assert "changed since review" in batch["error"]


def test_preview_does_not_create_memory_directory(tmp_path, monkeypatch):
    from tools.memory_review import list_memory_reviews
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "Generic entry."},
                   summary="Generic", origin="foreground")
    before = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")}
    assert list_memory_reviews()["batches"][0]["can_approve"]
    assert {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")} == before


def test_review_is_read_only_and_matches_atomic_application(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    store = MemoryStore()
    assert store.add("memory", "Generic original entry with a retained clause.")["success"]
    payload = {"action": "batch", "target": "memory", "operations": [
        {"action": "replace", "old_text": "original", "matched_entry": "Generic original entry with a retained clause.", "content": "Generic replacement."},
        {"action": "add", "new_text": "Generic second entry."},
    ]}
    record = wa.stage_write(wa.MEMORY, payload, summary="Generic batch", origin="background_review")
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    from tools.memory_review import list_memory_reviews, decide_memory_review
    review = list_memory_reviews()
    batch = review["batches"][0]
    assert batch["id"] == record["id"] and batch["can_approve"]
    assert not review["write_approval"]
    assert batch["before"] == "Generic original entry with a retained clause."
    assert batch["after"] == ENTRY_DELIMITER.join(["Generic replacement.", "Generic second entry."])
    assert "-Generic original entry with a retained clause." in batch["diff"]
    assert "+Generic replacement." in batch["diff"]
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    decision = decide_memory_review(batch["id"], "approve", batch["revision"])
    assert decision["success"]
    assert (tmp_path / "memories/MEMORY.md").read_text() == batch["after"]
    assert not wa.list_pending(wa.MEMORY)
