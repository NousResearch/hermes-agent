"""Contracts of the public write-review API (tools.write_review_api), using a disposable profile."""
import pytest

from tools import write_approval as wa


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def test_plugin_list_is_paged_metadata_and_deciding_needs_a_preview():
    from tools.write_review_api import decide_write, list_writes, preview_write

    body = "sample memory detail"
    record = wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": body},
                            summary=body, origin="foreground")
    pid = record["id"]
    rows = list_writes("memory", page=1, page_size=1)
    assert rows["counts"]["memory"] == rows["total"] == 1
    assert rows["items"][0]["id"] == pid
    assert body not in str(rows)
    assert rows["gates"]["memory"] is False  # seeing a staged record doesn't enable staging

    assert decide_write(wa.MEMORY, pid, "approve", expect=None)["status"] == "conflict"
    shown = preview_write(wa.MEMORY, pid)
    assert shown["status"] == "ready" and shown["ops"][0]["content"] == body
    applied = decide_write(wa.MEMORY, pid, "approve", expect=shown["expect"])
    assert applied["status"] == "applied" and "result" not in applied
    repeated = decide_write(wa.MEMORY, pid, "approve", expect=shown["expect"])
    assert (repeated["status"], repeated["prior"]) == ("already_resolved", "applied")


def test_plugin_can_reject_but_not_approve_a_blocked_preview():
    from tools.write_review_api import decide_write, preview_write

    record = wa.stage_write(wa.MEMORY, {"action": "remove", "target": "memory", "old_text": "missing"},
                            summary="legacy removal", origin="foreground")
    pid = record["id"]
    shown = preview_write(wa.MEMORY, pid)
    assert shown["status"] == "conflict" and shown["expect"]
    assert decide_write(wa.MEMORY, pid, "approve", expect=shown["expect"])["status"] == "conflict"
    assert wa.get_pending(wa.MEMORY, pid) is not None
    assert decide_write(wa.MEMORY, pid, "reject", expect=shown["expect"])["status"] == "rejected"


def test_plugin_list_counts_writes_that_need_review_apart_from_pending_ones():
    from tools.write_review_api import list_writes

    pid = wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "interrupted"},
                         summary="interrupted", origin="foreground")["id"]
    pending = wa._pending_path(wa.MEMORY, pid)
    pending.replace(pending.with_suffix(".claimed"))  # an approval that died mid-write
    wa._write_outcome(wa.MEMORY, pid, "applying", {})

    rows = list_writes()
    assert rows["counts"] == {"memory": 0, "skills": 0, "total": 0}
    assert rows["needs_review"] == {"memory": 1, "skills": 0, "total": 1}
    assert "interrupted" not in str(rows)


def test_a_batch_mixing_delete_with_other_ops_is_refused_by_the_resolver_not_just_the_preview(home):
    from tools.write_review_api import decide_write, preview_write

    skill = home / "skills" / "probe"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: probe\ndescription: p\n---\n# body\n", encoding="utf-8")
    before = (skill / "SKILL.md").read_bytes()
    ops = [{"action": "patch", "name": "probe", "old_string": "# body", "new_string": "# body\nADDED"},
           {"action": "delete", "name": "probe"}]
    pid = wa.stage_write(wa.SKILLS, {"action": "batch", "operations": ops}, summary="mixed", origin="foreground")["id"]
    shown = preview_write(wa.SKILLS, pid)
    assert shown["status"] == "conflict" and shown["expect"]

    for approved in (decide_write(wa.SKILLS, pid, "approve", expect=shown["expect"]),
                     wa.resolve_pending(wa.SKILLS, pid, "approve")):
        assert approved["status"] == "conflict" and "atomically" in approved["reason"], approved
    assert (skill / "SKILL.md").read_bytes() == before
    assert decide_write(wa.SKILLS, pid, "reject", expect=shown["expect"])["status"] == "rejected"


def test_a_memory_store_that_cannot_load_is_a_conflict_and_the_write_stays_pending(monkeypatch):
    from tools import memory_tool
    from tools.write_review_api import decide_write, preview_write

    pid = wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "kept"},
                         summary="kept", origin="foreground")["id"]
    shown = preview_write(wa.MEMORY, pid)

    def fail():
        raise OSError("memory store unreadable")

    monkeypatch.setattr(memory_tool, "load_on_disk_store", fail)
    r = decide_write(wa.MEMORY, pid, "approve", expect=shown["expect"])
    assert r["status"] == "conflict" and "memory store unreadable" in r["reason"], r
    assert wa.get_pending(wa.MEMORY, pid) is not None
