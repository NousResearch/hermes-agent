"""Skill proposal intake preserves approval records without redundant queue entries."""

import json

import pytest


@pytest.fixture
def skill_home(monkeypatch, tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("skills:\n  write_approval: true\n", encoding="utf-8")
    skill = home / "skills" / "routine"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: routine\ndescription: Use when checking a routine.\n---\n\nCheck the result.\n",
        encoding="utf-8",
    )
    return home


def _propose(operations=None):
    import tools.skill_manager_tool  # registers the real tool handler
    from tools.registry import registry

    entry = registry.get_entry("skill_manage")
    assert entry is not None
    arguments = operations if isinstance(operations, dict) else {"operations": operations if operations is not None else [{
        "action": "patch", "name": "routine", "old_string": "Check the result.",
        "new_string": "Check the result against the exact requested target.",
    }]}
    return json.loads(entry.handler(arguments))


def test_identical_skill_proposals_reuse_immutable_pending_record(skill_home):
    first = _propose()
    pending = skill_home / "pending" / "skills" / (first["pending_id"] + ".json")
    original = pending.read_bytes()

    second = _propose()

    assert second["pending_id"] == first["pending_id"]
    assert pending.read_bytes() == original
    assert len(list(pending.parent.glob("*.json"))) == 1
    assert (skill_home / "skills" / "routine" / "SKILL.md").read_text().endswith("Check the result.\n")


def test_identical_new_skill_packages_reuse_one_pending_proposal(skill_home):
    operations = [
        {"action": "create", "name": "new-routine", "content":
         "---\nname: new-routine\ndescription: Use when checking a new routine.\n---\n\nCheck it.\n"},
        {"action": "write_file", "name": "new-routine", "file_path": "references/detail.md",
         "file_content": "Check the exact target before a retry.\n"},
    ]
    first = _propose(operations)
    second = _propose(operations)

    assert second["pending_id"] == first["pending_id"]
    assert len(list((skill_home / "pending" / "skills").glob("*.json"))) == 1
    assert not (skill_home / "skills" / "new-routine").exists()


@pytest.mark.parametrize("error_type", [PermissionError, AttributeError])
def test_unreadable_base_does_not_lose_the_staged_proposal(skill_home, monkeypatch, error_type):
    from pathlib import Path

    original_read = Path.read_bytes
    target = skill_home / "skills" / "routine" / "SKILL.md"

    def deny_base(path):
        if path == target:
            raise error_type("base unavailable")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", deny_base)
    result = _propose()

    assert result["staged"]
    pending = skill_home / "pending" / "skills" / (result["pending_id"] + ".json")
    assert pending.is_file()
    assert "base_files" not in json.loads(pending.read_text())


@pytest.mark.platforms("posix")
def test_symlink_loop_disables_fingerprinting_without_losing_proposal(skill_home):
    skill = skill_home / "skills" / "routine"
    original = (skill / "SKILL.md").read_bytes()
    references = skill / "references"
    references.mkdir()
    loop = references / "loop.md"
    loop.symlink_to("loop.md")

    result = _propose([{"action": "write_file", "name": "routine",
                        "file_path": "references/loop.md", "file_content": "do not apply"}])

    assert result["staged"]
    pending = skill_home / "pending" / "skills" / (result["pending_id"] + ".json")
    assert pending.is_file()
    assert "base_files" not in json.loads(pending.read_text())
    assert (skill / "SKILL.md").read_bytes() == original
    assert loop.is_symlink()
    assert str(loop.readlink()) == "loop.md"


def test_invalid_old_queue_metadata_does_not_lose_a_new_proposal(skill_home):
    pending = skill_home / "pending" / "skills"
    pending.mkdir(parents=True)
    for pending_id, timestamp in [("00000001", 0), ("00000002", "invalid")]:
        (pending / (pending_id + ".json")).write_text(json.dumps({
            "id": pending_id, "created_at": timestamp, "payload": {},
        }))

    result = _propose()

    assert (pending / (result["pending_id"] + ".json")).exists()
    assert len(list(pending.glob("*.json"))) == 3


@pytest.mark.parametrize("corrupt_id", ["missing", "01234567", "../not-a-pending-id", 12])
def test_invalid_reusable_identity_gets_a_fresh_persisted_record(skill_home, corrupt_id):
    first = _propose()
    pending = skill_home / "pending" / "skills" / (first["pending_id"] + ".json")
    invalid = json.loads(pending.read_text())
    if corrupt_id == "missing":
        invalid.pop("id")
    else:
        invalid["id"] = corrupt_id
    pending.write_text(json.dumps(invalid))
    before = pending.read_bytes()

    second = _propose()

    assert second["staged"]
    assert (pending.parent / (second["pending_id"] + ".json")).is_file()
    assert pending.read_bytes() == before
    assert len(list(pending.parent.glob("*.json"))) == 2


@pytest.mark.parametrize("lock_failure", ["directory", "permission"])
def test_unavailable_optional_lock_does_not_lose_a_new_proposal(skill_home, monkeypatch, lock_failure):
    pending = skill_home / "pending" / "skills"
    pending.mkdir(parents=True)
    lock = pending / ".intake.lock"
    if lock_failure == "directory":
        lock.mkdir()
    else:
        real_open = open

        def deny_lock(path, *args, **kwargs):
            if path == lock or path == str(lock):
                raise PermissionError("optional intake lock unavailable")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", deny_lock)

    result = _propose()

    assert result["staged"]
    assert (pending / (result["pending_id"] + ".json")).is_file()
    assert len(list(pending.glob("*.json"))) == 1


@pytest.mark.parametrize("change", ["base", "payload", "create-root"])
def test_different_proposal_identity_gets_a_new_record(skill_home, change):
    operations = None
    if change == "create-root":
        operations = [{"action": "create", "name": "fresh", "content":
                       "---\nname: fresh\ndescription: Use when checking fresh input.\n---\n\nCheck it.\n"}]
    first = _propose(operations)
    pending = skill_home / "pending" / "skills" / (first["pending_id"] + ".json")
    before = pending.read_bytes()
    if change == "base":
        with (skill_home / "skills" / "routine" / "SKILL.md").open("a") as handle:
            handle.write("Keep unrelated improvements.\n")
    elif change == "payload":
        operations = [{"action": "patch", "name": "routine", "old_string": "Check the result.",
                       "new_string": "Check the result and retain the receipt."}]
    else:
        config = skill_home / "config.yaml"
        config.write_text(config.read_text() + "  create_dir: " + str(skill_home / "shared") + "\n")
    second = _propose(operations)

    assert second["pending_id"] != first["pending_id"]
    assert pending.read_bytes() == before


@pytest.mark.parametrize("action", ["edit", "patch"])
def test_full_rewrite_fingerprints_skill_even_with_an_ignored_supporting_path(skill_home, action):
    target = skill_home / "skills" / "routine" / "SKILL.md"
    operations = [{"action": action, "name": "routine", "file_path": "references/ignored.md",
                   "content": target.read_text() + "Keep this general lesson.\n"}]
    arguments = operations[0] if action == "edit" else operations
    first = _propose(arguments)
    assert first.get("staged"), first
    with target.open("a") as handle:
        handle.write("An independent instruction changed.\n")

    second = _propose(arguments)
    assert second.get("staged"), second

    assert second["pending_id"] != first["pending_id"]


def test_legacy_records_are_not_rewritten_or_assumed_to_match(skill_home):
    first = _propose()
    pending = skill_home / "pending" / "skills" / (first["pending_id"] + ".json")
    legacy = json.loads(pending.read_text())
    legacy.pop("base_files")
    pending.write_text(json.dumps(legacy))
    before = pending.read_bytes()

    second = _propose()

    assert second["pending_id"] != first["pending_id"]
    assert pending.read_bytes() == before
    assert _propose()["pending_id"] == second["pending_id"]


def test_profile_queues_remain_isolated(skill_home, tmp_path, monkeypatch):
    import shutil

    first = _propose()
    other = tmp_path / "other-profile"
    shutil.copytree(skill_home, other, ignore=shutil.ignore_patterns("pending"))
    monkeypatch.setenv("HERMES_HOME", str(other))
    second = _propose()

    assert first["pending_id"] != second["pending_id"]
    assert len(list((skill_home / "pending" / "skills").glob("*.json"))) == 1
    assert len(list((other / "pending" / "skills").glob("*.json"))) == 1


def test_context_bound_profile_uses_its_own_queue_for_shared_target(skill_home, tmp_path):
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    first = _propose()
    primary_pending = skill_home / "pending" / "skills" / (first["pending_id"] + ".json")
    before = primary_pending.read_bytes()
    secondary = tmp_path / "served-profile"
    secondary.mkdir()
    (secondary / "config.yaml").write_text(
        "skills:\n  write_approval: true\n  external_dirs: " + json.dumps([str(skill_home / "skills")]) + "\n")
    token = set_hermes_home_override(secondary)
    try:
        assert get_hermes_home() == secondary
        other = _propose()
        assert _propose()["pending_id"] == other["pending_id"]
        secondary_pending = secondary / "pending" / "skills" / (other["pending_id"] + ".json")
        assert json.loads(secondary_pending.read_text())["base_files"] == json.loads(before)["base_files"]
    finally:
        reset_hermes_home_override(token)

    assert primary_pending.read_bytes() == before
    assert len(list((secondary / "pending" / "skills").glob("*.json"))) == 1
    assert len(list((skill_home / "pending" / "skills").glob("*.json"))) == 1


def test_unsafe_supporting_path_is_not_read_for_deduplication(skill_home, monkeypatch):
    from pathlib import Path

    outside = skill_home / "private.txt"
    outside.write_text("not a skill")
    references = skill_home / "skills" / "routine" / "references"
    references.mkdir()
    (references / "linked.md").symlink_to(outside)
    original_read = Path.read_bytes

    def forbid_outside(path):
        assert path.resolve() != outside.resolve()
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_outside)
    result = _propose([{"action": "write_file", "name": "routine",
                        "file_path": "references/linked.md", "file_content": "do not write"}])

    assert result["staged"]
    record = json.loads((skill_home / "pending" / "skills" / (result["pending_id"] + ".json")).read_text())
    assert "base_files" not in record
    assert outside.read_text() == "not a skill"


def test_approval_still_requires_explicit_replay(skill_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    proposal = _propose()
    assert _propose()["pending_id"] == proposal["pending_id"]
    assert "Check the exact" not in (skill_home / "skills" / "routine" / "SKILL.md").read_text()

    response = handle_pending_subcommand(wa.SKILLS, ["approve", proposal["pending_id"]])

    assert response is not None
    assert "Approved 1" in response
    assert wa.pending_count(wa.SKILLS) == 0
    assert "Check the result against the exact requested target." in (
        skill_home / "skills" / "routine" / "SKILL.md").read_text()


def test_memory_staging_keeps_existing_append_semantics(skill_home):
    from tools import write_approval as wa

    payload = {"action": "add", "target": "user", "content": "A test preference"}
    first = wa.stage_write(wa.MEMORY, payload, summary="test", origin="foreground")
    second = wa.stage_write(wa.MEMORY, payload, summary="test", origin="foreground")

    assert first["id"] != second["id"]
    assert wa.pending_count(wa.MEMORY) == 2
    assert "base_files" not in first


def _parallel_propose(home, ready, start, results):
    import os

    os.environ["HERMES_HOME"] = home
    ready.put(True)
    assert start.wait(15)
    results.put(_propose()["pending_id"])


def test_concurrent_processes_stage_only_one_identical_proposal(skill_home):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    ready, results, start = context.Queue(), context.Queue(), context.Event()
    workers = [context.Process(target=_parallel_propose, args=(str(skill_home), ready, start, results))
               for _ in range(3)]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            assert ready.get(timeout=15)
        start.set()
        pending_ids = [results.get(timeout=15) for _ in workers]
        for worker in workers:
            worker.join(timeout=15)
            assert worker.exitcode == 0
        assert len(set(pending_ids)) == 1
        assert len(list((skill_home / "pending" / "skills").glob("*.json"))) == 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=15)
        ready.close()
        results.close()
