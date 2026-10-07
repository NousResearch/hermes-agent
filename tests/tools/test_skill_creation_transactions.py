"""Creation approval must agree with native discovery and preserve concurrent writes."""

import json
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from threading import Event, local

import pytest

from agent.skill_utils import iter_skill_index_files
from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import skill_manager_tool as smt, write_approval as wa
from tools.registry import registry
from tools.skill_provenance import reset_current_write_origin, set_current_write_origin


CONTENT = "---\nname: display\ndescription: Transaction tests.\n---\n\nOriginal body.\n"
SAMPLE = "---\nname: sample\ndescription: Sample skill.\n---\n\nSample body.\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (home / "config.yaml").write_text("skills:\n  write_approval: true\n  write_approval_mode: create\n")
    root = home / "skills" / "existing"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(CONTENT)
    return home


def dispatch(*ops):
    return json.loads(registry.dispatch("skill_manage", {"operations": list(ops)}))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("origin", ["foreground", "background_review"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("dangling", [False, True])
def test_discoverable_alias_of_support_sample_requires_review(home, origin, flat, dangling):
    root = home / "skills" / "existing"
    target = root / "references" / "sample"
    if not dangling:
        target.mkdir(parents=True)
    (root / "alias").symlink_to(target, target_is_directory=True)
    token = set_current_write_origin(origin)
    op = {"action": "write_file", "name": "existing", "file_path": "references/sample/SKILL.md",
          "file_content": SAMPLE}
    try:
        result = json.loads(smt.skill_manage(**op)) if flat else dispatch(op)
    finally:
        reset_current_write_origin(token)
    assert result.get("staged") is True, result
    assert not (target / "SKILL.md").exists()
    assert wa.get_pending(wa.SKILLS, result["pending_id"])["payload"]
    out = handle_pending_subcommand(wa.SKILLS, ["approve", result["pending_id"]])
    assert "Approved 1" in out
    assert root / "alias" / "SKILL.md" in set(iter_skill_index_files(home / "skills", "SKILL.md"))
    assert (target / "SKILL.md").read_text() == SAMPLE


@pytest.mark.platforms("posix")
def test_alias_in_another_category_is_part_of_native_discovery(home):
    root = home / "skills" / "existing"
    target = root / "references" / "sample"
    target.mkdir(parents=True)
    category = home / "skills" / "other-category"
    category.mkdir()
    (category / "alias").symlink_to(target, target_is_directory=True)
    result = dispatch({"action": "write_file", "name": "existing",
                       "file_path": "references/sample/SKILL.md", "file_content": SAMPLE})
    assert result.get("staged") is True, result
    assert not (target / "SKILL.md").exists()


def test_parallel_patches_through_display_and_directory_names_retain_every_edit(home):
    def patch(index):
        op = {"action": "patch", "name": "display" if index % 2 else "existing",
              "old_string": "Original body.", "new_string": f"Edit {index}.\nOriginal body."}
        return op, dispatch(op)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(patch, range(24)))
    pending_ids = []
    for op, result in results:
        assert result["success"], result
        if result.get("staged"):
            pid = result["pending_id"]
            record = wa.get_pending(wa.SKILLS, pid)
            assert record and record["payload"]["operations"] == [op], (record, op)
            pending_ids.append(pid)
    assert len(set(pending_ids)) == len(pending_ids)
    assert wa.pending_count(wa.SKILLS) == len(pending_ids)
    for pid in pending_ids:
        assert "Approved 1" in handle_pending_subcommand(wa.SKILLS, ["approve", pid])
    text = (home / "skills" / "existing" / "SKILL.md").read_text()
    for index in range(24):
        assert text.count(f"Edit {index}.\n") == 1
    assert wa.pending_count(wa.SKILLS) == 0


def test_alias_writer_cannot_interleave_after_manifest_removal_was_checked(home, monkeypatch):
    checked, release, attempted, finished = Event(), Event(), Event(), Event()
    actor = local()
    native_gate = smt._apply_skill_write_gate

    def hold_checked_removal(action, name, **args):
        result = native_gate(action, name, **args)
        if getattr(actor, "remover", False):
            actor.checks += 1
            if actor.checks == 2 and result is None:
                checked.set()
                assert release.wait(15), "remover was never released"
        return result

    monkeypatch.setattr(smt, "_apply_skill_write_gate", hold_checked_removal)

    def remove():
        actor.remover, actor.checks = True, 0
        return json.loads(smt.skill_manage(action="remove_file", name="existing", file_path="SKILL.md"))

    def write():
        attempted.set()
        try:
            return json.loads(smt.skill_manage(action="write_file", name="display",
                                              file_path="references/sample/SKILL.md", file_content=SAMPLE))
        finally:
            finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        remover = pool.submit(remove)
        try:
            assert checked.wait(15)
            writer = pool.submit(write)
            assert attempted.wait(15)
            # Event-based bounded wait: the old alias lock lets the writer finish here.
            interleaved = finished.wait(2)
        finally:
            release.set()
        removed, written = remover.result(timeout=15), writer.result(timeout=15)
    assert not interleaved, written
    assert removed["success"], removed
    assert not written["success"] or written.get("staged"), written
    assert not (home / "skills" / "existing" / "references" / "sample" / "SKILL.md").exists()


@pytest.mark.parametrize("legacy_lock", [False, True])
def test_contended_write_is_durable_reviewable_and_does_not_hang(home, monkeypatch, legacy_lock):
    from tools import skill_write_approval as policy
    monkeypatch.setattr(policy, "WRITE_WAIT_SECONDS", 2.0)
    held = smt._skill_mutation_lock("existing") if legacy_lock else policy.creation_transaction()
    op = {"action": "patch", "name": "existing", "old_string": "Original body.", "new_string": "Retried body."}
    with ThreadPoolExecutor(max_workers=1) as pool:
        with held:
            result = pool.submit(dispatch, op).result(timeout=15)
    assert result.get("staged") is True, result
    assert (home / "skills" / "existing" / "SKILL.md").read_text() == CONTENT
    pid = result["pending_id"]
    assert pid in handle_pending_subcommand(wa.SKILLS, ["pending"])
    assert "Retried body." in handle_pending_subcommand(wa.SKILLS, ["diff", pid])
    assert "Approved 1" in handle_pending_subcommand(wa.SKILLS, ["approve", pid])
    assert "Retried body." in (home / "skills" / "existing" / "SKILL.md").read_text()
    assert wa.pending_count(wa.SKILLS) == 0


def test_busy_approval_retains_original_id_and_can_be_retried(home, monkeypatch):
    from tools import skill_write_approval as policy
    monkeypatch.setattr(policy, "WRITE_WAIT_SECONDS", 2.0)
    result = dispatch({"action": "create", "name": "new-skill", "content": SAMPLE.replace("sample", "new-skill")})
    pid = result["pending_id"]
    record = wa.get_pending(wa.SKILLS, pid)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with policy.creation_transaction():
            out = pool.submit(handle_pending_subcommand, wa.SKILLS, ["approve", pid]).result(timeout=15)
    assert "busy" in out and "not changed" in out
    assert wa.get_pending(wa.SKILLS, pid) == record
    assert wa.pending_count(wa.SKILLS) == 1
    assert "Approved 1" in handle_pending_subcommand(wa.SKILLS, ["approve", pid])
    assert wa.pending_count(wa.SKILLS) == 0


def test_failed_batch_releases_fence_and_restores_links_and_content(home):
    root = home / "skills" / "existing"
    target = root / "references" / "guide.md"
    target.parent.mkdir()
    target.write_text("Original guide.")
    result = dispatch(
        {"action": "patch", "name": "existing", "old_string": "Original body.", "new_string": "Temporary body."},
        {"action": "patch", "name": "existing", "old_string": "Does not exist.", "new_string": "Failure."})
    assert not result["success"] and "rolled back" in result["error"], result
    assert (root / "SKILL.md").read_text() == CONTENT
    assert target.read_text() == "Original guide."
    assert dispatch({"action": "patch", "name": "display", "old_string": "Original body.",
                     "new_string": "After rollback."})["success"]


@pytest.mark.platforms("posix")
def test_rollback_preserves_directory_aliases(home):
    root = home / "skills" / "existing"
    (root / "references").mkdir()
    (root / "alias").symlink_to("references", target_is_directory=True)
    result = dispatch(
        {"action": "write_file", "name": "existing", "file_path": "references/guide.md", "file_content": "Temporary."},
        {"action": "patch", "name": "existing", "old_string": "Absent.", "new_string": "Failure."})
    assert not result["success"], result
    assert (root / "alias").is_symlink()
    assert (root / "alias").readlink().as_posix() == "references"
    assert not (root / "references" / "guide.md").exists()
    assert (root / "SKILL.md").read_text() == CONTENT


def test_profile_b_does_not_wait_for_profile_a_writer(home):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.skill_write_approval import creation_transaction
    other = home.parent / "other"
    root = other / "skills" / "existing"
    root.mkdir(parents=True)
    (other / "config.yaml").write_text((home / "config.yaml").read_text())
    (root / "SKILL.md").write_text(CONTENT)

    def write_other():
        token = set_hermes_home_override(other)
        try:
            return dispatch({"action": "patch", "name": "existing", "old_string": "Original body.",
                             "new_string": "Other profile."})
        finally:
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with creation_transaction():
            assert pool.submit(write_other).result(timeout=15)["success"]
    assert "Other profile." in (root / "SKILL.md").read_text()
    assert (home / "skills" / "existing" / "SKILL.md").read_text() == CONTENT


def test_pending_disk_failure_is_not_acknowledged_as_saved(home, monkeypatch):
    def no_space(*args, **kwargs):
        raise OSError("No space available")
    monkeypatch.setattr(wa, "atomic_json_write", no_space)
    result = dispatch({"action": "create", "name": "new-skill", "content": SAMPLE})
    assert result["success"] is False and result["error_type"] == "pending_write_failed", result
    assert not result.get("pending_id")
    assert not (home / "skills" / "new-skill").exists()
    assert wa.pending_count(wa.SKILLS) == 0


def test_queued_full_rewrite_does_not_silently_overwrite_concurrent_edit(home, monkeypatch):
    from tools import skill_write_approval as policy
    observed = Event()
    native_revisions = policy.replacement_revisions

    def capture(args):
        revisions = native_revisions(args)
        if args["content"]:
            observed.set()
        return revisions

    monkeypatch.setattr(policy, "replacement_revisions", capture)
    replacement = CONTENT.replace("Original body.", "Whole replacement.")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with policy.creation_transaction():
            future = pool.submit(smt.skill_manage, action="edit", name="display", content=replacement)
            assert observed.wait(15)
            assert dispatch({"action": "patch", "name": "existing", "old_string": "Original body.",
                             "new_string": "Concurrent edit."})["success"]
        result = json.loads(future.result(timeout=15))
    assert result.get("staged") is True, result
    assert "Concurrent edit." in (home / "skills" / "existing" / "SKILL.md").read_text()
    assert wa.get_pending(wa.SKILLS, result["pending_id"])["payload"]["content"] == replacement
    diff = handle_pending_subcommand(wa.SKILLS, ["diff", result["pending_id"]])
    assert "Concurrent edit." in diff and "Whole replacement." in diff


def test_independent_processes_retain_every_patch_across_aliases(home):
    code = '''
import json, sys
from tools import skill_manager_tool as smt
i = int(sys.argv[1])
op = {"action": "patch", "name": "display" if i % 2 else "existing",
      "old_string": "Original body.", "new_string": f"Process {i}.\\nOriginal body."}
print(smt.skill_manage(**op) if i % 2 else smt.skill_manage(action="", name="", operations=[op]), flush=True)
'''

    def write(index):
        process = subprocess.run([sys.executable, "-c", code, str(index)],
                                 env=dict(os.environ, HERMES_HOME=str(home), PYTHONPATH=str(Path(smt.__file__).parents[1])),
                                 stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
        assert process.returncode == 0, process.stderr
        return json.loads(process.stdout)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write, range(8)))
    assert all(r["success"] and not r.get("staged") for r in results), results
    text = (home / "skills" / "existing" / "SKILL.md").read_text()
    for index in range(8):
        assert text.count(f"Process {index}.\n") == 1
    assert wa.pending_count(wa.SKILLS) == 0


def test_process_exit_releases_creation_fence(home):
    code = '''
import os
from tools.skill_write_approval import creation_transaction
with creation_transaction():
    print("LOCKED", flush=True)
    os._exit(0)
'''
    process = subprocess.run([sys.executable, "-c", code],
                             env=dict(os.environ, HERMES_HOME=str(home), PYTHONPATH=str(Path(smt.__file__).parents[1])),
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0 and process.stdout.strip() == "LOCKED", process.stderr
    assert dispatch({"action": "patch", "name": "display", "old_string": "Original body.",
                     "new_string": "After process exit."})["success"]


def test_concurrent_approval_of_same_pending_id_applies_it_only_once(home):
    op = {"action": "patch", "name": "existing", "old_string": "Original body.",
          "new_string": "Applied once.\nOriginal body."}
    record = wa.stage_write(wa.SKILLS, op, summary="Old queued patch", origin="foreground")
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda _: handle_pending_subcommand(wa.SKILLS, ["approve", record["id"]]), range(8)))
    assert sum("Approved 1" in out for out in outputs) == 1, outputs
    assert (home / "skills" / "existing" / "SKILL.md").read_text().count("Applied once.") == 1
    assert wa.pending_count(wa.SKILLS) == 0


def test_concurrent_creation_requests_keep_distinct_ids_and_payloads(home):
    def create(index):
        return dispatch({"action": "create", "name": f"new-{index}", "content": SAMPLE})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(create, range(24)))
    ids = {result["pending_id"] for result in results}
    assert len(ids) == 24 and all(result["staged"] for result in results)
    records = [wa.get_pending(wa.SKILLS, pid) for pid in ids]
    assert {rec["payload"]["operations"][0]["name"] for rec in records} == {f"new-{i}" for i in range(24)}
    assert wa.pending_count(wa.SKILLS) == 24
    assert not list((home / "skills").glob("new-*"))
    assert "Rejected 24" in handle_pending_subcommand(wa.SKILLS, ["reject", "all"])
    assert wa.pending_count(wa.SKILLS) == 0


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("origin", ["foreground", "background_review"])
def test_non_manifest_leaf_symlink_cannot_create_unapproved_skill(home, flat, origin):
    root = home / "skills" / "existing"
    (root / "references").mkdir()
    target = root / "nested" / "SKILL.md"
    target.parent.mkdir()
    link = root / "references" / "guide.md"
    link.symlink_to(target)
    op = {"action": "write_file", "name": "existing", "file_path": "references/guide.md", "file_content": SAMPLE}
    token = set_current_write_origin(origin)
    try:
        result = json.loads(smt.skill_manage(**op)) if flat else dispatch(op)
    finally:
        reset_current_write_origin(token)
    assert result.get("staged") is True, result
    assert not target.exists() and link.is_symlink()
    assert "Approved 1" in handle_pending_subcommand(wa.SKILLS, ["approve", result["pending_id"]])
    assert target.read_text() == SAMPLE and link.is_symlink()
    assert target in set(iter_skill_index_files(home / "skills", "SKILL.md"))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("action", ["edit", "patch"])
def test_rewriting_dangling_root_manifest_cannot_create_unapproved_nested_skill(home, action):
    root = home / "skills" / "existing"
    target = root / "nested" / "SKILL.md"
    target.parent.mkdir()
    (root / "SKILL.md").unlink()
    (root / "SKILL.md").symlink_to(target)
    result = json.loads(smt.skill_manage(action=action, name="existing", content=SAMPLE))
    assert result.get("staged") is True, result
    assert not target.exists()


@pytest.mark.parametrize("metadata", ["ledger", "usage"])
@pytest.mark.parametrize("flat", [False, True])
def test_metadata_lock_contention_cannot_hang_an_applied_write(home, monkeypatch, metadata, flat):
    from tools import skill_usage, skill_write_approval as policy
    monkeypatch.setattr(policy, "WRITE_WAIT_SECONDS", 0.5)
    path = (home / "skills" / ".locks" / "ledger.lock" if metadata == "ledger"
            else home / "skills" / ".usage.json.lock")
    op = {"action": "patch", "name": "existing", "old_string": "Original body.", "new_string": "Metadata bounded."}
    with ThreadPoolExecutor(max_workers=1) as pool:
        with skill_usage.skill_file_lock(path):
            future = pool.submit(lambda: json.loads(smt.skill_manage(**op)) if flat else dispatch(op))
            # The holder exits before executor teardown even when this assertion fails.
            result = future.result(timeout=3)
    assert result["success"] and not result.get("staged"), result
    assert "Metadata bounded." in (home / "skills" / "existing" / "SKILL.md").read_text()
    assert wa.pending_count(wa.SKILLS) == 0


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("action", ["edit", "patch"])
def test_root_rewrite_predicts_unconfined_actual_leaf_target(home, action):
    root = home / "skills" / "existing"
    target = home / "skills" / "new-sibling" / "SKILL.md"
    target.parent.mkdir()
    (root / "SKILL.md").unlink()
    (root / "SKILL.md").symlink_to(target)
    result = json.loads(smt.skill_manage(action=action, name="existing", content=SAMPLE))
    assert result.get("staged") is True, result
    assert not target.exists()


def test_manifest_write_that_activates_missing_create_root_requires_review(home):
    root = home / "skills" / "existing"
    destination = root / "references" / "sample"
    with (home / "config.yaml").open("a", encoding="utf-8") as config:
        config.write(f"  create_dir: {destination}\n")
    result = dispatch({"action": "write_file", "name": "existing", "file_path": "references/sample/SKILL.md",
                       "file_content": SAMPLE})
    assert result.get("staged") is True, result
    assert not (destination / "SKILL.md").exists()


@pytest.mark.parametrize("rewrite", ["patch-content", "patch-header"])
def test_renamed_manifest_removal_predicts_discovery_but_batch_rejects_clobber(home, rewrite):
    root = home / "skills" / "existing"
    sample = root / "references" / "sample" / "SKILL.md"
    sample.parent.mkdir(parents=True)
    sample.write_text(SAMPLE, encoding="utf-8")
    if rewrite == "patch-header":
        first = {"action": "patch", "name": "existing", "old_string": "name: display", "new_string": "name: renamed"}
    else:
        first = {"action": "patch", "name": "existing",
                 "content": CONTENT.replace("name: display", "name: renamed")}
    operations = [first, {"action": "remove_file", "name": "renamed", "file_path": "SKILL.md"}]
    from tools.skill_write_approval import requires_creation_approval
    # The read-only predictor still resolves the renamed alias and newly exposed sample.
    assert requires_creation_approval(operations)
    result = dispatch(*operations)
    assert not result["success"] and "silently discard" in result["error"], result
    assert not result.get("staged")
    assert (root / "SKILL.md").read_text() == CONTENT
    assert sample.read_text() == SAMPLE
    assert wa.pending_count(wa.SKILLS) == 0


@pytest.mark.platforms("posix")
def test_batch_rollback_restores_real_leaf_target_and_preserves_root_symlink(home):
    root = home / "skills" / "existing"
    sibling = home / "skills" / "sibling" / "SKILL.md"
    sibling.parent.mkdir()
    original = CONTENT.replace("name: display", "name: sibling")
    sibling.write_text(original, encoding="utf-8")
    (root / "SKILL.md").unlink()
    (root / "SKILL.md").symlink_to(sibling)
    result = dispatch(
        {"action": "patch", "name": "existing", "content": original.replace("Original body.", "Whole replacement.")},
        {"action": "patch", "name": "existing", "old_string": "Absent.", "new_string": "Failure."})
    assert not result["success"], result
    assert sibling.read_text() == original
    assert (root / "SKILL.md").read_text() == original
    assert (root / "SKILL.md").is_symlink()


@pytest.mark.parametrize("setting", ["external_dirs", "create_dir"])
@pytest.mark.parametrize("rollback", [False, True])
def test_shared_resource_profiles_cannot_overwrite_acknowledged_edits(home, monkeypatch, setting, rollback):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    shared = home.parent / "shared"
    manifest = shared / "existing" / "SKILL.md"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(CONTENT, encoding="utf-8")
    profiles = []
    for name in ("profile-a", "profile-b"):
        profile = home.parent / name
        profile.mkdir()
        value = [str(shared)] if setting == "external_dirs" else str(shared)
        (profile / "config.yaml").write_text(
            "skills:\n  write_approval: true\n  write_approval_mode: create\n"
            f"  {setting}: {json.dumps(value)}\n", encoding="utf-8")
        profiles.append(profile)
    entered, release, attempted, finished = Event(), Event(), Event(), Event()
    actor = local()
    native_write, native_dispatch = smt._guarded_write, smt._skill_manage_from

    def hold_write(*args, **kwargs):
        if not rollback and getattr(actor, "a", False):
            entered.set()
            assert release.wait(15)
        return native_write(*args, **kwargs)

    def hold_dispatch(payload, **kwargs):
        if rollback and payload.get("old_string") == "Absent.":
            entered.set()
            assert release.wait(15)
        return native_dispatch(payload, **kwargs)

    monkeypatch.setattr(smt, "_guarded_write", hold_write)
    monkeypatch.setattr(smt, "_skill_manage_from", hold_dispatch)

    def write(index):
        actor.a = index == 0
        token = set_hermes_home_override(profiles[index])
        try:
            if index:
                attempted.set()
            patch = {"action": "patch", "name": "existing", "old_string": "Original body.",
                     "new_string": f"Shared {'A' if index == 0 else 'B'}.\nOriginal body."}
            if rollback and index == 0:
                return dispatch(patch, {"action": "patch", "name": "existing", "old_string": "Absent.",
                                        "new_string": "Failure."})
            return dispatch(patch)
        finally:
            reset_hermes_home_override(token)
            if index:
                finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(write, 0)
        try:
            assert entered.wait(15)
            b = pool.submit(write, 1)
            assert attempted.wait(15)
            interleaved = finished.wait(1)
        finally:
            release.set()
        result_a, result_b = a.result(timeout=15), b.result(timeout=15)
    assert not interleaved, result_b
    assert result_b["success"] and not result_b.get("staged"), result_b
    assert "Shared B." in manifest.read_text()
    if rollback:
        assert not result_a["success"] and "Shared A." not in manifest.read_text()
    else:
        assert result_a["success"] and "Shared A." in manifest.read_text()


@pytest.mark.parametrize("setting", ["external_dirs", "create_dir"])
def test_independent_processes_and_profiles_preserve_shared_edits(home, setting):
    import subprocess
    import sys
    shared = home.parent / "shared-processes"
    manifest = shared / "existing" / "SKILL.md"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(CONTENT, encoding="utf-8")
    code = (
        "import json,sys;from tools.skill_manager_tool import skill_manage;"
        "r=json.loads(skill_manage(action='patch',name='existing',old_string='Original body.',"
        "new_string='Original body. Process '+sys.argv[1]+'.'));print(json.dumps(r));"
        "sys.exit(0 if r.get('success') and not r.get('staged') else 2)"
    )
    processes = []
    try:
        for i in range(4):
            profile = home.parent / f"process-profile-{i}"
            profile.mkdir()
            value = [str(shared)] if setting == "external_dirs" else str(shared)
            (profile / "config.yaml").write_text(
                "skills:\n  write_approval: true\n  write_approval_mode: create\n"
                f"  {setting}: {json.dumps(value)}\n", encoding="utf-8")
            env = dict(os.environ, HERMES_HOME=str(profile), HOME=str(home.parent), USERPROFILE=str(home.parent),
                       PYTHONPATH=str(Path(smt.__file__).parents[1]), PYTHONDONTWRITEBYTECODE="1")
            processes.append(subprocess.Popen([sys.executable, "-c", code, str(i)], env=env,
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for process in processes:
            out, err = process.communicate(timeout=30)
            assert process.returncode == 0, (out, err)
            result = json.loads(out.strip())
            assert result["success"] and not result.get("staged"), result
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    for i in range(4):
        assert f"Process {i}." in manifest.read_text()


def test_empty_support_directory_cleanup_is_visible_to_later_operations(home):
    root = home / "skills" / "existing"
    removed = root / "references" / "sample" / "SKILL.md"
    exposed = root / "references" / "other" / "SKILL.md"
    for target in (removed, exposed):
        target.parent.mkdir(parents=True)
        target.write_text(CONTENT, encoding="utf-8")
    result = dispatch(
        {"action": "remove_file", "name": "existing", "file_path": "references/sample/SKILL.md"},
        {"action": "write_file", "name": "existing", "file_path": "references/sample", "file_content": "Ordinary note."},
        {"action": "remove_file", "name": "existing", "file_path": "SKILL.md"})
    assert result["success"] and result["staged"], result
    assert removed.is_file() and (root / "SKILL.md").is_file()
