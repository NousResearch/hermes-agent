"""Creation-only approval uses real config, tool dispatch, files and review commands."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import skill_manager_tool, write_approval as wa
from tools.registry import registry
from tools.skill_provenance import reset_current_write_origin, set_current_write_origin


CONTENT = "---\nname: {name}\ndescription: Test approval policy.\n---\n\n# Test\n\nOriginal body.\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _configure(home)
    path = home / "skills" / "existing" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(CONTENT.format(name="existing"), encoding="utf-8")
    return home


def _configure(home, mode="create", enabled=True):
    # Real on-disk config: the policy must not read import-time or launch-profile state.
    (home / "config.yaml").write_text(
        f"skills:\n  write_approval: {str(enabled).lower()}\n"
        f"  write_approval_mode: {mode}\n", encoding="utf-8")


def _dispatch(*ops):
    result = registry.dispatch("skill_manage", {"operations": list(ops)})
    assert isinstance(result, str)
    return json.loads(result)


def _create(name):
    return {"action": "create", "name": name, "content": CONTENT.format(name=name)}


def _patch(name="existing", old="Original body.", new="Updated body."):
    return {"action": "patch", "name": name, "old_string": old, "new_string": new}


@contextmanager
def _origin(origin):
    from tools.skill_manager_guards import _reset_background_review_read_marks
    token = set_current_write_origin(origin)
    _reset_background_review_read_marks()
    try:
        yield
    finally:
        reset_current_write_origin(token)


@pytest.mark.parametrize("origin", ["foreground", "background_review"])
@pytest.mark.parametrize("shape", ["flat", "batch"])
def test_existing_edits_apply_but_creations_wait(home, origin, shape):
    from tools.skill_manager_guards import mark_background_review_skill_read
    path = home / "skills" / "existing" / "SKILL.md"
    with _origin(origin):
        mark_background_review_skill_read(path)
        if shape == "flat":
            edited = json.loads(skill_manager_tool.skill_manage(**_patch()))
            created = json.loads(skill_manager_tool.skill_manage(**_create("new-skill")))
        else:
            edited = _dispatch(_patch())
            created = _dispatch(_create("new-skill"))
    assert edited["success"] and not edited.get("staged"), edited
    assert "Updated body." in path.read_text(encoding="utf-8")
    assert created["staged"] is True, created
    assert not (home / "skills" / "new-skill").exists()
    assert wa.pending_count(wa.SKILLS) == 1
    record = wa.get_pending(wa.SKILLS, created["pending_id"])
    assert record is not None
    assert record["origin"] == origin


@pytest.mark.parametrize("review", ["approve", "reject"])
def test_mixed_batch_stays_atomic_until_review(home, review):
    path = home / "skills" / "existing" / "SKILL.md"
    original = path.read_text(encoding="utf-8")
    result = _dispatch(_patch(), _create("new-skill"), {
        "action": "write_file", "name": "new-skill", "file_path": "references/guide.md",
        "file_content": "New knowledge.\n"})
    assert result["staged"] is True, result
    pid = result["pending_id"]
    assert path.read_text(encoding="utf-8") == original
    assert not (home / "skills" / "new-skill").exists()
    assert pid in handle_pending_subcommand(wa.SKILLS, ["pending"])
    diff = handle_pending_subcommand(wa.SKILLS, ["diff", pid])
    assert diff is not None and "New knowledge." in diff
    out = handle_pending_subcommand(wa.SKILLS, [review, pid])
    assert out is not None
    assert wa.pending_count(wa.SKILLS) == 0, out
    if review == "approve":
        assert "Approved 1" in out
        assert "Updated body." in path.read_text(encoding="utf-8")
        assert (home / "skills" / "new-skill" / "references" / "guide.md").read_text() == "New knowledge.\n"
    else:
        assert "Rejected" in out
        assert path.read_text(encoding="utf-8") == original
        assert not (home / "skills" / "new-skill").exists()
    # Replay's bypass must not leak into the next tool call.
    assert _dispatch(_create("another-new"))["staged"] is True


def test_existing_support_files_and_full_rewrites_are_automatic(home):
    path = home / "skills" / "existing" / "SKILL.md"
    rewritten = CONTENT.format(name="existing").replace("Original body.", "Rewritten body.")
    result = _dispatch({"action": "patch", "name": "existing", "content": rewritten}, {
        "action": "write_file", "name": "existing", "file_path": "references/guide.md",
        "file_content": "Existing skill knowledge.\n"})
    assert result["success"] and not result.get("staged"), result
    assert path.read_text() == rewritten
    result = _dispatch({"action": "remove_file", "name": "existing", "file_path": "references/guide.md"})
    assert result["success"] and not result.get("staged"), result
    assert not (path.parent / "references" / "guide.md").exists()
    assert wa.pending_count(wa.SKILLS) == 0


@pytest.mark.parametrize("mode,enabled,staged", [("all", True, True), ("create", False, False)])
def test_original_all_and_disabled_policies_are_preserved(home, mode, enabled, staged):
    _configure(home, mode=mode, enabled=enabled)
    result = _dispatch(_patch(), _create("new-skill"))
    assert result["success"], result
    assert bool(result.get("staged")) == staged
    assert (home / "skills" / "new-skill").exists() is not staged


def test_missing_scope_stages_and_memory_policy_is_unchanged(home):
    assert wa.evaluate_gate(wa.SKILLS).stage is True
    (home / "config.yaml").write_text(
        "skills:\n  write_approval: true\n  write_approval_mode: create\n"
        "memory:\n  write_approval: true\n", encoding="utf-8")
    with _origin("background_review"):
        assert wa.evaluate_gate(wa.MEMORY, inline_summary="Remember this").stage is True


@pytest.mark.parametrize("mode", ["typo", "null", "[]"])
def test_invalid_policy_refuses_before_mutation(home, mode):
    _configure(home, mode=mode)
    path = home / "skills" / "existing" / "SKILL.md"
    original = path.read_text()
    result = _dispatch(_patch())
    assert result["success"] is False, result
    assert result["error_type"] == "invalid_config"
    assert "write_approval_mode" in result["error"]
    assert path.read_text() == original
    assert wa.pending_count(wa.SKILLS) == 0


def test_threaded_creation_and_edit_do_not_share_replay_bypass(home):
    from threading import Barrier
    barrier = Barrier(2)

    def run(op):
        barrier.wait(timeout=10)
        return _dispatch(op)

    with ThreadPoolExecutor(max_workers=2) as pool:
        edit = pool.submit(run, _patch())
        create = pool.submit(run, _create("new-skill"))
        edited, created = edit.result(timeout=15), create.result(timeout=15)
    assert edited["success"] and not edited.get("staged"), edited
    assert created["staged"] is True, created
    assert not (home / "skills" / "new-skill").exists()
    assert "Updated body." in (home / "skills" / "existing" / "SKILL.md").read_text()


def test_scope_is_profile_local_across_a_b_a_switches(home):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    other = home.parent / "other-home"
    other.mkdir()
    _configure(other, mode="all")
    path = other / "skills" / "existing" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(CONTENT.format(name="existing"))
    for active, old, new, staged in [
        (home, "Original body.", "First edit.", False),
        (other, "Original body.", "Other edit.", True),
        (home, "First edit.", "Second edit.", False),
    ]:
        token = set_hermes_home_override(active)
        try:
            result = _dispatch(_patch(old=old, new=new))
            assert bool(result.get("staged")) is staged, result
        finally:
            reset_hermes_home_override(token)
    assert "Second edit." in (home / "skills" / "existing" / "SKILL.md").read_text()
    assert "Original body." in path.read_text()
    assert wa.pending_count(wa.SKILLS) == 0
    assert len(list((other / "pending" / "skills").glob("*.json"))) == 1


def test_pending_creation_is_reviewable_in_a_fresh_process(home):
    import os
    import subprocess
    import sys
    result = _dispatch(_create("new-skill"))
    pid = result["pending_id"]
    code = (
        "from hermes_cli.write_approval_commands import handle_pending_subcommand; "
        f"print(handle_pending_subcommand('skills', ['approve', '{pid}']))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], env={**os.environ, "HERMES_HOME": str(home)},
        capture_output=True, text=True, timeout=30, check=True)
    assert "Approved 1" in completed.stdout
    assert (home / "skills" / "new-skill" / "SKILL.md").read_text() == CONTENT.format(name="new-skill")
    assert wa.pending_count(wa.SKILLS) == 0


def test_existing_pending_edits_do_not_auto_apply_when_scope_changes(home):
    _configure(home, mode="all")
    staged = _dispatch(_patch())
    pid = staged["pending_id"]
    pending_path = home / "pending" / "skills" / f"{pid}.json"
    record = pending_path.read_bytes()
    _configure(home, mode="create")
    result = _dispatch(_patch(new="Fresh edit."))
    assert result["success"] and not result.get("staged"), result
    assert pending_path.read_bytes() == record
    status = handle_pending_subcommand(wa.SKILLS, [])
    assert status is not None and "scope: create" in status
    assert pid in handle_pending_subcommand(wa.SKILLS, ["pending"])


def test_omitted_mode_preserves_all_write_approval(home):
    (home / "config.yaml").write_text("skills:\n  write_approval: true\n")
    assert _dispatch(_patch())["staged"] is True


@pytest.mark.parametrize("origin", ["foreground", "background_review"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("file_path", ["nested-skill/SKILL.md", "./nested-skill/SKILL.md"])
def test_new_nested_skill_via_write_file_requires_approval(home, origin, flat, file_path):
    content = CONTENT.format(name="nested-skill")
    operation = {"action": "write_file", "name": "existing", "file_path": file_path,
                 "file_content": content}
    with _origin(origin):
        result = json.loads(skill_manager_tool.skill_manage(**operation)) if flat else _dispatch(operation)
    assert result.get("staged") is True, result
    assert not (home / "skills" / "existing" / file_path).exists()
    assert skill_manager_tool._find_skill("nested-skill") is None
    pid = result["pending_id"]
    diff = handle_pending_subcommand(wa.SKILLS, ["diff", pid])
    assert diff is not None and "Original body." in diff
    approved = handle_pending_subcommand(wa.SKILLS, ["approve", pid])
    assert approved is not None and "Approved 1" in approved
    assert (home / "skills" / "existing" / file_path).read_text() == content
    assert skill_manager_tool._find_skill("nested-skill") is not None


def test_nested_creation_stages_whole_batch_and_existing_nested_rewrite_is_automatic(home):
    content = CONTENT.format(name="nested-skill")
    operation = {"action": "write_file", "name": "existing",
                 "file_path": "nested-skill/SKILL.md", "file_content": content}
    result = _dispatch(_patch(), operation)
    assert result.get("staged") is True, result
    assert "Original body." in (home / "skills" / "existing" / "SKILL.md").read_text()
    assert not (home / "skills" / "existing" / "nested-skill").exists()
    pid = result["pending_id"]
    approved = handle_pending_subcommand(wa.SKILLS, ["approve", pid])
    assert approved is not None and "Approved 1" in approved
    result = _dispatch({**operation, "file_content": content.replace("Original body.", "New body.")})
    assert result["success"] and not result.get("staged"), result
    assert wa.pending_count(wa.SKILLS) == 0
    assert "New body." in (home / "skills" / "existing" / "nested-skill" / "SKILL.md").read_text()


def test_support_sample_is_automatic_but_exposing_it_as_a_skill_needs_approval(home):
    sample = {"action": "write_file", "name": "existing",
              "file_path": "references/sample/SKILL.md", "file_content": CONTENT.format(name="sample")}
    result = _dispatch(sample)
    assert result["success"] and not result.get("staged"), result
    assert skill_manager_tool._find_skill("sample") is None
    result = _dispatch({"action": "remove_file", "name": "existing", "file_path": "SKILL.md"})
    assert result.get("staged") is True, result
    assert (home / "skills" / "existing" / "SKILL.md").exists()
    assert skill_manager_tool._find_skill("sample") is None
    approved = handle_pending_subcommand(wa.SKILLS, ["approve", result["pending_id"]])
    assert approved is not None and "Approved 1" in approved
    assert skill_manager_tool._find_skill("sample") is not None


@pytest.mark.parametrize("origin", ["foreground", "background_review"])
def test_batch_support_write_then_manifest_removal_cannot_expose_new_skill(home, origin):
    from tools.skills_tool import skill_view
    with _origin(origin):
        skill_view("existing")
        result = _dispatch(
            {"action": "write_file", "name": "existing", "file_path": "references/sample/SKILL.md",
             "file_content": CONTENT.format(name="sample")},
            {"action": "remove_file", "name": "existing", "file_path": "SKILL.md"})
    assert result.get("staged") is True, result
    assert (home / "skills" / "existing" / "SKILL.md").exists()
    assert not (home / "skills" / "existing" / "references" / "sample").exists()
    assert skill_manager_tool._find_skill("sample") is None


@pytest.mark.parametrize("origin", ["foreground", "background_review"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.platforms("posix")
def test_symlinked_support_directory_cannot_hide_new_discoverable_manifest(home, origin, flat):
    root = home / "skills" / "existing"
    (root / "nested-skill").mkdir()
    (root / "references").mkdir()
    (root / "references" / "alias").symlink_to(root / "nested-skill", target_is_directory=True)
    operation = {"action": "write_file", "name": "existing",
                 "file_path": "references/alias/SKILL.md", "file_content": CONTENT.format(name="nested-skill")}
    with _origin(origin):
        result = json.loads(skill_manager_tool.skill_manage(**operation)) if flat else _dispatch(operation)
    assert result.get("staged") is True, result
    assert not (root / "nested-skill" / "SKILL.md").exists()
    assert skill_manager_tool._find_skill("nested-skill") is None


@pytest.mark.parametrize("origin", ["foreground", "background_review"])
@pytest.mark.parametrize("flat", [False, True])
def test_permanently_excluded_sample_does_not_gate_manifest_removal(home, origin, flat):
    from tools.skills_tool import skill_view
    path = home / "skills" / "existing" / "scripts" / ".venv" / "sample" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(CONTENT.format(name="sample"))
    operation = {"action": "remove_file", "name": "existing", "file_path": "SKILL.md"}
    with _origin(origin):
        skill_view("existing")
        result = json.loads(skill_manager_tool.skill_manage(**operation)) if flat else _dispatch(operation)
    assert result["success"] and not result.get("staged"), result
    assert not (home / "skills" / "existing" / "SKILL.md").exists()
    assert wa.pending_count(wa.SKILLS) == 0
    assert skill_manager_tool._find_skill("sample") is None


def test_manifest_gate_rechecks_after_waiting_for_a_concurrent_support_writer(home, monkeypatch):
    import threading
    from tools import skill_write_approval as policy
    queued = threading.Event()
    actor = threading.local()
    native_transaction = policy.creation_transaction

    @contextmanager
    def observed_transaction(args=None):
        if getattr(actor, "remover", False):
            queued.set()
        with native_transaction(args):
            yield

    monkeypatch.setattr(policy, "creation_transaction", observed_transaction)

    def remove():
        actor.remover = True
        return json.loads(skill_manager_tool.skill_manage(action="remove_file", name="existing", file_path="SKILL.md"))

    with ThreadPoolExecutor(max_workers=1) as pool:
        with native_transaction():
            pending_remove = pool.submit(remove)
            assert queued.wait(10), "remover did not reach the transaction lock"
            sample = _dispatch({"action": "write_file", "name": "existing",
                                "file_path": "references/sample/SKILL.md", "file_content": CONTENT.format(name="sample")})
            assert sample["success"] and not sample.get("staged"), sample
        result = pending_remove.result(timeout=15)
    assert result.get("staged") is True, result
    assert (home / "skills" / "existing" / "SKILL.md").exists()
    assert skill_manager_tool._find_skill("sample") is None
    assert wa.pending_count(wa.SKILLS) == 1
