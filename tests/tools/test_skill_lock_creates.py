"""Cross-profile skill locks for creates and for batches that change which folders are skills: a
folder's lock depends on its real path only, so no create, category, SKILL.md added or removed, or
symlinked category can move it, and nothing is written into the skills tree (split from
test_skill_lock_owning_root.py, whose fixtures and helpers it reuses)."""

import json
import os
import sys

import pytest

from tests.tools.test_skill_lock_owning_root import SKILL, _folder, _lock, _locks_module, _make_fleet, _no_lock_in, _probe


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    return _make_fleet(tmp_path, monkeypatch)


def _brain(fleet, tmp_path):
    brain = tmp_path / "brain"
    (fleet["b"] / "config.yaml").write_text(
        "skills:\n  write_approval: false\n  create_dir: " + str(brain) + "\n", encoding="utf-8")
    return brain


def _probing(monkeypatch, *locks):
    """Record, from another process, whether each lock is held when the batch takes its snapshot and
    when it starts its rollback."""
    import tools.skill_manager_batch as batch
    seen = {}
    real_snapshot, real_rollback = batch._snapshot_skills, batch._rollback

    def probe_then_snapshot(*a, **kw):
        seen["snapshot"] = _probe(*locks)
        return real_snapshot(*a, **kw)

    def probe_then_rollback(*a, **kw):
        seen["rollback"] = _probe(*locks)
        return real_rollback(*a, **kw)

    monkeypatch.setattr(batch, "_snapshot_skills", probe_then_snapshot)
    monkeypatch.setattr(batch, "_rollback", probe_then_rollback)
    return seen


def _create(name, **extra):
    return {"action": "create", "name": name, "content": SKILL.replace("shared-skill", name), **extra}


FAIL = {"action": "patch", "name": "shared-skill", "old_string": "not in the file", "new_string": "X."}


def _files(tree):
    return {p for p in tree.rglob("*") if not p.is_dir()}


def _warmed_files(fleet, tmp_path):
    """The files under tmp_path after one ordinary call, so profile bootstrap files Hermes writes on
    first use (SOUL.md and the like) are not counted as the footprint of the call under test."""
    from tools.skill_manager_tool import skill_manage
    assert json.loads(skill_manage(action="patch", name="shared-skill", old_string="Step 1: Do the thing.",
                                   new_string="Step 1: Do the thing."))["success"] in (True, False)
    return _files(tmp_path)


def _unexpected_files(fleet, new):
    """The new files that are NOT locks: neither the profile's own (<skills>/.locks/*.lock) nor the
    install's folder lock (<root>/.skill_locks/*.lock)."""
    own, folder = fleet["b"] / "skills" / ".locks", fleet["root"] / ".skill_locks"
    return sorted(str(p) for p in new if not (p.parent in (own, folder) and p.suffix == ".lock"))


def test_the_footprint_check_reports_every_file_that_is_not_a_lock(fleet, tmp_path):
    """Negative control for the footprint assertions below."""
    allowed = {fleet["b"] / "skills" / ".locks" / "a.lock", fleet["root"] / ".skill_locks" / "b.lock"}
    forbidden = {tmp_path / "escape" / "x" / "notes.txt", tmp_path / "brain" / ".locks" / "c.lock",
                 fleet["root"] / ".skill_locks" / "d.txt"}
    assert _unexpected_files(fleet, allowed) == []
    assert _unexpected_files(fleet, allowed | forbidden) == sorted(map(str, forbidden))


def test_a_categorized_create_takes_the_lock_of_the_folder_it_creates(fleet, tmp_path):
    brain = _brain(fleet, tmp_path)
    from tools.skill_manager_tool import skill_manage
    created = json.loads(skill_manage(action="create", name="cat-new", category="cat", content=SKILL.replace(
        "shared-skill", "cat-new")))
    assert created["success"] is True, created
    assert (brain / "cat" / "cat-new" / "SKILL.md").exists()
    assert _folder(fleet, brain / "cat" / "cat-new").exists() and _no_lock_in(brain)


@pytest.mark.skipif(sys.platform == "win32", reason="probes with fcntl")
def test_a_batch_creating_into_a_category_holds_that_folders_lock_through_its_rollback(fleet, tmp_path, monkeypatch):
    """A batch that creates cat-new in category cat, patches it, then fails: the lock a later patch of
    cat-new takes must be held from before the snapshot through the rollback, or another profile's
    patch can land in between and be erased by the rollback."""
    from tools.skill_manager_tool import skill_manage
    brain = _brain(fleet, tmp_path)
    seen = _probing(monkeypatch, _folder(fleet, brain / "cat" / "cat-new"))
    result = json.loads(skill_manage(action="create", name="", operations=[
        _create("cat-new", category="cat"),
        {"action": "patch", "name": "cat-new", "old_string": "Step 1: Do the thing.", "new_string": "C."},
        {"action": "patch", "name": "cat-new", "old_string": "not in the file", "new_string": "X."}]))
    assert result["success"] is False, result
    assert not (brain / "cat" / "cat-new").exists(), "the failed batch's create was rolled back"
    assert seen == {"snapshot": ["held"], "rollback": ["held"]}, seen


@pytest.mark.parametrize("category", ["..", "../escape", 5])
def test_a_rejected_category_is_refused_and_writes_nothing(fleet, tmp_path, category):
    """Including a non-string category: planning the create's lock must not raise before create's
    own validation refuses it (main returns the same JSON error)."""
    brain = _brain(fleet, tmp_path)
    brain.mkdir()
    from tools.skill_manager_tool import skill_manage
    before = _warmed_files(fleet, tmp_path)
    single = json.loads(skill_manage(action="create", name="cat-new", category=category, content=SKILL))
    batched = json.loads(skill_manage(action="create", name="", operations=[
        {"action": "create", "name": "cat-new", "category": category, "content": SKILL}]))
    assert single["success"] is False and batched["success"] is False, (single, batched)
    assert _unexpected_files(fleet, _files(tmp_path) - before) == []


@pytest.mark.parametrize("bad", ["../escape/x", "ABSOLUTE"])
def test_a_batch_create_name_that_create_rejects_places_no_lock(fleet, tmp_path, bad):
    """A traversal or absolute create name in a batch is refused (as on main); the only files the
    call leaves are the profile's own lock and the install's folder lock (nothing in or near a
    skills tree, nothing where the name points)."""
    brain = _brain(fleet, tmp_path)
    brain.mkdir()
    name = str(tmp_path / "abs" / "x") if bad == "ABSOLUTE" else bad
    from tools.skill_manager_tool import skill_manage
    before = _warmed_files(fleet, tmp_path)
    result = json.loads(skill_manage(action="create", name="", operations=[
        {"action": "create", "name": name, "content": SKILL}]))
    assert result["success"] is False, result
    assert _unexpected_files(fleet, _files(tmp_path) - before) == []


def test_a_batch_can_create_a_skill_inside_a_skill_it_creates(fleet, tmp_path):
    """create(outer) then create(inner, category=outer) succeeds as on main, and nothing but the
    two skills is written into the tree."""
    brain = _brain(fleet, tmp_path)
    from tools.skill_manager_tool import skill_manage
    result = json.loads(skill_manage(action="create", name="", operations=[
        _create("outer-skill"), _create("inner-skill", category="outer-skill")]))
    assert result["success"] is True, result
    assert (brain / "outer-skill" / "inner-skill" / "SKILL.md").exists()
    assert sorted(p.name for p in (brain / "outer-skill").iterdir()) == ["SKILL.md", "inner-skill"]
    assert _folder(fleet, brain / "outer-skill" / "inner-skill").exists() and _no_lock_in(brain)


@pytest.mark.skipif(sys.platform == "win32", reason="probes with fcntl and symlinks")
def test_a_batch_creating_through_a_symlinked_category_into_folders_it_creates_succeeds(fleet, tmp_path, monkeypatch):
    """Category deep is a symlink to outer/middle, which the batch itself creates before creating
    inner in deep. Planning inner's lock writes nothing into the tree (main succeeds; so must this),
    and the lock it holds is the one every later writer of outer/middle/inner takes."""
    from tools.skill_manager_tool import skill_manage
    brain = _brain(fleet, tmp_path)
    brain.mkdir()
    (brain / "deep").symlink_to(brain / "outer-skill" / "middle-skill", target_is_directory=True)
    seen = _probing(monkeypatch, _folder(fleet, brain / "outer-skill" / "middle-skill" / "inner-skill"))
    result = json.loads(skill_manage(action="create", name="", operations=[
        _create("outer-skill"), _create("middle-skill", category="outer-skill"), _create("inner-skill", category="deep")]))
    assert result["success"] is True, result
    assert (brain / "outer-skill" / "middle-skill" / "inner-skill" / "SKILL.md").exists()
    assert seen == {"snapshot": ["held"]}, seen


def test_a_failed_create_leaves_nothing_in_the_skills_tree(fleet, tmp_path):
    brain = _brain(fleet, tmp_path)
    from tools.skill_manager_tool import skill_manage
    failed = json.loads(skill_manage(action="create", name="cat-new", category="cat", content="no frontmatter"))
    assert failed["success"] is False, failed
    assert not brain.exists() or not any(brain.iterdir())


@pytest.mark.parametrize("entry", ["file", "dir"])
def test_a_folder_holding_a_dot_locks_entry_is_not_adopted_by_a_create(fleet, tmp_path, entry):
    """As on main, a create only adopts an EMPTY leftover folder: a user's file or folder named
    .locks makes it someone else's (a later delete would remove it with the skill)."""
    brain = _brain(fleet, tmp_path)
    (brain / "cat").mkdir(parents=True)
    if entry == "file":
        (brain / "cat" / ".locks").write_text("user data", encoding="utf-8")
    else:
        (brain / "cat" / ".locks").mkdir()
        (brain / "cat" / ".locks" / "notes.txt").write_text("user data", encoding="utf-8")
    from tools.skill_manager_tool import skill_manage
    created = json.loads(skill_manage(action="create", name="cat", content=SKILL.replace("shared-skill", "cat")))
    assert created["success"] is False, created
    assert not (brain / "cat" / "SKILL.md").exists()


def _existing_nested(brain, outer_is_skill):
    """brain/outer-skill/inner-skill is a skill; outer-skill is a skill too, or only a category."""
    inner = brain / "outer-skill" / "inner-skill"
    inner.mkdir(parents=True)
    (inner / "SKILL.md").write_text(SKILL.replace("shared-skill", "inner-skill"), encoding="utf-8")
    if outer_is_skill:
        (brain / "outer-skill" / "SKILL.md").write_text(SKILL.replace("shared-skill", "outer-skill"), encoding="utf-8")
    shared = brain / "shared-skill"
    shared.mkdir()
    (shared / "SKILL.md").write_text(SKILL, encoding="utf-8")
    return inner


PATCH_INNER = {"action": "patch", "name": "inner-skill", "old_string": "Step 1: Do the thing.", "new_string": "Inner."}


@pytest.mark.skipif(sys.platform == "win32", reason="probes with fcntl")
@pytest.mark.parametrize("ops", ["create-outer-first", "patch-first", "remove-outer-skill-md"])
def test_a_batch_changing_which_folders_are_skills_never_moves_a_skills_lock(fleet, tmp_path, monkeypatch, ops):
    """A batch that makes an existing skill's enclosing folder a skill (create, which fails here:
    outer is not empty) or stops it being one (remove_file of outer's SKILL.md), then fails: the lock
    another profile's writer of inner takes, before or after that change, is the one the batch
    holds from before its snapshot through its rollback, so the rollback cannot erase that write."""
    from tools.skill_manager_tool import skill_manage
    locks = _locks_module()
    brain = _brain(fleet, tmp_path)
    inner = _existing_nested(brain, outer_is_skill=ops == "remove-outer-skill-md")
    lock = _folder(fleet, inner)
    seen = _probing(monkeypatch, lock)
    if ops == "remove-outer-skill-md":
        batch_ops = [PATCH_INNER, {"action": "remove_file", "name": "outer-skill", "file_path": "SKILL.md"}, FAIL]
    else:
        create = _create("outer-skill")
        batch_ops = [create, PATCH_INNER] if ops == "create-outer-first" else [PATCH_INNER, create]
    result = json.loads(skill_manage(action="create", name="", operations=batch_ops))
    assert result["success"] is False, result
    assert seen == {"snapshot": ["held"], "rollback": ["held"]}, seen
    assert locks._target_folder_lock("inner-skill") == lock, "a writer of inner now takes another lock"


def test_a_folder_lock_depends_on_the_real_path_only(fleet, tmp_path):
    """No SKILL.md above the folder, no spelling of it, and no state left by an earlier call changes
    its lock."""
    locks = _locks_module()
    assert hasattr(locks, "_folder_lock"), "no real-path folder lock in this code"
    brain = tmp_path / "brain"
    inner = brain / "outer-skill" / "inner-skill"
    inner.mkdir(parents=True)
    before = locks._folder_lock(inner)
    (brain / "outer-skill" / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (inner / "SKILL.md").write_text(SKILL, encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(brain / "outer-skill", target_is_directory=True)
    assert locks._folder_lock(inner) == before == locks._folder_lock(link / "inner-skill")
    assert before == _folder(fleet, inner) and os.path.dirname(before) == str(fleet["root"] / ".skill_locks")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock semantics")
def test_a_batch_op_whose_lock_stays_busy_fails_and_the_batch_rolls_back(fleet, tmp_path, monkeypatch):
    """Profile b has a local foo shadowing the shared foo. A batch removes local foo's SKILL.md, so its
    next op resolves foo to the SHARED folder, whose lock the batch did not take and profile a holds.
    That op must fail as a result after the bounded wait (not raise past the batch, which skipped
    the rollback and deleted the snapshot), writing nothing to the shared foo; the batch then reaches
    its rollback. (Restoring across a lookup change is a separate upstream fix, #156492.)"""
    import importlib.util
    from tests.tools.test_skill_lock_owning_root import HOLD, REPO, _wait_for
    import subprocess
    locks = _locks_module()
    monkeypatch.setattr(locks, "_OUT_OF_ORDER_WAIT", 0.5)
    shared_foo = fleet["shared"] / "foo"
    shared_foo.mkdir()
    (shared_foo / "SKILL.md").write_text(SKILL.replace("shared-skill", "foo"), encoding="utf-8")
    local_foo = fleet["b"] / "skills" / "foo"
    local_foo.mkdir()
    original = SKILL.replace("shared-skill", "foo").replace("Do the thing.", "Local thing.")
    (local_foo / "SKILL.md").write_text(original, encoding="utf-8")
    from tools.skill_manager_tool import _find_skill, skill_manage
    found = _find_skill("foo")
    assert importlib.util.find_spec("tools.skill_manager_locks") and found and str(found["path"]) == str(local_foo), found
    import tools.skill_manager_batch as batch
    rolled_back = []
    real_rollback = batch._rollback

    def record_rollback(*a, **kw):  # what the shared skill holds when the rollback starts
        rolled_back.append((shared_foo / "SKILL.md").read_text(encoding="utf-8"))
        return real_rollback(*a, **kw)

    monkeypatch.setattr(batch, "_rollback", record_rollback)
    ready, release = tmp_path / "ready", tmp_path / "release"
    env = {**os.environ, "HERMES_HOME": str(fleet["a"]), "PYTHONPATH": str(REPO)}
    holder = subprocess.Popen([sys.executable, "-c", HOLD, str(ready), str(release), "foo"], cwd=REPO, env=env)
    try:
        _wait_for(ready, holder)
        try:
            outcome = json.loads(skill_manage(action="patch", name="", operations=[
                {"action": "remove_file", "name": "foo", "file_path": "SKILL.md"},
                {"action": "patch", "name": "foo", "old_string": "Do the thing.", "new_string": "Shared, changed."}]))
        except TimeoutError as exc:
            outcome = {"raised": repr(exc)}
    finally:
        release.write_text("go")
        holder.wait(timeout=30)
    assert outcome.get("success") is False, outcome
    assert rolled_back == [SKILL.replace("shared-skill", "foo")], "no rollback, or the busy op wrote the shared foo"


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, non-root")
def test_a_folder_lock_that_cannot_be_acquired_falls_back_inside_a_batch(fleet, monkeypatch):
    """Acquiring the folder lock fails (read-only lock directory, e.g. permissions changed after any
    access check): the write keeps the profile's own lock as on main, and a failing batch still rolls
    back instead of an OSError skipping its rollback."""
    locks = _locks_module()
    lock_dir = fleet["root"] / ".skill_locks"
    lock_dir.mkdir(parents=True)
    lock_dir.chmod(0o555)
    # Code that checks access before acquiring: that check passed before the permissions changed.
    monkeypatch.setattr(locks, "_can_open_lock", lambda _lock: True, raising=False)
    from tools.skill_manager_tool import skill_manage
    skill_md = fleet["shared"] / "shared-skill" / "SKILL.md"
    original = skill_md.read_text(encoding="utf-8")
    try:
        try:
            single = json.loads(skill_manage(action="patch", name="shared-skill",
                                             old_string="Step 1: Do the thing.", new_string="Step 1: Own lock."))
            patched = skill_md.read_text(encoding="utf-8")
            skill_md.write_text(original, encoding="utf-8")
            batch = json.loads(skill_manage(action="patch", name="", operations=[
                {"action": "patch", "name": "shared-skill", "old_string": "Step 1: Do the thing.", "new_string": "B."},
                FAIL]))
        except OSError as exc:
            single, patched, batch = {"raised": repr(exc)}, None, None
    finally:
        lock_dir.chmod(0o755)
    assert single.get("success") is True and "Step 1: Own lock." in patched, single
    assert batch["success"] is False, batch
    assert skill_md.read_text(encoding="utf-8") == original, "the failed batch was not rolled back"
    assert _lock(fleet["b"] / "skills", "shared-skill").exists()
