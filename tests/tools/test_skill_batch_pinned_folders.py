"""An atomic ``skill_manage`` batch acts only on the skill folders it snapshotted.

A batch resolves each name once, snapshots that folder, and must both run its ops in and roll back
into exactly those folders. Resolving a name again mid-batch can land somewhere else: when a local
skill shadows a same-named skill in ``skills.external_dirs``, removing the local SKILL.md reveals the
external copy. A later op must not write that external copy (it was never snapshotted, so a rollback
could not undo it), and a rollback must not "restore" the local snapshot over it.
"""

import json

import pytest


def _md(name, body):
    return f"---\nname: {name}\ndescription: Use when testing batch folders.\n---\n\n# {name}\n\n{body}\n"


@pytest.fixture
def shadowed(tmp_path, monkeypatch):
    """Local ``foo`` shadowing an external ``foo`` that has a reference file of its own."""
    home = tmp_path / ".hermes"
    local = home / "skills"
    (local / "foo").mkdir(parents=True)
    (local / "foo" / "SKILL.md").write_text(_md("foo", "Local."), encoding="utf-8")
    vault = tmp_path / "vault"
    (vault / "foo" / "references").mkdir(parents=True)
    (vault / "foo" / "SKILL.md").write_text(_md("foo", "External."), encoding="utf-8")
    (vault / "foo" / "references" / "ext.md").write_text("External reference.\n", encoding="utf-8")
    (home / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {vault}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_hermes_home_cache", None, raising=False)
    from agent import skill_utils as su
    from tools import skills_tool
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", local)
    monkeypatch.setattr(smt, "SKILLS_DIR", local)
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()
    yield {"local": local / "foo", "external": vault / "foo", "tmp": tmp_path}
    su._external_dirs_cache_clear()


def _tree(folder):
    if not folder.exists():
        return "MISSING"
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def _batch(*ops):
    from tools.skill_manager_tool import skill_manage
    return json.loads(skill_manage(action="", name="", operations=list(ops)))


REVEAL = {"action": "remove_file", "name": "foo", "file_path": "SKILL.md"}


def test_a_rollback_after_a_shadow_reveal_leaves_the_external_skill_where_it_is(shadowed):
    """main: the rollback resolves `foo` again, gets the external folder, and moves it into the profile."""
    local, external = _tree(shadowed["local"]), _tree(shadowed["external"])
    out = _batch(REVEAL, {"action": "patch", "name": "foo", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False, out
    assert _tree(shadowed["external"]) == external
    assert _tree(shadowed["local"]) == local
    assert not list(shadowed["tmp"].rglob("*.rollback-*")), "a half-done rollback was left behind"


@pytest.mark.parametrize("op", [
    {"action": "write_file", "name": "foo", "file_path": "references/new.md", "file_content": "Batch.\n"},
    {"action": "patch", "name": "foo", "old_string": "External.", "new_string": "Batch."},
    {"action": "remove_file", "name": "foo", "file_path": "references/ext.md"},
], ids=["write_file", "patch", "remove_file"])
def test_an_op_whose_name_now_resolves_to_another_folder_is_refused(shadowed, op):
    """main: the op writes the revealed external copy, which the batch never snapshotted."""
    local, external = _tree(shadowed["local"]), _tree(shadowed["external"])
    out = _batch(REVEAL, op)
    assert out.get("success") is False and out.get("failed_index") == 1, out
    assert "did not snapshot or create" in out.get("error", ""), out
    assert _tree(shadowed["external"]) == external
    assert _tree(shadowed["local"]) == local


def test_a_batch_that_stays_in_its_folders_is_unchanged(shadowed):
    out = _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Patched."},
                 {"action": "write_file", "name": "foo", "file_path": "references/r.md", "file_content": "R.\n"})
    assert out.get("success") is True, out
    assert b"Patched." in (shadowed["local"] / "SKILL.md").read_bytes()
    assert (shadowed["local"] / "references" / "r.md").exists()


def test_a_skill_created_in_the_batch_is_rolled_back_from_where_it_was_created(shadowed):
    content = _md("bar", "New.")
    out = _batch({"action": "create", "name": "bar", "content": content},
                 {"action": "patch", "name": "bar", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False, out
    assert not (shadowed["local"].parent / "bar").exists()


def test_a_name_that_found_no_skill_at_the_start_cannot_reach_one_later(shadowed):
    """Two skills share the display name `common`, so the name finds nothing when the batch starts and
    has no snapshot. Renaming one of them makes `common` find the other: an op on it would change a
    folder the batch never snapshotted, so it is refused and both folders come back byte-identical."""
    local_root = shadowed["local"].parent
    for folder in ("a", "b"):
        (local_root / folder).mkdir()
        (local_root / folder / "SKILL.md").write_text(_md("common", f"Body {folder}."), encoding="utf-8")
    a, b = _tree(local_root / "a"), _tree(local_root / "b")
    out = _batch({"action": "patch", "name": "a", "old_string": "name: common", "new_string": "name: renamed"},
                 {"action": "patch", "name": "common", "old_string": "Body b.", "new_string": "Changed."},
                 {"action": "patch", "name": "a", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False and out.get("failed_index") == 1, out
    assert "snapshot" in out.get("error", ""), out
    assert _tree(local_root / "a") == a
    assert _tree(local_root / "b") == b


@pytest.mark.parametrize("first", [
    {"action": "remove_file", "name": "baz", "file_path": "SKILL.md"},
    {"action": "write_file", "name": "baz", "file_path": "references/new.md", "file_content": "Batch.\n"},
], ids=["skill-md-removed", "skill-still-found"])
def test_a_rollback_never_touches_a_folder_next_to_the_skill(shadowed, first):
    """The rollback moves the broken folder aside before restoring the snapshot. A folder that already
    sits next to the skill under a rollback-looking name was never snapshotted and must survive, and
    the skill itself comes back byte-identical (main: a removed SKILL.md could not be restored)."""
    local_root = shadowed["local"].parent
    (local_root / "baz" / "references").mkdir(parents=True)
    (local_root / "baz" / "SKILL.md").write_text(_md("baz", "Baz."), encoding="utf-8")
    (local_root / "baz" / "references" / "a.md").write_text("Keep me.\n", encoding="utf-8")
    neighbour = local_root / "baz.rollback-broken"
    neighbour.mkdir()
    (neighbour / "notes.md").write_text("Not the batch's.\n", encoding="utf-8")
    baz, kept = _tree(local_root / "baz"), _tree(neighbour)
    out = _batch(first, {"action": "patch", "name": "baz", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False, out
    assert _tree(neighbour) == kept
    assert _tree(local_root / "baz") == baz
    assert not list(local_root.glob(".baz.rollback-*")), "this rollback's aside folder was left behind"


@pytest.mark.parametrize("make_name", [
    lambda outside: str(outside),                   # absolute name: rejected before anything is written
    lambda outside: "../" + outside.parent.name + "/" + outside.name,  # traversal: rejected the same way
    lambda outside: "taken",                        # occupied folder (found, so snapshotted): a guard
], ids=["absolute-name", "traversal-name", "occupied-folder"])
def test_a_create_that_was_refused_leaves_its_target_folder_alone(shadowed, make_name):
    """Rollback removes a skill only if this batch's create of it succeeded. A planned create target
    that the create refused (nothing written) belongs to someone else and comes back byte-identical."""
    outside = shadowed["local"].parent.parent / "outside" / "victim"  # next to the skills folder
    taken = shadowed["local"].parent / "taken"
    for folder in (outside, taken):
        (folder / "notes").mkdir(parents=True)
        (folder / "SKILL.md").write_text("Not a skill this batch made.\n", encoding="utf-8")
        (folder / "notes" / "keep.md").write_text("Keep.\n", encoding="utf-8")
    name = make_name(outside)
    before = {str(f): _tree(f) for f in (outside, taken)}
    out = _batch({"action": "create", "name": name, "content": _md("victim", "Batch.")},
                 {"action": "patch", "name": "foo", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False and out.get("failed_index") == 0, out
    assert {str(f): _tree(f) for f in (outside, taken)} == before


def test_a_rollback_that_cannot_move_the_skill_aside_leaves_no_aside_folder(shadowed, monkeypatch):
    """If moving the broken folder aside fails, the rollback reports it, keeps the snapshot, and removes
    the empty aside folder it made for the attempt."""
    import pathlib
    local_root = shadowed["local"].parent
    real_rename = pathlib.Path.rename

    def rename(self, target):
        if pathlib.Path(target).parent.name.startswith(".foo.rollback-"):
            raise PermissionError("simulated: cannot move the skill folder")
        return real_rename(self, target)
    monkeypatch.setattr(pathlib.Path, "rename", rename)
    out = _batch({"action": "write_file", "name": "foo", "file_path": "references/r.md", "file_content": "R.\n"},
                 {"action": "patch", "name": "foo", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False and "ROLLBACK FAILED" in out.get("error", ""), out
    assert not list(local_root.glob(".foo.rollback-*")), "the aside folder was left behind"


def test_a_skill_created_in_the_batch_can_be_changed_through_its_categorized_name(shadowed):
    """`category/bar` finds nothing when the batch starts, so it has no pin of its own; after the
    create of `bar` in `category` it finds the folder this batch just made, which the batch owns."""
    out = _batch({"action": "create", "name": "bar", "category": "category", "content": _md("bar", "Body.")},
                 {"action": "patch", "name": "category/bar", "old_string": "Body.", "new_string": "Changed."})
    assert out.get("success") is True, out
    assert b"Changed." in (shadowed["local"].parent / "category" / "bar" / "SKILL.md").read_bytes()


def test_files_written_through_another_name_are_rolled_back_with_the_skill(shadowed):
    """An adopted empty folder keeps existing after rollback, so the rollback removes the files the
    batch wrote there by folder, whichever name wrote them."""
    folder = shadowed["local"].parent / "category" / "bar"
    folder.mkdir(parents=True)  # an empty leftover folder: create adopts it
    out = _batch({"action": "create", "name": "bar", "category": "category", "content": _md("bar", "Body.")},
                 {"action": "write_file", "name": "category/bar", "file_path": "references/r.md", "file_content": "R.\n"},
                 {"action": "patch", "name": "bar", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False and out.get("failed_index") == 2, out
    assert _tree(folder) in ("MISSING", {}), _tree(folder)


@pytest.mark.parametrize("ops_before", [0, 1], ids=["nothing-applied", "another-skill-changed"])
def test_a_rollback_leaves_a_folder_the_batch_did_not_change_alone(shadowed, monkeypatch, ops_before):
    """The batch snapshots `baz` but never changes it: its only op on `baz` fails without writing.
    Right after the batch's snapshot, before any op runs, another writer adds a file to `baz`. The rollback restores
    only folders the batch changed (measured around each op), so that writer's file survives; `foo`,
    which the batch did change, is restored. (A writer changing `baz` while the batch's own op on it
    runs cannot be told apart from that op: serializing it is #156263's folder lock.)"""
    local_root = shadowed["local"].parent
    (local_root / "baz").mkdir()
    (local_root / "baz" / "SKILL.md").write_text(_md("baz", "Baz."), encoding="utf-8")
    foo = _tree(shadowed["local"])
    import tools.skill_manager_batch as smb
    real = smb._snapshot_skills

    def other_writer_lands_after_the_snapshot(*args, **kwargs):
        result = real(*args, **kwargs)
        (local_root / "baz" / "notes.md").write_text("Another writer's change.\n", encoding="utf-8")
        return result
    monkeypatch.setattr(smb, "_snapshot_skills", other_writer_lands_after_the_snapshot)
    ops = [{"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."}][:ops_before]
    out = _batch(*ops, {"action": "patch", "name": "baz", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False and out.get("failed_index") == ops_before, out
    notes = local_root / "baz" / "notes.md"
    assert notes.exists(), "the rollback undid another writer's change to a folder the batch never changed"
    assert notes.read_bytes() == b"Another writer's change.\n"
    assert _tree(shadowed["local"]) == foo


def test_a_create_whose_name_first_found_another_skill_is_rolled_back_where_it_was_made(shadowed):
    """`a` declares the display name `bar`, so `bar` finds folder `a` when the batch starts. Renaming
    `a` frees the name; `create bar` then makes a new folder `bar`, which the batch owns, so a patch
    on `bar` acts there. A later failure removes `bar` and restores `a`: a create is undone in the
    folder it made, whatever the name found at the start."""
    local_root = shadowed["local"].parent
    (local_root / "a").mkdir()
    (local_root / "a" / "SKILL.md").write_text(_md("bar", "Body a."), encoding="utf-8")
    a = _tree(local_root / "a")
    out = _batch({"action": "patch", "name": "a", "old_string": "name: bar", "new_string": "name: renamed"},
                 {"action": "create", "name": "bar", "content": _md("bar", "New.")},
                 {"action": "patch", "name": "bar", "old_string": "New.", "new_string": "Changed."},
                 {"action": "patch", "name": "a", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False and out.get("failed_index") == 3, out
    assert not (local_root / "bar").exists(), "the skill the batch created was left behind"
    assert _tree(local_root / "a") == a


def test_a_batch_op_is_refused_when_its_name_finds_another_folder_by_the_time_it_runs(shadowed, monkeypatch):
    """The batch checks `foo` (the local copy) and binds the op to it. Before the handler runs, another
    writer (through a display name, so a different lock) removes that copy's SKILL.md, and `foo` now
    finds the external skill. The handler refuses instead of following the name; nothing is written
    there, and the batch recorded nothing it would have to undo."""
    external = _tree(shadowed["external"])
    import tools.skill_manager_tool as smt
    real = smt._skill_manage_from
    calls = []

    def other_writer_removes_the_local_copy(payload, **extra):
        if not calls:
            (shadowed["local"] / "SKILL.md").unlink()
        calls.append(payload.get("action"))
        return real(payload, **extra)
    monkeypatch.setattr(smt, "_skill_manage_from", other_writer_removes_the_local_copy)
    out = _batch({"action": "write_file", "name": "foo", "file_path": "references/new.md", "file_content": "Batch.\n"},
                 {"action": "patch", "name": "foo", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False and out.get("failed_index") == 0, out
    assert "this batch checked" in out.get("error", ""), out
    assert _tree(shadowed["external"]) == external


def test_the_binding_covers_only_the_ops_own_name(tmp_path):
    """A hook writing another skill from inside a batch op is not bound to the op's folder; the op's own
    name is bound to the folder checked, and a name checked as finding nothing must still find nothing."""
    from tools import skill_manager_guards as g
    assert hasattr(g, "_batch_binding_error") and hasattr(g, "_batch_bound_folder"), "no per-op folder binding"
    checked, other = (tmp_path / "a").resolve(), tmp_path / "b"
    token = g._batch_bound_folder.set(("a", checked))
    try:
        assert g._batch_binding_error("a", checked) is None
        assert g._batch_binding_error("a", other) is not None
        assert g._batch_binding_error("hook-target", other) is None
    finally:
        g._batch_bound_folder.reset(token)
    token = g._batch_bound_folder.set(("a", g._BOUND_NOTHING))
    try:
        assert g._batch_binding_error("a", checked) is not None
    finally:
        g._batch_bound_folder.reset(token)


def test_a_failed_op_that_changed_bytes_is_rolled_back(shadowed, monkeypatch):
    """A write the security scan refuses puts the old text back decoded (a UTF-8 BOM and CRLF line
    endings are lost). The op failed, but it changed the folder, so the rollback restores it byte for
    byte. Whether a failed op changed its folder is measured, not inferred from its result."""
    original = b"\xef\xbb\xbfOriginal line.\r\nSecond line.\r\n"
    (shadowed["local"] / "references").mkdir()
    (shadowed["local"] / "references" / "data.md").write_bytes(original)
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(smt, "_security_scan_skill", lambda *a, **k: "Security scan blocked this skill (test).")
    out = _batch({"action": "write_file", "name": "foo", "file_path": "references/data.md", "file_content": "New.\n"})
    assert out.get("success") is False and out.get("failed_index") == 0, out
    assert (shadowed["local"] / "references" / "data.md").read_bytes() == original


def test_a_create_is_refused_when_its_folder_moved_after_the_batch_planned_it(shadowed, monkeypatch):
    """The batch plans `bar` at one folder; by the time the create runs, folder resolution gives another
    (a skills.create_dir change, say). The create is bound to the planned folder and refuses, so no
    skill appears in a folder the rollback would not know about."""
    import tools.skill_manager_tool as smt
    real = smt._resolve_skill_dir
    planned, moved = shadowed["local"].parent / "bar", shadowed["tmp"] / "elsewhere" / "bar"
    calls = []

    def drifting(name, category=None):
        calls.append(name)
        return real(name, category) if len(calls) == 1 else moved
    monkeypatch.setattr(smt, "_resolve_skill_dir", drifting)
    out = _batch({"action": "create", "name": "bar", "content": _md("bar", "New.")},
                 {"action": "patch", "name": "foo", "old_string": "no such text", "new_string": "x"})
    assert out.get("success") is False and out.get("failed_index") == 0, out
    assert not moved.exists() and not planned.exists()


@pytest.fixture
def baz(shadowed):
    """A second local skill `baz` the batch snapshots and whose op fails without writing."""
    folder = shadowed["local"].parent / "baz"
    folder.mkdir()
    (folder / "SKILL.md").write_text(_md("baz", "Baz."), encoding="utf-8")
    return folder


FAILS_ON_BAZ = {"action": "patch", "name": "baz", "old_string": "no such text", "new_string": "x"}


@pytest.mark.parametrize("hook_does", ["patch", "delete", "nested-batch"])
def test_a_lifecycle_hooks_write_to_a_snapshotted_folder_is_rolled_back(shadowed, baz, monkeypatch, hook_does):
    """foo's patch succeeds and its lifecycle hook (here the usage bump) changes `baz` through its own
    skill_manage call; then the batch's own op on `baz` fails without writing. The hook's write is part
    of the batch's execution, recorded where the bytes changed, so the rollback restores `baz` too."""
    before = _tree(baz)
    import tools.skill_usage as usage
    from tools.skill_manager_tool import skill_manage
    real, fired = usage.bump_patch, []

    def hook(name, *args, **kwargs):
        if name == "foo" and not fired:
            fired.append(name)
            if hook_does == "patch":
                skill_manage(action="patch", name="baz", old_string="Baz.", new_string="Hooked.")
            elif hook_does == "delete":
                skill_manage(action="delete", name="baz")
            else:
                skill_manage(action="", name="", operations=[
                    {"action": "write_file", "name": "baz", "file_path": "references/h.md", "file_content": "Hooked.\n"}])
            assert _tree(baz) != before, "the hook did not change baz"
        return real(name, *args, **kwargs)
    monkeypatch.setattr(usage, "bump_patch", hook)
    out = _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."}, FAILS_ON_BAZ)
    assert fired and out.get("success") is False and out.get("failed_index") == 1, out
    assert "rolled back" in out["error"], out
    assert _tree(baz) == before, "the hook's write to a snapshotted folder survived the rollback"
    assert _tree(shadowed["local"])["SKILL.md"] == _md("foo", "Local.").encode()


def _snapshot_dirs(tmp):
    return sorted(p.name for p in tmp.iterdir() if p.name.startswith("skill_batch_"))


def test_an_op_that_raises_after_writing_is_rolled_back(shadowed, baz, monkeypatch, tmp_path):
    """The second op writes `baz` and then its handler raises. The error propagates as before, but the
    batch first undoes what it wrote (foo's patch and the partial write to baz)."""
    before_baz, scratch = _tree(baz), tmp_path / "scratch"
    scratch.mkdir()
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    import tools.skill_manager_tool as smt
    real, calls = smt._security_scan_skill, []

    def scan(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("scanner crashed")
        return real(*args, **kwargs)
    monkeypatch.setattr(smt, "_security_scan_skill", scan)
    with pytest.raises(RuntimeError, match="scanner crashed"):
        _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."},
               {"action": "write_file", "name": "baz", "file_path": "references/new.md", "file_content": "Partial.\n"})
    assert _tree(baz) == before_baz
    assert _tree(shadowed["local"])["SKILL.md"] == _md("foo", "Local.").encode()
    assert _snapshot_dirs(scratch) == [], "a successful rollback leaves no snapshot behind"


def test_an_interrupted_batch_keeps_its_snapshots(shadowed, baz, monkeypatch, tmp_path):
    """An interrupt (KeyboardInterrupt) mid-op leaves recovery uncertain: the snapshots stay on disk."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    import tools.skill_manager_tool as smt
    real, calls = smt._security_scan_skill, []

    def scan(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)
    monkeypatch.setattr(smt, "_security_scan_skill", scan)
    with pytest.raises(KeyboardInterrupt):
        _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."},
               {"action": "write_file", "name": "baz", "file_path": "references/new.md", "file_content": "Partial.\n"})
    assert len(_snapshot_dirs(scratch)) == 1


@pytest.mark.parametrize("folder_was", ["new", "adopted-empty"])
@pytest.mark.parametrize("raises", ["after-writing", "before-writing"])
def test_a_create_that_raises_is_rolled_back(shadowed, monkeypatch, tmp_path, folder_was, raises):
    """The second op creates `bar` and raises partway: after SKILL.md is written (the scanner crashes) or
    before (the write itself fails). The create recorded the folder when it made or adopted it, so the
    rollback removes it (an adopted dir only loses what the batch wrote) and restores `foo`; the error propagates."""
    bar, scratch = shadowed["local"].parent / "bar", tmp_path / "scratch"
    scratch.mkdir()
    if folder_was == "adopted-empty":
        bar.mkdir()
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    import tools.skill_manager_tool as smt
    if raises == "after-writing":
        real_scan = smt._security_scan_skill

        def scan(skill_dir, *args, **kwargs):
            if skill_dir.name == "bar":
                assert (bar / "SKILL.md").exists()
                raise RuntimeError("scanner crashed")
            return real_scan(skill_dir, *args, **kwargs)
        monkeypatch.setattr(smt, "_security_scan_skill", scan)
    else:
        real_write = smt.atomic_write_text

        def write(target, *args, **kwargs):
            if target.parent.name == "bar":
                raise OSError(28, "No space left on device")
            return real_write(target, *args, **kwargs)
        monkeypatch.setattr(smt, "atomic_write_text", write)
    with pytest.raises((RuntimeError, OSError)):
        _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."},
               {"action": "create", "name": "bar", "content": _md("bar", "New.")})
    if folder_was == "new":
        assert not bar.exists(), "the raising create's folder survived the rollback"
    else:  # emptied of what the batch wrote, then rmdir'd if empty (anything foreign would stay)
        assert not bar.exists() or not any(bar.iterdir()), "the raising create's SKILL.md survived the rollback"
    assert _tree(shadowed["local"])["SKILL.md"] == _md("foo", "Local.").encode()
    assert _snapshot_dirs(scratch) == []


@pytest.mark.parametrize("op_does", ["fail", "raise"])
def test_an_interrupt_during_rollback_keeps_the_snapshots(shadowed, baz, monkeypatch, tmp_path, op_does):
    """The second op fails (or raises), and the rollback that follows is interrupted. Recovery is
    unfinished, so the snapshots stay on disk: they may hold the only copy of the original bytes."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    import tools.skill_manager_batch as smb
    import tools.skill_manager_tool as smt

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(smb, "_restore_snapshot", interrupted)
    second = FAILS_ON_BAZ
    if op_does == "raise":
        real, calls = smt._security_scan_skill, []

        def scan(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("scanner crashed")
            return real(*args, **kwargs)
        monkeypatch.setattr(smt, "_security_scan_skill", scan)
        second = {"action": "write_file", "name": "baz", "file_path": "references/new.md", "file_content": "Partial.\n"}
    with pytest.raises(KeyboardInterrupt):
        _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."}, second)
    assert len(_snapshot_dirs(scratch)) == 1, "an interrupted rollback deleted the snapshots"


@pytest.mark.parametrize("hook_uses", ["single-op", "nested-batch"])
def test_a_skill_a_lifecycle_hook_creates_during_the_batch_is_rolled_back(shadowed, baz, monkeypatch, hook_uses):
    """foo's patch succeeds and its lifecycle hook creates `qux` (directly, or through a batch of its own);
    the batch then fails. `qux` was made by the batch's execution, recorded when its folder was made,
    so the rollback removes it."""
    qux = shadowed["local"].parent / "qux"
    import tools.skill_usage as usage
    from tools.skill_manager_tool import skill_manage
    real, fired = usage.bump_patch, []

    def hook(name, *args, **kwargs):
        if name == "foo" and not fired:
            fired.append(name)
            op = {"action": "create", "name": "qux", "content": _md("qux", "Hooked.")}
            if hook_uses == "single-op":
                skill_manage(**op)
            else:
                skill_manage(action="", name="", operations=[op])
            assert (qux / "SKILL.md").exists(), "the hook did not create qux"
        return real(name, *args, **kwargs)
    monkeypatch.setattr(usage, "bump_patch", hook)
    out = _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."}, FAILS_ON_BAZ)
    assert fired and out.get("success") is False and out.get("failed_index") == 1, out
    assert not qux.exists(), "a skill the batch's hook created survived the rollback"


def test_a_rollback_that_fails_keeps_the_snapshots(shadowed, baz, monkeypatch, tmp_path):
    """The second op fails and restoring `foo` fails too (a locked file, a full disk). The result says
    so, and the snapshots stay on disk for recovery by hand."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    import tools.skill_manager_batch as smb

    def restore_fails(*args, **kwargs):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(smb, "_restore_snapshot", restore_fails)
    out = _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."}, FAILS_ON_BAZ)
    assert out.get("success") is False and "ROLLBACK FAILED" in out["error"], out
    assert len(_snapshot_dirs(scratch)) == 1, "a failed rollback deleted the snapshots"


def test_a_refused_adopted_create_leaves_a_file_it_did_not_write(shadowed, monkeypatch):
    """The create adopts an empty leftover `bar`; before it writes, another writer saves bar/SKILL.md, and
    the create is refused (here the scan baseline). The batch never wrote that file, so the rollback
    leaves it."""
    bar = shadowed["local"].parent / "bar"
    bar.mkdir()
    import tools.skill_manager_tool as smt

    real_baseline = smt._scan_baseline

    def baseline_fails(skill_dir, *args, **kwargs):
        if skill_dir.name != "bar":
            return real_baseline(skill_dir, *args, **kwargs)
        (skill_dir / "SKILL.md").write_text("Another writer's skill.\n", encoding="utf-8")
        return None, None, "scan baseline unavailable (test)"
    monkeypatch.setattr(smt, "_scan_baseline", baseline_fails)
    out = _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."},
                 {"action": "create", "name": "bar", "content": _md("bar", "New.")})
    assert out.get("success") is False and out.get("failed_index") == 1, out
    assert (bar / "SKILL.md").read_text(encoding="utf-8") == "Another writer's skill.\n"
    assert _tree(shadowed["local"])["SKILL.md"] == _md("foo", "Local.").encode()


def test_a_rollback_that_cannot_remove_a_written_file_keeps_the_snapshots(shadowed, monkeypatch, tmp_path):
    """An adopted create writes SKILL.md and raises; the rollback cannot unlink that file (permissions).
    The rollback did not finish, so the snapshots stay on disk."""
    bar, scratch = shadowed["local"].parent / "bar", tmp_path / "scratch"
    bar.mkdir()
    scratch.mkdir()
    import pathlib
    import tempfile
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    real_scan, real_unlink = smt._security_scan_skill, pathlib.Path.unlink

    def scan(skill_dir, *args, **kwargs):
        if skill_dir.name == "bar":
            raise RuntimeError("scanner crashed")
        return real_scan(skill_dir, *args, **kwargs)

    def unlink(self, *args, **kwargs):
        if self.parent.name == "bar" and self.name == "SKILL.md":
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)
    monkeypatch.setattr(smt, "_security_scan_skill", scan)
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    with pytest.raises(RuntimeError, match="scanner crashed"):
        _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."},
               {"action": "create", "name": "bar", "content": _md("bar", "New.")})
    assert len(_snapshot_dirs(scratch)) == 1, "an unfinished rollback deleted the snapshots"


@pytest.mark.parametrize("write_fails", ["before-replacing", "after-replacing"])
def test_a_write_that_fails_comes_back_as_it_was(shadowed, monkeypatch, write_fails):
    """The batch's write to foo fails, before the file is replaced or after (restoring owner or mode),
    having made references/ for it. The attempt is recorded when it starts, so the rollback restores
    foo exactly as it was: no references/ folder, no new file. (Whether another writer's change to foo
    at that moment survives is the concurrency #156263 serializes; the batch cannot tell the two apart.)"""
    before = _tree(shadowed["local"])
    import tools.skill_manager_tool as smt
    real_write = smt.atomic_write_text

    def write(target, *args, **kwargs):
        if target.name == "new.md":
            if write_fails == "after-replacing":
                real_write(target, *args, **kwargs)
            raise PermissionError(13, "Permission denied")
        return real_write(target, *args, **kwargs)
    monkeypatch.setattr(smt, "atomic_write_text", write)
    with pytest.raises(PermissionError):
        _batch({"action": "write_file", "name": "foo", "file_path": "references/new.md", "file_content": "New.\n"})
    assert _tree(shadowed["local"]) == before
    assert not (shadowed["local"] / "references").exists()


def test_a_rollback_removes_an_aside_folder_with_read_only_folders_inside(shadowed, baz, tmp_path):
    """foo holds a read-only folder. The batch patches foo and then fails; the rollback moves the changed
    foo aside, restores the snapshot and must still remove the aside copy, read-only folder and all."""
    locked = shadowed["local"] / "references" / "locked"
    locked.mkdir(parents=True)
    (locked / "data.md").write_text("Data.\n", encoding="utf-8")
    locked.chmod(0o555)
    try:
        before = _tree(shadowed["local"])
        out = _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."}, FAILS_ON_BAZ)
        assert out.get("success") is False and "all touched skills rolled back" in out["error"], out
        assert _tree(shadowed["local"]) == before
        leftovers = [p.name for p in shadowed["local"].parent.iterdir() if ".rollback-" in p.name]
        assert leftovers == [], f"the rollback left {leftovers} behind"
    finally:
        for p in shadowed["local"].parent.rglob("locked"):
            p.chmod(0o755)


def test_a_rollback_that_cannot_remove_its_aside_folder_says_so_and_keeps_the_snapshots(shadowed, baz, monkeypatch, tmp_path):
    """The skill is restored, but its aside copy cannot be deleted. The batch reports that rather than a
    clean rollback, and keeps its snapshots."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    import shutil
    import tempfile
    import tools.skill_manager_batch as smb
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    real_rmtree = shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if ".rollback-" in str(path) and not kwargs.get("ignore_errors"):
            raise PermissionError(13, "Permission denied")
        return real_rmtree(path, *args, **kwargs)
    monkeypatch.setattr(smb.shutil, "rmtree", rmtree)
    out = _batch({"action": "patch", "name": "foo", "old_string": "Local.", "new_string": "Changed."}, FAILS_ON_BAZ)
    assert out.get("success") is False and "could not be removed" in out["error"], out
    assert _tree(shadowed["local"])["SKILL.md"] == _md("foo", "Local.").encode()
    assert len(_snapshot_dirs(scratch)) == 1

