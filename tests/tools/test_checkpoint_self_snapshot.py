"""A checkpoint must not capture the checkpoint store that lives inside its workdir.

A profile-home checkpoint (``~/.hermes/profiles/<name>``) contains the active checkpoint base,
so an unfiltered ``git add -A`` snapshots the store's own object database into the very tree
the store is keeping.  The exclusion is derived from the live ``_resolve_checkpoint_base()``
path on every take, so an existing store that already carries its own store stops carrying it
on the next take, and an unrelated user directory that merely happens to be named
``checkpoints`` is still snapshotted.
"""

from __future__ import annotations

import tools.checkpoint_manager as checkpoints


def _tree_paths(project) -> set[str]:
    """Relative paths in the project's newest checkpoint commit."""
    store = checkpoints._store_path()
    ref = checkpoints._ref_name(checkpoints._project_hash(str(project)))
    ok, out, err = checkpoints._run_git(
        ["ls-tree", "-r", "--name-only", ref], store, str(project),
    )
    assert ok, f"ls-tree failed: {err}"
    return set(out.splitlines())


def _manager() -> checkpoints.CheckpointManager:
    return checkpoints.CheckpointManager(
        enabled=True, max_total_size_mb=0, max_file_size_mb=0,
    )


def test_profile_home_checkpoint_omits_its_own_store(tmp_path, monkeypatch):
    """The store sits at ``<workdir>/checkpoints/store``; the snapshot must contain neither."""
    project = tmp_path / "profile"
    project.mkdir()
    (project / "config.yaml").write_text("model: test\n", encoding="utf-8")
    base = project / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)

    assert _manager().ensure_checkpoint(str(project), "profile home")

    paths = _tree_paths(project)
    assert "config.yaml" in paths
    assert not [path for path in paths if path.startswith("checkpoints/")], paths
    # The store really was created inside the workdir, so the assertion above
    # is about exclusion rather than about a store that never existed.
    assert (base / "store" / "HEAD").exists()


def test_exclusion_follows_the_resolved_base_not_the_directory_name(tmp_path, monkeypatch):
    """A base nested below some other directory name is still excluded by its derived path."""
    project = tmp_path / "profile"
    (project / "state").mkdir(parents=True)
    (project / "state" / "notes.txt").write_text("keep me\n", encoding="utf-8")
    base = project / "state" / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)

    assert _manager().ensure_checkpoint(str(project), "nested base")

    paths = _tree_paths(project)
    assert "state/notes.txt" in paths
    assert not [path for path in paths if path.startswith("state/checkpoints/")], paths


def test_unrelated_checkpoints_directory_is_still_snapshotted(tmp_path, monkeypatch):
    """Only the active base is excluded - a user project's own ``checkpoints/`` is data."""
    project = tmp_path / "project"
    (project / "checkpoints").mkdir(parents=True)
    (project / "checkpoints" / "keep.txt").write_text("user data\n", encoding="utf-8")
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", tmp_path / "hermes-home" / "checkpoints")

    assert _manager().ensure_checkpoint(str(project), "user checkpoints dir")

    assert "checkpoints/keep.txt" in _tree_paths(project)


def test_checkpoint_of_the_base_itself_is_refused(tmp_path, monkeypatch):
    """A workdir that is the base holds only the store; there is no tree worth committing."""
    project = tmp_path / "profile"
    project.mkdir()
    (project / "config.yaml").write_text("model: test\n", encoding="utf-8")
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", project)

    assert not _manager().ensure_checkpoint(str(project), "base itself")
    assert not (project / "store").exists()


def test_existing_store_stops_carrying_its_own_store_on_the_next_take(tmp_path, monkeypatch):
    """A store that already holds a recursive snapshot drops those paths on the next take."""
    project = tmp_path / "profile"
    project.mkdir()
    (project / "config.yaml").write_text("model: test\n", encoding="utf-8")
    base = project / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    manager = _manager()

    # Reproduce the pre-fix snapshot: stage with no exclusion, as the old code did.
    real_exclusion = checkpoints._checkpoint_base_rel_path
    monkeypatch.setattr(checkpoints, "_checkpoint_base_rel_path", lambda _workdir: None)
    assert manager.ensure_checkpoint(str(project), "legacy recursive snapshot")
    assert [path for path in _tree_paths(project) if path.startswith("checkpoints/")]

    # The fix runs on an existing store: the seeded index still holds the store
    # paths, and they must be dropped before the new tree is written.
    monkeypatch.setattr(checkpoints, "_checkpoint_base_rel_path", real_exclusion)
    manager.new_turn()
    assert manager.ensure_checkpoint(str(project), "first clean snapshot")

    paths = _tree_paths(project)
    assert "config.yaml" in paths
    assert not [path for path in paths if path.startswith("checkpoints/")], paths

    # Converged: nothing changed on disk, so no further snapshot is taken.
    manager.new_turn()
    assert not manager.ensure_checkpoint(str(project), "nothing changed")


def test_diff_omits_store_paths_when_base_is_inside_workdir(tmp_path, monkeypatch):
    """``_diff()`` must not report store paths even when comparing against an old checkpoint."""
    project = tmp_path / "profile"
    project.mkdir()
    (project / "config.yaml").write_text("model: test\n", encoding="utf-8")
    base = project / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    manager = _manager()

    # Take a checkpoint with the old unfiltered staging (simulating a pre-fix store).
    real_exclusion = checkpoints._checkpoint_base_rel_path
    monkeypatch.setattr(checkpoints, "_checkpoint_base_rel_path", lambda _workdir: None)
    assert manager.ensure_checkpoint(str(project), "legacy recursive snapshot")
    assert [p for p in _tree_paths(project) if p.startswith("checkpoints/")]

    # Now switch to the fixed staging and take another checkpoint.
    monkeypatch.setattr(checkpoints, "_checkpoint_base_rel_path", real_exclusion)
    manager.new_turn()
    (project / "config.yaml").write_text("model: test-v2\n", encoding="utf-8")
    assert manager.ensure_checkpoint(str(project), "first clean snapshot")

    # Diff against the old checkpoint must not mention any store paths.
    checkpoints_list = manager.list_checkpoints(str(project))
    assert len(checkpoints_list) >= 2
    old_hash = checkpoints_list[-1]["hash"]
    result = manager.diff(str(project), old_hash)
    assert result["success"], result

    # Parse the diff into file sections and verify no path matches base_rel or is under it.
    import re
    diff_text = result.get("diff", "")
    stat_text = result.get("stat", "")

    # Check diff output: every "diff --git a/<path> b/<path>" header must not reference store paths.
    for match in re.finditer(r"^diff --git a/(\S+) b/(\S+)", diff_text, re.MULTILINE):
        a_path, b_path = match.group(1), match.group(2)
        assert not a_path.startswith("checkpoints/"), f"Store path in diff header: {a_path}"
        assert not b_path.startswith("checkpoints/"), f"Store path in diff header: {b_path}"

    # Check stat output: count the "|" rows and compare to the declared file total.
    stat_rows = [line for line in stat_text.splitlines() if "|" in line and not line.strip().startswith("---")]
    # The summary line is "N file(s) changed, ..."
    summary_match = re.search(r"(\d+) files? changed", stat_text)
    if summary_match:
        declared_count = int(summary_match.group(1))
        assert len(stat_rows) == declared_count, f"Stat row count {len(stat_rows)} != declared {declared_count}"

    # Verify no stat row mentions a store path.
    for row in stat_rows:
        path_part = row.split("|")[0].strip()
        assert not path_part.startswith("checkpoints/"), f"Store path in stat row: {path_part}"


def test_diff_preserves_user_content_quoting_store_path(tmp_path, monkeypatch):
    """A user file whose content mentions the store path must keep that line in the diff."""
    project = tmp_path / "profile"
    project.mkdir()
    (project / "config.yaml").write_text("model: test\n", encoding="utf-8")
    # Create a user file that quotes the store path in its content.
    (project / "notes.md").write_text("the store lives at checkpoints/store and is big\n", encoding="utf-8")
    base = project / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    manager = _manager()

    # Take a checkpoint with the old unfiltered staging (simulating a pre-fix store).
    real_exclusion = checkpoints._checkpoint_base_rel_path
    monkeypatch.setattr(checkpoints, "_checkpoint_base_rel_path", lambda _workdir: None)
    assert manager.ensure_checkpoint(str(project), "legacy recursive snapshot")

    # Now switch to the fixed staging and modify the user file.
    monkeypatch.setattr(checkpoints, "_checkpoint_base_rel_path", real_exclusion)
    manager.new_turn()
    (project / "notes.md").write_text("the store lives at checkpoints/store and is big\nnow with a second line\n", encoding="utf-8")
    assert manager.ensure_checkpoint(str(project), "first clean snapshot")

    # Diff against the old checkpoint must preserve the user content line.
    checkpoints_list = manager.list_checkpoints(str(project))
    assert len(checkpoints_list) >= 2
    old_hash = checkpoints_list[-1]["hash"]
    result = manager.diff(str(project), old_hash)
    assert result["success"], result

    diff_text = result.get("diff", "")
    # The user file's content line must be preserved in the diff body.
    assert "the store lives at checkpoints/store and is big" in diff_text, \
        f"User content line was removed from diff:\n{diff_text}"
    # The hunk header must be intact (not malformed by line removal).
    assert "@@" in diff_text, f"Diff has no hunk headers:\n{diff_text}"


def test_safe_restore_plan_omits_store_paths_when_base_is_inside_workdir(tmp_path, monkeypatch):
    """``_safe_restore_plan()`` must not list store paths in restore/skipped."""
    project = tmp_path / "profile"
    project.mkdir()
    (project / "config.yaml").write_text("model: test\n", encoding="utf-8")
    base = project / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    manager = _manager()

    # Take a checkpoint with the old unfiltered staging.
    real_exclusion = checkpoints._checkpoint_base_rel_path
    monkeypatch.setattr(checkpoints, "_checkpoint_base_rel_path", lambda _workdir: None)
    assert manager.ensure_checkpoint(str(project), "legacy recursive snapshot")
    assert [p for p in _tree_paths(project) if p.startswith("checkpoints/")]

    # Switch to fixed staging and take another checkpoint.
    monkeypatch.setattr(checkpoints, "_checkpoint_base_rel_path", real_exclusion)
    manager.new_turn()
    (project / "config.yaml").write_text("model: test-v2\n", encoding="utf-8")
    assert manager.ensure_checkpoint(str(project), "first clean snapshot")

    # Record an agent write so the ledger is non-empty.
    store = checkpoints._store_path()
    abs_config = str(project / "config.yaml")
    checkpoints._load_ledger(store, manager._ledger_key(str(project)))  # ensure ledger exists
    ledger = checkpoints._load_ledger(store, manager._ledger_key(str(project)))
    ledger[abs_config] = {"sha256": checkpoints._hash_file(project / "config.yaml") or ""}
    checkpoints._save_ledger(store, manager._ledger_key(str(project)), ledger)

    # Safe restore plan against the old checkpoint must not mention store paths.
    checkpoints_list = manager.list_checkpoints(str(project))
    old_hash = checkpoints_list[-1]["hash"]
    plan = manager._safe_restore_plan(str(project), old_hash)
    assert plan["success"], plan
    for path in plan.get("restore", []) + plan.get("skipped", []):
        assert not path.startswith("checkpoints/"), f"Store path in plan: {path}"
