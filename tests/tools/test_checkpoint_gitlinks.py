"""Real-Git rollback contracts for checkpoints containing uncaptured gitlinks."""
import os
import shutil
import subprocess

import pytest

from tools import checkpoint_manager as cm


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    work = tmp_path / "workspace"
    work.mkdir()
    nested = work / "tool"
    nested.mkdir()
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_NOSYSTEM": "1"}
    for args in (["init", "-q"], ["config", "user.name", "Test"],
                 ["config", "user.email", "test@example.invalid"]):
        subprocess.run(["git", *args], cwd=nested, env=env, check=True, capture_output=True)
    (nested / "main.py").write_text("committed\n")
    subprocess.run(["git", "add", "."], cwd=nested, env=env, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=nested, env=env, check=True, capture_output=True)
    (nested / "main.py").write_text("user work not captured\n")
    (nested / "untracked.txt").write_text("not captured either\n")
    (work / "notes.txt").write_text("before\n")
    mgr = cm.CheckpointManager(enabled=True)
    assert mgr.ensure_checkpoint(str(work), "baseline")
    commit = mgr.list_checkpoints(str(work))[0]["hash"]
    ok, tree, _ = cm._run_git(["ls-tree", "-r", "-z", commit], cm._store_path(), str(work))
    assert ok and "160000 commit " in tree
    (work / "notes.txt").write_text("after\n")
    (nested / "main.py").write_text("agent overwrite\n")
    return mgr, work, commit


def durable_state(work):
    """Include refs, indexes, ledger and working files, not transient lock files."""
    roots = [work, cm._store_path()]
    return {(str(root), str(path.relative_to(root))): path.read_bytes() if path.is_file() else None
            for root in roots for path in root.rglob("*")}


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("spec", [None, ".", "tool", "tool/", "tool/main.py", "*",
                                   "t*", ":(glob)*", ":(top,glob)**", ":(icase)TOOL"])
def test_refusal_is_nonmutating(checkpoint, safe, spec):
    mgr, work, commit = checkpoint
    before = durable_state(work)
    result = mgr.restore(str(work), commit, file_path=spec, safe=safe)
    assert result["success"] is False
    assert result["nested_repositories"] == ["tool"]
    assert "not captured" in result["error"]
    assert durable_state(work) == before


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("spec", ["notes.txt", "*.txt", ":(glob)*.txt", ":(exclude)tool",
                                   ":(literal)notes.txt"])
def test_unrelated_selection_still_restores(checkpoint, safe, spec):
    mgr, work, commit = checkpoint
    result = mgr.restore(str(work), commit, file_path=spec, safe=safe)
    assert result["success"] is True, result
    assert (work / "notes.txt").read_text() == "before\n"
    assert (work / "tool" / "main.py").read_text() == "agent overwrite\n"


@pytest.mark.parametrize("name", ["space name.txt", "unicode-雪.txt",
                                  pytest.param("tab\tline\n.txt", marks=pytest.mark.skipif(os.name == "nt", reason="POSIX control characters")),
                                  pytest.param(os.fsdecode(b"bad-\xff.txt"),
                                               marks=pytest.mark.skipif(os.name != "posix", reason="POSIX filename bytes"))])
def test_unusual_filename_does_not_break_tree_inspection(checkpoint, name):
    mgr, work, _ = checkpoint
    target = work / name
    target.write_text("original\n")
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(work), "unusual filename")
    commit = mgr.list_checkpoints(str(work))[0]["hash"]
    target.write_text("changed\n")
    result = mgr.restore(str(work), commit, file_path=name)
    assert result["success"] is True, result
    assert target.read_text() == "original\n"


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("spec", [None, "*", "tool/main.py"])
def test_deleted_nested_repository_is_not_reported_restored(checkpoint, safe, spec):
    mgr, work, commit = checkpoint
    shutil.rmtree(work / "tool")
    before = durable_state(work)
    result = mgr.restore(str(work), commit, file_path=spec, safe=safe)
    assert result["success"] is False
    assert result["nested_repositories"] == ["tool"]
    assert not (work / "tool").exists()
    assert durable_state(work) == before


@pytest.mark.parametrize("name", ["nested space", "nested-雪", "vendor/tool",
                                  pytest.param("nested\tline\n", marks=pytest.mark.skipif(os.name == "nt", reason="POSIX control characters")),
                                  pytest.param(os.fsdecode(b"nested-\xff"),
                                               marks=pytest.mark.skipif(os.name != "posix", reason="POSIX filename bytes"))])
def test_gitlink_name_roundtrips_in_refusal(checkpoint, name):
    mgr, work, _ = checkpoint
    (work / name).parent.mkdir(parents=True, exist_ok=True)
    (work / "tool").rename(work / name)
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(work), "renamed nested repository")
    commit = mgr.list_checkpoints(str(work))[0]["hash"]
    before = durable_state(work)
    result = mgr.restore(str(work), commit, file_path="*")
    assert result["success"] is False
    assert result["nested_repositories"] == [name]
    assert durable_state(work) == before


def test_checkpoint_listing_discloses_uncaptured_nested_repository(checkpoint, tmp_path):
    mgr, work, _ = checkpoint
    entry = mgr.list_checkpoints(str(work))[0]
    assert entry["reason"] == "baseline [nested git repos not captured: tool]"
    assert "[nested git repos not captured: tool]" in cm.format_checkpoint_list([entry], str(work))

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.txt").write_text("a\n")
    assert mgr.ensure_checkpoint(str(plain), "no nested repo")
    assert mgr.list_checkpoints(str(plain))[0]["reason"] == "no nested repo"


def test_checkpoint_disclosure_caps_listed_repositories(checkpoint):
    assert cm._uncaptured_note(["a", "b", "c", "d", "e"]) == " [nested git repos not captured: a, b, c (+2)]"


@pytest.fixture
def deep_checkpoint(checkpoint):
    mgr, work, _ = checkpoint
    (work / "packages").mkdir()
    (work / "tool").rename(work / "packages" / "nested")
    (work / "notes.txt").write_text("before\n")
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(work), "deeper nested repository")
    commit = mgr.list_checkpoints(str(work))[0]["hash"]
    (work / "notes.txt").write_text("after\n")
    return mgr, work, commit


@pytest.mark.parametrize("spec", [
    "packages", "packages/", "packages/nested", "packages/nested/", "packages/nested/main.py",
    "packages/./nested", "notes/../packages/nested",
    pytest.param("packages\\nested", marks=pytest.mark.skipif(os.name != "nt", reason="Windows separator")),
    pytest.param("packages\\nested\\main.py", marks=pytest.mark.skipif(os.name != "nt", reason="Windows separator")),
])
def test_deeper_gitlink_scope_refuses_in_any_spelling(deep_checkpoint, spec):
    mgr, work, commit = deep_checkpoint
    before = durable_state(work)
    result = mgr.restore(str(work), commit, file_path=spec)
    assert result["success"] is False
    assert result["nested_repositories"] == ["packages/nested"]
    assert durable_state(work) == before


@pytest.mark.parametrize("spec", ["notes.txt", "packages/../notes.txt"])
def test_deeper_gitlink_leaves_unrelated_recovery_available(deep_checkpoint, spec):
    mgr, work, commit = deep_checkpoint
    result = mgr.restore(str(work), commit, file_path=spec)
    assert result["success"] is True, result
    assert (work / "notes.txt").read_text() == "before\n"
    assert (work / "packages" / "nested" / "main.py").read_text() == "agent overwrite\n"


@pytest.mark.parametrize("spec", [":(literal)tool/main.py", ":(glob)tool/*.py", ":(invalid)tool"])
def test_unresolvable_selection_is_nonmutating(checkpoint, spec):
    mgr, work, commit = checkpoint
    before = durable_state(work)
    result = mgr.restore(str(work), commit, file_path=spec)
    assert result["success"] is False
    assert durable_state(work) == before


@pytest.mark.parametrize("command", ["read-tree", "ls-files"])
def test_inspection_failure_leaves_files_and_history_untouched(checkpoint, monkeypatch, command):
    mgr, work, commit = checkpoint
    run_git = cm._run_git

    def fail_inspection(args, *a, **kw):
        if args[0] == command:
            return False, "", "inspection unavailable"
        return run_git(args, *a, **kw)

    monkeypatch.setattr(cm, "_run_git", fail_inspection)
    before = durable_state(work)
    result = mgr.restore(str(work), commit, file_path="*")
    assert result["success"] is False
    assert "inspection unavailable" in result["error"]
    assert durable_state(work) == before
    assert not list(cm._store_path().glob("restore-inspect-*"))


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
@pytest.mark.parametrize("safe", [False, True])
def test_captured_file_replaced_by_symlink_into_nested_repo_restores(checkpoint, safe):
    mgr, work, commit = checkpoint
    notes = work / "notes.txt"
    notes.unlink()
    notes.symlink_to(work / "tool" / "main.py")
    result = mgr.restore(str(work), commit, file_path="notes.txt", safe=safe)
    assert result["success"] is True, result
    assert not notes.is_symlink()
    assert notes.read_text() == "before\n"
    assert (work / "tool" / "main.py").read_text() == "agent overwrite\n"


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
@pytest.mark.parametrize("spec", [None, "tool", "tool/main.py"])
def test_gitlink_replaced_by_unrelated_symlink_still_refuses(checkpoint, tmp_path, spec):
    mgr, work, commit = checkpoint
    (work / "tool").rename(tmp_path / "moved-tool")
    other = work / "other"
    other.mkdir()
    (other / "main.py").write_text("unrelated\n")
    (work / "tool").symlink_to(other, target_is_directory=True)
    before = durable_state(work)
    result = mgr.restore(str(work), commit, file_path=spec)
    assert result["success"] is False
    assert result["nested_repositories"] == ["tool"]
    assert (work / "tool").is_symlink()
    assert durable_state(work) == before
