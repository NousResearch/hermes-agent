"""Review regressions: profile HOME contention and ordered physical batch identity."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from hermes_constants import apply_subprocess_home_env
from tools import skill_manager_tool as smt, write_approval as wa
from tools.registry import registry

BODY = "---\nname: display\ndescription: Review regression.\n---\n\nOriginal body.\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (profile / "config.yaml").write_text("skills:\n  write_approval: true\n  write_approval_mode: create\n")
    directory = profile / "skills" / "category" / "existing"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(BODY)
    return profile


def dispatch(*ops):
    # smt is imported above to register the real tool handler.
    return json.loads(registry.dispatch("skill_manage", {"operations": list(ops)}))


@pytest.mark.parametrize("first", ["absolute", "category", "display"])
@pytest.mark.parametrize("destructive", ["write_file", "remove_file", "patch"])
def test_alias_destructive_overlap_rejected_before_any_write(home, first, destructive):
    target = home / "skills/category/existing/SKILL.md"
    name = {"absolute": str(target.parent), "category": "category/existing", "display": "display"}[first]
    last = {"action": destructive, "name": "existing"}
    if destructive == "patch":
        last["content"] = BODY.replace("Original body.", "Full rewrite.")
    else:
        last["file_path"] = "SKILL.md"
        if destructive == "write_file":
            last["file_content"] = BODY.replace("Original body.", "Full rewrite.")
    result = dispatch({"action": "patch", "name": name, "old_string": "Original body.",
                       "new_string": "Must survive.\nOriginal body."}, last)
    assert not result["success"], result
    assert "silently discard" in result["error"]
    assert target.read_text() == BODY
    assert wa.pending_count(wa.SKILLS) == 0


@pytest.mark.parametrize("destructive", [False, True])
def test_frontmatter_rename_uses_ordered_identity_without_rejecting_patch_chains(home, destructive):
    target = home / "skills/category/existing/SKILL.md"
    first = {"action": "patch", "name": str(target.parent), "old_string": "name: display",
             "new_string": "name: renamed"}
    last = ({"action": "write_file", "name": "renamed", "file_path": "SKILL.md", "file_content": BODY}
            if destructive else {"action": "patch", "name": "renamed", "old_string": "Original body.",
                                 "new_string": "Patch chain survives."})
    result = dispatch(first, last)
    assert result["success"] is not destructive, result
    if destructive:
        assert "silently discard" in result["error"]
        assert target.read_text() == BODY
    else:
        assert "name: renamed" in target.read_text()
        assert "Patch chain survives." in target.read_text()


@pytest.mark.parametrize("stage_route", ["profile", "physical", "replacement-changed"])
def test_invalid_overlap_never_enters_contention_or_replacement_review_queue(home, monkeypatch, stage_route):
    from tools import skill_write_approval as policy
    from tools.skill_resource_fences import acquire_resources
    monkeypatch.setattr(policy, "WRITE_WAIT_SECONDS", 1)
    target = home / "skills/category/existing/SKILL.md"
    ops = [{"action": "patch", "name": str(target.parent), "old_string": "Original body.",
            "new_string": "Must survive.\nOriginal body."},
           {"action": "write_file", "name": "existing", "file_path": "SKILL.md", "file_content": BODY}]
    if stage_route == "replacement-changed":
        monkeypatch.setattr(policy, "replacements_changed", lambda revisions: True)
        result = dispatch(*ops)
    else:
        holder = (policy.creation_transaction() if stage_route == "profile"
                  else acquire_resources({home / "skills"}, time.monotonic() + 10))
        with ThreadPoolExecutor(max_workers=1) as pool:
            with holder:
                result = pool.submit(dispatch, *ops).result(timeout=15)
    assert not result["success"], result
    assert "silently discard" in result["error"]
    assert not result.get("staged") and "pending_id" not in result
    assert target.read_text() == BODY
    assert wa.pending_count(wa.SKILLS) == 0


@pytest.mark.platforms("posix")
def test_support_leaf_aliases_share_destructive_identity(home):
    directory = home / "skills/category/existing"
    support = directory / "references"
    support.mkdir()
    target = support / "target.md"
    target.write_text("Original support.")
    alias = support / "alias.md"
    alias.symlink_to(target)
    result = dispatch({"action": "patch", "name": str(directory), "file_path": "references/alias.md",
                       "old_string": "Original support.", "new_string": "Must survive."},
                      {"action": "write_file", "name": "existing", "file_path": "references/target.md",
                       "file_content": "Clobber."})
    assert not result["success"], result
    assert "silently discard" in result["error"]
    assert target.read_text() == "Original support."
    assert alias.is_symlink()


@pytest.mark.parametrize("stage_route", ["timeout", "replacement-changed"])
def test_staging_rechecks_alias_identity_after_another_writer_changes_it(home, monkeypatch, stage_route):
    from tools import skill_write_approval as policy
    monkeypatch.setattr(policy, "WRITE_WAIT_SECONDS", 5)
    first = home / "skills/category/existing/SKILL.md"
    second = home / "skills/category/second/SKILL.md"
    second.parent.mkdir()
    second.write_text(BODY.replace("name: display", "name: display-second"))
    ops = [{"action": "patch", "name": str(first.parent), "old_string": "Original body.",
            "new_string": "Must survive.\nOriginal body."},
           {"action": "write_file", "name": "display-second", "file_path": "SKILL.md", "file_content": BODY}]
    observed = Event()
    native = policy.replacement_revisions

    def captured_revisions(args):
        revisions = native(args)
        if args["operations"] is not None:
            observed.set()
        return revisions

    monkeypatch.setattr(policy, "replacement_revisions", captured_revisions)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with policy.creation_transaction():
            future = pool.submit(dispatch, *ops)
            assert observed.wait(15), "Initial valid batch was not checked"
            for path, old, new in ((second, "display-second", "display-third"),
                                   (first, "display", "display-second")):
                changed = json.loads(smt.skill_manage(action="patch", name=str(path.parent),
                                                     old_string=f"name: {old}", new_string=f"name: {new}"))
                assert changed["success"] and not changed.get("staged"), changed
            if stage_route == "timeout":
                future.result(timeout=15)
        result = future.result(timeout=15)
    assert not result["success"] and "silently discard" in result["error"], result
    assert not result.get("staged") and "pending_id" not in result
    assert first.read_text() == BODY.replace("name: display", "name: display-second")
    assert second.read_text() == BODY.replace("name: display", "name: display-third")
    assert wa.pending_count(wa.SKILLS) == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_all_and_off_keep_existing_batch_validation(home, enabled):
    (home / "config.yaml").write_text(
        f"skills:\n  write_approval: {str(enabled).lower()}\n  write_approval_mode: all\n")
    target = home / "skills/category/existing/SKILL.md"
    result = dispatch({"action": "patch", "name": "display", "old_string": "Original body.",
                       "new_string": "Legacy patch."},
                      {"action": "write_file", "name": "existing", "file_path": "SKILL.md",
                       "file_content": BODY.replace("Original body.", "Legacy rewrite.")})
    assert result["success"], result
    if enabled:
        assert result["staged"] and target.read_text() == BODY
        assert wa.get_pending(wa.SKILLS, result["pending_id"])
    else:
        assert "Legacy rewrite." in target.read_text()
        assert not result.get("staged")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("setting", ["external_dirs", "create_dir"])
def test_different_profile_home_preserves_both_subprocess_edits(tmp_path, setting):
    shared = tmp_path / "shared"
    target = shared / "existing" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text(BODY)
    envs = []
    for name in ("profile-a", "profile-b"):
        profile = tmp_path / name
        (profile / "home").mkdir(parents=True)
        value = [str(shared)] if setting == "external_dirs" else str(shared)
        (profile / "config.yaml").write_text(
            "skills:\n  write_approval: true\n  write_approval_mode: create\n"
            f"  {setting}: {json.dumps(value)}\n")
        env = dict(os.environ, HERMES_HOME=str(profile), HERMES_REAL_HOME=str(tmp_path),
                   HOME=str(tmp_path), TERMINAL_HOME_MODE="profile")
        apply_subprocess_home_env(env)
        assert env["HOME"] == str(profile / "home")
        envs.append(env)
    ready, release = tmp_path / "ready", tmp_path / "release"
    code_a = '''import sys,time
from pathlib import Path
from tools import skill_manager_tool as smt
native = smt._guarded_write
def hold(*args, **kwargs):
    Path(sys.argv[1]).touch()
    deadline = time.monotonic() + 30
    while not Path(sys.argv[2]).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Test did not release A")
        time.sleep(.02)
    return native(*args, **kwargs)
smt._guarded_write = hold
print(smt.skill_manage(action="patch", name="existing", old_string="Original body.",
                       new_string="A edit.\\nOriginal body."), flush=True)
'''
    code_b = '''from tools import skill_manager_tool as smt, skill_write_approval as policy
policy.WRITE_WAIT_SECONDS = 1
print(smt.skill_manage(action="patch", name="existing", old_string="Original body.",
                       new_string="B edit.\\nOriginal body."), flush=True)
'''
    cwd = str(Path(__file__).resolve().parents[2])
    a = subprocess.Popen([sys.executable, "-c", code_a, str(ready), str(release)], cwd=cwd,
                         env=envs[0], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), "A never reached the read-modify-write barrier"
        b = subprocess.run([sys.executable, "-c", code_b], cwd=cwd, env=envs[1],
                           capture_output=True, text=True, timeout=15)
    finally:
        release.touch()
        out, err = a.communicate(timeout=15)
    assert a.returncode == 0, err
    assert json.loads(out)["success"]
    assert b.returncode == 0, b.stderr
    result = json.loads(b.stdout)
    assert result["success"] and result.get("staged"), result
    assert "A edit." in target.read_text() and "B edit." not in target.read_text()
    replay = '''import sys
from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import write_approval as wa
print(handle_pending_subcommand(wa.SKILLS, ["approve", sys.argv[1]]))
print("PENDING", wa.pending_count(wa.SKILLS))
'''
    applied = subprocess.run([sys.executable, "-c", replay, result["pending_id"]], cwd=cwd, env=envs[1],
                             capture_output=True, text=True, timeout=20)
    assert applied.returncode == 0, applied.stderr
    assert "Approved 1" in applied.stdout and "PENDING 0" in applied.stdout, applied.stdout
    assert "A edit." in target.read_text() and "B edit." in target.read_text()
