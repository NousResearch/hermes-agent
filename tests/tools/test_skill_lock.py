"""A user skill lock holds on every agent write path: write_file, patch and terminal.

skill_manage's side lives in test_skill_manager_tool.py::TestLockedGuard.
"""

import json

import pytest


@pytest.fixture
def locked_home(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    skills = home / "skills"
    for rel, display in (("ops/locked-skill", "locked-skill"), ("ops/free-skill", "free-skill"),
                         ("training/accelerate", "huggingface-accelerate")):
        (skills / rel / "scripts").mkdir(parents=True)
        (skills / rel / "SKILL.md").write_text(f"---\nname: {display}\ndescription: x\n---\nbody\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from tools import skill_usage
    assert skill_usage.set_locked("locked-skill", True)
    return home


def test_file_tools_refuse_a_locked_skill_and_its_record(locked_home):
    from tools.file_tools import patch_tool, write_file_tool
    locked = locked_home / "skills/ops/locked-skill/SKILL.md"
    free = locked_home / "skills/ops/free-skill/SKILL.md"
    before = locked.read_text()

    refused = [json.loads(write_file_tool(str(locked), "replaced\n")),
               json.loads(patch_tool(path=str(locked), old_string="body", new_string="changed")),
               json.loads(write_file_tool(str(locked.parent / "references/new.md"), "x\n")),
               json.loads(write_file_tool(str(locked_home / "skills/.usage.json"), "{}\n"))]
    allowed = json.loads(patch_tool(path=str(free), old_string="body", new_string="changed"))

    for res in refused:
        assert "locked" in res["error"], res
    assert locked.read_text() == before
    assert "error" not in allowed, allowed
    assert "changed" in free.read_text()


@pytest.mark.parametrize("command, blocked", [
    ("echo x > {L}/SKILL.md", True),
    ("sed -i s/a/b/ {L}/SKILL.md", True),
    ("cd {L} && echo x > SKILL.md", True),
    ("rm -rf {H}/skills/ops", True),
    ("git -C {L} checkout -- .", True),
    ("bash -c 'cp /etc/hosts {L}/SKILL.md'", True),
    ("python3 -c \"open('{L}/SKILL.md','w').write('x')\"", True),
    ("echo {{}} > {H}/skills/.usage.json", True),
    ("hermes curator unlock locked-skill", True),
    ("echo checking\nsed -i s/body/x/ {L}/SKILL.md", True),
    ("sudo -u root rm -rf {L}", True),
    ("cd {L} && (cd /) ; echo x > SKILL.md", True),
    ("cp -t {L} {H}/SKILL.md", True),
    ("cp --target-directory={L} {H}/SKILL.md", True),
    ("cp -r {H}/locked-skill {H}/skills/ops", True),
    ("tar xf {H}/a.tar -C {H}/skills/ops", True),
    ("sed -i s/body/x/ {H}/skills/ops/*/SKILL.md", True),
    ("p={L}/SKILL.md; printf x > \"$p\"", True),
    ("export D={L} && cp /etc/hosts $D/SKILL.md", True),
    ("cd {L} && python3 -c \"from pathlib import Path; Path('SKILL.md').write_text('x')\"", True),
    ("find {H}/skills -name SKILL.md -exec sed -i s/a/b/ {{}} +", True),
    ("git -C {H} add {L}/SKILL.md", True),
    ("git -C {H} add .", True),
    ("git -C {H} commit -a -m all", True),
    ("git -C {H}/skills/ops/free-skill rm {L}/SKILL.md", True),
    ("cd {H}/skills/ops/free-skill && git mv {L}/SKILL.md {H}/moved.md", True),
    ("git -C {H} pull --rebase", True),
    ("git -C {H} switch main", True),
    ("cat {L}/SKILL.md", False),
    ("python3 {L}/scripts/run.py > {H}/out.log 2>&1", False),
    ("cp {L}/SKILL.md {H}/copy.md", False),
    ("echo x > {H}/skills/ops/free-skill/SKILL.md", False),
    ("git -C {L} log --oneline", False),
    ("git -C {H} add README.md", False),
    ("git -C {H} commit -m 'unrelated notes'", False),
    ("git -C {H} commit -m 'release' && git -C {H} push origin main", False),
    ("git -C {H} push", False),
    ("git -C {H} fetch --all", False),
    ("tar xf {H}/a.tar -C {H}", False),
    ("python3 -c \"print(open('{L}/SKILL.md').read())\"", False),
    ("find {H}/skills -name SKILL.md -exec cat {{}} +", False),
    ("p={H}/out.txt; echo x > \"$p\"", False),
    ("cp -r {H}/new-skill {H}/skills/ops/new-skill", False),
])
def test_terminal_refuses_writes_into_a_locked_skill_but_not_reads(locked_home, command, blocked):
    from tools.skill_lock import command_refusal
    lock_dir = locked_home / "skills/ops/locked-skill"
    reason = command_refusal(command.format(L=lock_dir, H=locked_home), str(locked_home))
    assert (reason is not None) is blocked, reason


def test_git_in_ancestor_repo_checks_staged_paths_and_reports_tree_refusal(locked_home):
    import subprocess
    from tools.skill_lock import command_refusal
    subprocess.run(["git", "-C", str(locked_home), "init", "-q"], check=True)
    notes = locked_home / "notes.md"
    notes.write_text("hello\n")
    subprocess.run(["git", "-C", str(locked_home), "add", "notes.md"], check=True)
    assert command_refusal(f"git -C {locked_home} commit -m notes", str(locked_home)) is None

    subprocess.run(["git", "-C", str(locked_home), "add", "skills/ops/locked-skill/SKILL.md"], check=True)
    staged_refusal = command_refusal(f"git -C {locked_home} commit -m sneak", str(locked_home))
    assert staged_refusal and "locked-skill" in staged_refusal

    tree_refusal = command_refusal(f"git -C {locked_home} pull --rebase", str(locked_home))
    assert tree_refusal and "contains locked skill 'locked-skill'" in tree_refusal
    assert "git pull" in tree_refusal


def test_tree_wide_git_from_a_sibling_subdirectory_is_refused(locked_home):
    import subprocess
    from tools.skill_lock import command_refusal
    subprocess.run(["git", "-C", str(locked_home), "init", "-q"], check=True)
    docs = locked_home / "docs"
    docs.mkdir()
    for cmd in ("git reset --hard HEAD", "git stash", "git pull --rebase", "git checkout -- ..",
                "git add -A", "git commit -a -m x"):
        assert command_refusal(cmd, str(docs)), cmd
    assert command_refusal("git add notes.md", str(docs)) is None


def test_lock_resolves_by_display_name_as_well_as_directory(locked_home):
    """Locks are keyed by directory; the curator's rows carry the frontmatter name."""
    from tools import skill_usage
    from tools.skill_lock import is_locked_skill
    assert skill_usage.set_locked("accelerate", True)
    assert is_locked_skill("huggingface-accelerate")
    assert is_locked_skill("training/accelerate")
    assert not is_locked_skill("free-skill")
