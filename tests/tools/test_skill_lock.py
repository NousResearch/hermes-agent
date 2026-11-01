"""A user skill lock holds on every agent write path: write_file, patch and terminal.

skill_manage's side lives in test_skill_manager_tool.py::TestLockedGuard.
"""

import json

import pytest


@pytest.fixture
def locked_home(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    skills = home / "skills"
    for rel in ("ops/locked-skill", "ops/free-skill"):
        (skills / rel / "scripts").mkdir(parents=True)
        (skills / rel / "SKILL.md").write_text(f"---\nname: {rel.split('/')[1]}\ndescription: x\n---\nbody\n")
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
    ("cat {L}/SKILL.md", False),
    ("python3 {L}/scripts/run.py > {H}/out.log 2>&1", False),
    ("cp {L}/SKILL.md {H}/copy.md", False),
    ("echo x > {H}/skills/ops/free-skill/SKILL.md", False),
    ("git -C {L} log --oneline", False),
])
def test_terminal_refuses_writes_into_a_locked_skill_but_not_reads(locked_home, command, blocked):
    from tools.skill_lock import command_refusal
    lock_dir = locked_home / "skills/ops/locked-skill"
    reason = command_refusal(command.format(L=lock_dir, H=locked_home), str(locked_home))
    assert (reason is not None) is blocked, reason
