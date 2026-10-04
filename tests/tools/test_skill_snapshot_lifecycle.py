import json
import os
import subprocess
import sys
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.credential_files import get_skills_directory_mount


def make_home(path, text):
    skills = path / 'skills'
    (skills / 'case/references').mkdir(parents=True)
    (skills / 'case/SKILL.md').write_text(text)
    (skills / 'case/references/proof.md').write_text('reference')
    (path / 'private.txt').write_text('private')
    (skills / 'private-link').symlink_to(path / 'private.txt')
    (skills / '.hub').mkdir()
    (skills / '.hub/SKILL.md').write_text('excluded')
    return path


def mount(home):
    token = set_hermes_home_override(home)
    try:
        return Path(get_skills_directory_mount()[0]['host_path'])
    finally:
        reset_hermes_home_override(token)


def test_profile_mounts_survive_alternation_and_skill_updates(tmp_path):
    a = make_home(tmp_path / 'a', 'A')
    b = make_home(tmp_path / 'b', 'B')
    first = mount(a)
    inode = first.stat().st_ino
    second = mount(b)
    assert (first / 'case/SKILL.md').read_text() == 'A'
    assert (second / 'case/SKILL.md').read_text() == 'B'
    again = mount(a)
    assert again == first and again.stat().st_ino == inode
    assert (first / 'case/references/proof.md').read_text() == 'reference'
    assert not (first / 'private-link').exists()
    assert not (first / '.hub').exists()
    (a / 'skills/case/SKILL.md').write_text('new A')
    updated = mount(a)
    assert (updated / 'case/SKILL.md').read_text() == 'new A'
    assert (first / 'case/SKILL.md').read_text() == 'A'
    assert updated != first


def test_published_mount_survives_creator_exit_and_reuses_on_next_process(tmp_path):
    home = make_home(tmp_path / 'profile', 'A')
    program = 'import json; from tools.credential_files import get_skills_directory_mount; print(json.dumps(get_skills_directory_mount()[0]))'
    child_env = {**os.environ, 'HERMES_HOME': str(home)}
    first = json.loads(subprocess.check_output([sys.executable, '-c', program], env=child_env, text=True))
    path = Path(first['host_path'])
    assert (path / 'case/SKILL.md').read_text() == 'A'
    second = json.loads(subprocess.check_output([sys.executable, '-c', program], env=child_env, text=True))
    assert second['host_path'] == first['host_path']
    assert (path / 'case/SKILL.md').read_text() == 'A'
