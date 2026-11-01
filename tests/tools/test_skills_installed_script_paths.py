"""Script paths an installed bundled SKILL.md tells the agent to run must exist after install.

Skills are copied into ``$HERMES_HOME/skills/<category>/<name>/``. When a skill moves category
or is merged, commands that spell out its old install path (``~/.hermes/skills/maps/scripts/...``)
keep shipping and fail only when the agent runs them.
"""

import re

from hermes_constants import get_hermes_home
from tools.skills_sync import sync_skills

_HOME_SKILL_SCRIPT = re.compile(
    r"(?:\$\{HERMES_HOME:-\$HOME/\.hermes\}|\$HERMES_HOME|~/\.hermes)/skills/"
    r"([\w.-]+(?:/[\w.-]+)*/scripts/[\w.-]+\.\w+)"
)


def test_installed_skill_md_script_paths_exist():
    sync_skills(quiet=True)
    skills_dir = get_hermes_home() / "skills"
    skill_mds = list(skills_dir.rglob("SKILL.md"))
    assert skill_mds, "bundled skill sync installed nothing"

    missing = sorted({
        f"{path.relative_to(skills_dir)} -> {rel}"
        for path in skill_mds
        for rel in _HOME_SKILL_SCRIPT.findall(path.read_text(encoding="utf-8", errors="replace"))
        if not (skills_dir / rel).is_file()
    })
    assert not missing, "SKILL.md files name script paths that are not installed:\n" + "\n".join(missing)
