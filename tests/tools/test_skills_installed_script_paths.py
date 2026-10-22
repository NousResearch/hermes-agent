"""Script paths that installed bundled skills tell the agent to run must exist after install.

Skills are copied into ``$HERMES_HOME/skills/<category>/<name>/``. When a skill is moved or
merged, commands that spell out its old install path (``$HERMES_HOME/skills/github/github-auth/
scripts/...``) keep shipping and fail only when the agent runs them.
"""

import re

from hermes_constants import get_hermes_home
from tools.skills_sync import sync_skills

_HOME_SKILL_SCRIPT = re.compile(
    r"(?:\$\{HERMES_HOME:-\$HOME/\.hermes\}|\$HERMES_HOME|~/\.hermes)/skills/"
    r"([\w.-]+(?:/[\w.-]+)*/scripts/[\w.-]+\.\w+)"
)


def test_installed_skill_script_paths_exist():
    sync_skills(quiet=True)
    skills_dir = get_hermes_home() / "skills"
    texts = [p for p in skills_dir.rglob("*") if p.is_file() and p.suffix in {".md", ".sh"}]
    assert texts, "bundled skill sync installed nothing"

    missing = sorted({
        f"{path.relative_to(skills_dir)} -> {rel}"
        for path in texts
        for rel in _HOME_SKILL_SCRIPT.findall(path.read_text(encoding="utf-8", errors="replace"))
        if not (skills_dir / rel).is_file()
    })
    assert not missing, "skills name script paths that are not installed:\n" + "\n".join(missing)
