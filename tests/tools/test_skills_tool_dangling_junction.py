"""Regression for #121698: a stale Windows skill junction must not poison lookups."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.skills_tool import skill_view


@pytest.mark.windows_only
def test_profile_skill_lookup_skips_dangling_junction(tmp_path, monkeypatch):
    homes = [tmp_path / name for name in ("profile-a", "profile-b")]
    for home in homes:
        skill = home / "skills" / "working"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: working\ndescription: A working skill.\n---\n# Working\n",
            encoding="utf-8",
        )

    target = tmp_path / "removed-target"
    target.mkdir()
    junction = homes[1] / "skills" / "broken"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)],
                   check=True, capture_output=True)
    shutil.rmtree(target)
    try:
        for home in (homes[0], homes[1], homes[0]):
            monkeypatch.setenv("HERMES_HOME", str(home))
            result = json.loads(skill_view("working", preprocess=False))
            assert result["success"] is True, result
            assert Path(result["skill_dir"]) == home / "skills" / "working"
        monkeypatch.setenv("HERMES_HOME", str(homes[1]))
        missing = json.loads(skill_view("missing", preprocess=False))
        assert missing["success"] is False
        assert "not found" in missing["error"]
    finally:
        os.rmdir(junction)
