"""Tests for tools/skills_tool_plugin.py — plugin-skill linked-file support dirs.

Covers the plugin path of skill_view's linked_files: evals/ must be surfaced
for plugin-provided skills the same as local ones (the two paths share
support-dir lists that must stay in sync).
"""

from __future__ import annotations

from pathlib import Path

from tools.skills_tool_plugin import (
    _available_skill_files,
    _plugin_skill_linked_files,
)


def _make_plugin_skill(root: Path) -> Path:
    skill = root / "my-plugin-skill"
    (skill / "references").mkdir(parents=True)
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: my-plugin-skill\n---\n")
    (skill / "references" / "api.md").write_text("# api\n")
    (skill / "evals" / "eval_cases.yaml").write_text("static_checks: []\n")
    (skill / "evals" / "run_evals.py").write_text("print('evals')\n")
    return skill


def test_plugin_linked_files_surface_evals_dir(tmp_path):
    skill = _make_plugin_skill(tmp_path)
    linked = _plugin_skill_linked_files(skill)
    assert linked is not None
    assert "evals" in linked, f"evals missing from plugin linked_files: {linked}"
    assert "evals/eval_cases.yaml" in linked["evals"]
    assert "evals/run_evals.py" in linked["evals"]
    assert "references/api.md" in linked["references"]


def test_available_skill_files_groups_evals_dir(tmp_path):
    skill = _make_plugin_skill(tmp_path)
    available = _available_skill_files(skill)
    assert "evals" in available, f"evals missing from available files: {available}"
    assert "evals/eval_cases.yaml" in available["evals"]
    assert "evals/run_evals.py" in available["evals"]
