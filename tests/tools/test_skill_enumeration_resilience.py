"""Skill enumeration survives a subtree that vanishes mid-walk.

Skill installs/updates/removals mutate ``skills/`` while other surfaces enumerate it
(profile listings, the curator, ``skill_manage``). A directory that disappears between
the parent listing and its own scan must skip that subtree, never abort the whole
enumeration. Regression: ``FileNotFoundError: [WinError 3]`` out of
``skill_manager_tool._iter_skill_dirs`` while a skill directory was being replaced.
"""
import os

import pytest


def _make_skill(root, name: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\nbody\n", encoding="utf-8")


@pytest.fixture
def skills_root(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    for name in ("alpha", "beta", "gamma"):
        _make_skill(root, name)
    return root


def _scandir_failing_on(target: str):
    """``os.scandir`` that raises FileNotFoundError for *target* only (the directory
    the OS reports as gone once the walker reaches it)."""
    real = os.scandir

    def fake(path=".", *args, **kwargs):
        if os.path.basename(str(path)) == target:
            raise FileNotFoundError(3, "System cannot find the path specified", str(path))
        return real(path, *args, **kwargs)

    return fake


def test_iter_skill_dirs_skips_vanished_subtree(skills_root, monkeypatch):
    from tools.skill_manager_tool import _iter_skill_dirs

    monkeypatch.setattr(os, "scandir", _scandir_failing_on("beta"))
    names = {d.name for d in _iter_skill_dirs(skills_root)}

    assert "beta" not in names, "a vanished skill dir must be skipped, not reported"
    assert {"alpha", "gamma"} <= names, "surviving skills must still enumerate"


def test_curator_scan_skips_vanished_subtree(skills_root, monkeypatch):
    from tools.skill_usage import _iter_skill_mds

    monkeypatch.setattr(os, "scandir", _scandir_failing_on("beta"))
    names = {name for name, _ in _iter_skill_mds(skills_root, local_only=False)}

    assert "beta" not in names
    assert {"alpha", "gamma"} <= names
