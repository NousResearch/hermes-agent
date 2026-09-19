"""Test coverage for tools/skills_sync.py — path and manifest helpers."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.skills_sync import (
    _hermes_home,
    _skills_dir,
    _manifest_file,
    _build_external_skill_index,
    _read_manifest,
    _read_suppressed_names,
)

@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "skills").mkdir()
    return tmp_path

class TestHermesHome:
    def test_returns_path_object(self, fake_home):
        assert isinstance(_hermes_home(), Path)
        assert _hermes_home() == fake_home

class TestSkillsDir:
    def test_is_hermes_home_skills(self, fake_home):
        assert _skills_dir() == fake_home / "skills"

class TestManifestFile:
    def test_is_dotfile_in_skills_dir(self, fake_home):
        mf = _manifest_file()
        assert ".bundled_manifest" in str(mf)

class TestBuildExternalSkillIndex:
    def test_empty_dir_returns_empty_set(self, fake_home):
        result = _build_external_skill_index()
        assert isinstance(result, set)
        assert len(result) == 0

    def test_valid_manifest_parsed(self, fake_home):
        mf = _manifest_file()
        mf.parent.mkdir(parents=True, exist_ok=True)
        mf.write_text("skill-a:hash-a\nskill-b:hash-b\n", encoding="utf-8")
        result = _read_manifest()
        assert "skill-a" in result
        assert "skill-b" in result
        assert result["skill-a"] == "hash-a"

class TestReadSuppressedNames:
    def test_missing_file_returns_empty_set(self, fake_home):
        assert _read_suppressed_names() == set()
