"""Tests for tools/skills_hub_local.py — the "local-dir" source and its LocalDirsManager."""

from pathlib import Path

import pytest

from tools.skills_hub import LocalDirsManager
from tools.skills_hub_local import LocalFolderSource


def _patch_local_dirs_manager(monkeypatch, path):
    """LocalDirsManager.DEFAULT_PATH is bound to the ``_local_dirs_file`` resolver at class
    definition time, so patching the module-level function is a no-op for instances that don't
    pass an explicit ``path=``. Patch the class attribute itself instead — this is what every
    ``LocalDirsManager()`` call inside ``tools.skills_hub_local`` resolves against."""
    monkeypatch.setattr(LocalDirsManager, "DEFAULT_PATH", staticmethod(lambda: path))


def _write_skill(root: Path, name: str, description: str = "A test skill.", extra_files=None):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nlicense: MIT\n"
        "metadata:\n  hermes:\n    tags: [test]\n---\n\n# Skill\n\nBody.\n",
        encoding="utf-8",
    )
    for rel, content in (extra_files or {}).items():
        path = skill_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return skill_dir


class TestLocalDirsManager:
    def test_add_is_idempotent_and_resolves_path(self, tmp_path):
        mgr = LocalDirsManager(path=tmp_path / "local_dirs.json")
        target = tmp_path / "skills-folder"
        target.mkdir()
        assert mgr.add(str(target)) is True
        assert mgr.add(str(target)) is False  # already configured
        assert mgr.list_dirs() == [str(target.resolve())]

    def test_remove_existing_and_missing(self, tmp_path):
        mgr = LocalDirsManager(path=tmp_path / "local_dirs.json")
        target = tmp_path / "skills-folder"
        target.mkdir()
        mgr.add(str(target))
        assert mgr.remove(str(target)) is True
        assert mgr.list_dirs() == []
        assert mgr.remove(str(target)) is False

    def test_load_corrupt_json(self, tmp_path):
        dirs_file = tmp_path / "local_dirs.json"
        dirs_file.write_text("not json")
        mgr = LocalDirsManager(path=dirs_file)
        assert mgr.load() == []


class TestLocalFolderSource:
    @pytest.fixture(autouse=True)
    def _configure_dir(self, tmp_path, monkeypatch):
        self.root = tmp_path / "other-agent-skills"
        self.root.mkdir()
        _write_skill(self.root, "ab-test-setup", "Design and run A/B tests.")
        _write_skill(self.root, "yaml-linter", "Lint YAML config files.")
        mgr_path = tmp_path / "local_dirs.json"
        _patch_local_dirs_manager(monkeypatch, mgr_path)
        from tools.skills_hub import LocalDirsManager as _Mgr
        _Mgr().add(str(self.root))
        self.src = LocalFolderSource()

    def test_list_all_returns_every_configured_skill(self):
        metas = self.src.list_all()
        assert {m.name for m in metas} == {"ab-test-setup", "yaml-linter"}
        assert all(m.trust_level == "community" for m in metas)
        assert all(m.source == "local-dir" for m in metas)

    def test_search_filters_by_query(self):
        results = self.src.search("a/b test")
        assert len(results) == 1
        assert results[0].name == "ab-test-setup"

    def test_search_empty_query_matches_all(self):
        results = self.src.search("")
        assert {r.name for r in results} == {"ab-test-setup", "yaml-linter"}

    def test_inspect_and_fetch_round_trip(self):
        meta = self.src.inspect("ab-test-setup")
        assert meta is not None
        assert meta.identifier.startswith("local-dir:")
        bundle = self.src.fetch(meta.identifier)
        assert bundle is not None
        assert bundle.name == "ab-test-setup"
        assert "SKILL.md" in bundle.files
        assert bundle.trust_level == "community"

    def test_fetch_by_bare_name(self):
        bundle = self.src.fetch("ab-test-setup")
        assert bundle is not None
        assert bundle.name == "ab-test-setup"

    def test_fetch_unknown_skill_returns_none(self):
        assert self.src.fetch("does-not-exist") is None

    def test_fetch_skips_dotfiles_and_pycache(self):
        _write_skill(self.root, "with-junk", extra_files={
            ".DS_Store": "junk", "__pycache__/mod.pyc": "junk", "scripts/run.py": "print(1)",
        })
        bundle = self.src.fetch("with-junk")
        assert bundle is not None
        assert ".DS_Store" not in bundle.files
        assert not any("__pycache__" in f for f in bundle.files)
        assert "scripts/run.py" in bundle.files

    def test_fetch_rejects_symlink_escape(self, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        skill_dir = _write_skill(self.root, "with-symlink")
        (skill_dir / "leak.txt").symlink_to(outside)
        bundle = self.src.fetch("with-symlink")
        assert bundle is not None
        assert "leak.txt" not in bundle.files

    def test_no_configured_dirs_returns_empty(self, tmp_path, monkeypatch):
        _patch_local_dirs_manager(monkeypatch, tmp_path / "empty_local_dirs.json")
        empty_src = LocalFolderSource()
        assert empty_src.list_all() == []
        assert empty_src.search("anything") == []
        assert empty_src.fetch("ab-test-setup") is None
