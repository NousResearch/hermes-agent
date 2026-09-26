"""Runtime caches must not take ownership of updater-managed skill packages."""

import hashlib
import py_compile

import pytest

from tools import skills_sync as ss
from tools.skills_sync_bundled_ops import diff_bundled_skill, list_user_modified_bundled_skills
from tools.skills_sync_optional import _skill_file_list


def _legacy_hash(directory):
    digest = hashlib.md5()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _write(root, rel, text):
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


@pytest.fixture
def skill_tree(tmp_path, monkeypatch):
    base = tmp_path / "profile"
    bundled = base / "bundled"
    src = bundled / "coding" / "demo"
    _write(src, "SKILL.md", "---\nname: demo\n---\n# Demo\n")
    _write(src, "scripts/helper.py", "ANSWER = 1\n")
    skills = base / "skills"
    monkeypatch.setattr(ss, "HERMES_HOME", base)
    monkeypatch.setattr(ss, "SKILLS_DIR", skills)
    monkeypatch.setattr(ss, "MANIFEST_FILE", skills / ".bundled_manifest")
    monkeypatch.setattr(ss, "_get_bundled_dir", lambda: bundled)
    monkeypatch.setattr(ss, "_get_optional_dir", lambda: base / "no-optional")
    monkeypatch.setattr(ss, "_build_external_skill_index", set)
    monkeypatch.setattr(ss, "_read_suppressed_names", set)
    assert ss.sync_skills(quiet=True)["copied"] == ["demo"]
    return src, skills / "coding" / "demo"


def test_real_python_compilation_does_not_freeze_update(skill_tree):
    src, dest = skill_tree
    origin = ss._read_manifest()["demo"]
    py_compile.compile(str(dest / "scripts/helper.py"), doraise=True)
    assert list((dest / "scripts/__pycache__").glob("*.pyc"))
    assert ss._dir_hash(dest) == origin
    assert list_user_modified_bundled_skills() == []
    assert diff_bundled_skill("demo")["diffs"] == []
    _write(src, "scripts/helper.py", "ANSWER = 2\n")
    result = ss.sync_skills(quiet=True)
    assert result["updated"] == ["demo"]
    assert (dest / "scripts/helper.py").read_text() == "ANSWER = 2\n"
    assert ss._read_manifest()["demo"] == ss._dir_hash(src)


@pytest.mark.parametrize("cache_path", [
    "scripts/__pycache__/helper.cpython-311.pyc",
    ".pytest_cache/v/cache/nodeids",
    "scripts/.mypy_cache/3.11/helper.data.json",
    "scripts/.ruff_cache/0.15.0/abc",
    "scripts/helper.pyc", "scripts/helper.pyo",
])
def test_runtime_cache_is_not_hash_or_diff_content(skill_tree, cache_path):
    src, dest = skill_tree
    _write(dest, cache_path, "generated")
    assert ss._dir_hash(dest) == ss._dir_hash(src)
    assert cache_path not in _skill_file_list(dest)
    assert diff_bundled_skill("demo")["modified"] is False


@pytest.mark.parametrize("edited_path", ["SKILL.md", "scripts/helper.py", "references/notes.md", "cache/data.json", "scripts/bytecode_only.pyc"])
def test_genuine_edits_next_to_cache_are_preserved(skill_tree, edited_path):
    src, dest = skill_tree
    _write(dest, "scripts/__pycache__/helper.cpython-311.pyc", "generated")
    _write(dest, edited_path, "user-owned content")
    _write(src, "references/new-upstream.md", "new upstream documentation")
    assert [entry["name"] for entry in list_user_modified_bundled_skills()] == ["demo"]
    result = ss.sync_skills(quiet=True)
    assert result["user_modified"] == ["demo"]
    assert (dest / edited_path).read_text() == "user-owned content"
    assert not (dest / "references/new-upstream.md").exists()


def test_source_cache_is_neither_seeded_nor_recorded(skill_tree):
    src, dest = skill_tree
    _write(src, "scripts/__pycache__/helper.cpython-311.pyc", "source generated")
    _write(src, "SKILL.md", "---\nname: demo\n---\n# Changed upstream\n")
    assert ss.sync_skills(quiet=True)["updated"] == ["demo"]
    assert not (dest / "scripts/__pycache__").exists()
    ss._rmtree_writable(dest)
    ss._write_manifest({})
    assert ss.sync_skills(quiet=True)["copied"] == ["demo"]
    assert not (dest / "scripts/__pycache__").exists()


def test_legacy_cache_hash_migrates_only_with_matching_origin(skill_tree):
    src, dest = skill_tree
    _write(dest, "scripts/__pycache__/helper.cpython-311.pyc", "legacy generated")
    ss._write_manifest({"demo": _legacy_hash(dest)})
    assert list_user_modified_bundled_skills() == []
    _write(src, "scripts/helper.py", "ANSWER = 2\n")
    assert ss.sync_skills(quiet=True)["updated"] == ["demo"]
    assert ss._read_manifest()["demo"] == ss._dir_hash(src)


def test_legacy_hash_mismatch_never_rebaselines_user_edits(skill_tree):
    src, dest = skill_tree
    cache = _write(dest, "scripts/__pycache__/helper.cpython-311.pyc", "legacy generated")
    origin = _legacy_hash(dest)
    ss._write_manifest({"demo": origin})
    cache.unlink()
    _write(dest, "scripts/helper.py", "user edit\n")
    _write(src, "scripts/helper.py", "upstream edit\n")
    assert ss.sync_skills(quiet=True)["user_modified"] == ["demo"]
    assert ss._read_manifest()["demo"] == origin
    assert (dest / "scripts/helper.py").read_text() == "user edit\n"


def test_clean_manifest_hash_is_backwards_compatible(skill_tree):
    src, dest = skill_tree
    assert ss._dir_hash(src) == _legacy_hash(src)
    assert ss._dir_hash(dest) == _legacy_hash(dest)


def test_runtime_cache_does_not_hide_dotfile_edit(skill_tree):
    src, dest = skill_tree
    _write(dest, ".settings.json", "user settings")
    _write(dest, ".pytest_cache/v/cache/nodeids", "[]")
    assert ss._dir_hash(dest) != ss._dir_hash(src)
    assert ".settings.json" in _skill_file_list(dest)
    assert [e["path"] for e in diff_bundled_skill("demo")["diffs"]] == [".settings.json"]


def test_legacy_cache_origin_allows_rename_without_freezing(skill_tree):
    src, dest = skill_tree
    _write(dest, "scripts/__pycache__/helper.cpython-311.pyc", "legacy generated")
    ss._write_manifest({"demo": _legacy_hash(dest)})
    new_src = src.parent.parent / "recategorized" / "demo"
    new_src.parent.mkdir(parents=True)
    src.rename(new_src)
    _write(new_src, "scripts/helper.py", "ANSWER = 3\n")
    result = ss.sync_skills(quiet=True)
    assert result["updated"] == ["demo"]
    assert not dest.exists()
    assert (ss._skills_dir() / "recategorized/demo/scripts/helper.py").read_text() == "ANSWER = 3\n"


def test_hash_filter_is_skill_relative(tmp_path):
    # A Python environment/install prefix can itself contain a cache name.
    src = tmp_path / "__pycache__" / "demo"
    _write(src, "SKILL.md", "real skill content")
    _write(src, "scripts/helper.py", "ANSWER = 1\n")
    assert ss._dir_hash(src) == _legacy_hash(src)
    assert set(_skill_file_list(src)) == {"SKILL.md", "scripts/helper.py"}
