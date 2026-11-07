"""Skill discovery/read/write contracts for custom supporting directories (#30999)."""

import json

import pytest

from tools import skill_manager_tool, skills_tool


@pytest.fixture
def skill_tree(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    skill = home / "skills" / "linked-files-contract"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: linked-files-contract\ndescription: Supporting file contract.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    for relative, content in {
        "references/guide.md": "Reference guide.",
        "steps/phase1/details.md": "Nested instructions.",
        "checks/verify.py": "print('verified')\n",
        "steps/.private/notes.md": "Hidden instructions.",
        ".internal/notes.md": "Hidden top-level instructions.",
    }.items():
        target = skill / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (skill / "empty").mkdir()
    (skill / "hidden-only" / ".private").mkdir(parents=True)
    (skill / "hidden-only" / ".private" / "notes.md").write_text("Hidden.", encoding="utf-8")
    return skill


def _view(**args):
    return json.loads(skills_tool.registry.dispatch("skill_view", {"name": "linked-files-contract", **args}))


def _manage(**args):
    token = skill_manager_tool._skill_gate_bypass.set(True)
    try:
        return json.loads(skill_manager_tool.registry.dispatch("skill_manage", {"name": "linked-files-contract", **args}))
    finally:
        skill_manager_tool._skill_gate_bypass.reset(token)


def test_registered_view_discovers_and_reads_custom_support_files(skill_tree):
    loaded = _view()
    assert loaded["success"] is True, loaded
    linked = loaded["linked_files"]
    assert linked["steps"] == ["steps/phase1/details.md"]
    assert linked["checks"] == ["checks/verify.py"]
    assert linked["references"] == ["references/guide.md"]
    assert "empty" not in linked
    assert "hidden-only" not in linked
    assert ".internal" not in linked

    for relative in (linked["steps"][0], linked["checks"][0]):
        opened = _view(file_path=relative)
        assert opened["success"] is True, opened
        assert opened["content"] == (skill_tree / relative).read_text(encoding="utf-8")


def test_missing_view_suggests_custom_files_without_hidden_directories(skill_tree):
    missing = _view(file_path="steps/missing.md")
    assert missing["success"] is False
    available = [path for group in missing["available_files"].values() for path in group]
    assert "steps/phase1/details.md" in available
    assert "checks/verify.py" in available
    assert "steps/.private/notes.md" not in available
    assert ".internal/notes.md" not in available
    assert "hidden-only/.private/notes.md" not in available


def test_visible_directories_keep_dotfiles_in_read_discovery(skill_tree):
    for relative in ("assets/.gitignore", "templates/.guide.md"):
        target = skill_tree / relative
        target.parent.mkdir()
        target.write_text("Useful supporting file.", encoding="utf-8")

    loaded = _view()
    assert loaded["success"] is True, loaded
    assert "assets/.gitignore" in loaded["linked_files"]["assets"]
    assert "templates/.guide.md" in loaded["linked_files"]["templates"]
    missing = _view(file_path="missing.md")
    assert "assets/.gitignore" in missing["available_files"]["assets"]
    assert "templates/.guide.md" in missing["available_files"]["templates"]
    for relative in ("assets/.gitignore", "templates/.guide.md"):
        opened = _view(file_path=relative)
        assert opened["success"] is True, opened
        assert opened["content"] == "Useful supporting file."


@pytest.mark.parametrize("action", ["write_file", "remove_file"])
def test_custom_read_discovery_does_not_expand_write_paths(skill_tree, action):
    target = skill_tree / "steps" / "phase1" / "details.md"
    original = target.read_text(encoding="utf-8")
    result = _manage(action=action, file_path="steps/phase1/details.md", file_content="Changed.")
    assert result["success"] is False, result
    assert "must be under" in result["error"].lower(), result
    assert target.read_text(encoding="utf-8") == original

    # A standard support directory still reaches the mutation handler.
    allowed = _manage(action="write_file", file_path="references/new.md", file_content="Allowed.")
    assert allowed["success"] is True, allowed
    assert (skill_tree / "references" / "new.md").read_text(encoding="utf-8") == "Allowed."


def test_missing_remove_suggests_only_writable_support_paths(skill_tree):
    missing = _manage(action="remove_file", file_path="references/missing.md")
    assert missing["success"] is False
    assert "references/guide.md" in missing["available_files"]
    assert "steps/phase1/details.md" not in missing["available_files"]
    assert "checks/verify.py" not in missing["available_files"]
    assert (skill_tree / "steps" / "phase1" / "details.md").is_file()


def test_hidden_content_does_not_enable_custom_directory(skill_tree):
    # A hidden eligible file must not make unrelated visible binary files discoverable.
    (skill_tree / "hidden-only" / "payload.bin").write_bytes(b"\x00\x01")
    loaded = _view()
    assert loaded["success"] is True, loaded
    assert "hidden-only" not in loaded["linked_files"]
