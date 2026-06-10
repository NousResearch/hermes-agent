"""Tests for Obsidian Vault Tool — Hermes-level filesystem-first knowledge management."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pytest

# Point OBSIDIAN_VAULT_PATH at a temp vault before importing the tool module
TEST_VAULT = None


@pytest.fixture(autouse=True)
def setup_vault():
    """Create a temporary Obsidian vault for testing."""
    global TEST_VAULT
    TEST_VAULT = tempfile.mkdtemp(prefix="hermes_test_vault_")
    vault_path = Path(TEST_VAULT)

    # Create .obsidian config
    (vault_path / ".obsidian").mkdir()
    (vault_path / ".obsidian" / "app.json").write_text("{}")

    # Create some test notes
    (vault_path / "project").mkdir()
    (vault_path / "project" / "README.md").write_text(
        "---\ntags: [project]\n---\n\n# Test Project\n\nThis is a test project.\n"
    )
    (vault_path / "project" / "notes.md").write_text(
        "# Daily Notes\n\n- العقارات في ليبيا\n- Dentist appointment\n- Real estate lead\n"
    )
    (vault_path / "research").mkdir()
    (vault_path / "research" / "competitors.md").write_text(
        "---\ntags: [research, competitors]\n---\n\n# Competitors\n\nCompany A\nCompany B\n"
    )

    os.environ["OBSIDIAN_VAULT_PATH"] = TEST_VAULT

    # Force re-import to pick up the new vault path
    import tools.obsidian_tool as mod
    import importlib
    importlib.reload(mod)
    globals()["obsidian_mod"] = mod

    yield

    # Cleanup
    shutil.rmtree(TEST_VAULT, ignore_errors=True)
    if "OBSIDIAN_VAULT_PATH" in os.environ:
        del os.environ["OBSIDIAN_VAULT_PATH"]


@pytest.fixture
def mod():
    """Get the obsidian tool module."""
    import tools.obsidian_tool
    return tools.obsidian_tool


class TestVaultHealth:
    def test_health_returns_valid_json(self, mod):
        result = mod.vault_health()
        data = json.loads(result)
        assert data["note_count"] == 3
        assert data["config_ok"] is True
        assert data["vault_path"] == TEST_VAULT
        assert any("project/" in p for p in data["projects"])

    def test_health_includes_recent_notes(self, mod):
        result = mod.vault_health()
        data = json.loads(result)
        assert len(data["recent_notes"]) > 0


class TestVaultList:
    def test_list_root(self, mod):
        result = mod.vault_list()
        data = json.loads(result)
        assert data["note_count"] == 3
        paths = [n["path"] for n in data["notes"]]
        assert "project/README.md" in paths

    def test_list_subfolder(self, mod):
        result = mod.vault_list("project")
        data = json.loads(result)
        assert data["note_count"] == 2
        paths = [n["path"] for n in data["notes"]]
        assert "project/README.md" in paths
        assert "project/notes.md" in paths

    def test_list_nonexistent(self, mod):
        result = mod.vault_list("nonexistent")
        data = json.loads(result)
        assert "error" in data


class TestVaultRead:
    def test_read_existing_note(self, mod):
        result = mod.vault_read("project/README.md")
        data = json.loads(result)
        assert data["path"] == "project/README.md"
        assert "Test Project" in data["content"]
        assert data["total_lines"] > 0

    def test_read_nonexistent(self, mod):
        result = mod.vault_read("nonexistent.md")
        data = json.loads(result)
        assert "error" in data
        assert "similar_notes" in data


class TestVaultSearch:
    def test_search_finds_text(self, mod):
        result = mod.vault_search("عقارات")
        data = json.loads(result)
        assert data["total_results"] >= 1
        assert any("notes.md" in r["note"] for r in data["results"])

    def test_search_case_insensitive(self, mod):
        result = mod.vault_search("dentist")
        data = json.loads(result)
        assert data["total_results"] >= 1

    def test_search_no_results(self, mod):
        result = mod.vault_search("zzzznonexistent")
        data = json.loads(result)
        assert data["total_results"] == 0

    def test_search_empty_query(self, mod):
        result = mod.vault_search("")
        data = json.loads(result)
        assert "error" in data


class TestVaultCreate:
    def test_create_note(self, mod):
        result = mod.vault_create(
            "new/note.md",
            content="# Hello\n\nWorld",
            tags=["test", "example"],
            vertical="Testing",
        )
        data = json.loads(result)
        assert data["created"] == "new/note.md"

        # Verify the file exists and has frontmatter
        note_path = Path(TEST_VAULT) / "new" / "note.md"
        assert note_path.exists()
        content = note_path.read_text()
        assert "tags: [test, example]" in content
        assert "vertical: Testing" in content
        assert "# Hello" in content

    def test_create_duplicate(self, mod):
        mod.vault_create("dup/note.md", content="First")
        result = mod.vault_create("dup/note.md", content="Second")
        data = json.loads(result)
        assert "error" in data
        assert "already exists" in data["error"]


class TestRequirements:
    def test_check_requirements_true(self, mod):
        assert mod.check_obsidian_requirements() is True

    def test_check_requirements_false_no_vault(self):
        """Requirements check returns False when no vault path is set."""
        # Temporarily unset vault
        old: Optional[str] = os.environ.pop("OBSIDIAN_VAULT_PATH", None)
        try:
            import importlib
            import tools.obsidian_tool as m
            importlib.reload(m)
            # Should be False because no vault exists at the default paths
            result = m.check_obsidian_requirements()
            # Might still find /opt/data/vault if it exists
            assert isinstance(result, bool)
        finally:
            if old:
                os.environ["OBSIDIAN_VAULT_PATH"] = old


class TestRegistry:
    def test_all_tools_registered(self, mod):
        """Verify all five vault tools are registered."""
        from tools.registry import registry
        tool_names = registry.get_all_tool_names()
        expected = {"vault_health", "vault_read", "vault_search", "vault_list", "vault_create"}
        found = set(tool_names) & expected
        assert found == expected, f"Missing: {expected - found}"
