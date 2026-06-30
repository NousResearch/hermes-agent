"""Tests for tools/memory_migrate.py — flat → scoped graph migration.

Migration must be SAFE: back up first, never silently delete, be idempotent,
and use a temp HERMES_HOME (never touch real user profiles).
"""

import json
import pytest
from pathlib import Path

import yaml

from tools.memory_migrate import migrate_profile_to_scoped


@pytest.fixture()
def profile(tmp_path, monkeypatch):
    """A temp profile: HERMES_HOME with memories/ and a config.yaml."""
    mem_dir = tmp_path / "memories"
    mem_dir.mkdir(parents=True)
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: mem_dir)
    monkeypatch.setattr("tools.memory_migrate.get_memory_dir", lambda: mem_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "memory:\n  memory_enabled: true\n  user_profile_enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("tools.memory_migrate.get_config_path", lambda: config_path)
    return mem_dir, config_path


def _scoped_nodes(mem_dir):
    path = mem_dir / "SCOPED_MEMORY.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_migration_creates_backup_scoped_file_and_enables_config(profile):
    mem_dir, config_path = profile
    (mem_dir / "MEMORY.md").write_text(
        "Botson sends need explicit final confirmation.\n"
        "§\n"
        "TermFleet keeps Watchpost-compatible ID tables.\n",
        encoding="utf-8",
    )
    (mem_dir / "USER.md").write_text(
        "User prefers concise replies.\n",
        encoding="utf-8",
    )

    report = migrate_profile_to_scoped()

    assert report["success"] is True
    # Backups taken for the files that existed.
    for key in ("MEMORY.md", "USER.md", "config.yaml"):
        assert key in report["backups"]
        assert Path(report["backups"][key]).exists()

    # Scoped file created with nodes.
    nodes = _scoped_nodes(mem_dir)
    assert len(nodes) >= 1
    for node in nodes:
        assert node["type"] in {
            "user_preference", "project", "entity", "workflow",
            "vocabulary", "environment_fact", "safety_rule",
        }
        assert node["id"]

    # Config enabled.
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg["memory"]["scoped_memory_enabled"] is True
    assert cfg["memory"]["scoped_memory_char_limit"] == 4000


def test_migration_preserves_data(profile):
    mem_dir, _ = profile
    (mem_dir / "MEMORY.md").write_text(
        "Botson sends need explicit final confirmation.\n"
        "§\n"
        "TermFleet keeps Watchpost-compatible ID tables.\n",
        encoding="utf-8",
    )

    migrate_profile_to_scoped()

    # The project-specific facts must survive somewhere — scoped or flat.
    scoped_text = "\n".join(n["content"] for n in _scoped_nodes(mem_dir))
    flat_text = ""
    if (mem_dir / "MEMORY.md").exists():
        flat_text = (mem_dir / "MEMORY.md").read_text(encoding="utf-8")
    combined = scoped_text + flat_text
    assert "Botson sends need explicit final confirmation." in combined
    assert "TermFleet keeps Watchpost-compatible ID tables." in combined


def test_migration_classifies_safety_and_keeps_global_pref_flat(profile):
    mem_dir, _ = profile
    (mem_dir / "MEMORY.md").write_text(
        "Botson sends need explicit final confirmation.\n",
        encoding="utf-8",
    )
    (mem_dir / "USER.md").write_text(
        "User prefers concise replies.\n",
        encoding="utf-8",
    )

    migrate_profile_to_scoped()

    nodes = _scoped_nodes(mem_dir)
    botson = next(n for n in nodes if "Botson" in n["content"])
    assert botson["type"] == "safety_rule"
    assert any(e.lower() == "botson" for e in botson["entities"])

    # Global communication preference stays in USER.md (tiny global profile).
    user_text = (mem_dir / "USER.md").read_text(encoding="utf-8")
    assert "User prefers concise replies." in user_text


def test_migration_is_idempotent(profile):
    mem_dir, _ = profile
    (mem_dir / "MEMORY.md").write_text(
        "Botson sends need explicit final confirmation.\n"
        "§\n"
        "TermFleet keeps Watchpost-compatible ID tables.\n",
        encoding="utf-8",
    )

    migrate_profile_to_scoped()
    first = _scoped_nodes(mem_dir)
    migrate_profile_to_scoped()
    second = _scoped_nodes(mem_dir)

    # Re-running must not duplicate nodes.
    assert len(second) == len(first)
    contents = [n["content"] for n in second]
    assert len(contents) == len(set(contents))


def test_migration_then_retrieval_loads_relevant_skips_unrelated(profile):
    from tools.memory_tool import MemoryStore

    mem_dir, _ = profile
    (mem_dir / "MEMORY.md").write_text(
        "Botson sends need explicit final confirmation.\n"
        "§\n"
        "TermFleet keeps Watchpost-compatible ID tables.\n",
        encoding="utf-8",
    )
    migrate_profile_to_scoped()

    store = MemoryStore(scoped_memory_enabled=True)
    store.load_from_disk()
    block = store.format_for_system_prompt(
        "memory", query="Confirm a Botson send", cwd="", session_source="telegram",
    )
    assert "Botson sends need explicit final confirmation." in block
    assert "TermFleet keeps Watchpost-compatible ID tables." not in block
