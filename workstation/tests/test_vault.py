import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
import pytest

from workstation.vault import (
    VaultManager,
    canonicalize_title,
    extract_tags,
    extract_wikilinks,
    parse_frontmatter_and_content,
)


def test_canonicalize_title():
    assert canonicalize_title("My Note.md") == "My Note"
    assert canonicalize_title("folder/subfolder/Deep Note.md") == "Deep Note"
    assert canonicalize_title("Plain Title") == "Plain Title"


def test_extract_wikilinks():
    text = """
    Check out [[Artificial Intelligence]] and [[Machine Learning#Supervised|ML]].
    Also see [[Deep Learning|DL]] and standard text without links.
    """
    links = extract_wikilinks(text)
    assert len(links) == 3

    assert links[0].target == "Artificial Intelligence"
    assert links[0].section is None
    assert links[0].alias is None

    assert links[1].target == "Machine Learning"
    assert links[1].section == "Supervised"
    assert links[1].alias == "ML"

    assert links[2].target == "Deep Learning"
    assert links[2].alias == "DL"


def test_extract_tags():
    text = """
    # Markdown Header 1
    This is an inline #research tag and another #agent/memory tag.
    ```python
    # this is a python comment, not a tag
    x = 1
    ```
    """
    tags = extract_tags(text, frontmatter_tags=["pkm", "obsidian"])
    assert "pkm" in tags
    assert "obsidian" in tags
    assert "research" in tags
    assert "agent/memory" in tags
    assert "this" not in tags  # python comment inside code block excluded


def test_parse_frontmatter():
    text = """---
title: Custom Note
tags: [alpha, beta]
aliases: [CustomAlias]
status: in-progress
---

# Real Content
Here is the note body.
"""
    fm, body = parse_frontmatter_and_content(text)
    assert fm["title"] == "Custom Note"
    assert fm["tags"] == ["alpha", "beta"]
    assert fm["aliases"] == ["CustomAlias"]
    assert fm["status"] == "in-progress"
    assert body.startswith("# Real Content")


@pytest.fixture
def temp_vault():
    temp_dir = tempfile.mkdtemp()
    mgr = VaultManager(vault_dir=temp_dir)
    yield mgr
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_vault_lifecycle_and_bidirectional_links(temp_vault: VaultManager):
    # Create Note A
    note_a = temp_vault.write_note(
        title="Note A",
        content="This is note A referencing [[Note B]] and [[Note C|Alias C]].",
        tags=["project-alpha"],
    )
    assert note_a["title"] == "Note A"
    assert "Note B" in note_a["forward_links"]
    assert "Note C" in note_a["forward_links"]

    # Create Note B
    note_b = temp_vault.write_note(
        title="Note B",
        content="This is note B with backlinks from A.",
        tags=["status/todo"],
    )
    assert note_b["title"] == "Note B"
    assert "Note A" in note_b["backlinks"]

    # Verify Note A backlinks and forward links via get_note
    data_b = temp_vault.get_note("Note B")
    assert data_b is not None
    assert "Note A" in data_b["backlinks"]

    # Test Append
    temp_vault.append_note("Note B", "Appended line with link to [[Note A]].")
    data_b_updated = temp_vault.get_note("Note B")
    assert "Note A" in data_b_updated["forward_links"]
    data_a = temp_vault.get_note("Note A")
    assert "Note B" in data_a["backlinks"]

    # Test Search
    results = temp_vault.search("project-alpha")
    assert len(results) >= 1
    assert results[0]["title"] == "Note A"

    # Test Graph Generation
    graph = temp_vault.get_graph()
    node_ids = {n["id"] for n in graph["nodes"]}
    assert "Note A" in node_ids
    assert "Note B" in node_ids

    # Test Link Suggestions (Autocomplete)
    suggestions = temp_vault.suggest_wikilinks("Note")
    titles = {s["title"] for s in suggestions}
    assert "Note A" in titles
    assert "Note B" in titles

    # Test Delete
    assert temp_vault.delete_note("Note A") is True
    assert temp_vault.get_note("Note A") is None
    data_b_post_delete = temp_vault.get_note("Note B")
    assert "Note A" not in data_b_post_delete["backlinks"]


@pytest.mark.parametrize("subfolder", ["../outside", "C:\\outside", "\\\\server\\share", "/tmp/outside"])
def test_write_rejects_escaping_subfolders_without_side_effect(tmp_path: Path, subfolder: str):
    vault = tmp_path / "vault"
    outside = tmp_path / "outside"
    manager = VaultManager(str(vault))

    with pytest.raises(ValueError):
        manager.write_note("escaped", "must stay inside", subfolder=subfolder)

    assert not outside.exists()
    assert list(vault.rglob("*.md")) == []


@pytest.mark.parametrize("title", ["", "   ", "../escape", "folder/note", "folder\\note", "bad:name"])
def test_write_rejects_malformed_titles(tmp_path: Path, title: str):
    manager = VaultManager(str(tmp_path / "vault"))
    with pytest.raises(ValueError):
        manager.write_note(title, "unsafe")
    assert list((tmp_path / "vault").rglob("*.md")) == []


def test_custom_vault_root_accepts_normal_nested_note(tmp_path: Path):
    custom = tmp_path / "existing-obsidian"
    custom.mkdir()
    (custom / ".obsidian").mkdir()
    manager = VaultManager(str(custom))

    note = manager.write_note("Safe Note", "hello", subfolder="Projects/Hermes")

    assert note["rel_path"] == "Projects/Hermes/Safe Note.md"
    assert (custom / "Projects" / "Hermes" / "Safe Note.md").is_file()


def test_scan_and_mutations_ignore_file_symlink_escape(tmp_path: Path):
    vault = tmp_path / "vault"
    outside = tmp_path / "outside.md"
    vault.mkdir()
    outside.write_text("external secret", encoding="utf-8")
    link = vault / "linked.md"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are not supported on this host")

    manager = VaultManager(str(vault))
    assert manager.get_note("linked") is None
    assert manager.delete_note("linked") is False
    with pytest.raises(ValueError):
        manager.write_note("linked", "overwrite attempt")
    assert outside.read_text(encoding="utf-8") == "external secret"


def test_scan_ignores_directory_symlink_escape(tmp_path: Path):
    vault = tmp_path / "vault"
    outside = tmp_path / "outside"
    vault.mkdir()
    outside.mkdir()
    (outside / "secret.md").write_text("external secret", encoding="utf-8")
    link = vault / "linked-dir"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not supported on this host")

    manager = VaultManager(str(vault))
    assert manager.list_notes() == []
    with pytest.raises(ValueError):
        manager.write_note("escape", "unsafe", subfolder="linked-dir")
    assert not (outside / "escape.md").exists()


def test_external_change_watcher_coalesces_and_stops(tmp_path: Path):
    vault = tmp_path / "vault"
    manager = VaultManager(str(vault))
    changed = threading.Event()
    manager.start_watcher(interval=0.01, debounce=0.03, on_change=lambda _result: changed.set())
    try:
        note = vault / "External.md"
        note.write_text("first", encoding="utf-8")
        note.write_text("second #updated", encoding="utf-8")
        assert changed.wait(2)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and manager.get_note("External") is None:
            time.sleep(0.01)
        assert manager.get_note("External")["tags"] == ["updated"]

        changed.clear()
        note.rename(vault / "Renamed.md")
        assert changed.wait(2)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and manager.get_note("Renamed") is None:
            time.sleep(0.01)
        assert manager.get_note("External") is None
        assert manager.get_note("Renamed") is not None
    finally:
        manager.stop_watcher()

    assert manager._watch_thread is None
