import os
import shutil
import tempfile
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
