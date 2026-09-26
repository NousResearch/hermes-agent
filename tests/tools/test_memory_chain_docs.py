"""Chain documents (#116564): MEMORY.md as an anchor for a chain of memory documents.

The anchor (MEMORY.md) stays small and keeps its char cap; documents it references with
``@doc:<name>`` live in ``memories/docs/<name>.md`` as first-class memory the tool can read
and append to, with NO char limit and without entering the system prompt.
"""

import json

import pytest

from tools.memory_tool import MemoryStore, memory_tool

ANCHOR_REF = "Dated memory chain: @doc:conventions-2026-09"
DOC = "conventions-2026-09"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=200, user_char_limit=150)
    s.load_from_disk()
    return s


def _call(store, **kwargs):
    return json.loads(memory_tool(store=store, **kwargs))


def test_referenced_chain_doc_appends_uncapped(store, tmp_path):
    assert _call(store, action="add", target="memory", content=ANCHOR_REF)["success"] is True

    big = "x" * 500  # 2.5x the anchor's cap: chain docs are not budgeted
    result = _call(store, action="add", target="memory", doc=DOC, content=big)

    assert result["success"] is True, result
    assert (tmp_path / "docs" / f"{DOC}.md").read_text(encoding="utf-8") == big
    # The anchor stayed small: the doc's size never lands on the anchor's budget.
    assert store._char_count("memory") < store.memory_char_limit


def test_unreferenced_chain_doc_write_is_refused(store, tmp_path):
    result = _call(store, action="add", target="memory", doc="orphan", content="nope")

    assert result["success"] is False
    assert "@doc:orphan" in result["error"]  # tells the model how to link it
    assert not (tmp_path / "docs" / "orphan.md").exists()


def test_read_returns_doc_entries_and_anchor_chain_list(store):
    _call(store, action="add", target="memory", content=ANCHOR_REF)
    _call(store, action="add", target="memory", doc=DOC, content="First note")
    _call(store, action="add", target="memory", doc=DOC, content="Second note")

    doc = _call(store, action="read", target="memory", doc=DOC)
    assert doc["entries"] == ["First note", "Second note"]
    assert "uncapped" in doc["usage"]
    # A '.md' suffix is accepted for a doc name.
    assert _call(store, action="read", target="memory", doc=f"{DOC}.md")["entries"] == doc["entries"]

    anchor = _call(store, action="read", target="memory")
    assert anchor["entries"] == [ANCHOR_REF]
    assert anchor["chain_docs"] == [{"name": DOC, "entries": 2, "chars": len("First note\n§\nSecond note")}]


def test_anchor_cap_still_applies(store):
    _call(store, action="add", target="memory", content=ANCHOR_REF)

    result = _call(store, action="add", target="memory", content="y" * 400)

    assert result["success"] is False
    assert "would exceed the limit" in result["error"]


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", " "])
def test_unsafe_doc_names_are_rejected(store, tmp_path, name):
    result = _call(store, action="add", target="memory", doc=name, content="x")

    assert result["success"] is False, name
    assert list(tmp_path.rglob("*.md")) == []  # nothing written, anywhere


def test_chain_docs_hang_off_memory_not_user(store):
    result = _call(store, action="add", target="user", doc=DOC, content="x")

    assert result["success"] is False
    assert "MEMORY.md anchor" in result["error"]


def test_anchor_block_names_chain_docs_without_load(store):
    _call(store, action="add", target="memory", content=ANCHOR_REF)
    _call(store, action="add", target="memory", doc=DOC, content="secret detail")
    store.load_from_disk()

    block = store.format_for_system_prompt("memory")

    assert DOC in block and "Chain docs" in block
    assert "secret detail" not in block  # contents stay out of the prompt


def test_orphaned_chain_doc_is_reported_on_anchor_read(store):
    """Editing away the anchor entry that named a doc leaves the file behind: the doc keeps
    its entries and stays readable, but the chain line no longer mentions it — so an anchor
    read is the only place the model can learn it still exists (#116564)."""
    _call(store, action="add", target="memory", content=ANCHOR_REF)
    _call(store, action="add", target="memory", doc=DOC, content="First note")
    assert _call(store, action="replace", target="memory", old_text="@doc:",
                 new_text="Chain index: see docs/")["success"] is True

    anchor = _call(store, action="read", target="memory")

    assert anchor["chain_docs"] == []
    assert anchor["orphan_docs"] == [{"name": DOC, "entries": 1, "chars": len("First note")}]
    # Still readable: that is exactly why being listed matters.
    assert _call(store, action="read", target="memory", doc=DOC)["entries"] == ["First note"]


def test_orphaned_chain_doc_write_names_the_file_and_the_ways_out(store, tmp_path):
    _call(store, action="add", target="memory", content=ANCHOR_REF)
    _call(store, action="add", target="memory", doc=DOC, content="First note")
    _call(store, action="replace", target="memory", old_text="@doc:", new_text="Chain index: see docs/")

    result = _call(store, action="add", target="memory", doc=DOC, content="more")

    assert result["success"] is False
    assert str(tmp_path / "docs" / f"{DOC}.md") in result["error"]
    assert "Re-link" in result["error"]


def test_anchor_at_cap_names_the_way_to_link_a_doc(store):
    """The reference entry itself costs anchor budget, so at the cap the order has to be
    named: free space first, then link, then write (#116564)."""
    assert _call(store, action="add", target="memory", content="z" * 200)["success"] is True

    result = _call(store, action="add", target="memory", doc="fresh", content="x")

    assert result["success"] is False
    assert "free space FIRST" in result["error"]
    assert "@doc:fresh" in result["error"]


def test_anchor_at_cap_with_orphan_present(store, tmp_path):
    """Anchor at its char cap while an unlinked chain-doc file is on disk: every write path has
    to stay distinct - the anchor write is refused on capacity without blaming the orphan (the
    orphan frees no anchor budget), the orphan write names both ways out, and a fresh doc at the
    cap names the order: free space first, then link, then write. Leading with the anchor read
    pins the information gap the fix closes - on the pre-fix base this is the first assertion to
    fail (KeyError: 'orphan_docs') (#116564)."""
    _call(store, action="add", target="memory", content=ANCHOR_REF)
    _call(store, action="add", target="memory", doc=DOC, content="First note")
    assert _call(store, action="remove", target="memory", old_text="@doc:")["success"] is True
    assert _call(store, action="add", target="memory", content="z" * 200)["success"] is True

    anchor = _call(store, action="read", target="memory")
    assert anchor["orphan_docs"] == [{"name": DOC, "entries": 1, "chars": len("First note")}]
    assert anchor["chain_docs"] == []
    assert _call(store, action="read", target="memory", doc=DOC)["entries"] == ["First note"]

    result = _call(store, action="add", target="memory", content="y" * 400)
    assert result["success"] is False
    assert "would exceed the limit" in result["error"]
    assert "orphan" not in result["error"]

    result = _call(store, action="add", target="memory", doc=DOC, content="more")
    assert result["success"] is False
    assert str(tmp_path / "docs" / f"{DOC}.md") in result["error"]
    assert "Re-link" in result["error"]
    assert "delete the file" in result["error"]

    result = _call(store, action="add", target="memory", doc="fresh", content="x")
    assert result["success"] is False
    assert "free space FIRST" in result["error"]
    assert "@doc:fresh" in result["error"]


def test_linked_doc_is_not_reported_as_orphan(store):
    _call(store, action="add", target="memory", content=ANCHOR_REF)
    _call(store, action="add", target="memory", doc=DOC, content="First note")

    anchor = _call(store, action="read", target="memory")

    assert "orphan_docs" not in anchor
    assert anchor["chain_docs"][0]["name"] == DOC
