"""Regression coverage for repeated todo filler (#107407)."""

from tools.todo_tool import MAX_TODO_CONTENT_CHARS, TodoStore


def test_repeated_multibyte_filler_is_collapsed():
    store = TodoStore()
    store.write([{"id": "1", "content": ("1 选 " * 57).strip(), "status": "pending"}])
    content = store.read()[0]["content"]
    assert "[repeated]" in content
    assert content.count("1 选") <= 3
    assert len(content) < 80


def test_short_runs_and_normal_text_are_preserved():
    store = TodoStore()
    normal = "write the report"
    four = ("1 选 " * 4).strip()
    store.write([{"id": "1", "content": normal, "status": "pending"}])
    assert store.read()[0]["content"] == normal
    store.write([{"id": "2", "content": four, "status": "pending"}])
    assert store.read()[0]["content"] == four


def test_nonshrinking_short_runs_do_not_expand_content():
    store = TodoStore()
    short_run = "x" * 5
    store.write([{"id": "1", "content": short_run, "status": "pending"}])
    assert store.read()[0]["content"] == short_run

    capped = short_run + "-".join(str(i) for i in range(3000))
    store.write([{"id": "2", "content": capped, "status": "pending"}])
    content = store.read()[0]["content"]
    assert len(content) <= MAX_TODO_CONTENT_CHARS
    assert content.endswith("… [truncated]")


def test_merge_and_restore_apply_sanitization_without_breaking_truncation():
    store = TodoStore()
    store.write([{"id": "1", "content": "original", "status": "pending"}])
    store.write([{"id": "1", "content": "x" * 57, "status": "pending"}], merge=True)
    assert "[repeated]" in store.read()[0]["content"]
    store.restore([{"id": "2", "content": "y" * 57, "status": "pending"}])
    assert "[repeated]" in store.read()[0]["content"]
    unique = "-".join(str(i) for i in range(3000))
    store.write([{"id": "3", "content": unique, "status": "pending"}])
    content = store.read()[0]["content"]
    assert len(content) <= MAX_TODO_CONTENT_CHARS
    assert content.endswith("… [truncated]")


def test_truncated_repeated_content_stays_capped_and_keeps_both_markers():
    store = TodoStore()
    store.write([{"id": "1", "content": "abcdef" * 1000, "status": "pending"}])
    content = store.read()[0]["content"]
    assert len(content) <= MAX_TODO_CONTENT_CHARS
    assert "[repeated]" in content
    assert content.endswith("… [truncated]")
