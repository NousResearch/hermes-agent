"""Repeat-pattern collapse for todo content (#107407).

``MAX_TODO_CONTENT_CHARS = 4000`` (GHSA-5g4g-6jrg-mw3g) does not stop
short filler such as ``\"1 选 \" * 57`` (~200 chars). These pin collapse
of 5+ consecutive repeated units on every write/merge/restore path that
already goes through ``TodoStore._cap_content``.
"""

from tools.todo_tool import TodoStore


class TestRepeatCollapse:
    def test_reported_garbage_is_collapsed(self):
        store = TodoStore()
        garbage = ("1 选 " * 57).strip()
        store.write([{"id": "1", "content": garbage, "status": "pending"}])
        content = store.read()[0]["content"]
        assert "[repeated]" in content
        assert content.count("1 选") <= 3
        assert len(content) < 80

    def test_single_char_run_is_collapsed(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "x" * 57, "status": "pending"}])
        content = store.read()[0]["content"]
        assert "[repeated]" in content
        assert content.count("x") <= 3

    def test_normal_unique_text_unchanged(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "write the report", "status": "pending"}])
        assert store.read()[0]["content"] == "write the report"

    def test_four_repeats_below_threshold_unchanged(self):
        store = TodoStore()
        four = ("1 选 " * 4).strip()
        store.write([{"id": "1", "content": four, "status": "pending"}])
        assert store.read()[0]["content"] == four


class TestFailOpen:
    def test_empty_content_keeps_placeholder(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "   ", "status": "pending"}])
        assert store.read()[0]["content"] == "(no description)"

    def test_nonconsecutive_similar_phrases_unchanged(self):
        store = TodoStore()
        text = "step 1 do A; step 2 do B"
        store.write([{"id": "1", "content": text, "status": "pending"}])
        assert store.read()[0]["content"] == text

    def test_numbered_list_and_path_unchanged(self):
        store = TodoStore()
        text = "1. edit src/app.py\n2. run tests\n3. open /tmp/out.log"
        store.write([{"id": "1", "content": text, "status": "pending"}])
        assert store.read()[0]["content"] == text

    def test_merge_omitting_content_leaves_existing(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "write the report", "status": "pending"}])
        store.write([{"id": "1", "status": "in_progress"}], merge=True)
        item = store.read()[0]
        assert item["content"] == "write the report"
        assert item["status"] == "in_progress"

    def test_invalid_non_dict_keeps_placeholder(self):
        store = TodoStore()
        store.write(["not-a-dict"])
        assert store.read()[0]["content"] == "(invalid item)"

    def test_unique_oversized_content_still_truncated(self):
        from tools.todo_tool import MAX_TODO_CONTENT_CHARS

        store = TodoStore()
        unique = "-".join(str(i) for i in range(3000))
        store.write([{"id": "1", "content": unique, "status": "pending"}])
        content = store.read()[0]["content"]
        assert len(content) <= MAX_TODO_CONTENT_CHARS
        assert content.endswith("… [truncated]")
        assert "[repeated]" not in content
