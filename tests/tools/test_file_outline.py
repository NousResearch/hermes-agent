"""Synthetic documents exercised through real registry and local backend I/O."""

import json

import pytest

from tools import file_outline, file_state, file_tools
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


@pytest.fixture
def document(tmp_path, monkeypatch):
    operations = ShellFileOperations(LocalEnvironment(str(tmp_path)))
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _: operations)
    return tmp_path / "spec.md"


def call(path, task="outline-test", **arguments):
    return json.loads(registry.dispatch("read_file", {"path": str(path), **arguments}, task_id=task))


def pages(path, **arguments):
    cursor = None
    for _ in range(100):
        page = call(path, mode="outline", cursor=cursor, **arguments)
        assert "error" not in page, page
        yield page
        if page["scan_complete"]:
            return
        cursor = page["next_cursor"]
    pytest.fail("continuation did not terminate")


def test_structure_then_targeted_body_read(document):
    document.write_bytes(b"\xef\xbb\xbf# Plan\r\nIntro\r\nAcceptance\r\n---\r\nMust pass.\r\n## Acceptance\r\n")
    page = call(document, mode="outline")
    assert page["outline"] == [
        {"line": 1, "level": 1, "heading": "Plan"},
        {"line": 3, "level": 2, "heading": "Acceptance"},
        {"line": 6, "level": 2, "heading": "Acceptance"}]
    assert page["body_read"] is False
    body = call(document, offset=page["outline"][1]["line"], limit=3)
    assert "Must pass." in body["content"]


@pytest.mark.parametrize("fence", ["`", "~"])
def test_chunk_boundaries_keep_fences_and_utf8(document, monkeypatch, fence):
    monkeypatch.setattr(file_outline, "WINDOW_BYTES", 32)
    document.write_bytes((f"{fence * 4}\r\n{fence * 3}\r\n# Hidden\r\n{fence * 4}\r\n"
                          "标题\r\n===\r\n###### Last ###").encode())
    result = list(pages(document))
    assert [item for p in result for item in p["outline"]] == [
        {"line": 5, "level": 1, "heading": "标题"},
        {"line": 7, "level": 6, "heading": "Last"}]


def test_empty_pages_continue_and_totals_are_not_premature(document, monkeypatch):
    monkeypatch.setattr(file_outline, "WINDOW_BYTES", 32)
    monkeypatch.setattr(file_outline, "CALL_BYTES", 32)
    document.write_text("body\n" * 30 + "# End\n", encoding="utf-8")
    result = list(pages(document))
    assert len(result) > 4 and all("_warning" not in p for p in result)
    assert not result[0]["outline"] and not result[0]["scan_complete"]
    assert all("total_headings" not in p for p in result[:-1])
    assert result[-1]["total_headings"] == 1
    previous = 0
    for page in result:
        assert 0 < page["scanned_bytes"] - previous <= 32
        previous = page["scanned_bytes"]


def test_output_pages_retain_duplicate_positions(document):
    document.write_text("# Repeated\n" * 505, encoding="utf-8")
    result = list(pages(document))
    assert len(result[0]["outline"]) == 500
    assert [entry["line"] for p in result for entry in p["outline"]] == list(range(1, 506))


@pytest.mark.parametrize("content", ["", "Only body\n", "\n\n"])
def test_no_headings_complete_without_cursor(document, content):
    document.write_text(content, encoding="utf-8")
    result = call(document, mode="outline")
    assert result["outline"] == [] and result["scan_complete"]
    assert "next_cursor" not in result


def test_cursor_cannot_mix_tasks_paths_or_versions(document):
    document.write_text("# First\n# Second\n", encoding="utf-8")
    cursor = call(document, mode="outline", limit=1)["next_cursor"]
    assert "cursor" in call(document, task="another", mode="outline", cursor=cursor)["error"]
    other = document.with_name("other.md")
    assert "cursor" in call(other, mode="outline", cursor=cursor)["error"]
    document.write_text("# Replaced\n", encoding="utf-8")
    assert "changed" in call(document, mode="outline", cursor=cursor)["error"]


def test_cursor_expires_and_retries_are_idempotent(document, monkeypatch):
    document.write_text("# One\n# Two\n# Three\n", encoding="utf-8")
    cursor = call(document, mode="outline", limit=1)["next_cursor"]
    a = call(document, mode="outline", limit=1, cursor=cursor)
    b = call(document, mode="outline", limit=1, cursor=cursor)
    assert a["outline"] == b["outline"]
    now = file_outline.time.monotonic()
    monkeypatch.setattr(file_outline.time, "monotonic", lambda: now + 601)
    assert "expired" in call(document, mode="outline", cursor=cursor)["error"]


def test_outline_does_not_satisfy_or_refresh_body_read(document):
    document.write_text("# Initial\nPrivate body\n", encoding="utf-8")
    identity = str(document.resolve())
    file_state.record_read("outline-test", identity, partial=True)
    call(document, mode="outline")
    assert "partial" in file_state.check_stale("outline-test", identity).lower()
    file_state.record_read("outline-test", identity)
    document.write_text("# Changed heading\nChanged body\n", encoding="utf-8")
    stale = file_state.check_stale("outline-test", identity)
    assert stale
    call(document, mode="outline")
    assert file_state.check_stale("outline-test", identity) == stale


def test_titles_use_existing_redaction_before_truncation(document):
    synthetic = "ghp_" + "a" * 36
    document.write_text("# " + "x" * 140 + " " + synthetic + "\n", encoding="utf-8")
    heading = call(document, mode="outline")["outline"][0]
    assert "ghp_aaaa" not in heading["heading"]
    assert heading["heading_truncated"] and len(heading["heading"]) <= 150


def test_nonmarkdown_binary_and_guarded_files(document):
    result = call(document.with_suffix(".txt"), mode="outline")
    assert "Markdown only" in result["note"]
    document.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    assert "Binary" in call(document, mode="outline")["error"]
    blocked = document.with_name(".env")
    blocked.write_text("# Synthetic blocked heading\n", encoding="utf-8")
    assert "Access denied" in call(blocked, mode="outline")["error"]


def test_overlong_line_fails_explicitly(document, monkeypatch):
    monkeypatch.setattr(file_outline, "WINDOW_BYTES", 1024)
    document.write_text("# ok\n# " + "x" * 2048 + "\n", encoding="utf-8")
    assert "ordinary offset/limit" in call(document, mode="outline")["error"]


def test_encoded_output_budget_has_lossless_continuation(document, monkeypatch):
    monkeypatch.setattr(file_outline, "PAGE_CHARS", 180)
    document.write_text(("# " + 'a"\\' * 20 + "\n") * 8, encoding="utf-8")
    result = list(pages(document))
    assert len(result) > 1
    assert [e["line"] for p in result for e in p["outline"]] == list(range(1, 9))


def test_initial_heading_offset_and_literal_source_lines(document):
    document.write_text("text\u2028same physical line\n# ###\n## Next\n", encoding="utf-8")
    assert call(document, mode="outline", offset=2)["outline"] == [
        {"line": 3, "level": 2, "heading": "Next"}]


def test_invalid_options_are_errors(document):
    assert "mode" in call(document, mode="other")["error"]
    assert "cursor" in call(document, cursor="not-for-body")["error"]


def test_one_call_spans_many_backend_windows(document, monkeypatch):
    monkeypatch.setattr(file_outline, "WINDOW_BYTES", 32)
    document.write_text("".join(f"# H{i}\nbody text line\n" for i in range(40)), encoding="utf-8")
    page = call(document, mode="outline")
    assert page["scan_complete"] and "next_cursor" not in page
    assert [e["heading"] for e in page["outline"]] == [f"H{i}" for i in range(40)]


def test_repeated_first_page_of_unchanged_file_is_deduplicated(document):
    document.write_text("# Same\n", encoding="utf-8")
    assert call(document, mode="outline")["outline"]
    assert call(document, mode="outline")["dedup"] is True
    document.write_text("# Changed\n", encoding="utf-8")
    assert call(document, mode="outline")["outline"][0]["heading"] == "Changed"
