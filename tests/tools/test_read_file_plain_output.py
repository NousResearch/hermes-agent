"""File rendering must preserve literal text without changing read safeguards."""

import json

from tools import file_tools
from tools.registry import registry


def _call(path, task, **arguments):
    return json.loads(registry.dispatch(
        "read_file", {"path": str(path), **arguments}, task_id=task))


def test_plain_output_preserves_literal_lines_and_page_contract(tmp_path, monkeypatch):
    path = tmp_path / "literal.txt"
    path.write_text("alpha\n1|keep\n\n9|尾", encoding="utf-8")
    cases = [
        (1, 2000, 100_000, "alpha\n1|keep\n\n9|尾"),
        (2, 3, 100_000, "1|keep\n\n9|尾"),
        (1, 2000, 14, "alpha"),
        (1, 2000, 1, ""),
    ]
    for index, (offset, limit, budget, expected) in enumerate(cases):
        monkeypatch.setattr(file_tools, "_get_max_read_chars", lambda: budget)
        numbered_task, plain_task = f"numbered-{index}", f"plain-{index}"
        try:
            numbered = _call(path, numbered_task, offset=offset, limit=limit)
            plain = _call(path, plain_task, offset=offset, limit=limit, line_numbers=False)
            assert plain.pop("content") == expected
            assert numbered.pop("content").startswith(str(offset))
            assert plain == numbered  # same paging, truncation and safety metadata
        finally:
            file_tools.clear_file_ops_cache(numbered_task)
            file_tools.clear_file_ops_cache(plain_task)

    # The document-extraction path obeys the same rendering choice.
    monkeypatch.setattr(file_tools, "_get_max_read_chars", lambda: 100_000)
    notebook = tmp_path / "literal.ipynb"
    notebook.write_text(json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["1|keep\n", "\n", "9|尾"]},
    ]}), encoding="utf-8")
    try:
        plain = _call(notebook, "plain-document", line_numbers=False)
        assert plain.get("extracted_document"), plain
        assert plain["content"].splitlines()[1:] == ["1|keep", "", "9|尾"]
    finally:
        file_tools.clear_file_ops_cache("plain-document")


def test_output_format_keeps_dedup_write_and_denial_boundaries(tmp_path):
    path = tmp_path / "guarded.txt"
    path.write_text("1|keep", encoding="utf-8")
    task = "format-switch"
    try:
        assert _call(path, task)["content"] == "1|1|keep"
        assert _call(path, task, line_numbers=False)["content"] == "1|keep"
        for arguments in ({"line_numbers": False}, {}):
            assert _call(path, task, **arguments)["content_returned"] is False
        blocked = _call(path, task, line_numbers=False)
        assert "BLOCKED" in blocked["error"]
        written = json.loads(registry.dispatch("write_file", {
            "path": str(path), "content": "verified replacement",
        }, task_id=task))
        assert "error" not in written, written
        assert path.read_text(encoding="utf-8") == "verified replacement"
    finally:
        file_tools.clear_file_ops_cache(task)

    denied = tmp_path / ".env"
    denied.write_text("SYNTHETIC_ONLY=value", encoding="utf-8")
    for value in (False, True):
        result = _call(denied, f"deny-{value}", line_numbers=value)
        assert result.get("error") and "content" not in result
    for index, value in enumerate(("false", 0, None, [])):
        result = _call(path, f"bad-format-{index}", line_numbers=value)
        assert "line_numbers" in result.get("error", "") and "content" not in result

    secret = "AKIA" + "A" * 16  # synthetic, never a usable credential
    path.write_text("1|" + secret, encoding="utf-8")
    try:
        redacted = _call(path, "redacted-plain", line_numbers=False)
        assert redacted["content"].startswith("1|")
        assert secret not in redacted["content"]
        refused = json.loads(registry.dispatch("write_file", {
            "path": str(path), "content": "replacement",
        }, task_id="redacted-plain"))
        assert refused.get("stale_write_blocked"), refused
        assert path.read_text(encoding="utf-8") == "1|" + secret
    finally:
        file_tools.clear_file_ops_cache("redacted-plain")
