"""Batch entrypoints preserve each single-item tool's guards and result contract."""

import json
from unittest.mock import patch

from tools.file_tools import (
    READ_FILE_SCHEMA, WRITE_FILE_SCHEMA, _handle_read_file, _handle_write_file,
)
from tools.skills_tool import SKILL_VIEW_SCHEMA, _skill_view_with_bump


def _parsed(raw):
    return json.loads(raw)


def test_skill_view_batch_loads_main_and_reference_and_keeps_partial_errors(tmp_path):
    skill = tmp_path / "demo"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: test\n---\n# Demo\n")
    (skill / "references" / "api.md").write_text("API details\n")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        payload = _parsed(_skill_view_with_bump({"requests": [
            {"name": "demo"},
            {"name": "demo", "file_path": "references/api.md"},
            {"name": "demo", "file_path": "../../outside.md"},
        ]}, task_id="batch-skill-view"))
    assert payload["success"] is False
    assert payload["completed"] == 2
    assert [r["request"]["name"] for r in payload["results"]] == ["demo"] * 3
    assert "# Demo" in payload["results"][0]["result"]["content"]
    assert "API details" in payload["results"][1]["result"]["content"]
    assert payload["results"][2]["result"]["success"] is False


def test_skill_view_batch_rejects_malformed_and_mixed_modes_before_loading():
    assert "error" in _parsed(_skill_view_with_bump({"requests": []}))
    assert "error" in _parsed(_skill_view_with_bump({"name": "demo", "requests": [{"name": "demo"}]}))
    assert "error" in _parsed(_skill_view_with_bump({"requests": [{"name": "demo"}, {"name": 42}]}))
    assert "error" in _parsed(_skill_view_with_bump({"requests": [{"name": "demo"}] * 5}))
    assert "duplicate" in _parsed(_skill_view_with_bump({"requests": [
        {"name": "demo"}, {"name": "demo"},
    ]}))["error"]
    assert "requests" in SKILL_VIEW_SCHEMA["parameters"]["properties"]


def test_read_file_batch_preserves_result_and_per_item_guards(tmp_path):
    first = tmp_path / "first.txt"
    first.write_text("first\n")
    second = tmp_path / "second.txt"
    second.write_text("second\n")
    payload = _parsed(_handle_read_file({"requests": [
        {"path": str(first)}, {"path": "/dev/zero"}, {"path": str(second)},
    ]}, task_id="batch-file-read"))
    assert payload["success"] is False
    assert payload["completed"] == 2
    assert "first" in payload["results"][0]["result"]["content"]
    assert "device" in payload["results"][1]["result"]["error"].lower()
    assert "second" in payload["results"][2]["result"]["content"]
    assert "requests" in READ_FILE_SCHEMA["parameters"]["properties"]


def test_read_file_batch_budget_has_continuation_and_does_not_dedup_single_read(tmp_path):
    big = tmp_path / "large.txt"
    big.write_text("line\n" * 7000)
    batch = _parsed(_handle_read_file({"requests": [{"path": str(big)}]}, task_id="batch-large-read"))
    page = batch["results"][0]["result"]
    assert page["truncated_by"] == "bytes"
    assert page["next_offset"] > 1
    assert len(page["content"]) <= 16_000
    single = _parsed(_handle_read_file({"path": str(big)}, task_id="batch-large-read"))
    assert "content" in single
    assert len(single["content"]) > len(page["content"])


def test_write_file_batch_validates_all_inputs_before_mutation(tmp_path):
    candidate = tmp_path / "not-created.txt"
    invalid = _parsed(_handle_write_file({"files": [
        {"path": str(candidate), "content": "hello"},
        {"path": str(tmp_path / "bad.txt")},
    ]}, task_id="batch-file-invalid"))
    assert "error" in invalid
    assert not candidate.exists()
    duplicate = _parsed(_handle_write_file({"files": [
        {"path": str(candidate), "content": "one"},
        {"path": str(candidate), "content": "two"},
    ]}, task_id="batch-file-duplicate"))
    assert "error" in duplicate
    assert not candidate.exists()
    assert "files" in WRITE_FILE_SCHEMA["parameters"]["properties"]


def test_write_file_batch_results_are_verified_individually(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    payload = _parsed(_handle_write_file({"files": [
        {"path": str(first), "content": "first\n"},
        {"path": str(second), "content": "second\n"},
    ]}, task_id="batch-file-writes"))
    assert payload["success"] is True
    assert payload["completed"] == 2
    assert all(r["result"]["verified"] is True for r in payload["results"])
    assert first.read_text() == "first\n"
    assert second.read_text() == "second\n"


def test_write_file_batch_preflights_protected_and_stale_paths(tmp_path):
    candidate = tmp_path / "candidate.txt"
    blocked = _parsed(_handle_write_file({"files": [
        {"path": str(candidate), "content": "first"},
        {"path": "/etc/hosts", "content": "never"},
    ]}, task_id="batch-preflight-protected"))
    assert "error" in blocked
    assert not candidate.exists()

    existing = tmp_path / "existing.txt"
    existing.write_text("original")
    stale = _parsed(_handle_write_file({"files": [
        {"path": str(candidate), "content": "first"},
        {"path": str(existing), "content": "replacement"},
    ]}, task_id="batch-preflight-stale"))
    assert stale.get("error") or stale.get("success") is False
    assert not candidate.exists()
    assert existing.read_text() == "original"


def test_write_file_batch_reports_runtime_partial_failure(tmp_path):
    from tools.file_tools import write_file_tool

    first, second = tmp_path / "first.txt", tmp_path / "second.txt"

    def write_or_fail(*, path, **kw):
        if path == str(second):
            return json.dumps({"error": "simulated failure"})
        return write_file_tool(path=path, **kw)

    with patch("tools.file_tools.write_file_tool", side_effect=write_or_fail):
        result = _parsed(_handle_write_file({"files": [
            {"path": str(first), "content": "first"},
            {"path": str(second), "content": "second"},
        ]}, task_id="batch-runtime-failure"))
    assert result["success"] is False
    assert result["completed"] == 1
    assert result["results"][1]["result"]["error"] == "simulated failure"
    assert first.read_text() == "first"
    assert not second.exists()


def test_write_file_batch_rejects_symlink_alias_before_mutation(tmp_path):
    original = tmp_path / "original.txt"
    alias = tmp_path / "alias.txt"
    alias.symlink_to(original)
    result = _parsed(_handle_write_file({"files": [
        {"path": str(original), "content": "first"},
        {"path": str(alias), "content": "second"},
    ]}, task_id="batch-alias"))
    assert "error" in result
    assert not original.exists()


def test_write_file_batch_uses_single_write_metrics_and_lsp_owner(tmp_path):
    from tools.file_tools import _owned

    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    with (patch("hermes_cli.observability.shared_metrics_harness.record_file_edit",
                side_effect=lambda _tool, _kind, thunk: thunk()) as metrics,
          patch("tools.file_tools._owned", wraps=_owned) as owned):
        result = _parsed(_handle_write_file({"files": [
            {"path": str(first), "content": "first"},
            {"path": str(second), "content": "second"},
        ]}, task_id="batch-ownership", session_id="session-ownership"))
    assert result["success"] is True
    assert metrics.call_count == 2
    assert owned.call_count == 2
    assert all(call.args[0] == "session-ownership" for call in owned.call_args_list)


def test_read_file_batch_keeps_dynamic_ocr_description():
    from tools.file_tools import _read_file_schema_overrides

    with patch("tools.read_extract.hosted_ocr_available", return_value=True):
        assert "PDF (scanned or text)" in _read_file_schema_overrides()["description"]
