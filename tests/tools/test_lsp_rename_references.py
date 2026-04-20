import io
import json

from tools.lsp_tools import lsp_prepare_rename_tool, lsp_references_tool, lsp_rename_tool


def _frame(payload):
    body = json.dumps(payload).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


class _FakeProcess:
    def __init__(self, frames):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(b"".join(frames))
        self.stderr = io.BytesIO()
        self.returncode = None
        self.command = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


class TestLspRenameAndReferences:
    def test_prepare_rename_returns_structured_range_and_placeholder(self, monkeypatch, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text("value = old_name\n", encoding="utf-8")

        fake_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {
                            "range": {
                                "start": {"line": 0, "character": 8},
                                "end": {"line": 0, "character": 16},
                            },
                            "placeholder": "old_name",
                        },
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 3, "result": None}),
            ]
        )

        monkeypatch.setattr("tools.lsp_tools.subprocess.Popen", lambda command, **kwargs: fake_process)

        result = json.loads(
            lsp_prepare_rename_tool(
                path=str(source_path),
                line=0,
                character=10,
                server_command="fake-lsp --stdio",
            )
        )

        assert result["ok"] is True
        assert result["can_rename"] is True
        assert result["prepare_rename"]["placeholder"] == "old_name"
        assert result["prepare_rename"]["range"]["start"] == {"line": 0, "character": 8}

    def test_references_and_rename_return_realistic_workspace_edit_preview(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        main_path = workspace / "main.py"
        helper_path = workspace / "helper.py"
        helper_path.write_text(
            "def old_name(value):\n    return value + 1\n",
            encoding="utf-8",
        )
        main_path.write_text(
            "from helper import old_name\n\nresult = old_name(1)\n",
            encoding="utf-8",
        )

        reference_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": [
                            {
                                "uri": helper_path.resolve().as_uri(),
                                "range": {
                                    "start": {"line": 0, "character": 4},
                                    "end": {"line": 0, "character": 12},
                                },
                            },
                            {
                                "uri": main_path.resolve().as_uri(),
                                "range": {
                                    "start": {"line": 0, "character": 19},
                                    "end": {"line": 0, "character": 27},
                                },
                            },
                            {
                                "uri": main_path.resolve().as_uri(),
                                "range": {
                                    "start": {"line": 2, "character": 9},
                                    "end": {"line": 2, "character": 17},
                                },
                            },
                        ],
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 3, "result": None}),
            ]
        )
        rename_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {
                            "range": {
                                "start": {"line": 0, "character": 4},
                                "end": {"line": 0, "character": 12},
                            },
                            "placeholder": "old_name",
                        },
                    }
                ),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "result": {
                            "changes": {
                                helper_path.resolve().as_uri(): [
                                    {
                                        "range": {
                                            "start": {"line": 0, "character": 4},
                                            "end": {"line": 0, "character": 12},
                                        },
                                        "newText": "new_name",
                                    }
                                ],
                                main_path.resolve().as_uri(): [
                                    {
                                        "range": {
                                            "start": {"line": 0, "character": 19},
                                            "end": {"line": 0, "character": 27},
                                        },
                                        "newText": "new_name",
                                    },
                                    {
                                        "range": {
                                            "start": {"line": 2, "character": 9},
                                            "end": {"line": 2, "character": 17},
                                        },
                                        "newText": "new_name",
                                    },
                                ],
                            }
                        },
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 4, "result": None}),
            ]
        )
        processes = [reference_process, rename_process]

        monkeypatch.setattr("tools.lsp_tools.subprocess.Popen", lambda command, **kwargs: processes.pop(0))

        references_result = json.loads(
            lsp_references_tool(
                path=str(helper_path),
                line=0,
                character=6,
                server_command="fake-lsp --stdio",
                workspace_root=str(workspace),
            )
        )
        rename_result = json.loads(
            lsp_rename_tool(
                path=str(helper_path),
                line=0,
                character=6,
                new_name="new_name",
                server_command="fake-lsp --stdio",
                workspace_root=str(workspace),
            )
        )

        assert references_result["ok"] is True
        assert references_result["reference_count"] == 3
        assert {entry["path"] for entry in references_result["references"]} == {
            str(helper_path.resolve()),
            str(main_path.resolve()),
        }

        assert rename_result["ok"] is True
        assert rename_result["prepare_rename"]["placeholder"] == "old_name"
        assert rename_result["workspace_edit"]["edit_count"] == 3
        assert rename_result["workspace_edit"]["touched_paths"] == [
            str(helper_path.resolve()),
            str(main_path.resolve()),
        ]
        assert rename_result["workspace_edit"]["preview_by_path"][str(helper_path.resolve())] == (
            "def new_name(value):\n    return value + 1\n"
        )
        assert rename_result["workspace_edit"]["preview_by_path"][str(main_path.resolve())] == (
            "from helper import new_name\n\nresult = new_name(1)\n"
        )

    def test_rename_fails_safely_when_prepare_rename_rejects_symbol(self, monkeypatch, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text("value = 1\n", encoding="utf-8")

        fake_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame({"jsonrpc": "2.0", "id": 2, "result": None}),
                _frame({"jsonrpc": "2.0", "id": 3, "result": None}),
            ]
        )

        monkeypatch.setattr("tools.lsp_tools.subprocess.Popen", lambda command, **kwargs: fake_process)

        result = json.loads(
            lsp_rename_tool(
                path=str(source_path),
                line=0,
                character=0,
                new_name="renamed",
                server_command="fake-lsp --stdio",
            )
        )

        assert result["error"] == "LSP server reported that the selected symbol cannot be renamed"
        assert result["failure_kind"] == "prepare_rename_rejected"
        assert result["prepare_rename"] == {"can_rename": False}

    def test_prepare_rename_classifies_method_not_found_as_unsupported_capability(self, monkeypatch, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text("value = old_name\n", encoding="utf-8")

        fake_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "error": {"code": -32601, "message": "Method not found: textDocument/prepareRename"},
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 3, "result": None}),
            ]
        )

        monkeypatch.setattr("tools.lsp_tools.subprocess.Popen", lambda command, **kwargs: fake_process)

        result = json.loads(
            lsp_prepare_rename_tool(
                path=str(source_path),
                line=0,
                character=10,
                server_command="fake-lsp --stdio",
            )
        )

        assert result["failure_kind"] == "unsupported_capability"
        assert result["requested_method"] == "textDocument/prepareRename"
        assert result["lsp_error_code"] == -32601
        assert result["lsp_error_message"] == "Method not found: textDocument/prepareRename"

    def test_references_classifies_method_not_found_as_unsupported_capability(self, monkeypatch, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text("value = old_name\n", encoding="utf-8")

        fake_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "error": {"code": -32601, "message": "Method not found: textDocument/references"},
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 3, "result": None}),
            ]
        )

        monkeypatch.setattr("tools.lsp_tools.subprocess.Popen", lambda command, **kwargs: fake_process)

        result = json.loads(
            lsp_references_tool(
                path=str(source_path),
                line=0,
                character=10,
                server_command="fake-lsp --stdio",
            )
        )

        assert result["failure_kind"] == "unsupported_capability"
        assert result["requested_method"] == "textDocument/references"
        assert result["lsp_error_code"] == -32601
        assert result["lsp_error_message"] == "Method not found: textDocument/references"

    def test_rename_classifies_protocol_failure_after_prepare_rename_as_unsupported_capability(self, monkeypatch, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text("value = old_name\n", encoding="utf-8")

        fake_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {
                            "range": {
                                "start": {"line": 0, "character": 8},
                                "end": {"line": 0, "character": 16},
                            },
                            "placeholder": "old_name",
                        },
                    }
                ),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "error": {"code": -32601, "message": "Method not found: textDocument/rename"},
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 4, "result": None}),
            ]
        )

        monkeypatch.setattr("tools.lsp_tools.subprocess.Popen", lambda command, **kwargs: fake_process)

        result = json.loads(
            lsp_rename_tool(
                path=str(source_path),
                line=0,
                character=10,
                new_name="new_name",
                server_command="fake-lsp --stdio",
            )
        )

        assert result["failure_kind"] == "unsupported_capability"
        assert result["requested_method"] == "textDocument/rename"
        assert result["lsp_error_code"] == -32601
        assert result["lsp_error_message"] == "Method not found: textDocument/rename"

    def test_references_report_missing_server_for_supported_language(self, monkeypatch, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text("value = old_name\n", encoding="utf-8")
        monkeypatch.setattr("tools.lsp_tools.shutil.which", lambda name: None)

        result = json.loads(lsp_references_tool(path=str(source_path), line=0, character=8))

        assert result["failure_kind"] == "missing_server"
        assert result["language"] == "python"
        assert "candidate_commands" in result

    def test_rename_reports_unsupported_language_when_auto_detect_has_no_mapping(self, tmp_path):
        source_path = tmp_path / "sample.customlang"
        source_path.write_text("symbol\n", encoding="utf-8")

        result = json.loads(lsp_rename_tool(path=str(source_path), line=0, character=0, new_name="other"))

        assert result["failure_kind"] == "unsupported_language"
        assert result["language"] == "plaintext"
        assert result["candidate_commands"] == []
