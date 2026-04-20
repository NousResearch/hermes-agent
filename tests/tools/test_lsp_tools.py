import io
import json

from model_tools import get_tool_definitions
from toolsets import resolve_toolset
from tools.lsp_tools import (
    LSP_TOOLS,
    LSP_DIAGNOSTICS_SCHEMA,
    LSP_DOCUMENT_SYMBOLS_SCHEMA,
    LSP_DEFINITION_SCHEMA,
    LSP_PREPARE_RENAME_SCHEMA,
    LSP_REFERENCES_SCHEMA,
    LSP_RENAME_SCHEMA,
    get_host_lsp_capabilities,
    lsp_definition_tool,
    lsp_diagnostics_tool,
    lsp_document_symbols_tool,
    lsp_prepare_rename_tool,
    lsp_references_tool,
    lsp_rename_tool,
)


class TestLspRepoCodeKnowledgeFraming:
    def test_repo_code_knowledge_toolset_treats_lsp_primitives_as_canonical(self):
        repo_code_tools = set(resolve_toolset("repo-code-knowledge"))
        assert {
            "lsp_document_symbols",
            "lsp_definition",
            "lsp_diagnostics",
            "lsp_prepare_rename",
            "lsp_references",
            "lsp_rename",
        } <= repo_code_tools

    def test_lsp_schemas_mention_repo_code_knowledge_grounding(self):
        assert "repo/code knowledge primitive" in LSP_DOCUMENT_SYMBOLS_SCHEMA["description"]
        assert "semantic local source grounding" in LSP_DOCUMENT_SYMBOLS_SCHEMA["description"]
        assert "repo/code knowledge primitive" in LSP_DEFINITION_SCHEMA["description"]
        assert "repo/code knowledge primitive" in LSP_DIAGNOSTICS_SCHEMA["description"]
        assert "repo/code knowledge primitive" in LSP_PREPARE_RENAME_SCHEMA["description"]
        assert "repo/code knowledge primitive" in LSP_REFERENCES_SCHEMA["description"]
        assert "repo/code knowledge primitive" in LSP_RENAME_SCHEMA["description"]


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


class TestLspToolRegistration:
    def test_exports_expected_tools_and_schemas(self):
        names = {tool["name"] for tool in LSP_TOOLS}
        assert names == {
            "lsp_document_symbols",
            "lsp_definition",
            "lsp_diagnostics",
            "lsp_prepare_rename",
            "lsp_references",
            "lsp_rename",
        }

        for schema in [
            LSP_DOCUMENT_SYMBOLS_SCHEMA,
            LSP_DEFINITION_SCHEMA,
            LSP_DIAGNOSTICS_SCHEMA,
            LSP_PREPARE_RENAME_SCHEMA,
            LSP_REFERENCES_SCHEMA,
            LSP_RENAME_SCHEMA,
        ]:
            assert "name" in schema
            assert "description" in schema
            assert "properties" in schema["parameters"]

    def test_code_intel_toolset_lists_lsp_tools(self):
        code_intel_tools = set(resolve_toolset("code_intel"))
        assert {
            "lsp_document_symbols",
            "lsp_definition",
            "lsp_diagnostics",
            "lsp_prepare_rename",
            "lsp_references",
            "lsp_rename",
        } <= code_intel_tools

    def test_model_tool_definitions_expose_lsp_tools_when_server_exists(self, monkeypatch):
        monkeypatch.setattr("tools.lsp_tools.shutil.which", lambda name: "/usr/bin/fake-lsp")
        model_tool_names = {
            tool["function"]["name"]
            for tool in get_tool_definitions(enabled_toolsets=["code_intel"], quiet_mode=True)
        }
        assert {
            "lsp_document_symbols",
            "lsp_definition",
            "lsp_diagnostics",
            "lsp_prepare_rename",
            "lsp_references",
            "lsp_rename",
        } <= model_tool_names

    def test_model_tool_definitions_hide_lsp_tools_when_no_server_exists(self, monkeypatch):
        monkeypatch.setattr("tools.lsp_tools.shutil.which", lambda name: None)
        model_tool_names = {
            tool["function"]["name"]
            for tool in get_tool_definitions(enabled_toolsets=["code_intel"], quiet_mode=True)
        }
        assert {
            "lsp_document_symbols",
            "lsp_definition",
            "lsp_diagnostics",
            "lsp_prepare_rename",
            "lsp_references",
            "lsp_rename",
        }.isdisjoint(model_tool_names)

    def test_host_capability_helper_reports_language_specific_auto_detection(self, monkeypatch):
        available_binaries = {
            "pylsp": "/usr/bin/pylsp",
            "typescript-language-server": "/usr/bin/typescript-language-server",
        }
        monkeypatch.setattr("tools.lsp_tools.shutil.which", lambda name: available_binaries.get(name))

        capabilities = get_host_lsp_capabilities()

        assert capabilities["available"] is True
        assert capabilities["auto_detect_languages"] == [
            "javascript",
            "javascriptreact",
            "python",
            "typescript",
            "typescriptreact",
        ]
        assert capabilities["auto_detect_commands"]["python"] == ["pylsp"]
        assert capabilities["auto_detect_commands"]["typescript"] == ["typescript-language-server --stdio"]

    def test_model_tool_definitions_enrich_lsp_descriptions_with_host_languages(self, monkeypatch):
        available_binaries = {
            "pylsp": "/usr/bin/pylsp",
            "rust-analyzer": "/usr/bin/rust-analyzer",
        }
        monkeypatch.setattr("tools.lsp_tools.shutil.which", lambda name: available_binaries.get(name))

        tool_definitions = get_tool_definitions(enabled_toolsets=["code_intel"], quiet_mode=True)
        descriptions = {
            tool["function"]["name"]: tool["function"]["description"]
            for tool in tool_definitions
            if tool["function"]["name"] in {
                "lsp_document_symbols",
                "lsp_definition",
                "lsp_diagnostics",
                "lsp_prepare_rename",
                "lsp_references",
                "lsp_rename",
            }
        }

        assert descriptions
        for description in descriptions.values():
            assert "Auto-detect currently supports: python, rust." in description
            assert "server_command" in description


class TestLspHandlers:
    def test_document_symbols_returns_structured_error_when_no_server_is_available(self, monkeypatch, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text("def demo():\n    return 1\n", encoding="utf-8")
        monkeypatch.setattr("tools.lsp_tools.shutil.which", lambda name: None)

        result = json.loads(lsp_document_symbols_tool(path=str(source_path)))

        assert "error" in result
        assert "No LSP server found" in result["error"]
        assert result["language"] == "python"
        assert result["requested_method"] == "textDocument/documentSymbol"

    def test_document_symbols_round_trip_uses_explicit_server_command(self, monkeypatch, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text("class Example:\n    pass\n", encoding="utf-8")

        fake_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": [
                            {
                                "name": "Example",
                                "kind": 5,
                                "range": {
                                    "start": {"line": 0, "character": 0},
                                    "end": {"line": 1, "character": 8},
                                },
                                "selectionRange": {
                                    "start": {"line": 0, "character": 6},
                                    "end": {"line": 0, "character": 13},
                                },
                                "children": [],
                            }
                        ],
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 3, "result": None}),
            ]
        )

        def _fake_popen(command, **kwargs):
            fake_process.command = command
            return fake_process

        monkeypatch.setattr("tools.lsp_tools.subprocess.Popen", _fake_popen)

        result = json.loads(
            lsp_document_symbols_tool(
                path=str(source_path),
                server_command="fake-lsp --stdio",
            )
        )

        assert result["ok"] is True
        assert result["server_command"] == ["fake-lsp", "--stdio"]
        assert fake_process.command == ["fake-lsp", "--stdio"]
        assert result["symbols"][0]["name"] == "Example"
        sent_payload = fake_process.stdin.getvalue().decode("utf-8")
        assert "textDocument/documentSymbol" in sent_payload
        assert "textDocument/didOpen" in sent_payload

    def test_definition_and_diagnostics_round_trip(self, monkeypatch, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text("value = helper()\n", encoding="utf-8")

        definition_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": [
                            {
                                "uri": source_path.resolve().as_uri(),
                                "range": {
                                    "start": {"line": 0, "character": 0},
                                    "end": {"line": 0, "character": 5},
                                },
                            }
                        ],
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 3, "result": None}),
            ]
        )
        diagnostics_process = _FakeProcess(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {
                            "kind": "full",
                            "items": [
                                {
                                    "message": "Undefined name 'helper'",
                                    "severity": 1,
                                    "source": "fake-lsp",
                                    "range": {
                                        "start": {"line": 0, "character": 8},
                                        "end": {"line": 0, "character": 14},
                                    },
                                }
                            ],
                        },
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 3, "result": None}),
            ]
        )
        processes = [definition_process, diagnostics_process]

        def _fake_popen(command, **kwargs):
            return processes.pop(0)

        monkeypatch.setattr("tools.lsp_tools.subprocess.Popen", _fake_popen)

        definition_result = json.loads(
            lsp_definition_tool(
                path=str(source_path),
                line=0,
                character=8,
                server_command="fake-lsp --stdio",
            )
        )
        diagnostics_result = json.loads(
            lsp_diagnostics_tool(
                path=str(source_path),
                server_command="fake-lsp --stdio",
            )
        )

        assert definition_result["ok"] is True
        assert definition_result["definitions"][0]["path"] == str(source_path.resolve())
        assert diagnostics_result["ok"] is True
        assert diagnostics_result["diagnostics"][0]["message"] == "Undefined name 'helper'"
        assert diagnostics_result["diagnostics"][0]["severity"] == 1
