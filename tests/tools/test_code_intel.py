"""Tests for tools/code_intel.py."""

from tools.code_intel import (
    code_definition,
    code_intel_capabilities,
    code_references,
)
from tools.registry import registry


class TestCodeIntelCapabilities:
    def test_python_capabilities_report_supported_operations(self, tmp_path):
        path = tmp_path / "sample.py"
        path.write_text("def demo(value):\n    total = value + 1\n    return total\n", encoding="utf-8")

        result = code_intel_capabilities(path=str(path))

        assert result["ok"] is True
        assert result["language"] == "python"
        assert result["backend"] == "python_ast"
        assert result["external_server_required"] is False
        assert result["operations"] == {
            "rename": True,
            "references": True,
            "definition": True,
            "symbols": True,
            "diagnostics": True,
        }

    def test_unsupported_language_returns_structured_refusal(self, tmp_path):
        path = tmp_path / "sample.js"
        path.write_text("const value = 1;\n", encoding="utf-8")

        result = code_intel_capabilities(path=str(path))

        assert result["ok"] is False
        assert result["code"] == "unsupported_language"
        assert result["language"] == "javascript"
        assert result["supported_languages"] == ["python"]


class TestCodeIntelReadOnlyApis:
    def test_code_references_returns_structured_python_results(self, tmp_path):
        path = tmp_path / "sample.py"
        path.write_text(
            "def demo(value):\n"
            "    total = value + 1\n"
            "    return total\n",
            encoding="utf-8",
        )

        result = code_references(path=str(path), line=2, column=4)

        assert result["ok"] is True
        assert result["backend"] == "python_ast"
        assert result["symbol_name"] == "total"
        assert [(item["line"], item["column"]) for item in result["references"]] == [(2, 4), (3, 11)]
        assert all(item["file"] == str(path) for item in result["references"])
        assert all(item["snippet"] for item in result["references"])

    def test_code_definition_returns_binding_location(self, tmp_path):
        path = tmp_path / "sample.py"
        path.write_text(
            "def demo(value):\n"
            "    total = value + 1\n"
            "    return total\n",
            encoding="utf-8",
        )

        result = code_definition(path=str(path), line=3, column=11)

        assert result["ok"] is True
        assert result["backend"] == "python_ast"
        assert result["symbol_name"] == "total"
        assert result["definition"] == {
            "file": str(path),
            "line": 2,
            "column": 4,
            "snippet": "    total = value + 1",
            "symbol_name": "total",
            "backend": "python_ast",
        }

    def test_code_references_refuses_unsupported_language(self, tmp_path):
        path = tmp_path / "sample.ts"
        path.write_text("const value = 1;\n", encoding="utf-8")

        result = code_references(path=str(path), line=1, column=6)

        assert result["ok"] is False
        assert result["code"] == "unsupported_language"
        assert result["backend"] is None


class TestCodeIntelToolExposure:
    def test_read_only_tools_are_registered_in_code_intel_toolset(self):
        import tools.code_intel  # noqa: F401 - ensure self-registering tool is imported

        references = registry.get_entry("code_references")
        definition = registry.get_entry("code_definition")

        assert references is not None
        assert references.toolset == "code_intel"
        assert definition is not None
        assert definition.toolset == "code_intel"
        assert references.schema["parameters"]["required"] == ["path", "line", "column"]
        assert definition.schema["parameters"]["required"] == ["path", "line", "column"]


class TestCodeIntelCommonPythonDefinitions:
    def test_code_definition_resolves_module_binding_used_inside_function(self, tmp_path):
        path = tmp_path / "sample.py"
        path.write_text("VALUE = 1\n\ndef demo():\n    return VALUE\n", encoding="utf-8")

        result = code_definition(path=str(path), line=4, column=11)

        assert result["ok"] is True
        assert result["symbol_name"] == "VALUE"
        assert result["definition"]["line"] == 1
        assert result["definition"]["column"] == 0

    def test_code_references_include_module_binding_uses_inside_function(self, tmp_path):
        path = tmp_path / "sample.py"
        path.write_text("VALUE = 1\n\ndef demo():\n    return VALUE\n", encoding="utf-8")

        result = code_references(path=str(path), line=1, column=0)

        assert result["ok"] is True
        assert [(item["line"], item["column"]) for item in result["references"]] == [(1, 0), (4, 11)]

    def test_code_definition_resolves_imported_symbol_to_import_binding(self, tmp_path):
        path = tmp_path / "sample.py"
        path.write_text("from math import sqrt\n\nvalue = sqrt(4)\n", encoding="utf-8")

        result = code_definition(path=str(path), line=3, column=8)

        assert result["ok"] is True
        assert result["symbol_name"] == "sqrt"
        assert result["definition"]["line"] == 1
        assert result["definition"]["column"] == 17
        assert result["definition"]["snippet"] == "from math import sqrt"
