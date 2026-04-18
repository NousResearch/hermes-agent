import json

from model_tools import get_tool_definitions
from toolsets import resolve_toolset
from tools.ast_tools import (
    AST_TOOLS,
    AST_FIND_NODES_SCHEMA,
    AST_LIST_DEFS_SCHEMA,
    ast_find_nodes_tool,
    ast_list_defs_tool,
)


class TestAstToolRegistration:
    def test_exports_expected_tools_and_schemas(self):
        names = {tool["name"] for tool in AST_TOOLS}
        assert names == {"ast_list_defs", "ast_find_nodes"}

        for schema in [AST_LIST_DEFS_SCHEMA, AST_FIND_NODES_SCHEMA]:
            assert "name" in schema
            assert "description" in schema
            assert "properties" in schema["parameters"]

        assert "language" in AST_LIST_DEFS_SCHEMA["parameters"]["properties"]
        assert "language" in AST_FIND_NODES_SCHEMA["parameters"]["properties"]

    def test_code_intel_toolset_exposes_ast_tools(self):
        code_intel_tools = set(resolve_toolset("code_intel"))
        assert {"ast_list_defs", "ast_find_nodes"} <= code_intel_tools

        model_tool_names = {
            tool["function"]["name"]
            for tool in get_tool_definitions(enabled_toolsets=["code_intel"], quiet_mode=True)
        }
        assert {"ast_list_defs", "ast_find_nodes"} <= model_tool_names


class TestAstHandlers:
    def test_list_defs_reports_top_level_items_and_optional_nested_items(self, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text(
            "import os\n\n"
            "class Example:\n"
            "    def method(self):\n"
            "        return 1\n\n"
            "async def run():\n"
            "    return 2\n\n"
            "def outer():\n"
            "    def inner():\n"
            "        return 3\n"
            "    return inner()\n",
            encoding="utf-8",
        )

        top_level = json.loads(ast_list_defs_tool(path=str(source_path)))
        nested = json.loads(ast_list_defs_tool(path=str(source_path), include_nested=True))

        assert [item["name"] for item in top_level["definitions"]] == ["Example", "run", "outer"]
        assert "inner" not in [item["name"] for item in top_level["definitions"]]
        nested_names = [item["name"] for item in nested["definitions"]]
        assert "inner" in nested_names
        assert nested["definitions"][-1]["parent"] == "outer"

    def test_find_nodes_filters_by_type_and_name(self, tmp_path):
        source_path = tmp_path / "sample.py"
        source_path.write_text(
            "def helper():\n"
            "    return 1\n\n"
            "def build():\n"
            "    value = helper()\n"
            "    return helper()\n",
            encoding="utf-8",
        )

        calls = json.loads(ast_find_nodes_tool(path=str(source_path), node_type="Call", name="helper"))
        defs = json.loads(ast_find_nodes_tool(path=str(source_path), node_type="FunctionDef", name="build"))

        assert calls["count"] == 2
        assert all(match["display_name"] == "helper" for match in calls["matches"])
        assert defs["count"] == 1
        assert defs["matches"][0]["name"] == "build"
        assert defs["matches"][0]["parent"] is None

    def test_syntax_error_returns_structured_error(self, tmp_path):
        source_path = tmp_path / "broken.py"
        source_path.write_text("def broken(:\n    pass\n", encoding="utf-8")

        result = json.loads(ast_list_defs_tool(path=str(source_path)))

        assert "error" in result
        assert result["error_type"] == "SyntaxError"
        assert result["path"] == str(source_path)
        assert result["lineno"] == 1

    def test_list_defs_supports_json_structural_traversal(self, tmp_path):
        source_path = tmp_path / "config.json"
        source_path.write_text(
            '{\n'
            '  "name": "demo",\n'
            '  "build": {\n'
            '    "entry": "src/index.js",\n'
            '    "flags": {"watch": true}\n'
            '  },\n'
            '  "metadata": {"owner": "hermes"}\n'
            '}\n',
            encoding="utf-8",
        )

        top_level = json.loads(ast_list_defs_tool(path=str(source_path)))
        nested = json.loads(ast_list_defs_tool(path=str(source_path), include_nested=True))
        watch_matches = json.loads(
            ast_find_nodes_tool(path=str(source_path), node_type="ObjectProperty", name="watch")
        )

        assert top_level["language"] == "json"
        assert top_level["parser"] == "json_structure_v1"
        assert top_level["confidence"] == "high"
        assert [item["name"] for item in top_level["definitions"]] == ["name", "build", "metadata"]
        assert "flags" in [item["name"] for item in nested["definitions"]]
        assert watch_matches["count"] == 1
        assert watch_matches["matches"][0]["parent"] == "flags"
        assert watch_matches["matches"][0]["node_type"] == "ObjectProperty"

    def test_javascript_heuristics_list_defs_and_find_nodes(self, tmp_path):
        source_path = tmp_path / "sample.js"
        source_path.write_text(
            'export class Example {\n'
            '  async method() {\n'
            '    function innerHelper() {\n'
            '      return 1;\n'
            '    }\n'
            '    return innerHelper();\n'
            '  }\n'
            '}\n\n'
            'export async function runTask() {\n'
            '  return 2;\n'
            '}\n\n'
            'const helper = () => 3;\n',
            encoding="utf-8",
        )

        top_level = json.loads(ast_list_defs_tool(path=str(source_path)))
        nested = json.loads(ast_list_defs_tool(path=str(source_path), include_nested=True))
        method_matches = json.loads(
            ast_find_nodes_tool(path=str(source_path), node_type="MethodDefinition", name="method")
        )

        assert top_level["language"] == "javascript"
        assert top_level["parser"] == "hermes_js_structure_v1"
        assert top_level["confidence"] == "medium"
        assert [item["name"] for item in top_level["definitions"]] == ["Example", "runTask", "helper"]
        nested_items = {item["name"]: item for item in nested["definitions"]}
        assert nested_items["method"]["parent"] == "Example"
        assert nested_items["innerHelper"]["parent"] == "method"
        assert method_matches["count"] == 1
        assert method_matches["matches"][0]["parent"] == "Example"

    def test_typescript_heuristics_support_interfaces_and_type_aliases(self, tmp_path):
        source_path = tmp_path / "sample.ts"
        source_path.write_text(
            'export interface User {\n'
            '  name: string;\n'
            '}\n\n'
            'export type UserId = string;\n\n'
            'export const loadUser = async (): Promise<User> => ({ name: "demo" });\n',
            encoding="utf-8",
        )

        result = json.loads(ast_list_defs_tool(path=str(source_path)))
        interface_matches = json.loads(
            ast_find_nodes_tool(path=str(source_path), node_type="InterfaceDeclaration", name="User")
        )

        assert result["language"] == "typescript"
        assert result["parser"] == "hermes_js_structure_v1"
        assert [item["name"] for item in result["definitions"]] == ["User", "UserId", "loadUser"]
        assert result["definitions"][0]["node_type"] == "InterfaceDeclaration"
        assert interface_matches["count"] == 1
        assert interface_matches["matches"][0]["name"] == "User"

    def test_unsupported_language_returns_structured_error(self, tmp_path):
        source_path = tmp_path / "notes.md"
        source_path.write_text("# hello\n", encoding="utf-8")

        result = json.loads(ast_list_defs_tool(path=str(source_path)))

        assert "error" in result
        assert result["error_type"] == "UnsupportedLanguage"
        assert result["path"] == str(source_path)
        assert result["detected_language"] == "unknown"
        assert set(result["supported_languages"]) == {"python", "javascript", "typescript", "json"}
