"""Tests for feishu_doc_tool and feishu_drive_tool — registration and schema validation."""

import importlib
import json
import unittest
from unittest.mock import Mock

from tools.registry import registry

# Trigger tool discovery so feishu tools get registered
importlib.import_module("tools.feishu_doc_tool")
importlib.import_module("tools.feishu_drive_tool")


class TestFeishuToolRegistration(unittest.TestCase):
    """Verify feishu tools are registered and have valid schemas."""

    EXPECTED_TOOLS = {
        "feishu_doc_read": "feishu_doc",
        "feishu_drive_list_comments": "feishu_drive",
        "feishu_drive_list_comment_replies": "feishu_drive",
        "feishu_drive_reply_comment": "feishu_drive",
        "feishu_drive_add_comment": "feishu_drive",
    }

    def test_all_tools_registered(self):
        for tool_name, toolset in self.EXPECTED_TOOLS.items():
            entry = registry.get_entry(tool_name)
            self.assertIsNotNone(entry, f"{tool_name} not registered")
            self.assertEqual(entry.toolset, toolset)

    def test_schemas_have_required_fields(self):
        for tool_name in self.EXPECTED_TOOLS:
            entry = registry.get_entry(tool_name)
            schema = entry.schema
            self.assertIn("name", schema)
            self.assertEqual(schema["name"], tool_name)
            self.assertIn("description", schema)
            self.assertIn("parameters", schema)
            self.assertIn("type", schema["parameters"])
            self.assertEqual(schema["parameters"]["type"], "object")

    def test_handlers_are_callable(self):
        for tool_name in self.EXPECTED_TOOLS:
            entry = registry.get_entry(tool_name)
            self.assertTrue(callable(entry.handler))

    def test_doc_read_schema_params(self):
        entry = registry.get_entry("feishu_doc_read")
        props = entry.schema["parameters"].get("properties", {})
        self.assertIn("doc_token", props)

    def test_drive_tools_require_file_token(self):
        for tool_name in self.EXPECTED_TOOLS:
            if tool_name == "feishu_doc_read":
                continue
            entry = registry.get_entry(tool_name)
            props = entry.schema["parameters"].get("properties", {})
            self.assertIn("file_token", props, f"{tool_name} missing file_token param")
            self.assertIn("file_type", props, f"{tool_name} missing file_type param")


class TestFeishuToolDocumentScope(unittest.TestCase):
    """Verify Feishu tools reject model-supplied tokens outside the comment document."""

    def tearDown(self):
        from tools import feishu_doc_tool, feishu_drive_tool

        feishu_doc_tool.set_client(None)
        feishu_drive_tool.set_client(None)

    def test_doc_read_rejects_cross_document_token(self):
        from tools import feishu_doc_tool

        client = Mock()
        feishu_doc_tool.set_client(client, allowed_file_type="docx", allowed_file_token="SRC_DOC")

        result = json.loads(feishu_doc_tool._handle_feishu_doc_read({"doc_token": "OTHER_DOC"}))

        self.assertIn("outside the authorized comment document", result["error"])
        client.request.assert_not_called()

    def test_drive_tools_reject_cross_document_file_token(self):
        from tools import feishu_drive_tool

        client = Mock()
        feishu_drive_tool.set_client(client, allowed_file_type="docx", allowed_file_token="SRC_DOC")

        cases = [
            (feishu_drive_tool._handle_list_comments, {"file_token": "OTHER_DOC"}),
            (feishu_drive_tool._handle_list_replies, {"file_token": "OTHER_DOC", "comment_id": "c1"}),
            (feishu_drive_tool._handle_reply_comment, {"file_token": "OTHER_DOC", "comment_id": "c1", "content": "hi"}),
            (feishu_drive_tool._handle_add_comment, {"file_token": "OTHER_DOC", "content": "hi"}),
        ]

        for handler, args in cases:
            with self.subTest(handler=handler.__name__):
                result = json.loads(handler(args))
                self.assertIn("outside the authorized comment document", result["error"])

        client.request.assert_not_called()

    def test_drive_tools_reject_cross_document_file_type(self):
        from tools import feishu_drive_tool

        client = Mock()
        feishu_drive_tool.set_client(client, allowed_file_type="docx", allowed_file_token="SRC_DOC")

        result = json.loads(feishu_drive_tool._handle_list_comments({"file_token": "SRC_DOC", "file_type": "sheet"}))

        self.assertIn("outside the authorized comment document", result["error"])
        client.request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
