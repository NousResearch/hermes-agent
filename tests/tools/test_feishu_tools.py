"""Tests for feishu_doc_tool and feishu_drive_tool — registration and schema validation."""

import importlib
import unittest
from concurrent.futures import ThreadPoolExecutor

from tools.registry import registry
from tools.thread_context import propagate_context_to_thread

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


    def test_drive_tools_require_file_token(self):
        for tool_name in self.EXPECTED_TOOLS:
            if tool_name == "feishu_doc_read":
                continue
            entry = registry.get_entry(tool_name)
            props = entry.schema["parameters"].get("properties", {})
            self.assertIn("file_token", props, f"{tool_name} missing file_token param")
            self.assertIn("file_type", props, f"{tool_name} missing file_type param")


class TestFeishuCommentClientContext(unittest.TestCase):
    """The comment client must survive concurrent tool dispatch."""

    def setUp(self):
        self.doc_tool = importlib.import_module("tools.feishu_doc_tool")
        self.drive_tool = importlib.import_module("tools.feishu_drive_tool")
        self.doc_tool.set_client(None)
        self.drive_tool.set_client(None)

    def tearDown(self):
        self.doc_tool.set_client(None)
        self.drive_tool.set_client(None)

    def test_comment_client_reaches_propagated_tool_worker(self):
        client = object()
        self.doc_tool.set_client(client)
        self.drive_tool.set_client(client)

        with ThreadPoolExecutor(max_workers=1) as executor:
            observed = executor.submit(
                propagate_context_to_thread(
                    lambda: (self.doc_tool.get_client(), self.drive_tool.get_client())
                )
            ).result()

        self.assertEqual(observed, (client, client))

    def test_cleared_comment_client_does_not_leak_to_reused_worker(self):
        client = object()
        self.doc_tool.set_client(client)
        self.drive_tool.set_client(client)

        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(
                propagate_context_to_thread(
                    lambda: (self.doc_tool.get_client(), self.drive_tool.get_client())
                )
            ).result()

            self.doc_tool.set_client(None)
            self.drive_tool.set_client(None)
            second = executor.submit(
                propagate_context_to_thread(
                    lambda: (self.doc_tool.get_client(), self.drive_tool.get_client())
                )
            ).result()

        self.assertEqual(first, (client, client))
        self.assertEqual(second, (None, None))


if __name__ == "__main__":
    unittest.main()
