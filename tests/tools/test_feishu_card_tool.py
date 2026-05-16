"""Tests for feishu_card_tool — card senders, file upload/send, and template builders."""

import importlib
import json
import unittest
from unittest.mock import MagicMock, patch

from tools.registry import registry


# Trigger tool discovery
importlib.import_module("tools.feishu_card_tool")


class TestCardTemplateBuilders(unittest.TestCase):
    """Verify card template functions produce valid Feishu Message Card JSON."""

    def test_task_card_pending(self):
        from tools.feishu_card_tool import task_card

        card = task_card("Review PR #42", "pending", "2026-04-30", "张三")
        self.assertEqual(card["schema"], "2.0")
        self.assertEqual(card["header"]["template"], "purple")
        self.assertIn("⏳", card["header"]["title"]["content"])

    def test_task_card_completed(self):
        from tools.feishu_card_tool import task_card

        card = task_card("Deploy v2", "completed", assignee_name="李四")
        self.assertEqual(card["header"]["template"], "green")
        self.assertIn("✅", card["header"]["title"]["content"])

    def test_task_card_overdue(self):
        from tools.feishu_card_tool import task_card

        card = task_card("Overdue Task", "overdue", due_date="2026-04-01")
        self.assertEqual(card["header"]["template"], "red")
        self.assertIn("🔴", card["header"]["title"]["content"])

    def test_task_card_with_url(self):
        from tools.feishu_card_tool import task_card

        card = task_card("Task", "pending", task_url="https://example.com/task/1")
        # Has action button
        action_elements = [e for e in card["elements"] if e.get("tag") == "action"]
        self.assertEqual(len(action_elements), 1)
        actions = action_elements[0]["actions"]
        self.assertEqual(actions[0]["action_id"], "open_task")

    def test_calendar_card(self):
        from tools.feishu_card_tool import calendar_card

        card = calendar_card(
            event_title="Q2 Planning",
            start_time="2026-04-30 09:00",
            end_time="2026-04-30 11:00",
            location="会议室A",
            attendees=["张三", "李四"],
            organizer="王五",
        )
        self.assertEqual(card["schema"], "2.0")
        self.assertEqual(card["header"]["template"], "blue")
        # Should have time, location, organizer, attendees fields
        div_elements = [e for e in card["elements"] if e.get("tag") == "div"]
        self.assertGreaterEqual(len(div_elements), 4)

    def test_generic_card(self):
        from tools.feishu_card_tool import generic_card

        card = generic_card(
            title="项目状态",
            body_lines=[("状态", "进行中"), ("负责人", "张三")],
            description="这是描述",
            status="warning",
        )
        self.assertEqual(card["header"]["template"], "yellow")
        # Two fields -> one field-div (paired), plus description -> one text-div
        div_elements = [e for e in card["elements"] if e.get("tag") == "div"]
        self.assertEqual(len(div_elements), 2)
        # First div has 2 fields
        self.assertEqual(len(div_elements[0]["fields"]), 2)

    def test_generic_card_actions(self):
        from tools.feishu_card_tool import generic_card

        actions = [{"tag": "button", "text": {"tag": "plain_text", "content": "确认"}}]
        card = generic_card("标题", actions=actions)
        action_elements = [e for e in card["elements"] if e.get("tag") == "action"]
        self.assertEqual(len(action_elements), 1)


class TestToolRegistration(unittest.TestCase):
    """Verify all feishu_card_tool tools are registered."""

    EXPECTED = [
        "feishu_send_card",
        "feishu_send_message",
        "feishu_upload_file",
        "feishu_send_file",
    ]

    def test_all_tools_registered(self):
        for name in self.EXPECTED:
            entry = registry.get_entry(name)
            self.assertIsNotNone(entry, f"{name} not registered")

    def test_schemas_valid(self):
        for name in self.EXPECTED:
            entry = registry.get_entry(name)
            schema = entry.schema
            self.assertIn("name", schema)
            self.assertEqual(schema["name"], name)
            self.assertIn("description", schema)
            self.assertIn("parameters", schema)
            self.assertEqual(schema["parameters"]["type"], "object")

    def test_send_card_requires_card_and_receive_id(self):
        entry = registry.get_entry("feishu_send_card")
        props = entry.schema["parameters"]["properties"]
        required = entry.schema["parameters"].get("required", [])
        self.assertIn("card", required)
        self.assertIn("receive_id", required)
        self.assertIn("card", props)
        self.assertIn("receive_id", props)

    def test_send_file_requires_file_path_and_receive_id(self):
        entry = registry.get_entry("feishu_send_file")
        required = entry.schema["parameters"].get("required", [])
        self.assertIn("file_path", required)
        self.assertIn("receive_id", required)


class TestSendCardHandler(unittest.TestCase):
    """Test feishu_send_card handler logic."""

    @patch("tools.feishu_card_tool._get_client")
    def test_send_card_success(self, mock_get_client):
        from tools.feishu_card_tool import _handle_feishu_send_card

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        mock_resp.data.message_id = "om_123"
        mock_client.im.v1.message.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        card = {"schema": "2.0", "header": {}, "elements": []}
        result = _handle_feishu_send_card({
            "card": card,
            "receive_id": "ou_abc",
            "receive_id_type": "open_id",
        })
        self.assertIn("om_123", result)
        self.assertIn("success", result.lower())

    def test_send_card_missing_card(self):
        from tools.feishu_card_tool import _handle_feishu_send_card

        result = _handle_feishu_send_card({"receive_id": "ou_abc"})
        self.assertIn("error", result.lower())

    def test_send_card_missing_receive_id(self):
        from tools.feishu_card_tool import _handle_feishu_send_card

        card = {"schema": "2.0", "header": {}, "elements": []}
        result = _handle_feishu_send_card({"card": card})
        self.assertIn("error", result.lower())


class TestSendMessageHandler(unittest.TestCase):
    """Test feishu_send_message handler."""

    @patch("tools.feishu_card_tool._get_client")
    def test_send_text_success(self, mock_get_client):
        from tools.feishu_card_tool import _handle_feishu_send_message

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        mock_resp.data.message_id = "om_456"
        mock_client.im.v1.message.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        result = _handle_feishu_send_message({
            "text": "Hello, world!",
            "receive_id": "ou_abc",
        })
        self.assertIn("om_456", result)
        self.assertIn("success", result.lower())

    def test_send_text_missing_text(self):
        from tools.feishu_card_tool import _handle_feishu_send_message

        result = _handle_feishu_send_message({"receive_id": "ou_abc"})
        self.assertIn("error", result.lower())


class TestUploadFileHandler(unittest.TestCase):
    """Test feishu_upload_file handler."""

    @patch("tools.feishu_card_tool._get_client")
    def test_upload_file_missing(self, mock_get_client):
        from tools.feishu_card_tool import _handle_feishu_upload_file

        result = _handle_feishu_upload_file({"file_path": "/nonexistent/file.pdf"})
        self.assertIn("not found", result.lower())

    def test_upload_file_no_path(self):
        from tools.feishu_card_tool import _handle_feishu_upload_file

        result = _handle_feishu_upload_file({})
        self.assertIn("error", result.lower())


class TestSendFileHandler(unittest.TestCase):
    """Test feishu_send_file handler."""

    def test_send_file_missing_path(self):
        from tools.feishu_card_tool import _handle_feishu_send_file

        result = _handle_feishu_send_file({"receive_id": "ou_abc"})
        self.assertIn("error", result.lower())

    def test_send_file_missing_receive_id(self):
        from tools.feishu_card_tool import _handle_feishu_send_file

        result = _handle_feishu_send_file({"file_path": "/tmp/test.pdf"})
        self.assertIn("error", result.lower())


if __name__ == "__main__":
    unittest.main()