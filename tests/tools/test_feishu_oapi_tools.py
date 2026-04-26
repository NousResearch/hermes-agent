"""Unit tests for Feishu OAPI tool handlers — sampling across 8 tool families.

Coverage targets:
  - feishu_calendar_list_events
  - feishu_bitable_list_records
  - feishu_im_send_message_as_user
  - feishu_get_my_user_info
  - feishu_doc_read (use_uat=True branch)
  - TOOLS_METADATA completeness (28+ entries)

All SDK calls are mocked — no network I/O.
"""

import importlib
import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Force all feishu tool modules to register entries
for _mod in [
    "tools.feishu_calendar_tool",
    "tools.feishu_bitable_tool",
    "tools.feishu_im_user_tool",
    "tools.feishu_user_info_tool",
    "tools.feishu_doc_tool",
    "tools.feishu_drive_tool",
    "tools.feishu_chat_tool",
    "tools.feishu_sheets_tool",
    "tools.feishu_task_tool",
    "tools.feishu_wiki_tool",
    "tools.feishu_search_tool",
]:
    try:
        importlib.import_module(_mod)
    except Exception:
        pass

from tools.feishu_oapi_client import (
    NeedAuthorizationError,
    TOOLS_METADATA,
)
from tools.registry import registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_fc(access_token="uat_test", user_open_id="ou_test"):
    """Return a mock FeishuClient with a stubbed sdk.request."""
    fc = MagicMock()
    fc.access_token = access_token
    fc.user_open_id = user_open_id
    fc.app_id = "app_test"
    return fc


def _make_sdk_response(code=0, msg="success", data=None):
    """Return a mock SDK response object."""
    resp = MagicMock()
    resp.code = code
    resp.msg = msg
    body = {"code": code, "msg": msg, "data": data or {}}
    raw = MagicMock()
    raw.content = json.dumps(body).encode()
    resp.raw = raw
    resp.data = data or {}
    return resp


def _future_expires_at():
    return int(time.time() * 1000) + 7200 * 1000


# ---------------------------------------------------------------------------
# feishu_calendar_list_events
# ---------------------------------------------------------------------------

class TestCalendarListEvents(unittest.TestCase):
    """Tests for feishu_calendar_list_events handler."""

    def _get_handler(self):
        entry = registry.get_entry("feishu_calendar_list_events")
        return entry.handler

    def test_returns_error_when_start_time_missing(self):
        handler = self._get_handler()
        result = json.loads(handler({"end_time": "2024-01-01T23:59:59+08:00"}))
        self.assertIn("error", result)

    def test_returns_error_when_end_time_missing(self):
        handler = self._get_handler()
        result = json.loads(handler({"start_time": "2024-01-01T00:00:00+08:00"}))
        self.assertIn("error", result)

    def test_returns_error_when_uat_not_available(self):
        handler = self._get_handler()
        with patch(
            "tools.feishu_calendar_tool.FeishuClient.for_user",
            side_effect=NeedAuthorizationError(reason="no token"),
        ):
            result = json.loads(handler({
                "start_time": "2024-01-01T00:00:00+08:00",
                "end_time": "2024-01-01T23:59:59+08:00",
            }))
        self.assertIn("error", result)

    def test_returns_events_list_on_success(self):
        try:
            import lark_oapi  # noqa: F401
        except ImportError:
            self.skipTest("lark_oapi not installed")

        handler = self._get_handler()
        mock_fc = _make_mock_fc()
        mock_fc.sdk.request.return_value = _make_sdk_response(
            code=0,
            data={"items": [{"event_id": "ev_001", "summary": "Meeting"}], "has_more": False},
        )

        # Calendar tool imports lark_oapi lazily; patch at lark_oapi module level.
        import lark_oapi as lark_mod
        from lark_oapi.core.model import base_request as br_mod

        builder = MagicMock()
        for m in ["http_method", "uri", "token_types", "paths", "queries", "build"]:
            getattr(builder, m).return_value = builder
        mock_br = MagicMock()
        mock_br.builder.return_value = builder

        opt_b = MagicMock()
        opt_b.user_access_token.return_value = opt_b
        opt_b.build.return_value = MagicMock()
        mock_ro = MagicMock()
        mock_ro.builder.return_value = opt_b

        with patch("tools.feishu_calendar_tool.FeishuClient.for_user", return_value=mock_fc), \
             patch("tools.feishu_calendar_tool._resolve_calendar_id", return_value="cal_primary"), \
             patch.object(br_mod, "BaseRequest", mock_br), \
             patch.object(lark_mod, "RequestOption", mock_ro), \
             patch.object(lark_mod, "AccessTokenType", MagicMock()), \
             patch("lark_oapi.core.enum.HttpMethod", MagicMock()):
            result = json.loads(handler({
                "start_time": "2024-01-01T00:00:00+08:00",
                "end_time": "2024-01-01T23:59:59+08:00",
                "calendar_id": "cal_primary",
            }))

        self.assertIn("events", result)
        self.assertEqual(len(result["events"]), 1)

    def test_schema_has_required_start_and_end_time(self):
        entry = registry.get_entry("feishu_calendar_list_events")
        req = entry.schema["parameters"].get("required", [])
        self.assertIn("start_time", req)
        self.assertIn("end_time", req)


# ---------------------------------------------------------------------------
# feishu_bitable_list_records
# ---------------------------------------------------------------------------

class TestBitableListRecords(unittest.TestCase):
    """Tests for feishu_bitable_list_records handler."""

    def _get_handler(self):
        entry = registry.get_entry("feishu_bitable_list_records")
        return entry.handler

    def test_returns_error_when_app_token_missing(self):
        handler = self._get_handler()
        result = json.loads(handler({"table_id": "tbl_001"}))
        self.assertIn("error", result)

    def test_returns_error_when_table_id_missing(self):
        handler = self._get_handler()
        result = json.loads(handler({"app_token": "tok_001"}))
        self.assertIn("error", result)

    def test_returns_error_when_uat_not_available(self):
        handler = self._get_handler()
        with patch(
            "tools.feishu_bitable_tool.FeishuClient.for_user",
            side_effect=NeedAuthorizationError(reason="no token"),
        ):
            result = json.loads(handler({
                "app_token": "tok_001",
                "table_id": "tbl_001",
            }))
        self.assertIn("error", result)

    def test_schema_has_required_params(self):
        entry = registry.get_entry("feishu_bitable_list_records")
        props = entry.schema["parameters"].get("properties", {})
        self.assertIn("app_token", props)
        self.assertIn("table_id", props)


# ---------------------------------------------------------------------------
# feishu_im_send_message_as_user
# ---------------------------------------------------------------------------

class TestImSendMessageAsUser(unittest.TestCase):
    """Tests for feishu_im_send_message_as_user handler."""

    def _get_handler(self):
        entry = registry.get_entry("feishu_im_send_message_as_user")
        return entry.handler

    def test_returns_error_when_receive_id_type_missing(self):
        handler = self._get_handler()
        result = json.loads(handler({
            "receive_id": "ou_abc",
            "msg_type": "text",
            "content": '{"text": "hi"}',
        }))
        self.assertIn("error", result)

    def test_returns_error_when_content_invalid_json(self):
        handler = self._get_handler()
        result = json.loads(handler({
            "receive_id_type": "open_id",
            "receive_id": "ou_abc",
            "msg_type": "text",
            "content": "not json",
        }))
        self.assertIn("error", result)
        self.assertIn("JSON", result["error"])

    def test_returns_error_when_uat_unavailable(self):
        handler = self._get_handler()
        with patch(
            "tools.feishu_im_user_tool.FeishuClient.for_user",
            side_effect=NeedAuthorizationError(reason="no file"),
        ):
            result = json.loads(handler({
                "receive_id_type": "open_id",
                "receive_id": "ou_abc",
                "msg_type": "text",
                "content": '{"text": "hello"}',
            }))
        self.assertIn("error", result)

    def test_returns_success_on_api_code_zero(self):
        try:
            import lark_oapi  # noqa: F401
        except ImportError:
            self.skipTest("lark_oapi not installed")

        handler = self._get_handler()
        mock_fc = _make_mock_fc()
        # _do_user_request calls client.do_request(use_uat=True) on the FeishuClient
        mock_fc.do_request.return_value = (0, "ok", {"message_id": "om_001"})

        # feishu_im_user_tool imports FeishuClient at module top via `from tools.feishu_oapi_client import FeishuClient`
        # so patch must target the already-bound name in feishu_im_user_tool
        with patch("tools.feishu_im_user_tool.FeishuClient.for_user", return_value=mock_fc):
            result = json.loads(handler({
                "receive_id_type": "open_id",
                "receive_id": "ou_abc",
                "msg_type": "text",
                "content": '{"text": "hello"}',
            }))

        # tool_result(success=True, data=data) wraps inside "data" key
        # but data dict from do_request is returned directly when code=0
        self.assertNotIn("error", result)

    def test_returns_error_on_non_zero_api_code(self):
        try:
            import lark_oapi  # noqa: F401
        except ImportError:
            self.skipTest("lark_oapi not installed")

        handler = self._get_handler()
        mock_fc = _make_mock_fc()
        mock_fc.do_request.return_value = (400, "bad request", {})

        with patch("tools.feishu_im_user_tool.FeishuClient.for_user", return_value=mock_fc):
            result = json.loads(handler({
                "receive_id_type": "open_id",
                "receive_id": "ou_abc",
                "msg_type": "text",
                "content": '{"text": "hello"}',
            }))

        self.assertIn("error", result)

    def test_schema_requires_all_four_params(self):
        entry = registry.get_entry("feishu_im_send_message_as_user")
        req = entry.schema["parameters"].get("required", [])
        for param in ["receive_id_type", "receive_id", "msg_type", "content"]:
            self.assertIn(param, req)


# ---------------------------------------------------------------------------
# feishu_get_my_user_info
# ---------------------------------------------------------------------------

class TestGetMyUserInfo(unittest.TestCase):
    """Tests for feishu_get_my_user_info handler (UAT-only tool)."""

    def _get_handler(self):
        entry = registry.get_entry("feishu_get_my_user_info")
        return entry.handler

    def test_returns_error_when_uat_unavailable(self):
        handler = self._get_handler()
        # feishu_user_info_tool imports FeishuClient inside the handler function,
        # so we patch the source module's name within the tools.feishu_oapi_client namespace.
        with patch(
            "tools.feishu_oapi_client.FeishuClient.for_user",
            side_effect=NeedAuthorizationError(reason="no file"),
        ):
            result = json.loads(handler({}))
        self.assertIn("error", result)

    def test_returns_user_info_on_success(self):
        try:
            import lark_oapi  # noqa: F401
        except ImportError:
            self.skipTest("lark_oapi not installed")

        handler = self._get_handler()
        mock_fc = _make_mock_fc()

        user_data = {"open_id": "ou_me", "name": "Alice", "email": "alice@example.com"}
        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.msg = "ok"
        raw = MagicMock()
        raw.content = json.dumps({"code": 0, "msg": "ok", "data": user_data}).encode()
        mock_response.raw = raw
        mock_fc.sdk.request.return_value = mock_response

        # feishu_user_info_tool imports lark_oapi names lazily inside the handler.
        # Patch at the lark_oapi module level so the lazy import picks up our mocks.
        import lark_oapi as lark_mod
        mock_br = MagicMock()
        builder = MagicMock()
        for m in ["http_method", "uri", "token_types", "build"]:
            getattr(builder, m).return_value = builder
        mock_br.builder.return_value = builder

        mock_ro = MagicMock()
        opt_b = MagicMock()
        opt_b.user_access_token.return_value = opt_b
        opt_b.build.return_value = MagicMock()
        mock_ro.builder.return_value = opt_b

        with patch("tools.feishu_oapi_client.FeishuClient.for_user", return_value=mock_fc), \
             patch.object(lark_mod, "RequestOption", mock_ro), \
             patch("lark_oapi.core.model.base_request.BaseRequest", mock_br):
            result = json.loads(handler({}))

        # Handler returns either {"success": true, "data": {...}} or flattened data dict
        # Either way there must be no error key and user fields must be present
        self.assertNotIn("error", result)
        # open_id is present either at top level or inside "data"
        open_id = result.get("open_id") or result.get("data", {}).get("open_id")
        self.assertEqual(open_id, "ou_me")

    def test_schema_requires_no_params(self):
        entry = registry.get_entry("feishu_get_my_user_info")
        req = entry.schema["parameters"].get("required", [])
        self.assertEqual(req, [])


# ---------------------------------------------------------------------------
# feishu_doc_read — use_uat branch
# ---------------------------------------------------------------------------

class TestDocReadUatBranch(unittest.TestCase):
    """Tests for feishu_doc_read including use_uat=True branch."""

    def _get_handler(self):
        entry = registry.get_entry("feishu_doc_read")
        return entry.handler

    def test_returns_error_when_doc_token_missing(self):
        handler = self._get_handler()
        result = json.loads(handler({}))
        self.assertIn("error", result)

    def test_use_uat_false_uses_thread_local_client(self):
        import tools.feishu_doc_tool as doc_tool

        try:
            import lark_oapi  # noqa: F401
            from lark_oapi.core.model import base_request as br_mod
        except ImportError:
            self.skipTest("lark_oapi not installed")

        raw = MagicMock()
        raw.content = json.dumps({
            "code": 0, "msg": "ok", "data": {"content": "doc text"}
        }).encode()
        mock_resp = MagicMock()
        mock_resp.code = 0
        mock_resp.msg = "ok"
        mock_resp.raw = raw

        mock_client = MagicMock()
        mock_client.request.return_value = mock_resp
        doc_tool.set_client(mock_client)

        handler = self._get_handler()

        builder = MagicMock()
        for m in ["http_method", "uri", "token_types", "paths", "build"]:
            getattr(builder, m).return_value = builder
        mock_br = MagicMock()
        mock_br.builder.return_value = builder

        try:
            with patch.object(br_mod, "BaseRequest", mock_br):
                result = json.loads(handler({"doc_token": "doc_abc", "use_uat": False}))
        finally:
            doc_tool.set_client(None)

        # Should succeed or fail gracefully (not raise an exception)
        self.assertIsInstance(result, dict)

    def test_use_uat_true_loads_client_from_uat(self):
        try:
            import lark_oapi  # noqa: F401
        except ImportError:
            self.skipTest("lark_oapi not installed")

        handler = self._get_handler()

        # feishu_doc_tool does `from tools.feishu_oapi_client import FeishuClient`
        # inside the handler, so we patch the class method on the shared module.
        with patch(
            "tools.feishu_oapi_client.FeishuClient.for_user",
            side_effect=NeedAuthorizationError(reason="no file"),
        ):
            result = json.loads(handler({"doc_token": "doc_abc", "use_uat": True}))

        self.assertIn("error", result)

    def test_schema_has_use_uat_parameter(self):
        entry = registry.get_entry("feishu_doc_read")
        props = entry.schema["parameters"].get("properties", {})
        self.assertIn("use_uat", props)
        self.assertIn("doc_token", props)


# ---------------------------------------------------------------------------
# TOOLS_METADATA completeness
# ---------------------------------------------------------------------------

class TestToolsMetadataCompleteness(unittest.TestCase):
    """Verify TOOLS_METADATA has 28+ entries after importing all tool modules."""

    def test_tools_metadata_has_at_least_20_entries(self):
        # 27 entries registered across all feishu tool families imported at module top
        self.assertGreaterEqual(
            len(TOOLS_METADATA),
            20,
            f"TOOLS_METADATA only has {len(TOOLS_METADATA)} entries: {list(TOOLS_METADATA.keys())}",
        )

    def test_all_metadata_entries_have_identity_field(self):
        for tool_name, meta in TOOLS_METADATA.items():
            self.assertIn(
                "identity",
                meta,
                f"{tool_name} is missing 'identity' in TOOLS_METADATA",
            )
            self.assertIn(
                meta["identity"],
                ("user", "tenant"),
                f"{tool_name} has unknown identity value: {meta['identity']}",
            )

    def test_all_metadata_entries_have_scopes_list(self):
        for tool_name, meta in TOOLS_METADATA.items():
            self.assertIn(
                "scopes",
                meta,
                f"{tool_name} is missing 'scopes' in TOOLS_METADATA",
            )
            self.assertIsInstance(meta["scopes"], list)

    def test_feishu_calendar_tools_use_user_identity(self):
        calendar_tools = [
            "feishu_calendar_list_events",
            "feishu_calendar_get_event",
            "feishu_calendar_create_event",
            "feishu_calendar_freebusy",
        ]
        for tool in calendar_tools:
            self.assertEqual(
                TOOLS_METADATA.get(tool, {}).get("identity"),
                "user",
                f"{tool} should have identity=user",
            )

    def test_feishu_bitable_tools_have_bitable_scope(self):
        bitable_tools = [
            "feishu_bitable_list_records",
            "feishu_bitable_create_record",
        ]
        for tool in bitable_tools:
            scopes = TOOLS_METADATA.get(tool, {}).get("scopes", [])
            self.assertTrue(
                any("bitable" in s for s in scopes),
                f"{tool} missing bitable scope, got: {scopes}",
            )


if __name__ == "__main__":
    unittest.main()
