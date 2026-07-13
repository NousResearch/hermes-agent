"""Tests for feishu_request_tool -- endpoint validation ("navigation map")."""

import importlib
import json
import unittest

from tools.registry import registry

importlib.import_module("tools.feishu_request_tool")
from tools.feishu_request_tool import (  # noqa: E402
    _handle_feishu_request,
    set_client,
    validate_endpoint,
)


class TestEndpointValidation(unittest.TestCase):
    """The map accepts known-good paths and rejects guessed ones."""

    def test_list_folder_correct_query_form_is_valid(self):
        template, paths = validate_endpoint("GET", "/open-apis/drive/v1/files")
        self.assertEqual(template, "/open-apis/drive/v1/files")
        self.assertEqual(paths, {})

    def test_list_folder_with_query_string_strips_and_validates(self):
        template, paths = validate_endpoint(
            "GET", "/open-apis/drive/v1/files?folder_token=fldcnABC123"
        )
        self.assertEqual(template, "/open-apis/drive/v1/files")
        self.assertEqual(paths, {})

    def test_wrong_children_path_is_rejected_with_suggestion(self):
        # The exact mistake from the incident.
        template, suggestions = validate_endpoint(
            "GET", "/open-apis/drive/v1/files/fldcnK0sP9zb1TejQsaN0S54cHc/children"
        )
        self.assertIsNone(template)
        self.assertIn("/open-apis/drive/v1/files", suggestions)

    def test_token_segment_is_extracted_into_paths(self):
        template, paths = validate_endpoint(
            "GET", "/open-apis/drive/v1/files/Lv4SdXIhAou70nxGvwLc/comments"
        )
        self.assertEqual(template, "/open-apis/drive/v1/files/:file_token/comments")
        self.assertEqual(paths, {"file_token": "Lv4SdXIhAou70nxGvwLc"})

    def test_add_global_comment_official_path(self):
        # Official docs: POST /open-apis/drive/v1/files/:file_token/comments
        template, paths = validate_endpoint(
            "POST", "/open-apis/drive/v1/files/Lv4SdXIhAou70nxGvwLc/comments"
        )
        self.assertEqual(template, "/open-apis/drive/v1/files/:file_token/comments")
        self.assertEqual(paths, {"file_token": "Lv4SdXIhAou70nxGvwLc"})

    def test_two_token_segments_extracted(self):
        template, paths = validate_endpoint(
            "PUT",
            "/open-apis/bitable/v1/apps/APPTOKEN123456789/tables/tblXYZ/records/recABC",
        )
        self.assertEqual(
            template,
            "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/:record_id",
        )
        self.assertEqual(paths["app_token"], "APPTOKEN123456789")
        self.assertEqual(paths["table_id"], "tblXYZ")
        self.assertEqual(paths["record_id"], "recABC")

    def test_method_matters(self):
        # The records collection is GET (list) and POST (create), not DELETE.
        template, _ = validate_endpoint(
            "DELETE",
            "/open-apis/bitable/v1/apps/APPTOKEN123456789/tables/tblXYZ/records",
        )
        self.assertIsNone(template)

    def test_unknown_root_returns_suggestions_list(self):
        template, suggestions = validate_endpoint("GET", "/open-apis/totally/made/up")
        self.assertIsNone(template)
        self.assertIsInstance(suggestions, list)

    def test_full_url_is_normalized_and_validated(self):
        # The agent's actual mistake: pasting a full URL with scheme+host.
        template, paths = validate_endpoint(
            "GET", "https://open.feishu.cn/open-apis/drive/v1/files?page_size=50"
        )
        self.assertEqual(template, "/open-apis/drive/v1/files")
        self.assertEqual(paths, {})

    def test_bare_host_url_is_normalized(self):
        template, _ = validate_endpoint(
            "GET", "open.feishu.cn/open-apis/drive/v1/files"
        )
        self.assertEqual(template, "/open-apis/drive/v1/files")

    def test_static_segment_wins_over_wildcard(self):
        # create_folder is a literal segment, not a :file_token.
        template, paths = validate_endpoint(
            "POST", "/open-apis/drive/v1/files/create_folder"
        )
        self.assertEqual(template, "/open-apis/drive/v1/files/create_folder")
        self.assertEqual(paths, {})


class TestHandler(unittest.TestCase):
    """Handler-level guards that don't need the lark SDK."""

    def tearDown(self):
        set_client(None)

    def test_missing_args(self):
        out = json.loads(_handle_feishu_request({"method": "GET"}))
        self.assertIn("error", out)

    def test_bad_method(self):
        out = json.loads(
            _handle_feishu_request({"method": "FETCH", "path": "/open-apis/drive/v1/files"})
        )
        self.assertIn("error", out)

    def test_invalid_path_rejected_before_client_lookup(self):
        # Even with no client set, an invalid path is rejected at validation.
        out = json.loads(
            _handle_feishu_request(
                {"method": "GET", "path": "/open-apis/drive/v1/files/TOKEN123456789/children"}
            )
        )
        self.assertIn("error", out)
        self.assertIn("valid_suggestions", out)

    def test_valid_path_without_client_reports_client_error(self):
        # Valid path passes validation, then fails on missing client (not 404).
        set_client(None)
        out = json.loads(
            _handle_feishu_request({"method": "GET", "path": "/open-apis/drive/v1/files"})
        )
        self.assertIn("error", out)
        self.assertIn("client not available", out["error"])


class TestToolsetWiring(unittest.TestCase):
    """feishu_request must be reachable exactly where the lark client is injected.

    resolve_toolset() reads the explicit ``tools`` list for statically-defined
    toolsets (it does NOT merge registry toolset membership), so the name must
    appear in toolsets.py to be reachable at all.

    The client is only injected on the comment-agent path (feishu_comment.py),
    which enables ["feishu_doc", "feishu_drive"]. So the tool belongs in
    feishu_drive — but NOT in the general hermes-feishu platform set, where no
    client is injected and the tool would only ever return "client not
    available". This test locks that placement so the tool is never exposed
    without a client behind it.
    """

    def test_resolves_via_feishu_drive(self):
        from toolsets import resolve_toolset
        self.assertIn("feishu_request", resolve_toolset("feishu_drive"))

    def test_reachable_by_comment_agent_toolsets(self):
        from toolsets import resolve_multiple_toolsets
        self.assertIn(
            "feishu_request",
            resolve_multiple_toolsets(["feishu_doc", "feishu_drive"]),
        )

    def test_not_exposed_on_general_feishu_platform(self):
        # hermes-feishu has no client injection — must not surface feishu_request.
        from toolsets import resolve_toolset
        self.assertNotIn("feishu_request", resolve_toolset("hermes-feishu"))


class TestRegistration(unittest.TestCase):
    def test_registered(self):
        entry = registry.get_entry("feishu_request")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.toolset, "feishu_drive")
        self.assertTrue(callable(entry.handler))

    def test_schema_shape(self):
        entry = registry.get_entry("feishu_request")
        props = entry.schema["parameters"]["properties"]
        self.assertIn("method", props)
        self.assertIn("path", props)
        self.assertEqual(entry.schema["parameters"]["required"], ["method", "path"])


if __name__ == "__main__":
    unittest.main()
