"""Tests for Lumenfall image-to-video (input_reference) support.

Tests cover:
  - submit_video() correctly includes/excludes input_reference in payload
  - Video tool schema exposes image_url property
  - Video tool handler passes image_url through to the client
"""

import json
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def set_lumenfall_api_key(monkeypatch):
    """Ensure LUMENFALL_API_KEY is set for all tests."""
    monkeypatch.setenv("LUMENFALL_API_KEY", "lmnfl_test_key_123")


# ---------------------------------------------------------------------------
# submit_video — input_reference wire format
# ---------------------------------------------------------------------------


class TestSubmitVideoInputReference:
    """Verify that submit_video() adds input_reference when image_url is given."""

    @patch("tools.lumenfall_client.httpx.Client")
    def test_input_reference_present_when_image_url_provided(self, mock_client_cls):
        """When image_url is passed, payload must contain input_reference."""
        from tools.lumenfall_client import submit_video

        # Set up the mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "vid_123", "status": "pending"}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = submit_video(
            prompt="A cat walking",
            model="veo-2",
            image_url="https://example.com/cat.png",
        )

        # Verify the POST was called
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        # input_reference must be present with the correct structure
        assert "input_reference" in payload, "input_reference missing from payload"
        assert payload["input_reference"] == {
            "image_url": "https://example.com/cat.png"
        }

        # The top-level payload should NOT have a bare image_url key
        assert "image_url" not in payload

    @patch("tools.lumenfall_client.httpx.Client")
    def test_input_reference_absent_when_no_image_url(self, mock_client_cls):
        """When image_url is not passed, input_reference must NOT appear."""
        from tools.lumenfall_client import submit_video

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "vid_456", "status": "pending"}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = submit_video(
            prompt="A dog running",
            model="wan-2.1",
        )

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        assert "input_reference" not in payload, (
            "input_reference should not be in payload when image_url is not provided"
        )

    @patch("tools.lumenfall_client.httpx.Client")
    def test_input_reference_absent_when_image_url_is_none(self, mock_client_cls):
        """Explicitly passing image_url=None should not add input_reference."""
        from tools.lumenfall_client import submit_video

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "vid_789", "status": "pending"}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = submit_video(
            prompt="A bird flying",
            image_url=None,
        )

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        assert "input_reference" not in payload


# ---------------------------------------------------------------------------
# Video tool schema — image_url property
# ---------------------------------------------------------------------------


class TestVideoToolSchema:
    """Verify the video tool schema includes the image_url property."""

    def test_schema_has_image_url_property(self):
        from tools.lumenfall_video_tool import LUMENFALL_VIDEO_GENERATE_SCHEMA

        properties = LUMENFALL_VIDEO_GENERATE_SCHEMA["parameters"]["properties"]
        assert "image_url" in properties, "image_url missing from schema properties"

    def test_schema_image_url_is_string_type(self):
        from tools.lumenfall_video_tool import LUMENFALL_VIDEO_GENERATE_SCHEMA

        prop = LUMENFALL_VIDEO_GENERATE_SCHEMA["parameters"]["properties"]["image_url"]
        assert prop["type"] == "string"

    def test_schema_image_url_has_description(self):
        from tools.lumenfall_video_tool import LUMENFALL_VIDEO_GENERATE_SCHEMA

        prop = LUMENFALL_VIDEO_GENERATE_SCHEMA["parameters"]["properties"]["image_url"]
        assert "description" in prop
        assert len(prop["description"]) > 10  # meaningful description

    def test_image_url_not_required(self):
        """image_url should be optional — not in the required list."""
        from tools.lumenfall_video_tool import LUMENFALL_VIDEO_GENERATE_SCHEMA

        required = LUMENFALL_VIDEO_GENERATE_SCHEMA["parameters"].get("required", [])
        assert "image_url" not in required

    @patch("tools.lumenfall_video_tool.submit_video")
    @patch("tools.lumenfall_video_tool.poll_video")
    def test_handler_passes_image_url_to_client(self, mock_poll, mock_submit):
        """_handle_lumenfall_video_generate must pass image_url to submit_video."""
        from tools.lumenfall_video_tool import _handle_lumenfall_video_generate

        mock_submit.return_value = {"id": "vid_handler_test", "status": "pending"}
        mock_poll.return_value = {
            "status": "completed",
            "output": {"url": "https://example.com/video.mp4"},
            "metadata": {
                "provider_name": "test_provider",
                "executed_model": "veo-2",
            },
            "seconds": 5,
        }

        _handle_lumenfall_video_generate({
            "prompt": "Animate this image",
            "image_url": "https://example.com/source.png",
        })

        mock_submit.assert_called_once()
        call_kwargs = mock_submit.call_args
        # image_url should be passed as a keyword argument
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("image_url") == "https://example.com/source.png"
        else:
            # Fallback: check all args
            assert "https://example.com/source.png" in str(call_kwargs)

    @patch("tools.lumenfall_video_tool.submit_video")
    @patch("tools.lumenfall_video_tool.poll_video")
    def test_handler_omits_image_url_when_absent(self, mock_poll, mock_submit):
        """When image_url is not in args, handler should pass None (or omit)."""
        from tools.lumenfall_video_tool import _handle_lumenfall_video_generate

        mock_submit.return_value = {"id": "vid_no_img", "status": "pending"}
        mock_poll.return_value = {
            "status": "completed",
            "output": {"url": "https://example.com/video2.mp4"},
            "metadata": {
                "provider_name": "test_provider",
                "executed_model": "wan-2.1",
            },
            "seconds": 5,
        }

        _handle_lumenfall_video_generate({
            "prompt": "A sunset over the ocean",
        })

        mock_submit.assert_called_once()
        call_kwargs = mock_submit.call_args
        # image_url should be None or not present
        if call_kwargs.kwargs:
            image_url_val = call_kwargs.kwargs.get("image_url")
            assert image_url_val is None, (
                f"image_url should be None when not in args, got {image_url_val!r}"
            )


# ---------------------------------------------------------------------------
# lumenfall_list_models tool
# ---------------------------------------------------------------------------


MOCK_MODELS = [
    {"id": "flux-2-pro", "name": "FLUX 2 Pro", "modes": ["text-to-image"]},
    {"id": "veo-2", "name": "Veo 2", "modes": ["text-to-video", "image-to-video"]},
    {"id": "gpt-image-1", "name": "GPT Image 1", "modes": ["text-to-image", "image-edit"]},
    {"id": "wan-2.1", "name": "Wan 2.1", "modes": ["text-to-video"]},
    {"id": "bare-model", "modes": ["text-to-image"]},  # no "name" key — should fallback to id
]


class TestListModelsTool:
    """Verify the lumenfall_list_models tool function and handler."""

    @patch("tools.lumenfall_models_tool.list_models")
    def test_returns_all_models_when_no_capability_filter(self, mock_list):
        """When capability is None, all models should be returned."""
        from tools.lumenfall_models_tool import lumenfall_list_models_tool

        mock_list.return_value = MOCK_MODELS

        result = json.loads(lumenfall_list_models_tool())

        assert result["success"] is True
        assert result["total"] == 5
        assert len(result["models"]) == 5

    @patch("tools.lumenfall_models_tool.list_models")
    def test_filters_by_capability(self, mock_list):
        """When capability is provided, only matching models are returned."""
        from tools.lumenfall_models_tool import lumenfall_list_models_tool

        mock_list.return_value = MOCK_MODELS

        result = json.loads(lumenfall_list_models_tool(capability="text-to-video"))

        assert result["success"] is True
        assert result["total"] == 2
        model_ids = [m["id"] for m in result["models"]]
        assert "veo-2" in model_ids
        assert "wan-2.1" in model_ids
        assert "flux-2-pro" not in model_ids

    @patch("tools.lumenfall_models_tool.list_models")
    def test_returns_empty_when_no_matches(self, mock_list):
        """When capability filter matches nothing, return empty list."""
        from tools.lumenfall_models_tool import lumenfall_list_models_tool

        mock_list.return_value = MOCK_MODELS

        result = json.loads(lumenfall_list_models_tool(capability="nonexistent-mode"))

        assert result["success"] is True
        assert result["total"] == 0
        assert result["models"] == []

    @patch("tools.lumenfall_models_tool.list_models")
    def test_handles_api_errors_gracefully(self, mock_list):
        """LumenfallError should result in success=False with error message."""
        from tools.lumenfall_models_tool import lumenfall_list_models_tool
        from tools.lumenfall_client import LumenfallError

        mock_list.side_effect = LumenfallError("API rate limit exceeded", status_code=429)

        result = json.loads(lumenfall_list_models_tool())

        assert result["success"] is False
        assert result["models"] == []
        assert "API rate limit exceeded" in result["error"]

    @patch("tools.lumenfall_models_tool.list_models")
    def test_model_name_fallback_to_id(self, mock_list):
        """Models without a 'name' field should use 'id' as name."""
        from tools.lumenfall_models_tool import lumenfall_list_models_tool

        mock_list.return_value = MOCK_MODELS

        result = json.loads(lumenfall_list_models_tool())

        bare = [m for m in result["models"] if m["id"] == "bare-model"][0]
        assert bare["name"] == "bare-model"

    @patch("tools.lumenfall_models_tool.list_models")
    def test_model_output_fields(self, mock_list):
        """Each model in output should have id, name, modes."""
        from tools.lumenfall_models_tool import lumenfall_list_models_tool

        mock_list.return_value = MOCK_MODELS

        result = json.loads(lumenfall_list_models_tool())

        for model in result["models"]:
            assert "id" in model
            assert "name" in model
            assert "modes" in model

    @patch("tools.lumenfall_models_tool.list_models")
    def test_handler_passes_capability(self, mock_list):
        """_handle_lumenfall_list_models passes capability arg through."""
        from tools.lumenfall_models_tool import _handle_lumenfall_list_models

        mock_list.return_value = MOCK_MODELS

        result_str = _handle_lumenfall_list_models({"capability": "image-edit"})
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["total"] == 1
        assert result["models"][0]["id"] == "gpt-image-1"

    @patch("tools.lumenfall_models_tool.list_models")
    def test_handler_no_capability(self, mock_list):
        """_handle_lumenfall_list_models with empty args returns all models."""
        from tools.lumenfall_models_tool import _handle_lumenfall_list_models

        mock_list.return_value = MOCK_MODELS

        result_str = _handle_lumenfall_list_models({})
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["total"] == 5


# ---------------------------------------------------------------------------
# edit_image client function
# ---------------------------------------------------------------------------


class TestEditImageClient:
    """Verify that edit_image() sends the correct URL, data, files, and headers."""

    @patch("tools.lumenfall_client.httpx.Client")
    def test_edit_image_sends_correct_url_data_files(self, mock_client_cls):
        """edit_image() should POST to /images/edits with multipart form data."""
        from tools.lumenfall_client import edit_image

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"url": "https://cdn.lumenfall.ai/edited.png"}],
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = edit_image(
            image_url="https://example.com/source.png",
            prompt="Remove the background",
            model="gpt-image-1",
            mask_url="https://example.com/mask.png",
        )

        # Verify POST was called
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check URL ends with /images/edits
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert url.endswith("/images/edits"), f"Expected URL to end with /images/edits, got {url}"

        # Check data dict contains expected fields
        data = call_args.kwargs.get("data") or call_args[1].get("data", {})
        assert data["n"] == 1
        assert data["output_format"] == "png"
        assert data["response_format"] == "url"
        assert data["model"] == "gpt-image-1"
        assert data["prompt"] == "Remove the background"

        # Check files dict contains image and mask
        files = call_args.kwargs.get("files") or call_args[1].get("files", {})
        assert "image" in files
        assert "mask" in files

        # Headers should NOT contain Content-Type (httpx sets it for multipart)
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers", {})
        assert "Content-Type" not in headers

        # Check return value
        assert result == mock_response.json.return_value

    @patch("tools.lumenfall_client.httpx.Client")
    def test_edit_image_without_optional_fields(self, mock_client_cls):
        """edit_image() without prompt, model, mask_url should omit them from data/files."""
        from tools.lumenfall_client import edit_image

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"url": "https://cdn.lumenfall.ai/upscaled.png"}],
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = edit_image(image_url="https://example.com/source.png")

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        data = call_args.kwargs.get("data") or call_args[1].get("data", {})
        assert "model" not in data
        assert "prompt" not in data

        files = call_args.kwargs.get("files") or call_args[1].get("files", {})
        assert "image" in files
        assert "mask" not in files

    @patch("tools.lumenfall_client.httpx.Client")
    def test_edit_image_returns_parsed_response(self, mock_client_cls):
        """edit_image() should return the parsed JSON response."""
        from tools.lumenfall_client import edit_image

        expected = {
            "data": [{"url": "https://cdn.lumenfall.ai/result.png"}],
            "metadata": {"provider_name": "openai"},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = edit_image(image_url="https://example.com/img.png")
        assert result == expected


# ---------------------------------------------------------------------------
# lumenfall_image_edit tool
# ---------------------------------------------------------------------------


class TestImageEditTool:
    """Verify the lumenfall_image_edit tool function, schema, and handler."""

    @patch("tools.lumenfall_edit_tool.edit_image")
    def test_successful_edit_returns_success_and_image(self, mock_edit):
        """Successful edit should return {"success": True, "image": url}."""
        from tools.lumenfall_edit_tool import lumenfall_image_edit_tool

        mock_edit.return_value = {
            "data": [{"url": "https://cdn.lumenfall.ai/edited.png"}],
        }

        result = json.loads(lumenfall_image_edit_tool(
            image_url="https://example.com/source.png",
            prompt="Remove background",
        ))

        assert result["success"] is True
        assert result["image"] == "https://cdn.lumenfall.ai/edited.png"

    def test_schema_requires_image_url(self):
        """Schema must list image_url as required."""
        from tools.lumenfall_edit_tool import LUMENFALL_IMAGE_EDIT_SCHEMA

        required = LUMENFALL_IMAGE_EDIT_SCHEMA["parameters"]["required"]
        assert "image_url" in required

    def test_missing_image_url_returns_error(self):
        """When image_url is empty, tool should return an error."""
        from tools.lumenfall_edit_tool import lumenfall_image_edit_tool

        result = json.loads(lumenfall_image_edit_tool(image_url=""))
        assert result["success"] is False
        assert result["image"] is None

    def test_handler_checks_image_url_present(self):
        """_handle_lumenfall_image_edit should error when image_url is missing."""
        from tools.lumenfall_edit_tool import _handle_lumenfall_image_edit

        result = _handle_lumenfall_image_edit({})
        # tool_error returns a JSON string with "error" key
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("tools.lumenfall_edit_tool.edit_image")
    def test_handler_passes_all_args(self, mock_edit):
        """_handle_lumenfall_image_edit should pass all args through."""
        from tools.lumenfall_edit_tool import _handle_lumenfall_image_edit

        mock_edit.return_value = {
            "data": [{"url": "https://cdn.lumenfall.ai/result.png"}],
        }

        _handle_lumenfall_image_edit({
            "image_url": "https://example.com/img.png",
            "prompt": "Add a hat",
            "model": "gpt-image-1",
            "mask_url": "https://example.com/mask.png",
            "output_format": "jpeg",
        })

        mock_edit.assert_called_once()
        call_kwargs = mock_edit.call_args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs["image_url"] == "https://example.com/img.png"
            assert call_kwargs.kwargs["prompt"] == "Add a hat"
            assert call_kwargs.kwargs["model"] == "gpt-image-1"
            assert call_kwargs.kwargs["mask_url"] == "https://example.com/mask.png"
            assert call_kwargs.kwargs["output_format"] == "jpeg"

    def test_schema_has_expected_properties(self):
        """Schema should have image_url, prompt, model, mask_url properties."""
        from tools.lumenfall_edit_tool import LUMENFALL_IMAGE_EDIT_SCHEMA

        properties = LUMENFALL_IMAGE_EDIT_SCHEMA["parameters"]["properties"]
        assert "image_url" in properties
        assert "prompt" in properties
        assert "model" in properties
        assert "mask_url" in properties


class TestErrorGuidance:
    """Error responses should suggest using lumenfall_list_models."""

    def test_image_generate_error_suggests_list_models(self, monkeypatch):
        from tools import lumenfall_client, lumenfall_image_tool

        def raise_error(**kw):
            raise lumenfall_client.LumenfallError("model not found", status_code=404)

        monkeypatch.setattr(lumenfall_image_tool, "generate_image", raise_error)

        result = json.loads(lumenfall_image_tool.lumenfall_image_generate_tool(
            prompt="a cat", model="nonexistent-model",
        ))

        assert result["success"] is False
        assert "lumenfall_list_models" in result.get("hint", "")

    def test_video_generate_error_suggests_list_models(self, monkeypatch):
        from tools import lumenfall_client, lumenfall_video_tool

        def raise_error(**kw):
            raise lumenfall_client.LumenfallError("model not found", status_code=404)

        monkeypatch.setattr(lumenfall_video_tool, "submit_video", raise_error)

        result = json.loads(lumenfall_video_tool.lumenfall_video_generate_tool(
            prompt="a cat walking", model="nonexistent-model",
        ))

        assert result["success"] is False
        assert "lumenfall_list_models" in result.get("hint", "")
