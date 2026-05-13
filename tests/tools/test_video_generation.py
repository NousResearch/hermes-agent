"""Tests for video_generation_tool with mocked FAL client."""

import json
import pytest
from unittest.mock import MagicMock, patch
from tools import video_generation_tool as vgt


class TestVideoGenerateMock:
    """Test video generation without actual API calls."""

    @patch.object(vgt, '_submit_fal_request')
    @patch.object(vgt, 'fal_key_is_configured', return_value=True)
    def test_basic_video_generation(self, mock_auth, mock_submit):
        """Test basic video generation flow."""
        # Mock handler
        mock_handler = MagicMock()
        mock_handler.status.return_value = {"status": "COMPLETED"}
        mock_handler.get.return_value = {
            "video": {"url": "https://fal.media/test.mp4"}
        }
        mock_submit.return_value = mock_handler

        # Test
        result = json.loads(vgt.video_generate_tool(
            prompt="A cat walking",
            duration=5
        ))

        assert result["success"] is True
        assert "test.mp4" in result["video"]["url"]
        assert result["video"]["duration"] == 5
        assert result["video"]["generation_time_seconds"] >= 0

    @patch.object(vgt, 'fal_key_is_configured', return_value=True)
    @patch.object(vgt, '_submit_fal_request')
    def test_duration_too_short(self, mock_submit, mock_auth):
        """Test duration below minimum."""
        result = json.loads(vgt.video_generate_tool(
            prompt="test",
            duration=1  # Min is 3
        ))
        assert result["success"] is False
        assert "Duration must be between" in result["error"]

    @patch.object(vgt, 'fal_key_is_configured', return_value=True)
    @patch.object(vgt, '_submit_fal_request')
    def test_duration_too_long(self, mock_submit, mock_auth):
        """Test duration above maximum."""
        result = json.loads(vgt.video_generate_tool(
            prompt="test",
            duration=100  # Max is 15
        ))
        assert result["success"] is False
        assert "Duration must be between" in result["error"]

    @patch.object(vgt, 'fal_key_is_configured', return_value=True)
    def test_missing_prompt(self, mock_auth):
        """Test empty prompt validation."""
        result = json.loads(vgt.video_generate_tool(prompt=""))
        assert result["success"] is False
        assert "Prompt is required" in result["error"]

    @patch.object(vgt, 'fal_key_is_configured', return_value=True)
    @patch.object(vgt, '_submit_fal_request')
    def test_aspect_ratio_mapping(self, mock_submit, mock_auth):
        """Test aspect ratio is mapped correctly."""
        mock_handler = MagicMock()
        mock_handler.status.return_value = {"status": "COMPLETED"}
        mock_handler.get.return_value = {"video": {"url": "https://test.mp4"}}
        mock_submit.return_value = mock_handler

        result = json.loads(vgt.video_generate_tool(
            prompt="test",
            aspect_ratio="9:16"
        ))

        assert result["success"] is True
        # Verify the payload sent to FAL
        call_args = mock_submit.call_args
        assert call_args[1]['arguments']['aspect_ratio'] == "9:16"

    @patch.object(vgt, 'fal_key_is_configured', return_value=False)
    @patch.object(vgt, '_resolve_managed_fal_gateway', return_value=None)
    def test_no_auth_raises_error(self, mock_gateway, mock_auth):
        """Test error when neither FAL_KEY nor managed gateway available."""
        result = json.loads(vgt.video_generate_tool(prompt="test"))
        assert result["success"] is False
        assert "FAL_KEY" in result["error"]

    @patch.object(vgt, 'fal_key_is_configured', return_value=False)
    @patch.object(vgt, '_resolve_managed_fal_gateway')
    @patch.object(vgt, '_submit_fal_request')
    def test_managed_gateway_fallback(self, mock_submit, mock_gateway, mock_auth):
        """Test managed gateway is used when FAL_KEY not set."""
        mock_gateway_instance = MagicMock()
        mock_gateway_instance.gateway_origin = "https://fal-queue-gateway.nousresearch.com"
        mock_gateway_instance.nous_user_token = "test-token"
        mock_gateway.return_value = mock_gateway_instance

        mock_handler = MagicMock()
        mock_handler.status.return_value = {"status": "COMPLETED"}
        mock_handler.get.return_value = {"video": {"url": "https://test.mp4"}}
        mock_submit.return_value = mock_handler

        result = json.loads(vgt.video_generate_tool(prompt="test"))
        assert result["success"] is True
        # Verify managed gateway was called
        mock_gateway.assert_called_once()


class TestModelResolution:
    """Test model selection logic."""

    @patch('hermes_cli.config.load_config')
    def test_default_model(self, mock_load_config):
        """Test default model is used when no config."""
        mock_load_config.return_value = {}
        model_id, meta = vgt._resolve_fal_video_model()
        assert model_id == vgt.DEFAULT_VIDEO_MODEL
        assert "Seedance" in meta["display"]

    @patch('hermes_cli.config.load_config')
    def test_custom_model_from_config(self, mock_load_config):
        """Test custom model from config.yaml."""
        mock_load_config.return_value = {
            "video_gen": {"model": "kling-video/v3/pro/text-to-video}
        }
        model_id, meta = vgt._resolve_fal_video_model()
        assert model_id == "kling-video/v3/pro/text-to-video"
        assert "Kling" in meta["display"]

    @patch('hermes_cli.config.load_config')
    def test_fallback_on_invalid_model(self, mock_load_config):
        """Test fallback to default on invalid model."""
        mock_load_config.return_value = {
            "video_gen": {"model": "invalid-model"}
        }
        model_id, meta = vgt._resolve_fal_video_model()
        assert model_id == vgt.DEFAULT_VIDEO_MODEL


class TestPayloadBuilder:
    """Test payload construction."""

    def test_basic_payload(self):
        """Test basic payload construction."""
        payload = vgt._build_fal_video_payload(
            "bytedance/seedance-2.0/text-to-video",
            prompt="A cat walking",
            duration=5,
            aspect_ratio="16:9"
        )
        assert payload["prompt"] == "A cat walking"
        assert payload["duration"] == 5
        assert payload["aspect_ratio"] == "16:9"

    def test_payload_filters_unsupported_keys(self):
        """Test that unsupported keys are filtered out."""
        payload = vgt._build_fal_video_payload(
            "kling-video/v3/pro/text-to-video"
            prompt="test",
            duration=5,
            negative_prompt="blurry"  # Kling doesn't support negative_prompt
        )
        assert "negative_prompt" not in payload
        assert payload["prompt"] == "test"

    def test_seed_included_when_provided(self):
        """Test seed is included when provided."""
        payload = vgt._build_fal_video_payload(
            "bytedance/seedance-2.0/text-to-video",
            prompt="test",
            duration=5,
            seed=42
        )
        assert payload["seed"] == 42

    def test_seed_excluded_when_none(self):
        """Test seed is excluded when None."""
        payload = vgt._build_fal_video_payload(
            "bytedance/seedance-2.0/text-to-video",
            prompt="test",
            duration=5,
            seed=None
        )
        assert "seed" not in payload
