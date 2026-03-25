"""
Tests for video_generation_tool.py

Covers:
- Parameter validation (prompt, model, duration, aspect_ratio)
- _extract_video_url for different FAL.ai response shapes
- _build_arguments per model family
- check_video_generation_requirements
- video_generate_tool success path (mocked fal_client)
- video_generate_tool failure paths (missing key, bad response, download error)
- Registry registration
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to isolate fal_client and urllib from the real network
# ---------------------------------------------------------------------------

def _make_fal_mock(video_url: str):
    """Return a mock fal_client whose submit().get() yields a video URL."""
    mock_handler = MagicMock()
    mock_handler.get.return_value = {"video": {"url": video_url, "width": 1280, "height": 720}}

    mock_fal = MagicMock()
    mock_fal.submit.return_value = mock_handler
    return mock_fal


# ---------------------------------------------------------------------------
# Unit tests — internal helpers (no network)
# ---------------------------------------------------------------------------

class TestExtractVideoUrl:
    def setup_method(self):
        from tools.video_generation_tool import _extract_video_url
        self._extract = _extract_video_url

    def test_single_video_object(self):
        result = {"video": {"url": "https://cdn.fal.ai/video.mp4"}}
        assert self._extract(result) == "https://cdn.fal.ai/video.mp4"

    def test_videos_list(self):
        result = {"videos": [{"url": "https://cdn.fal.ai/v1.mp4"}, {"url": "https://cdn.fal.ai/v2.mp4"}]}
        assert self._extract(result) == "https://cdn.fal.ai/v1.mp4"

    def test_none_result(self):
        assert self._extract(None) is None

    def test_empty_dict(self):
        assert self._extract({}) is None

    def test_empty_videos_list(self):
        assert self._extract({"videos": []}) is None

    def test_video_without_url_key(self):
        result = {"video": {"width": 1280, "height": 720}}
        assert self._extract(result) is None


class TestBuildArguments:
    def setup_method(self):
        from tools.video_generation_tool import _build_arguments
        self._build = _build_arguments

    def test_kling_includes_duration_and_aspect(self):
        args = self._build("kling", "a dog running", 5, "landscape", None)
        assert args["prompt"] == "a dog running"
        assert args["duration"] == "5"
        assert args["aspect_ratio"] == "16:9"
        assert "negative_prompt" not in args

    def test_kling_10s_duration(self):
        args = self._build("kling", "sunset timelapse", 10, "portrait", None)
        assert args["duration"] == "10"
        assert args["aspect_ratio"] == "9:16"

    def test_kling_negative_prompt_included(self):
        args = self._build("kling", "a scene", 5, "square", "blurry, shaky")
        assert args["negative_prompt"] == "blurry, shaky"
        assert args["aspect_ratio"] == "1:1"

    def test_luma_has_aspect_and_loop(self):
        args = self._build("luma", "a galaxy", 5, "landscape", None)
        assert args["aspect_ratio"] == "16:9"
        assert args["loop"] is False
        assert "duration" not in args

    def test_minimax_prompt_only(self):
        args = self._build("minimax", "waves crashing", 5, "landscape", None)
        assert args["prompt"] == "waves crashing"
        assert "aspect_ratio" not in args
        assert "duration" not in args

    def test_prompt_stripped(self):
        args = self._build("kling", "  spaced prompt  ", 5, "landscape", None)
        assert args["prompt"] == "spaced prompt"


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

class TestCheckRequirements:
    def test_returns_false_without_fal_key(self):
        from tools.video_generation_tool import check_video_generation_requirements
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FAL_KEY", None)
            assert check_video_generation_requirements() is False

    def test_returns_true_with_fal_key(self):
        from tools.video_generation_tool import check_video_generation_requirements
        with patch.dict(os.environ, {"FAL_KEY": "test-key"}):
            assert check_video_generation_requirements() is True

    def test_returns_false_when_fal_client_missing(self):
        from tools.video_generation_tool import check_video_generation_requirements
        with patch.dict(os.environ, {"FAL_KEY": "test-key"}):
            with patch.dict(sys.modules, {"fal_client": None}):
                assert check_video_generation_requirements() is False


# ---------------------------------------------------------------------------
# video_generate_tool — validation paths (no network)
# ---------------------------------------------------------------------------

class TestVideoGenerateToolValidation:
    def _call(self, **kwargs):
        from tools.video_generation_tool import video_generate_tool
        return json.loads(video_generate_tool(**kwargs))

    def test_empty_prompt_fails(self):
        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            result = self._call(prompt="")
        assert result["success"] is False
        assert result["video_path"] is None

    def test_whitespace_prompt_fails(self):
        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            result = self._call(prompt="   ")
        assert result["success"] is False

    def test_missing_fal_key_fails(self):
        env = {k: v for k, v in os.environ.items() if k != "FAL_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = self._call(prompt="a scene")
        assert result["success"] is False

    def test_invalid_model_fails(self):
        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            result = self._call(prompt="a scene", model="unknown_model")
        assert result["success"] is False

    def test_invalid_duration_defaults_silently(self):
        """Invalid duration should fall back to 5 (no crash)."""
        fake_fal = _make_fal_mock("https://cdn.fal.ai/video.mp4")
        fake_content = b"\x00" * 100

        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            with patch("tools.video_generation_tool.fal_client", fake_fal):
                with patch("tools.video_generation_tool.urllib.request.urlretrieve") as mock_dl:
                    # Simulate download writing bytes to the temp file
                    def fake_download(url, path):
                        with open(path, "wb") as f:
                            f.write(fake_content)
                    mock_dl.side_effect = fake_download

                    result = self._call(prompt="a scene", duration=999)
        assert result["success"] is True

    def test_invalid_aspect_ratio_defaults_silently(self):
        fake_fal = _make_fal_mock("https://cdn.fal.ai/video.mp4")
        fake_content = b"\x00" * 100

        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            with patch("tools.video_generation_tool.fal_client", fake_fal):
                with patch("tools.video_generation_tool.urllib.request.urlretrieve") as mock_dl:
                    def fake_download(url, path):
                        with open(path, "wb") as f:
                            f.write(fake_content)
                    mock_dl.side_effect = fake_download

                    result = self._call(prompt="a scene", aspect_ratio="widescreen_banana")
        assert result["success"] is True


# ---------------------------------------------------------------------------
# video_generate_tool — success path (mocked fal_client + download)
# ---------------------------------------------------------------------------

class TestVideoGenerateToolSuccess:
    VIDEO_URL = "https://cdn.fal.ai/generated/video.mp4"
    FAKE_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50

    def _run(self, prompt="a cat walking", model="kling", duration=5, aspect_ratio="landscape"):
        from tools.video_generation_tool import video_generate_tool

        fake_fal = _make_fal_mock(self.VIDEO_URL)

        def fake_download(url, path):
            with open(path, "wb") as f:
                f.write(self.FAKE_BYTES)

        with patch.dict(os.environ, {"FAL_KEY": "test-key"}):
            with patch("tools.video_generation_tool.fal_client", fake_fal):
                with patch("tools.video_generation_tool.urllib.request.urlretrieve", side_effect=fake_download):
                    raw = video_generate_tool(
                        prompt=prompt, model=model,
                        duration=duration, aspect_ratio=aspect_ratio,
                    )
        return json.loads(raw), fake_fal

    def test_returns_success_true(self):
        result, _ = self._run()
        assert result["success"] is True

    def test_returns_video_path(self):
        result, _ = self._run()
        assert result["video_path"] is not None
        assert result["video_path"].endswith(".mp4")

    def test_returns_video_url(self):
        result, _ = self._run()
        assert result["video_url"] == self.VIDEO_URL

    def test_returns_model_name(self):
        result, _ = self._run(model="luma")
        assert result["model"] == "luma"

    def test_returns_duration(self):
        result, _ = self._run(duration=10)
        assert result["duration_seconds"] == 10

    def test_fal_submit_called_with_kling_model_id(self):
        _, fake_fal = self._run(model="kling")
        call_args = fake_fal.submit.call_args
        assert call_args[0][0] == "fal-ai/kling-video/v2/master/text-to-video"

    def test_fal_submit_called_with_luma_model_id(self):
        _, fake_fal = self._run(model="luma")
        call_args = fake_fal.submit.call_args
        assert call_args[0][0] == "fal-ai/luma-dream-machine"

    def test_fal_submit_called_with_minimax_model_id(self):
        _, fake_fal = self._run(model="minimax")
        call_args = fake_fal.submit.call_args
        assert call_args[0][0] == "fal-ai/minimax/video-01-live"

    def test_video_path_is_local_file(self):
        result, _ = self._run()
        # File was created during the mocked download
        assert os.path.exists(result["video_path"])
        # Clean up
        os.unlink(result["video_path"])

    def test_model_case_insensitive(self):
        from tools.video_generation_tool import video_generate_tool

        fake_fal = _make_fal_mock(self.VIDEO_URL)

        def fake_download(url, path):
            with open(path, "wb") as f:
                f.write(self.FAKE_BYTES)

        with patch.dict(os.environ, {"FAL_KEY": "test-key"}):
            with patch("tools.video_generation_tool.fal_client", fake_fal):
                with patch("tools.video_generation_tool.urllib.request.urlretrieve", side_effect=fake_download):
                    raw = video_generate_tool(prompt="test", model="KLING")
        result = json.loads(raw)
        assert result["success"] is True
        if result["video_path"]:
            os.unlink(result["video_path"])


# ---------------------------------------------------------------------------
# video_generate_tool — failure paths
# ---------------------------------------------------------------------------

class TestVideoGenerateToolFailures:
    def test_fal_api_raises_exception(self):
        from tools.video_generation_tool import video_generate_tool

        mock_fal = MagicMock()
        mock_fal.submit.side_effect = RuntimeError("API unavailable")

        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            with patch("tools.video_generation_tool.fal_client", mock_fal):
                result = json.loads(video_generate_tool(prompt="a scene"))

        assert result["success"] is False
        assert result["video_path"] is None

    def test_no_video_url_in_response(self):
        from tools.video_generation_tool import video_generate_tool

        mock_handler = MagicMock()
        mock_handler.get.return_value = {"images": []}  # wrong key
        mock_fal = MagicMock()
        mock_fal.submit.return_value = mock_handler

        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            with patch("tools.video_generation_tool.fal_client", mock_fal):
                result = json.loads(video_generate_tool(prompt="a scene"))

        assert result["success"] is False

    def test_download_failure_handled(self):
        from tools.video_generation_tool import video_generate_tool

        mock_handler = MagicMock()
        mock_handler.get.return_value = {"video": {"url": "https://cdn.fal.ai/v.mp4"}}
        mock_fal = MagicMock()
        mock_fal.submit.return_value = mock_handler

        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            with patch("tools.video_generation_tool.fal_client", mock_fal):
                with patch("tools.video_generation_tool.urllib.request.urlretrieve",
                           side_effect=OSError("network error")):
                    result = json.loads(video_generate_tool(prompt="a scene"))

        assert result["success"] is False


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

class TestRegistryRegistration:
    def test_video_generate_registered(self):
        from tools.registry import registry
        import tools.video_generation_tool  # noqa: F401 — ensure module loaded

        entry = registry._tools.get("video_generate")
        assert entry is not None

    def test_toolset_is_video_gen(self):
        from tools.registry import registry
        import tools.video_generation_tool  # noqa: F401
        assert registry._tools["video_generate"].toolset == "video_gen"

    def test_requires_fal_key(self):
        from tools.registry import registry
        import tools.video_generation_tool  # noqa: F401
        assert "FAL_KEY" in registry._tools["video_generate"].requires_env

    def test_is_sync(self):
        from tools.registry import registry
        import tools.video_generation_tool  # noqa: F401
        assert registry._tools["video_generate"].is_async is False

    def test_schema_has_required_prompt(self):
        from tools.registry import registry
        import tools.video_generation_tool  # noqa: F401
        schema = registry._tools["video_generate"].schema
        assert "prompt" in schema["parameters"]["required"]

    def test_schema_model_enum(self):
        from tools.registry import registry
        import tools.video_generation_tool  # noqa: F401
        schema = registry._tools["video_generate"].schema
        model_enum = schema["parameters"]["properties"]["model"]["enum"]
        assert set(model_enum) == {"kling", "luma", "minimax"}


# ---------------------------------------------------------------------------
# _handle_video_generate — registry handler argument mapping
# ---------------------------------------------------------------------------

class TestHandleVideoGenerate:
    """_handle_video_generate maps args dict → video_generate_tool kwargs."""

    FAKE_BYTES = b"\x00" * 50
    VIDEO_URL = "https://cdn.fal.ai/v.mp4"

    def _dispatch(self, args: dict):
        from tools.video_generation_tool import _handle_video_generate

        fake_fal = _make_fal_mock(self.VIDEO_URL)

        def fake_download(url, path):
            with open(path, "wb") as f:
                f.write(self.FAKE_BYTES)

        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            with patch("tools.video_generation_tool.fal_client", fake_fal):
                with patch("tools.video_generation_tool.urllib.request.urlretrieve",
                           side_effect=fake_download):
                    return json.loads(_handle_video_generate(args)), fake_fal

    def test_missing_prompt_returns_error(self):
        from tools.video_generation_tool import _handle_video_generate
        result = json.loads(_handle_video_generate({}))
        assert "error" in result

    def test_empty_prompt_returns_error(self):
        from tools.video_generation_tool import _handle_video_generate
        result = json.loads(_handle_video_generate({"prompt": ""}))
        assert "error" in result

    def test_prompt_forwarded(self):
        result, fake_fal = self._dispatch({"prompt": "a volcano erupting"})
        assert result["success"] is True
        call_kwargs = fake_fal.submit.call_args[1]["arguments"]
        assert call_kwargs["prompt"] == "a volcano erupting"

    def test_model_forwarded(self):
        result, fake_fal = self._dispatch({"prompt": "ocean", "model": "luma"})
        assert result["model"] == "luma"

    def test_duration_forwarded_as_int(self):
        """Handler must cast duration to int (model may pass it as string)."""
        result, fake_fal = self._dispatch({"prompt": "sky", "duration": "10"})
        assert result["success"] is True
        # Kling receives duration as string "10" in API args
        call_kwargs = fake_fal.submit.call_args[1]["arguments"]
        assert call_kwargs["duration"] == "10"

    def test_aspect_ratio_forwarded(self):
        result, fake_fal = self._dispatch({"prompt": "forest", "aspect_ratio": "portrait"})
        assert result["success"] is True
        call_kwargs = fake_fal.submit.call_args[1]["arguments"]
        assert call_kwargs["aspect_ratio"] == "9:16"

    def test_negative_prompt_forwarded(self):
        result, fake_fal = self._dispatch({
            "prompt": "a city", "negative_prompt": "blurry, shaky"
        })
        assert result["success"] is True
        call_kwargs = fake_fal.submit.call_args[1]["arguments"]
        assert call_kwargs["negative_prompt"] == "blurry, shaky"

    def test_defaults_applied_when_optional_args_absent(self):
        result, fake_fal = self._dispatch({"prompt": "clouds"})
        assert result["success"] is True
        assert result["model"] == "kling"
        assert result["duration_seconds"] == 5

    def cleanup_paths(self, result):
        path = result.get("video_path")
        if path and os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# _download_video — extension / suffix logic
# ---------------------------------------------------------------------------

class TestDownloadVideoExtension:
    """_download_video picks the correct temp file suffix from the URL."""

    FAKE_BYTES = b"\x00" * 50

    def _download(self, url: str) -> str:
        from tools.video_generation_tool import _download_video
        with patch("tools.video_generation_tool.urllib.request.urlretrieve") as mock_dl:
            def fake_retrieve(u, path):
                with open(path, "wb") as f:
                    f.write(self.FAKE_BYTES)
            mock_dl.side_effect = fake_retrieve
            return _download_video(url)

    def test_mp4_url_gives_mp4_suffix(self):
        path = self._download("https://cdn.fal.ai/output/video.mp4")
        assert path.endswith(".mp4")
        os.unlink(path)

    def test_webm_url_gives_webm_suffix(self):
        path = self._download("https://cdn.fal.ai/output/video.webm")
        assert path.endswith(".webm")
        os.unlink(path)

    def test_mov_url_gives_mov_suffix(self):
        path = self._download("https://cdn.fal.ai/output/clip.mov")
        assert path.endswith(".mov")
        os.unlink(path)

    def test_avi_url_gives_avi_suffix(self):
        path = self._download("https://cdn.fal.ai/output/clip.avi")
        assert path.endswith(".avi")
        os.unlink(path)

    def test_unknown_extension_defaults_to_mp4(self):
        path = self._download("https://cdn.fal.ai/output/video.ts")
        assert path.endswith(".mp4")
        os.unlink(path)

    def test_no_extension_defaults_to_mp4(self):
        path = self._download("https://cdn.fal.ai/output/abc123")
        assert path.endswith(".mp4")
        os.unlink(path)

    def test_query_params_ignored(self):
        """URL with ?token=... should still parse the extension correctly."""
        path = self._download("https://cdn.fal.ai/output/video.mp4?token=xyz&expires=99")
        assert path.endswith(".mp4")
        os.unlink(path)

    def test_file_has_hermes_prefix(self):
        path = self._download("https://cdn.fal.ai/output/video.mp4")
        assert os.path.basename(path).startswith("hermes_video_")
        os.unlink(path)


# ---------------------------------------------------------------------------
# Integration — registry.dispatch end-to-end
# ---------------------------------------------------------------------------

class TestRegistryDispatchIntegration:
    """Full pipeline: registry.dispatch('video_generate', args) → JSON result."""

    FAKE_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50
    VIDEO_URL = "https://cdn.fal.ai/integration/video.mp4"

    def _dispatch(self, args: dict):
        from tools.registry import registry
        import tools.video_generation_tool  # noqa: F401 — ensure registered

        fake_fal = _make_fal_mock(self.VIDEO_URL)

        def fake_download(url, path):
            with open(path, "wb") as f:
                f.write(self.FAKE_BYTES)

        with patch.dict(os.environ, {"FAL_KEY": "key"}):
            with patch("tools.video_generation_tool.fal_client", fake_fal):
                with patch("tools.video_generation_tool.urllib.request.urlretrieve",
                           side_effect=fake_download):
                    raw = registry.dispatch("video_generate", args)
        return json.loads(raw)

    def test_dispatch_returns_success(self):
        result = self._dispatch({"prompt": "a waterfall at dawn"})
        assert result["success"] is True

    def test_dispatch_returns_video_path(self):
        result = self._dispatch({"prompt": "a waterfall at dawn"})
        assert result["video_path"] is not None
        assert os.path.exists(result["video_path"])
        os.unlink(result["video_path"])

    def test_dispatch_unknown_tool_returns_error(self):
        from tools.registry import registry
        result = json.loads(registry.dispatch("nonexistent_tool", {}))
        assert "error" in result

    def test_dispatch_missing_prompt_returns_error(self):
        result = self._dispatch({})
        assert "error" in result

    def test_dispatch_all_params_forwarded(self):
        result = self._dispatch({
            "prompt": "a volcano",
            "model": "kling",
            "duration": 10,
            "aspect_ratio": "portrait",
        })
        assert result["success"] is True
        assert result["model"] == "kling"
        assert result["duration_seconds"] == 10
        if result.get("video_path"):
            os.unlink(result["video_path"])
