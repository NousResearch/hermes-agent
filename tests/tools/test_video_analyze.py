"""Tests for video_analyze tool in tools/vision_tools.py."""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.vision_tools import (
    _detect_video_mime_type,
    _video_to_base64_data_url,
    _handle_video_analyze,
    video_analyze_tool,
)

# ---------------------------------------------------------------------------
# _detect_video_mime_type
# ---------------------------------------------------------------------------

class TestDetectVideoMimeType:
    """Extension-based MIME detection for video files."""

    def test_mp4(self, tmp_path):
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

    def test_webm(self, tmp_path):
        p = tmp_path / "clip.webm"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/webm"

    def test_case_insensitive(self, tmp_path):
        p = tmp_path / "clip.MP4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

# ---------------------------------------------------------------------------
# _video_to_base64_data_url
# ---------------------------------------------------------------------------

class TestVideoToBase64DataUrl:
    """Base64 encoding of video files."""

    def test_produces_data_url(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = _video_to_base64_data_url(p)
        assert result.startswith("data:video/mp4;base64,")

    def test_default_mime_for_unknown_ext(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = _video_to_base64_data_url(p)
        # Falls back to video/mp4
        assert result.startswith("data:video/mp4;base64,")

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _handle_video_analyze handler
# ---------------------------------------------------------------------------

class TestHandleVideoAnalyze:
    """Tests for the registry handler wrapper."""

    def test_falls_back_to_vision_model_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "google/gemini-flash")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "ok"})
            asyncio.get_event_loop().run_until_complete(
                _handle_video_analyze({"video_url": "/tmp/test.mp4", "question": "test"})
            )
            args = mock_tool.call_args[0]
            assert args[2] == "google/gemini-flash"

# ---------------------------------------------------------------------------
# video_analyze_tool — integration-style tests with mocked LLM
# ---------------------------------------------------------------------------

class TestVideoAnalyzeTool:
    """Core video analysis function tests."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_local_file_success(self, tmp_path, monkeypatch):
        """Analyze a local video file — happy path."""
        video = tmp_path / "demo.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A short video showing a demo."

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock, return_value=mock_response):
            with patch("tools.vision_tools.extract_content_or_reasoning", return_value="A short video showing a demo."):
                result = self._run(video_analyze_tool(str(video), "What is this?"))

        data = json.loads(result)
        assert data["success"] is True
        assert "demo" in data["analysis"].lower()

    @pytest.mark.require_symlinks
    def test_local_file_read_guard_blocks_env_via_video_extension(self, tmp_path):
        """A .env file symlinked with a video extension must still be blocked.

        _detect_video_mime_type only checks the file extension, not file
        content, so without a read guard a model could point video_url at
        any credential-store file (renamed/symlinked to look like a video)
        and have its raw bytes base64-encoded and sent to the vision
        provider. Regression for the shared agent.file_safety chokepoint
        added to video_analyze_tool's local-file branch.
        """
        secret = tmp_path / ".env"
        secret.write_text("OPENAI_API_KEY=sk-super-secret\n", encoding="utf-8")
        disguised = tmp_path / "video.mp4"
        disguised.symlink_to(secret)

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool(str(disguised), "What is this?"))

        data = json.loads(result)
        assert data["success"] is False
        assert "secret-bearing environment file" in data["error"]
        mock_llm.assert_not_awaited()

    def test_unsupported_format(self, tmp_path):
        """Unsupported extension raises error."""
        video = tmp_path / "clip.flv"
        video.write_bytes(b"\x00" * 100)

        result = self._run(video_analyze_tool(str(video), "What is this?"))
        data = json.loads(result)
        assert data["success"] is False
        assert "unsupported video format" in data["analysis"].lower()

    def test_api_message_format(self, tmp_path):
        """Verify the message sent to LLM uses video_url content type."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "OK"
            return mock_response

        with patch("tools.vision_tools.async_call_llm", side_effect=capture_llm):
            with patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
                self._run(video_analyze_tool(str(video), "Describe this"))

        messages = captured_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "video_url"
        assert "video_url" in content[1]
        assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")
        # No hardcoded output cap — the aux client omits max_tokens so the
        # provider uses its full output budget (max-tokens-knob policy).
        assert "max_tokens" not in captured_kwargs

    def test_non_local_backend_reads_video_from_terminal_backend(self, tmp_path, monkeypatch):
        """Non-local terminal backends must not read local host video paths.

        The read routes through the shared media resolver
        (tools.image_source, ``permitted=("video",)``) which exec-reads the
        bytes inside the sandbox — so the analyzed video is the container's
        file, never the host's.
        """
        host_video = tmp_path / "clip.mp4"
        host_video.write_bytes(b"HOST-VIDEO")
        remote_bytes = b"REMOTE-SANDBOX-VIDEO"
        remote_b64 = base64.b64encode(remote_bytes).decode("ascii")
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        import tools.image_source as isrc
        import tools.terminal_tool as tt

        env_lookups = []

        def fake_get_active(task_id):
            env_lookups.append(task_id)
            return SimpleNamespace(
                execute=lambda cmd, **kw: {"returncode": 0, "output": remote_b64}
            )

        monkeypatch.setattr(tt, "ensure_task_env", lambda *a, **k: None)
        monkeypatch.setattr(isrc, "_get_active_env", fake_get_active)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "sandbox video"
            return mock_response

        with (
            patch("tools.vision_tools.async_call_llm", side_effect=capture_llm),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="sandbox video"),
        ):
            result = self._run(
                video_analyze_tool(str(host_video), "Describe this", task_id="task-123")
            )

        data = json.loads(result)
        assert data["success"] is True
        assert env_lookups == ["task-123"]
        video_url = captured_kwargs["messages"][0]["content"][1]["video_url"]["url"]
        uploaded_bytes = base64.b64decode(video_url.split(",", 1)[1])
        assert uploaded_bytes == remote_bytes
        assert uploaded_bytes != host_video.read_bytes()

# ---------------------------------------------------------------------------
# Toolset registration
# ---------------------------------------------------------------------------

class TestVideoToolsetRegistration:
    """Verify the tool is registered correctly."""

    def test_registered_in_video_toolset(self):
        from tools.registry import registry
        entry = registry.get_entry("video_analyze")
        assert entry is not None
        assert entry.toolset == "video"
        assert entry.is_async is True


# ---------------------------------------------------------------------------
# Native video & YouTube fast-path (Fixes #88141)
# ---------------------------------------------------------------------------

class TestNativeVideoFastPath:
    """Behavior-contract & E2E tests for native video routing (#88141)."""

    def test_native_fast_path_local_mp4_via_registry_dispatch(self, tmp_path, monkeypatch):
        """Real registry.dispatch returns a _multimodal envelope for a native Gemini 3+/4.x main model."""
        from tools.registry import registry

        video = tmp_path / "demo.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

        with patch("tools.vision_tools._should_use_native_video_fast_path", return_value=True):
            result = registry.dispatch(
                "video_analyze",
                {"video_url": str(video), "question": "What happens in this clip?"},
            )

        assert isinstance(result, dict)
        assert result.get("_multimodal") is True
        assert result["meta"]["native_video"] is True
        assert result["content"][1]["type"] == "video_url"
        assert result["content"][1]["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_native_youtube_fast_path_zero_download(self):
        """Gemini native route passes YouTube URL directly without downloading the webpage."""
        from tools.registry import registry

        with (
            patch("tools.vision_tools._should_use_native_video_fast_path", return_value=True),
            patch("tools.vision_tools._materialize_video", side_effect=AssertionError("must not download YouTube")),
        ):
            result = registry.dispatch(
                "video_analyze",
                {"video_url": "https://youtu.be/dQw4w9WgXcQ?t=15", "question": "Summarize"},
            )

        assert isinstance(result, dict)
        assert result.get("_multimodal") is True
        assert result["meta"]["native_youtube"] is True
        assert result["content"][1] == {
            "type": "video_url",
            "video_url": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        }

    def test_non_gemini_or_openai_compat_route_falls_back_to_auxiliary_for_youtube(self, monkeypatch):
        """Non-Gemini models (Claude/OpenAI) and OpenRouter/OpenAI-compat Vertex never take the native YouTube path."""
        from agent.image_routing import is_native_gemini_video_route

        assert is_native_gemini_video_route("anthropic", "claude-sonnet-4-6", {}) is False
        assert is_native_gemini_video_route("openai", "gpt-5", {}) is False
        assert is_native_gemini_video_route("openrouter", "google/gemini-3.8-flash", {}) is False
        assert is_native_gemini_video_route(
            "vertex",
            "gemini-3.8-flash",
            {"model": {"base_url": "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/endpoints/openapi"}},
        ) is False
        assert is_native_gemini_video_route("gemini", "gemini-2.5-flash", {}) is False
        assert is_native_gemini_video_route("gemini", "gemini-3.8-flash", {}) is True
        assert is_native_gemini_video_route("gemini", "gemini-4-neon", {}) is True
        assert is_native_gemini_video_route("gemini", "gemini-flash-latest", {}) is True
        assert is_native_gemini_video_route(
            "gemini",
            "gemini-3.8-flash",
            {"model": {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"}},
        ) is True
        assert is_native_gemini_video_route(
            "gemini",
            "gemini-3.8-flash",
            {},
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        ) is False

        with (
            patch("tools.vision_tools._should_use_native_video_fast_path", return_value=False),
            patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_aux,
        ):
            mock_aux.return_value = json.dumps({"success": True, "analysis": "aux summary"})
            out = asyncio.run(
                _handle_video_analyze({"video_url": "https://youtu.be/dQw4w9WgXcQ", "question": "Summarize"})
            )
        mock_aux.assert_awaited_once()
        assert json.loads(out)["analysis"] == "aux summary"

    def test_native_youtube_enforces_website_policy_and_host_feature(self):
        """Native YouTube fast-path respects both host_feature_enabled('media_processing') and check_website_access."""
        from tools.registry import registry

        with (
            patch("tools.vision_tools._should_use_native_video_fast_path", return_value=True),
            patch("tools.hosted_tool_policy.host_feature_enabled", return_value=False),
        ):
            res_hosted = registry.dispatch(
                "video_analyze",
                {"video_url": "https://youtu.be/dQw4w9WgXcQ", "question": "Summarize"},
            )
        assert isinstance(res_hosted, str)
        assert json.loads(res_hosted)["success"] is False

        with (
            patch("tools.vision_tools._should_use_native_video_fast_path", return_value=True),
            patch("tools.vision_tools.check_website_access", return_value={"message": "Blocked by website policy"}),
        ):
            res_blocked = registry.dispatch(
                "video_analyze",
                {"video_url": "https://youtu.be/dQw4w9WgXcQ", "question": "Summarize"},
            )
        assert isinstance(res_blocked, str)
        parsed = json.loads(res_blocked)
        assert parsed["success"] is False
        assert "Blocked by website policy" in parsed["error"]

    def test_html_disguised_as_mp4_rejected_on_both_native_and_auxiliary_paths(self, tmp_path):
        """HTML webpages masquerading as .mp4 fail loudly on both native and auxiliary paths."""
        from tools.registry import registry

        fake_mp4 = tmp_path / "webpage.mp4"
        fake_mp4.write_bytes(b"<!DOCTYPE html><html><body>Not a video</body></html>")

        with patch("tools.vision_tools._should_use_native_video_fast_path", return_value=True):
            native_res = registry.dispatch(
                "video_analyze",
                {"video_url": str(fake_mp4), "question": "Describe"},
            )
        assert isinstance(native_res, str)
        native_data = json.loads(native_res)
        assert native_data["success"] is False
        assert "html" in native_data["error"].lower()

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            aux_res = asyncio.run(video_analyze_tool(str(fake_mp4), "Describe"))
        aux_data = json.loads(aux_res)
        assert aux_data["success"] is False
        assert "html" in (aux_data.get("error") or aux_data.get("analysis") or "").lower()
        mock_llm.assert_not_awaited()

    def test_native_inline_video_enforces_20mb_cap(self, tmp_path):
        """Native inline videos exceeding _MAX_NATIVE_VIDEO_BASE64_BYTES (20 MB) fail loudly."""
        from tools.registry import registry

        clip = tmp_path / "big.mp4"
        clip.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

        with (
            patch("tools.vision_tools._should_use_native_video_fast_path", return_value=True),
            patch("tools.vision_tools._MAX_NATIVE_VIDEO_BASE64_BYTES", 10),
        ):
            res = registry.dispatch(
                "video_analyze",
                {"video_url": str(clip), "question": "Describe"},
            )
        assert isinstance(res, str)
        data = json.loads(res)
        assert data["success"] is False
        assert "too large" in data["error"].lower()

    def test_check_video_requirements_true_for_native_gemini_without_aux_client(self):
        """#47149 parity: video_analyze stays registered when main model supports native video even without an aux client."""
        from tools.vision_tools import check_video_requirements

        with (
            patch("tools.vision_tools._should_use_native_video_fast_path", return_value=True),
            patch("tools.vision_tools._has_aux_vision_client", return_value=False),
        ):
            assert check_video_requirements() is True

    def test_profile_isolation_a_b_a_video_input_mode_e2e(self, tmp_path, monkeypatch):
        """Real-import E2E across two isolated HERMES_HOME profiles (A -> B -> A) verifying agent.video_input_mode and 'text' alias."""
        from agent.runtime_scope import bind_profile_scope
        from tools.registry import registry

        home_a = tmp_path / "profile_a"
        home_b = tmp_path / "profile_b"
        home_a.mkdir()
        home_b.mkdir()
        (home_a / "config.yaml").write_text(
            "model:\n  provider: gemini\n  default: gemini-3.8-flash\nagent:\n  video_input_mode: native\n",
            encoding="utf-8",
        )
        (home_b / "config.yaml").write_text(
            "model:\n  provider: gemini\n  default: gemini-3.8-flash\nagent:\n  video_input_mode: text\n",
            encoding="utf-8",
        )
        clip = tmp_path / "sample.mp4"
        clip.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

        with (
            patch("agent.auxiliary_client._read_main_provider", return_value="gemini"),
            patch("agent.auxiliary_client._read_main_model", return_value="gemini-3.8-flash"),
            patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_aux,
        ):
            mock_aux.return_value = json.dumps({"success": True, "analysis": "aux-b"})

            with bind_profile_scope(home_a):
                res_a1 = registry.dispatch("video_analyze", {"video_url": str(clip), "question": "q"})
            assert isinstance(res_a1, dict) and res_a1.get("_multimodal") is True

            with bind_profile_scope(home_b):
                res_b = registry.dispatch("video_analyze", {"video_url": str(clip), "question": "q"})
            assert isinstance(res_b, str) and json.loads(res_b)["analysis"] == "aux-b"

            with bind_profile_scope(home_a):
                res_a2 = registry.dispatch("video_analyze", {"video_url": str(clip), "question": "q"})
            assert isinstance(res_a2, dict) and res_a2.get("_multimodal") is True

    def test_auto_mode_unmocked_models_dev_capabilities_and_history_budget(self, tmp_path, monkeypatch):
        """Default auto mode works end-to-end against a real seeded models.dev cache & model_overrides, real website_blocklist, and canonical YouTube repeat budget."""
        from agent import models_dev
        from agent.image_routing import decide_video_input_mode
        from agent.runtime_scope import bind_profile_scope
        from hermes_cli.config import set_config_value
        from tools.registry import registry
        from tools.vision_tools import check_video_requirements
        from tools.vision_tools_history_budget import _image_key, reserve_embed

        monkeypatch.delenv("AUXILIARY_VIDEO_MODEL", raising=False)
        home = tmp_path / "seeded_home"
        home.mkdir()
        (home / "config.yaml").write_text(
            "model:\n  provider: gemini\n  default: gemini-3.8-flash\n"
            "security:\n  website_blocklist:\n    enabled: true\n    domains:\n      - youtu.be\n",
            encoding="utf-8",
        )
        fake_catalog = {
            "google": {
                "models": {
                    "gemini-3.8-flash": {"modalities": {"input": ["text", "image", "video"], "output": ["text"]}, "tool_call": True},
                    "gemini-text-only": {"modalities": {"input": ["text"], "output": ["text"]}, "tool_call": True},
                }
            }
        }
        monkeypatch.setattr(models_dev, "_models_dev_cache", fake_catalog)
        monkeypatch.setattr(models_dev, "_models_dev_cache_time", float("inf"))

        with bind_profile_scope(home):
            # Real unmocked models.dev catalog lookup -> native vs auxiliary
            assert decide_video_input_mode("gemini", "gemini-3.8-flash", {}, allow_network=False) == "native"
            assert decide_video_input_mode(
                "gemini",
                "gemini-3.8-flash",
                {"auxiliary": {"vision": {"provider": "", "model": "auto"}}},
                allow_network=False,
            ) == "native"
            assert decide_video_input_mode("gemini", "gemini-text-only", {}, allow_network=False) == "auxiliary"
            # Strict bool override ("false" must not coerce to True)
            caps_overridden = models_dev.get_model_capabilities(
                "gemini",
                "gemini-3.8-flash",
                config={"model_overrides": {"google": {"gemini-3.8-flash": {"supports_video": "false"}}}},
            )
            assert caps_overridden is not None and caps_overridden.supports_video is False

            # Real check_video_requirements without mocking _should_use_native_video_fast_path
            with patch("tools.vision_tools._has_aux_vision_client", return_value=False):
                assert check_video_requirements() is True

            # Real website_blocklist blocks raw youtu.be URL before normalization bypasses it
            blocked_res = registry.dispatch(
                "video_analyze",
                {"video_url": "https://youtu.be/dQw4w9WgXcQ", "question": "Summarize"},
            )
            assert isinstance(blocked_res, str)
            assert json.loads(blocked_res)["success"] is False

            # file:// URI with spaces + data:video rejection
            spaced = tmp_path / "my clip.mp4"
            spaced.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
            file_uri_res = registry.dispatch(
                "video_analyze",
                {"video_url": f"file://{spaced.as_posix().replace(' ', '%20')}", "question": "q"},
            )
            assert isinstance(file_uri_res, dict) and file_uri_res.get("_multimodal") is True

            data_uri_res = registry.dispatch(
                "video_analyze",
                {"video_url": "data:video/mp4;base64,AAAA", "question": "q"},
            )
            assert isinstance(data_uri_res, str) and json.loads(data_uri_res)["success"] is False

            # config set round-trip for agent.video_input_mode
            from hermes_cli.config import load_config
            set_config_value("agent.video_input_mode", "auxiliary")
            assert load_config()["agent"]["video_input_mode"] == "auxiliary"

            # Canonical YouTube keys + video refusal wording + profile-scoped _repeat_counts
            assert _image_key("https://youtu.be/AbCdEfGhIjK?t=10") == _image_key("https://www.youtube.com/watch?v=AbCdEfGhIjK")
            with patch("tools.vision_tools_history_budget.resolve_repeat_cap", return_value=1):
                ref1, key1 = reserve_embed("https://youtu.be/AbCdEfGhIjK", tool_name="video_analyze", media_label="video")
                assert ref1 is None and key1 is not None
                ref2, key2 = reserve_embed("https://www.youtube.com/watch?v=AbCdEfGhIjK", tool_name="video_analyze", media_label="video")
                assert ref2 is not None and key2 is None
                assert "video_analyze refused: this video" in ref2



