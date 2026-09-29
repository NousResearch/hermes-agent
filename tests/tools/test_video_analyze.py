"""Tests for video_analyze tool in tools/vision_tools.py."""

import asyncio
import base64
import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.vision_tools import (
    _detect_video_mime_type,
    _ensure_ytdlp_available,
    _resolve_video_provider_model,
    _ytdlp_command,
    _ytdlp_js_runtime_args,
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

    def test_page_url_uses_ytdlp_and_direct_gemini(self, tmp_path):
        """A YouTube watch page is extracted, not downloaded as HTML."""
        extracted = tmp_path / "extracted.mp4"
        extracted.write_bytes(b"\x00" * 100)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A person demonstrates a task."

        with patch("tools.url_safety.async_is_safe_url", new_callable=AsyncMock, return_value=True), \
             patch("tools.vision_tools._ensure_ytdlp_available") as ensure_ytdlp, \
             patch("tools.vision_tools._download_video_via_ytdlp", new_callable=AsyncMock, return_value=extracted) as download, \
             patch("tools.vision_tools._download_media", new_callable=AsyncMock) as http_download, \
             patch("tools.vision_tools._resolve_video_provider_model", return_value=("gemini", "gemini-3-flash-preview")), \
             patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock, return_value=mock_response) as llm, \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="A person demonstrates a task."):
            result = self._run(video_analyze_tool(
                "https://youtu.be/GhSdkmMt4LE?feature=shared", "What happens?"
            ))

        data = json.loads(result)
        assert data["success"] is True
        ensure_ytdlp.assert_called_once_with()
        download.assert_awaited_once()
        http_download.assert_not_awaited()
        assert llm.await_args.kwargs["provider"] == "gemini"
        assert llm.await_args.kwargs["model"] == "gemini-3-flash-preview"

    def test_page_url_to_private_address_is_refused_before_ytdlp(self):
        """SSRF guard: a page URL resolving to a private address never reaches yt-dlp."""
        with patch("tools.url_safety.async_is_safe_url", new_callable=AsyncMock, return_value=False), \
             patch("tools.vision_tools._ensure_ytdlp_available") as ensure_ytdlp, \
             patch("tools.vision_tools._download_video_via_ytdlp", new_callable=AsyncMock) as download, \
             patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as llm:
            result = self._run(video_analyze_tool(
                "http://169.254.169.254/latest/meta-data/", "What happens?"
            ))

        data = json.loads(result)
        assert data["success"] is False
        ensure_ytdlp.assert_not_called()
        download.assert_not_awaited()
        llm.assert_not_awaited()


class TestVideoDependenciesAndRouting:
    def test_ytdlp_child_can_import_from_lazy_target(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy-packages"
        package = target / "yt_dlp"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "__main__.py").write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['YTDLP_TEST_RESULT']).write_text("
            "json.dumps(sys.argv[1:]), encoding='utf-8')\n",
            encoding="utf-8",
        )

        fake_module = ModuleType("yt_dlp")
        fake_module.__file__ = str(package / "__init__.py")
        monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

        result_path = tmp_path / "result.json"
        child_env = os.environ.copy()
        child_env.pop("PYTHONPATH", None)
        child_env["YTDLP_TEST_RESULT"] = str(result_path)
        subprocess.run(
            [*_ytdlp_command(), "--probe", "value"],
            check=True,
            capture_output=True,
            text=True,
            env=child_env,
            stdin=subprocess.DEVNULL,
        )

        assert json.loads(result_path.read_text(encoding="utf-8")) == [
            "--probe", "value"
        ]

    def test_ytdlp_enables_installed_node_runtime(self):
        def which(executable):
            return "/opt/node/bin/node" if executable == "node" else None

        with patch("shutil.which", side_effect=which):
            assert _ytdlp_js_runtime_args() == [
                "--js-runtimes", "node:/opt/node/bin/node"
            ]

    def test_missing_ytdlp_installs_the_video_extra(self):
        real_import = __import__

        def import_without_ytdlp(name, *args, **kwargs):
            if name == "yt_dlp":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_ytdlp), \
             patch("pm.ensure_import") as ensure:
            _ensure_ytdlp_available()

        ensure.assert_called_once_with("video")

    def test_video_extra_anchor_is_ytdlp(self):
        from pm.extras import ANCHORS

        assert ANCHORS["video"] == "yt_dlp"

    def test_auto_routing_falls_through_to_direct_gemini(self):
        values = {
            ("auxiliary", "vision", "video_provider"): "auto",
            ("auxiliary", "vision", "video_model"): "",
            ("model", "provider"): "openai-codex",
            ("model", "default"): "gpt-5.6-sol",
        }

        def config_value(*path, default=""):
            return values.get(path, default)

        def resolve(*, provider, model):
            if provider == "gemini":
                return provider, object(), model
            return provider, None, None

        with patch("tools.vision_tools._cfg_get_safe", side_effect=config_value), \
             patch("agent.auxiliary_client.resolve_vision_provider_client", side_effect=resolve):
            assert _resolve_video_provider_model() == (
                "gemini", "gemini-3-flash-preview"
            )


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
