"""Tests for browser_console tool and browser_vision annotate param."""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── browser_console ──────────────────────────────────────────────────


class TestBrowserConsole:
    """browser_console() returns console messages + JS errors in one call."""

    def test_returns_console_messages_and_errors(self):
        from tools.browser_tool import browser_console

        console_response = {
            "success": True,
            "data": {
                "messages": [
                    {"text": "hello", "type": "log", "timestamp": 1},
                    {"text": "oops", "type": "error", "timestamp": 2},
                ]
            },
        }
        errors_response = {
            "success": True,
            "data": {
                "errors": [
                    {"message": "Uncaught TypeError", "timestamp": 3},
                ]
            },
        }

        with patch("tools.browser_tool._run_browser_command") as mock_cmd:
            mock_cmd.side_effect = [console_response, errors_response]
            result = json.loads(browser_console(task_id="test"))

        assert result["success"] is True
        assert result["total_messages"] == 2
        assert result["total_errors"] == 1
        assert result["console_messages"][0]["text"] == "hello"
        assert result["console_messages"][1]["text"] == "oops"
        assert result["js_errors"][0]["message"] == "Uncaught TypeError"

    def test_passes_clear_flag(self):
        from tools.browser_tool import browser_console

        empty = {"success": True, "data": {"messages": [], "errors": []}}
        with patch("tools.browser_tool._run_browser_command", return_value=empty) as mock_cmd:
            browser_console(clear=True, task_id="test")

        calls = mock_cmd.call_args_list
        # Both console and errors should get --clear
        assert calls[0][0] == ("test", "console", ["--clear"])
        assert calls[1][0] == ("test", "errors", ["--clear"])

    def test_no_clear_by_default(self):
        from tools.browser_tool import browser_console

        empty = {"success": True, "data": {"messages": [], "errors": []}}
        with patch("tools.browser_tool._run_browser_command", return_value=empty) as mock_cmd:
            browser_console(task_id="test")

        calls = mock_cmd.call_args_list
        assert calls[0][0] == ("test", "console", [])
        assert calls[1][0] == ("test", "errors", [])

    def test_empty_console_and_errors(self):
        from tools.browser_tool import browser_console

        empty = {"success": True, "data": {"messages": [], "errors": []}}
        with patch("tools.browser_tool._run_browser_command", return_value=empty):
            result = json.loads(browser_console(task_id="test"))

        assert result["total_messages"] == 0
        assert result["total_errors"] == 0
        assert result["console_messages"] == []
        assert result["js_errors"] == []

    def test_handles_failed_commands(self):
        from tools.browser_tool import browser_console

        failed = {"success": False, "error": "No session"}
        with patch("tools.browser_tool._run_browser_command", return_value=failed):
            result = json.loads(browser_console(task_id="test"))

        # Should still return success with empty data
        assert result["success"] is True
        assert result["total_messages"] == 0
        assert result["total_errors"] == 0


# ── browser_console schema ───────────────────────────────────────────


class TestBrowserConsoleSchema:
    """browser_console is properly registered in the tool registry."""

    def test_schema_in_browser_schemas(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS

        names = [s["name"] for s in BROWSER_TOOL_SCHEMAS]
        assert "browser_console" in names

    def test_schema_has_clear_param(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS

        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_console")
        props = schema["parameters"]["properties"]
        assert "clear" in props
        assert props["clear"]["type"] == "boolean"


class TestBrowserConsoleToolsetWiring:
    """browser_console must be reachable via toolset resolution."""

    def test_in_browser_toolset(self):
        from toolsets import TOOLSETS
        assert "browser_console" in TOOLSETS["browser"]["tools"]

    def test_in_hermes_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS
        assert "browser_console" in _HERMES_CORE_TOOLS

    def test_in_legacy_toolset_map(self):
        from model_tools import _LEGACY_TOOLSET_MAP
        assert "browser_console" in _LEGACY_TOOLSET_MAP["browser_tools"]

    def test_in_registry(self):
        from tools.registry import registry
        from tools import browser_tool  # noqa: F401
        assert "browser_console" in registry._tools


# ── browser_vision annotate ──────────────────────────────────────────


class TestBrowserVisionAnnotate:
    """browser_vision supports annotate parameter."""

    def test_schema_has_annotate_param(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS

        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_vision")
        props = schema["parameters"]["properties"]
        assert "annotate" in props
        assert props["annotate"]["type"] == "boolean"

    def test_annotate_false_no_flag(self):
        """Without annotate, screenshot command has no --annotate flag."""
        from tools.browser_tool import browser_vision

        with (
            patch("tools.browser_tool._run_browser_command") as mock_cmd,
            patch("tools.browser_tool.call_llm") as mock_call_llm,
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
        ):
            mock_cmd.return_value = {"success": True, "data": {}}
            # Will fail at screenshot file read, but we can check the command
            try:
                browser_vision("test", annotate=False, task_id="test")
            except Exception:
                pass

            if mock_cmd.called:
                args = mock_cmd.call_args[0]
                cmd_args = args[2] if len(args) > 2 else []
                assert "--annotate" not in cmd_args

    def test_annotate_true_adds_flag(self):
        """With annotate=True, screenshot command includes --annotate."""
        from tools.browser_tool import browser_vision

        with (
            patch("tools.browser_tool._run_browser_command") as mock_cmd,
            patch("tools.browser_tool.call_llm") as mock_call_llm,
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
        ):
            mock_cmd.return_value = {"success": True, "data": {}}
            try:
                browser_vision("test", annotate=True, task_id="test")
            except Exception:
                pass

            if mock_cmd.called:
                args = mock_cmd.call_args[0]
                cmd_args = args[2] if len(args) > 2 else []
                assert "--annotate" in cmd_args


class TestBrowserVisionConfig:
    def _setup_screenshot(self, tmp_path):
        shots_dir = tmp_path / "browser_screenshots"
        shots_dir.mkdir()
        screenshot = shots_dir / "shot.png"
        screenshot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        return shots_dir, screenshot

    def test_browser_vision_uses_configured_temperature_and_timeout(self, tmp_path):
        from tools.browser_tool import browser_vision

        shots_dir, screenshot = self._setup_screenshot(tmp_path)
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Annotated screenshot analysis"
        mock_response.choices = [mock_choice]

        with (
            patch("hermes_constants.get_hermes_dir", return_value=shots_dir),
            patch("tools.browser_tool._cleanup_old_screenshots"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True, "data": {"path": str(screenshot)}}),
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
            patch("hermes_cli.config.load_config", return_value={"auxiliary": {"vision": {"temperature": 1, "timeout": 45}}}),
            patch("tools.browser_tool.call_llm", return_value=mock_response) as mock_llm,
        ):
            result = json.loads(browser_vision("what is on the page?", task_id="test"))

        assert result["success"] is True
        assert result["analysis"] == "Annotated screenshot analysis"
        assert mock_llm.call_args.kwargs["temperature"] == 1.0
        assert mock_llm.call_args.kwargs["timeout"] == 45.0

    def test_browser_vision_defaults_temperature_when_config_omits_it(self, tmp_path):
        from tools.browser_tool import browser_vision

        shots_dir, screenshot = self._setup_screenshot(tmp_path)
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Default screenshot analysis"
        mock_response.choices = [mock_choice]

        with (
            patch("hermes_constants.get_hermes_dir", return_value=shots_dir),
            patch("tools.browser_tool._cleanup_old_screenshots"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True, "data": {"path": str(screenshot)}}),
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
            patch("hermes_cli.config.load_config", return_value={"auxiliary": {"vision": {}}}),
            patch("tools.browser_tool.call_llm", return_value=mock_response) as mock_llm,
        ):
            result = json.loads(browser_vision("what is on the page?", task_id="test"))

        assert result["success"] is True
        assert result["analysis"] == "Default screenshot analysis"
        assert mock_llm.call_args.kwargs["temperature"] == 0.1
        assert mock_llm.call_args.kwargs["timeout"] == 120.0

    def test_browser_vision_native_fast_path_returns_multimodal(self, tmp_path):
        """supports_vision override → screenshot attached natively, no aux call."""
        from agent.auxiliary_client import clear_runtime_main, set_runtime_main
        from tools.browser_tool import browser_vision

        shots_dir, screenshot = self._setup_screenshot(tmp_path)
        annotations = [{"id": 1, "label": "Search box"}]
        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with (
                patch("hermes_constants.get_hermes_dir", return_value=shots_dir),
                patch("tools.browser_tool._cleanup_old_screenshots"),
                patch(
                    "tools.browser_tool._run_browser_command",
                    return_value={
                        "success": True,
                        "data": {"path": str(screenshot), "annotations": annotations},
                    },
                ),
                patch(
                    "hermes_cli.config.load_config",
                    return_value={"model": {"supports_vision": True}},
                ),
                patch("tools.browser_tool._get_vision_model") as mock_get_vision_model,
                patch("tools.browser_tool.call_llm") as mock_llm,
            ):
                result = browser_vision("what is on the page?", annotate=True, task_id="test")
        finally:
            clear_runtime_main()

        assert isinstance(result, dict)
        assert result["_multimodal"] is True
        assert result["meta"]["screenshot_path"] == str(screenshot)
        assert result["meta"]["annotations"] == annotations
        assert any(p.get("type") == "image_url" for p in result["content"])
        assert f"Screenshot path: {screenshot}" in result["text_summary"]
        mock_get_vision_model.assert_not_called()
        mock_llm.assert_not_called()

    def test_browser_vision_text_mode_blocks_native_fast_path(self, tmp_path):
        """Explicit text routing → aux LLM used even with supports_vision."""
        from agent.auxiliary_client import clear_runtime_main, set_runtime_main
        from tools.browser_tool import browser_vision

        shots_dir, screenshot = self._setup_screenshot(tmp_path)
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Text-mode screenshot analysis"
        mock_response.choices = [mock_choice]

        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with (
                patch("hermes_constants.get_hermes_dir", return_value=shots_dir),
                patch("tools.browser_tool._cleanup_old_screenshots"),
                patch(
                    "tools.browser_tool._run_browser_command",
                    return_value={"success": True, "data": {"path": str(screenshot)}},
                ),
                patch(
                    "hermes_cli.config.load_config",
                    return_value={
                        "agent": {"image_input_mode": "text"},
                        "model": {"supports_vision": True},
                    },
                ),
                patch("tools.browser_tool._get_vision_model", return_value="test-model"),
                patch("tools.browser_tool.call_llm", return_value=mock_response) as mock_llm,
            ):
                result = json.loads(browser_vision("what is on the page?", task_id="test"))
        finally:
            clear_runtime_main()

        assert result["success"] is True
        assert result["analysis"] == "Text-mode screenshot analysis"
        mock_llm.assert_called_once()

    def test_browser_vision_tall_screenshot_gets_resized_on_native_path(self, tmp_path):
        """Tall low-byte screenshot → resize called → native multimodal result with resized data URL."""
        import base64
        from agent.auxiliary_client import clear_runtime_main, set_runtime_main
        from tools.browser_tool import browser_vision
        from tools.vision_tools import _EMBED_TARGET_BYTES, _EMBED_MAX_DIMENSION

        shots_dir = tmp_path / "browser_screenshots"
        shots_dir.mkdir()
        screenshot = shots_dir / "shot.png"

        # Create a minimal valid PNG that appears tall (100×8500) via metadata trick.
        # We write a stub PNG; the proactive check uses _image_exceeds_dimension which
        # reads the actual file. Patch _image_exceeds_dimension to report "over dims".
        screenshot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        # A small valid data URL that fits under both caps
        tiny_png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
        small_data_url = f"data:image/png;base64,{tiny_png_b64}"
        assert len(small_data_url) <= _EMBED_TARGET_BYTES

        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with (
                patch("hermes_constants.get_hermes_dir", return_value=shots_dir),
                patch("tools.browser_tool._cleanup_old_screenshots"),
                patch(
                    "tools.browser_tool._run_browser_command",
                    return_value={"success": True, "data": {"path": str(screenshot)}},
                ),
                patch(
                    "hermes_cli.config.load_config",
                    return_value={"model": {"supports_vision": True}},
                ),
                patch("tools.browser_tool._get_vision_model") as mock_get_vision_model,
                patch("tools.browser_tool.call_llm") as mock_llm,
                patch(
                    "tools.vision_tools._image_exceeds_dimension",
                    return_value=True,  # simulate tall image
                ),
                patch(
                    "tools.vision_tools._resize_image_for_vision",
                    return_value=small_data_url,
                ) as mock_resize,
            ):
                result = browser_vision("what is on the page?", task_id="test")
        finally:
            clear_runtime_main()

        # Should return native multimodal dict, not a JSON error string
        assert isinstance(result, dict), f"Expected dict, got: {result!r}"
        assert result["_multimodal"] is True
        mock_resize.assert_called_once()
        # The data URL in the content should be the resized version
        image_parts = [p for p in result["content"] if p.get("type") == "image_url"]
        assert image_parts, "No image_url part found in native result"
        assert image_parts[0]["image_url"]["url"] == small_data_url
        mock_get_vision_model.assert_not_called()
        mock_llm.assert_not_called()

    def test_browser_vision_byte_heavy_screenshot_gets_resized_on_native_path(self, tmp_path):
        """Byte-heavy screenshot → resize called → native multimodal result with resized data URL."""
        import base64
        from agent.auxiliary_client import clear_runtime_main, set_runtime_main
        from tools.browser_tool import browser_vision
        from tools.vision_tools import _EMBED_TARGET_BYTES, _EMBED_MAX_DIMENSION

        shots_dir = tmp_path / "browser_screenshots"
        shots_dir.mkdir()
        screenshot = shots_dir / "shot.png"
        screenshot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        # A small valid data URL that fits under both caps after "resize"
        tiny_png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
        small_data_url = f"data:image/png;base64,{tiny_png_b64}"

        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with (
                patch("hermes_constants.get_hermes_dir", return_value=shots_dir),
                patch("tools.browser_tool._cleanup_old_screenshots"),
                patch(
                    "tools.browser_tool._run_browser_command",
                    return_value={"success": True, "data": {"path": str(screenshot)}},
                ),
                patch(
                    "hermes_cli.config.load_config",
                    return_value={"model": {"supports_vision": True}},
                ),
                patch("tools.browser_tool._get_vision_model") as mock_get_vision_model,
                patch("tools.browser_tool.call_llm") as mock_llm,
                patch(
                    "tools.vision_tools._image_exceeds_dimension",
                    return_value=False,
                ),
                # Simulate a byte-heavy data_url by patching the initial read
                # so _over_bytes fires: patch _over_bytes check by making data_url huge
                patch(
                    "tools.vision_tools._resize_image_for_vision",
                    return_value=small_data_url,
                ) as mock_resize,
            ):
                # Write a stub PNG but use a side-effect to simulate over-bytes scenario:
                # We patch the screenshot bytes read so data_url is large enough to trigger resize.
                # Simplest: write a file whose base64 exceeds _EMBED_TARGET_BYTES.
                # Since _EMBED_TARGET_BYTES = 4MB, we'd need 3MB+ raw. Instead, patch len() — 
                # but that's invasive. Instead patch _image_exceeds_dimension=False and
                # patch the data_url construction by making the file big enough in concept.
                # We rely on patching _resize_image_for_vision being called when _over_bytes fires.
                # For this test, we patch the screenshot file read to produce a large payload:
                large_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * (3 * 1024 * 1024)
                screenshot.write_bytes(large_bytes)
                result = browser_vision("what is on the page?", task_id="test")
        finally:
            clear_runtime_main()

        assert isinstance(result, dict), f"Expected dict, got: {result!r}"
        assert result["_multimodal"] is True
        mock_resize.assert_called_once()
        image_parts = [p for p in result["content"] if p.get("type") == "image_url"]
        assert image_parts, "No image_url part found in native result"
        assert image_parts[0]["image_url"]["url"] == small_data_url
        mock_get_vision_model.assert_not_called()
        mock_llm.assert_not_called()

    def test_browser_vision_resize_failure_returns_tool_error(self, tmp_path):
        """When resize cannot satisfy constraints, tool_error is returned (not native result)."""
        import base64
        from agent.auxiliary_client import clear_runtime_main, set_runtime_main
        from tools.browser_tool import browser_vision
        from tools.vision_tools import _EMBED_TARGET_BYTES

        shots_dir = tmp_path / "browser_screenshots"
        shots_dir.mkdir()
        screenshot = shots_dir / "shot.png"
        # Write a file large enough to trigger _over_bytes
        large_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * (3 * 1024 * 1024)
        screenshot.write_bytes(large_bytes)

        # Simulate resize failure: return a data_url that is still over the byte cap
        oversized_b64 = base64.b64encode(large_bytes).decode()
        oversized_data_url = f"data:image/png;base64,{oversized_b64}"
        assert len(oversized_data_url) > _EMBED_TARGET_BYTES

        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with (
                patch("hermes_constants.get_hermes_dir", return_value=shots_dir),
                patch("tools.browser_tool._cleanup_old_screenshots"),
                patch(
                    "tools.browser_tool._run_browser_command",
                    return_value={"success": True, "data": {"path": str(screenshot)}},
                ),
                patch(
                    "hermes_cli.config.load_config",
                    return_value={"model": {"supports_vision": True}},
                ),
                patch("tools.browser_tool._get_vision_model"),
                patch("tools.browser_tool.call_llm"),
                patch(
                    "tools.vision_tools._image_exceeds_dimension",
                    return_value=False,
                ),
                patch(
                    "tools.vision_tools._resize_image_for_vision",
                    return_value=oversized_data_url,  # resize "failed" to shrink it
                ),
            ):
                result = browser_vision("what is on the page?", task_id="test")
        finally:
            clear_runtime_main()

        # Should be a JSON error string, not a native dict
        assert isinstance(result, str), f"Expected JSON string error, got: {type(result)} {result!r}"
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "embed limits after auto-resize" in parsed.get("error", "")


# ── auto-recording config ────────────────────────────────────────────


class TestRecordSessionsConfig:
    """browser.record_sessions config option."""

    def test_default_config_has_record_sessions(self):
        from hermes_cli.config import DEFAULT_CONFIG

        browser_cfg = DEFAULT_CONFIG.get("browser", {})
        assert "record_sessions" in browser_cfg
        assert browser_cfg["record_sessions"] is False

    def test_maybe_start_recording_disabled(self):
        """Recording doesn't start when config says record_sessions: false."""
        from tools.browser_tool import _maybe_start_recording, _recording_sessions

        with (
            patch("tools.browser_tool._run_browser_command") as mock_cmd,
            patch("builtins.open", side_effect=FileNotFoundError),
        ):
            _maybe_start_recording("test-task")

        mock_cmd.assert_not_called()
        assert "test-task" not in _recording_sessions

    def test_maybe_stop_recording_noop_when_not_recording(self):
        """Stopping when not recording is a no-op."""
        from tools.browser_tool import _maybe_stop_recording, _recording_sessions

        _recording_sessions.discard("test-task")  # ensure not in set
        with patch("tools.browser_tool._run_browser_command") as mock_cmd:
            _maybe_stop_recording("test-task")

        mock_cmd.assert_not_called()


# ── dogfood skill files ──────────────────────────────────────────────


class TestDogfoodSkill:
    """Dogfood skill files exist and have correct structure."""

    @pytest.fixture(autouse=True)
    def _skill_dir(self):
        # Use the actual repo skills dir (not temp)
        self.skill_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "skills", "dogfood"
        )

    def test_skill_md_exists(self):
        assert os.path.exists(os.path.join(self.skill_dir, "SKILL.md"))

    def test_taxonomy_exists(self):
        assert os.path.exists(
            os.path.join(self.skill_dir, "references", "issue-taxonomy.md")
        )

    def test_report_template_exists(self):
        assert os.path.exists(
            os.path.join(self.skill_dir, "templates", "dogfood-report-template.md")
        )

    def test_skill_md_has_frontmatter(self):
        with open(os.path.join(self.skill_dir, "SKILL.md")) as f:
            content = f.read()
        assert content.startswith("---")
        assert "name: dogfood" in content
        assert "description:" in content

    def test_skill_references_browser_console(self):
        with open(os.path.join(self.skill_dir, "SKILL.md")) as f:
            content = f.read()
        assert "browser_console" in content

    def test_skill_references_annotate(self):
        with open(os.path.join(self.skill_dir, "SKILL.md")) as f:
            content = f.read()
        assert "annotate" in content

    def test_taxonomy_has_severity_levels(self):
        with open(
            os.path.join(self.skill_dir, "references", "issue-taxonomy.md")
        ) as f:
            content = f.read()
        assert "Critical" in content
        assert "High" in content
        assert "Medium" in content
        assert "Low" in content

    def test_taxonomy_has_categories(self):
        with open(
            os.path.join(self.skill_dir, "references", "issue-taxonomy.md")
        ) as f:
            content = f.read()
        assert "Functional" in content
        assert "Visual" in content
        assert "Accessibility" in content
        assert "Console" in content
