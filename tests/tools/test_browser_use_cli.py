"""Tests for the Browser Use CLI 3.0 backend (tools/browser_use_cli.py).

Covers the three seams the integration relies on:

* Mode detection — ``browser.backend: browser-use`` in config (set via the
  ``hermes tools`` picker); off by default.
* Tool-surface swap — when the mode is on, ``check_browser_requirements``
  returns False so every legacy ``browser_*`` tool (including
  browser_cdp/browser_dialog, whose check_fns funnel through it) is hidden,
  and ``browser_exec`` is advertised instead.
* ``browser_exec`` execution — code is piped on stdin, ``session`` becomes
  ``BU_NAME``, bad session names and a missing CLI produce actionable errors.
"""
import json
import os
import stat
import time

import pytest

import tools.browser_use_cli as bu_cli


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("BU_NAME", raising=False)
    monkeypatch.delenv("BU_AUTOSPAWN", raising=False)
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    yield


def _fake_cli(tmp_path, body):
    """Write an executable fake browser-use CLI and return its path."""
    script = tmp_path / "browser-use"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


class TestModeDetection:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_config_opt_in(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "browser-use"}},
        )
        assert bu_cli.is_browser_use_cli_mode() is True

    def test_other_backend_value_is_not_cli_mode(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "something-else"}},
        )
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_config_read_failure_fails_safe(self, monkeypatch):
        def boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("hermes_cli.config.read_raw_config", boom)
        assert bu_cli.is_browser_use_cli_mode() is False


class TestToolSurfaceSwap:
    def test_legacy_browser_tools_hidden_in_cli_mode(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.setattr(browser_tool, "_is_browser_use_cli_mode", lambda: True)
        assert browser_tool.check_browser_requirements() is False
        assert browser_tool.check_browser_vision_requirements() is False

    def test_browser_exec_registered_with_mode_check(self):
        from tools.registry import registry

        entry = registry.get_entry("browser_exec")
        assert entry is not None
        assert entry.check_fn is bu_cli.is_browser_use_cli_mode
        assert entry.toolset == "browser-use"

    def test_browser_exec_in_browser_toolsets(self):
        from toolsets import TOOLSETS, _HERMES_CORE_TOOLS

        assert "browser_exec" in _HERMES_CORE_TOOLS
        assert "browser_exec" in TOOLSETS["browser"]["tools"]
        assert "browser_exec" in TOOLSETS["coding"]["tools"]


class TestFindCli:
    def test_prefers_installed_binary(self, monkeypatch):
        monkeypatch.setattr(
            bu_cli.shutil, "which",
            lambda name: "/usr/local/bin/browser-use" if name == "browser-use" else "/usr/local/bin/uvx",
        )
        assert bu_cli._find_cli() == ["/usr/local/bin/browser-use"]

    def test_falls_back_to_uvx(self, monkeypatch):
        monkeypatch.setattr(
            bu_cli.shutil, "which",
            lambda name: "/usr/local/bin/uvx" if name == "uvx" else None,
        )
        assert bu_cli._find_cli() == ["/usr/local/bin/uvx", "browser-use"]

    def test_none_when_neither_available(self, monkeypatch):
        monkeypatch.setattr(bu_cli.shutil, "which", lambda name: None)
        assert bu_cli._find_cli() is None


class TestLegacyCloudMigration:
    """Pre-CLI direct-API Browser Use cloud configs (cloud_provider:
    "browser-use" + BROWSER_USE_API_KEY) auto-route to the CLI backend;
    Nous-gateway users stay on the legacy provider path."""

    _LEGACY = {"browser": {"cloud_provider": "browser-use"}}

    def test_direct_api_config_migrates(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: self._LEGACY)
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        assert bu_cli.is_browser_use_cli_mode() is True

    def test_gateway_config_stays_on_legacy_path(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "browser-use", "use_gateway": True}},
        )
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_no_api_key_stays_on_legacy_path(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: self._LEGACY)
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_explicit_other_backend_wins(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "browser-use", "backend": "something-else"}},
        )
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_other_cloud_provider_does_not_migrate(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "browserbase"}},
        )
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_explicit_local_does_not_migrate(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "local"}},
        )
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_auto_detect_with_key_migrates(self, monkeypatch):
        """No cloud_provider configured + BROWSER_USE_API_KEY set: credential
        auto-detection prefers Browser Use (even when Browserbase creds are
        also present), which now means Browser Use mode."""
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        monkeypatch.setenv("BROWSERBASE_API_KEY", "bb-key")
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "bb-project")
        assert bu_cli.is_browser_use_cli_mode() is True

    def test_auto_detect_without_key_does_not_migrate(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_migrated_config_gets_bu_autospawn(self, tmp_path, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: self._LEGACY)
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "autospawn:$BU_AUTOSPAWN"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print(1)"))
        assert "autospawn:1" in result["output"]

    def test_explicit_backend_does_not_set_bu_autospawn(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "browser-use"}},
        )
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "autospawn:[$BU_AUTOSPAWN]"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print(1)"))
        assert "autospawn:[]" in result["output"]

    def test_picker_highlights_cli_row_for_migrated_config(self, monkeypatch):
        from hermes_cli.tools_config import TOOL_CATEGORIES, _is_provider_active

        cli_row = next(
            r for r in TOOL_CATEGORIES["browser"]["providers"] if r.get("browser_backend")
        )
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        assert _is_provider_active(cli_row, dict(self._LEGACY)) is True
        monkeypatch.delenv("BROWSER_USE_API_KEY")
        assert _is_provider_active(cli_row, dict(self._LEGACY)) is False


class TestProviderPickerIntegration:
    """The `hermes tools` Browser Automation picker row (browser_backend
    marker) must enter/leave CLI mode cleanly and highlight correctly."""

    def _rows(self):
        from hermes_cli.tools_config import TOOL_CATEGORIES

        return TOOL_CATEGORIES["browser"]["providers"]

    def test_picker_has_browser_use_cli_row(self):
        row = next(r for r in self._rows() if r.get("browser_backend"))
        assert row["browser_backend"] == "browser-use"
        assert row["name"] == "Browser Use"

    def test_picker_row_names_stay_unique(self):
        """The CLI row is named "Browser Use"; the legacy plugin API row must
        keep a distinct name — apply_provider_selection matches by name."""
        from hermes_cli.tools_config import TOOL_CATEGORIES, _plugin_browser_providers

        names = [r["name"] for r in TOOL_CATEGORIES["browser"]["providers"]]
        names += [r["name"] for r in _plugin_browser_providers()]
        assert len(names) == len(set(names))

    def test_selecting_cli_row_writes_backend_and_keeps_cloud_provider(self):
        from hermes_cli.tools_config import _write_provider_config

        row = next(r for r in self._rows() if r.get("browser_backend"))
        config = {"browser": {"cloud_provider": "browserbase"}}
        assert row["name"] == "Browser Use"
        _write_provider_config(row, config, managed_feature=None)
        assert config["browser"]["backend"] == "browser-use"
        assert config["browser"]["cloud_provider"] == "browserbase"

    def test_selecting_provider_row_leaves_cli_mode(self):
        from hermes_cli.tools_config import _write_provider_config

        local_row = next(
            r for r in self._rows() if r.get("browser_provider") == "local"
        )
        config = {"browser": {"backend": "browser-use"}}
        _write_provider_config(local_row, config, managed_feature=None)
        assert "backend" not in config["browser"]
        assert config["browser"]["cloud_provider"] == "local"

    def test_active_row_highlight_is_mutually_exclusive(self):
        from hermes_cli.tools_config import _is_provider_active

        cli_row = next(r for r in self._rows() if r.get("browser_backend"))
        local_row = next(
            r for r in self._rows() if r.get("browser_provider") == "local"
        )
        cli_config = {"browser": {"cloud_provider": "local", "backend": "browser-use"}}
        assert _is_provider_active(cli_row, cli_config) is True
        assert _is_provider_active(local_row, cli_config) is False

        local_config = {"browser": {"cloud_provider": "local"}}
        assert _is_provider_active(cli_row, local_config) is False
        assert _is_provider_active(local_row, local_config) is True


class TestBrowserUseSlashCommand:
    """/browser use [off] toggles browser.backend and resets the session,
    mirroring the /tools enable/disable flow."""

    class _Stub:
        def __init__(self):
            self.session_resets = 0

        def new_session(self):
            self.session_resets += 1

    def _run(self, cmd, config, monkeypatch):
        import hermes_cli.config as hc
        from hermes_cli.cli_commands_mixin import CLICommandsMixin

        saved = {}
        monkeypatch.setattr(hc, "load_config", lambda: config)
        monkeypatch.setattr(hc, "save_config", lambda c: saved.update(c))
        stub = self._Stub()
        CLICommandsMixin._handle_browser_command(stub, cmd)
        return stub, saved

    def test_use_enables_backend_and_resets_session(self, monkeypatch):
        stub, saved = self._run("/browser use", {}, monkeypatch)
        assert saved["browser"]["backend"] == "browser-use"
        assert stub.session_resets == 1

    def test_use_off_removes_backend(self, monkeypatch):
        config = {"browser": {"backend": "browser-use"}}
        stub, saved = self._run("/browser use off", config, monkeypatch)
        assert "backend" not in saved["browser"]
        assert stub.session_resets == 1

    def test_use_bad_arg_prints_usage_without_writing(self, monkeypatch):
        stub, saved = self._run("/browser use whatever", {}, monkeypatch)
        assert saved == {}
        assert stub.session_resets == 0


class TestNativeScreenshots:
    """Screenshots printed by capture_screenshot() attach directly to the
    model's context when it has native vision — no aux vision-LLM detour."""

    def _shot(self, tmp_path):
        shot = tmp_path / "shot.png"
        shot.write_bytes(b"\x89PNG fake")
        return str(shot)

    def test_find_screenshot_returns_last_fresh_path(self, tmp_path):
        a, b = self._shot(tmp_path), str(tmp_path / "b.png")
        (tmp_path / "b.png").write_bytes(b"\x89PNG fake2")
        out = f"step one saved {a}\nthen saved {b}\n"
        assert bu_cli._find_screenshot(out, since=time.time() - 5) == b

    def test_find_screenshot_rejects_stale_and_missing(self, tmp_path):
        stale = self._shot(tmp_path)
        os.utime(stale, (time.time() - 900, time.time() - 900))
        out = f"{stale}\n/nonexistent/dir/x.png\n"
        assert bu_cli._find_screenshot(out, since=time.time()) is None

    def test_vision_model_gets_multimodal_envelope(self, tmp_path, monkeypatch):
        shot = self._shot(tmp_path)
        cli = _fake_cli(tmp_path, f'cat > /dev/null\necho "{shot}"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        monkeypatch.setattr(
            "tools.vision_tools._should_use_native_vision_fast_path", lambda: True
        )
        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda p, **kw: "data:image/png;base64,QUJD",
        )
        result = bu_cli.browser_exec("print(capture_screenshot())")
        assert isinstance(result, dict) and result["_multimodal"] is True
        kinds = [part["type"] for part in result["content"]]
        assert kinds == ["text", "image_url"]
        assert result["meta"]["screenshot_path"] == shot
        assert shot in result["text_summary"]

    def test_text_only_model_gets_plain_result_with_path(self, tmp_path, monkeypatch):
        shot = self._shot(tmp_path)
        cli = _fake_cli(tmp_path, f'cat > /dev/null\necho "{shot}"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        monkeypatch.setattr(
            "tools.vision_tools._should_use_native_vision_fast_path", lambda: False
        )
        result = json.loads(bu_cli.browser_exec("print(capture_screenshot())"))
        assert result["screenshot_path"] == shot

    def test_no_screenshot_keeps_string_result(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "no images here"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print(1)"))
        assert "screenshot_path" not in result


class TestStepLabels:
    """browser_exec code leads with a `# …` comment (per the tool
    description); the TUI surfaces it as the step label and keeps the code
    collapsed behind display.tool_preview_length."""

    _CODE = "# Searching Amazon for paper towels\nnew_tab('https://amazon.com')\nwait_for_load()"

    def test_leading_comment_becomes_step_label(self):
        from agent.display import _browser_exec_step_label

        assert _browser_exec_step_label({"code": self._CODE}) == "Searching Amazon for paper towels"

    def test_no_comment_returns_none(self):
        from agent.display import _browser_exec_step_label

        assert _browser_exec_step_label({"code": "new_tab('x')"}) is None
        assert _browser_exec_step_label({"code": ""}) is None
        assert _browser_exec_step_label({"code": "#   "}) is None

    def test_label_hard_capped_regardless_of_global_setting(self):
        from agent.display import _browser_exec_step_label

        long = "# " + "x" * 200
        label = _browser_exec_step_label({"code": long})
        assert len(label) <= 80 and label.endswith("…")

    def test_preview_prefers_comment_over_code(self):
        from agent.display import build_tool_preview

        assert build_tool_preview("browser_exec", {"code": self._CODE}) == (
            "Searching Amazon for paper towels"
        )
        assert "new_tab" in build_tool_preview("browser_exec", {"code": "new_tab('x')"})

    def test_progress_line_shows_label(self):
        from agent.display import get_cute_tool_message

        line = get_cute_tool_message("browser_exec", {"code": self._CODE}, 1.2)
        assert "Searching Amazon for paper towels" in line
        assert "new_tab" not in line

    def test_header_instructs_leading_comment(self):
        assert "one-line comment" in bu_cli._HEADER_BASE
        assert "step label" in bu_cli._HEADER_BASE


class TestHeaderVariants:
    def test_vision_header_forbids_vision_tool_detour(self, monkeypatch):
        monkeypatch.setattr(
            "tools.vision_tools._should_use_native_vision_fast_path", lambda: True
        )
        header = bu_cli._description_header()
        assert header.startswith(bu_cli._HEADER_BASE)
        assert "attached to your context automatically" in header

    def test_text_only_header_teaches_text_workflow(self, monkeypatch):
        monkeypatch.setattr(
            "tools.vision_tools._should_use_native_vision_fast_path", lambda: False
        )
        header = bu_cli._description_header()
        assert "cannot view images" in header
        assert "page_info()" in header


class TestSkillTextDescription:
    @pytest.fixture(autouse=True)
    def _reset_skill_cache(self, monkeypatch):
        monkeypatch.setattr(bu_cli, "_skill_text_cache", None)
        monkeypatch.setattr(bu_cli, "_skill_text_fetched", False)
        yield

    def test_description_is_verbatim_cli_skill_text(self, tmp_path, monkeypatch):
        cli = _fake_cli(
            tmp_path,
            'if [ "$1" = "skill" ]; then echo "# Browser Use\nverbatim skill body"; fi\n',
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        overrides = bu_cli._dynamic_schema_overrides()
        assert overrides["description"].startswith(bu_cli._DESCRIPTION_HEADER)
        assert overrides["description"].endswith("# Browser Use\nverbatim skill body")

    def test_skill_text_cached_after_first_fetch(self, tmp_path, monkeypatch):
        calls = []
        cli = _fake_cli(tmp_path, 'echo "skill text"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: (calls.append(1), [cli])[1])
        assert "skill text" in bu_cli._cli_skill_text()
        assert "skill text" in bu_cli._cli_skill_text()
        assert len(calls) == 1

    def test_no_override_when_cli_missing(self, monkeypatch):
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert bu_cli._dynamic_schema_overrides() == {}


class TestBrowserExec:
    def test_missing_cli_returns_install_hint(self, monkeypatch):
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        result = json.loads(bu_cli.browser_exec("print(page_info())"))
        assert "uv tool install browser-use" in result["error"]

    def test_empty_code_rejected(self):
        result = json.loads(bu_cli.browser_exec("   "))
        assert "error" in result

    def test_code_piped_on_stdin(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, 'code=$(cat)\necho "got:$code"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec('print("hi")'))
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert 'got:print("hi")' in result["output"]
        assert "session" not in result

    def test_session_sets_bu_name(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "bu:$BU_NAME"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print(1)", session="r7k2"))
        assert "bu:r7k2" in result["output"]
        assert result["session"] == "r7k2"

    def test_invalid_session_name_rejected(self, monkeypatch, tmp_path):
        cli = _fake_cli(tmp_path, "cat > /dev/null\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print(1)", session="bad name!"))
        assert "error" in result
        assert "session" in result["error"].lower()

    def test_nonzero_exit_reports_failure_and_stderr(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "boom" >&2\nexit 3\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print(1)"))
        assert result["success"] is False
        assert result["exit_code"] == 3
        assert "boom" in result["stderr"]

    def test_timeout_returns_actionable_error(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, "cat > /dev/null\nsleep 30\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        monkeypatch.setattr(bu_cli, "_MIN_TIMEOUT_S", 1)
        result = json.loads(bu_cli.browser_exec("print(1)", timeout_s=1))
        assert "timed out" in result["error"]
