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
    def test_default_on_when_cli_available(self, monkeypatch):
        """Backend unset: Browser Use mode is the default when the CLI runs."""
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/usr/bin/browser-use"])
        assert bu_cli.is_browser_use_cli_mode() is True

    def test_default_off_when_cli_unavailable(self, monkeypatch):
        """Backend unset + no runnable CLI: keep the built-in browser tools."""
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_explicit_off_wins_over_default(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": bu_cli.BACKEND_DISABLED}},
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/usr/bin/browser-use"])
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_yaml_bool_off_means_disabled(self, monkeypatch):
        """YAML 1.1 parses unquoted `off` as False — must mean disabled."""
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": False}},
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/usr/bin/browser-use"])
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

    def test_config_read_failure_uses_default(self, monkeypatch):
        def boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("hermes_cli.config.read_raw_config", boom)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert bu_cli.is_browser_use_cli_mode() is False


class TestSubprocessEnvironment:
    def test_browser_use_telemetry_defaults_off(self, monkeypatch):
        import sys
        from types import ModuleType

        browser_tool = ModuleType("tools.browser_tool")
        browser_tool._build_browser_env = lambda: {}
        monkeypatch.setitem(sys.modules, "tools.browser_tool", browser_tool)
        env = bu_cli._base_subprocess_env()
        assert env["ANONYMIZED_TELEMETRY"] == "false"


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

    def test_browser_exec_stripped_without_terminal(self, monkeypatch):
        """Sessions without the terminal surface must not regain host code
        execution through browser_exec (arbitrary Python via the CLI)."""
        monkeypatch.setattr(bu_cli, "is_browser_use_cli_mode", lambda: True)
        from tools.registry import registry

        entry = registry.get_entry("browser_exec")
        monkeypatch.setattr(entry, "check_fn", lambda: True)
        import model_tools

        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["browser"], quiet_mode=False
        )
        names = {t["function"]["name"] for t in defs}
        assert "browser_exec" not in names

    def test_browser_exec_present_with_terminal(self, monkeypatch):
        monkeypatch.setattr(bu_cli, "is_browser_use_cli_mode", lambda: True)
        from tools.registry import registry

        entry = registry.get_entry("browser_exec")
        monkeypatch.setattr(entry, "check_fn", lambda: True)
        import model_tools

        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["browser", "terminal"], quiet_mode=False
        )
        names = {t["function"]["name"] for t in defs}
        assert "browser_exec" in names


class TestFindCli:
    """The tests/tools conftest pins _find_cli to None (host isolation);
    exercise the real function via the preserved _find_cli_unpatched."""

    def test_prefers_installed_binary(self, monkeypatch):
        monkeypatch.setattr(
            bu_cli.shutil, "which",
            lambda name, path=None: "/usr/local/bin/browser-use" if name == "browser-use" and path is None else ("/usr/local/bin/uvx" if path is None else None),
        )
        assert bu_cli._find_cli_unpatched() == ["/usr/local/bin/browser-use"]

    def test_falls_back_to_uvx(self, monkeypatch):
        monkeypatch.setattr(
            bu_cli.shutil, "which",
            lambda name, path=None: "/usr/local/bin/uvx" if name == "uvx" and path is None else None,
        )
        assert bu_cli._find_cli_unpatched() == ["/usr/local/bin/uvx", "browser-use"]

    def test_none_when_neither_available(self, monkeypatch):
        monkeypatch.setattr(bu_cli.shutil, "which", lambda name, path=None: None)
        assert bu_cli._find_cli_unpatched() is None


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
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_no_api_key_stays_on_legacy_path(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: self._LEGACY)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_camofox_user_does_not_migrate(self, monkeypatch):
        """A Camofox user (env-var selected, cloud_provider unset) with a
        stray BROWSER_USE_API_KEY keeps Camofox — no silent mode flip."""
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config", lambda: {"browser": {}}
        )
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        import tools.browser_camofox as camofox

        monkeypatch.setattr(camofox, "is_camofox_mode", lambda: True)
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_camofox_overrides_explicit_backend(self, monkeypatch):
        """Even with browser.backend: browser-use, an active Camofox setup
        falls back to the built-in tools (no CDP surface to drive)."""
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "browser-use"}},
        )
        import tools.browser_camofox as camofox

        monkeypatch.setattr(camofox, "is_camofox_mode", lambda: True)
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
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_explicit_local_does_not_migrate(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"cloud_provider": "local"}},
        )
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-key")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
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
        """No key, no CLI: nothing to migrate and no default flip."""
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
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


class TestBackendCdpResolution:
    """browser_exec routes through the configured browser backend by reusing
    the legacy stack's provider session machinery (_get_session_info)."""

    def _env(self):
        return {}

    def test_existing_bu_env_wins(self, monkeypatch):
        env = {"BU_CDP_WS": "ws://operator-override:9222"}
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env["BU_CDP_WS"] == "ws://operator-override:9222"

    def test_cdp_override_exported(self, monkeypatch):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "http://127.0.0.1:9222")
        env = self._env()
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env["BU_CDP_URL"] == "http://127.0.0.1:9222"

    def test_ws_override_uses_bu_cdp_ws(self, monkeypatch):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "wss://connect.example/x")
        env = self._env()
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env["BU_CDP_WS"] == "wss://connect.example/x"

    def test_cloud_provider_session_exported(self, monkeypatch):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(
            bt, "_get_session_info",
            lambda task_id: {"cdp_url": "wss://browser.example/cdp/abc"},
        )
        env = self._env()
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env["BU_CDP_WS"] == "wss://browser.example/cdp/abc"

    def test_no_provider_leaves_env_untouched(self, monkeypatch):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        env = self._env()
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert "BU_CDP_WS" not in env and "BU_CDP_URL" not in env

    def test_provider_failure_returns_error(self, monkeypatch):
        import tools.browser_tool as bt

        def boom(task_id):
            raise RuntimeError("api down")

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(bt, "_get_session_info", boom)
        err = bu_cli._resolve_backend_cdp(self._env(), "t1")
        assert err and "api down" in err

    def test_provider_without_cdp_returns_error(self, monkeypatch):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(bt, "_get_session_info", lambda task_id: {"cdp_url": None})
        err = bu_cli._resolve_backend_cdp(self._env(), "t1")
        assert err and "no" in err.lower() and "CDP" in err

    def test_named_session_skips_backend_resolution(self, tmp_path, monkeypatch):
        """session=<name> (BU_NAME cloud browser) must not consume a backend
        provider session."""
        import tools.browser_tool as bt

        def fail(task_id):
            raise AssertionError("backend resolution must be skipped")

        monkeypatch.setattr(bt, "_get_session_info", fail)
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "bu:$BU_NAME"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print(1)", session="r7k2"))
        assert result["success"] is True
        assert "bu:r7k2" in result["output"]


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

    def test_selecting_provider_row_keeps_cli_mode(self):
        """Backend composes with the provider: switching browser source
        (local/Browserbase/Firecrawl/gateway) keeps the driver choice."""
        from hermes_cli.tools_config import _write_provider_config

        local_row = next(
            r for r in self._rows() if r.get("browser_provider") == "local"
        )
        config = {"browser": {"backend": "browser-use"}}
        _write_provider_config(local_row, config, managed_feature=None)
        assert config["browser"]["backend"] == "browser-use"
        assert config["browser"]["cloud_provider"] == "local"

    def test_provider_row_stays_active_alongside_cli_mode(self, monkeypatch):
        from hermes_cli.tools_config import _is_provider_active

        cli_row = next(r for r in self._rows() if r.get("browser_backend"))
        local_row = next(
            r for r in self._rows() if r.get("browser_provider") == "local"
        )
        cli_config = {"browser": {"cloud_provider": "local", "backend": "browser-use"}}
        assert _is_provider_active(cli_row, cli_config) is True
        # Provider row remains highlighted: it supplies the browser the CLI
        # driver attaches to.
        assert _is_provider_active(local_row, cli_config) is True

        # Explicit off: the CLI row must not highlight even with the CLI
        # installed (default-on only applies while backend is unset).
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/usr/bin/browser-use"])
        off_config = {"browser": {"cloud_provider": "local", "backend": "off"}}
        assert _is_provider_active(cli_row, off_config) is False
        assert _is_provider_active(local_row, off_config) is True

        # Backend unset: default-on — the CLI row highlights when the CLI
        # is runnable, and not when it isn't.
        default_config = {"browser": {"cloud_provider": "local"}}
        assert _is_provider_active(cli_row, default_config) is True
        assert _is_provider_active(local_row, default_config) is True
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert _is_provider_active(cli_row, default_config) is False


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

    def test_use_off_pins_backend_off(self, monkeypatch):
        """`off` must be written explicitly (BACKEND_DISABLED), not removed:
        with the key merely deleted, is_legacy_browser_use_cloud_config()
        would re-activate CLI mode on the next start for anyone with
        BROWSER_USE_API_KEY set, so /browser use off wouldn't stick."""
        config = {"browser": {"backend": "browser-use"}}
        stub, saved = self._run("/browser use off", config, monkeypatch)
        assert saved["browser"]["backend"] == bu_cli.BACKEND_DISABLED
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
    """The schema description is fully pinned: header + _HELPERS_DIGEST.

    The live ``browser-use skill`` fetch was removed after A/B benchmarking
    showed the pinned digest matches the full skill dump on success rate
    (36/36 vs 36/36, opus-4.8 + kimi-k3) — see tools/browser_use_cli.py.
    """

    def test_description_is_pinned_header_plus_digest(self, monkeypatch):
        # Even with a CLI present, the description must NOT shell out.
        monkeypatch.setattr(
            bu_cli, "_find_cli",
            lambda: (_ for _ in ()).throw(AssertionError("schema must not invoke the CLI")),
        )
        overrides = bu_cli._dynamic_schema_overrides()
        assert overrides["description"].startswith(bu_cli._HEADER_BASE)
        assert overrides["description"].endswith(bu_cli._HELPERS_DIGEST)

    def test_digest_names_core_helpers(self):
        for helper in ("new_tab(", "page_info()", "js(", "fill_input(",
                       "click_at_xy(", "capture_screenshot()", "cdp("):
            assert helper in bu_cli._HELPERS_DIGEST

    def test_static_fallback_carries_digest_and_install_hint(self):
        desc = bu_cli.BROWSER_EXEC_SCHEMA["description"]
        assert bu_cli._HELPERS_DIGEST in desc
        assert "uv tool install browser-use" in desc


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


class TestFindCliManagedBin:
    """_find_cli probes ~/.local/bin and $HERMES_HOME/bin after PATH."""

    @pytest.fixture(autouse=True)
    def _hermetic_home(self, tmp_path, monkeypatch):
        """Pin HOME so the ~/.local/bin probe can't leak the host's real
        user-level installs into these real-PATH-probing tests."""
        monkeypatch.setenv("HOME", str(tmp_path / "userhome"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    def test_managed_bin_browser_use_found(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "home" / "bin"
        bin_dir.mkdir(parents=True)
        bu = bin_dir / "browser-use"
        bu.write_text("#!/bin/sh\n")
        bu.chmod(bu.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(bu)]

    def test_managed_bin_uvx_fallback(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "home" / "bin"
        bin_dir.mkdir(parents=True)
        uvx = bin_dir / "uvx"
        uvx.write_text("#!/bin/sh\n")
        uvx.chmod(uvx.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(uvx), "browser-use"]

    def test_nothing_found(self, tmp_path, monkeypatch):
        assert bu_cli._find_cli_unpatched() is None

    def test_user_local_bin_browser_use_found(self, tmp_path, monkeypatch):
        """#83788: Desktop/TUI workers spawn with a minimal PATH that omits
        ~/.local/bin, where `uv tool install browser-use` links the binary
        by default — _find_cli must probe it explicitly."""
        cli_dir = tmp_path / "userhome" / ".local" / "bin"
        cli_dir.mkdir(parents=True)
        cli = cli_dir / "browser-use"
        cli.write_text("#!/bin/sh\n")
        cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(cli)]

    def test_user_local_bin_precedes_managed_bin(self, tmp_path, monkeypatch):
        """A user-level install wins over Hermes' managed copy — mirrors the
        PATH-first preference (user intent beats bootstrap)."""
        user_dir = tmp_path / "userhome" / ".local" / "bin"
        user_dir.mkdir(parents=True)
        user_cli = user_dir / "browser-use"
        user_cli.write_text("#!/bin/sh\n")
        user_cli.chmod(user_cli.stat().st_mode | stat.S_IXUSR)
        managed_dir = tmp_path / "home" / "bin"
        managed_dir.mkdir(parents=True)
        managed_cli = managed_dir / "browser-use"
        managed_cli.write_text("#!/bin/sh\n")
        managed_cli.chmod(managed_cli.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(user_cli)]

    def test_user_local_bin_uvx_fallback(self, tmp_path, monkeypatch):
        cli_dir = tmp_path / "userhome" / ".local" / "bin"
        cli_dir.mkdir(parents=True)
        uvx = cli_dir / "uvx"
        uvx.write_text("#!/bin/sh\n")
        uvx.chmod(uvx.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(uvx), "browser-use"]


class TestInstallCli:
    def test_already_installed_on_path(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, "")
        monkeypatch.setattr(bu_cli.shutil, "which", lambda name, path=None: cli if name == "browser-use" and path is None else None)
        ok, msg = bu_cli.install_cli()
        assert ok is True
        assert "already installed" in msg

    def test_no_uv_anywhere_fails_with_guidance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        import sys as _sys
        import types as _types
        fake = _types.ModuleType("hermes_cli.managed_uv")
        fake.ensure_uv = lambda **kw: None
        monkeypatch.setitem(_sys.modules, "hermes_cli.managed_uv", fake)
        ok, msg = bu_cli.install_cli()
        assert ok is False
        assert "uv" in msg

    def test_successful_install_via_fake_uv(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        bin_dir = home / "bin"
        bin_dir.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        # install_cli verifies via _find_cli(), which the tests/tools conftest
        # pins to None — restore the real resolver for this test.
        monkeypatch.setattr(bu_cli, "_find_cli", bu_cli._find_cli_unpatched)
        # fake uv: `uv tool install browser-use` drops a binary into UV_TOOL_BIN_DIR.
        # Absolute /bin/chmod: PATH is emptied above, so bare chmod won't resolve.
        uv = tmp_path / "uv"
        uv.write_text(
            "#!/bin/sh\n"
            'target="$UV_TOOL_BIN_DIR/browser-use"\n'
            'echo "#!/bin/sh" > "$target"\n'
            '/bin/chmod +x "$target"\n'
        )
        uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
        import sys as _sys
        import types as _types
        fake = _types.ModuleType("hermes_cli.managed_uv")
        fake.ensure_uv = lambda **kw: str(uv)
        monkeypatch.setitem(_sys.modules, "hermes_cli.managed_uv", fake)
        ok, msg = bu_cli.install_cli()
        assert ok is True, msg
        assert (bin_dir / "browser-use").exists()

    def test_failed_install_surfaces_stderr_tail(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        uv = tmp_path / "uv"
        uv.write_text('#!/bin/sh\necho "no network" >&2\nexit 1\n')
        uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
        import sys as _sys
        import types as _types
        fake = _types.ModuleType("hermes_cli.managed_uv")
        fake.ensure_uv = lambda **kw: str(uv)
        monkeypatch.setitem(_sys.modules, "hermes_cli.managed_uv", fake)
        ok, msg = bu_cli.install_cli()
        assert ok is False
        assert "no network" in msg


class TestDefaultDowngradeNotice:
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})

    def test_notice_when_default_and_cli_missing(self, tmp_path, monkeypatch):
        self._isolate(tmp_path, monkeypatch)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        notice = bu_cli.default_downgrade_notice()
        assert notice is not None
        assert "hermes tools" in notice

    def test_rate_limited_within_24h(self, tmp_path, monkeypatch):
        self._isolate(tmp_path, monkeypatch)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert bu_cli.default_downgrade_notice() is not None
        assert bu_cli.default_downgrade_notice() is None

    def test_no_notice_when_cli_runnable(self, tmp_path, monkeypatch):
        self._isolate(tmp_path, monkeypatch)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/usr/bin/browser-use"])
        assert bu_cli.default_downgrade_notice() is None

    def test_no_notice_on_explicit_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": bu_cli.BACKEND_DISABLED}},
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert bu_cli.default_downgrade_notice() is None
