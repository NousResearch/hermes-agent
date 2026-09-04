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
import subprocess
import sys
import threading
import time
from pathlib import Path

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
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
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

    def test_subprocess_env_strips_parent_python_import_paths(self, monkeypatch):
        """#83427/#84841/#86006/#86104: the browser-use CLI runs under its
        own Python — inherited PYTHONPATH/PYTHONHOME pointing at Hermes's
        venv make it import wrong-ABI C-extensions (pydantic_core) and
        crash. Both must be stripped; unrelated vars survive."""
        import sys
        from types import ModuleType

        browser_tool = ModuleType("tools.browser_tool")
        browser_tool._build_browser_env = lambda: {
            "PYTHONPATH": "/hermes:/hermes/venv/lib/site-packages",
            "PYTHONHOME": "/hermes/venv",
            "KEEP_ME": "yes",
        }
        monkeypatch.setitem(sys.modules, "tools.browser_tool", browser_tool)

        env = bu_cli._base_subprocess_env()

        assert "PYTHONPATH" not in env
        assert "PYTHONHOME" not in env
        assert env["KEEP_ME"] == "yes"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX PATH-floor semantics")
    def test_subprocess_env_floors_version_manager_only_path(self, monkeypatch):
        """Profile workers (kanban bots, cron) can inherit a PATH of only
        version-manager dirs (observed in the wild: one nvm dir repeated
        7x). The uv browser-use trampoline resolves dirname/realpath
        through PATH, so /usr/bin must be guaranteed or the CLI dies
        'realpath: not found' (exit 127) before its Python starts."""
        import sys
        from types import ModuleType

        browser_tool = ModuleType("tools.browser_tool")
        browser_tool._build_browser_env = lambda: {
            "PATH": os.pathsep.join(
                ["/home/u/.nvm/versions/node/v24.18.0/bin"] * 7
            ),
        }
        monkeypatch.setitem(sys.modules, "tools.browser_tool", browser_tool)

        env = bu_cli._base_subprocess_env()

        parts = env["PATH"].split(os.pathsep)
        assert "/usr/bin" in parts
        assert "/bin" in parts

    @pytest.mark.skipif(os.name == "nt", reason="POSIX PATH-floor semantics")
    def test_floor_preserves_existing_entries_and_order(self):
        """The floor only adds dirs — never drops or reorders what the
        caller's environment already had."""
        original = "/opt/toolchain/bin:/usr/bin:/snap/bin"
        merged = bu_cli._floor_subprocess_path(original).split(os.pathsep)

        assert set(original.split(os.pathsep)) <= set(merged)
        positions = [merged.index(p) for p in original.split(os.pathsep)]
        assert positions == sorted(positions)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX PATH-floor semantics")
    def test_floor_survives_missing_sibling_helper(self, monkeypatch):
        """If browser_tool stops exporting _merge_browser_path, the floor
        degrades to appending FHS bin dirs instead of vanishing."""
        import sys
        from types import ModuleType

        browser_tool = ModuleType("tools.browser_tool")
        browser_tool._build_browser_env = lambda: {
            "PATH": "/home/u/.nvm/versions/node/v24.18.0/bin"
        }
        monkeypatch.setitem(sys.modules, "tools.browser_tool", browser_tool)

        env = bu_cli._base_subprocess_env()

        parts = env["PATH"].split(os.pathsep)
        assert "/usr/bin" in parts
        assert "/home/u/.nvm/versions/node/v24.18.0/bin" in parts


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

    def test_named_session_composes_with_provider_backend(self, tmp_path, monkeypatch):
        """session=<name> composes with a configured provider backend: the
        name keys its OWN provider browser (bu-named-<name>), so concurrent
        named sessions never share one browser (#86894)."""
        import tools.browser_tool as bt

        seen = []

        def fake_session_info(key):
            seen.append(key)
            return {"cdp_url": "wss://browser.example/cdp/" + key}

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(bt, "_get_session_info", fake_session_info)
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "bu:$BU_NAME ws:$BU_CDP_WS"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print(1)", session="r7k2"))
        assert result["success"] is True
        assert seen == ["bu-named-r7k2"]
        assert "bu:r7k2" in result["output"]
        assert "ws:wss://browser.example/cdp/bu-named-r7k2" in result["output"]

    def test_named_session_key_stable_across_tasks(self, monkeypatch):
        """The same session name maps to the same provider cache key no
        matter which task calls it — that is what lets a follow-up call
        reattach to the same cloud browser."""
        import tools.browser_tool as bt

        seen = []
        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(
            bt, "_get_session_info",
            lambda key: seen.append(key) or {"cdp_url": "wss://x/cdp/a"},
        )
        env1, env2 = {}, {}
        assert bu_cli._resolve_backend_cdp(env1, "task-A", session_name="research") is None
        assert bu_cli._resolve_backend_cdp(env2, "task-B", session_name="research") is None
        assert seen == ["bu-named-research", "bu-named-research"]

    def test_named_session_direct_api_bu_cloud_still_skips_provider(
        self, tmp_path, monkeypatch
    ):
        """Direct-API Browser Use cloud configs keep the native named-daemon
        path: resolving through the provider would double-session and
        double-bill."""
        import tools.browser_tool as bt

        class _BUProvider:
            name = "browser-use"

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: _BUProvider())
        monkeypatch.setattr(
            bt, "_get_session_info",
            lambda key: (_ for _ in ()).throw(AssertionError("must skip provider")),
        )
        monkeypatch.setattr(bu_cli, "_read_browser_cfg", lambda: {"cloud_provider": "browser-use"})
        env = {}
        assert bu_cli._resolve_backend_cdp(env, "t1", session_name="r7k2") is None
        assert "BU_CDP_WS" not in env and "BU_CDP_URL" not in env


class TestOwnTabPreamble:
    """Named sessions on SHARED browsers get the own-tab preamble prepended;
    private per-name browsers and unnamed sessions do not."""

    def _run(self, tmp_path, monkeypatch, *, session="", private=False, provider=False):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        if provider:
            monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
            monkeypatch.setattr(
                bt, "_get_session_info",
                lambda key: {"cdp_url": "wss://browser.example/cdp/" + key},
            )
        else:
            monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        # fake CLI echoes stdin back so we can inspect what code was sent
        cli = _fake_cli(tmp_path, "cat\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        return json.loads(bu_cli.browser_exec("print('payload')", session=session))

    def test_named_shared_browser_gets_preamble(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, session="r7k2")
        assert result["success"] is True
        assert "_hermes_ensure_own_tab" in result["output"]
        # model code still present, after the preamble
        assert result["output"].index("_hermes_ensure_own_tab") < result["output"].index("print('payload')")

    def test_unnamed_session_gets_no_preamble(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, session="")
        assert result["success"] is True
        assert "_hermes_ensure_own_tab" not in result["output"]

    def test_named_provider_browser_skips_preamble(self, tmp_path, monkeypatch):
        """Per-name provider browsers are private — preamble would leak a tab."""
        result = self._run(tmp_path, monkeypatch, session="r7k2", provider=True)
        assert result["success"] is True
        assert "_hermes_ensure_own_tab" not in result["output"]

    def test_sentinel_never_reaches_subprocess_env(self, tmp_path, monkeypatch):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(
            bt, "_get_session_info",
            lambda key: {"cdp_url": "wss://browser.example/cdp/" + key},
        )
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "sentinel:${_HERMES_BU_PRIVATE_BROWSER:-unset}"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print(1)", session="r7k2"))
        assert "sentinel:unset" in result["output"]

    def test_preamble_is_valid_python(self):
        import ast

        ast.parse(bu_cli._OWN_TAB_PREAMBLE)
        # and composes with model code
        ast.parse(bu_cli._OWN_TAB_PREAMBLE + "print('x')")

    def test_preamble_reselects_existing_logical_tab_every_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BH_RUNTIME_DIR", str(tmp_path))
        monkeypatch.setenv("HERMES_BU_LOGICAL_SESSION", "research")
        targets, created, switched = {}, [], []

        def fake_cdp(method, **kwargs):
            if method == "Target.getTargets":
                return {"targetInfos": list(targets.values())}
            target_id = f"target-{len(created) + 1}"
            targets[target_id] = {"targetId": target_id, "type": "page"}
            created.append(target_id)
            return {"targetId": target_id}

        for _ in range(2):
            exec(bu_cli._OWN_TAB_PREAMBLE, {"cdp": fake_cdp, "switch_tab": switched.append})
        assert created == ["target-1"]
        assert switched == ["target-1", "target-1"]


class TestSessionScopedTransport:
    def test_lineage_reuses_runtime_across_logical_names_and_compression(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
        env1, env2 = {}, {}
        one = bu_cli._configure_session_transport(env1, conversation_id="root", logical_session="a", private_browser=False)
        two = bu_cli._configure_session_transport(env2, conversation_id="root", logical_session="b", private_browser=False)
        assert one is not None and two is not None
        assert one.key == two.key
        assert env1["BH_RUNTIME_DIR"] == env2["BH_RUNTIME_DIR"]
        assert env1["HERMES_BU_LOGICAL_SESSION"] == "a"
        assert env2["HERMES_BU_LOGICAL_SESSION"] == "b"

    def test_different_lineage_endpoint_or_profile_gets_different_transport(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
        a = bu_cli._configure_session_transport({}, conversation_id="root-a", logical_session="x", private_browser=False)
        b = bu_cli._configure_session_transport({}, conversation_id="root-b", logical_session="x", private_browser=False)
        c = bu_cli._configure_session_transport({"BU_CDP_WS": "ws://127.0.0.1:9/other"}, conversation_id="root-a", logical_session="x", private_browser=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
        d = bu_cli._configure_session_transport({}, conversation_id="root-a", logical_session="x", private_browser=False)
        assert a is not None and b is not None and c is not None and d is not None
        assert len({a.key, b.key, c.key, d.key}) == 4

    def test_private_browser_is_not_multiplexed(self):
        env = {}
        assert bu_cli._configure_session_transport(env, conversation_id="root", logical_session="x", private_browser=True) is None
        assert "BH_RUNTIME_DIR" not in env

    @pytest.mark.skipif(os.name == "nt", reason="AF_UNIX path applies on POSIX")
    def test_runtime_socket_path_stays_under_macos_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ("very-long-" * 20)))
        info = bu_cli._configure_session_transport({}, conversation_id="root", logical_session="x", private_browser=False)
        assert info is not None
        assert len(os.fsencode(str(info.runtime_dir / "bu.sock"))) < 100

    @pytest.mark.skipif(os.name == "nt", reason="POSIX flock integration")
    def test_transport_file_lock_serializes_processes(self, tmp_path):
        runtime = tmp_path / "run"
        scratch = tmp_path / "tmp"
        runtime.mkdir(); scratch.mkdir()
        output = tmp_path / "order.log"
        script = tmp_path / "lock_probe.py"
        script.write_text(
            "import sys, threading, time\n"
            "from pathlib import Path\n"
            "from tools.browser_use_cli import _SessionTransport, _TransportExecutionLock\n"
            "t=_SessionTransport('key', Path(sys.argv[2]), Path(sys.argv[3]), threading.RLock())\n"
            "lock=_TransportExecutionLock(t, 5); lock.acquire()\n"
            "with open(sys.argv[4], 'a', encoding='utf-8') as f: f.write('start '+sys.argv[1]+'\\n')\n"
            "time.sleep(.2)\n"
            "with open(sys.argv[4], 'a', encoding='utf-8') as f: f.write('end '+sys.argv[1]+'\\n')\n"
            "lock.release()\n",
            encoding="utf-8",
        )
        args = [str(runtime), str(scratch), str(output)]
        repo_root = str(Path(__file__).resolve().parents[2])
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = repo_root + (
            os.pathsep + child_env["PYTHONPATH"]
            if child_env.get("PYTHONPATH")
            else ""
        )
        first = subprocess.Popen(
            [sys.executable, str(script), "a", *args],
            cwd=repo_root,
            env=child_env,
        )
        second = subprocess.Popen(
            [sys.executable, str(script), "b", *args],
            cwd=repo_root,
            env=child_env,
        )
        assert first.wait(timeout=10) == 0
        assert second.wait(timeout=10) == 0
        lines = output.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4
        assert lines[0].split()[1] == lines[1].split()[1]
        assert lines[2].split()[1] == lines[3].split()[1]
        assert lines[0].split()[1] != lines[2].split()[1]

    def test_registry_forwards_conversation_lineage(self, monkeypatch):
        from tools.registry import registry
        captured = {}
        monkeypatch.setattr(bu_cli, "browser_exec", lambda **kw: captured.update(kw) or "ok")
        entry = registry.get_entry("browser_exec")
        assert entry is not None
        assert entry.handler({"code": "print(1)"}, session_id="child", conversation_id="root") == "ok"
        assert captured["conversation_id"] == "root"

    def test_fake_daemon_reports_attach_reuse_and_drop_reconnect(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
        cli = _fake_cli(tmp_path, 'cat > /dev/null\nif [ ! -f "$BH_RUNTIME_DIR/alive" ]; then n=$(($(cat "$BH_RUNTIME_DIR/n" 2>/dev/null || echo 0)+1)); echo "$n" > "$BH_RUNTIME_DIR/n"; echo "$n" > "$BH_RUNTIME_DIR/bu.pid"; touch "$BH_RUNTIME_DIR/alive"; fi\necho "runtime:$BH_RUNTIME_DIR"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        monkeypatch.setattr(
            bu_cli,
            "_probe_daemon_pid",
            lambda runtime: (
                int((runtime / "bu.pid").read_text(encoding="utf-8"))
                if (runtime / "alive").exists() and (runtime / "bu.pid").exists()
                else None
            ),
        )
        raw1 = bu_cli.browser_exec("print(1)", session="a", conversation_id="root")
        raw2 = bu_cli.browser_exec("print(2)", session="b", conversation_id="root")
        assert isinstance(raw1, str) and isinstance(raw2, str)
        one, two = json.loads(raw1), json.loads(raw2)
        assert one["cdp_transport"]["attach_count"] == 1
        assert two["cdp_transport"]["attach_count"] == 1
        assert two["cdp_transport"]["reused"] is True
        runtime = Path(one["output"].split("runtime:", 1)[1].strip())
        (runtime / "alive").unlink(); (runtime / "bu.pid").unlink()
        raw3 = bu_cli.browser_exec("print(3)", session="a", conversation_id="root")
        assert isinstance(raw3, str)
        three = json.loads(raw3)
        assert three["cdp_transport"]["attach_count"] == 2
        assert three["cdp_transport"]["attached_this_call"] == 1

    def test_daemon_identity_requires_live_ping_matching_pid_file(
        self, tmp_path, monkeypatch
    ):
        runtime = tmp_path / "run"
        runtime.mkdir()
        (runtime / "bu.pid").write_text("123", encoding="utf-8")

        monkeypatch.setattr(bu_cli, "_probe_daemon_pid", lambda _runtime: None)
        assert bu_cli._daemon_identity(runtime) is None

        monkeypatch.setattr(bu_cli, "_probe_daemon_pid", lambda _runtime: 124)
        assert bu_cli._daemon_identity(runtime) is None

        monkeypatch.setattr(bu_cli, "_probe_daemon_pid", lambda _runtime: 123)
        assert bu_cli._daemon_identity(runtime) is not None

    def test_permission_failure_counts_attempt_not_attach_success(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
        info = bu_cli._configure_session_transport(
            {}, conversation_id="root", logical_session="a", private_browser=False
        )
        assert info is not None

        report = bu_cli._record_transport_outcome(
            info,
            None,
            None,
            "a",
            False,
            "permission-blocked: Allow remote debugging? was not accepted",
        )

        assert report["attach_count"] == 1
        assert report["attach_success_count"] == 0
        assert report["attached_this_call"] == 1
        assert report["reused"] is False
        telemetry = json.loads(
            (info.tmp_dir / "transport-telemetry.json").read_text(encoding="utf-8")
        )
        assert telemetry["events"][-1]["event"] == "attach_failed"
        assert telemetry["events"][-1]["reason"] == "permission_blocked"

    def test_telemetry_never_persists_endpoint_or_raw_session_ids(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
        endpoint = "ws://127.0.0.1:9/devtools/browser/SECRET-ENDPOINT"
        env = {"BU_CDP_WS": endpoint}
        info = bu_cli._configure_session_transport(
            env, conversation_id="SECRET-LINEAGE", logical_session="SECRET-LOGICAL",
            private_browser=False,
        )
        assert info is not None
        bu_cli._record_transport_outcome(
            info, None, "hashed-daemon", "SECRET-LOGICAL", True, ""
        )
        persisted = (info.tmp_dir / "transport-telemetry.json").read_text(
            encoding="utf-8"
        )
        assert "SECRET-ENDPOINT" not in persisted
        assert "SECRET-LINEAGE" not in persisted
        assert "SECRET-LOGICAL" not in persisted


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
    """MANAGED-FIRST: _find_cli probes $HERMES_HOME/bin before PATH and
    ~/.local/bin, so the Hermes-installed copy always wins."""

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
        bu.write_text("#!/bin/sh\n", encoding="utf-8")
        bu.chmod(bu.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(bu)]

    def test_managed_bin_uvx_fallback(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "home" / "bin"
        bin_dir.mkdir(parents=True)
        uvx = bin_dir / "uvx"
        uvx.write_text("#!/bin/sh\n", encoding="utf-8")
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
        cli.write_text("#!/bin/sh\n", encoding="utf-8")
        cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(cli)]

    def test_managed_bin_precedes_user_local_bin(self, tmp_path, monkeypatch):
        """MANAGED-FIRST: Hermes' managed copy wins over a user-level side
        install — every backend selection provisions/updates the managed
        copy, so resolution must land on the binary we control (no version
        drift from stray `uv tool install` runs)."""
        user_dir = tmp_path / "userhome" / ".local" / "bin"
        user_dir.mkdir(parents=True)
        user_cli = user_dir / "browser-use"
        user_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        user_cli.chmod(user_cli.stat().st_mode | stat.S_IXUSR)
        managed_dir = tmp_path / "home" / "bin"
        managed_dir.mkdir(parents=True)
        managed_cli = managed_dir / "browser-use"
        managed_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        managed_cli.chmod(managed_cli.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(managed_cli)]

    def test_managed_bin_precedes_path(self, tmp_path, monkeypatch):
        """MANAGED-FIRST: the managed copy also wins over one on PATH."""
        path_dir = tmp_path / "onpath"
        path_dir.mkdir()
        path_cli = path_dir / "browser-use"
        path_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        path_cli.chmod(path_cli.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("PATH", str(path_dir))
        managed_dir = tmp_path / "home" / "bin"
        managed_dir.mkdir(parents=True)
        managed_cli = managed_dir / "browser-use"
        managed_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        managed_cli.chmod(managed_cli.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(managed_cli)]

    def test_user_local_bin_uvx_fallback(self, tmp_path, monkeypatch):
        cli_dir = tmp_path / "userhome" / ".local" / "bin"
        cli_dir.mkdir(parents=True)
        uvx = cli_dir / "uvx"
        uvx.write_text("#!/bin/sh\n", encoding="utf-8")
        uvx.chmod(uvx.stat().st_mode | stat.S_IXUSR)
        assert bu_cli._find_cli_unpatched() == [str(uvx), "browser-use"]


class TestInstallCli:
    def test_path_install_does_not_short_circuit(self, tmp_path, monkeypatch):
        """MANAGED-FIRST: a browser-use on PATH is a user-level side install
        and must NOT satisfy install_cli() — only the managed copy does,
        otherwise resolution stays pinned to a binary Hermes can't update."""
        cli = _fake_cli(tmp_path, "")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr(bu_cli.shutil, "which", lambda name, path=None: cli if name == "browser-use" and path is None else None)
        import sys as _sys
        import types as _types
        fake = _types.ModuleType("hermes_cli.managed_uv")
        fake.ensure_uv = lambda **kw: None
        monkeypatch.setitem(_sys.modules, "hermes_cli.managed_uv", fake)
        ok, msg = bu_cli.install_cli()
        # No uv available in this fixture, so the attempted managed install
        # fails — the point is that the PATH copy did not short-circuit.
        assert ok is False
        assert "already installed" not in msg

    def test_already_installed_in_managed_bin(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "home" / "bin"
        bin_dir.mkdir(parents=True)
        cli = bin_dir / "browser-use"
        cli.write_text("#!/bin/sh\n", encoding="utf-8")
        cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
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
        uv.write_text(
            '#!/bin/sh\necho "no network" >&2\nexit 1\n', encoding="utf-8"
        )
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


class TestLightpandaBackendResolution:
    """browser.engine: lightpanda in Browser Use mode — Hermes spawns
    ``lightpanda serve`` through the same _get_session_info machinery and
    exports its endpoint, but only when nothing with higher precedence
    (BU_CDP_* env, a CDP override, a cloud provider) claimed the session."""

    def _setup(self, monkeypatch, *, engine=True, info=None, boom=None):
        import tools.browser_tool as bt

        seen = []

        def fake_session_info(key):
            seen.append(key)
            if boom:
                raise boom
            return info if info is not None else {"cdp_url": "http://127.0.0.1:43111"}

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr(bt, "_using_lightpanda_engine", lambda: engine)
        monkeypatch.setattr(bt, "_get_session_info", fake_session_info)
        return seen

    def test_exports_bu_cdp_url_and_private_sentinel(self, monkeypatch):
        seen = self._setup(monkeypatch)
        env = {}
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env["BU_CDP_URL"] == "http://127.0.0.1:43111"
        assert env[bu_cli._PRIVATE_BROWSER_SENTINEL] == "1"
        assert seen == ["t1"]

    def test_named_session_keys_its_own_process(self, monkeypatch):
        seen = self._setup(monkeypatch)
        assert bu_cli._resolve_backend_cdp({}, "t1", session_name="r7k2") is None
        assert seen == ["bu-named-r7k2"]

    def test_default_key_without_task(self, monkeypatch):
        seen = self._setup(monkeypatch)
        assert bu_cli._resolve_backend_cdp({}, None) is None
        assert seen == ["browser-exec-default"]

    def test_launch_failure_returns_actionable_error(self, monkeypatch):
        self._setup(monkeypatch, boom=RuntimeError("no lightpanda binary was found"))
        err = bu_cli._resolve_backend_cdp({}, "t1")
        assert err and "no lightpanda binary was found" in err
        assert "browser.engine" in err

    def test_missing_cdp_returns_error(self, monkeypatch):
        self._setup(monkeypatch, info={"cdp_url": None})
        err = bu_cli._resolve_backend_cdp({}, "t1")
        assert err and "no CDP endpoint" in err

    def test_engine_auto_leaves_env_untouched(self, monkeypatch):
        seen = self._setup(monkeypatch, engine=False)
        env = {}
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env == {}
        assert seen == []

    def test_bu_env_wins(self, monkeypatch):
        seen = self._setup(monkeypatch)
        env = {"BU_CDP_WS": "ws://operator:9222"}
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env["BU_CDP_WS"] == "ws://operator:9222"
        assert seen == []

    def test_cdp_override_wins(self, monkeypatch):
        import tools.browser_tool as bt

        seen = self._setup(monkeypatch)
        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "http://127.0.0.1:9222")
        env = {}
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env["BU_CDP_URL"] == "http://127.0.0.1:9222"
        assert seen == []

    def test_cloud_provider_wins(self, monkeypatch):
        import tools.browser_tool as bt

        seen = self._setup(monkeypatch, info={"cdp_url": "wss://cloud.example/x"})
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        env = {}
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env["BU_CDP_WS"] == "wss://cloud.example/x"
        assert seen == ["t1"]  # provider path, same cache key


class TestLightpandaPreamble:
    def test_lightpanda_session_skips_own_tab_preamble(self, tmp_path, monkeypatch):
        """A Lightpanda process is private to its session: no sibling daemon
        to collide with, and Target.createTarget would fail anyway
        (lightpanda-io/browser#1962)."""
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr(bt, "_using_lightpanda_engine", lambda: True)
        monkeypatch.setattr(
            bt, "_get_session_info", lambda key: {"cdp_url": "http://127.0.0.1:43111"}
        )
        cli = _fake_cli(tmp_path, "cat\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        result = json.loads(bu_cli.browser_exec("print('payload')", session="r7k2"))
        assert result["success"] is True
        assert "_hermes_ensure_own_tab" not in result["output"]
        assert "print('payload')" in result["output"]

    def test_lightpanda_skips_tab_lifecycle_even_when_enabled(
        self, tmp_path, monkeypatch
    ):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr(bt, "_using_lightpanda_engine", lambda: True)
        monkeypatch.setattr(
            bt,
            "_get_session_info",
            lambda key: {"cdp_url": "http://127.0.0.1:43111"},
        )
        monkeypatch.setattr(bu_cli, "_resource_hygiene_enabled", lambda: True)
        lifecycle_calls = []
        monkeypatch.setattr(
            bt,
            "prepare_browser_tab_lifecycle",
            lambda **kwargs: lifecycle_calls.append(kwargs) or (None, None),
        )
        cli = _fake_cli(tmp_path, "cat\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])

        raw = bu_cli.browser_exec("print('payload')", session="r7k2")
        assert isinstance(raw, str)
        result = json.loads(raw)

        assert result["success"] is True
        assert lifecycle_calls == []


class TestLightpandaHeader:
    def test_lightpanda_header_is_text_first_even_for_vision_models(self, monkeypatch):
        monkeypatch.setattr(bu_cli, "_resource_hygiene_enabled", lambda: True)
        monkeypatch.setattr(
            "tools.vision_tools._should_use_native_vision_fast_path", lambda: True
        )
        monkeypatch.setattr(
            "tools.browser_tool.lightpanda_engine_status", lambda: (True, "used")
        )
        header = bu_cli._description_header()
        assert header.startswith(bu_cli._HEADER_BASE)
        assert header.endswith(bu_cli._HEADER_LIGHTPANDA)
        assert "goto_url(url)" in header
        assert "attached to your context automatically" not in header
        assert "TAB LIFECYCLE" not in header
        overrides = bu_cli._dynamic_schema_overrides()
        assert overrides["description"].startswith(bu_cli._HEADER_BASE)
        assert overrides["description"].endswith(bu_cli._HELPERS_DIGEST)

    def test_shadowed_engine_keeps_default_header(self, monkeypatch):
        monkeypatch.setattr(
            "tools.vision_tools._should_use_native_vision_fast_path", lambda: True
        )
        monkeypatch.setattr(
            "tools.browser_tool.lightpanda_engine_status", lambda: (False, "cloud")
        )
        assert bu_cli._description_header() == bu_cli._HEADER_BASE + bu_cli._HEADER_VISION


class TestLightpandaPickerRow:
    def _rows(self):
        from hermes_cli.tools_config import TOOL_CATEGORIES

        return TOOL_CATEGORIES["browser"]["providers"]

    def _row(self, name):
        return next(r for r in self._rows() if r["name"] == name)

    def test_lightpanda_row_shape(self):
        row = self._row("Lightpanda")
        assert row["browser_provider"] == "local"
        assert row["browser_engine"] == "lightpanda"
        assert row["post_setup"] == "lightpanda"
        assert row["env_vars"] == []
        # Local Browser stays the default-highlighted first row.
        assert self._rows()[0]["name"] == "Local Browser"

    def test_selecting_lightpanda_writes_engine_and_local_keeps_backend(self):
        from hermes_cli.tools_config import _write_provider_config

        config = {"browser": {"backend": "browser-use", "cloud_provider": "browserbase"}}
        _write_provider_config(self._row("Lightpanda"), config, managed_feature=None)
        assert config["browser"]["cloud_provider"] == "local"
        assert config["browser"]["engine"] == "lightpanda"
        assert config["browser"]["backend"] == "browser-use"

    def test_selecting_local_browser_resets_engine(self):
        from hermes_cli.tools_config import _write_provider_config

        config = {"browser": {"cloud_provider": "local", "engine": "lightpanda"}}
        _write_provider_config(self._row("Local Browser"), config, managed_feature=None)
        assert config["browser"]["engine"] == "auto"

    def test_active_row_follows_engine(self):
        from hermes_cli.tools_config import _is_provider_active

        lp_row, local_row = self._row("Lightpanda"), self._row("Local Browser")
        lp_cfg = {"browser": {"cloud_provider": "local", "engine": "lightpanda"}}
        assert _is_provider_active(lp_row, lp_cfg) is True
        assert _is_provider_active(local_row, lp_cfg) is False
        default_cfg = {"browser": {"cloud_provider": "local"}}
        assert _is_provider_active(lp_row, default_cfg) is False
        assert _is_provider_active(local_row, default_cfg) is True
        cloud_cfg = {"browser": {"cloud_provider": "browserbase", "engine": "lightpanda"}}
        assert _is_provider_active(lp_row, cloud_cfg) is False


class TestLightpandaStatusLine:
    def _status(self, monkeypatch, *, used, reason, binary="/opt/lightpanda"):
        import contextlib
        import io

        import tools.browser_tool as bt
        from hermes_cli.cli_commands_mixin import CLICommandsMixin

        monkeypatch.setattr(bu_cli, "is_browser_use_cli_mode", lambda: True)
        monkeypatch.setattr(bt, "_using_lightpanda_engine", lambda: True)
        monkeypatch.setattr(bt, "lightpanda_engine_status", lambda: (used, reason))
        monkeypatch.setattr("tools.browser_lightpanda.find_lightpanda_binary", lambda: binary)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CLICommandsMixin._handle_browser_command(object(), "/browser status")
        return buf.getvalue()

    def test_status_reports_lightpanda_in_use(self, monkeypatch):
        out = self._status(monkeypatch, used=True, reason="Browser Use mode: Hermes spawns `lightpanda serve` per session")
        assert "Engine: Lightpanda" in out
        assert "spawns `lightpanda serve`" in out
        assert "Binary: /opt/lightpanda" in out

    def test_status_reports_missing_binary(self, monkeypatch):
        out = self._status(monkeypatch, used=True, reason="x", binary=None)
        assert "lightpanda binary not found" in out

    def test_status_reports_shadowed_engine(self, monkeypatch):
        out = self._status(monkeypatch, used=False, reason="cloud provider Browserbase is selected")
        assert "NOT in use" in out
        assert "Browserbase" in out


class TestBrowserResourceHygiene:
    class FakeGuard:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.target_id = "owned-target"
            self.started = False
            self.finished = False
            type(self).instances.append(self)

        def start(self):
            self.started = True
            return None

        def finish(self):
            self.finished = True
            return {
                "managed": True,
                "ok": True,
                "created": 1,
                "repurposed": 0,
                "closed": 1,
                "leased": 0,
                "remaining_pages": 1,
                "errors": [],
            }

    @pytest.fixture(autouse=True)
    def _guard(self, monkeypatch):
        import tools.browser_tool as browser_tool

        self.FakeGuard.instances = []

        def fake_prepare(**kwargs):
            if kwargs.get("private_browser"):
                return None, None
            guard = self.FakeGuard(**kwargs)
            return guard, guard.start()

        monkeypatch.setattr(browser_tool, "prepare_browser_tab_lifecycle", fake_prepare)
        monkeypatch.setattr(bu_cli, "_resource_hygiene_enabled", lambda: True)
        monkeypatch.setattr(
            bu_cli,
            "_base_subprocess_env",
            lambda: {"BU_CDP_URL": "http://127.0.0.1:9222"},
        )
        monkeypatch.setattr(bu_cli, "_resolve_backend_cdp", lambda env, task_id, **kw: None)

    def test_schema_exposes_explicit_bounded_lease(self):
        properties = bu_cli.BROWSER_EXEC_SCHEMA["parameters"]["properties"]
        assert properties["lease_minutes"]["minimum"] == 0
        assert properties["lease_minutes"]["maximum"] == 120
        assert "lease_reason" in properties

    def test_canonical_config_declares_tab_lifecycle_default(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["browser"]["resource_hygiene"] == {"enabled": False}

    def test_lease_requires_reason_before_browser_execution(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "should-not-run"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        raw = bu_cli.browser_exec("print(1)", lease_minutes=10)
        assert isinstance(raw, str)
        result = json.loads(raw)
        assert "lease_reason is required" in result["error"]
        assert not self.FakeGuard.instances

    def test_managed_local_call_starts_and_finishes_guard(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, 'cat\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        raw = bu_cli.browser_exec(
            "print(1)",
            task_id="task-123",
            turn_id="turn-456",
            lease_minutes=15,
            lease_reason="continue pagination",
        )
        assert isinstance(raw, str)
        result = json.loads(raw)
        guard = self.FakeGuard.instances[0]
        assert guard.started is True
        assert guard.finished is True
        assert guard.kwargs["owner_key"] == "turn-456"
        assert guard.kwargs["lease_minutes"] == 15
        assert 'switch_tab("owned-target")' in result["output"]
        assert result["hygiene"]["closed"] == 1

    def test_timeout_still_finishes_guard(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, "cat > /dev/null\nsleep 30\n")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        monkeypatch.setattr(bu_cli, "_MIN_TIMEOUT_S", 1)
        raw = bu_cli.browser_exec("print(1)", timeout_s=1)
        assert isinstance(raw, str)
        result = json.loads(raw)
        assert "timed out" in result["error"]
        assert self.FakeGuard.instances[0].finished is True

    def test_finish_exception_never_wedges_transport_lock(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "ok"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])

        def finish_boom(_guard):
            raise RuntimeError("finish boom")

        monkeypatch.setattr(self.FakeGuard, "finish", finish_boom)
        with pytest.raises(RuntimeError, match="finish boom"):
            bu_cli.browser_exec("print(1)", conversation_id="lineage-root")

        info = bu_cli._configure_session_transport(
            {"BU_CDP_URL": "http://127.0.0.1:9222"},
            conversation_id="lineage-root",
            logical_session="default",
            private_browser=False,
        )
        assert info is not None
        acquired = []

        def acquire_from_another_thread():
            lock = bu_cli._TransportExecutionLock(info, 0.2)
            try:
                lock.acquire()
                acquired.append(True)
            finally:
                lock.release()

        thread = threading.Thread(target=acquire_from_another_thread)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert acquired == [True]

    def test_named_shared_local_session_is_managed(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "local"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        raw = bu_cli.browser_exec("print(1)", session="local-a")
        assert isinstance(raw, str)
        result = json.loads(raw)
        assert result["success"] is True
        assert len(self.FakeGuard.instances) == 1
        assert self.FakeGuard.instances[0].kwargs["session_name"] == "local-a"

    def test_named_private_cloud_session_is_never_managed(self, tmp_path, monkeypatch):
        cli = _fake_cli(tmp_path, 'cat > /dev/null\necho "cloud"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])

        def mark_private(env, task_id, **kwargs):
            env[bu_cli._PRIVATE_BROWSER_SENTINEL] = "1"
            return None

        monkeypatch.setattr(bu_cli, "_resolve_backend_cdp", mark_private)
        raw = bu_cli.browser_exec("print(1)", session="cloud-a")
        assert isinstance(raw, str)
        result = json.loads(raw)
        assert result["success"] is True
        assert not self.FakeGuard.instances
