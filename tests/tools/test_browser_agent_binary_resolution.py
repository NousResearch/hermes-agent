import os
import stat

import pytest

import tools.browser_tool as browser_tool


def _make_executable(path, content):
    path.write_bytes(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture(autouse=True)
def reset_agent_browser_cache(monkeypatch):
    monkeypatch.setattr(browser_tool, "_cached_agent_browser", None)
    monkeypatch.setattr(browser_tool, "_agent_browser_resolved", False)
    monkeypatch.setattr(browser_tool, "_merge_browser_path", lambda path="": path)
    yield
    browser_tool._cached_agent_browser = None
    browser_tool._agent_browser_resolved = False


def test_agent_browser_native_binary_names_include_published_package_variants(monkeypatch):
    monkeypatch.setattr(browser_tool.sys, "platform", "linux")
    monkeypatch.setattr(browser_tool.platform, "machine", lambda: "x86_64")
    assert browser_tool._agent_browser_native_binary_names() == (
        "agent-browser-linux-x64",
        "agent-browser-linux-musl-x64",
    )

    monkeypatch.setattr(browser_tool.platform, "machine", lambda: "aarch64")
    assert browser_tool._agent_browser_native_binary_names() == (
        "agent-browser-linux-arm64",
        "agent-browser-linux-musl-arm64",
    )

    monkeypatch.setattr(browser_tool.sys, "platform", "win32")
    assert browser_tool._agent_browser_native_binary_names() == (
        "agent-browser-win32-arm64.exe",
        "agent-browser-windows-arm64.exe",
    )


def test_find_agent_browser_uses_native_sibling_of_global_npm_shim_without_node(tmp_path, monkeypatch):
    package_bin = tmp_path / "lib" / "node_modules" / "agent-browser" / "bin"
    package_bin.mkdir(parents=True)
    js_shim = _make_executable(
        package_bin / "agent-browser.js",
        b"#!/usr/bin/env node\nrequire('../dist/cli.js')\n",
    )
    native = _make_executable(
        package_bin / browser_tool._agent_browser_native_binary_names()[0],
        b"\x7fELFfake",
    )
    global_bin = tmp_path / "bin"
    global_bin.mkdir()
    global_shim = global_bin / "agent-browser"
    global_shim.symlink_to(js_shim)

    monkeypatch.setattr(browser_tool, "_candidate_agent_browser_native_bins", lambda: [])

    def fake_which(cmd, path=None):
        if cmd == "agent-browser" and path is None:
            return str(global_shim)
        return None

    monkeypatch.setattr(browser_tool.shutil, "which", fake_which)

    assert browser_tool._find_agent_browser() == str(native)


def test_find_agent_browser_chmods_native_binary_when_node_wrapper_is_unavailable(tmp_path, monkeypatch):
    package_bin = tmp_path / "lib" / "node_modules" / "agent-browser" / "bin"
    package_bin.mkdir(parents=True)
    js_shim = _make_executable(
        package_bin / "agent-browser.js",
        b"#!/usr/bin/env node\nrequire('../dist/cli.js')\n",
    )
    native = package_bin / browser_tool._agent_browser_native_binary_names()[0]
    native.write_bytes(b"\x7fELFfake")
    native.chmod(stat.S_IRUSR | stat.S_IWUSR)
    global_shim = tmp_path / "agent-browser"
    global_shim.symlink_to(js_shim)

    monkeypatch.setattr(browser_tool, "_candidate_agent_browser_native_bins", lambda: [])

    def fake_which(cmd, path=None):
        if cmd == "agent-browser" and path is None:
            return str(global_shim)
        return None

    monkeypatch.setattr(browser_tool.shutil, "which", fake_which)

    assert browser_tool._find_agent_browser() == str(native)
    assert native.stat().st_mode & stat.S_IXUSR


def test_find_agent_browser_prefers_native_binary_when_path_shim_needs_missing_node(tmp_path, monkeypatch):
    shim = _make_executable(
        tmp_path / "agent-browser",
        b"#!/bin/sh\nexec node ../agent-browser/dist/cli.js \"$@\"\n",
    )
    native = _make_executable(tmp_path / "agent-browser-linux-x64", b"\x7fELFfake")

    monkeypatch.setattr(browser_tool, "_candidate_agent_browser_native_bins", lambda: [tmp_path / "agent-browser-linux-x64"])

    def fake_which(cmd, path=None):
        if cmd == "agent-browser" and path is None:
            return shim
        return None

    monkeypatch.setattr(browser_tool.shutil, "which", fake_which)

    assert browser_tool._find_agent_browser() == native


def test_find_agent_browser_keeps_path_shim_when_node_is_available(tmp_path, monkeypatch):
    shim = _make_executable(
        tmp_path / "agent-browser",
        b"#!/bin/sh\nexec node ../agent-browser/dist/cli.js \"$@\"\n",
    )
    node = _make_executable(tmp_path / "node", b"\x7fELFfake")
    native = _make_executable(tmp_path / "agent-browser-linux-x64", b"\x7fELFfake")

    monkeypatch.setattr(browser_tool, "_candidate_agent_browser_native_bins", lambda: [tmp_path / "agent-browser-linux-x64"])

    def fake_which(cmd, path=None):
        if cmd == "agent-browser" and path is None:
            return shim
        if cmd == "node":
            return node
        return None

    monkeypatch.setattr(browser_tool.shutil, "which", fake_which)

    assert browser_tool._find_agent_browser() == shim
    assert browser_tool._find_agent_browser() != native


def test_find_agent_browser_does_not_use_npx_without_node(monkeypatch):
    monkeypatch.setattr(browser_tool, "_candidate_agent_browser_native_bins", lambda: [])
    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", lambda _name: False)

    def fake_which(cmd, path=None):
        if cmd == "npx":
            return "/usr/bin/npx"
        return None

    monkeypatch.setattr(browser_tool.shutil, "which", fake_which)

    with pytest.raises(FileNotFoundError):
        browser_tool._find_agent_browser()


def test_legacy_chrome_flags_do_not_suppress_no_sandbox_injection(tmp_path, monkeypatch):
    captured_env = {}

    class FakeProc:
        returncode = 0

        def __init__(self, args, stdout, stderr, stdin, env, **kwargs):
            captured_env.update(env)
            os.write(stdout, b'{"success": true}')

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setenv("AGENT_BROWSER_CHROME_FLAGS", "--legacy-only")
    monkeypatch.delenv("AGENT_BROWSER_ARGS", raising=False)
    monkeypatch.setattr(browser_tool, "_find_agent_browser", lambda: "/bin/agent-browser")
    monkeypatch.setattr(browser_tool, "_requires_real_termux_browser_install", lambda _cmd: False)
    monkeypatch.setattr(browser_tool, "_is_local_mode", lambda: True)
    monkeypatch.setattr(browser_tool, "_chromium_installed", lambda: True)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda _task_id: {"session_name": "test-session"})
    monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(browser_tool, "_write_owner_pid", lambda *_args: None)
    monkeypatch.setattr(browser_tool.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(browser_tool.subprocess, "Popen", FakeProc)

    result = browser_tool._run_browser_command("task-1", "snapshot", timeout=1)

    assert result == {"success": True}
    assert captured_env["AGENT_BROWSER_ARGS"] == "--no-sandbox,--disable-dev-shm-usage"
    assert captured_env["AGENT_BROWSER_CHROME_FLAGS"] == "--legacy-only"
