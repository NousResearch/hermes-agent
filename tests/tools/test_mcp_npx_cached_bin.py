"""``npx -y <pkg>`` should spawn the cached binary, not a resident `npm exec`.

`npx` resolves the package and then FORKS, staying alive as the real server's
parent for the whole process lifetime while doing no work. Measured on a
4-agent host that is ~48 MB of private memory per MCP server — and it buys
nothing, because Hermes already wraps the child in its own parent-death
watchdog, so npx's supervision is a second parent nobody reads.

Removing it must stay conservative: a cache miss, a version-pinned spec, or an
ambiguous ``bin`` map all fall back to plain `npx` so a cold machine still
installs normally.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.mcp_tool import _npx_cached_bin


def _cache(tmp_path, *, package, deps=None, bin_field, make_bin=True, entry="abc123", cache_dir=None):
    """Build a fake npx cache entry the way npm lays one out."""
    root = (cache_dir or tmp_path / ".npm") / "_npx" / entry
    (root / "node_modules" / package).mkdir(parents=True)
    (root / "package.json").write_text(
        json.dumps({"dependencies": deps if deps is not None else {package: "^1.0.0"}}),
        encoding="utf-8",
    )
    (root / "node_modules" / package / "package.json").write_text(
        json.dumps({"name": package, "bin": bin_field}), encoding="utf-8"
    )
    bindir = root / "node_modules" / ".bin"
    bindir.mkdir(parents=True, exist_ok=True)
    name = bin_field if isinstance(bin_field, str) else list(bin_field)[0]
    launcher = os.path.basename(package) if isinstance(bin_field, str) else name
    target = bindir / (launcher + ".cmd" if os.name == "nt" else launcher)
    if make_bin:
        target.write_text("@echo off\n" if os.name == "nt" else "#!/usr/bin/env node\n", encoding="utf-8")
        target.chmod(0o755)
    return target


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("npm_config_cache", str(tmp_path / ".npm"))
    yield


def test_cached_package_resolves_to_its_binary(tmp_path):
    target = _cache(tmp_path, package="mcp-linear", bin_field={"mcp-linear": "dist/index.js"})

    got = _npx_cached_bin(["-y", "mcp-linear"])

    assert got == (str(target), [])


def test_scoped_package_and_trailing_args_survive(tmp_path):
    target = _cache(
        tmp_path,
        package="@tacticlaunch/mcp-linear",
        bin_field={"mcp-linear": "dist/index.js"},
    )

    got = _npx_cached_bin(["-y", "@tacticlaunch/mcp-linear", "--port", "7"])

    assert got == (str(target), ["--port", "7"])


def test_uncached_package_falls_back_to_npx(tmp_path):
    _cache(tmp_path, package="something-else", bin_field={"something-else": "i.js"})

    assert _npx_cached_bin(["-y", "mcp-linear"]) is None


def test_version_pinned_spec_is_left_to_npx(tmp_path):
    _cache(tmp_path, package="mcp-linear", bin_field={"mcp-linear": "dist/index.js"})

    # The user pinned a build; npx owns that resolution and the cache key for
    # a different version would not match this entry.
    assert _npx_cached_bin(["-y", "mcp-linear@1.2.3"]) is None


def test_ambiguous_bin_map_is_left_to_npx(tmp_path):
    _cache(
        tmp_path,
        package="multi",
        bin_field={"one": "a.js", "two": "b.js"},
    )

    # Which bin npx would choose is not ours to guess.
    assert _npx_cached_bin(["-y", "multi"]) is None


def test_missing_or_non_executable_binary_falls_back(tmp_path):
    _cache(
        tmp_path,
        package="mcp-linear",
        bin_field={"mcp-linear": "dist/index.js"},
        make_bin=False,
    )

    assert _npx_cached_bin(["-y", "mcp-linear"]) is None


def test_no_cache_directory_at_all(tmp_path, monkeypatch):
    monkeypatch.setenv("npm_config_cache", str(tmp_path / "nope"))

    assert _npx_cached_bin(["-y", "mcp-linear"]) is None


@pytest.mark.windows_only
def test_windows_default_localappdata_cache_is_used_without_override(tmp_path, monkeypatch):
    """A normal Windows npm install should not fall back to a slow resident npx process."""
    local_app_data = tmp_path / "AppData" / "Local"
    target = _cache(
        tmp_path,
        package="mcp-linear",
        bin_field={"mcp-linear": "dist/index.js"},
        cache_dir=local_app_data / "npm-cache",
    )
    monkeypatch.delenv("npm_config_cache", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    assert _npx_cached_bin(["-y", "mcp-linear"]) == (str(target), [])


@pytest.mark.windows_only
def test_windows_uppercase_server_cache_override_is_relative_to_child_cwd(tmp_path):
    """Windows treats a server's uppercase cache setting as the same npm setting."""
    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
    target = _cache(
        tmp_path,
        package="mcp-linear",
        bin_field={"mcp-linear": "dist/index.js"},
        cache_dir=child_cwd / "configured-npm-cache",
    )

    assert _npx_cached_bin(
        ["-y", "mcp-linear"],
        env={"NPM_CONFIG_CACHE": "configured-npm-cache"},
        cwd=str(child_cwd),
    ) == (str(target), [])


@pytest.mark.linux_only
def test_linux_uppercase_server_cache_override_is_relative_to_child_cwd(tmp_path):
    """npm's conventional uppercase cache setting is honored on POSIX too."""
    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
    target = _cache(
        tmp_path,
        package="mcp-linear",
        bin_field={"mcp-linear": "dist/index.js"},
        cache_dir=child_cwd / "configured-npm-cache",
    )

    assert _npx_cached_bin(
        ["-y", "mcp-linear"],
        env={"NPM_CONFIG_CACHE": "configured-npm-cache"},
        cwd=str(child_cwd),
    ) == (str(target), [])


def test_server_home_relative_cache_uses_the_child_home_not_the_server_cwd(tmp_path):
    """npm's ``~`` cache alias is resolved from the MCP child's HOME."""
    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
    child_home = tmp_path / "child-home"
    target = _cache(
        tmp_path,
        package="mcp-linear",
        bin_field={"mcp-linear": "dist/index.js"},
        cache_dir=child_home / ".npm-custom",
    )

    assert _npx_cached_bin(
        ["-y", "mcp-linear"],
        env={"npm_config_cache": "~/.npm-custom", "HOME": str(child_home)},
        cwd=str(child_cwd),
    ) == (str(target), [])


def _paired_windows_npm_layout(tmp_path):
    """Build the local npx/npm files needed to mock only npm's config response."""
    bin_dir = tmp_path / "node-bin"
    bin_dir.mkdir()
    npx = bin_dir / "npx.cmd"
    node = bin_dir / "node.exe"
    npm_cli = bin_dir / "node_modules" / "npm" / "bin" / "npm-cli.js"
    npx.write_text("@echo off\n", encoding="utf-8")
    node.write_bytes(b"")
    npm_cli.parent.mkdir(parents=True)
    npm_cli.write_text("// fixture\n", encoding="utf-8")
    return npx, node, npm_cli


@pytest.mark.windows_only
def test_effective_cache_honors_uppercase_server_override_without_npm_probe(tmp_path):
    """The conventional Windows spelling bypasses npm config and remains the child setting."""
    from tools.mcp_tool_config import _effective_npx_cache_env

    env = {"NPM_CONFIG_CACHE": "configured-npm-cache"}
    with patch("tools.mcp_tool_config.subprocess.run") as run:
        resolved = _effective_npx_cache_env("npx", env, str(tmp_path))

    assert resolved == env
    run.assert_not_called()


@pytest.mark.windows_only
def test_effective_cache_reads_the_paired_npm_configuration(tmp_path):
    """A project npmrc cache wins over a stale platform-default cache shortcut."""
    from tools.mcp_tool_config import _effective_npx_cache_env

    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
    (child_cwd / ".npmrc").write_text("cache=configured-npm-cache\n", encoding="utf-8")
    npx, node, npm_cli = _paired_windows_npm_layout(tmp_path)
    observed = {}

    def _run(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="configured-npm-cache\n")

    with patch("tools.mcp_tool_config.subprocess.run", side_effect=_run):
        env = _effective_npx_cache_env(str(npx), {"PATH": str(npx.parent)}, str(child_cwd))

    assert env is not None
    assert env["npm_config_cache"] == str(child_cwd / "configured-npm-cache")
    assert observed["argv"] == [
        str(node), str(npm_cli), "--silent", "--no-update-notifier", "--offline", "config", "get", "cache",
    ]
    assert observed["kwargs"]["cwd"] == str(child_cwd)
    assert observed["kwargs"]["env"]["PATH"] == str(npx.parent)


@pytest.mark.windows_only
def test_effective_cache_uses_managed_node_when_paired_npx_omits_one(tmp_path):
    """A paired npm CLI never resolves an arbitrary system Node from PATH."""
    from tools.mcp_tool_config import _effective_npx_cache_env

    npx, node, npm_cli = _paired_windows_npm_layout(tmp_path)
    node.unlink()
    managed_node = tmp_path / "hermes-node" / "node.exe"
    managed_node.parent.mkdir()
    managed_node.write_bytes(b"")
    observed = {}

    def _run(argv, **_kwargs):
        observed["argv"] = argv
        return SimpleNamespace(returncode=0, stdout=str(tmp_path / "npm-cache") + "\n")

    with patch("tools.mcp_tool_config.find_node_executable", return_value=str(managed_node)) as find_node, \
         patch("tools.mcp_tool_config.subprocess.run", side_effect=_run):
        env = _effective_npx_cache_env(str(npx), {"PATH": str(npx.parent)}, str(tmp_path))

    assert env is not None
    assert env["npm_config_cache"] == str(tmp_path / "npm-cache")
    assert observed["argv"] == [
        str(managed_node), str(npm_cli), "--silent", "--no-update-notifier", "--offline", "config", "get", "cache",
    ]
    find_node.assert_called_once_with("node")


@pytest.mark.windows_only
def test_effective_cache_expands_a_home_alias_from_the_child_environment(tmp_path):
    """A paired npm result uses the server's HOME, not Hermes's or the cwd."""
    from tools.mcp_tool_config import _effective_npx_cache_env

    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
    child_home = tmp_path / "child-home"
    npx, _node, _npm_cli = _paired_windows_npm_layout(tmp_path)

    with patch(
        "tools.mcp_tool_config.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="~/.npm-custom\n"),
    ):
        env = _effective_npx_cache_env(
            str(npx),
            {"PATH": str(npx.parent), "HOME": str(child_home)},
            str(child_cwd),
        )

    assert env is not None
    assert env["npm_config_cache"] == str(child_home / ".npm-custom")


@pytest.mark.windows_only
def test_effective_cache_failure_returns_no_configured_cache(tmp_path):
    """A failed npm config subprocess does not invent a custom cache root."""
    from tools.mcp_tool_config import _effective_npx_cache_env

    npx, _node, _npm_cli = _paired_windows_npm_layout(tmp_path)
    with patch(
        "tools.mcp_tool_config.subprocess.run",
        side_effect=subprocess.TimeoutExpired("npm config get cache", 2),
    ):
        assert _effective_npx_cache_env(str(npx), {"PATH": str(npx.parent)}, str(tmp_path)) is None


@pytest.mark.windows_only
def test_preflight_keeps_npx_when_effective_lookup_fails_on_windows():
    """Windows must not guess a default cache after the authoritative probe fails."""
    from tools.mcp_tool import _preflight_stdio_command

    env = {"LOCALAPPDATA": "C:/fixture/AppData/Local"}
    with patch("tools.osv_check.check_package_for_malware", return_value=None), \
         patch("tools.mcp_tool._effective_npx_cache_env", return_value=None), \
         patch("tools.mcp_tool._npx_cached_bin") as cached:
        command, args = asyncio.run(_preflight_stdio_command(
            "server", "npx", ["-y", "mcp-linear"], env=env, cwd="server-cwd"))

    assert (command, args) == ("npx", ["-y", "mcp-linear"])
    cached.assert_not_called()


@pytest.mark.linux_only
def test_preflight_keeps_npx_when_effective_lookup_fails_on_linux(tmp_path):
    """A failed POSIX npm probe never guesses the default cache root."""
    from tools.mcp_tool import _preflight_stdio_command

    with patch("tools.osv_check.check_package_for_malware", return_value=None), \
         patch("tools.mcp_tool._effective_npx_cache_env", return_value=None), \
         patch("tools.mcp_tool._npx_cached_bin") as cached:
        command, args = asyncio.run(_preflight_stdio_command(
            "server", "npx", ["-y", "mcp-linear"], env={"HOME": str(tmp_path)}, cwd=str(tmp_path)))

    assert (command, args) == ("npx", ["-y", "mcp-linear"])
    cached.assert_not_called()


def test_preflight_keeps_custom_npx_config_flag_and_child_env():
    """An invocation that npx owns never probes or pins a different cache."""
    from tools.mcp_tool import _preflight_stdio_command

    env = {"PATH": "C:/fixture/bin", "UNCHANGED": "1"}
    original_env = dict(env)
    with patch("tools.osv_check.check_package_for_malware", return_value=None), \
         patch("tools.mcp_tool._effective_npx_cache_env") as effective, \
         patch("tools.mcp_tool._npx_cached_bin") as cached:
        command, args = asyncio.run(_preflight_stdio_command(
            "server",
            "npx",
            ["--userconfig", "C:/fixture/custom.npmrc", "-y", "mcp-linear"],
            env=env,
            cwd="C:/fixture/project",
        ))

    assert (command, args) == ("npx", ["--userconfig", "C:/fixture/custom.npmrc", "-y", "mcp-linear"])
    assert env == original_env
    effective.assert_not_called()
    cached.assert_not_called()


@pytest.mark.windows_only
def test_preflight_keeps_npx_when_paired_npm_fails(tmp_path):
    """A real failed npm CLI probe never guesses a Windows cache launcher."""
    from tools.mcp_tool import _preflight_stdio_command
    from tools.mcp_tool_config import _resolve_stdio_command

    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
    local_app_data = tmp_path / "appdata" / "local"
    npx_dir = tmp_path / "npx-bin"
    npx_dir.mkdir()
    (npx_dir / "npx.cmd").write_text("@echo off\r\nexit /b 1\r\n", encoding="utf-8")
    npm_cli = npx_dir / "node_modules" / "npm" / "bin" / "npm-cli.js"
    npm_cli.parent.mkdir(parents=True)
    npm_cli.write_text("process.exit(1);\n", encoding="utf-8")
    env = {
        "APPDATA": str(tmp_path / "appdata" / "roaming"),
        "HOME": str(tmp_path / "home"),
        "LOCALAPPDATA": str(local_app_data),
        "PATH": str(npx_dir) + os.pathsep + os.environ.get("PATH", ""),
        "ProgramFiles": str(tmp_path / "program-files"),
        "USERPROFILE": str(tmp_path / "profile"),
    }
    command, safe_env = _resolve_stdio_command("npx", env)

    with patch("tools.osv_check.check_package_for_malware", return_value=None):
        direct_command, direct_args = asyncio.run(_preflight_stdio_command(
            "server", command, ["-y", "mcp-linear"], env=safe_env, cwd=str(child_cwd)))

    assert (direct_command, direct_args) == (command, ["-y", "mcp-linear"])


def test_preflight_keeps_npx_when_default_cache_may_be_overridden():
    """A failed config lookup does not scan a default cache when a custom one may apply."""
    from tools.mcp_tool import _preflight_stdio_command

    with patch("tools.osv_check.check_package_for_malware", return_value=None), \
         patch("tools.mcp_tool._effective_npx_cache_env", return_value=None), \
         patch("tools.mcp_tool._npx_cached_bin") as cached:
        command, args = asyncio.run(_preflight_stdio_command("server", "npx", ["-y", "mcp-linear"]))

    assert (command, args) == ("npx", ["-y", "mcp-linear"])
    cached.assert_not_called()


def test_corrupt_cache_manifest_is_skipped(tmp_path):
    root = tmp_path / ".npm" / "_npx" / "broken"
    root.mkdir(parents=True)
    (root / "package.json").write_text("{ not json", encoding="utf-8")

    assert _npx_cached_bin(["-y", "mcp-linear"]) is None


@pytest.mark.parametrize("args", [[], ["-y"], ["--yes"], ["-p", "x"], None, "notalist"])
def test_unusable_args_are_ignored(args):
    assert _npx_cached_bin(args) is None


def test_osv_preflight_runs_before_the_swap():
    """The malware gate must still see `npx` + the package name.

    `_infer_ecosystem` keys off the command basename, so a command already
    rewritten to `.../node_modules/.bin/mcp-linear` yields no ecosystem and
    `check_package_for_malware` returns None — the gate silently becomes a
    no-op. This pins the ordering: OSV inspects the original invocation.
    """
    from tools.osv_check import _infer_ecosystem, _parse_package_from_args

    # What the preflight sees today, before any swap.
    assert _infer_ecosystem("npx") == "npm"
    assert _parse_package_from_args(["-y", "@tacticlaunch/mcp-linear"], "npm")[0] == (
        "@tacticlaunch/mcp-linear"
    )

    # What it would see if the swap happened first — nothing.
    assert _infer_ecosystem("/home/u/.npm/_npx/abc/node_modules/.bin/mcp-linear") is None


def test_preflight_checks_npx_before_using_its_cached_binary():
    """The malware scan sees the original invocation before the direct swap."""
    events = []

    def _check(command, args):
        events.append(("osv", command, list(args)))
        return None

    def _cached(args, *, env=None, cwd=None):
        events.append(("cached", list(args), env, cwd))
        return "cached-server", ["--from-cache"]

    with patch("tools.osv_check.check_package_for_malware", side_effect=_check), \
         patch("tools.mcp_tool._npx_cached_bin", side_effect=_cached):
        from tools.mcp_tool import _preflight_stdio_command

        command, args = asyncio.run(_preflight_stdio_command(
            "server", "npx", ["-y", "mcp-linear"], env={"npm_config_cache": "configured"},
            cwd="server-cwd"))

    assert (command, args) == ("cached-server", ["--from-cache"])
    assert events == [
        ("osv", "npx", ["-y", "mcp-linear"]),
        ("cached", ["-y", "mcp-linear"], {"npm_config_cache": "configured"}, "server-cwd"),
    ]


def test_windows_selects_launchers_never_the_sh_script():
    """On Windows the extensionless sh script must never be chosen.

    npm lays down three siblings per bin — `<name>`, `<name>.cmd`,
    `<name>.ps1` — and spawning the sh one from a Windows process fails, while
    `os.access(X_OK)` there is effectively an existence check and cannot tell
    them apart. Tested through the injectable helper rather than by patching
    `os.name`, which breaks path handling process-wide (it took pytest's own
    traceback formatting down when I tried).
    """
    from tools.mcp_tool_config import _npx_bin_candidates

    win = _npx_bin_candidates("/c/bin", "mcp-linear", windows=True)
    assert win == [
        os.path.join("/c/bin", "mcp-linear.cmd"),
        os.path.join("/c/bin", "mcp-linear.exe"),
    ]
    assert not any(c.endswith("mcp-linear") for c in win), "sh script must not be a candidate"

    assert _npx_bin_candidates("/bin", "mcp-linear", windows=False) == [
        os.path.join("/bin", "mcp-linear")
    ]


def test_posix_resolution_uses_the_helper(tmp_path):
    """The resolver honours the helper's ordering (POSIX path end-to-end)."""
    target = _cache(tmp_path, package="mcp-linear", bin_field={"mcp-linear": "i.js"})

    assert _npx_cached_bin(["-y", "mcp-linear"]) == (str(target), [])


def test_flag_after_the_spec_is_left_to_npx(tmp_path):
    """`npx pkg -y` would forward -y to the server; that shape stays with npx."""
    _cache(tmp_path, package="mcp-linear", bin_field={"mcp-linear": "i.js"})

    assert _npx_cached_bin(["mcp-linear", "-y"]) is None
