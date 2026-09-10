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

import json
import os

import pytest

from tools.mcp_tool import _npx_cached_bin


def _cache(tmp_path, *, package, deps=None, bin_field, make_bin=True, entry="abc123"):
    """Build a fake npx cache entry the way npm lays one out."""
    root = tmp_path / ".npm" / "_npx" / entry
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
    target = bindir / (os.path.basename(package) if isinstance(bin_field, str) else name)
    if make_bin:
        target.write_text("#!/usr/bin/env node\n", encoding="utf-8")
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


def test_swap_happens_after_the_osv_call_in_source():
    """Structural guard for the ordering above.

    The swap and the preflight live in one async function; a future edit that
    moves the swap earlier would disable the malware gate silently, and no
    unit test of either piece alone would notice.
    """
    from pathlib import Path as _P

    src = _P(__file__).resolve().parents[2] / "tools" / "mcp_tool.py"
    text = src.read_text(encoding="utf-8")
    osv_needle = "check_package_for_malware, command, args"
    swap_needle = "cached = _npx_cached_bin(args)"
    # Report a rename explicitly: a bare .index() ValueError here reads like a
    # broken test rather than "someone renamed the thing this guards".
    assert osv_needle in text, (
        f"cannot find the OSV preflight call ({osv_needle!r}) — it was renamed; "
        "update this guard and re-verify the swap still happens after it"
    )
    assert swap_needle in text, (
        f"cannot find the npx swap ({swap_needle!r}) — it was renamed; update "
        "this guard and re-verify it still happens after the OSV preflight"
    )

    assert text.index(osv_needle) < text.index(swap_needle), (
        "the npx swap now precedes the OSV malware preflight, which silently "
        "disables it: _infer_ecosystem keys off the command basename being "
        "npx/uvx/pipx, so a rewritten command yields no ecosystem and "
        "check_package_for_malware returns None"
    )


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
    assert win == ["/c/bin/mcp-linear.cmd", "/c/bin/mcp-linear.exe"]
    assert not any(c.endswith("mcp-linear") for c in win), "sh script must not be a candidate"

    assert _npx_bin_candidates("/bin", "mcp-linear", windows=False) == ["/bin/mcp-linear"]


def test_posix_resolution_uses_the_helper(tmp_path):
    """The resolver honours the helper's ordering (POSIX path end-to-end)."""
    target = _cache(tmp_path, package="mcp-linear", bin_field={"mcp-linear": "i.js"})

    assert _npx_cached_bin(["-y", "mcp-linear"]) == (str(target), [])


def test_flag_after_the_spec_is_left_to_npx(tmp_path):
    """`npx pkg -y` would forward -y to the server; that shape stays with npx."""
    _cache(tmp_path, package="mcp-linear", bin_field={"mcp-linear": "i.js"})

    assert _npx_cached_bin(["mcp-linear", "-y"]) is None
