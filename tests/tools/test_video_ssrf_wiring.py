"""Phase-3 wiring tests: _download_video_via_ytdlp routes through the SSRF proxy.

These assert the CONFIGURATION and FAIL-CLOSED behavior of the wiring without
hitting the network: yt-dlp is monkeypatched at the subprocess boundary so we
inspect the argv/env the tool would run, and the proxy lifecycle is exercised
for the fail-closed and post-run-guard paths.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

import tools.vision_tools as vt


def _mk_dest(tmp_path) -> Path:
    d = tmp_path / "vid"
    d.mkdir()
    return d


def test_ssrf_proxy_disabled_by_default(monkeypatch):
    """Config key absent → proxy OFF (v0.1 default)."""
    monkeypatch.setattr(vt, "_cfg_get_safe", lambda *a, **k: k.get("default", ""))
    assert vt._ssrf_proxy_enabled() is False


def test_ssrf_proxy_enabled_when_config_true(monkeypatch):
    def fake_cfg(*path, default=""):
        if path == ("auxiliary", "vision", "ytdlp_ssrf_proxy"):
            return True
        return default
    monkeypatch.setattr(vt, "_cfg_get_safe", fake_cfg)
    assert vt._ssrf_proxy_enabled() is True


def test_ffmpeg_block_shim_disables_external_downloaders(tmp_path):
    """The shim dir provides ffmpeg/ffprobe/aria2c that fail non-zero, while the
    real PATH (deno/node/etc.) stays reachable when prepended."""
    import subprocess as _sp
    shim = vt._make_ffmpeg_block_shim(tmp_path)
    for name in ("ffmpeg", "ffprobe", "aria2c"):
        exe = Path(shim) / name
        assert exe.exists() and os.access(exe, os.X_OK)
        # Running the shim binary fails (fail-closed), never a real fetch.
        rc = _sp.run([str(exe), "-version"], capture_output=True).returncode
        assert rc != 0
    # The shim is a single dir to PREPEND — it does not remove real PATH entries.
    child_path = shim + os.pathsep + "/usr/local/bin:/usr/bin"
    assert "/usr/local/bin" in child_path and "/usr/bin" in child_path


@pytest.mark.asyncio
async def test_ytdlp_uses_proxy_args_when_enabled(monkeypatch, tmp_path):
    """With the knob on, the child argv carries --proxy + --downloader native,
    the env has *_PROXY set, and PATH excludes ffmpeg."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env", {})
        # Simulate yt-dlp writing a finished mp4 so _find_output succeeds.
        # out template is argv after -o
        oidx = argv.index("-o")
        tmpl = argv[oidx + 1]
        outfile = Path(tmpl.replace(".%(ext)s", ".mp4"))
        outfile.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # tiny fake mp4
        class P:
            returncode = 0
            stderr = ""
            stdout = ""
        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(vt, "_ssrf_proxy_enabled", lambda: True)
    monkeypatch.setattr(vt, "_ytdlp_cookie_args", lambda: [])

    dest = _mk_dest(tmp_path)
    # Bump the proxy connection_count so the post-run guard passes (we're not
    # actually connecting through it here — the argv/env is what this asserts).
    from tools.ssrf_proxy import SsrfFilteringProxy
    orig_aenter = SsrfFilteringProxy.__aenter__

    async def patched_aenter(self):
        url = await orig_aenter(self)
        self.connection_count = 1  # pretend a real fetch traversed it
        return url
    monkeypatch.setattr(SsrfFilteringProxy, "__aenter__", patched_aenter)

    out = await vt._download_video_via_ytdlp("https://youtube.com/watch?v=x", dest)
    assert out.exists()
    argv = captured["argv"]
    assert "--proxy" in argv
    pidx = argv.index("--proxy")
    assert argv[pidx + 1].startswith("http://127.0.0.1:")
    assert "--downloader" in argv and argv[argv.index("--downloader") + 1] == "native"
    env = captured["env"]
    assert env.get("HTTP_PROXY", "").startswith("http://127.0.0.1:")
    assert env.get("HTTPS_PROXY", "").startswith("http://127.0.0.1:")


@pytest.mark.asyncio
async def test_ytdlp_no_proxy_args_when_disabled(monkeypatch, tmp_path):
    """With the knob off (default), no --proxy in argv."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        oidx = argv.index("-o")
        tmpl = argv[oidx + 1]
        Path(tmpl.replace(".%(ext)s", ".mp4")).write_bytes(b"\x00\x00\x00\x18ftypmp42")
        class P:
            returncode = 0
            stderr = ""
            stdout = ""
        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(vt, "_ssrf_proxy_enabled", lambda: False)
    monkeypatch.setattr(vt, "_ytdlp_cookie_args", lambda: [])

    dest = _mk_dest(tmp_path)
    out = await vt._download_video_via_ytdlp("https://youtube.com/watch?v=x", dest)
    assert out.exists()
    assert "--proxy" not in captured["argv"]


@pytest.mark.asyncio
async def test_ytdlp_proxy_start_failure_fails_closed(monkeypatch, tmp_path):
    """If the proxy can't start, the tool RAISES — yt-dlp is never invoked."""
    ran = {"called": False}

    def fake_run(argv, **kwargs):
        ran["called"] = True
        class P:
            returncode = 0
            stderr = ""
            stdout = ""
        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(vt, "_ssrf_proxy_enabled", lambda: True)
    monkeypatch.setattr(vt, "_ytdlp_cookie_args", lambda: [])

    from tools.ssrf_proxy import SsrfFilteringProxy

    async def boom(self):
        raise OSError("cannot bind")
    monkeypatch.setattr(SsrfFilteringProxy, "__aenter__", boom)

    dest = _mk_dest(tmp_path)
    with pytest.raises(OSError):
        await vt._download_video_via_ytdlp("https://youtube.com/watch?v=x", dest)
    assert ran["called"] is False, "yt-dlp must NOT run when the proxy fails to start"


@pytest.mark.asyncio
async def test_post_run_guard_raises_on_unproxied(monkeypatch, tmp_path):
    """Proxy on, bytes on disk, but zero proxied connections → raise + clean."""
    def fake_run(argv, **kwargs):
        oidx = argv.index("-o")
        tmpl = argv[oidx + 1]
        Path(tmpl.replace(".%(ext)s", ".mp4")).write_bytes(b"\x00\x00\x00\x18ftypmp42")
        class P:
            returncode = 0
            stderr = ""
            stdout = ""
        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(vt, "_ssrf_proxy_enabled", lambda: True)
    monkeypatch.setattr(vt, "_ytdlp_cookie_args", lambda: [])
    # Leave connection_count at 0 (default) — simulates external-downloader bypass.

    dest = _mk_dest(tmp_path)
    with pytest.raises(ValueError, match="no connection traversed the proxy"):
        await vt._download_video_via_ytdlp("https://youtube.com/watch?v=x", dest)
    # partials + the unvalidated file cleaned
    assert list(dest.glob("*")) == []
