"""Tests for the tool_output_compresr plugin.

The plugin compresses large tool outputs via the Compresr API, caches the
verbatim original under Hermes's managed cache (PR #6 cache-authority: host write
+ per-backend agent-visible path), and passes the API's compressed text through
UNCHANGED with a footer pointing the agent at the original (recoverable with
``read_file``/``search_files``). These cover the footer/pointer contract, the
transform hook's gating and fail-open behavior, the cache-authority path
translation, and size-based retention.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402

from plugins.tool_output_compresr import ToolOutputCompressor  # noqa: E402
from plugins.tool_output_compresr import cache, compress  # noqa: E402

# Large enough that a small compressed body is a genuine win even after the
# recovery footer's token budget (~90) is added back — the size gate now
# accounts for the footer, so a marginal fixture would (correctly) fail open.
ORIGINAL = "\n".join(
    [f"line{i} {word}" for i, word in enumerate(
        ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"] * 40
    )]
)

# A short stand-in for the API's compressed output. Keeps a couple of lines and
# an inline drop marker, which the plugin now passes through verbatim.
COMPRESSED = "line0 alpha\nline1 bravo\n[4 lines removed]"

CACHE_ROOT = "/root/.hermes/cache/compresr/tool-output"


def _cache_path(cache_id: str) -> str:
    return f"{CACHE_ROOT}/{cache_id}"


def _fake_env(name: str, **attrs):
    env = type(name, (), {})()
    for key, value in attrs.items():
        setattr(env, key, value)
    return env


class _FakeSyncManager:
    def __init__(self, boom: Exception | None = None):
        self.calls = []
        self.boom = boom

    def sync(self, force=False):
        self.calls.append(force)
        if self.boom is not None:
            raise self.boom


# --------------------------------------------------------------------------- #
# compress_tool_output: footer contract + fail-open
# --------------------------------------------------------------------------- #
class _FakeClient:
    def __init__(self, out=COMPRESSED, stats=None, boom=False):
        self.out, self.stats, self.boom = out, stats or {}, boom

    def compress(self, **kw):
        if self.boom:
            raise RuntimeError("HTTP 500")
        return self.out, self.stats


def test_compress_appends_footer_with_cache_pointer(monkeypatch):
    monkeypatch.setattr(cache, "store_original", lambda cid, content, task_id="default", **_: _cache_path(cid))
    out, info = compress.compress_tool_output(
        query="q", content=ORIGINAL, tool_name="grep",
        cache_id="abc", client=_FakeClient(), task_id="t",
    )
    assert info["shortened"] is True
    assert out.startswith(COMPRESSED)                 # API output verbatim
    assert "[4 lines removed]" in out                 # marker preserved, not rewritten
    assert compress.FOOTER_MARKER in out
    assert info["cache_path"] in out                  # pointer names the cache path
    assert "read_file" in out and "search_files" in out


def test_compress_failopen_on_api_error():
    out, info = compress.compress_tool_output(
        query="q", content=ORIGINAL, tool_name="grep", cache_id="c",
        client=_FakeClient(boom=True), task_id="t",
    )
    assert out == ORIGINAL
    assert info["called_api"] is False and info["shortened"] is False


def test_compress_failopen_when_not_shorter():
    out, info = compress.compress_tool_output(
        query="q", content=ORIGINAL, tool_name="grep", cache_id="c",
        client=_FakeClient(out=ORIGINAL + "\nplus more padding text here"),
        task_id="t",
    )
    assert out == ORIGINAL
    assert info["called_api"] is True and info["shortened"] is False


def test_compress_failopen_on_cache_write_failure(monkeypatch):
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: None)
    out, info = compress.compress_tool_output(
        query="q", content=ORIGINAL, tool_name="grep", cache_id="c",
        client=_FakeClient(), task_id="t",
    )
    assert out == ORIGINAL
    assert info["shortened"] is False and info["error"] == "cache write failed"


# --------------------------------------------------------------------------- #
# transform_tool_result hook
# --------------------------------------------------------------------------- #
def test_hook_skips_small_output():
    c = ToolOutputCompressor()
    c.enabled, c.api_key = True, "cmp_test"
    assert c.on_transform_tool_result(tool_name="grep", args={}, result="tiny") is None


def test_hook_compresses_large_output(monkeypatch):
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    stored = {}
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: stored.update({cid: content}) or _cache_path(cid),
    )
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "line"}, result=ORIGINAL, tool_call_id="tc1"
    )
    assert out is not None
    assert compress.FOOTER_MARKER in out
    assert out.startswith(COMPRESSED)                 # API markers preserved
    assert CACHE_ROOT in out                          # pointer present
    assert ".compresr/cache" not in out               # not the old workspace path
    assert c._cache_id(ORIGINAL) in stored


def test_hook_folds_footer_into_json_envelope(monkeypatch):
    """For unwrappable JSON tools, the footer must land inside the inner payload
    so the returned result stays valid JSON."""
    result = json.dumps({"content": ORIGINAL, "path": "/x"})
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: _cache_path(cid),
    )
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 1
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    out = c.on_transform_tool_result(
        tool_name="read_file", args={"file_path": "/x"}, result=result,
        task_id="t", tool_call_id="tc2",
    )
    assert out is not None
    parsed = json.loads(out)                           # still valid JSON
    assert compress.FOOTER_MARKER in parsed["content"]
    assert parsed["path"] == "/x"                       # sibling fields preserved


def test_hook_skips_already_compressed(monkeypatch):
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 1
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    already = "some output\n" + compress.FOOTER_MARKER + " ... cached at /x"
    assert c.on_transform_tool_result(tool_name="grep", args={}, result=already) is None


def test_hook_failopen_on_api_error(monkeypatch):
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5

    def _boom(**kw):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(c._client, "compress", _boom)
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "x"}, result=ORIGINAL, tool_call_id="tc3"
    )
    assert out is None


def test_hook_failopen_on_cache_write_failure(monkeypatch):
    """If persisting the original fails, the pointer would dangle — the hook must
    fail open to the original output rather than emit it."""
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: None)
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "x"}, result=ORIGINAL, tool_call_id="tc4"
    )
    assert out is None


def test_hook_failopen_when_not_shorter(monkeypatch):
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: _cache_path("c"))
    monkeypatch.setattr(c._client, "compress", lambda **kw: (ORIGINAL + "\nlonger now", {}))
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "x"}, result=ORIGINAL, tool_call_id="tc5"
    )
    assert out is None


# --------------------------------------------------------------------------- #
# Cache-authority (PR #6): host write + per-backend agent-visible path
# --------------------------------------------------------------------------- #
def _docker_env(byte_count=None, returncode=0):
    """A fake DockerEnvironment whose `execute` simulates the in-container
    visibility probe. byte_count None → echo the real host size back (mount
    present); otherwise return the given count / returncode (mount absent)."""
    def _execute(cmd, cwd="", **kw):
        out = "" if byte_count is None else str(byte_count)
        return {"returncode": returncode, "output": out}
    env = _fake_env("DockerEnvironment", execute=_execute)
    env._byte_count = byte_count
    return env


def test_store_original_writes_host_cache_and_returns_visible_path(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Docker mount is present: the probe reads back the real host byte size.
    real_bytes = len(ORIGINAL.encode("utf-8"))
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: _docker_env(byte_count=real_bytes))

    path = cache.store_original("abc123", ORIGINAL, task_id="task-cache", max_cache_mb=0)

    assert path == f"{CACHE_ROOT}/abc123"
    host_file = hermes_home / "cache" / "compresr" / "tool-output" / "abc123"
    assert host_file.read_text(encoding="utf-8") == ORIGINAL


def test_store_original_docker_mount_missing_fails_open(monkeypatch, tmp_path):
    """Fix #1: a reused Docker container that never bind-mounted the cache dir
    (the probe can't read the file) must fail open, not hand out a dangling
    recovery pointer. The host file is retained for the pruner."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        cache, "_get_active_env", lambda task_id: _docker_env(byte_count=0, returncode=1)
    )
    assert cache.store_original("missing-mount", ORIGINAL, task_id="t", max_cache_mb=0) is None
    assert (hermes_home / "cache" / "compresr" / "tool-output" / "missing-mount").exists()


def test_store_original_docker_size_mismatch_fails_open(monkeypatch, tmp_path):
    """Fix #1: probe returns a different byte count (a stale/other file shadows
    the mount point) → treat as not visible and fail open."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        cache, "_get_active_env", lambda task_id: _docker_env(byte_count=999999)
    )
    assert cache.store_original("stale-shadow", ORIGINAL, task_id="t", max_cache_mb=0) is None


def test_store_original_docker_no_exec_primitive_fails_open(monkeypatch, tmp_path):
    """Fix #1: a Docker env we can't probe (no callable execute) can't prove
    visibility → fail open."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: _fake_env("DockerEnvironment"))
    assert cache.store_original("no-exec", ORIGINAL, task_id="t", max_cache_mb=0) is None


def test_store_original_fails_open_when_visibility_unknown(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: _fake_env("SingularityEnvironment"))

    assert cache.store_original("no-visible-path", ORIGINAL, task_id="task-cache", max_cache_mb=0) is None
    # Content-addressed: the host file is retained (not unlinked) on the
    # fail-open path so a concurrent sibling's recovery pointer can't dangle;
    # the size-based pruner reclaims it later.
    assert (hermes_home / "cache" / "compresr" / "tool-output" / "no-visible-path").exists()


def test_store_original_with_no_active_env_uses_host_path(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: None)

    path = cache.store_original("host-path", ORIGINAL, task_id="task-cache", max_cache_mb=0)

    assert path == str(hermes_home / "cache" / "compresr" / "tool-output" / "host-path")
    assert (hermes_home / "cache" / "compresr" / "tool-output" / "host-path").exists()


def test_store_original_force_syncs_remote_cache_before_return(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    sync_manager = _FakeSyncManager()
    env = _fake_env("SSHEnvironment", _remote_home="/home/agent", _sync_manager=sync_manager)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: env)

    path = cache.store_original("synced", ORIGINAL, task_id="task-sync", max_cache_mb=0)

    assert sync_manager.calls == [True]
    assert path == "/home/agent/.hermes/cache/compresr/tool-output/synced"
    assert (hermes_home / "cache" / "compresr" / "tool-output" / "synced").exists()


def test_store_original_force_sync_failure_retains_host_file(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    sync_manager = _FakeSyncManager(boom=RuntimeError("sync failed"))
    env = _fake_env("SSHEnvironment", _remote_home="/home/agent", _sync_manager=sync_manager)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: env)

    assert cache.store_original("sync-fail", ORIGINAL, task_id="task-sync", max_cache_mb=0) is None
    assert sync_manager.calls == [True]
    # Content-addressed: retained, not unlinked — a sibling compression of
    # identical content may already hold a recovery pointer to this exact file.
    assert (hermes_home / "cache" / "compresr" / "tool-output" / "sync-fail").exists()


# --------------------------------------------------------------------------- #
# Retention (size-based prune)
# --------------------------------------------------------------------------- #
def test_cache_prune_evicts_oldest_over_budget(tmp_path):
    from plugins.tool_output_compresr.cache import _prune_cache_dir

    old = tmp_path / "old"
    newer = tmp_path / "newer"
    current = tmp_path / "current"
    old.write_text("x" * 10, encoding="utf-8")
    newer.write_text("y" * 10, encoding="utf-8")
    current.write_text("z" * 10, encoding="utf-8")
    os.utime(old, (1, 1))
    os.utime(newer, (2, 2))
    os.utime(current, (3, 3))

    _prune_cache_dir(str(tmp_path), 20, str(current))

    assert not old.exists()
    assert newer.exists()
    assert current.exists()
    assert sum(path.stat().st_size for path in tmp_path.iterdir() if path.is_file()) <= 20


def test_cache_prune_disabled_leaves_files(tmp_path):
    from plugins.tool_output_compresr.cache import _prune_cache_dir

    old = tmp_path / "old"
    current = tmp_path / "current"
    old.write_text("x" * 10, encoding="utf-8")
    current.write_text("z" * 10, encoding="utf-8")

    _prune_cache_dir(str(tmp_path), 0, str(current))

    assert old.exists()
    assert current.exists()


def test_cache_prune_host_root(tmp_path):
    from plugins.tool_output_compresr.cache import _prune_cache_dir

    old = tmp_path / "old"
    current = tmp_path / "current"
    old.write_text("x" * 10, encoding="utf-8")
    current.write_text("z" * 10, encoding="utf-8")
    # Age the eviction candidate past the recovery pin window so size-based
    # prune reclaims it (recent entries are pinned; see pin-recent).
    os.utime(old, (1, 1))

    _prune_cache_dir(str(tmp_path), 1, str(current))

    assert not old.exists()
    assert current.exists()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_tool_output_api_key_is_env_only(monkeypatch, tmp_path):
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compresr:\n"
        "  api_key: cmp_from_config\n"
        "  tool_output_enabled: true\n"
        "  tool_output_min_tokens: 7\n"
        "  tool_output_max_cache_mb: 3\n",
        encoding="utf-8",
    )

    c = ToolOutputCompressor()
    assert c.api_key == ""            # secrets are env-only
    assert c.enabled is True
    assert c.min_tokens == 7
    assert c.max_cache_mb == 3
    assert not c.active               # no key → inactive


# --------------------------------------------------------------------------- #
# H1: recovery reads of a cached original must not be re-compressed
# --------------------------------------------------------------------------- #
def test_hook_skips_recovery_read_of_cached_original(monkeypatch):
    """A read_file targeting a compresr cache file returns the verbatim original
    (None from the hook) instead of being re-compressed into a lossy summary."""
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    # If the guard fails, this would fire and mangle the recovery.
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: _cache_path("x"))

    # Container-translated path (backend-agnostic substring match).
    out = c.on_transform_tool_result(
        tool_name="read_file",
        args={"file_path": f"{CACHE_ROOT}/deadbeef"},
        result=ORIGINAL,
        tool_call_id="rec1",
    )
    assert out is None
    assert c.recoveries == 1
    assert c.get_status()["recoveries"] == 1


def test_hook_recovery_guard_matches_host_cache_root(monkeypatch, tmp_path):
    """The guard also matches a resolved host cache path under get_cache_root()."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    host_path = str(cache.get_cache_root() / "abc123")
    out = c.on_transform_tool_result(
        tool_name="read_file", args={"file_path": host_path}, result=ORIGINAL,
    )
    assert out is None
    assert c.recoveries == 1


# --------------------------------------------------------------------------- #
# H2: search_files unwraps its real dense shape ("matches_text")
# --------------------------------------------------------------------------- #
def test_hook_unwraps_search_files_matches_text(monkeypatch):
    """search_files emits {"total_count":N,"matches_text":"..."} (never a
    top-level "content"). The plugin must unwrap matches_text, keep the result
    valid JSON, and fold the footer into matches_text."""
    matches = "\n".join(f"file{i}.py\n  {i}: match line {i}" for i in range(60))
    result = json.dumps({"total_count": 60, "matches_text": matches})
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: _cache_path(cid),
    )
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 1
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    out = c.on_transform_tool_result(
        tool_name="search_files", args={"pattern": "match"}, result=result,
        tool_call_id="sf1",
    )
    assert out is not None
    parsed = json.loads(out)                            # still valid JSON
    assert parsed["total_count"] == 60                  # sibling field preserved
    assert compress.FOOTER_MARKER in parsed["matches_text"]
    assert parsed["matches_text"].startswith(COMPRESSED)
    assert "content" not in parsed


# --------------------------------------------------------------------------- #
# H3: a real FileSyncManager whose upload raises fails open (host file retained)
# --------------------------------------------------------------------------- #
def test_store_original_real_sync_upload_failure_fails_open(monkeypatch, tmp_path):
    from tools.environments.file_sync import FileSyncManager

    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _boom_upload(host_path, remote_path):
        raise RuntimeError("transport down")

    # A real manager: sync() catches the transport error internally, so without
    # raise_on_error the failure would be swallowed and store_original would hand
    # back a remote path for a file that was never uploaded.
    sm = FileSyncManager(
        get_files_fn=lambda: [(str(hermes_home / "x"), "/root/.hermes/x")],
        upload_fn=_boom_upload,
        delete_fn=lambda paths: None,
    )
    # Force the initial file to look new so sync attempts an upload.
    (hermes_home).mkdir(parents=True, exist_ok=True)
    (hermes_home / "x").write_text("seed", encoding="utf-8")

    env = _fake_env("SSHEnvironment", _remote_home="/home/agent", _sync_manager=sm)
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: env)

    assert cache.store_original("sync-boom", ORIGINAL, task_id="t", max_cache_mb=0) is None
    # Fail open (returns None) but retain the content-addressed host file so a
    # concurrent sibling's recovery pointer can't dangle; pruner reclaims later.
    assert (hermes_home / "cache" / "compresr" / "tool-output" / "sync-boom").exists()


# --------------------------------------------------------------------------- #
# M1: a marginally-shorter compression must not come back net-larger + footer
# --------------------------------------------------------------------------- #
def test_compress_failopen_on_marginal_shrink(monkeypatch):
    """The size gate now accounts for the ~90-token footer, so a body only a few
    tokens smaller than the original fails open (no net growth, no cache write)."""
    stored = {}
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: stored.update({cid: content}) or _cache_path(cid),
    )
    base = "x" * 1000                                   # ~250 tokens
    marginal = "x" * 990                                # ~2.5 tokens shorter — below footer budget
    out, info = compress.compress_tool_output(
        query="q", content=base, tool_name="grep", cache_id="marg",
        client=_FakeClient(out=marginal), task_id="t",
    )
    assert out == base
    assert info["shortened"] is False
    assert "marg" not in stored                         # cache NOT written on failure


# --------------------------------------------------------------------------- #
# M2: whitespace-only compressed output is treated as a failure (fail open)
# --------------------------------------------------------------------------- #
def test_compress_failopen_on_whitespace_only_output(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: stored.update({cid: content}) or _cache_path(cid),
    )
    out, info = compress.compress_tool_output(
        query="q", content=ORIGINAL, tool_name="grep", cache_id="ws",
        client=_FakeClient(out="  \n  \t "), task_id="t",
    )
    assert out == ORIGINAL
    assert info["shortened"] is False
    assert info["error"] == "empty compressed output"
    assert "ws" not in stored


def test_hook_failopen_on_whitespace_only_output(monkeypatch):
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: _cache_path("c"))
    monkeypatch.setattr(c._client, "compress", lambda **kw: ("   \n  ", {}))
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "x"}, result=ORIGINAL, tool_call_id="ws1"
    )
    assert out is None


# --------------------------------------------------------------------------- #
# M4: recently-written cache entries are pinned against size-based prune
# --------------------------------------------------------------------------- #
def test_prune_pins_recently_written_entries(tmp_path):
    from plugins.tool_output_compresr.cache import _prune_cache_dir

    old = tmp_path / "old"
    fresh = tmp_path / "fresh"
    current = tmp_path / "current"
    for p in (old, fresh, current):
        p.write_text("x" * 10, encoding="utf-8")
    # Only `old` is aged past the pin window; `fresh` is a footer path a sibling
    # just handed to the model and must survive even though the dir is over budget.
    os.utime(old, (1, 1))

    _prune_cache_dir(str(tmp_path), 1, str(current))

    assert not old.exists()      # aged → evictable
    assert fresh.exists()        # recent → pinned, survives prune
    assert current.exists()      # explicit keep


import plugins.tool_output_compresr as toc  # noqa: E402


def test_gutter_helpers():
    assert toc._is_fully_guttered("1|def f():\n2|    return 1")
    assert not toc._is_fully_guttered("plain text\nno gutter")
    # a line that merely contains a pipe must not be mistaken for a gutter
    assert not toc._is_fully_guttered("a | b table")
    assert toc._strip_line_gutter("3|def f():\n4|    return 1") == "def f():\n    return 1"
    assert toc._strip_line_gutter("a | b") == "a | b"  # untouched


def test_read_file_output_cached_denumbered(monkeypatch):
    """read_file content arrives line-numbered (N|code); the cache must store a
    DE-numbered copy so a recovery read_file re-adds exactly one gutter, not two."""
    body = "\n".join("%d|line %d content here" % (i, i) for i in range(1, 60))
    result = json.dumps({"content": body, "path": "/x"})
    stored = {}
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: stored.update({"content": content})
        or "/hermes/cache/compresr/tool-output/" + cid,
    )
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 1
    monkeypatch.setattr(c._client, "compress", lambda **kw: ("line 1 content here\n[56 lines removed]", {}))
    out = c.on_transform_tool_result(
        tool_name="read_file", args={"file_path": "/x"}, result=result,
        task_id="t", tool_call_id="tc-num",
    )
    assert out is not None
    # what was cached is the DE-numbered content (no "N|" gutters)
    cached = stored["content"]
    assert cached.startswith("line 1 content here")
    assert "1|line 1" not in cached
    assert cached == "\n".join("line %d content here" % i for i in range(1, 60))


# --------------------------------------------------------------------------- #
# C: the size gate is EXACT — a long cache path whose footer exceeds the nominal
# FOOTER_TOKEN_BUDGET must not sneak a net-larger output past the gate.
# --------------------------------------------------------------------------- #
def test_compress_failopen_when_long_cache_path_makes_output_net_larger(monkeypatch):
    """A body a bit under the nominal budget passes the cheap pre-filter, but a
    very long remote cache path makes the real footer cost far more than
    FOOTER_TOKEN_BUDGET. The exact post-footer gate must catch it and fail open."""
    long_path = "/root/.hermes/cache/compresr/tool-output/" + ("a" * 560)
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: long_path,
    )
    base = "x" * 4000            # ~1000 tokens
    body = "x" * 3400            # ~850 tokens: 850 + 90 (budget) < 1000 → passes pre-filter
    out, info = compress.compress_tool_output(
        query="q", content=base, tool_name="grep", cache_id="longpath",
        client=_FakeClient(out=body), task_id="t",
    )
    assert out == base                                   # failed open, original returned
    assert info["shortened"] is False
    # A benign "no net win" is NOT a failure — no error key (so the hook doesn't
    # back off / count it); it is recorded as a skipped_reason instead.
    assert "error" not in info
    assert info["skipped_reason"] == "not smaller after footer"


# --------------------------------------------------------------------------- #
# D: HTTP contract of the tool-output client (payload shape, headers, envelope).
# Kept unmocked at the urllib layer so envelope-key drift (compressed_output vs
# the engine's compressed_context) fails loudly.
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, body):
        self._b = body.encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_tool_output_client_builds_request_and_parses_compressed_output(monkeypatch):
    from plugins.tool_output_compresr.client import CompresrToolOutputClient

    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _FakeResp(json.dumps({"success": True, "data": {
            "compressed_output": "SHORT\n[3 lines removed]", "tokens_saved": 7,
        }}))

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    client = CompresrToolOutputClient(api_key="cmp_secret", model="toc_latte_v2")
    text, data = client.compress(
        tool_output="a huge dump", query="grep: foo", tool_name="grep",
        target_ratio=2.0,
    )

    assert text == "SHORT\n[3 lines removed]"
    assert data["tokens_saved"] == 7
    req = captured["req"]
    assert req.full_url.endswith("/compress/tool-output/")
    assert req.get_header("X-api-key") == "cmp_secret"
    assert req.get_header("User-agent") == CompresrToolOutputClient.USER_AGENT
    assert req.get_header("Content-type") == "application/json"
    payload = json.loads(req.data)
    assert payload["tool_output"] == "a huge dump"
    assert payload["query"] == "grep: foo"
    assert payload["tool_name"] == "grep"
    assert payload["compression_model_name"] == "toc_latte_v2"
    assert payload["source"] == "integration:hermes"
    assert payload["coarse"] is True
    assert payload["disable_placeholders"] is False
    assert payload["target_compression_ratio"] == 2.0


def test_tool_output_client_raises_on_api_failure_and_empty(monkeypatch):
    import pytest as _pytest
    import urllib.request
    from plugins.tool_output_compresr.client import CompresrToolOutputClient

    client = CompresrToolOutputClient(api_key="cmp_secret")

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(json.dumps(
                            {"success": False, "message": "bad source"})))
    with _pytest.raises(RuntimeError):
        client.compress(tool_output="x", query="q", tool_name="grep")

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(json.dumps(
                            {"success": True, "data": {"compressed_output": ""}})))
    with _pytest.raises(RuntimeError):
        client.compress(tool_output="x", query="q", tool_name="grep")


def test_tool_output_client_raises_on_http_error_with_detail(monkeypatch):
    import io
    import pytest as _pytest
    import urllib.error
    import urllib.request
    from plugins.tool_output_compresr.client import CompresrToolOutputClient

    def _http_err(req, timeout=None):
        raise urllib.error.HTTPError(
            "u", 422, "Unprocessable", None, io.BytesIO(b'{"detail":"bad source"}')
        )

    monkeypatch.setattr(urllib.request, "urlopen", _http_err)
    client = CompresrToolOutputClient(api_key="cmp_secret")
    with _pytest.raises(RuntimeError) as ei:
        client.compress(tool_output="x", query="q", tool_name="grep")
    assert "422" in str(ei.value)


def test_tool_output_client_requires_key_and_output():
    import pytest as _pytest
    from plugins.tool_output_compresr.client import CompresrToolOutputClient

    with _pytest.raises(RuntimeError):
        CompresrToolOutputClient(api_key="").compress(
            tool_output="x", query="q", tool_name="grep")
    with _pytest.raises(RuntimeError):
        CompresrToolOutputClient(api_key="k").compress(
            tool_output="", query="q", tool_name="grep")


def test_secure_base_url_rejects_cloud_metadata_ip():
    from plugins.tool_output_compresr import _secure_base_url

    default = "https://api.compresr.ai"
    assert _secure_base_url("https://169.254.169.254/", default) == default
    assert _secure_base_url("https://metadata.google.internal/", default) == default


def test_client_url_error_routed_through_runtime_error(monkeypatch):
    # Regression: urllib.error.URLError (DNS failure, connect refused,
    # socket.timeout) is not a subclass of HTTPError; must be caught explicitly
    # so callers get a fail-open-worthy RuntimeError instead of a raw URLError.
    import urllib.error
    import urllib.request as ureq
    from plugins.tool_output_compresr.client import CompresrToolOutputClient

    def _boom(*_a, **_kw):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(ureq, "urlopen", _boom)
    with pytest.raises(RuntimeError, match="connection error"):
        CompresrToolOutputClient(api_key="cmp_test").compress(
            tool_output="hello", query="q", tool_name="grep"
        )


def test_client_json_decode_error_routed_through_runtime_error(monkeypatch):
    import urllib.request as ureq
    from plugins.tool_output_compresr.client import CompresrToolOutputClient

    class _Resp:
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def read(self):
            return b"<html>gateway error</html>"

    monkeypatch.setattr(ureq, "urlopen", lambda *a, **kw: _Resp())
    with pytest.raises(RuntimeError, match="non-JSON response"):
        CompresrToolOutputClient(api_key="cmp_test").compress(
            tool_output="hello", query="q", tool_name="grep"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
