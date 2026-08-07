"""_fuzzy_cache eviction: expired repo listings must not outlive their TTL.

The fuzzy-picker cache stores an entire repo file listing per root. Its TTL
was only consulted on read, so a root that stopped being queried kept its
listing pinned for the TUI gateway's lifetime — systematic with worktree
flows, where every task queries a unique ``.worktrees/<task-id>`` root once
and never again. The write path now sweeps expired entries.
"""

import time

import tui_gateway.server as srv


def test_expired_roots_swept_on_write(tmp_path):
    (tmp_path / "x.py").write_text("print('hi')\n")
    root = str(tmp_path)
    now = time.monotonic()

    with srv._fuzzy_cache_lock:
        saved = dict(srv._fuzzy_cache)
        srv._fuzzy_cache.clear()
        srv._fuzzy_cache["/worktree-a"] = (now - 3600.0, ["stale.py"] * 100)
        srv._fuzzy_cache["/worktree-b"] = (now - srv._FUZZY_CACHE_TTL_S, ["old.py"])
        srv._fuzzy_cache["/still-fresh"] = (now, ["fresh.py"])

    try:
        files = srv._list_repo_files(root)
        assert "x.py" in files

        with srv._fuzzy_cache_lock:
            assert "/worktree-a" not in srv._fuzzy_cache
            assert "/worktree-b" not in srv._fuzzy_cache
            assert "/still-fresh" in srv._fuzzy_cache
            assert root in srv._fuzzy_cache
    finally:
        with srv._fuzzy_cache_lock:
            srv._fuzzy_cache.clear()
            srv._fuzzy_cache.update(saved)


def test_fresh_entry_still_served_from_cache(tmp_path):
    """The sweep must not break the cache's purpose: a fresh entry is
    returned without relisting."""
    root = str(tmp_path)
    with srv._fuzzy_cache_lock:
        saved = dict(srv._fuzzy_cache)
        srv._fuzzy_cache.clear()
        srv._fuzzy_cache[root] = (time.monotonic(), ["cached-answer.py"])
    try:
        assert srv._list_repo_files(root) == ["cached-answer.py"]
    finally:
        with srv._fuzzy_cache_lock:
            srv._fuzzy_cache.clear()
            srv._fuzzy_cache.update(saved)
