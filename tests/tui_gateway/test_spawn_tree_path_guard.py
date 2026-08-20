"""Regression tests: spawn_tree session dirs must survive hostile ids.

``_spawn_tree_session_dir`` builds a directory name from ``session_id``.
Callers pass it through ``str(params.get("session_id") or "")``, so a client
can make the name arbitrarily long (a long id, or the str() of a dict/list
param). The sanitizer filters characters but not length, so the resulting
path can exceed filesystem limits. On Windows, ``Path.mkdir`` raises
``OSError [WinError 123]`` once the full path passes MAX_PATH (260).

``spawn_tree.list`` and ``spawn_tree.save`` run inline on the gateway's
stdin reader thread, where no exception is caught, so that one frame kills
the gateway process.

The fix caps the sanitized segment at 64 chars (truncate + sha256 tail so
long ids still map to distinct directories).
"""

import tui_gateway.server as server


def test_long_session_id_still_builds_dir():
    d = server._spawn_tree_session_dir("a" * 300)
    name = d.name
    assert d.is_dir()
    assert len(name) <= 64

def test_two_long_ids_map_to_different_dirs():
    d1 = server._spawn_tree_session_dir("a" * 300)
    d2 = server._spawn_tree_session_dir("b" * 300)
    assert d1 != d2

def test_same_long_id_maps_to_same_dir():
    d1 = server._spawn_tree_session_dir("x" * 300)
    d2 = server._spawn_tree_session_dir("x" * 300)
    assert d1 == d2

def test_normal_id_keeps_full_name():
    d = server._spawn_tree_session_dir("abc-123_def")
    assert d.name == "abc-123_def"

def test_spawn_tree_list_with_hostile_id_returns_response_not_crash():
    resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "r1",
            "method": "spawn_tree.list",
            "params": {"session_id": "a" * 300},
        }
    )
    assert isinstance(resp, dict)
    assert "error" in resp or "result" in resp
