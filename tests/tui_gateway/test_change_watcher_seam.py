"""Seam tests for the R2-S1 extraction: change watcher → tui_gateway/change_watcher.py.

Proves the extraction's load-bearing seam (consensus R2 §4, tests T1–T8):
- re-export identity: every moved name on ``tui_gateway.server`` IS the object
  in ``tui_gateway.change_watcher``;
- patch-inertness: state replaced via ``monkeypatch.setattr(server, ...)`` /
  ``monkeypatch.setitem(server._CHANGE_WATCHES, ...)`` is read through the
  server module object at call time by the moved code;
- import-order cycle permutation: change_watcher-first and server-first both
  import clean;
- aggressive behavior: file create/modify/delete lifecycle and a broken-probe
  error path never kill the pass.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from tui_gateway import change_watcher, server

MOVED_NAMES = (
    "resolve_skin",
    "_skin_sig",
    "_note_skin_broadcast",
    "_broadcast_skin_if_changed",
    "_watcher_home",
    "_pet_sig",
    "_pet_changed_payload",
    "_cron_sig",
    "_sessions_sig",
    "_platforms_sig",
    "_pairing_sig",
    "_broadcast_watched_changes",
    "_ensure_skin_watcher",
)


@pytest.fixture()
def watcher_home(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("display: {}\n")
    (tmp_path / "cron").mkdir()

    monkeypatch.setattr(server, "_hermes_home", str(tmp_path))
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_change_sigs", {})
    monkeypatch.setattr(server, "_change_checked_at", {})
    monkeypatch.setattr(server, "_change_broadcast_at", {})
    # No broadcast floor for the lifecycle tests: raw signature moves.
    monkeypatch.setattr(server, "_CHANGE_BROADCAST_FLOOR_S", {})

    events = []
    monkeypatch.setattr(
        server,
        "_broadcast_global_event",
        lambda ev, payload=None: events.append((ev, payload)),
    )
    return tmp_path, events


def test_reexport_identity_per_moved_name():
    """T1: every re-exported name on server is the moved object (identity)."""
    for name in MOVED_NAMES:
        assert getattr(server, name) is getattr(change_watcher, name), name


def test_global_state_placement():
    """`_skin_watcher_started` moved with the cluster (global-bound, unpatched);
    `_last_skin_sig` stayed server-owned (test_protocol patches it on server)."""
    assert change_watcher._skin_watcher_started is False
    assert not hasattr(server, "_skin_watcher_started")
    assert server._last_skin_sig is None
    assert not hasattr(change_watcher, "_last_skin_sig")


def test_patch_liveness_setattr_seen_by_moved_code(watcher_home):
    """T2: setattr(server, "_change_sigs", {...}) is read at call time.

    The moved ``_broadcast_watched_changes`` must seed the NEW dict we
    installed on the server module — a stale import-time binding would seed a
    dead dict and this assertion would fail.
    """
    home, events = watcher_home
    (home / "cron" / "jobs.json").write_text("[]")
    (home / "state.db").write_text("x")

    server._broadcast_watched_changes(now=0.0)

    assert server._change_sigs  # seeded into the freshly setattr'd dict
    assert events == []


def test_setitem_identity_broken_probe_never_kills_pass(watcher_home):
    """T3 + error path: setitem on server._CHANGE_WATCHES is seen by identity,
    and a raising probe is skipped while a healthy probe still broadcasts."""
    home, events = watcher_home
    server._broadcast_watched_changes(now=0.0)

    def _boom():
        raise RuntimeError("probe exploded")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(
        server._CHANGE_WATCHES,
        "cron.changed",
        (1.0, _boom, lambda: {}),
    )
    try:
        (home / "state.db").write_text("x")
        server._broadcast_watched_changes(now=10.0)

        assert ("sessions.changed", {}) in events
        assert not [e for e in events if e[0] == "cron.changed"]
    finally:
        monkeypatch.undo()


def test_hermes_home_patch_visibility(watcher_home):
    """T4: _hermes_home patched on server is what the moved probes see."""
    home, _ = watcher_home
    assert change_watcher._watcher_home() == Path(str(home))
    # The skin signature must resolve against the patched home too.
    assert change_watcher._skin_sig()[0] == "default"
    assert change_watcher._skin_sig()[1] is None  # no skins dir → no mtime


def test_file_create_modify_delete_lifecycle(watcher_home):
    """Aggressive: state.db create → broadcast, modify → broadcast,
    delete → broadcast (signature returns to None and must move again)."""
    home, events = watcher_home
    db = home / "state.db"
    server._broadcast_watched_changes(now=0.0)  # seed: absent → None

    db.write_text("x")  # create
    server._broadcast_watched_changes(now=10.0)
    assert ("sessions.changed", {}) in events
    events.clear()

    time.sleep(0.02)  # NTFS mtime granularity
    db.write_text("xy")  # modify
    server._broadcast_watched_changes(now=11.0)
    assert ("sessions.changed", {}) in events
    events.clear()

    db.unlink()  # delete
    server._broadcast_watched_changes(now=12.0)
    assert ("sessions.changed", {}) in events


def test_import_order_cycle_permutation():
    """T8: change_watcher-first and server-first both import clean, and the
    external pins (entry, ws) still import."""
    repo_root = Path(__file__).resolve().parents[2]
    probes = (
        "import tui_gateway.change_watcher; import tui_gateway.server",
        "import tui_gateway.server; import tui_gateway.change_watcher",
        "import tui_gateway.server; import tui_gateway.entry; import tui_gateway.ws",
    )
    for probe in probes:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"{probe!r} failed:\n{proc.stdout}\n{proc.stderr}"
