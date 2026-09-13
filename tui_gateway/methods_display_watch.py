"""Cross-process lease watcher: ``display.lease`` for transitions made OUTSIDE ``hermes serve``.

The lease (``tools.bot_desktop.lease``) lives on disk and is changed by whatever process hosts the
agent — the messaging gateway's ``request_handoff``, a ``hermes chat`` takeover, a cron worker's
release. ``lease.on_change`` only fires in the writing process, so ``methods_display``'s in-process
listener never sees those; the Desktop's "Bot needs you" badge and hero tone stayed stale until the
pane was reopened. One daemon thread stats every served home's ``bot-desktop/lease.json`` (launch
home + ``_served_profile_homes``) every 0.5s and broadcasts the SAME ``display.lease`` payload when
the epoch moves. Bodies are rebound onto server.py's globals (method_ctx.bind_module).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .method_ctx import bind_module

_LEASE_POLL_S = 0.5
_lease_watcher_started = threading.Event()
# profile key → last epoch broadcast or seen (in-process transitions record theirs too, so a change
# made by THIS process is not re-broadcast when its file write is noticed a tick later).
_lease_epochs: dict[str, int] = {}
_lease_mtimes: dict[str, int | None] = {}
# profile key → (env mtime, launcher.pid mtime): the screen's running/display identity. A start,
# stop or crash made by another process (CLI, gateway auto-start) moves one of these.
_runtime_marks: dict[str, tuple] = {}


def _lease_event_payload(profile_key: str, lease) -> dict:
    # Same shape and the same redaction (viewer_hash, never the raw id) as the in-process broadcast.
    from tui_gateway.methods_display import _lease_view
    return {"profile_key": profile_key, "lease": _lease_view(lease)}


def _watched_lease_homes() -> list[Path]:
    return [Path(_hermes_home), *_served_profile_homes]


def _mtime(path: Path):
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _poll_runtime_files() -> None:
    """Broadcast ``display.status`` when a home's screen started/stopped outside this process."""
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
    for home in _watched_lease_homes():
        key = hermes_home_key(home)
        sd = home / "bot-desktop"
        mark = (_mtime(sd / "env"), _mtime(sd / "launcher.pid"))
        first = key not in _runtime_marks
        if _runtime_marks.get(key) == mark:
            continue
        _runtime_marks[key] = mark
        if first:  # seeding: display.status carries the current state
            continue
        token = set_hermes_home_override(home)
        try:
            payload = _display_snapshot()
        finally:
            reset_hermes_home_override(token)
        _broadcast_global_event("display.status", payload)


def _poll_lease_files() -> None:
    """One pass: read a home's lease only when its file mtime moved; broadcast when the epoch did."""
    from hermes_constants import hermes_home_key
    from tools.bot_desktop import lease as _bd_lease
    for home in _watched_lease_homes():
        key = hermes_home_key(home)
        try:
            mtime = (home / "bot-desktop" / "lease.json").stat().st_mtime_ns
        except OSError:
            mtime = None
        if mtime == _lease_mtimes.get(key, 0):
            continue
        _lease_mtimes[key] = mtime
        lease = _bd_lease.get(str(home))
        if key not in _lease_epochs:  # first sighting seeds silently: display.status carries it
            _lease_epochs[key] = lease.epoch
            continue
        if lease.epoch == _lease_epochs[key]:
            continue
        _lease_epochs[key] = lease.epoch
        _broadcast_global_event("display.lease", _lease_event_payload(key, lease))


def _ensure_lease_watcher() -> None:
    """Once per process, from the first display.* call: start the lease-file poll thread and mark
    in-process transitions as seen so they broadcast exactly once (via the in-process listener)."""
    if _lease_watcher_started.is_set():
        return
    _lease_watcher_started.set()
    from tools.bot_desktop import lease as _bd_lease

    def _seen_locally(profile_key: str, lease) -> None:
        # methods_display's listener broadcasts in-process transitions; only then is the file
        # move ours to skip. Before display.status installed it, the poll below carries them.
        if _lease_listener_installed.is_set():
            _lease_epochs[profile_key] = lease.epoch
    _bd_lease.on_change(_seen_locally)

    def _loop() -> None:
        while True:
            try:
                _poll_lease_files()
                _poll_runtime_files()
            except Exception:  # noqa: BLE001 - a torn read must not kill the watcher
                logger.debug("lease watcher poll failed", exc_info=True)
            time.sleep(_LEASE_POLL_S)
    threading.Thread(target=_loop, name="hermes-lease-watcher", daemon=True).start()


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
