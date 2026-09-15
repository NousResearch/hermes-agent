"""Who may drive a profile's Bot Desktop screen: the agent (default) or exactly one human viewer.

The lease is the single truth shared by the RFB bridge (drops human input from non-holders), the
``computer_use`` tool (refuses to act while a human holds control — the person may be typing a
credential, so even screenshots are refused; fail closed rather than trusting the agent to pause
itself) and the Desktop UI (Watch / Take over / Hand back).

Authority lives ON DISK, ``<HERMES_HOME>/bot-desktop/lease.json`` under an fcntl lock, because the
processes that must agree do not share memory: ``hermes serve`` (viewer bridge), the messaging
gateway, a CLI turn and isolated workers all drive the same display. Every read goes to the file;
the in-process Condition only wakes local waiters early. ``epoch`` increments on every transition so
an action admitted under one lease can tell that control changed underneath it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from hermes_constants import get_hermes_home, hermes_home_key, secure_parent_dir

try:
    import fcntl
except ImportError:  # Windows/macOS without fcntl: computer_use imports this module on every call, and no
    fcntl = None     # multi-process Bot Desktop exists there, so the cross-process lock degrades to a no-op.

logger = logging.getLogger(__name__)

AGENT = "agent"
HUMAN = "human"


class HumanHasControl(RuntimeError):
    """Raised by screen-driving tools while a human holds the lease."""


@dataclass
class Lease:
    holder: str = AGENT
    viewer_id: Optional[str] = None
    since: float = field(default_factory=time.time)
    reason: str = ""
    epoch: int = 0

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


_lock = threading.Condition()
_listeners: List[Callable[[str, Lease], None]] = []


def _path(profile_key: Optional[str]) -> Path:
    """``profile_key`` is the HERMES_HOME path of the profile whose lease is meant (the RFB bridge
    serves several profiles from one process); ``None`` means the current profile."""
    home = Path(profile_key) if profile_key else get_hermes_home()
    return home / "bot-desktop" / "lease.json"


def _read(path: Path) -> Lease:
    """No file = fresh profile, agent holds. A file that exists but cannot be parsed is a torn write
    or tampering: fail CLOSED (human holds) — an unreadable lease must never let the agent act on a
    screen a human may be using; the next successful write repairs it."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Lease()
    except OSError:
        return Lease(holder=HUMAN, viewer_id="unreadable-lease", reason="lease file unreadable")
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if not isinstance(data, dict) or data.get("holder") not in (AGENT, HUMAN):
        return Lease(holder=HUMAN, viewer_id="unreadable-lease", reason="lease file corrupt")
    try:
        return Lease(**{k: v for k, v in data.items() if k in Lease.__dataclass_fields__})
    except TypeError:
        return Lease(holder=HUMAN, viewer_id="unreadable-lease", reason="lease file corrupt")


def _private_dir(path: Path) -> None:
    """``bot-desktop/`` owner-only even when the lease is the first thing written there (a takeover can be
    recorded before start() ever ran, and the umask would otherwise leave it 0755)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(path)


def _open_private(path: str | bytes | os.PathLike, flags: int) -> int:
    """``open(..., opener=_open_private)``: the file is created 0600 regardless of the umask."""
    return os.open(path, flags, 0o600)


def _write(path: Path, lease: Lease) -> None:
    _private_dir(path)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", opener=_open_private) as fh:
        fh.write(json.dumps(lease.as_dict()))
    os.replace(tmp, path)


class _locked:
    """Cross-process critical section over the lease file (fcntl on a sibling lock file)."""

    def __init__(self, path: Path):
        self._lockfile = path.with_suffix(".lock")
        self._fh = None

    def __enter__(self):
        if fcntl is None:
            return self
        _private_dir(self._lockfile)
        self._fh = open(self._lockfile, "a+", encoding="utf-8", opener=_open_private)  # noqa: SIM115 — closed in __exit__
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self._fh is None:
            return
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()


def get(profile_key: Optional[str] = None) -> Lease:
    return _read(_path(profile_key))


def on_change(listener: Callable[[str, Lease], None]) -> Callable[[], None]:
    """Subscribe to lease transitions made IN THIS PROCESS (the gateway broadcasts them to Desktop
    clients). Transitions made by another process are observed by reading, not by callback."""
    with _lock:
        _listeners.append(listener)

    def _off() -> None:
        with _lock:
            if listener in _listeners:
                _listeners.remove(listener)
    return _off


def _notify(key: str, lease: Lease) -> None:
    for cb in list(_listeners):
        try:
            cb(key, lease)
        except Exception:  # a broken subscriber must not wedge the handoff
            pass


def _transition(profile_key: Optional[str], mutate: Callable[[Lease], bool]) -> Lease:
    key, path = hermes_home_key(profile_key) if profile_key else hermes_home_key(), _path(profile_key)
    with _locked(path):
        lease = _read(path)
        if not mutate(lease):
            return lease
        lease.epoch += 1
        _write(path, lease)
    with _lock:
        _lock.notify_all()
    _notify(key, lease)
    return lease


def acquire(viewer_id: str, *, profile_key: Optional[str] = None, reason: str = "") -> Lease:
    """Human ``viewer_id`` takes control. Last writer wins: a second viewer evicts the first, and the
    RFB bridge closes the evicted socket so its UI drops to view-only."""
    def _m(lease: Lease) -> bool:
        if lease.holder == HUMAN and lease.viewer_id == viewer_id:
            return False  # already theirs: no epoch bump, `since` and the reason on screen stay put
        # The agent's ask ("please log in to X") stays as the takeover reason: the human needs it
        # on screen WHILE they act, not only before they clicked Take over.
        lease.holder, lease.viewer_id, lease.since = HUMAN, viewer_id, time.time()
        lease.reason = reason or ""
        return True
    return _transition(profile_key, _m)


def release(viewer_id: Optional[str] = None, *, profile_key: Optional[str] = None,
            unless_human: bool = False) -> Lease:
    """Return control to the agent. With ``viewer_id`` only that holder may release (a stale viewer
    closing its window must not yank control from the one who took over after it). ``unless_human``
    makes a bare release a no-op while any human holds: the decision is taken under the file lock,
    so a takeover racing a check-then-release cannot be silently revoked. Callers read the returned
    lease's holder to learn whether anything happened."""
    def _m(lease: Lease) -> bool:
        if unless_human and lease.holder == HUMAN:
            logger.info("bot-desktop lease: bare release ignored, a human holds")
            return False
        if viewer_id is not None and lease.holder == HUMAN and lease.viewer_id != viewer_id:
            # Ignored, not an error: the returned lease still shows the real holder. Logged so a
            # caller that never inspects the return value leaves a trace.
            logger.info("bot-desktop lease: release by %r ignored, another viewer holds", viewer_id)
            return False
        if lease.holder == AGENT:
            # Already the agent's. Bumping the epoch here would make a legitimately admitted in-flight
            # agent action (a double-clicked Hand back, a stray CLI stop) look overtaken and get voided.
            return False
        lease.holder, lease.viewer_id, lease.since, lease.reason = AGENT, None, time.time(), ""
        return True
    return _transition(profile_key, _m)


def human_holds(profile_key: Optional[str] = None) -> bool:
    return get(profile_key).holder == HUMAN


def viewer_may_send_input(viewer_id: str, *, profile_key: Optional[str] = None) -> bool:
    lease = get(profile_key)
    return lease.holder == HUMAN and lease.viewer_id == viewer_id


def assert_agent_may_act(profile_key: Optional[str] = None) -> Lease:
    """The lease as of now, or ``HumanHasControl``. Callers keep the returned ``epoch`` and compare it
    with ``get().epoch`` after an admitted action: a change means a human took over mid-flight."""
    lease = get(profile_key)
    if lease.holder == HUMAN:
        raise HumanHasControl(
            "A human has taken over this desktop (they may be entering a credential). Screen actions and "
            "captures are refused until they hand control back. Tell the user what you need in your reply.")
    return lease


def _reset_for_tests() -> None:
    with _lock:
        _listeners.clear()
    p = get_hermes_home() / "bot-desktop" / "lease.json"
    for f in (p, p.with_suffix(".lock"), p.with_suffix(".json.tmp")):
        try:
            f.unlink()
        except OSError:
            pass
