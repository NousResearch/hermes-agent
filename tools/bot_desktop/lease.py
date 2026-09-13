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
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home, hermes_home_key

try:
    import fcntl
except ImportError:  # Windows/macOS without fcntl: computer_use imports this module on every call, and no
    fcntl = None  # multi-process Bot Desktop exists there, so the cross-process lock degrades to a no-op.

logger = logging.getLogger(__name__)

AGENT = "agent"
HUMAN = "human"
_POLL_SECONDS = 0.25


class HumanHasControl(RuntimeError):
    """Raised by screen-driving tools while a human holds the lease."""


class ViewerLeaseHeld(RuntimeError):
    """Raised when another authenticated viewer session already holds this profile's screen."""


@dataclass
class Lease:
    holder: str = AGENT
    viewer_id: Optional[str] = None
    since: float = field(default_factory=time.time)
    reason: str = ""
    pending_handoff: Optional[str] = None  # agent's reason for asking, until the human takes over
    resume_hash: str = ""  # hash of the holder's client-only recovery capability; never broadcast
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


def _write(path: Path, lease: Lease) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lease.as_dict()), encoding="utf-8")
    os.replace(tmp, path)


class _locked:
    """Cross-process critical section over the lease file (fcntl on a sibling lock file)."""

    def __init__(self, path: Path):
        self._lockfile = path.with_suffix(".lock")
        self._fh = None

    def __enter__(self):
        if fcntl is None:
            return self
        self._lockfile.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._lockfile, "a+", encoding="utf-8")  # noqa: SIM115 — closed in __exit__
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self._fh is None:
            return
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()


def get(profile_key: Optional[str] = None) -> Lease:
    return _read(_path(profile_key))


@contextmanager
def locked_snapshot(profile_key: Optional[str] = None) -> Iterator[Lease]:
    """Yield one read-only lease snapshot while excluding cross-process transitions.

    Admission code may bind other process-local state to this snapshot before releasing the lock;
    callers must not mutate it because this context deliberately performs no write.
    """
    path = _path(profile_key)
    with _locked(path):
        yield _read(path)


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
    key, path = (
        hermes_home_key(profile_key) if profile_key else hermes_home_key(),
        _path(profile_key),
    )
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


def acquire(
    viewer_id: str,
    *,
    profile_key: Optional[str] = None,
    reason: str = "",
    resume_hash: str = "",
) -> Lease:
    """Let ``viewer_id`` take control unless another authenticated viewer session holds it."""

    def _m(lease: Lease) -> bool:
        if lease.holder == HUMAN and lease.viewer_id != viewer_id:
            raise ViewerLeaseHeld("another viewer already holds private control")
        # The agent's ask ("please log in to X") stays as the takeover reason: the human needs it
        # on screen WHILE they act, not only before they clicked Take over.
        lease.holder, lease.viewer_id, lease.since = HUMAN, viewer_id, time.time()
        lease.reason = reason or lease.pending_handoff or ""
        lease.pending_handoff = None
        lease.resume_hash = resume_hash
        return True

    return _transition(profile_key, _m)


def release(viewer_id: Optional[str] = None, *, profile_key: Optional[str] = None) -> Lease:
    """Return control to the agent. With ``viewer_id`` only that holder may release (a stale viewer
    closing its window must not yank control from the one who took over after it)."""

    def _m(lease: Lease) -> bool:
        if viewer_id is not None and lease.holder == HUMAN and lease.viewer_id != viewer_id:
            # Ignored, not an error: the returned lease still shows the real holder. Logged so a
            # caller that never inspects the return value leaves a trace.
            logger.info("bot-desktop lease: release by %r ignored, another viewer holds", viewer_id)
            return False
        lease.holder, lease.viewer_id, lease.since, lease.reason = (
            AGENT,
            None,
            time.time(),
            "",
        )
        lease.resume_hash = ""
        lease.pending_handoff = None  # "hand back" answers an open request even if nobody formally took over
        return True

    return _transition(profile_key, _m)


def request_handoff(reason: str, *, profile_key: Optional[str] = None) -> Lease:
    """Agent asks a human to take over (login, 2FA, CAPTCHA, payment). Recorded so the UI can show
    why and the bridge can page the user; control itself still flips only on ``acquire``."""

    def _m(lease: Lease) -> bool:
        lease.pending_handoff = reason
        return True

    return _transition(profile_key, _m)


def wait_for_release(*, timeout: float, profile_key: Optional[str] = None) -> bool:
    """Block until the agent holds the lease (and no handoff is pending) or ``timeout`` elapses.
    True when control is back with the agent. Polls the file so a release made by another process
    is seen; the local Condition just shortens the wait for same-process transitions."""
    return _wait_until(
        lambda lease: lease.holder == AGENT and lease.pending_handoff is None,
        timeout,
        profile_key,
    )


def wait_for_takeover_or_release(*, timeout: float, profile_key: Optional[str] = None) -> bool:
    """False when, after ``timeout``, the agent still holds with a handoff pending: nobody answered."""
    return _wait_until(
        lambda lease: lease.holder == HUMAN or lease.pending_handoff is None,
        timeout,
        profile_key,
    )


def _wait_until(done: Callable[[Lease], bool], timeout: float, profile_key: Optional[str]) -> bool:
    path = _path(profile_key)
    deadline = time.monotonic() + timeout
    while True:
        if done(_read(path)):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        with _lock:
            _lock.wait(min(remaining, _POLL_SECONDS))


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
            "captures are refused until they hand control back; call computer_use action='wait_for_human' "
            "to block until then."
        )
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
