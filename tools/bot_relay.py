"""Bot Mode cross-connection relay — connections ARE the peer set.

Every gateway connected to the user's Desktop (local, remote URL, SSH,
Hermes Cloud, docker) is a persistent line. This module is the gateway-side
half of the relay that rides those lines so agents on ANY connected gateway
can find and message agents on ANY other, with `message_agent` as the one
send path (Teknium ruling, Aug 2026 — the peers-vs-connections split was
itself the bug).

How the relay works (three files under ``<root>/bot_relay/``):

- ``roster.json`` — the union roster of agents on OTHER connections, pushed
  by the Desktop over each connection's WebSocket (``bot_relay.roster.sync``).
  ``tools/bot_mode_probe.py`` folds it into the Bot Chat protocol section so
  every bot knows every reachable teammate, and ``message_agent`` resolves
  cross-connection targets against it.
- ``outbox/`` — envelopes queued by ``message_agent`` for targets that live
  on another connection. The Desktop drains them (``bot_relay.outbox.drain``)
  and delivers each to the target connection (``bot_relay.deliver``).
- ``replies/`` — one JSON per envelope, written when the Desktop relays the
  target agent's reply back (``bot_relay.reply``). A background waiter
  spawned at send time watches for it, so the reply wakes the sender through
  the exact same completion-notification path local DMs already use.

The gateway never holds another connection's credentials; the Desktop owns
every socket and does all cross-connection I/O. Everything here is plain
file plumbing on the gateway's own HERMES root — no network. The public
helpers never raise, with one deliberate exception: ``enqueue_envelope``
raises ``EnvelopeRefusedError`` when the target is definitively offline, so
the sender fails fast instead of queueing a DM nobody will drain (#93091).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

RELAY_DIR_NAME = "bot_relay"
ROSTER_FILE = "roster.json"
OUTBOX_DIR = "outbox"
CLAIMED_DIR = "claimed"
REPLIES_DIR = "replies"
LOCKS_DIR = "locks"

# Fallback wait budget for a queued delivery turn when config is unreadable.
# The real knob is ``bot_mode.turn_wait_seconds`` in config.yaml.
TURN_WAIT_SECONDS_FALLBACK = 120

# A reply must arrive before the waiter gives up. Cross-connection turns can
# be slow (remote model, cold gateway) — generous, but bounded.
REPLY_WAIT_SECONDS = 900

# Envelopes and replies older than this are stale artifacts (Desktop was
# closed, connection died) and are swept opportunistically.
STALE_AFTER_SECONDS = 6 * 3600

# Fallback envelope TTL when config is unreachable — mirrors the
# ``bot_mode.envelope_ttl_seconds`` default in hermes_cli/config_defaults.py.
# Envelopes older than the TTL are refused at drain time with a
# 'queued_expired' error reply instead of being delivered late.
DEFAULT_ENVELOPE_TTL_SECONDS = 900

# A roster older than this proves nothing about who is offline: the Desktop
# pushes roster.sync on connection-state changes, so only a recently-written
# roster is treated as an authoritative view for the fail-fast check.
ROSTER_FRESH_SECONDS = 600


class EnvelopeRefusedError(RuntimeError):
    """``enqueue_envelope`` refused to queue — nothing was written to disk.

    ``reason`` is a stable machine code; ``str(exc)`` is the human text.
    'runtime_offline' matches the #93091 item-1 failure-reason enum (plain
    literal here so the branches merge cleanly).
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason

_HANDLE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def relay_root(root: Path | str) -> Path:
    return Path(root) / RELAY_DIR_NAME


def _ensure_dirs(root: Path | str) -> Path:
    base = relay_root(root)
    for sub in (OUTBOX_DIR, CLAIMED_DIR, REPLIES_DIR):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


# ── remote roster ────────────────────────────────────────────────────────────


def _normalize_roster_row(row: Any) -> Optional[dict]:
    """Validated, minimal roster row or None.

    Rows come from the Desktop over RPC — treat as untrusted input. A row
    names an agent on another connection: profile name, taggable handle,
    the connection id/label of the gateway that owns it, and optional
    friendly title/description for the protocol section.
    """
    if not isinstance(row, dict):
        return None
    profile = str(row.get("profile") or "").strip()
    handle = str(row.get("handle") or "").strip().lstrip("@")
    connection_id = str(row.get("connection_id") or "").strip()
    if not profile or not connection_id:
        return None
    if not handle:
        handle = "hermes" if profile == "default" else profile
    if (
        not _HANDLE_RE.match(handle)
        or not _HANDLE_RE.match(profile)
        or not _HANDLE_RE.match(connection_id)
    ):
        return None
    out = {
        "profile": profile,
        "handle": handle,
        "connection_id": connection_id,
        "connection_label": str(row.get("connection_label") or "").strip()[:80],
        "title": str(row.get("title") or "").strip()[:120],
        "description": " ".join(str(row.get("description") or "").split())[:160],
    }
    # Optional explicit liveness flag (additive — the Desktop may push it).
    # Preserved only when it is a real bool so absent stays distinguishable
    # from false: absent == liveness unknown == fail-open on enqueue.
    if isinstance(row.get("online"), bool):
        out["online"] = row["online"]
    return out


def write_remote_roster(root: Path | str, rows: Any) -> int:
    """Atomically persist the Desktop-pushed remote roster. Returns count."""
    base = _ensure_dirs(root)
    cleaned: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows if isinstance(rows, list) else []:
        norm = _normalize_roster_row(row)
        if not norm:
            continue
        key = (norm["connection_id"], norm["profile"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(norm)
    cleaned.sort(key=lambda r: (r["connection_id"], r["profile"]))
    payload = {"updated_at": int(time.time()), "agents": cleaned}
    target = base / ROSTER_FILE
    fd, tmp = tempfile.mkstemp(dir=str(base), prefix=".roster-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(cleaned)


def read_remote_roster(root: Path | str) -> list[dict]:
    """The current remote roster (possibly empty). Never raises."""
    try:
        raw = (relay_root(root) / ROSTER_FILE).read_text(encoding="utf-8")
        data = json.loads(raw)
        agents = data.get("agents") if isinstance(data, dict) else None
        if not isinstance(agents, list):
            return []
        return [r for r in (_normalize_roster_row(a) for a in agents) if r]
    except FileNotFoundError:
        return []
    except Exception:
        logger.debug("bot_relay roster read failed", exc_info=True)
        return []


def resolve_remote_target(raw_target: str, roster: list[dict]) -> Any:
    """Resolve ``raw_target`` against the remote roster.

    Accepted forms:
    - bare handle/profile (``moxie``) — must be unique across connections;
    - ``<handle>@<connection-id>`` / ``<profile>@<connection-id>`` — exact.

    Returns the matched row, the string ``"ambiguous"`` when a bare form
    matches agents on several connections, or None for no match.
    """
    want = str(raw_target or "").strip().lstrip("@")
    if not want:
        return None
    conn: Optional[str] = None
    if "@" in want:
        want, _, conn = want.partition("@")
        want = want.strip()
        conn = conn.strip()
        if not want or not conn:
            return None
    matches = []
    for row in roster:
        if want.lower() not in (row["handle"].lower(), row["profile"].lower()):
            continue
        if conn and row["connection_id"].lower() != conn.lower():
            continue
        matches.append(row)
    if not matches:
        return None
    if len(matches) > 1:
        return "ambiguous"
    return matches[0]


def remote_target_forms(roster: list[dict]) -> list[str]:
    """Human/agent-facing target strings, ambiguity-aware."""
    by_handle: dict[str, int] = {}
    for row in roster:
        by_handle[row["handle"].lower()] = by_handle.get(row["handle"].lower(), 0) + 1
    forms = []
    for row in roster:
        if by_handle[row["handle"].lower()] > 1:
            forms.append(f"{row['handle']}@{row['connection_id']}")
        else:
            forms.append(row["handle"])
    return forms


# ── outbox / replies ─────────────────────────────────────────────────────────


def _envelope_ttl_seconds() -> int:
    """Configured drain TTL (``bot_mode.envelope_ttl_seconds``), lazily read.

    tools/ must not pull heavy CLI config at import time, so the read happens
    per-drain and falls back to ``DEFAULT_ENVELOPE_TTL_SECONDS`` when config
    is unavailable (tests, stripped installs). ``0`` (or negative) disables
    drain-time expiry.
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        val = (cfg.get("bot_mode") or {}).get("envelope_ttl_seconds")
        if val is not None:
            return int(val)
    except Exception:
        logger.debug("bot_relay TTL config read failed", exc_info=True)
    return DEFAULT_ENVELOPE_TTL_SECONDS


def _target_liveness(root: Path | str, target: dict) -> Optional[bool]:
    """Tri-state liveness for ``target``: True / False / None (unknown).

    Roster rows carry no heartbeat today, so 'definitively offline' is keyed
    off the two signals roster.json actually gives us:

    - an explicit ``online: false`` on the target's row (additive field,
      honored when the Desktop starts pushing it);
    - the target's (connection_id, profile) being ABSENT from a *fresh*
      roster — the Desktop re-pushes the whole roster on connection-state
      changes, so a recently-synced roster that dropped the target means its
      connection is gone.

    A missing, unreadable, or stale (older than ``ROSTER_FRESH_SECONDS``)
    roster proves nothing → None, and callers fail open. Never raises.
    """
    try:
        roster_path = relay_root(root) / ROSTER_FILE
        try:
            age = time.time() - roster_path.stat().st_mtime
        except OSError:
            return None  # no roster ever synced — unknown
        if age > ROSTER_FRESH_SECONDS:
            return None  # stale view — unknown
        roster = read_remote_roster(root)
        if not roster:
            return None  # empty/corrupt roster — treat as unknown, fail open
        key = (str(target.get("connection_id") or ""), str(target.get("profile") or ""))
        for row in roster:
            if (row["connection_id"], row["profile"]) == key:
                online = row.get("online")
                if online is False:
                    return False
                return True if online is True else None
        return False  # fresh roster no longer lists the target — offline
    except Exception:
        logger.debug("bot_relay liveness check failed", exc_info=True)
        return None


def enqueue_envelope(
    root: Path | str,
    *,
    target: dict,
    message: str,
    sender_profile: str,
    sender_handle: str,
) -> dict:
    """Queue a cross-connection DM for the Desktop relay. Returns envelope.

    Raises ``EnvelopeRefusedError`` (reason ``'runtime_offline'``) instead of
    writing the outbox file when the target is definitively offline per
    ``_target_liveness``. Unknown liveness enqueues as before (fail-open).
    """
    if _target_liveness(root, target) is False:
        label = (
            f"@{target.get('handle') or target.get('profile') or '?'} on "
            f"{target.get('connection_label') or target.get('connection_id') or '?'}"
        )
        # 'runtime_offline' matches the #93091 item-1 reason enum.
        raise EnvelopeRefusedError(
            "runtime_offline",
            f"{label} is offline right now — the message was NOT queued. "
            "Try again once that machine reconnects to the Desktop.",
        )
    base = _ensure_dirs(root)
    envelope = {
        "id": uuid.uuid4().hex,
        "created_at": int(time.time()),
        "from_profile": sender_profile,
        "from_handle": sender_handle,
        "target_connection": target["connection_id"],
        "target_profile": target["profile"],
        "target_handle": target["handle"],
        "message": message,
    }
    path = base / OUTBOX_DIR / f"{envelope['id']}.json"
    fd, tmp = tempfile.mkstemp(dir=str(base / OUTBOX_DIR), prefix=".env-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False)
    os.replace(tmp, path)
    return envelope


def claim_pending_envelopes(root: Path | str) -> list[dict]:
    """Drain the outbox (rename → claimed/, so a second drain can't double-
    deliver). Sweeps stale claimed/reply artifacts opportunistically.

    Envelopes older than ``bot_mode.envelope_ttl_seconds`` are NOT delivered:
    each gets an error reply (reason ``'queued_expired'``) so the sender's
    waiter resolves, and its outbox file is removed (#93091 item 2).
    """
    base = _ensure_dirs(root)
    _sweep_stale(base)
    ttl = _envelope_ttl_seconds()
    now = time.time()
    out: list[dict] = []
    outbox = base / OUTBOX_DIR
    for path in sorted(outbox.glob("*.json")):
        if ttl > 0:
            expired = False
            try:
                env = json.loads(path.read_text(encoding="utf-8"))
                created = float(env.get("created_at") or path.stat().st_mtime)
                if now - created > ttl:
                    expired = True
                    handle = str(env.get("target_handle") or "?")
                    conn = str(env.get("target_connection") or "?")
                    # 'queued_expired' matches the #93091 item-1 reason enum.
                    write_reply(
                        root,
                        str(env.get("id") or ""),
                        error=(
                            f"queued message to @{handle} on {conn} expired after "
                            f"{ttl}s waiting for the Desktop to drain it — it was "
                            "NOT delivered. Resend once the Desktop reconnects."
                        ),
                        reason="queued_expired",
                    )
            except (OSError, ValueError):
                # Unreadable envelope or invalid id: if it already counted as
                # expired, still remove it below; otherwise let the normal
                # claim attempt below deal with it.
                pass
            if expired:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
        claimed = base / CLAIMED_DIR / path.name
        try:
            os.replace(path, claimed)  # atomic claim
            out.append(json.loads(claimed.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def write_reply(
    root: Path | str, envelope_id: str, *, reply: str = "", error: str = "", reason: str = ""
) -> Path:
    """Persist the relayed reply (or delivery error) for the waiter.

    ``reason`` is an optional typed failure code (see
    ``tools.bot_failure_reasons``, e.g. 'queued_expired'); when omitted and
    ``error`` is non-empty it is classified from the error text. The waiter
    only surfaces the human ``error``.
    """
    base = _ensure_dirs(root)
    safe = str(envelope_id or "").strip()
    if not re.match(r"^[0-9a-f]{32}$", safe):
        raise ValueError(f"invalid envelope id: {envelope_id!r}")
    err = str(error or "")
    code = str(reason or "")
    if not code and err:
        from tools.bot_failure_reasons import classify_agent_error

        code = classify_agent_error(err)
    path = base / REPLIES_DIR / f"{safe}.json"
    payload = {
        "id": safe,
        "at": int(time.time()),
        "reply": str(reply or ""),
        "error": err,
        "reason": code,
    }
    fd, tmp = tempfile.mkstemp(dir=str(base / REPLIES_DIR), prefix=".rep-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def _sweep_stale(base: Path, *, now: float | None = None) -> int:
    cutoff = (time.time() if now is None else now) - STALE_AFTER_SECONDS
    removed = 0
    for sub in (CLAIMED_DIR, REPLIES_DIR, OUTBOX_DIR):
        try:
            for path in (base / sub).glob("*.json"):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
        except OSError:
            continue
    return removed


def cleanup_bot_relay_artifacts(max_age_hours: float | None = None) -> int:
    """Sweep stale relay artifacts (envelopes/replies hold DM plaintext).

    ``_sweep_stale`` otherwise runs only when the Desktop drains the outbox
    (``claim_pending_envelopes``) — if the Desktop never reconnects, queued
    plaintext envelopes would sit on disk forever. Same contract as the
    ``cleanup_*_cache`` helpers so the gateway housekeeping loop can call it
    hourly. ``max_age_hours`` is accepted for signature compatibility but the
    relay's own ``STALE_AFTER_SECONDS`` governs staleness.
    """
    del max_age_hours  # relay staleness is governed by STALE_AFTER_SECONDS
    try:
        home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
        root = home.parent.parent if home.parent.name == "profiles" else home
        base = relay_root(root)
        if not base.is_dir():
            return 0
        return _sweep_stale(base)
    except Exception:
        logger.debug("bot_relay artifact sweep failed", exc_info=True)
        return 0


# ── waiter (runs on the sender gateway via terminal background process) ─────


def waiter_command(root: Path | str, envelope: dict) -> str:
    """Shell command that blocks until the reply file appears, then prints it.

    Spawned with ``terminal_tool(background=True, notify_on_complete=True)``
    so its stdout — the teammate's reply — arrives as the same completion
    notification local DMs use. Stdlib-only; runs under the sender gateway's
    interpreter.
    """
    reply_path = str(relay_root(root) / REPLIES_DIR / f"{envelope['id']}.json")
    label = (
        f"@{envelope.get('target_handle', '')} "
        f"on {envelope.get('target_connection', '')}"
    )
    # Encode label with !r so roster fields cannot break out of the generated
    # python -c source (quotes, parens, or extra statements in connection_id).
    code = (
        "import json,os,sys,time\n"
        f"p = {reply_path!r}\n"
        f"label = {label!r}\n"
        f"deadline = time.time() + {REPLY_WAIT_SECONDS}\n"
        "while time.time() < deadline:\n"
        "    if os.path.exists(p):\n"
        "        d = json.load(open(p, encoding='utf-8'))\n"
        "        if d.get('error'):\n"
        "            print('Delivery to ' + label + ' failed: ' + d['error'])\n"
        "            sys.exit(1)\n"
        "        print('Reply from ' + label + ':')\n"
        "        print(d.get('reply') or '(empty reply)')\n"
        "        sys.exit(0)\n"
        "    time.sleep(2)\n"
        f"print('No reply from ' + label + ' within {REPLY_WAIT_SECONDS}s. The message may "
        "still be delivered when the Desktop reconnects; do not resend blindly.')\n"
        "sys.exit(1)\n"
    )
    return f"{shlex.quote(sys.executable or 'python3')} -c {shlex.quote(code)}"


# ── delivery command (used by the deliver RPC on the TARGET gateway) ────────


def local_delivery_command(profile: str, query_file: str) -> list[str]:
    """argv that delivers a DM into ``profile``'s Bot Chat on THIS gateway."""
    return [
        "hermes",
        "-p",
        profile,
        "chat",
        "--in",
        "~",
        "-c",
        "Bot Chat",
        "--create-if-missing",
        "-Q",
        "--query-file",
        query_file,
    ]


# ── per-profile turn lock (#93091) ───────────────────────────────────────────
#
# Two deliveries into the SAME target profile must never run their Bot Chat
# turns concurrently: deliveries spawn separate ``hermes`` subprocesses, so
# an in-memory mutex is useless — the lock is a per-profile lockfile under
# ``<root>/bot_relay/locks/`` held with ``fcntl.flock`` for exactly the turn
# execution window. flock is released by the kernel when the holder's fd
# closes (including process death), so a crashed turn can never wedge the
# profile. A queued delivery waits up to ``bot_mode.turn_wait_seconds`` and
# then fails with a structured 'target_busy' refusal instead of blocking
# forever.


class TurnBusyError(RuntimeError):
    """A delivery turn is already running for the target profile.

    ``reason`` is 'target_busy' — extends the #93091 item-1 structured
    refusal enum. ``waited_seconds`` is roughly how long the caller queued
    behind the current turn before giving up.
    """

    reason = "target_busy"

    def __init__(self, profile: str, waited_seconds: float):
        self.profile = profile
        self.waited_seconds = waited_seconds
        super().__init__(
            f"target_busy: another delivery turn is already running for "
            f"profile '{profile}' — queued behind it for ~{int(round(waited_seconds))}s "
            "without it finishing. The message was NOT delivered; retry shortly."
        )


def turn_wait_seconds() -> float:
    """Wait budget for a queued delivery turn (config, lazily read)."""
    try:
        from hermes_cli.config import cfg_get, load_config

        val = cfg_get(load_config(), "bot_mode", "turn_wait_seconds", default=None)
        if val is not None:
            return max(0.0, float(val))
    except Exception:
        logger.debug("bot_mode.turn_wait_seconds read failed", exc_info=True)
    return float(TURN_WAIT_SECONDS_FALLBACK)


def turn_lock_path(root: Path | str, profile: str) -> Path:
    """Per-profile lockfile path (short — safe on macOS temp roots)."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(profile or ""))[:64] or "_"
    return relay_root(root) / LOCKS_DIR / f"{safe}.lock"


@contextlib.contextmanager
def acquire_turn_lock(
    root: Path | str, profile: str, timeout_seconds: float | None = None
) -> Iterator[Path]:
    """Hold ``profile``'s cross-process turn lock for the ``with`` body.

    Non-blocking flock probe + short-sleep retry loop up to the budget
    (``bot_mode.turn_wait_seconds`` unless ``timeout_seconds`` is given).
    No ordering guarantee among waiters — whichever probe lands first after
    release wins — but every waiter is bounded by the budget, so no
    deadlock. Raises :class:`TurnBusyError` when the budget is exhausted.
    On platforms without ``fcntl`` (Windows) the lock degrades to a no-op —
    those installs never had this race path in production.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover — Windows
        logger.debug("bot turn lock disabled: fcntl unavailable on this platform")
        yield turn_lock_path(root, profile)
        return

    budget = turn_wait_seconds() if timeout_seconds is None else max(0.0, float(timeout_seconds))
    path = turn_lock_path(root, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        start = time.monotonic()
        deadline = start + budget
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                now = time.monotonic()
                if now >= deadline:
                    raise TurnBusyError(profile, now - start)
                time.sleep(min(0.1, max(0.005, deadline - now)))
        try:
            yield path
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:  # pragma: no cover — kernel releases on close anyway
                pass
    finally:
        os.close(fd)
