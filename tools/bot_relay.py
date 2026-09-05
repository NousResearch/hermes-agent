"""Bot Mode cross-connection relay — connections ARE the peer set.

Gateway-side half of the relay letting agents on ANY Desktop-connected gateway
message agents on ANY other. Plain file plumbing under ``<root>/bot_relay/`` —
no network; the Desktop owns every socket: ``roster.json`` (union roster of
agents on OTHER connections, pushed via ``bot_relay.roster.sync``), ``outbox/``
(envelopes queued by ``message_agent``, drained via ``bot_relay.outbox.drain``),
``replies/`` (one JSON per envelope via ``bot_relay.reply``; a waiter spawned at
send time watches it so the reply wakes the sender like a local DM).
Public helpers never raise, except ``enqueue_envelope`` → ``EnvelopeRefusedError``
when the target is definitively offline (fail fast instead of queueing a DM nobody will drain).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

from tools.bot_mode_probe import _default_home, _hermes_root

logger = logging.getLogger(__name__)

RELAY_DIR_NAME = "bot_relay"
ROSTER_FILE = "roster.json"
OUTBOX_DIR = "outbox"
CLAIMED_DIR = "claimed"
REPLIES_DIR = "replies"
LOCKS_DIR = "locks"

# Config fallbacks (real knobs: ``bot_mode.turn_wait_seconds`` / ``bot_mode.envelope_ttl_seconds``).
TURN_WAIT_SECONDS_FALLBACK = 120
DEFAULT_ENVELOPE_TTL_SECONDS = 900  # older envelopes are refused at drain with 'queued_expired'
# Waiter give-up budget: cross-connection turns can be slow — generous, but bounded.
REPLY_WAIT_SECONDS = 900
# Envelopes/replies older than this are stale artifacts (Desktop closed) and are swept.
STALE_AFTER_SECONDS = 6 * 3600
# Only a recent roster is authoritative for the fail-fast offline check: the
# Desktop re-pushes roster.sync on connection-state changes.
ROSTER_FRESH_SECONDS = 600


class EnvelopeRefusedError(RuntimeError):
    """``enqueue_envelope`` refused to queue (nothing written); ``reason`` is a stable machine code.

    ``reason`` is a stable machine code; ``str(exc)`` is the human text. 'runtime_offline' matches the
    #93091 item-1 failure-reason enum (plain literal here so the branches merge cleanly).
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# Profile names, handles and connection ids share one shape (also the local
# ``message_agent`` target grammar in ``tools/bot_mode_dm.py``).
_HANDLE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# One turn in a profile's canonical Bot Chat: ``hermes -p <profile> *BOT_CHAT_TURN_ARGS``.
# ``-c "Bot Chat"`` must match ``bot_mode_probe.BOT_CHAT_TITLE``.
BOT_CHAT_TURN_ARGS = ("chat", "--in", "~", "-c", "Bot Chat", "--create-if-missing", "-Q")


def relay_root(root: Path | str) -> Path:
    return Path(root) / RELAY_DIR_NAME


def _ensure_dirs(root: Path | str) -> Path:
    base = relay_root(root)
    for sub in (OUTBOX_DIR, CLAIMED_DIR, REPLIES_DIR):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def _atomic_write_json(target: Path, payload: Any, *, prefix: str, sort_keys: bool = False) -> None:
    """tempfile + os.replace so readers never see a partial file; tempfile removed on failure."""
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=sort_keys)
        os.replace(tmp, target)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _bot_mode_cfg(key: str, *, loader: str) -> Any:
    """``bot_mode.<key>`` from config, read lazily (tools/ must not import CLI
    config at import time); None when absent or the config is unreadable."""
    try:
        import hermes_cli.config as cfgmod

        cfg = getattr(cfgmod, loader)() or {}
        return (cfg.get("bot_mode") or {}).get(key)
    except Exception:
        logger.debug("bot_mode.%s config read failed", key, exc_info=True)
        return None


def _normalize_roster_row(row: Any) -> Optional[dict]:
    """Validated, minimal roster row or None. Rows come from the Desktop over
    RPC — treat as untrusted input."""
    if not isinstance(row, dict):
        return None
    profile = str(row.get("profile") or "").strip()
    handle = str(row.get("handle") or "").strip().lstrip("@") or ("hermes" if profile == "default" else profile)
    connection_id = str(row.get("connection_id") or "").strip()
    if not profile or not connection_id or not all(_HANDLE_RE.match(v) for v in (handle, profile, connection_id)):
        return None
    out = {
        "profile": profile, "handle": handle, "connection_id": connection_id,
        "connection_label": str(row.get("connection_label") or "").strip()[:80],
        "title": str(row.get("title") or "").strip()[:120],
        "description": " ".join(str(row.get("description") or "").split())[:160],
    }
    # Liveness kept only when a real bool: absent == unknown == fail-open on enqueue.
    if isinstance(row.get("online"), bool):
        out["online"] = row["online"]
    return out


def write_remote_roster(root: Path | str, rows: Any) -> int:
    """Atomically persist the Desktop-pushed remote roster. Returns count."""
    base = _ensure_dirs(root)
    by_key: dict[tuple[str, str], dict] = {}
    for norm in filter(None, map(_normalize_roster_row, rows if isinstance(rows, list) else [])):
        by_key.setdefault((norm["connection_id"], norm["profile"]), norm)
    cleaned = [by_key[k] for k in sorted(by_key)]
    _atomic_write_json(base / ROSTER_FILE, {"updated_at": int(time.time()), "agents": cleaned},
                       prefix=".roster-", sort_keys=True)
    return len(cleaned)


def read_remote_roster(root: Path | str) -> list[dict]:
    """The current remote roster (possibly empty). Never raises."""
    try:
        data = json.loads((relay_root(root) / ROSTER_FILE).read_text(encoding="utf-8"))
        agents = data.get("agents") if isinstance(data, dict) else None
        return [r for r in map(_normalize_roster_row, agents) if r] if isinstance(agents, list) else []
    except FileNotFoundError:
        return []
    except Exception:
        logger.debug("bot_relay roster read failed", exc_info=True)
        return []


def resolve_remote_target(raw_target: str, roster: list[dict]) -> Any:
    """Matched row for a bare handle/profile (unique across connections) or
    ``<handle|profile>@<connection-id>``; ``"ambiguous"`` for a bare form on several connections; None otherwise."""
    want, at, conn = (p.strip() for p in str(raw_target or "").strip().lstrip("@").partition("@"))
    if not want or (at and not conn):
        return None
    matches = [row for row in roster if want.lower() in (row["handle"].lower(), row["profile"].lower())
               and (not conn or row["connection_id"].lower() == conn.lower())]
    if not matches:
        return None
    return matches[0] if len(matches) == 1 else "ambiguous"


def remote_target_forms(roster: list[dict]) -> list[str]:
    """Unambiguous target strings for agents on other connections.

    A handle unique within the remote roster can still collide with a profile
    on this gateway (most importantly every gateway's ``@hermes`` default), so
    remote forms always carry their connection id.
    """
    return [f"{row['handle']}@{row['connection_id']}" for row in roster]


def _envelope_ttl_seconds() -> int:
    """Configured drain TTL (``bot_mode.envelope_ttl_seconds``), read per-drain.
    ``0`` (or negative) disables expiry."""
    val = _bot_mode_cfg("envelope_ttl_seconds", loader="load_config_readonly")
    return DEFAULT_ENVELOPE_TTL_SECONDS if val is None else int(val)


def _target_liveness(root: Path | str, target: dict) -> Optional[bool]:
    """Tri-state liveness: True / False / None (unknown → callers fail open). Offline =
    explicit ``online: false`` or ABSENT from a *fresh* roster; a missing, unreadable,
    empty or stale roster proves nothing → None. Never raises."""
    try:
        try:
            age = time.time() - (relay_root(root) / ROSTER_FILE).stat().st_mtime
        except OSError:
            return None
        roster = read_remote_roster(root) if age <= ROSTER_FRESH_SECONDS else []
        if not roster:
            return None
        key = (str(target.get("connection_id") or ""), str(target.get("profile") or ""))
        row = next((r for r in roster if (r["connection_id"], r["profile"]) == key), None)
        if row is None:
            return False  # fresh roster no longer lists the target — offline
        return row["online"] if isinstance(row.get("online"), bool) else None
    except Exception:
        logger.debug("bot_relay liveness check failed", exc_info=True)
        return None


def enqueue_envelope(root: Path | str, *, target: dict, message: str, sender_profile: str, sender_handle: str) -> dict:
    """Queue a cross-connection DM for the Desktop relay; returns the envelope. Raises
    ``EnvelopeRefusedError`` ('runtime_offline') without writing when the target is
    definitively offline; unknown liveness enqueues (fail-open)."""
    if _target_liveness(root, target) is False:
        label = (f"@{target.get('handle') or target.get('profile') or '?'} on "
                 f"{target.get('connection_label') or target.get('connection_id') or '?'}")
        raise EnvelopeRefusedError("runtime_offline", f"{label} is offline right now — the message was NOT queued. "
                                   "Try again once that machine reconnects to the Desktop.")
    base = _ensure_dirs(root)
    envelope = {
        "id": uuid.uuid4().hex, "created_at": int(time.time()),
        "from_profile": sender_profile, "from_handle": sender_handle,
        "target_connection": target["connection_id"], "target_profile": target["profile"],
        "target_handle": target["handle"], "message": message,
    }
    _atomic_write_json(base / OUTBOX_DIR / f"{envelope['id']}.json", envelope, prefix=".env-")
    return envelope


def _expire_if_stale(root: Path | str, path: Path, ttl: float, now: float) -> bool:
    """True when the outbox envelope is older than ``ttl``; writes the 'queued_expired'
    reply so the sender's waiter resolves (best effort). Unreadable envelopes are left for the claim."""
    try:
        env = json.loads(path.read_text(encoding="utf-8"))
        created = float(env.get("created_at") or path.stat().st_mtime)
    except (OSError, ValueError):
        return False
    if now - created <= ttl:
        return False
    with contextlib.suppress(OSError, ValueError):
        write_reply(root, str(env.get("id") or ""), reason="queued_expired", error=(
            f"queued message to @{env.get('target_handle') or '?'} on {env.get('target_connection') or '?'} "
            f"expired after {ttl}s waiting for the Desktop to drain it — it was NOT delivered. "
            "Resend once the Desktop reconnects."))
    return True


def claim_pending_envelopes(root: Path | str) -> list[dict]:
    """Drain the outbox (rename → claimed/ so a second drain can't double-deliver).
    TTL-expired envelopes get a 'queued_expired' reply and are removed instead.

    Envelopes older than ``bot_mode.envelope_ttl_seconds`` are NOT delivered: each gets an error reply
    (reason ``'queued_expired'``) so the sender's waiter resolves, and its outbox file is removed (#93091
    item 2).
    """
    base = _ensure_dirs(root)
    _sweep_stale(base)
    ttl = _envelope_ttl_seconds()
    now = time.time()
    out: list[dict] = []
    for path in sorted((base / OUTBOX_DIR).glob("*.json")):
        if ttl > 0 and _expire_if_stale(root, path, ttl, now):
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        claimed = base / CLAIMED_DIR / path.name
        with contextlib.suppress(OSError, ValueError):
            os.replace(path, claimed)  # atomic claim
            out.append(json.loads(claimed.read_text(encoding="utf-8")))
    return out


def write_reply(root: Path | str, envelope_id: str, *, reply: str = "", error: str = "", reason: str = "") -> Path:
    """Persist the relayed reply (or delivery error) for the waiter. ``reason`` (typed
    code, ``tools.bot_failure_reasons``) is classified from ``error`` when omitted."""
    base = _ensure_dirs(root)
    safe = str(envelope_id or "").strip()
    if not re.match(r"^[0-9a-f]{32}$", safe):
        raise ValueError(f"invalid envelope id: {envelope_id!r}")
    err, code = str(error or ""), str(reason or "")
    if not code and err:
        from tools.bot_failure_reasons import classify_agent_error

        code = classify_agent_error(err)
    path = base / REPLIES_DIR / f"{safe}.json"
    _atomic_write_json(path, {"id": safe, "at": int(time.time()), "reply": str(reply or ""), "error": err, "reason": code},
                       prefix=".rep-")
    return path


def unlink_files_older_than(directory: Path, pattern: str, cutoff: float) -> int:
    """Unlink regular files matching ``pattern`` with mtime before ``cutoff``; returns count. Never raises."""
    removed = 0
    with contextlib.suppress(OSError):
        for path in directory.glob(pattern):
            with contextlib.suppress(OSError):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
    return removed


def _sweep_stale(base: Path, *, now: float | None = None) -> int:
    cutoff = (time.time() if now is None else now) - STALE_AFTER_SECONDS
    return sum(unlink_files_older_than(base / sub, "*.json", cutoff) for sub in (CLAIMED_DIR, REPLIES_DIR, OUTBOX_DIR))


def cleanup_bot_relay_artifacts(max_age_hours: float | None = None) -> int:
    """Hourly sweep of stale relay artifacts (DM plaintext; ``_sweep_stale`` otherwise runs
    only on Desktop drains). ``max_age_hours`` is for ``cleanup_*_cache`` signature parity only."""
    del max_age_hours
    try:
        base = relay_root(_hermes_root(Path(_default_home())))
        return _sweep_stale(base) if base.is_dir() else 0
    except Exception:
        logger.debug("bot_relay artifact sweep failed", exc_info=True)
        return 0


def waiter_command(root: Path | str, envelope: dict) -> str:
    """Shell command that blocks until the reply file appears, then prints it; spawned
    via ``terminal_tool(background=True, notify_on_complete=True)`` so its stdout arrives
    as the same completion notification local DMs use. Stdlib-only."""
    reply_path = str(relay_root(root) / REPLIES_DIR / f"{envelope['id']}.json")
    label = f"@{envelope.get('target_handle', '')} on {envelope.get('target_connection', '')}"
    # !r keeps roster fields from breaking out of the generated python -c source.
    # The r-prefix keeps Windows paths viable: the Windows execution layer folds
    # repr's "\\" back to "\", turning "\U" into an invalid unicode escape; a
    # raw literal parses the folded backslash literally. No-op on POSIX, and \'
    # still cannot terminate a raw literal, so the injection defense holds.
    code = (
        # Encode label with !r so roster fields cannot break out of the generated python -c source (quotes,
        # parens, or extra statements in connection_id). See #93590.
        "import json,os,sys,time\n"
        f"p = r{reply_path!r}\n"
        f"label = r{label!r}\n"
        f"deadline = time.time() + {REPLY_WAIT_SECONDS}\n"
        "while time.time() < deadline:\n"
        "    if os.path.exists(p):\n"
        "        d = json.load(open(p, encoding='utf-8'))\n"
        "        if d.get('error'):\n"
        # Typed reason code rides ahead of the free text so the sender can
        # branch on it without parsing provider prose.
        # See #93091.
        "            code = str(d.get('reason') or '').strip()\n"
        "            tag = ' [reason: ' + code + ']' if code else ''\n"
        "            print('Delivery to ' + label + ' failed' + tag + ': ' + d['error'])\n"
        "            sys.exit(1)\n"
        "        print('Reply from ' + label + ':')\n"
        "        print(d.get('reply') or '(empty reply)')\n"
        "        sys.exit(0)\n"
        # 250ms cadence: stat is cheap and a longer sleep is pure dead air.
        "    time.sleep(0.25)\n"
        f"print('No reply from ' + label + ' within {REPLY_WAIT_SECONDS}s. The message may "
        "still be delivered when the Desktop reconnects; do not resend blindly.')\n"
        "sys.exit(1)\n"
    )
    return f"{shlex.quote(sys.executable or 'python3')} -c {shlex.quote(code)}"


def _hermes_cli() -> str:
    """hermes CLI beside this interpreter, then ``shutil.which``, then the bare name
    (service contexts lack PATH, so a bare "hermes" died with ENOENT).

    The deliver RPC runs on the target gateway, whose process is the venv python — its bin/Scripts directory
    holds the matching ``hermes`` entrypoint. A bare ``"hermes"`` relies on PATH, which is exactly what
    service contexts (systemd units, desktop launchers, non-login SSH shells) do not provide, so delivery
    died with ENOENT there (#93590). When no sibling exists (e.g. running from a source tree without an
    installed script), a ``shutil.which`` lookup runs next — it honors whatever PATH the process does have —
    before falling back to the bare name, preserving today's behavior for interactive shells.
    """
    sibling = Path(sys.executable or "").parent / ("hermes.exe" if sys.platform == "win32" else "hermes")
    return str(sibling) if sibling.is_file() else shutil.which("hermes") or "hermes"


def local_delivery_command(profile: str, query_file: str) -> list[str]:
    """argv that delivers a DM into ``profile``'s Bot Chat on THIS gateway."""
    return [_hermes_cli(), "-p", profile, *BOT_CHAT_TURN_ARGS, "--query-file", query_file]


# Two deliveries into the SAME profile must never run Bot Chat turns concurrently.
# Deliveries are separate ``hermes`` subprocesses, so the lock is a per-profile
# lockfile under ``<root>/bot_relay/locks/`` held with ``fcntl.flock`` for exactly
# the turn window; the kernel releases it on fd close (incl. process death), so a
# crashed turn can never wedge the profile.


# ── per-profile turn lock (#93091) ─────────────────────────────────────────── Two deliveries into the SAME
# target profile must never run their Bot Chat turns concurrently: deliveries spawn separate ``hermes``
# subprocesses, so an in-memory mutex is useless — the lock is a per-profile lockfile under
# ``<root>/bot_relay/locks/`` held with ``fcntl.flock`` for exactly the turn execution window. flock is
# released by the kernel when the holder's fd closes (including process death), so a crashed turn can never
# wedge the profile. A queued delivery waits up to ``bot_mode.turn_wait_seconds`` and then fails with a
# structured 'target_busy' refusal instead of blocking forever.
class TurnBusyError(RuntimeError):
    """A delivery turn is already running for the target profile (``waited_seconds`` ≈ time queued).

    ``reason`` is 'target_busy' — extends the #93091 item-1 structured refusal enum. ``waited_seconds`` is
    roughly how long the caller queued behind the current turn before giving up.
    """

    reason = "target_busy"

    def __init__(self, profile: str, waited_seconds: float):
        self.profile, self.waited_seconds = profile, waited_seconds
        super().__init__(f"target_busy: another delivery turn is already running for profile '{profile}' — "
                         f"queued behind it for ~{int(round(waited_seconds))}s without it finishing. "
                         "The message was NOT delivered; retry shortly.")


def turn_wait_seconds() -> float:
    """Wait budget for a queued delivery turn (config, lazily read)."""
    val = _bot_mode_cfg("turn_wait_seconds", loader="load_config")
    return float(TURN_WAIT_SECONDS_FALLBACK) if val is None else max(0.0, float(val))


def turn_lock_path(root: Path | str, profile: str) -> Path:
    """Per-profile lockfile path (short — safe on macOS temp roots)."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(profile or ""))[:64] or "_"
    return relay_root(root) / LOCKS_DIR / f"{safe}.lock"


@contextlib.contextmanager
def acquire_turn_lock(root: Path | str, profile: str, timeout_seconds: float | None = None) -> Iterator[Path]:
    """Hold ``profile``'s cross-process turn lock for the ``with`` body: non-blocking
    flock probe + short-sleep retry up to the budget (``bot_mode.turn_wait_seconds``
    unless ``timeout_seconds``); raises :class:`TurnBusyError` when exhausted. No
    ordering among waiters, but every waiter is bounded. Without ``fcntl`` (Windows)
    the lock is a no-op — those installs never had this race path."""
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
            with contextlib.suppress(OSError):  # kernel releases on close anyway
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
