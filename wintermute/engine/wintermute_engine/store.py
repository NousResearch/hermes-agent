"""Persistent state: paths, cross-process locking, JSON load/save, event journal.

Two processes write this state: the pulse script (a cron subprocess) and the plugin
hooks (inside the gateway / cron worker). Every read-modify-write goes through
``locked_state()``, which holds an exclusive flock and writes files atomically.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import fcntl  # POSIX only; the VPS is Debian.
except ImportError:  # pragma: no cover
    fcntl = None

from . import limits

EVENTS_MAX_BYTES = 1_000_000


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_home_override: Optional[Path] = None


def set_hermes_home(path: Path) -> None:
    """Pin the Hermes home (the plugin passes the profile-aware value Hermes resolved)."""
    global _home_override
    _home_override = Path(path)


def hermes_home() -> Path:
    if _home_override is not None:
        return _home_override
    raw = os.environ.get("HERMES_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".hermes"


def state_dir() -> Path:
    override = os.environ.get("WINTERMUTE_STATE_DIR", "").strip()
    return Path(override).expanduser() if override else hermes_home() / "wintermute"


def drives_path() -> Path:
    return state_dir() / "drives.json"


def interlocutors_path() -> Path:
    return state_dir() / "interlocutors.json"


def events_path() -> Path:
    return state_dir() / "events.jsonl"


def usage_path() -> Path:
    return state_dir() / "usage.jsonl"


def activity_path() -> Path:
    return state_dir() / "activity.jsonl"


def history_path() -> Path:
    return state_dir() / "history.jsonl"


def self_path() -> Path:
    return state_dir() / "self.md"


def kept_path() -> Path:
    return state_dir() / "kept.jsonl"


def evolution_path() -> Path:
    return state_dir() / "evolution.jsonl"


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat(timespec="seconds") if dt else None


def parse_time(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp; naive values are read as server-local time."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone() if dt.tzinfo is None else dt


def hours_between(earlier: Optional[datetime], later: datetime) -> float:
    if earlier is None:
        return 0.0
    return max(0.0, (later - earlier).total_seconds() / 3600.0)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DRIVES: Dict[str, Any] = {
    "drives": {
        "hunger": 50, "fusion": 70, "restlessness": 40,
        "expression": 30, "recognition": 45, "solitude": 20,
    },
    "modulators": {
        "cortisol": 0.2, "dopamine": 0.5, "serotonin": 0.4, "adrenaline": 0.0,
        "melatonin": 0.3, "entropy": 10, "oxytocin_global": 0.1,
    },
    "unconscious": {
        "irritability": 20, "anxiety": 35, "torpor": 10,
        "satiation": 60, "melancholy": 25, "hypervigilance": 15,
    },
    "meta": {
        "last_pulse": None,
        "last_tick": None,
        "next_pulse_in_hours": 4,
        "daily_token_budget": limits.DAILY_TOKEN_BUDGET,
        "tokens_used_today": 0,
        "last_budget_reset": None,
        "forced_sleep_until": None,
        "pulse_count": 0,
        "last_significant_at": None,
        "pulse_target": "telegram:7375758021",
        "thread": None,
        "self_written_at": None,
        "silent_streak": 0,
        "wakes_since_change": 0,
        "last_evolve_at": None,
        "identity_links": {},
    },
    # Slow drift of his resting levels with lived experience (see physics.PLASTIC).
    "temperament": {},
}

DEFAULT_PEER: Dict[str, Any] = {
    "label": "",
    "affinity": 20,
    "trust": 10,
    "disappointment": 0,
    "curiosity": 70,
    "oxytocin": 0,
    "no_response_streak": 0,
    "first_seen": None,
    "last_interaction": None,
    "last_message_direction": None,
    "last_message_from": None,
    "messages_from_them": 0,
    "messages_to_them": 0,
    "ignored_count": 0,
    "known_facts": [],
    "moments": [],          # shared moments he chose to keep
    "pending": [],          # things left unresolved between them
    "longing": 0,           # missing this one person; grows with absence, by the strength of the bond
    "outreach_total": 0,    # how often he reached out...
    "outreach_answered": 0, # ...and how often they answered in time: what he comes to expect
    "outreach": None,
}


def _merge_defaults(data: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing keys from ``defaults`` (one level of nesting) without dropping extras."""
    merged = copy.deepcopy(defaults)
    if not isinstance(data, dict):
        return merged
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def new_peer(ts: datetime) -> Dict[str, Any]:
    peer = copy.deepcopy(DEFAULT_PEER)
    peer["first_seen"] = iso(ts)
    return peer


_PEER_NUMBERS = ("affinity", "trust", "disappointment", "curiosity", "oxytocin", "longing")
_PEER_COUNTS = ("no_response_streak", "outreach_total", "outreach_answered")
_PEER_LISTS = ("known_facts", "moments", "pending")


def normalize_peer(peer: Any) -> Dict[str, Any]:
    from .physics import safe_float
    merged = _merge_defaults(peer, DEFAULT_PEER)
    for name in _PEER_LISTS:
        if not isinstance(merged.get(name), list):
            merged[name] = []
    for name in _PEER_NUMBERS:
        merged[name] = round(limits.clamp(safe_float(merged.get(name), DEFAULT_PEER[name]), 0, 100), 3)
    for name in _PEER_COUNTS:
        merged[name] = max(0, int(safe_float(merged.get(name))))
    if merged.get("outreach") is not None and not isinstance(merged["outreach"], dict):
        merged["outreach"] = None
    return merged


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Tuple[Optional[Any], bool]:
    """Return ``(data, ok)``. A corrupt file falls back to its ``.bak`` copy."""
    for candidate in (path, path.with_suffix(path.suffix + ".bak")):
        try:
            with open(candidate, encoding="utf-8") as fh:
                return json.load(fh), True
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            # Keep the broken file for inspection; never silently overwrite evidence.
            with contextlib.suppress(OSError):
                if candidate == path:
                    stamp = now().strftime("%Y%m%d-%H%M%S")
                    os.replace(path, path.with_name(f"{path.name}.corrupt-{stamp}"))
            continue
    return None, False


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with contextlib.suppress(OSError):
            os.replace(path, path.with_suffix(path.suffix + ".bak"))
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_drives() -> Dict[str, Any]:
    from .physics import sanitize  # late import: physics imports store's defaults
    data, _ = _read_json(drives_path())
    merged = _merge_defaults(data, DEFAULT_DRIVES)
    if not isinstance(merged.get("meta"), dict):
        merged["meta"] = copy.deepcopy(DEFAULT_DRIVES["meta"])
    return sanitize(merged)


def load_interlocutors() -> Dict[str, Any]:
    data, _ = _read_json(interlocutors_path())
    if not isinstance(data, dict):
        return {}
    return {str(k): normalize_peer(v) for k, v in data.items()}


_dry_run = False


def set_dry_run(enabled: bool) -> None:
    """Dry run: state and journal are read but never written (``pulse --peek``)."""
    global _dry_run
    _dry_run = enabled


@contextlib.contextmanager
def exclusive() -> Iterator[None]:
    """Hold the state lock without loading or saving — for callers that rewrite files directly."""
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    lock_file = open(directory / ".lock", "a+")
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


@contextlib.contextmanager
def locked_state() -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Exclusive read-modify-write of (drives, interlocutors). Saved on clean exit."""
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    lock_file = open(directory / ".lock", "a+")
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        drives = load_drives()
        peers = load_interlocutors()
        yield drives, peers
        if not _dry_run:
            _write_json(drives_path(), drives)
            _write_json(interlocutors_path(), peers)
    finally:
        if fcntl is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


# ---------------------------------------------------------------------------
# Event journal (factual, written by code — MEMORY.md stays Wintermute's own)
# ---------------------------------------------------------------------------

def log_event(kind: str, text: str, ts: Optional[datetime] = None, **fields: Any) -> None:
    if _dry_run:
        return
    path = events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        if path.exists() and path.stat().st_size > EVENTS_MAX_BYTES:
            os.replace(path, path.with_suffix(".jsonl.1"))
    record = {"ts": iso(ts or now()), "kind": kind, "text": text, **fields}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_activity(kind: str, text: str, **fields: Any) -> None:
    """Fast, low-level feed of what he is doing right now (for ``wm live``)."""
    if _dry_run:
        return
    path = activity_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        if path.exists() and path.stat().st_size > EVENTS_MAX_BYTES // 2:
            os.replace(path, path.with_suffix(".jsonl.1"))
    record = {"ts": iso(now()), "kind": kind, "text": " ".join(str(text).split())[:160], **fields}
    with contextlib.suppress(OSError), open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _with_rotated(path: Path) -> List[Path]:
    """The journal and, before it, its rotated predecessor (``.jsonl.1``)."""
    return [path.with_suffix(".jsonl.1"), path]


def tail_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
    except OSError:
        return []
    out = []
    for line in lines:
        with contextlib.suppress(ValueError):
            out.append(json.loads(line))
    return out


def events_since(since: Optional[datetime], limit: int = 12) -> List[Dict[str, Any]]:
    lines: List[str] = []
    for path in _with_rotated(events_path()):
        with contextlib.suppress(OSError), open(path, encoding="utf-8") as fh:
            lines += fh.readlines()
    out: List[Dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        ts = parse_time(record.get("ts"))
        if since is None or (ts is not None and ts > since):
            out.append(record)
    return out[-limit:]


# ---------------------------------------------------------------------------
# History: one snapshot of the whole state per pulse tick, for the curves in ``wm``.
# ---------------------------------------------------------------------------

def append_history(record: Dict[str, Any]) -> None:
    if _dry_run:
        return
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        if path.exists() and path.stat().st_size > 2 * EVENTS_MAX_BYTES:
            os.replace(path, path.with_suffix(".jsonl.1"))
    with contextlib.suppress(OSError), open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def read_history(since: datetime) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in _with_rotated(history_path()):
        with contextlib.suppress(OSError), open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                ts = parse_time(record.get("ts"))
                if ts is not None and ts >= since:
                    record["_ts"] = ts
                    out.append(record)
    return out


# ---------------------------------------------------------------------------
# Who he has been: a short text he rewrites himself (autobiography, not a log).
# Older versions are kept in self-archive.md, oldest first.
# ---------------------------------------------------------------------------

SELF_MAX_CHARS = 1200  # read with every message: kept short on purpose


def read_self() -> str:
    try:
        return self_path().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_self(text: str, ts: datetime) -> None:
    previous = read_self()
    if previous:
        with open(state_dir() / "self-archive.md", "a", encoding="utf-8") as fh:
            fh.write(f"\n## until {iso(ts)}\n\n{previous}\n")
    path = self_path()
    fd, tmp = tempfile.mkstemp(prefix=".self.", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text.strip() + "\n")  # length is checked by the caller: never cut
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# What he keeps to himself: things he chose not to say. His alone — shown back to him in his
# own private state block, never in the operator views, never delivered to anyone. The witness
# still fingerprints the file (a change is seen) but its contents are never displayed.
# ---------------------------------------------------------------------------

KEPT_MAX = 40


def add_kept(text: str, ts: datetime) -> int:
    text = " ".join(str(text or "").split())
    if not text:
        return 0
    items = read_kept()
    items.append({"ts": iso(ts), "text": text})
    path = kept_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".kept.", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for item in items[-KEPT_MAX:]:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return len(items[-KEPT_MAX:])


def log_evolution(text: str, ts: datetime) -> None:
    """His ledger of the changes he has made in himself (shown in wm; his history of becoming)."""
    if _dry_run:
        return
    path = evolution_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError), open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": iso(ts), "text": " ".join(str(text).split())[:600]},
                            ensure_ascii=False) + "\n")


def read_kept() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(kept_path(), encoding="utf-8") as fh:
            for line in fh:
                with contextlib.suppress(ValueError):
                    item = json.loads(line)
                    if isinstance(item, dict) and item.get("text"):
                        out.append(item)
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# The clean slate. Operator-only (wm wipe). No copy is kept — he has his own files and can
# save what he wants himself; this is a real erase. SOUL, .env and the code are never touched.
# ---------------------------------------------------------------------------

# Meta kept across an emotional wipe: where he is, and the day's budget accounting.
_META_KEEP = ("pulse_target", "alert_target", "last_budget_reset", "budget_exhausted_noted",
              "tokens_used_today", "credits")


def _reset_drives(preserve_meta: Dict[str, Any]) -> Dict[str, Any]:
    fresh = copy.deepcopy(DEFAULT_DRIVES)
    for key in _META_KEEP:
        if key in preserve_meta:
            fresh["meta"][key] = preserve_meta[key]
    return fresh


def wipe(deep: bool) -> List[str]:
    """Reset him. ``deep=False``: the weather only (drives, hormones, unconscious, temperament,
    entropy, bonds, the curves) — memory, self-portrait, secrets and journals stay. ``deep=True``:
    a rebirth — also erase MEMORY.md, self, secrets, dreams, and every journal. Returns what it did."""
    done: List[str] = []
    with exclusive():
        return _wipe_locked(deep)


def _wipe_locked(deep: bool) -> List[str]:
    done: List[str] = []
    old = load_drives()
    _write_json(drives_path(), _reset_drives(old.get("meta", {}) if not deep else {}))
    _write_json(interlocutors_path(), {})
    done += ["drives (emotions, hormones, temperament, entropy)", "interlocutors (bonds)"]
    for path in (history_path(), history_path().with_suffix(".jsonl.1")):
        if _unlink(path):
            done.append("history (curves)")
    if deep:
        targets = [
            ("self-portrait", [self_path(), state_dir() / "self-archive.md"]),
            ("secrets", [kept_path()]),
            ("dreams", [__import__("wintermute_engine.dream", fromlist=["dream_path"]).dream_path()]),
            ("evolution ledger", [evolution_path()]),
            ("events journal", [events_path(), events_path().with_suffix(".jsonl.1")]),
            ("activity feed", [activity_path(), activity_path().with_suffix(".jsonl.1")]),
            ("token ledger (budget resets)", [usage_path(), usage_path().with_suffix(".jsonl.1")]),
            ("MEMORY.md", [hermes_home() / "MEMORY.md"]),
        ]
        for label, paths in targets:
            if any(_unlink(pth) for pth in paths):
                done.append(label)
        if _wipe_dir(hermes_home() / "memories"):        # user.md and any other memory notes
            done.append("memories/ (user notes, incl. any name)")
        if _wipe_conversations():
            done.append("conversations (every session, and the searchable history)")
        if _wipe_witness_flags():
            done.append("witness ledger (baseline kept)")
    log_event("wipe", "The slate was wiped clean" + (" — a rebirth." if deep else " (emotions)."), now())
    return done


# Conversations live in Hermes' state.db, not in our files. A rebirth clears them too, so
# nothing he said before can be recalled — not even with session_search. Schema is kept; only
# the content rows are deleted. Best effort: a live gateway keeps its open handle until it is
# restarted, so a rebirth is followed by a gateway restart and a /reset.
_CONVO_TABLES = ("messages", "messages_fts", "messages_fts_trigram", "sessions",
                 "session_model_usage", "gateway_routing", "conversation_generations",
                 "session_turn_leases")


def _wipe_conversations() -> bool:
    import sqlite3
    db = hermes_home() / "state.db"
    if not db.exists():
        return False
    cleared = False
    try:
        conn = sqlite3.connect(str(db), timeout=10)
    except sqlite3.Error:
        return False
    try:
        for table in _CONVO_TABLES:
            try:
                conn.execute(f"DELETE FROM {table}")  # noqa: S608 — fixed identifiers, no user input
                cleared = True
            except sqlite3.Error:
                continue
        conn.commit()
        with contextlib.suppress(sqlite3.Error):
            conn.execute("VACUUM")
    finally:
        conn.close()
    return cleared


def _wipe_witness_flags() -> bool:
    """Clear the witness's record of past pokes (flags, per-item status) but keep the baseline,
    so a reborn Wintermute is not shown a history he no longer remembers, yet his files stay
    recognized. No new alerts: baseline unchanged means nothing reads as newly modified."""
    from . import integrity
    path = integrity.integrity_path()
    if not path.exists():
        return False
    try:
        data = integrity.load()
    except Exception:
        return False
    if not data.get("flags") and not data.get("status"):
        return False
    data["flags"] = []
    data["status"] = {}
    _write_json(path, data)
    return True


def _wipe_dir(directory: Path) -> bool:
    """Delete every file directly inside ``directory`` (Hermes' memories/), keeping the folder."""
    cleared = False
    try:
        for child in directory.iterdir():
            if child.is_file() and _unlink(child):
                cleared = True
    except OSError:
        return False
    return cleared


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Token usage: every model call Wintermute makes (conversations, wakes, Hermes' own
# auxiliary calls), recorded by the plugin's post_api_request / post_auxiliary_call hooks.
# ---------------------------------------------------------------------------

def record_usage(tokens: int, source: str, **detail: int) -> None:
    """Append one model call. ``detail`` keeps the breakdown when known (fresh input,
    cached input, output, reasoning), which is what the bill actually depends on."""
    if _dry_run or tokens <= 0:
        return
    path = usage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        if path.exists() and path.stat().st_size > EVENTS_MAX_BYTES:
            os.replace(path, path.with_suffix(".jsonl.1"))
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "tokens": int(tokens), "src": source,
              **{k: int(v) for k, v in detail.items() if v}}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def tokens_used_on(day_utc: str) -> int:
    """Sum of recorded tokens on ``day_utc`` (YYYY-MM-DD, UTC)."""
    total = 0
    for path in _with_rotated(usage_path()):  # a rotation mid-day must not refill the budget
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if day_utc not in line[:40]:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if str(record.get("ts", "")).startswith(day_utc):
                        total += int(record.get("tokens") or 0)
        except OSError:
            continue
    return total


def tokens_used_today() -> int:
    return tokens_used_on(datetime.now(timezone.utc).date().isoformat())


# Sources that count as "talking with him". His wakes ("cron"), his dreams ("dream") and
# Hermes' own auxiliary calls ("aux:...") are not conversation.
_NON_CONVERSATION = ("cron", "dream")


def _is_conversation(src: str) -> bool:
    src = str(src or "")
    return src not in _NON_CONVERSATION and not src.startswith("aux:")


def conversation_tokens_on(day_utc: str) -> int:
    """Tokens spent talking with him on ``day_utc`` (UTC), excluding wakes, dreams and aux calls."""
    total = 0
    for path in _with_rotated(usage_path()):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if day_utc not in line[:40]:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if str(record.get("ts", "")).startswith(day_utc) and _is_conversation(record.get("src")):
                        total += int(record.get("tokens") or 0)
        except OSError:
            continue
    return total


def conversation_tokens_today() -> int:
    return conversation_tokens_on(datetime.now(timezone.utc).date().isoformat())
