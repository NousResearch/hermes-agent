"""Bounded, read-only telemetry projections. No renderer-selected paths."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
import time

from hermes_constants import hermes_home_key
from hermes_state_usage_events import SessionUsageEventsMixin


def profile_id(home):
    return hashlib.sha256(hermes_home_key(home).encode()).hexdigest()


def envelope(home, as_of_us):
    return {
        "version": 1,
        "scope": {"profile": profile_id(home), "profile_selection": "backend_launch",
                  "device": "backend", "device_local": True,
                  "renderer_device_local": "unknown", "remote_telemetry": "unsupported"},
        "as_of_us": as_of_us,
        "as_of": datetime.fromtimestamp(as_of_us / 1_000_000, timezone.utc).isoformat(),
    }


class _TimelineReader(SessionUsageEventsMixin):
    """Reuse the event-store aggregation without SessionDB migrations or write handles."""
    def __init__(self, home):
        self.db_path = Path(home) / "state.db"
        self._conn = None

    def _read_all(self, sql, params):
        if self._conn is None:
            raise sqlite3.OperationalError("telemetry unavailable")
        return self._conn.execute(sql, params).fetchall()

    def _read_one(self, sql, params):
        rows = self._read_all(sql, params)
        return rows[0] if rows else None


def timeline(home):
    reader = _TimelineReader(home)
    try:
        try:
            reader._conn = sqlite3.connect(reader.db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)
            reader._conn.row_factory = sqlite3.Row
            deadline = time.monotonic() + 0.25
            reader._conn.set_progress_handler(lambda: time.monotonic() >= deadline, 1000)
        except sqlite3.Error:
            # The mixin's unavailable shape still supplies exactly 24 NULL bins.
            pass
        result = reader.codex_usage_timeline(profile=profile_id(home))
    finally:
        if reader._conn is not None:
            reader._conn.close()
    return {**envelope(home, result["as_of_us"]), **result,
            "freshness": {"status": "fresh" if result["total"] is not None else "unknown",
                          "stale_after_seconds": 60}}


def quota(home):
    from agent.account_usage import fetch_codex_quota
    from agent.account_usage_quota import CodexQuotaFailure
    from hermes_cli.auth_codex import _read_codex_tokens
    from hermes_cli.auth_constants import AuthError
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(home)
    try:
        try:
            # Runtime resolution may import external CLI credentials or rotate the
            # pool. A read monitor must not do either. auth.json is atomically replaced.
            stored = _read_codex_tokens(_lock=False)["tokens"]
            result = fetch_codex_quota(api_key=stored["access_token"]).to_dict()
            if _read_codex_tokens(_lock=False)["tokens"] != stored:
                result = CodexQuotaFailure(time.time_ns() // 1_000_000_000, "auth").to_dict()
        except AuthError:
            result = CodexQuotaFailure(time.time_ns() // 1_000_000_000, "auth").to_dict()
    finally:
        reset_hermes_home_override(token)
    return {**envelope(home, result["fetched_at"] * 1_000_000),
            "provider": "openai-codex", "quota": result,
            "account_identity": "unknown", "source": "profile_singleton_usage_api",
            "coverage": {"status": "partial" if result["status"] == "ok" else "unavailable",
                         "reason": "account_identity_unverified"},
            "freshness": {"status": "fresh" if result["status"] == "ok" else "unknown",
                          "cached": False, "stale_after_seconds": 60}}
