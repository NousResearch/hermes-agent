#!/usr/bin/env python3
"""Local credit billing for EasyHermes KIE media generation.

A slim, single-tenant port of the Kari/Langflow ``lfx.kari_billing`` module:
pure pricing functions + a local SQLite ledger (one ``local`` account). The
multi-tenant / hub-relay / redeem-code / pay-order machinery is dropped —
EasyHermes is a single local instance.

Billing is **opt-in**. It only charges when enabled via config
(``billing.enabled: true`` in ``config.yaml``) or env (``KIE_BILLING=1``); when
disabled (the default) the KIE image/video tools generate freely and this
module never touches disk. Turn it on for the productized build where users buy
credits.

Pricing parity with Kari: 1 元 = 25 积分 (1 credit = 0.04 元). Image/video
credit numbers match the Kari tables exactly.

Granting credits (until the wallet UI lands)::

    uv run python -c "import tools.kie_billing as b; print(b.grant(1000, 'topup'))"
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

logger = logging.getLogger(__name__)

CREDIT_RMB = 0.04  # 1 credit = 0.04 元 (1 元 = 25 credits)
_LOCAL_UID = "local"
_lock = threading.Lock()


class InsufficientCreditsError(Exception):
    def __init__(self, needed: float, balance: float):
        self.needed = float(needed)
        self.balance = float(balance)
        super().__init__(
            f"积分不足:本次需要 {self.needed:.2f} 积分,当前余额 {self.balance:.2f}。请充值后重试。"
        )


# --------------------------- enable gate ---------------------------


def billing_enabled() -> bool:
    """Return True when credit billing should gate KIE generation.

    Enabled by env ``KIE_BILLING`` (1/true/yes/on) or config ``billing.enabled``.
    Default: False (generation is free; the ledger is never created).
    """
    env = (os.getenv("KIE_BILLING") or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("billing") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            return bool(section.get("enabled"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("billing_enabled: could not read config: %s", exc)
    return False


# --------------------------- pricing (pure) ---------------------------


def credits_for_image(resolution: str | None, kind: str = "text_to_image", count: int = 1) -> int:
    r = (resolution or "1K").strip().upper()
    n = max(1, int(count or 1))
    if kind == "image_to_image":
        unit = 30 if r == "4K" else 21 if r == "2K" else 13
    else:
        unit = 25 if r == "4K" else 16 if r == "2K" else 10
    return unit * n


_VIDEO_CREDIT_PER_USD_SEC = 1.3 * 7 / CREDIT_RMB  # markup 1.3 × fx 7 ÷ credit price
_VIDEO_USD = {
    "happyhorse": {"720p": 0.14, "1080p": 0.24},
    "seedance": {"480p": 0.095, "720p": 0.205, "1080p": 0.51},
    "seedance_video": {"480p": 0.057, "720p": 0.125, "1080p": 0.31},
}


def _video_usd_per_sec(model: str, resolution: str | None, *, has_video_input: bool = False) -> float:
    table = _VIDEO_USD["seedance_video" if (model == "seedance" and has_video_input) else model]
    return table.get((resolution or "1080p").strip().lower(), max(table.values()))


def credits_for_video(model: str, resolution: str | None, seconds: int, *, has_video_input: bool = False) -> float:
    secs = max(1, int(seconds or 1))
    return round(
        _video_usd_per_sec(model, resolution, has_video_input=has_video_input) * _VIDEO_CREDIT_PER_USD_SEC * secs,
        2,
    )


def pricing_table() -> dict:
    img = ["1K", "2K", "4K"]
    vid = ["480p", "720p", "1080p"]
    return {
        "credit_rmb": CREDIT_RMB,
        "image": {
            "text_to_image": {r: credits_for_image(r, "text_to_image", 1) for r in img},
            "image_to_image": {r: credits_for_image(r, "image_to_image", 1) for r in img},
        },
        "video_per_second": {
            "seedance": {r: credits_for_video("seedance", r, 1) for r in vid},
        },
    }


# --------------------------- local ledger ---------------------------


def _db_path() -> Path:
    override = os.getenv("KIE_BILLING_DB")
    if override:
        path = Path(override)
    else:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "kie_billing.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_db_path()), timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS account (user_id TEXT PRIMARY KEY, balance REAL NOT NULL DEFAULT 0)")
    c.execute(
        "CREATE TABLE IF NOT EXISTS txn (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, "
        "ts REAL NOT NULL, delta REAL NOT NULL, kind TEXT NOT NULL, note TEXT, balance_after REAL NOT NULL)"
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_txn_user ON txn (user_id, id DESC)")
    return c


def get_balance(uid: str = _LOCAL_UID) -> float:
    with closing(_conn()) as c:
        row = c.execute("SELECT balance FROM account WHERE user_id=?", (str(uid),)).fetchone()
        return float(row[0]) if row else 0.0


def _apply(delta: float, kind: str, note: str = "", uid: str = _LOCAL_UID) -> float:
    uid = str(uid)
    with _lock, closing(_conn()) as c:
        try:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT balance FROM account WHERE user_id=?", (uid,)).fetchone()
            bal = float(row[0]) if row else 0.0
            if delta < 0 and bal + delta < -1e-9:
                c.execute("ROLLBACK")
                raise InsufficientCreditsError(-delta, bal)
            newbal = round(bal + delta, 6)
            c.execute(
                "INSERT INTO account (user_id, balance) VALUES (?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET balance=?",
                (uid, newbal, newbal),
            )
            c.execute(
                "INSERT INTO txn (user_id, ts, delta, kind, note, balance_after) VALUES (?,?,?,?,?,?)",
                (uid, time.time(), delta, kind, note, newbal),
            )
            c.commit()
            return newbal
        except InsufficientCreditsError:
            raise
        except Exception:
            c.execute("ROLLBACK")
            raise


def ensure_balance(amount: float, uid: str = _LOCAL_UID) -> None:
    """Raise :class:`InsufficientCreditsError` if the balance can't cover ``amount``."""
    bal = get_balance(uid)
    if bal + 1e-9 < float(amount):
        raise InsufficientCreditsError(amount, bal)


def charge(amount: float, kind: str, note: str = "", uid: str = _LOCAL_UID) -> float:
    """Deduct ``amount`` credits. Returns the new balance. No-op for amount<=0."""
    amount = max(0.0, float(amount))
    return get_balance(uid) if amount == 0 else _apply(-amount, kind, note, uid)


def grant(amount: float, note: str = "topup", uid: str = _LOCAL_UID) -> float:
    """Add credits to the wallet. Returns the new balance."""
    return _apply(abs(float(amount)), "grant", note, uid)


def transactions(limit: int = 50, uid: str = _LOCAL_UID) -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT ts, delta, kind, note, balance_after FROM txn WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (str(uid), int(limit)),
        ).fetchall()
    return [{"ts": r[0], "delta": r[1], "kind": r[2], "note": r[3], "balance_after": r[4]} for r in rows]
