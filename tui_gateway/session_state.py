"""Typed view over a gateway session's mutable state — Phase 3 Step 1 of the
Central Brain + Action Runtime design (docs/architecture/central-brain-openclaw.md
§11 "Phase 3 — design detail").

``SessionState`` subclasses ``dict`` deliberately: every existing
``session["key"]`` / ``session.get(...)`` / ``session.setdefault(...)`` /
``"k" in session`` call in tui_gateway.server keeps EXACT dict semantics, so
introducing the type is **zero handler churn** (verified: the gateway has no
membership test, bare ``pop`` without default, or iteration that relies on the
session value dict; the only risk — ``.get()`` returning the default for an
absent key — is preserved because a dict subclass genuinely stores only the
keys that were set, so no field defaults need tuning).

The typed ``@property`` accessors below let handlers migrate from
``session["history"]`` to ``session.history`` incrementally (Phase 3 Steps 2-3).
Each accessor is sugar over the same dict item, so old and new styles stay in
sync. The per-session locks (``history_lock``, ``agent_build_lock``) remain dict
items for now; they become owned attributes once the subscript call-sites are
converted (Step 4).
"""

from __future__ import annotations

import threading
from typing import Any, Optional


class SessionState(dict):
    """A session's live state. A ``dict`` with typed accessors — see module doc."""

    # ── conversation (all guarded by history_lock) ──────────────────
    @property
    def history(self) -> list:
        return self["history"]

    @history.setter
    def history(self, value: list) -> None:
        self["history"] = value

    @property
    def history_lock(self) -> "threading.Lock":
        return self["history_lock"]

    @property
    def history_version(self) -> int:
        return self["history_version"]

    @history_version.setter
    def history_version(self, value: int) -> None:
        self["history_version"] = value

    @property
    def running(self) -> bool:
        return self["running"]

    @running.setter
    def running(self, value: bool) -> None:
        self["running"] = value

    @property
    def inflight_turn(self) -> Optional[dict]:
        return self["inflight_turn"]

    @inflight_turn.setter
    def inflight_turn(self, value: Optional[dict]) -> None:
        self["inflight_turn"] = value

    @property
    def last_active(self) -> float:
        return self["last_active"]

    @last_active.setter
    def last_active(self, value: float) -> None:
        self["last_active"] = value

    # ── agent / lifecycle ───────────────────────────────────────────
    @property
    def agent(self) -> Any:
        return self["agent"]

    @agent.setter
    def agent(self, value: Any) -> None:
        self["agent"] = value

    @property
    def session_key(self) -> str:
        return self["session_key"]

    @session_key.setter
    def session_key(self, value: str) -> None:
        self["session_key"] = value

    @property
    def slash_worker(self) -> Any:
        return self["slash_worker"]

    @slash_worker.setter
    def slash_worker(self, value: Any) -> None:
        self["slash_worker"] = value

    # ── environment / view ──────────────────────────────────────────
    @property
    def transport(self) -> Any:
        return self["transport"]

    @transport.setter
    def transport(self, value: Any) -> None:
        self["transport"] = value

    @property
    def cwd(self) -> str:
        return self["cwd"]

    @cwd.setter
    def cwd(self, value: str) -> None:
        self["cwd"] = value

    @property
    def cols(self) -> int:
        return self["cols"]

    @cols.setter
    def cols(self, value: int) -> None:
        self["cols"] = value
