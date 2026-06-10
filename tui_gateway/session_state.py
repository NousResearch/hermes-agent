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

Phase 3 Step 2 absorbs the gateway's inline lock-dances into methods so the
locking contract lives here instead of being re-spelled at every handler:
``snapshot_history`` / ``commit_compaction`` (the compaction snapshot +
compare-and-swap), ``begin_turn`` / ``end_turn`` (the turn-lifecycle CAS), and
``build_once`` (the one-shot agent-build guard). Each method mirrors the
server.py block it replaced EXACTLY — same lock, same checks, same order.
"""

from __future__ import annotations

import threading
import time
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

    @property
    def explicit_cwd(self) -> bool:
        return self["explicit_cwd"]

    @explicit_cwd.setter
    def explicit_cwd(self, value: bool) -> None:
        self["explicit_cwd"] = value

    # ── attachments / images (Phase 3 Step 4) ───────────────────────
    @property
    def attached_images(self) -> list:
        return self["attached_images"]

    @attached_images.setter
    def attached_images(self, value: list) -> None:
        self["attached_images"] = value

    @property
    def image_counter(self) -> int:
        return self["image_counter"]

    @image_counter.setter
    def image_counter(self, value: int) -> None:
        self["image_counter"] = value

    @property
    def edit_snapshots(self) -> dict:
        return self["edit_snapshots"]

    @edit_snapshots.setter
    def edit_snapshots(self, value: dict) -> None:
        self["edit_snapshots"] = value

    # ── display / per-session preferences (Phase 3 Step 4) ──────────
    @property
    def pending_title(self) -> Optional[str]:
        return self["pending_title"]

    @pending_title.setter
    def pending_title(self, value: Optional[str]) -> None:
        self["pending_title"] = value

    @property
    def show_reasoning(self) -> bool:
        return self["show_reasoning"]

    @show_reasoning.setter
    def show_reasoning(self, value: bool) -> None:
        self["show_reasoning"] = value

    @property
    def tool_progress_mode(self) -> str:
        return self["tool_progress_mode"]

    @tool_progress_mode.setter
    def tool_progress_mode(self, value: str) -> None:
        self["tool_progress_mode"] = value

    @property
    def tool_started_at(self) -> dict:
        return self["tool_started_at"]

    @tool_started_at.setter
    def tool_started_at(self, value: dict) -> None:
        self["tool_started_at"] = value

    @property
    def personality(self) -> str:
        return self["personality"]

    @personality.setter
    def personality(self, value: str) -> None:
        self["personality"] = value

    @property
    def model_override(self) -> dict:
        return self["model_override"]

    @model_override.setter
    def model_override(self, value: dict) -> None:
        self["model_override"] = value

    # ── lock dances (Phase 3 Step 2) ────────────────────────────────
    # Each method is a verbatim lift of an inline ``with history_lock:``
    # (or ``agent_build_lock``) block from tui_gateway.server — same lock,
    # same ``.get()`` defaults, same ordering — so call sites keep EXACT
    # observable behavior.

    def snapshot_history(self) -> "tuple[list, int]":
        """Copy (history, history_version) under the lock.

        The compaction path snapshots so the LLM-bound compression call does
        NOT hold ``history_lock`` for the duration of the request — otherwise
        other handlers acquiring the lock (prompt.submit etc.) block on the
        dispatcher loop while compaction runs.
        """
        with self["history_lock"]:
            return list(self.get("history", [])), int(self.get("history_version", 0))

    def commit_compaction(self, new_history: list, expected_version: int) -> bool:
        """Compare-and-swap the compacted history back in.

        Returns False (committing nothing) when ``history_version`` moved
        since :meth:`snapshot_history` — external mutation during compaction
        means the compressed result would clobber concurrent edits. On match,
        replaces history and bumps the version past the snapshot.
        """
        with self["history_lock"]:
            if int(self.get("history_version", 0)) != expected_version:
                return False
            self["history"] = new_history
            self["history_version"] = expected_version + 1
            return True

    def begin_turn(self) -> bool:
        """Claim the session for a turn: running False→True atomically.

        Returns False when a turn is already running (caller rejects with
        4009 "session busy"). On success also refreshes ``last_active``.
        """
        with self["history_lock"]:
            if self.get("running"):
                return False
            self["running"] = True
            self["last_active"] = time.time()
            return True

    def end_turn(self) -> None:
        """Release the session after a turn — the turn thread's ``finally``.

        Clears ``running``, refreshes ``last_active``, and drops the
        inflight-turn snapshot, all under one lock acquisition (mirrors the
        ``_run_prompt_submit`` finally block).
        """
        with self["history_lock"]:
            self["running"] = False
            self["last_active"] = time.time()
            self["inflight_turn"] = None

    def build_once(self) -> bool:
        """One-shot agent-build guard: True iff THIS caller should build.

        Mirrors ``_start_agent_build``'s dance: lazily create
        ``agent_build_lock``, and under it return False when the agent is
        already built (``agent_ready`` set) or a build was already claimed
        (``agent_build_started``); otherwise claim the build and return True.
        """
        ready = self.get("agent_ready")
        lock = self.setdefault("agent_build_lock", threading.Lock())
        with lock:
            if (ready is not None and ready.is_set()) or self.get(
                "agent_build_started"
            ):
                return False
            self["agent_build_started"] = True
            return True
