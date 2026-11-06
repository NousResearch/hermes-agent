"""Real compute-host child used by the two-home profile-ownership regression.

The child runs the production JSONL host loop.  It only seeds a synthetic, provider-free
session so the parent can exercise the real off-turn ``session.save`` control path without
making an LLM request.  The marker is read from the child home on disk, which keeps the
assertion about startup ownership independent from ambient process environment state.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from tui_gateway.compute_host import run_host


def _seed_probe_session(server) -> None:
    home = Path(os.environ["HERMES_HOME"]).resolve()
    sid = (home / "profile-owner-session.txt").read_text(encoding="utf-8").strip()
    if not sid:
        raise RuntimeError(f"profile-owner-session.txt is empty: {home}")
    marker = (home / "profile-owner-marker.txt").read_text(encoding="utf-8").strip()
    agent = SimpleNamespace(
        model="profile-owner-fixture",
        provider="fixture",
        tools=[],
        _cached_system_prompt="",
        session_start=None,
        session_id=sid,
    )
    now = time.time()
    server._sessions[sid] = {
        "agent": agent,
        "session_key": sid,
        "history": [{"role": "user", "content": f"profile-marker:{marker}"}],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "inflight_turn": None,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "edit_snapshots": {},
        "tool_started_at": {},
        "created_at": now,
        "last_active": now,
        "cwd": str(home),
        "profile_home": str(home),
        "source": "profile-owner-fixture",
        "model_override": None,
        "transport": None,
    }


def _run_child() -> None:
    # ``tui_gateway.server`` redirects the module's stdout during import.  Capture the real
    # supervisor pipe before that import and pass it explicitly to the production host loop.
    protocol_stdout = sys.stdout
    from tui_gateway import server

    _seed_probe_session(server)
    run_host(stdout=protocol_stdout)


if __name__ == "__main__":  # pragma: no cover - exercised in a real child process
    _run_child()
