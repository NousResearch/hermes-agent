"""Gateway fast-path handler for habit confirmations."""
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
HOME = Path.home()
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_ROOTS = [HOME / ".hermes" / "scripts", _REPO_ROOT / "scripts", _REPO_ROOT / "obsidian-repo" / "scripts"]
_KILLER_PHRASES = ["Could not detect habit", "does not look like a completion confirmation", "Unknown habit:"]

def _resolve_script():
    for root in _CANDIDATE_ROOTS:
        if (root / "log_habit.py").exists():
            return root / "log_habit.py"
    return None

async def handle_habit_fast_path(*, event, session_key: str, habit_decision: dict) -> Optional[str]:
    message = (habit_decision.get("text") or getattr(event, "text", "") or "").strip()
    if not message:
        return None
    def _run(dry_run: bool = True) -> Dict[str, Any]:
        script = _resolve_script()
        if not script:
            return {"ok": False, "error": "log_habit.py not found"}
        cmd = [sys.executable, str(script), message]
        if dry_run:
            cmd.append("--dry-run")
        try:
            p = __import__("subprocess").run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as exc:
            return {"ok": False, "error": f"exec: {exc}"}
        stdout = (p.stdout or "").strip()
        stderr = (p.stderr or "").strip()
        if p.returncode != 0 or not stdout:
            return {"ok": False, "error": stderr or "dry-run returned empty"}
        try:
            payload = json.loads(stdout)
        except Exception:
            return {"ok": False, "error": "json parse failed"}
        return payload if isinstance(payload, dict) else {"ok": True}

    # Phase 1: dry-run validation
    dry_result = await asyncio.get_running_loop().run_in_executor(None, _run, True)
    if not dry_result.get("ok"):
        reason = dry_result.get("error") or "unknown"
        if any(k.lower() in reason.lower() for k in _KILLER_PHRASES):
            return None  # Not a habit message → fall through
        return None  # Can't handle → fall through to agent

    # Phase 2: real persist
    persist_result = await asyncio.get_running_loop().run_in_executor(None, _run, False)
    if persist_result.get("ok"):
        row = persist_result.get("row") if isinstance(persist_result.get("row"), dict) else {}
        habit = row.get("habit") or persist_result.get("habit") or habit_decision.get("habit") or "habit"
        date = row.get("date") or ""
        return f"Logged ✅ {habit} · {date}" if date else f"Logged ✅ {habit}"

    return None  # Persist failed → fall through to agent
