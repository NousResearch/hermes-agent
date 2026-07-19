from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
HOME = Path.home()
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_ROOTS = [HOME / ".hermes" / "scripts", _REPO_ROOT / "scripts", _REPO_ROOT / "obsidian-repo" / "scripts"]

def _resolve_script():
    for root in _CANDIDATE_ROOTS:
        if (root / "log_work.py").exists():
            return root / "log_work.py"
    return None

async def handle_work_fast_path(*, event, session_key: str, decision: Dict[str, Any]) -> Optional[str]:
    message = (decision.get("text") or getattr(event, "text", "") or "").strip()
    if not message:
        return None

    def _run(dry_run: bool = True) -> Dict[str, Any]:
        script = _resolve_script()
        if not script:
            return {"ok": False, "error": "log_work.py not found"}
        cmd = [sys.executable, str(script), "message", message]
        if dry_run:
            cmd.append("--dry-run")
        try:
            p = __import__("subprocess").run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        stdout = (p.stdout or "").strip()
        if p.returncode != 0 or not stdout:
            err = (p.stderr or "").strip() or "no output"
            # If it returned needs_agent, don't force it
            if any(k.lower() in err.lower() for k in ["required", "could not", "unknown", "needs_agent"]):
                return {"ok": False, "error": err, "needs_agent": True}
            return {"ok": False, "error": err}
        try:
            payload = json.loads(stdout)
            return payload if isinstance(payload, dict) else {"ok": True, "raw": stdout[:160]}
        except Exception:
            return {"ok": True, "raw": stdout[:160]}

    # Phase 1: dry-run validation
    dry_result = await asyncio.get_running_loop().run_in_executor(None, _run, True)
    if dry_result.get("needs_agent") or not dry_result.get("ok"):
        return None  # Can't handle → fall through to agent

    # Phase 2: real persist
    persist_result = await asyncio.get_running_loop().run_in_executor(None, _run, False)
    if persist_result.get("ok"):
        return "Work ✅ logged"

    return None  # Persist failed → fall through to agent
