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
        if (root / "log_health.py").exists():
            return root / "log_health.py"
    return None

async def handle_health_fast_path(*, event, session_key: str, decision: Dict[str, Any]) -> Optional[str]:
    message = (decision.get("text") or getattr(event, "text", "") or "").strip()
    if not message:
        return None
    health_type = decision.get("type") or "yoga"

    def _run(dry_run: bool = True) -> Dict[str, Any]:
        script = _resolve_script()
        if not script:
            return {"ok": False, "error": "log_health.py not found"}
        args = [sys.executable, str(script), health_type]
        if dry_run:
            args.append("--dry-run")
        try:
            p = __import__("subprocess").run(args, capture_output=True, text=True, timeout=30)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        stdout = (p.stdout or "").strip()
        if p.returncode != 0 or not stdout:
            return {"ok": False, "error": (p.stderr or "").strip() or "log_health.py returned no output"}
        try:
            payload = json.loads(stdout)
        except Exception:
            return {"ok": True, "raw": stdout[:160]}
        return payload if isinstance(payload, dict) else {"ok": True, "raw": stdout[:160]}

    # Phase 1: dry-run validation
    dry_result = await asyncio.get_running_loop().run_in_executor(None, _run, True)
    if not dry_result.get("ok"):
        return None  # Can't handle → fall through to agent

    # Phase 2: real persist
    persist_result = await asyncio.get_running_loop().run_in_executor(None, _run, False)
    if persist_result.get("ok"):
        return f"Health ✅ {health_type} logged"

    return None  # Persist failed → fall through to agent
