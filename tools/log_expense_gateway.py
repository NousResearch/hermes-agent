"""Gateway fast-path handler for expense logging."""
from __future__ import annotations
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional
HOME = Path.home()
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_ROOTS = [HOME / ".hermes" / "scripts", _REPO_ROOT / "scripts", _REPO_ROOT / "obsidian-repo" / "scripts"]

_CURRENCY_AMOUNT_RE = re.compile(r'(?:₹|Rs\.?|INR|rs\.?|inr)?\s*(\d[\d,]*\.?\d*)')
_AMOUNT_RE = re.compile(r'(?:\b|^)(\d[\d,]*\.?\d*)(?:\b|$)')
_NOT_MERCHANT = {'via','paid','using','with','through','on','at','from','today','yesterday','rs','inr'}

def _clean(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9&.'’\-]", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip().rstrip(".,;")
    return s

def _guess_amount_merchant(text: str):
    msg = (text or "").strip()
    amount = None
    m = _CURRENCY_AMOUNT_RE.search(msg.replace(",", ""))
    if m:
        amount = m.group(1)
    else:
        m = _AMOUNT_RE.search(msg.replace(",", ""))
        if m:
            amount = m.group(1)
    stop_words = _NOT_MERCHANT | {str(amount), amount, "for", "and", "the", "a", "an", "of", "in", "kg", "kgs"}
    words = [w for w in re.split(r"\s+", msg) if w.lower() not in stop_words and w]
    if amount:
        words = [w for w in words if w.replace(".", "").replace(",", "") != amount]
    merchant = _clean(" ".join(words[:3])) if words else None
    if merchant and amount and merchant.startswith(amount):
        merchant = merchant[len(amount):].strip()
    if merchant:
        merchant = merchant.title()
    return amount, merchant

def _resolve_script():
    for root in _CANDIDATE_ROOTS:
        if (root / "log_expense.py").exists():
            return root / "log_expense.py"
    return None

async def handle_expense_fast_path(*, event, session_key: str, expense_decision: Dict[str, Any]) -> Optional[str]:
    message = (expense_decision.get("text") or getattr(event, "text", "") or "").strip()
    if not message:
        return None
    amount_hint, merchant_hint = _guess_amount_merchant(message)

    def _run(retry: bool = False, dry_run: bool = True) -> Dict[str, Any]:
        script = _resolve_script()
        if not script:
            return {"ok": False, "error": "log_expense.py not found"}
        if retry and amount_hint:
            cmd = [sys.executable, str(script), "--amount", amount_hint or "0", "--merchant", merchant_hint or "Expense"]
            if not merchant_hint:
                cmd += ["--note", message[:40]]
        else:
            cmd = [sys.executable, str(script), message]
        if dry_run:
            cmd.append("--dry-run")
        try:
            p = __import__("subprocess").run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        stdout = (p.stdout or "").strip()
        stderr = (p.stderr or "").strip()
        if p.returncode != 0 or not stdout:
            err_text = stdout or stderr or "log_expense.py returned no output"
            try:
                err_payload = json.loads(err_text)
                if isinstance(err_payload, dict):
                    return err_payload
            except Exception:
                pass
            return {"ok": False, "error": err_text[:400]}
        try:
            payload = json.loads(stdout)
            return payload if isinstance(payload, dict) else {"ok": True}
        except Exception:
            return {"ok": True}

    # Phase 1: dry-run with natural language
    dry_result = await asyncio.get_running_loop().run_in_executor(None, _run, False, True)
    if dry_result.get("ok"):
        # Phase 2: real persist
        persist_result = await asyncio.get_running_loop().run_in_executor(None, _run, False, False)
        if persist_result.get("ok"):
            parsed = persist_result.get("parsed") if isinstance(persist_result.get("parsed"), dict) else {}
            amount_out = parsed.get("amount") or persist_result.get("amount") or amount_hint
            merchant_out = parsed.get("merchant") or persist_result.get("merchant") or merchant_hint or "expense"
            category = parsed.get("category") or persist_result.get("category") or ""
            date = parsed.get("date") or persist_result.get("date") or ""
            parts = [f"Logged ✅ {merchant_out!s} ₹{amount_out!s}"]
            if category and category != "other":
                parts.append(f"· {category}")
            if date:
                parts.append(f"· {date}")
            return " ".join(parts)
        return None  # Persist failed → fall through to agent

    # NLP dry-run failed — try structured retry
    reason = dry_result.get("error") or ""
    if any(k.lower() in reason.lower() for k in ["amount is required", "merchant is required", "could not detect"]):
        retry_dry = await asyncio.get_running_loop().run_in_executor(None, _run, True, True)
        if retry_dry.get("ok"):
            retry_persist = await asyncio.get_running_loop().run_in_executor(None, _run, True, False)
            if retry_persist.get("ok"):
                amount_out = retry_persist.get("amount") or amount_hint
                merchant_out = retry_persist.get("merchant") or merchant_hint or "expense"
                return f"Logged ✅ {merchant_out!s} ₹{amount_out!s}"

    return None  # Can't handle → fall through to agent
