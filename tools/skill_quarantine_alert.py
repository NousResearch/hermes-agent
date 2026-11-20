"""Escalation path for skill-load scanning findings (P1 addendum §6).

The quarantine gate itself fails closed; this module is the alert/card side
and must fail open — an escalation error may never break the load hook
(same semantics injection_shield uses for its alerting).

Mechanics (addendum §6.2):
- Trigger file instead of direct kanban_create from plugin/gate code — the
  kanban kernel picks the file up and creates the audit card.
- The trigger file is a JSON **array** of pending events. The writer
  accumulates (never overwrites): a second, different content hash must not
  clobber an event the kernel has not consumed yet. Consumption contract for
  the kernel reader: create one audit card per pending event (deduped by
  ``content_hash`` within the batch), then rewrite the file to the events it
  did NOT consume — or remove it when empty. The writer merges with whatever
  the consumer left in place, so both sides race safely on the same file.
- Dedupe by content hash: re-loads of the same violating payload never ping
  again; the ledger only counts repeats.
- Findings are reduced to ``(file, line, rule, severity)`` — payload quotes
  never enter the trigger file, so the card the kernel creates passes the
  tool-call pattern check in the P0 families without further scrubbing.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

TRIGGER_FILE_NAME = ".skill-quarantine-trigger"
LEDGER_FILE_NAME = TRIGGER_FILE_NAME + ".ledger.json"


def _knowledge_dir() -> Path | None:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "knowledge"
    except Exception:
        return None


def notify_skill_finding(tier: str, skill_md: Path, provenance: dict, enforce: bool) -> None:
    """Record one scan finding for the audit-card pipeline. Fail-open all the
    way down: any error here degrades to a debug log, never an exception."""
    try:
        card = os.environ.get("HERMES_SKILL_QUARANTINE_CARDS", "").strip().lower() in ("1", "true", "yes")
        verdict = str(provenance.get("verdict") or "unknown")
        content_hash = str(provenance.get("bundle_hash") or "")
        findings = [{"file": str(f.get("file", "")), "line": f.get("line", 0),
                     "rule": str(f.get("pattern_id", "")), "severity": str(f.get("severity", ""))}
                    for f in provenance.get("findings") or [] if isinstance(f, dict)]
        if not (card and verdict in ("caution", "dangerous") and content_hash):
            return
        now = datetime.now(UTC).isoformat()
        event = {
            "skill": Path(skill_md).parent.name,
            "tier": tier,
            "verdict": verdict,
            "enforcement": "on" if enforce else "audit-only",
            "content_hash": content_hash,
            "first_seen": now,
            "count": 1,
            "findings": findings,
        }
        kdir = _knowledge_dir()
        if kdir is None:
            return
        kdir.mkdir(parents=True, exist_ok=True)
        ledger_path = kdir / LEDGER_FILE_NAME
        trigger_path = kdir / TRIGGER_FILE_NAME
        ledger: dict = {}
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if not isinstance(ledger, dict):
                ledger = {}
        except Exception:
            ledger = {}
        prior = ledger.get(content_hash) or {}
        if prior:
            event["first_seen"] = prior.get("first_seen") or now
            event["count"] = int(prior.get("count") or 0) + 1
            # Same content hash: the card pipeline already has (or dead-lettered)
            # this payload — counting is the only fresh information.
        else:
            _queue_trigger_event(trigger_path, event)
            logger.warning("skill finding triggered audit card: tier=%s verdict=%s skill=%s (see %s)",
                           tier, verdict, event["skill"], trigger_path)
        ledger[content_hash] = {"first_seen": event["first_seen"], "count": event["count"]}
        _atomic_write_text(ledger_path, json.dumps(ledger, indent=2) + "\n")
        if prior:
            logger.info("skill finding repeat (content_hash=%s, count=%s) — no new card", content_hash, event["count"])
    except Exception:
        logger.debug("skill quarantine escalation unavailable — failing open", exc_info=True)


def _queue_trigger_event(trigger_path: Path, event: dict) -> None:
    """Append one new event to the pending-events array, merging with whatever
    the kernel consumer left in the file. A same-content_hash entry that is
    still pending is dropped here: it will be carded by the earlier entry."""
    try:
        pending = json.loads(trigger_path.read_text(encoding="utf-8"))
        if not isinstance(pending, list):
            pending = []
    except Exception:
        pending = []
    new_hash = str(event.get("content_hash") or "")
    new_event_dirname = str(event.get("skill") or "")
    if not any(str(e.get("content_hash")) == new_hash and str(e.get("skill")) == new_event_dirname
               for e in pending if isinstance(e, dict)):
        pending.append(event)
    _atomic_write_text(trigger_path, json.dumps(pending, indent=2) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
