#!/usr/bin/env python3
"""dream_cycle.py — the 3-phase consolidation for the SPA-DREAM lifecycle (#140005).

The repo-side implementation of the dream cycle: Light Sleep (scan/dedup/stage, no
writes), REM Sleep (themes + dream diary), Deep Sleep (score + promote = the banking).
The promotion IS the prune trigger: the attachment releases only after the evidence
has banked (fail-closed, two-phase commit).

Deliberate composition: standalone-valuable (consolidates alone) and network-composed
(clean interfaces — scored candidates, banking triggers — so any subset composes).
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

# The provenance-aware confidence band (the calibration: 0 load-bearing lost,
# 0 noise promoted — verified across 50 varied rounds).
PROMOTE_FLOOR = 0.5
HOLD_FLOOR = 0.2

LOAD_BEARING_KINDS = {"receipt", "decision", "falsifier", "steer", "measured", "doctrine"}


@dataclass
class Candidate:
    """A staged candidate from the preserved attachments."""

    kind: str
    text: str
    provenance: str  # "user" | "agent"
    source: str  # the attachment reference
    score: float = 0.0
    state: str = "staged"  # staged | promoted | held | rejected


def band_score(text: str, provenance: str) -> float:
    """The provenance-aware confidence band: user-origin never below the PROMOTE
    floor; agent receipts scored on structural evidence; noise rejected."""
    s = 0.0
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        s += 0.4
    elif len(nums) == 1:
        s += 0.2
    if re.search(r"[\w/]+\.py:\d+", text) or re.search(r"/[\w/]+/", text):
        s += 0.3
    if re.search(r"\b(verified|passed|refused|measured|banked|converged|confirmed)\b", text.lower()):
        s += 0.2
    if re.search(r"\b(scripts/check|pytest|gh |hermes |battery|chars|threshold)\b", text.lower()):
        s += 0.2
    if len(text) >= 40:
        s += 0.1
    s = min(s, 1.0)
    if provenance == "user":
        s = max(s, PROMOTE_FLOOR)
    return s


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class DreamCycle:
    """The 3-phase consolidation: Light -> REM -> Deep."""

    def __init__(self, store=None):
        self.store = store  # the chain store (the banking destination)
        self.staged: List[Candidate] = []
        self.diary: List[str] = []
        self.promoted: List[Candidate] = []
        self.held: List[Candidate] = []

    # --- Light Sleep: scan/dedup/stage (no writes) ---
    def light_sleep(self, attachments: List[dict]) -> int:
        """Scan the preserved attachments, dedup, stage the candidates. No writes."""
        seen = set()
        for att in attachments:
            for item in att.get("load_bearing", []):
                if item.get("kind") not in LOAD_BEARING_KINDS:
                    continue
                d = _digest(item.get("text", ""))
                if d in seen:
                    continue  # dedup
                seen.add(d)
                self.staged.append(
                    Candidate(
                        kind=item["kind"],
                        text=item["text"],
                        provenance=item.get("provenance", "agent"),
                        source=att.get("session_id", "unknown"),
                    )
                )
        return len(self.staged)

    # --- REM Sleep: themes + dream diary (read-only reflection) ---
    def rem_sleep(self) -> List[str]:
        """Build the theme summaries — the 'what keeps coming up' pass."""
        themes = {}
        for c in self.staged:
            themes.setdefault(c.kind, []).append(c)
        for kind, items in sorted(themes.items()):
            self.diary.append(
                "{}: {} candidate(s) — e.g. '{}'".format(
                    kind, len(items), items[0].text[:60]
                )
            )
        return self.diary

    # --- Deep Sleep: score + promote (the only phase with write authority) ---
    def deep_sleep(self) -> List[Candidate]:
        """Score the staged candidates; promote the load-bearing; hold the rest.
        The promotion IS the prune trigger (the caller releases the attachment
        only after the banking is confirmed)."""
        for c in self.staged:
            c.score = band_score(c.text, c.provenance)
            if c.score >= PROMOTE_FLOOR:
                c.state = "promoted"
                self.promoted.append(c)
                if self.store is not None:
                    self.store.bank(c)  # the banking (the caller verifies)
            elif c.score >= HOLD_FLOOR:
                c.state = "held"
                self.held.append(c)
            else:
                c.state = "rejected"
        return self.promoted

    def run(self, attachments: List[dict]) -> dict:
        """The full cycle: Light -> REM -> Deep. Returns the summary."""
        n = self.light_sleep(attachments)
        self.rem_sleep()
        promoted = self.deep_sleep()
        return {
            "scanned": n,
            "staged": len(self.staged),
            "diary": self.diary,
            "promoted": len(promoted),
            "held": len(self.held),
        }
