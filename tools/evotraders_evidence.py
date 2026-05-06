#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence helpers for EvoTraders analysis flow.

This module intentionally keeps logic lightweight and defensive:
- Accepts tool payloads as dict/list/JSON-string.
- Produces a normalized evidence envelope for downstream policy checks.
- Provides a minimal merge/check API for future evidence gate rollout.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List
import re


def _to_obj(payload: Any) -> Any:
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return {"raw": payload}
    return {"raw": str(payload)}


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for it in node:
            yield from _walk(it)


def _pick_first_numeric(snapshot: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in snapshot:
            return snapshot.get(k)
    return None


def _extract_codes_from_value(v: Any) -> List[str]:
    codes: List[str] = []
    if isinstance(v, int):
        s = str(v)
        if re.fullmatch(r"\d{6}", s):
            codes.append(s)
        return codes
    if isinstance(v, str):
        for m in re.findall(r"\b\d{6}\b", v):
            codes.append(m)
        return codes
    if isinstance(v, list):
        for it in v:
            codes.extend(_extract_codes_from_value(it))
    elif isinstance(v, dict):
        for vv in v.values():
            codes.extend(_extract_codes_from_value(vv))
    return codes


def build_evidence_envelope(tool_name: str, payload: Any) -> Dict[str, Any]:
    """Build a normalized evidence envelope from one tool result."""
    obj = _to_obj(payload)
    now_ms = int(time.time() * 1000)

    envelope: Dict[str, Any] = {
        "tool": str(tool_name or ""),
        "captured_at_ms": now_ms,
        "snapshot": {},
        "has_snapshot": False,
        "has_structure": False,
        "has_timepoint": False,
        "structure_signals": [],
        "timepoint": "",
        "codes": [],
        "route_trace": {},
    }
    code_set = set()

    # Route trace (from evotraders_route_and_call response)
    if isinstance(obj, dict):
        for k in (
            "matched_rule_id",
            "selected_tool",
            "fallback_chain",
            "route_version",
            "route_latency_ms",
            "route_flags",
        ):
            if k in obj:
                envelope["route_trace"][k] = obj.get(k)

    # Heuristic scan across nested dict/list payloads.
    for d in _walk(obj):
        keys = set(d.keys())

        # Snapshot-like keys.
        if keys.intersection(
            {
                "latest_price",
                "last_price",
                "price",
                "close",
                "change_pct",
                "pct_chg",
                "turnover",
                "turnover_rate",
                "volume",
                "amount",
            }
        ):
            if not envelope["has_snapshot"]:
                snapshot = {
                    "price": _pick_first_numeric(d, ["latest_price", "last_price", "price", "close"]),
                    "change_pct": _pick_first_numeric(d, ["change_pct", "pct_chg", "chg_pct"]),
                    "turnover": _pick_first_numeric(d, ["turnover", "amount"]),
                    "volume": _pick_first_numeric(d, ["volume", "vol"]),
                    "turnover_rate": _pick_first_numeric(d, ["turnover_rate", "换手率"]),
                }
                envelope["snapshot"] = snapshot
            envelope["has_snapshot"] = True

        # Structure signals (fund flow / lianban / notice/news etc).
        if keys.intersection(
            {
                "net_inflow",
                "fund_flow",
                "big_order",
                "limit_up",
                "lianban",
                "announcement",
                "news",
                "financials",
                "shareholder",
            }
        ):
            envelope["has_structure"] = True
            for sk in (
                "net_inflow",
                "fund_flow",
                "big_order",
                "limit_up",
                "lianban",
                "announcement",
                "news",
                "financials",
                "shareholder",
            ):
                if sk in d and sk not in envelope["structure_signals"]:
                    envelope["structure_signals"].append(sk)

        # Timepoint hints.
        for tk in ("trade_date", "timestamp", "ts", "update_time", "datetime"):
            if tk in d and d.get(tk):
                envelope["has_timepoint"] = True
                if not envelope["timepoint"]:
                    envelope["timepoint"] = str(d.get(tk))
                break

        # Stock code hints.
        for ck in ("code", "stock_code", "symbol", "ts_code", "ticker"):
            if ck in d and d.get(ck) is not None:
                for c in _extract_codes_from_value(d.get(ck)):
                    if re.fullmatch(r"\d{6}", c):
                        code_set.add(c)

    # Fallback scan across whole object for 6-digit stock codes.
    for c in _extract_codes_from_value(obj):
        if re.fullmatch(r"\d{6}", c):
            code_set.add(c)
    envelope["codes"] = sorted(code_set)

    return envelope


def merge_evidence(envelopes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple envelopes into one check-friendly summary."""
    merged: Dict[str, Any] = {
        "count": 0,
        "has_snapshot": False,
        "has_structure": False,
        "has_timepoint": False,
        "latest_snapshot": {},
        "structure_signals": [],
        "latest_timepoint": "",
        "codes": [],
    }
    if not isinstance(envelopes, list):
        return merged

    signals = set()
    codes = set()
    latest = None
    for ev in envelopes:
        if not isinstance(ev, dict):
            continue
        merged["count"] += 1
        if ev.get("has_snapshot"):
            merged["has_snapshot"] = True
            latest = ev.get("snapshot", {}) or latest
        if ev.get("has_structure"):
            merged["has_structure"] = True
        if ev.get("has_timepoint"):
            merged["has_timepoint"] = True
            if ev.get("timepoint"):
                merged["latest_timepoint"] = str(ev.get("timepoint"))
        for s in ev.get("structure_signals", []) or []:
            signals.add(str(s))
        for c in ev.get("codes", []) or []:
            c = str(c).strip()
            if re.fullmatch(r"\d{6}", c):
                codes.add(c)
    merged["structure_signals"] = sorted(signals)
    merged["codes"] = sorted(codes)
    merged["latest_snapshot"] = latest or {}
    return merged


def check_min_evidence(
    evidence: Dict[str, Any],
    intent: str = "stock_analysis",
    required_codes: List[str] | None = None,
) -> Dict[str, Any]:
    """Check minimum evidence requirements for stock analysis."""
    missing: List[str] = []
    if intent == "stock_analysis":
        if not evidence.get("has_snapshot"):
            missing.append("snapshot")
        if not evidence.get("has_structure"):
            missing.append("structure")
        if not evidence.get("has_timepoint"):
            missing.append("timepoint")
        if required_codes:
            got = set(str(c).strip() for c in (evidence.get("codes") or []))
            need = [c for c in required_codes if c and c not in got]
            if need:
                missing.append("code:" + ",".join(sorted(need)))
    return {"ok": len(missing) == 0, "missing": missing}

