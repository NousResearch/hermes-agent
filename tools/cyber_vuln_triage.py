"""Cybersecurity Vulnerability Triage Tool.

Correlates three data sources to produce a prioritised remediation
recommendation for a given CVE:

  1. NVD API v2        — CVSS base score and severity
  2. EPSS API          — Exploit Prediction Scoring System probability
                         (probability a CVE will be exploited in the wild
                          within 30 days; sourced from first.org, free)
  3. Asset matching    — caller-supplied asset list checked against CVE CPE
                         data to confirm whether any owned asset is affected

Output includes a priority label (CRITICAL / HIGH / MEDIUM / LOW) derived
from the combined score, plus a short remediation recommendation.

Priority formula (intentionally conservative):
  CRITICAL — CVSS ≥ 9 OR (CVSS ≥ 7 AND EPSS ≥ 0.5)
  HIGH     — CVSS ≥ 7 OR (CVSS ≥ 4 AND EPSS ≥ 0.3)
  MEDIUM   — CVSS ≥ 4
  LOW      — otherwise
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_TIMEOUT = 10
_CVE_RE  = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

_NVD_API  = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_EPSS_API = "https://api.first.org/data/v1/epss"


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _fetch_nvd(cve_id: str) -> dict:
    url = f"{_NVD_API}?cveId={urllib.parse.quote(cve_id)}"
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agentcyber/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return {"error": str(exc)}

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"error": f"{cve_id} not found"}

    cve = vulns[0]["cve"]
    metrics = cve.get("metrics", {})
    cvss_score, cvss_severity, cvss_vector = None, None, None
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            c = entries[0].get("cvssData", {})
            cvss_score    = c.get("baseScore")
            cvss_severity = c.get("baseSeverity") or entries[0].get("baseSeverity")
            cvss_vector   = c.get("vectorString")
            break

    # Extract CPE URIs for asset matching
    cpes: list[str] = []
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if match.get("vulnerable"):
                    cpes.append(match.get("criteria", ""))

    desc = next(
        (d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"),
        "",
    )
    return {
        "cve_id": cve_id,
        "description": desc[:500],
        "published": cve.get("published", "")[:10],
        "cvss_score": cvss_score,
        "cvss_severity": cvss_severity,
        "cvss_vector": cvss_vector,
        "cpes": cpes[:20],
        "status": cve.get("vulnStatus"),
    }


def _fetch_epss(cve_id: str) -> dict:
    url = f"{_EPSS_API}?cve={urllib.parse.quote(cve_id)}"
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agentcyber/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return {"error": str(exc)}

    entries = data.get("data", [])
    if not entries:
        return {"epss": None, "percentile": None, "note": "EPSS score not yet available for this CVE"}
    e = entries[0]
    return {
        "epss": float(e.get("epss", 0)),
        "percentile": float(e.get("percentile", 0)),
        "date": e.get("date"),
    }


# ---------------------------------------------------------------------------
# Asset matching
# ---------------------------------------------------------------------------

def _match_assets(cpes: list[str], assets: list[str]) -> list[str]:
    """Return subset of assets that appear to be affected per CPE strings.

    CPE format: cpe:2.3:a:vendor:product:version:...
    We do fuzzy keyword matching — good enough for triage, not a replacement
    for a real vulnerability scanner.
    """
    matches: list[str] = []
    for asset in assets:
        asset_lower = asset.lower()
        for cpe in cpes:
            parts = cpe.split(":")
            if len(parts) < 5:
                continue
            vendor  = parts[3].lower().replace("_", " ")
            product = parts[4].lower().replace("_", " ")
            if vendor in asset_lower or product in asset_lower:
                matches.append(asset)
                break
    return matches


# ---------------------------------------------------------------------------
# Priority calculation
# ---------------------------------------------------------------------------

def _priority(cvss: float | None, epss: float | None) -> str:
    if cvss is None:
        return "UNKNOWN"
    if cvss >= 9.0 or (cvss >= 7.0 and epss is not None and epss >= 0.5):
        return "CRITICAL"
    if cvss >= 7.0 or (cvss >= 4.0 and epss is not None and epss >= 0.3):
        return "HIGH"
    if cvss >= 4.0:
        return "MEDIUM"
    return "LOW"


def _recommendation(priority: str, affected_assets: list[str], epss: float | None) -> str:
    if priority == "CRITICAL":
        timeframe = "immediately (within 24 hours)"
    elif priority == "HIGH":
        timeframe = "urgently (within 72 hours)"
    elif priority == "MEDIUM":
        timeframe = "within 14 days"
    else:
        timeframe = "within 30 days or next patch cycle"

    asset_note = (
        f"Confirmed affected assets: {', '.join(affected_assets[:5])}."
        if affected_assets
        else "No asset match found — verify scope manually."
    )
    epss_note = (
        f"EPSS exploitation probability: {epss:.1%}." if epss is not None else "EPSS score unavailable."
    )
    return (
        f"Patch or mitigate {timeframe}. "
        f"{asset_note} "
        f"{epss_note}"
    )


# ---------------------------------------------------------------------------
# Main triage function
# ---------------------------------------------------------------------------

def _triage(cve_id: str, assets: list[str]) -> dict:
    cve_id = cve_id.strip().upper()
    if not _CVE_RE.match(cve_id):
        return {"error": f"Invalid CVE ID: {cve_id!r}"}

    nvd  = _fetch_nvd(cve_id)
    epss = _fetch_epss(cve_id)

    if "error" in nvd:
        return {"error": f"NVD lookup failed: {nvd['error']}"}

    cvss_score    = nvd.get("cvss_score")
    epss_score    = epss.get("epss")
    cpes          = nvd.get("cpes", [])
    affected      = _match_assets(cpes, assets) if assets else []
    priority      = _priority(cvss_score, epss_score)
    recommendation = _recommendation(priority, affected, epss_score)

    return {
        "cve_id": cve_id,
        "description": nvd.get("description"),
        "published": nvd.get("published"),
        "status": nvd.get("status"),
        "cvss": {
            "score": cvss_score,
            "severity": nvd.get("cvss_severity"),
            "vector": nvd.get("cvss_vector"),
        },
        "epss": {
            "score": epss_score,
            "percentile": epss.get("percentile"),
            "date": epss.get("date"),
            "note": epss.get("note"),
        },
        "affected_assets": affected,
        "unmatched_assets": [a for a in assets if a not in affected],
        "triage": {
            "priority": priority,
            "recommendation": recommendation,
        },
    }


# ---------------------------------------------------------------------------
# Tool handler + schema
# ---------------------------------------------------------------------------

def _handle(args: dict, **_kw: Any) -> str:
    cve_id = args.get("cve_id", "")
    assets = args.get("assets", [])
    if not cve_id:
        return json.dumps({"error": "cve_id is required"})
    result = _triage(cve_id, assets)
    return json.dumps(result, indent=2)


SCHEMA = {
    "type": "function",
    "function": {
        "name": "vuln_triage",
        "description": (
            "Vulnerability triage: correlates CVSS severity (NVD), EPSS exploitation "
            "probability (first.org), and an optional asset list to produce a "
            "prioritised remediation recommendation (CRITICAL/HIGH/MEDIUM/LOW) "
            "for a given CVE. No API keys required."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "CVE identifier, e.g. CVE-2024-12345.",
                },
                "assets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of asset names/descriptions to check against "
                        "the CVE's affected product list (e.g. ['Apache 2.4 web server', "
                        "'nginx 1.24', 'Windows Server 2022']). Leave empty to skip "
                        "asset correlation."
                    ),
                },
            },
            "required": ["cve_id"],
        },
    },
}

from tools.registry import registry  # noqa: E402

registry.register(
    name="vuln_triage",
    toolset="cyber",
    schema=SCHEMA,
    handler=_handle,
    emoji="🛡️",
)
