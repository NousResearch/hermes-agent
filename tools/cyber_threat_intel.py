"""Cybersecurity Threat Intelligence Tool.

Provides three capabilities in a single tool:

  cve_lookup    — Query the NVD API v2 for CVE details (no API key required).
  ioc_reputation— Check IP/domain/hash reputation via AbuseIPDB or VirusTotal
                  (optional env vars: ABUSEIPDB_API_KEY, VT_API_KEY).
  mitre_ttp     — Fetch MITRE ATT&CK technique details from the TAXII 2.1 feed.

Design notes:
- All network calls time out in 10 s so the agent is never stuck.
- Missing API keys degrade gracefully: the tool returns what it can and
  notes which lookups were skipped.
- No credentials are ever echoed back in tool results.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds for every outbound request

# ---------------------------------------------------------------------------
# NVD CVE lookup (free, unauthenticated, rate-limited at ~5 req/30 s)
# ---------------------------------------------------------------------------

_NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)


def _nvd_cve(cve_id: str) -> dict:
    cve_id = cve_id.strip().upper()
    if not _CVE_RE.match(cve_id):
        return {"error": f"Invalid CVE ID format: {cve_id!r}"}
    url = f"{_NVD_API}?cveId={urllib.parse.quote(cve_id)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-agentcyber/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"NVD API HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:
        return {"error": f"NVD request failed: {exc}"}

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"error": f"CVE {cve_id} not found in NVD"}

    cve = vulns[0]["cve"]
    descriptions = {
        d["lang"]: d["value"]
        for d in cve.get("descriptions", [])
    }

    # Prefer CVSS v3.1, fall back to v3.0, then v2
    metrics = cve.get("metrics", {})
    cvss = None
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            c = entries[0].get("cvssData", {})
            cvss = {
                "version": c.get("version"),
                "score": c.get("baseScore"),
                "severity": c.get("baseSeverity") or entries[0].get("baseSeverity"),
                "vector": c.get("vectorString"),
            }
            break

    refs = [r.get("url") for r in cve.get("references", [])[:5]]

    return {
        "cve_id": cve_id,
        "published": cve.get("published", "")[:10],
        "last_modified": cve.get("lastModified", "")[:10],
        "description": descriptions.get("en", descriptions.get(next(iter(descriptions), ""), "")),
        "cvss": cvss,
        "cwe": [
            w.get("description", [{}])[0].get("value")
            for w in cve.get("weaknesses", [])
            if w.get("description")
        ][:3],
        "references": refs,
        "status": cve.get("vulnStatus"),
    }


# ---------------------------------------------------------------------------
# IOC reputation checks
# ---------------------------------------------------------------------------

def _vt_reputation(indicator: str, kind: str, api_key: str) -> dict:
    kind_map = {"ip": "ip_addresses", "domain": "domains", "hash": "files", "url": "urls"}
    resource = kind_map.get(kind)
    if not resource:
        return {"error": f"Unknown IOC type for VirusTotal: {kind!r}"}
    encoded = urllib.parse.quote(indicator, safe="")
    url = f"https://www.virustotal.com/api/v3/{resource}/{encoded}"
    req = urllib.request.Request(url, headers={"x-apikey": api_key, "User-Agent": "hermes-agentcyber/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"VirusTotal HTTP {exc.code}"}
    except Exception as exc:
        return {"error": str(exc)}

    attrs = data.get("data", {}).get("attributes", {})
    last = attrs.get("last_analysis_stats", {})
    return {
        "source": "VirusTotal",
        "indicator": indicator,
        "type": kind,
        "malicious": last.get("malicious", 0),
        "suspicious": last.get("suspicious", 0),
        "harmless": last.get("harmless", 0),
        "undetected": last.get("undetected", 0),
        "reputation": attrs.get("reputation"),
        "community_score": attrs.get("total_votes", {}),
    }


def _abuseipdb_reputation(ip: str, api_key: str) -> dict:
    url = (
        "https://api.abuseipdb.com/api/v2/check?"
        + urllib.parse.urlencode({"ipAddress": ip, "maxAgeInDays": 90, "verbose": ""})
    )
    req = urllib.request.Request(url, headers={"Key": api_key, "Accept": "application/json", "User-Agent": "hermes-agentcyber/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"AbuseIPDB HTTP {exc.code}"}
    except Exception as exc:
        return {"error": str(exc)}

    d = data.get("data", {})
    return {
        "source": "AbuseIPDB",
        "indicator": ip,
        "type": "ip",
        "abuse_confidence_score": d.get("abuseConfidenceScore"),
        "total_reports": d.get("totalReports"),
        "last_reported": d.get("lastReportedAt"),
        "country": d.get("countryCode"),
        "isp": d.get("isp"),
        "domain": d.get("domain"),
        "is_tor": d.get("isTor"),
        "usage_type": d.get("usageType"),
    }


def _ioc_reputation(indicator: str, ioc_type: str) -> dict:
    vt_key = os.environ.get("VT_API_KEY", "")
    abuse_key = os.environ.get("ABUSEIPDB_API_KEY", "")

    results: dict[str, Any] = {"indicator": indicator, "type": ioc_type, "lookups": []}

    if ioc_type == "ip" and abuse_key:
        results["lookups"].append(_abuseipdb_reputation(indicator, abuse_key))

    if vt_key:
        results["lookups"].append(_vt_reputation(indicator, ioc_type, vt_key))

    if not results["lookups"]:
        results["note"] = (
            "No API keys configured. Set VT_API_KEY (VirusTotal) and/or "
            "ABUSEIPDB_API_KEY (AbuseIPDB) environment variables to enable reputation lookups."
        )
    return results


# ---------------------------------------------------------------------------
# MITRE ATT&CK TAXII 2.1 lookup (free, unauthenticated)
# ---------------------------------------------------------------------------

_MITRE_TAXII_COLLECTION = (
    "https://attack-taxii.mitre.org/api/v21/collections/"
    "x-mitre-collection--1f5ab827-a5f5-420e-b549-ae43ce62ae7a/objects/"
)
_TECH_RE = re.compile(r"^T\d{4}(\.\d{3})?$", re.IGNORECASE)


def _mitre_ttp(technique_id: str) -> dict:
    technique_id = technique_id.strip().upper()
    if not _TECH_RE.match(technique_id):
        return {"error": f"Invalid technique ID format: {technique_id!r} — expected T####[.###]"}

    # Build STIX filter: type=attack-pattern and name/external_id match
    url = (
        _MITRE_TAXII_COLLECTION
        + "?"
        + urllib.parse.urlencode({
            "match[type]": "attack-pattern",
            "match[external_references.external_id]": technique_id,
        })
    )
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/taxii+json;version=2.1", "User-Agent": "hermes-agentcyber/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"MITRE TAXII HTTP {exc.code}"}
    except Exception as exc:
        return {"error": f"MITRE TAXII request failed: {exc}"}

    objects = data.get("objects", [])
    if not objects:
        return {"error": f"Technique {technique_id} not found in ATT&CK TAXII feed"}

    obj = objects[0]
    ext_refs = obj.get("external_references", [])
    mitre_ref = next((r for r in ext_refs if r.get("source_name") == "mitre-attack"), {})
    kill_chain = [
        p.get("phase_name")
        for p in obj.get("kill_chain_phases", [])
        if p.get("kill_chain_name") == "mitre-attack"
    ]
    platforms = obj.get("x_mitre_platforms", [])
    data_sources = obj.get("x_mitre_data_sources", [])

    return {
        "technique_id": mitre_ref.get("external_id", technique_id),
        "name": obj.get("name"),
        "description": (obj.get("description") or "")[:800],
        "tactics": kill_chain,
        "platforms": platforms,
        "data_sources": data_sources[:8],
        "url": mitre_ref.get("url"),
        "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
        "detection": (obj.get("x_mitre_detection") or "")[:400],
    }


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _handle(args: dict, **_kw: Any) -> str:
    action = args.get("action", "")
    if action == "cve_lookup":
        result = _nvd_cve(args.get("cve_id", ""))
    elif action == "ioc_reputation":
        result = _ioc_reputation(
            args.get("indicator", ""),
            args.get("ioc_type", "ip"),
        )
    elif action == "mitre_ttp":
        result = _mitre_ttp(args.get("technique_id", ""))
    else:
        result = {"error": f"Unknown action: {action!r}. Use cve_lookup, ioc_reputation, or mitre_ttp."}
    return json.dumps(result, indent=2)


SCHEMA = {
    "type": "function",
    "function": {
        "name": "threat_intel",
        "description": (
            "Threat intelligence lookups for cybersecurity operations. "
            "Supports CVE details from NVD, IOC reputation from VirusTotal/AbuseIPDB, "
            "and MITRE ATT&CK technique lookup via TAXII 2.1."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["cve_lookup", "ioc_reputation", "mitre_ttp"],
                    "description": "Which lookup to perform.",
                },
                "cve_id": {
                    "type": "string",
                    "description": "CVE identifier e.g. CVE-2024-12345 (required for cve_lookup).",
                },
                "indicator": {
                    "type": "string",
                    "description": "IP address, domain, file hash, or URL to check (required for ioc_reputation).",
                },
                "ioc_type": {
                    "type": "string",
                    "enum": ["ip", "domain", "hash", "url"],
                    "description": "Indicator type (required for ioc_reputation).",
                },
                "technique_id": {
                    "type": "string",
                    "description": "ATT&CK technique ID e.g. T1059 or T1059.001 (required for mitre_ttp).",
                },
            },
            "required": ["action"],
        },
    },
}

# ---------------------------------------------------------------------------
# Registry self-registration
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402

registry.register(
    name="threat_intel",
    toolset="cyber",
    schema=SCHEMA,
    handler=_handle,
    emoji="🔍",
)
