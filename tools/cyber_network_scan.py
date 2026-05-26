"""Hermes AgentCyber — Network Scan Tool.

Wraps nmap to produce structured scan output and optionally correlates
detected service versions against the NVD CVE database.

Actions:
  scan        — Run nmap against one or more targets; returns hosts, ports,
                services, and version banners as structured JSON.
  correlate   — Take service/version data from a previous scan and look up
                matching CVEs via NVD v2.  Feeds cleanly into vuln_triage.
  quick_scan  — scan + correlate in a single call (convenience).
  status      — Show whether nmap is installed and the last scan summary.

Scan profiles:
  fast        nmap -F           Top 100 ports, fastest
  standard    nmap -sV          Top 1000 ports + service/version detection
  full        nmap -sV -p-      All 65535 ports + version detection
  vuln        nmap -sV --script vuln  Version + built-in vuln scripts (needs root)

No API keys required.  CVE lookups use the free NVD v2 API.
nmap must be installed: apt-get install -y nmap
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

logger = logging.getLogger(__name__)

_NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_TIMEOUT = 10       # NVD HTTP timeout (seconds)
_MAX_CVES = 5       # CVEs returned per service


# ---------------------------------------------------------------------------
# nmap runner
# ---------------------------------------------------------------------------

def _nmap_available() -> bool:
    return bool(shutil.which("nmap"))


_PROFILE_FLAGS: dict[str, list[str]] = {
    "fast":     ["-F", "-sV", "--version-intensity", "3"],
    "standard": ["-sV", "--version-intensity", "5"],
    "full":     ["-sV", "-p-", "--version-intensity", "5"],
    "vuln":     ["-sV", "--script", "vuln"],
}


def _run_nmap(
    target: str,
    profile: str = "standard",
    ports: str = "",
    timeout: int = 300,
) -> dict:
    """Run nmap and return parsed results dict."""
    if not _nmap_available():
        return {"error": "nmap not found. Install with: apt-get install -y nmap"}

    flags = list(_PROFILE_FLAGS.get(profile, _PROFILE_FLAGS["standard"]))
    cmd = ["nmap", "-oX", "-"] + flags
    if ports:
        cmd += ["-p", ports]
    cmd += [target]

    logger.info("network_scan: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"nmap timed out after {timeout}s", "target": target}
    except Exception as exc:
        return {"error": str(exc), "target": target}

    if proc.returncode not in (0, 1):   # nmap exits 1 on "host not up"
        return {
            "error": proc.stderr.strip() or f"nmap exited {proc.returncode}",
            "target": target,
        }

    return _parse_nmap_xml(proc.stdout, target, profile)


# ---------------------------------------------------------------------------
# nmap XML parser
# ---------------------------------------------------------------------------

def _parse_nmap_xml(xml_text: str, target: str, profile: str) -> dict:
    """Parse nmap -oX output into a structured dict."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return {"error": f"nmap XML parse error: {exc}", "target": target}

    hosts = []
    for host_el in root.findall("host"):
        status = host_el.find("status")
        if status is None or status.get("state") != "up":
            continue

        # Address (use explicit None checks — ET elements are falsy when childless)
        addr_el = host_el.find("address[@addrtype='ipv4']")
        if addr_el is None:
            addr_el = host_el.find("address[@addrtype='ipv6']")
        ip = addr_el.get("addr", "unknown") if addr_el is not None else "unknown"

        # Hostname
        hn_el = host_el.find("hostnames/hostname[@type='user']")
        if hn_el is None:
            hn_el = host_el.find("hostnames/hostname")
        hostname = hn_el.get("name", "") if hn_el is not None else ""

        # OS guess
        os_match = host_el.find("os/osmatch")
        os_guess = os_match.get("name", "") if os_match is not None else ""
        os_accuracy = int(os_match.get("accuracy", 0)) if os_match is not None else 0

        # Ports
        open_ports: list[dict] = []
        for port_el in host_el.findall("ports/port"):
            state_el = port_el.find("state")
            if state_el is None or state_el.get("state") != "open":
                continue
            svc_el = port_el.find("service")
            port_entry: dict[str, Any] = {
                "port":     int(port_el.get("portid", 0)),
                "protocol": port_el.get("protocol", "tcp"),
                "service":  svc_el.get("name", "") if svc_el is not None else "",
                "product":  svc_el.get("product", "") if svc_el is not None else "",
                "version":  svc_el.get("version", "") if svc_el is not None else "",
                "extra":    svc_el.get("extrainfo", "") if svc_el is not None else "",
                "cpe":      [c.text for c in (port_el.findall("service/cpe") or [])
                             if c.text],
            }
            open_ports.append(port_entry)

        hosts.append({
            "ip":          ip,
            "hostname":    hostname,
            "os":          os_guess,
            "os_accuracy": os_accuracy,
            "open_ports":  open_ports,
            "port_count":  len(open_ports),
        })

    scan_stats = root.find("runstats/hosts")
    return {
        "target":   target,
        "profile":  profile,
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hosts_up":   int(scan_stats.get("up", 0)) if scan_stats is not None else len(hosts),
        "hosts_down": int(scan_stats.get("down", 0)) if scan_stats is not None else 0,
        "hosts":    hosts,
    }


# ---------------------------------------------------------------------------
# CVE correlation via NVD v2
# ---------------------------------------------------------------------------

def _nvd_search(keyword: str) -> list[dict]:
    """Search NVD for CVEs matching a product+version keyword."""
    url = f"{_NVD_BASE}?keywordSearch={urllib.parse.quote(keyword)}&resultsPerPage=5"
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agentcyber/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        logger.warning("NVD lookup failed for %r: %s", keyword, exc)
        return []

    cves = []
    for item in data.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id", "")
        desc_list = cve.get("descriptions", [])
        desc = next((d["value"] for d in desc_list if d.get("lang") == "en"), "")
        metrics = cve.get("metrics", {})
        cvss_score = 0.0
        cvss_severity = ""
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics and metrics[key]:
                m = metrics[key][0].get("cvssData", {})
                cvss_score = float(m.get("baseScore", 0))
                cvss_severity = m.get("baseSeverity", "")
                break
        cves.append({
            "cve_id":        cve_id,
            "cvss_score":    cvss_score,
            "cvss_severity": cvss_severity,
            "description":   desc[:300],
        })

    cves.sort(key=lambda c: c["cvss_score"], reverse=True)
    return cves[:_MAX_CVES]


def _correlate_scan(scan_result: dict) -> dict:
    """Enrich a scan result dict with CVE lookups per service/version."""
    findings: list[dict] = []

    for host in scan_result.get("hosts", []):
        for port in host.get("open_ports", []):
            product = port.get("product", "")
            version = port.get("version", "")
            if not product:
                continue
            keyword = f"{product} {version}".strip()
            cves = _nvd_search(keyword)
            if not cves:
                continue
            top_cvss = cves[0]["cvss_score"] if cves else 0.0
            findings.append({
                "host":      host["ip"],
                "hostname":  host.get("hostname", ""),
                "port":      port["port"],
                "protocol":  port["protocol"],
                "service":   port["service"],
                "product":   keyword,
                "cves":      cves,
                "top_cvss":  top_cvss,
                "risk":      _risk_label(top_cvss),
            })
            time.sleep(0.15)   # NVD rate-limit courtesy delay

    findings.sort(key=lambda f: f["top_cvss"], reverse=True)
    return {
        "target":         scan_result.get("target"),
        "scanned_at":     scan_result.get("scanned_at"),
        "hosts_up":       scan_result.get("hosts_up", 0),
        "findings_count": len(findings),
        "findings":       findings,
        "summary": _make_summary(findings),
    }


def _risk_label(cvss: float) -> str:
    if cvss >= 9.0:   return "CRITICAL"
    if cvss >= 7.0:   return "HIGH"
    if cvss >= 4.0:   return "MEDIUM"
    if cvss > 0:      return "LOW"
    return "INFO"


def _make_summary(findings: list[dict]) -> dict:
    by_risk: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
    for f in findings:
        by_risk[f["risk"]] = by_risk.get(f["risk"], 0) + 1
    critical = [f"{f['host']}:{f['port']} {f['product']} ({f['cves'][0]['cve_id']})"
                for f in findings if f["risk"] == "CRITICAL"]
    return {
        "by_risk":       by_risk,
        "critical_items": critical[:10],
    }


# ---------------------------------------------------------------------------
# Session scan store (per agent instance)
# ---------------------------------------------------------------------------

class _ScanStore:
    def __init__(self) -> None:
        self._scans: list[dict] = []

    def save(self, scan: dict) -> str:
        scan_id = f"scan-{len(self._scans) + 1:03d}"
        self._scans.append({"id": scan_id, **scan})
        return scan_id

    def get(self, scan_id: str) -> dict | None:
        return next((s for s in self._scans if s["id"] == scan_id), None)

    def last(self) -> dict | None:
        return self._scans[-1] if self._scans else None

    def list_summaries(self) -> list[dict]:
        return [
            {
                "id":        s["id"],
                "target":    s.get("target"),
                "scanned_at": s.get("scanned_at"),
                "hosts_up":  s.get("hosts_up", 0),
            }
            for s in self._scans
        ]


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _action_scan(args: dict, store: _ScanStore) -> dict:
    target = args.get("target", "")
    if not target:
        return {"error": "target is required (e.g. '192.168.1.0/24' or 'host.example.com')"}

    profile = args.get("profile", "standard")
    if profile not in _PROFILE_FLAGS:
        return {"error": f"Unknown profile: {profile!r}", "valid": list(_PROFILE_FLAGS)}

    result = _run_nmap(
        target=target,
        profile=profile,
        ports=args.get("ports", ""),
        timeout=int(args.get("timeout", 300)),
    )
    if "error" in result:
        return result

    scan_id = store.save(result)
    result["scan_id"] = scan_id
    result["hint"] = f"Run correlate with scan_id={scan_id!r} to look up CVEs for found services."
    return result


def _action_correlate(args: dict, store: _ScanStore) -> dict:
    scan_id = args.get("scan_id", "")
    if scan_id:
        scan = store.get(scan_id)
        if scan is None:
            return {"error": f"scan_id not found: {scan_id!r}. Use status to list scans."}
    else:
        scan = store.last()
        if scan is None:
            return {"error": "No scan in session. Run scan first."}

    return _correlate_scan(scan)


def _action_quick_scan(args: dict, store: _ScanStore) -> dict:
    scan_result = _action_scan(args, store)
    if "error" in scan_result:
        return scan_result
    corr = _correlate_scan(scan_result)
    corr["scan_id"] = scan_result.get("scan_id")
    return corr


def _action_status(args: dict, store: _ScanStore) -> dict:
    return {
        "nmap_available": _nmap_available(),
        "nmap_path":      shutil.which("nmap") or "not found",
        "running_as_root": os.geteuid() == 0,
        "note_on_root": (
            "SYN scan (-sS), OS detection (-O), and 'vuln' profile require root. "
            "TCP connect scan (-sT) works without root."
        ),
        "valid_profiles": list(_PROFILE_FLAGS),
        "session_scans":  store.list_summaries(),
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _handle(args: dict, **kw: Any) -> str:
    agent = kw.get("agent")
    if agent is not None and not hasattr(agent, "_scan_store"):
        agent._scan_store = _ScanStore()
    store: _ScanStore = getattr(agent, "_scan_store", None) or _ScanStore()

    action = args.get("action", "")
    dispatch = {
        "scan":       _action_scan,
        "correlate":  _action_correlate,
        "quick_scan": _action_quick_scan,
        "status":     _action_status,
    }
    fn = dispatch.get(action)
    if fn is None:
        return json.dumps({
            "error": f"Unknown action: {action!r}",
            "valid_actions": list(dispatch),
        })
    try:
        result = fn(args, store)
    except Exception as exc:
        logger.exception("network_scan tool error")
        result = {"error": str(exc)}
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Schema + registration
# ---------------------------------------------------------------------------

SCHEMA = {
    "type": "function",
    "function": {
        "name": "network_scan",
        "description": (
            "Network reconnaissance: run nmap against a host or CIDR range and "
            "optionally correlate discovered service versions against the NVD CVE "
            "database. Use quick_scan to get hosts + CVEs in one call. Results "
            "feed directly into vuln_triage (pass cve_id) and ir_incident "
            "(pass finding as evidence). nmap must be installed on the host."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["scan", "correlate", "quick_scan", "status"],
                    "description": (
                        "scan — run nmap and return structured results; "
                        "correlate — enrich last (or specified) scan with CVEs; "
                        "quick_scan — scan + correlate in one call; "
                        "status — check nmap availability and session scan list."
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Host, IP, or CIDR range to scan "
                        "(e.g. '192.168.1.1', '10.0.0.0/24', 'example.com'). "
                        "Required for scan and quick_scan."
                    ),
                },
                "profile": {
                    "type": "string",
                    "enum": ["fast", "standard", "full", "vuln"],
                    "description": (
                        "fast — top 100 ports (quickest); "
                        "standard — top 1000 ports + version detection (default); "
                        "full — all 65535 ports + version detection (slow); "
                        "vuln — version detection + nmap vuln scripts (requires root)."
                    ),
                },
                "ports": {
                    "type": "string",
                    "description": (
                        "Override port list, e.g. '80,443,8080' or '1-1024'. "
                        "If omitted, the profile's default port range is used."
                    ),
                },
                "scan_id": {
                    "type": "string",
                    "description": (
                        "Reference a specific previous scan for the correlate action "
                        "(e.g. 'scan-001'). If omitted, the most recent scan is used."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "nmap timeout in seconds (default: 300).",
                },
            },
            "required": ["action"],
        },
    },
}

from tools.registry import registry  # noqa: E402

registry.register(
    name="network_scan",
    toolset="cyber",
    schema=SCHEMA,
    handler=_handle,
    emoji="📡",
)
