"""Cybersecurity IOC Extractor Tool.

Extracts Indicators of Compromise from freeform text (logs, emails, reports,
alert payloads). Returns structured JSON with deduplicated, type-labelled IOCs
ready for further enrichment via threat_intel or vuln_triage.

IOC types extracted:
  ipv4          — public routable IPv4 addresses (RFC 1918 + loopback excluded)
  ipv6          — abbreviated and full IPv6 addresses
  domain        — FQDNs (TLD-validated against a short common list)
  url           — http/https/ftp URLs (defanged variants handled)
  md5           — 32-hex-char file hashes
  sha1          — 40-hex-char file hashes
  sha256        — 64-hex-char file hashes
  cve           — CVE identifiers
  email         — email addresses (useful for phishing/BEC cases)

Defanging: common analyst obfuscation patterns are normalised before
extraction (hxxp→http, [.]→., [at]→@) so pasted threat reports are
handled without preprocessing.
"""

from __future__ import annotations

import ipaddress
import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Defanging normalisation
# ---------------------------------------------------------------------------

_DEFANG_SUBS = [
    (re.compile(r"hxxps?://", re.IGNORECASE), "https://"),
    (re.compile(r"\[\.?\]|\(\.\)", re.IGNORECASE), "."),
    (re.compile(r"\[at\]", re.IGNORECASE), "@"),
    (re.compile(r"\[\+\]", re.IGNORECASE), "."),
]


def _defang(text: str) -> str:
    for pattern, replacement in _DEFANG_SUBS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# IPv4 — captured as dotted-quad, filtered for routability below
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# IPv6 — simplified; catches full and abbreviated forms
_IPV6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"
    r"|\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b"
    r"|\b[0-9a-fA-F]{1,4}::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b"
)

# URLs — http/https/ftp
_URL_RE = re.compile(
    r"https?://[^\s\"'>)\]]{8,256}|ftp://[^\s\"'>)\]]{8,128}",
    re.IGNORECASE,
)

# Domains — simple FQDN heuristic (avoids matching plain words)
_DOMAIN_RE = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)"
    r"+(?:com|net|org|io|gov|edu|mil|co|uk|de|ru|cn|jp|br|fr|au|nl|se|no|fi|"
    r"info|biz|xyz|online|site|live|space|store|tech|app|dev|cloud|security|"
    r"bank|pay|login|verify|update|secure|support)\b",
    re.IGNORECASE,
)

# File hashes — order matters (sha256 > sha1 > md5 to avoid prefix collisions)
_SHA256_RE = re.compile(r"\b[0-9a-fA-F]{64}\b")
_SHA1_RE   = re.compile(r"\b[0-9a-fA-F]{40}\b")
_MD5_RE    = re.compile(r"\b[0-9a-fA-F]{32}\b")

# CVE
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)

# Email
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")

# ---------------------------------------------------------------------------
# Private/reserved IPv4 filter
# ---------------------------------------------------------------------------

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
]


def _is_public_ipv4(addr: str) -> bool:
    try:
        ip = ipaddress.IPv4Address(addr)
        return not any(ip in net for net in _PRIVATE_NETS)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_iocs(text: str, include_private_ips: bool = False) -> dict:
    normalised = _defang(text)

    # --- URLs first (to avoid double-counting domains embedded in URLs) ---
    urls = list(dict.fromkeys(_URL_RE.findall(normalised)))

    # Strip URL matches from text before domain/IP scanning to avoid overlap
    stripped = _URL_RE.sub(" ", normalised)

    # --- Hashes (longest first to avoid prefix false positives) ---
    sha256 = list(dict.fromkeys(_SHA256_RE.findall(stripped)))
    # Remove sha256 matches before looking for sha1/md5 to avoid prefix collision
    hash_stripped = stripped
    for h in sha256:
        hash_stripped = hash_stripped.replace(h, " " * len(h))
    sha1 = list(dict.fromkeys(_SHA1_RE.findall(hash_stripped)))
    for h in sha1:
        hash_stripped = hash_stripped.replace(h, " " * len(h))
    md5 = list(dict.fromkeys(_MD5_RE.findall(hash_stripped)))

    # --- IPs ---
    raw_ipv4 = _IPV4_RE.findall(stripped)
    if include_private_ips:
        ipv4 = list(dict.fromkeys(raw_ipv4))
    else:
        ipv4 = list(dict.fromkeys(ip for ip in raw_ipv4 if _is_public_ipv4(ip)))
    ipv6 = list(dict.fromkeys(_IPV6_RE.findall(stripped)))

    # --- Domains (exclude bare hostnames that are already captured as IPs) ---
    raw_domains = _DOMAIN_RE.findall(stripped)
    # Remove anything that's already in ipv4 or is a pure numeric quad
    domains = list(dict.fromkeys(
        d for d in raw_domains
        if d not in ipv4 and not _IPV4_RE.fullmatch(d)
    ))

    # --- CVEs and email ---
    cves   = list(dict.fromkeys(m.upper() for m in _CVE_RE.findall(normalised)))
    emails = list(dict.fromkeys(_EMAIL_RE.findall(normalised)))

    total = sum(len(v) for v in [urls, ipv4, ipv6, domains, sha256, sha1, md5, cves, emails])

    return {
        "total": total,
        "iocs": {
            "url":    urls,
            "ipv4":   ipv4,
            "ipv6":   ipv6,
            "domain": domains,
            "sha256": sha256,
            "sha1":   sha1,
            "md5":    md5,
            "cve":    cves,
            "email":  emails,
        },
        "note": (
            "Private/RFC-1918 addresses excluded by default. "
            "Pass include_private_ips=true to include them."
        ) if not include_private_ips and any(_PRIVATE_NETS) else None,
    }


# ---------------------------------------------------------------------------
# Tool handler + schema
# ---------------------------------------------------------------------------

def _handle(args: dict, **_kw: Any) -> str:
    text = args.get("text", "")
    if not text:
        return json.dumps({"error": "text parameter is required"})
    include_private = bool(args.get("include_private_ips", False))
    result = _extract_iocs(text, include_private_ips=include_private)
    return json.dumps(result, indent=2)


SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_iocs",
        "description": (
            "Extract Indicators of Compromise (IOCs) from freeform text such as "
            "log entries, threat reports, email bodies, or alert payloads. "
            "Returns deduplicated, type-labelled IOCs: IPv4/IPv6, domains, URLs, "
            "MD5/SHA1/SHA256 hashes, CVE IDs, and email addresses. "
            "Handles analyst defanging (hxxp://, [.], [at]) automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The raw text to extract IOCs from (log lines, report, email body, etc.).",
                },
                "include_private_ips": {
                    "type": "boolean",
                    "description": "Include RFC-1918 private IP addresses (default false).",
                },
            },
            "required": ["text"],
        },
    },
}

from tools.registry import registry  # noqa: E402

registry.register(
    name="extract_iocs",
    toolset="cyber",
    schema=SCHEMA,
    handler=_handle,
    emoji="🔎",
)
