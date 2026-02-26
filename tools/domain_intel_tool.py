#!/usr/bin/env python3
"""
Domain Intelligence Tool Module

Passive domain reconnaissance using only public data sources and Python stdlib.
No external API keys required.

Available tools:
- domain_subdomains : Subdomain discovery via crt.sh certificate transparency logs
- domain_ssl        : Live SSL/TLS certificate inspection
- domain_whois      : WHOIS registration data — supports 100+ TLDs
- domain_dns        : DNS record lookup (A, AAAA, MX, NS, TXT, CNAME)
- domain_available  : Check if a domain is available for registration
- domain_bulk       : Run checks on multiple domains in parallel (up to 20)

All tools return structured JSON. Errors are returned as {"error": "..."} — never raised.
"""

import json
import logging
import re
import socket
import ssl
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "hermes-agent/domain-intel (https://github.com/NousResearch/hermes-agent)",
            "Accept": "application/json, text/plain, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _future(dt: datetime) -> bool:
    return dt > datetime.now(timezone.utc)


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _clean_domain(domain: str) -> str:
    return domain.strip().lower().lstrip("*. ").rstrip(". ")


# ---------------------------------------------------------------------------
# WHOIS server map — 100+ TLDs
# ---------------------------------------------------------------------------

WHOIS_SERVERS: Dict[str, str] = {
    # Generic TLDs
    "com":          "whois.verisign-grs.com",
    "net":          "whois.verisign-grs.com",
    "org":          "whois.pir.org",
    "info":         "whois.afilias.net",
    "biz":          "whois.biz",
    "name":         "whois.nic.name",
    "mobi":         "whois.dotmobiregistry.net",
    "tel":          "whois.nic.tel",
    "aero":         "whois.aero",
    "cat":          "whois.cat",
    "coop":         "whois.nic.coop",
    "museum":       "whois.museum",
    # Popular new gTLDs
    "io":           "whois.nic.io",
    "co":           "whois.nic.co",
    "ai":           "whois.nic.ai",
    "dev":          "whois.nic.google",
    "app":          "whois.nic.google",
    "page":         "whois.nic.google",
    "new":          "whois.nic.google",
    "shop":         "whois.nic.shop",
    "store":        "whois.nic.store",
    "online":       "whois.nic.online",
    "site":         "whois.nic.site",
    "tech":         "whois.nic.tech",
    "cloud":        "whois.nic.cloud",
    "digital":      "whois.nic.digital",
    "media":        "whois.nic.media",
    "blog":         "whois.nic.blog",
    "news":         "whois.nic.news",
    "email":        "whois.nic.email",
    "guru":         "whois.nic.guru",
    "ninja":        "whois.nic.ninja",
    "rocks":        "whois.nic.rocks",
    "social":       "whois.nic.social",
    "world":        "whois.nic.world",
    "network":      "whois.nic.network",
    "global":       "whois.nic.global",
    "systems":      "whois.nic.systems",
    "solutions":    "whois.nic.solutions",
    "services":     "whois.nic.services",
    "support":      "whois.nic.support",
    "agency":       "whois.nic.agency",
    "studio":       "whois.nic.studio",
    "design":       "whois.nic.design",
    "art":          "whois.nic.art",
    "music":        "whois.nic.music",
    "game":         "whois.nic.game",
    "games":        "whois.nic.games",
    "live":         "whois.nic.live",
    "tv":           "whois.nic.tv",
    "fm":           "whois.nic.fm",
    "space":        "whois.nic.space",
    "zone":         "whois.nic.zone",
    "link":         "whois.uniregistry.net",
    "click":        "whois.uniregistry.net",
    "host":         "whois.nic.host",
    "domains":      "whois.nic.domains",
    "software":     "whois.nic.software",
    "tools":        "whois.nic.tools",
    "codes":        "whois.nic.codes",
    "tips":         "whois.nic.tips",
    "fund":         "whois.nic.fund",
    "capital":      "whois.nic.capital",
    "finance":      "whois.nic.finance",
    "money":        "whois.nic.money",
    "cash":         "whois.nic.cash",
    "tax":          "whois.nic.tax",
    "law":          "whois.nic.law",
    "legal":        "whois.nic.legal",
    "care":         "whois.nic.care",
    "health":       "whois.nic.health",
    "fit":          "whois.nic.fit",
    "run":          "whois.nic.run",
    "bike":         "whois.nic.bike",
    "eco":          "whois.nic.eco",
    "green":        "whois.nic.green",
    "energy":       "whois.nic.energy",
    "land":         "whois.nic.land",
    "city":         "whois.nic.city",
    "estate":       "whois.nic.estate",
    "properties":   "whois.nic.properties",
    "international":"whois.nic.international",
    "pro":          "whois.registrypro.pro",
    "web":          "whois.nic.web",
    "cc":           "whois.nic.cc",
    "ws":           "whois.website.ws",
    "nu":           "whois.iis.nu",
    "pw":           "whois.nic.pw",
    "ly":           "whois.nic.ly",
    "to":           "whois.tonic.to",
    "mn":           "whois.nic.mn",
    "me":           "whois.nic.me",
    "sh":           "whois.nic.sh",
    "ac":           "whois.nic.ac",
    "gg":           "whois.gg",
    "im":           "whois.nic.im",
    "cx":           "whois.nic.cx",
    # Europe
    "uk":           "whois.nic.uk",
    "co.uk":        "whois.nic.uk",
    "de":           "whois.denic.de",
    "nl":           "whois.domain-registry.nl",
    "fr":           "whois.nic.fr",
    "it":           "whois.nic.it",
    "es":           "whois.nic.es",
    "pl":           "whois.dns.pl",
    "ru":           "whois.tcinet.ru",
    "se":           "whois.iis.se",
    "no":           "whois.norid.no",
    "fi":           "whois.fi",
    "dk":           "whois.dk-hostmaster.dk",
    "ch":           "whois.nic.ch",
    "at":           "whois.nic.at",
    "be":           "whois.dns.be",
    "pt":           "whois.dns.pt",
    "gr":           "whois.ics.forth.gr",
    "cz":           "whois.nic.cz",
    "sk":           "whois.sk-nic.sk",
    "hu":           "whois.nic.hu",
    "ro":           "whois.rotld.ro",
    "bg":           "whois.register.bg",
    "hr":           "whois.dns.hr",
    "rs":           "whois.rnids.rs",
    "ua":           "whois.ua",
    "gi":           "whois2.afilias-grs.net",
    "je":           "whois.je",
    # Americas
    "br":           "whois.registro.br",
    "ca":           "whois.cira.ca",
    "mx":           "whois.mx",
    "cl":           "whois.nic.cl",
    "ar":           "whois.nic.ar",
    "uy":           "whois.nic.org.uy",
    "pe":           "kero.yachay.pe",
    "ec":           "whois.nic.ec",
    "bo":           "whois.nic.bo",
    "bz":           "whois.belizenic.bz",
    "ag":           "whois2.afilias-grs.net",
    "lc":           "whois2.afilias-grs.net",
    "vc":           "whois2.afilias-grs.net",
    # Asia Pacific
    "au":           "whois.auda.org.au",
    "jp":           "whois.jprs.jp",
    "cn":           "whois.cnnic.cn",
    "in":           "whois.inregistry.net",
    "kr":           "whois.kr",
    "sg":           "whois.sgnic.sg",
    "hk":           "whois.hkirc.hk",
    "tw":           "whois.twnic.net.tw",
    "id":           "whois.pandi.or.id",
    "my":           "whois.mynic.my",
    "ph":           "whois.dot.ph",
    "th":           "whois.thnic.co.th",
    "vn":           "whois.vnnic.vn",
    "pk":           "whois.pknic.net.pk",
    "lk":           "whois.nic.lk",
    "nz":           "whois.irs.net.nz",
    # Middle East & Africa
    "tr":           "whois.nic.tr",
    "ae":           "whois.aeda.net.ae",
    "sa":           "whois.nic.net.sa",
    "il":           "whois.isoc.org.il",
    "za":           "whois.registry.net.za",
    "ng":           "whois.nic.net.ng",
    "ke":           "whois.kenic.or.ke",
    "ma":           "whois.registre.ma",
    "gh":           "whois.nic.gh",
    "tz":           "whois.tznic.or.tz",
}


def _find_whois_server(domain: str) -> Optional[str]:
    parts = domain.split(".")
    # Try second-level TLD first (e.g. co.uk)
    if len(parts) >= 3:
        second_level = ".".join(parts[-2:])
        if second_level in WHOIS_SERVERS:
            return WHOIS_SERVERS[second_level]
    tld = parts[-1]
    if tld in WHOIS_SERVERS:
        return WHOIS_SERVERS[tld]
    # Fallback: ask IANA
    try:
        resp = _whois_raw(tld, server="whois.iana.org")
        for line in resp.splitlines():
            if line.lower().startswith("whois:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 1. Subdomain discovery via crt.sh
# ---------------------------------------------------------------------------

def _fetch_crtsh(domain: str, include_expired: bool) -> List[Dict[str, Any]]:
    url = f"https://crt.sh/?q=%25.{urllib.parse.quote(domain)}&output=json"
    try:
        raw = _http_get(url)
    except Exception as e:
        raise RuntimeError(f"crt.sh unreachable: {e}") from e
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"crt.sh returned invalid JSON: {e}") from e
    if not isinstance(entries, list):
        return []

    seen: set = set()
    results: List[Dict[str, Any]] = []
    for entry in entries:
        not_after = entry.get("not_after", "")
        if not include_expired and not_after:
            dt = _parse_iso(not_after)
            if dt and not _future(dt):
                continue
        for name in entry.get("name_value", "").splitlines():
            name = name.strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            results.append({
                "subdomain": name,
                "issuer": entry.get("issuer_name", ""),
                "not_before": entry.get("not_before", ""),
                "not_after": not_after,
                "cert_id": entry.get("id", ""),
            })
    results.sort(key=lambda r: (r["subdomain"].startswith("*"), r["subdomain"]))
    return results


def domain_subdomains(domain: str, include_expired: bool = False, limit: int = 200) -> str:
    """Discover subdomains via certificate transparency logs (crt.sh)."""
    domain = _clean_domain(domain)
    if not domain or "." not in domain:
        return json.dumps({"error": f"Invalid domain: {domain!r}"})
    try:
        records = _fetch_crtsh(domain, include_expired)
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    total = len(records)
    truncated = total > limit
    records = records[:limit]
    wildcards = sum(1 for r in records if r["subdomain"].startswith("*"))
    issuers: Dict[str, int] = {}
    for r in records:
        cn = r["issuer"].split("CN=")[-1].split(",")[0].strip() if r["issuer"] else "Unknown"
        issuers[cn] = issuers.get(cn, 0) + 1
    top_issuers = sorted(issuers.items(), key=lambda x: -x[1])[:5]

    return json.dumps({
        "domain": domain,
        "total_found": total,
        "returned": len(records),
        "truncated": truncated,
        "include_expired": include_expired,
        "summary": {
            "unique_subdomains": len(records) - wildcards,
            "wildcard_entries": wildcards,
            "top_issuers": [{"issuer": k, "count": v} for k, v in top_issuers],
        },
        "subdomains": records,
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 2. SSL / TLS certificate inspection
# ---------------------------------------------------------------------------

def _ssl_info(host: str, port: int, timeout: int) -> Dict[str, Any]:
    def _connect(verify: bool):
        ctx = ssl.create_default_context()
        if not verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                return ssock.getpeercert(), ssock.cipher(), ssock.version()

    warning = None
    try:
        cert, cipher, proto = _connect(verify=True)
    except ssl.SSLCertVerificationError as e:
        warning = str(e)
        try:
            cert, cipher, proto = _connect(verify=False)
        except Exception as e2:
            raise RuntimeError(f"SSL connection failed: {e2}") from e2
    except socket.timeout:
        raise RuntimeError(f"Connection timed out after {timeout}s")
    except ConnectionRefusedError:
        raise RuntimeError(f"Connection refused on {host}:{port}")
    except socket.gaierror as e:
        raise RuntimeError(f"DNS resolution failed for {host}: {e}")
    except Exception as e:
        raise RuntimeError(f"SSL error: {e}")

    def _parse_ssl_date(s):
        for fmt in ("%b %d %H:%M:%S %Y %Z", "%b  %d %H:%M:%S %Y %Z"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _flat(rdns):
        result = {}
        for rdn in rdns:
            for item in rdn:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    result[item[0]] = item[1]
        return result

    def _extract_uris(entries):
        uris = []
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                uris.append(entry[-1])
            else:
                uris.append(str(entry))
        return uris

    not_after = _parse_ssl_date(cert.get("notAfter", ""))
    not_before = _parse_ssl_date(cert.get("notBefore", ""))
    now = datetime.now(timezone.utc)
    days_remaining = (not_after - now).days if not_after else None
    is_expired = days_remaining is not None and days_remaining < 0

    if is_expired:
        status = f"EXPIRED ({abs(days_remaining)} days ago)"
    elif days_remaining is not None and days_remaining <= 14:
        status = f"CRITICAL — expires in {days_remaining} day(s)"
    elif days_remaining is not None and days_remaining <= 30:
        status = f"WARNING — expires in {days_remaining} day(s)"
    else:
        status = f"OK — {days_remaining} day(s) remaining" if days_remaining is not None else "unknown"

    return {
        "host": host,
        "port": port,
        "subject": _flat(cert.get("subject", [])),
        "issuer": _flat(cert.get("issuer", [])),
        "subject_alt_names": [f"{t}:{v}" for t, v in cert.get("subjectAltName", [])],
        "not_before": not_before.isoformat() if not_before else cert.get("notBefore", ""),
        "not_after": not_after.isoformat() if not_after else cert.get("notAfter", ""),
        "days_remaining": days_remaining,
        "is_expired": is_expired,
        "expiry_status": status,
        "tls_version": proto,
        "cipher_suite": cipher[0] if cipher else None,
        "serial_number": cert.get("serialNumber", ""),
        "ocsp_urls": _extract_uris(cert.get("OCSP", [])),
        "ca_issuers": _extract_uris(cert.get("caIssuers", [])),
        "verification_warning": warning,
    }


def domain_ssl(host: str, port: int = 443, timeout: int = 10) -> str:
    """Inspect live SSL/TLS certificate details for a host."""
    host = host.strip().lower().rstrip(".")
    if not host:
        return json.dumps({"error": "host must not be empty"})
    try:
        return json.dumps(_ssl_info(host, port, timeout), ensure_ascii=False, indent=2)
    except RuntimeError as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# 3. WHOIS lookup
# ---------------------------------------------------------------------------

def _whois_raw(domain: str, server: str = "whois.iana.org", port: int = 43) -> str:
    with socket.create_connection((server, port), timeout=10) as s:
        s.sendall((domain + "\r\n").encode())
        chunks = []
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")


def _parse_whois(raw: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"raw_snippet": raw[:800]}
    patterns = {
        "registrar":        r"(?:Registrar|registrar):\s*(.+)",
        "creation_date":    r"(?:Creation Date|Created|created|Registration Time):\s*(.+)",
        "expiration_date":  r"(?:Registry Expiry Date|Expiration Date|Expiry Date|paid-till):\s*(.+)",
        "updated_date":     r"(?:Updated Date|Last Modified|Last updated):\s*(.+)",
        "name_servers":     r"(?:Name Server|nserver):\s*(.+)",
        "status":           r"(?:Domain Status|status):\s*(.+)",
        "registrant_org":   r"(?:Registrant Organization|org):\s*(.+)",
        "registrant_email": r"(?:Registrant Email|e-mail):\s*(.+)",
        "dnssec":           r"DNSSEC:\s*(.+)",
    }
    multi = {"name_servers", "status"}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, raw, re.IGNORECASE)
        if not matches:
            continue
        if key in multi:
            result[key] = list(dict.fromkeys(m.strip().lower() for m in matches))
        else:
            result[key] = matches[0].strip()

    for field in ("creation_date", "expiration_date", "updated_date"):
        raw_val = result.get(field, "")
        if raw_val:
            dt = _parse_iso(raw_val[:19])
            if dt:
                result[field] = dt.isoformat()
                if field == "expiration_date":
                    days = (dt - datetime.now(timezone.utc)).days
                    result["expiration_days_remaining"] = days
                    result["is_expired"] = days < 0
    return result


def domain_whois(domain: str) -> str:
    """Fetch WHOIS registration data for a domain."""
    domain = _clean_domain(domain)
    if not domain or "." not in domain:
        return json.dumps({"error": f"Invalid domain: {domain!r}"})

    server = _find_whois_server(domain)
    if not server:
        tld = domain.rsplit(".", 1)[-1]
        return json.dumps({"error": f"No WHOIS server found for TLD: .{tld}"})

    try:
        raw = _whois_raw(domain, server=server)
    except Exception as e:
        return json.dumps({"error": f"WHOIS query failed: {e}"})

    parsed = _parse_whois(raw)
    parsed["domain"] = domain
    parsed["whois_server"] = server
    return json.dumps(parsed, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 4. DNS record lookup
# ---------------------------------------------------------------------------

def _dns_query(domain: str, qtype: str) -> List[str]:
    if qtype == "A":
        try:
            infos = socket.getaddrinfo(domain, None, socket.AF_INET)
            return list(dict.fromkeys(i[4][0] for i in infos))
        except socket.gaierror:
            return []
    if qtype == "AAAA":
        try:
            infos = socket.getaddrinfo(domain, None, socket.AF_INET6)
            return list(dict.fromkeys(i[4][0] for i in infos))
        except socket.gaierror:
            return []
    url = f"https://dns.google/resolve?name={urllib.parse.quote(domain)}&type={qtype}"
    try:
        raw = _http_get(url)
        data = json.loads(raw)
    except Exception:
        return []
    return [ans.get("data", "").strip().rstrip(".") for ans in data.get("Answer", []) if ans.get("data")]


def domain_dns(domain: str, record_types: Optional[List[str]] = None) -> str:
    """Look up DNS records for a domain."""
    domain = _clean_domain(domain)
    if not domain or "." not in domain:
        return json.dumps({"error": f"Invalid domain: {domain!r}"})

    if not record_types:
        record_types = ["A", "AAAA", "MX", "NS", "TXT", "CNAME"]
    record_types = [r.upper() for r in record_types]

    results: Dict[str, Any] = {"domain": domain, "records": {}}
    for rtype in record_types:
        try:
            results["records"][rtype] = _dns_query(domain, rtype)
        except Exception as e:
            results["records"][rtype] = {"error": str(e)}

    results["has_dns_records"] = any(
        bool(v) for v in results["records"].values() if isinstance(v, list)
    )
    return json.dumps(results, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 5. Domain availability check
# ---------------------------------------------------------------------------

def domain_available(domain: str) -> str:
    """Check if a domain is likely available for registration."""
    domain = _clean_domain(domain)
    if not domain or "." not in domain:
        return json.dumps({"error": f"Invalid domain: {domain!r}"})

    signals: Dict[str, Any] = {}

    a_records = _dns_query(domain, "A")
    ns_records = _dns_query(domain, "NS")
    signals["dns_a_records"] = a_records
    signals["dns_ns_records"] = ns_records
    dns_exists = bool(a_records or ns_records)

    whois_available = None
    whois_note = ""
    server = _find_whois_server(domain)
    if server:
        try:
            raw = _whois_raw(domain, server=server)
            not_found_patterns = [
                "no match", "not found", "no data found", "no entries found",
                "object does not exist", "domain not found", "no information available",
                "status: free", "available",
            ]
            lower_raw = raw.lower()
            if any(p in lower_raw for p in not_found_patterns):
                whois_available = True
                whois_note = "WHOIS returned 'not found' response"
            elif "registrar:" in lower_raw or "creation date:" in lower_raw:
                whois_available = False
                whois_note = "WHOIS shows active registration"
            else:
                whois_note = "WHOIS response inconclusive"
        except Exception as e:
            whois_note = f"WHOIS query failed: {e}"
    else:
        tld = domain.rsplit(".", 1)[-1]
        whois_note = f"No WHOIS server found for .{tld}"
    signals["whois_available"] = whois_available
    signals["whois_note"] = whois_note

    ssl_reachable = False
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((domain, 443), timeout=3) as s:
            with ctx.wrap_socket(s, server_hostname=domain):
                ssl_reachable = True
    except Exception:
        pass
    signals["ssl_reachable"] = ssl_reachable

    if not dns_exists and whois_available is True:
        verdict, confidence = "LIKELY AVAILABLE", "high"
    elif dns_exists or whois_available is False or ssl_reachable:
        verdict, confidence = "REGISTERED / IN USE", "high"
    elif not dns_exists and whois_available is None:
        verdict, confidence = "POSSIBLY AVAILABLE", "medium — DNS clean but WHOIS inconclusive"
    else:
        verdict, confidence = "UNCERTAIN", "low — conflicting signals"

    return json.dumps({
        "domain": domain,
        "verdict": verdict,
        "confidence": confidence,
        "signals": signals,
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 6. Bulk domain intelligence (parallel)
# ---------------------------------------------------------------------------

def _bulk_single(domain: str, checks: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"domain": domain}
    try:
        if "subdomains" in checks:
            result["subdomains"] = json.loads(domain_subdomains(domain, limit=50))
        if "ssl" in checks:
            result["ssl"] = json.loads(domain_ssl(domain))
        if "whois" in checks:
            result["whois"] = json.loads(domain_whois(domain))
        if "dns" in checks:
            result["dns"] = json.loads(domain_dns(domain))
        if "available" in checks:
            result["available"] = json.loads(domain_available(domain))
    except Exception as e:
        result["error"] = str(e)
    return result


def domain_bulk(
    domains: List[str],
    checks: Optional[List[str]] = None,
    max_workers: int = 5,
) -> str:
    """
    Run domain intelligence checks on multiple domains in parallel.

    domains:     List of domain names to analyze (max 20).
    checks:      Which checks to run — any of: ssl, whois, dns, available, subdomains.
                 Defaults to: ssl, whois, dns, available.
    max_workers: Parallel threads (1–10). Default 5.
    """
    if not domains:
        return json.dumps({"error": "domains list is empty"})

    domains = [_clean_domain(d) for d in domains if _clean_domain(d) and "." in _clean_domain(d)]
    if not domains:
        return json.dumps({"error": "No valid domains provided"})
    if len(domains) > 20:
        domains = domains[:20]

    if not checks:
        checks = ["ssl", "whois", "dns", "available"]
    valid_checks = {"subdomains", "ssl", "whois", "dns", "available"}
    checks = [c for c in checks if c in valid_checks]
    if not checks:
        return json.dumps({"error": f"No valid checks. Valid options: {sorted(valid_checks)}"})

    max_workers = min(max(1, max_workers), 10)
    order = {d: i for i, d in enumerate(domains)}
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_bulk_single, d, checks): d for d in domains}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"domain": futures[future], "error": str(e)})

    results.sort(key=lambda r: order.get(r.get("domain", ""), 999))

    return json.dumps({
        "total": len(results),
        "checks_run": checks,
        "results": results,
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def check_domain_intel_requirements() -> bool:
    return True


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schemas
# ---------------------------------------------------------------------------

DOMAIN_SUBDOMAINS_SCHEMA = {
    "name": "domain_subdomains",
    "description": (
        "Discover subdomains for a target domain using certificate transparency logs (crt.sh). "
        "Passive OSINT — no active scanning, no target contact. "
        "Returns unique subdomains with issuer and validity info.\n\n"
        "Use for: passive subdomain enumeration, attack surface mapping, finding forgotten services, "
        "certificate inventory audits. No API key required."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "description": "Root domain (e.g. 'example.com')."},
            "include_expired": {
                "type": "boolean",
                "description": "Include subdomains from expired certs. Default: false.",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Max records to return. Default: 200.",
                "default": 200,
                "minimum": 1,
                "maximum": 2000,
            },
        },
        "required": ["domain"],
    },
}

DOMAIN_SSL_SCHEMA = {
    "name": "domain_ssl",
    "description": (
        "Inspect the live SSL/TLS certificate for a hostname. "
        "Returns subject, issuer, SANs, validity dates, days until expiry "
        "(OK / WARNING / CRITICAL / EXPIRED), TLS version, cipher suite. "
        "Warns on self-signed or mismatched certs.\n\n"
        "Use for: certificate expiry monitoring, SAN enumeration, TLS config auditing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "host": {"type": "string", "description": "Hostname or IP."},
            "port": {"type": "integer", "description": "TLS port. Default: 443.", "default": 443},
            "timeout": {"type": "integer", "description": "Timeout seconds. Default: 10.", "default": 10},
        },
        "required": ["host"],
    },
}

DOMAIN_WHOIS_SCHEMA = {
    "name": "domain_whois",
    "description": (
        "Fetch WHOIS registration data for a domain. Supports 100+ TLDs. "
        "Returns registrar, creation/expiration/update dates, nameservers, "
        "domain status, DNSSEC status, and days until expiry.\n\n"
        "Use for: ownership research, expiry tracking, registrar identification."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "description": "Domain to look up (e.g. 'example.com')."},
        },
        "required": ["domain"],
    },
}

DOMAIN_DNS_SCHEMA = {
    "name": "domain_dns",
    "description": (
        "Look up DNS records for a domain. Supports A, AAAA, MX, NS, TXT, CNAME. "
        "Uses system DNS for A/AAAA and DNS-over-HTTPS (Google) for other types.\n\n"
        "Use for: finding IPs, mail servers, SPF/DKIM/DMARC, infrastructure mapping."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "description": "Domain to query."},
            "record_types": {
                "type": "array",
                "items": {"type": "string", "enum": ["A", "AAAA", "MX", "NS", "TXT", "CNAME"]},
                "description": "Record types to fetch. Default: all.",
                "default": ["A", "AAAA", "MX", "NS", "TXT", "CNAME"],
            },
        },
        "required": ["domain"],
    },
}

DOMAIN_AVAILABLE_SCHEMA = {
    "name": "domain_available",
    "description": (
        "Check if a domain is likely available for registration. "
        "Uses three passive signals: DNS (A/NS), WHOIS, and SSL reachability. "
        "Returns verdict (LIKELY AVAILABLE / REGISTERED / POSSIBLY AVAILABLE / UNCERTAIN) "
        "with confidence level.\n\n"
        "Use for: domain availability checks, brand protection, finding dropped domains."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "description": "Domain to check."},
        },
        "required": ["domain"],
    },
}

DOMAIN_BULK_SCHEMA = {
    "name": "domain_bulk",
    "description": (
        "Run domain intelligence checks on multiple domains in parallel (up to 20). "
        "Supports any combination of: ssl, whois, dns, available, subdomains.\n\n"
        "Use for: batch domain audits, TLD variant checks (example.com/.io/.ai), "
        "bulk cert expiry monitoring, competitor infrastructure mapping."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of domains to analyze (max 20).",
                "maxItems": 20,
            },
            "checks": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["ssl", "whois", "dns", "available", "subdomains"],
                },
                "description": "Checks to run. Default: ['ssl', 'whois', 'dns', 'available'].",
                "default": ["ssl", "whois", "dns", "available"],
            },
            "max_workers": {
                "type": "integer",
                "description": "Parallel threads (1–10). Default: 5.",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["domains"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="domain_subdomains",
    toolset="domain_intel",
    schema=DOMAIN_SUBDOMAINS_SCHEMA,
    handler=lambda args, **_: domain_subdomains(
        domain=args["domain"],
        include_expired=args.get("include_expired", False),
        limit=args.get("limit", 200),
    ),
    check_fn=check_domain_intel_requirements,
    description=DOMAIN_SUBDOMAINS_SCHEMA["description"],
)

registry.register(
    name="domain_ssl",
    toolset="domain_intel",
    schema=DOMAIN_SSL_SCHEMA,
    handler=lambda args, **_: domain_ssl(
        host=args["host"],
        port=args.get("port", 443),
        timeout=args.get("timeout", 10),
    ),
    check_fn=check_domain_intel_requirements,
    description=DOMAIN_SSL_SCHEMA["description"],
)

registry.register(
    name="domain_whois",
    toolset="domain_intel",
    schema=DOMAIN_WHOIS_SCHEMA,
    handler=lambda args, **_: domain_whois(domain=args["domain"]),
    check_fn=check_domain_intel_requirements,
    description=DOMAIN_WHOIS_SCHEMA["description"],
)

registry.register(
    name="domain_dns",
    toolset="domain_intel",
    schema=DOMAIN_DNS_SCHEMA,
    handler=lambda args, **_: domain_dns(
        domain=args["domain"],
        record_types=args.get("record_types"),
    ),
    check_fn=check_domain_intel_requirements,
    description=DOMAIN_DNS_SCHEMA["description"],
)

registry.register(
    name="domain_available",
    toolset="domain_intel",
    schema=DOMAIN_AVAILABLE_SCHEMA,
    handler=lambda args, **_: domain_available(domain=args["domain"]),
    check_fn=check_domain_intel_requirements,
    description=DOMAIN_AVAILABLE_SCHEMA["description"],
)

registry.register(
    name="domain_bulk",
    toolset="domain_intel",
    schema=DOMAIN_BULK_SCHEMA,
    handler=lambda args, **_: domain_bulk(
        domains=args["domains"],
        checks=args.get("checks"),
        max_workers=args.get("max_workers", 5),
    ),
    check_fn=check_domain_intel_requirements,
    description=DOMAIN_BULK_SCHEMA["description"],
)
