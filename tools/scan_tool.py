"""Native web vulnerability scanner tool for Hermes Agent.

Wraps hermes_scan as a first-class tool — the agent calls scan_web() directly,
same as cancer-cli's native scan integration. No execute_code wrapper needed.

Usage by the agent:
    scan_web(target="https://target.com/page?id=1", profile="aggressive")
"""

import json
import logging
import sys
import os

logger = logging.getLogger(__name__)

# Ensure hermes_scan is importable
_SCAN_PATH = os.path.expanduser("~/.hermes/scripts")
if _SCAN_PATH not in sys.path:
    sys.path.insert(0, _SCAN_PATH)


SCAN_WEB_SCHEMA = {
    "name": "scan_web",
    "description": (
        "Run an automated web vulnerability scan against a target URL. "
        "Tests for SQLi, XSS, SSRF, LFI, command injection, SSTI, XXE, "
        "CORS, CSRF, JWT flaws, open redirects, default credentials, and "
        "security header issues.\n\n"
        "The scanner discovers endpoints (links, forms) on the target page, "
        "then runs 13 detection modules against all parameters found.\n\n"
        "Use this during penetration testing (Phase 4) for broad automated "
        "coverage before manual exploitation. Scanner findings need manual "
        "verification to confirm exploitability.\n\n"
        "Returns structured findings with severity, evidence, and CVSS scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Target URL to scan. Include query parameters for best results "
                    "(e.g. https://target.com/page?id=1&name=test). The scanner "
                    "will also discover additional endpoints by crawling the page."
                ),
            },
            "profile": {
                "type": "string",
                "enum": ["stealth", "normal", "aggressive", "brutal"],
                "description": (
                    "Scan intensity profile. "
                    "stealth: passive only, 2s between requests, 3 payloads/param. "
                    "normal: safe payloads, 0.5s rate limit, 10 payloads/param. "
                    "aggressive: full testing incl. time-based blind, 0.1s rate, 30 payloads/param. "
                    "brutal: everything including DoS probes, no rate limit."
                ),
            },
            "modules": {
                "type": "string",
                "description": (
                    "Comma-separated module names to run. Available: "
                    "sqli, xss, ssrf, lfi, cmdi, cors, csrf, jwt, xxe, ssti, "
                    "headers, redirect, default_creds. "
                    "Omit to run all modules."
                ),
            },
            "cookies": {
                "type": "string",
                "description": "Cookie string for authenticated scanning (e.g. 'PHPSESSID=abc123; token=xyz').",
            },
            "headers": {
                "type": "string",
                "description": (
                    "Additional HTTP headers as JSON object "
                    "(e.g. '{\"Authorization\": \"Bearer token123\"}')."
                ),
            },
            "proxy": {
                "type": "string",
                "description": "HTTP proxy URL (e.g. http://127.0.0.1:8080 for Burp Suite).",
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds (default: 30).",
            },
            "no_verify_tls": {
                "type": "boolean",
                "description": "Skip TLS certificate verification (useful with proxy).",
            },
        },
        "required": ["target"],
    },
}


def scan_web_handler(args, **kwargs):
    """Handle scan_web tool invocation."""
    target = args.get("target", "")
    if not target:
        return tool_error("target is required")

    if not target.startswith(("http://", "https://")):
        return tool_error("target must be a full URL starting with http:// or https://")

    profile = args.get("profile", "normal")
    modules_str = args.get("modules", "")
    cookies = args.get("cookies", "")
    headers_str = args.get("headers", "")
    proxy = args.get("proxy")
    timeout = args.get("timeout", 30)
    no_verify_tls = args.get("no_verify_tls", False)

    # Parse modules
    modules = None
    if modules_str:
        modules = [m.strip() for m in modules_str.split(",") if m.strip()]

    # Parse headers
    headers = {}
    if headers_str:
        try:
            headers = json.loads(headers_str)
            if not isinstance(headers, dict):
                return tool_error("headers must be a JSON object")
        except json.JSONDecodeError as e:
            return tool_error(f"Invalid headers JSON: {e}")

    # Import and run
    try:
        from hermes_scan import scan_web
    except ImportError as e:
        return tool_error(
            f"hermes_scan not installed: {e}. "
            f"Expected at ~/.hermes/scripts/hermes_scan/"
        )

    try:
        result = scan_web(
            target=target,
            profile=profile,
            modules=modules,
            headers=headers,
            cookies=cookies,
            proxy=proxy,
            timeout=timeout,
            verify_tls=not no_verify_tls,
        )
    except Exception as e:
        return tool_error(f"Scan failed: {e}")

    # Format output
    output = _format_result(result)
    return output


def _format_result(result) -> str:
    """Format ScanResult into a readable string for the agent."""
    lines = []
    lines.append(f"## Scan Complete")
    lines.append(f"")
    lines.append(f"**Target:** {result.target}")
    lines.append(f"**Profile:** {result.profile}")
    lines.append(f"**Duration:** {result.duration:.1f}s")
    lines.append(f"**Endpoints tested:** {result.endpoints_tested}")
    lines.append(f"**Requests sent:** {result.requests_sent}")
    lines.append(f"**Findings:** {len(result.findings)} "
                 f"(Critical: {result.critical_count}, High: {result.high_count})")
    lines.append("")

    if not result.findings:
        lines.append("No vulnerabilities detected.")
    else:
        # Group by severity
        from hermes_scan.types import Severity
        for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
            sev_findings = [f for f in result.findings if f.severity == sev]
            if not sev_findings:
                continue
            lines.append(f"### {sev.value.upper()} ({len(sev_findings)})")
            lines.append("")
            for f in sev_findings:
                lines.append(f"**[{f.module}] {f.title}** ({f.confidence})")
                lines.append(f"- URL: {f.url}")
                lines.append(f"- Parameter: `{f.parameter}`")
                lines.append(f"- CVSS: {f.cvss_score} ({f.cvss_vector})")
                if f.evidence_request:
                    lines.append(f"- Payload: `{f.evidence_request[:200]}`")
                if f.evidence_response:
                    lines.append(f"- Evidence: `{f.evidence_response[:300]}`")
                if f.description:
                    lines.append(f"- Description: {f.description}")
                lines.append("")

    if result.errors:
        lines.append(f"### Errors ({len(result.errors)})")
        for e in result.errors:
            lines.append(f"- {e}")

    # Also include JSON for programmatic use
    lines.append("")
    lines.append("<details><summary>Raw JSON</summary>")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result.to_dict(), indent=2))
    lines.append("```")
    lines.append("</details>")

    return "\n".join(lines)


# ── Host Scan ─────────────────────────────────────────────────────────────────

SCAN_HOST_SCHEMA = {
    "name": "scan_host",
    "description": (
        "Run a host/network vulnerability scan against an IP address, hostname, "
        "or CIDR range. Performs port scanning, service fingerprinting, default "
        "credential checks, SSL/TLS auditing, SMB null session testing, DNS "
        "zone transfer attempts, and HTTP service discovery.\n\n"
        "Use this for infrastructure-level reconnaissance before web application "
        "testing. Maps the attack surface of a network host or subnet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Target to scan: IP address (e.g. 10.0.0.1), hostname "
                    "(e.g. target.com), or CIDR range (e.g. 192.168.1.0/24)."
                ),
            },
            "profile": {
                "type": "string",
                "enum": ["stealth", "normal", "aggressive", "brutal"],
                "description": (
                    "Scan intensity profile. "
                    "stealth: top 100 ports, 30s timeout. "
                    "normal: top 100 ports, 10s timeout. "
                    "aggressive: top 1000 ports, 5s timeout. "
                    "brutal: all 65535 ports, 3s timeout."
                ),
            },
        },
        "required": ["target"],
    },
}


def scan_host_handler(args, **kwargs):
    """Handle scan_host tool invocation."""
    target = args.get("target", "")
    if not target:
        return tool_error("target is required")

    profile = args.get("profile", "normal")

    try:
        from hermes_scan.platforms.host import scan_host as _scan_host
    except ImportError as e:
        return tool_error(
            f"hermes_scan not installed: {e}. "
            f"Expected at ~/.hermes/scripts/hermes_scan/"
        )

    try:
        result = _scan_host(target=target, profile=profile)
    except Exception as e:
        return tool_error(f"Host scan failed: {e}")

    output = _format_result(result)
    return output


# ── Mobile Scan ───────────────────────────────────────────────────────────────

SCAN_MOBILE_SCHEMA = {
    "name": "scan_mobile",
    "description": (
        "Run a mobile application static security scan against an Android APK "
        "or iOS IPA file. Checks for exported components, dangerous permissions, "
        "hardcoded secrets, insecure network config, deep link extraction, API "
        "endpoint discovery, certificate pinning, and debug/backup flags.\n\n"
        "Use this during mobile application penetration testing (Phase 2: Static "
        "Analysis) to identify security weaknesses without running the app."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Path to the APK (.apk) or IPA (.ipa) file to scan.",
            },
            "profile": {
                "type": "string",
                "enum": ["stealth", "normal", "aggressive", "brutal"],
                "description": "Scan intensity profile. Default: normal.",
            },
        },
        "required": ["target"],
    },
}


def scan_mobile_handler(args, **kwargs):
    """Handle scan_mobile tool invocation."""
    target = args.get("target", "")
    if not target:
        return tool_error("target is required")

    profile = args.get("profile", "normal")

    import asyncio

    try:
        from hermes_scan.platforms.mobile import scan_mobile as _scan_mobile
        from hermes_scan.types import ScanConfig, ScanProfile
    except ImportError as e:
        return tool_error(
            f"hermes_scan not installed: {e}. "
            f"Expected at ~/.hermes/scripts/hermes_scan/"
        )

    config = ScanConfig(
        target=target,
        profile=ScanProfile(profile),
    )

    try:
        result = asyncio.run(_scan_mobile(target_path=target, config=config))
    except Exception as e:
        return tool_error(f"Mobile scan failed: {e}")

    output = _format_result(result)
    return output


# ── Source Scan ───────────────────────────────────────────────────────────────

SCAN_SOURCE_SCHEMA = {
    "name": "scan_source",
    "description": (
        "Run a source code security scan against a local directory. Performs "
        "pure-Python regex-based static analysis for hardcoded secrets (API keys, "
        "passwords, private keys), SQL injection sinks, command injection sinks, "
        "path traversal sinks, XSS sinks, deserialization sinks, insecure crypto, "
        "dependency auditing, sensitive file detection, and debug/dev code left "
        "in production.\n\n"
        "Use this for white-box security reviews of application source code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Path to the source code directory to scan.",
            },
            "profile": {
                "type": "string",
                "enum": ["stealth", "normal", "aggressive", "brutal"],
                "description": "Scan intensity profile. Default: normal.",
            },
        },
        "required": ["target"],
    },
}


def scan_source_handler(args, **kwargs):
    """Handle scan_source tool invocation."""
    target = args.get("target", "")
    if not target:
        return tool_error("target is required")

    profile = args.get("profile", "normal")

    try:
        from hermes_scan.scanner import scan_source as _scan_source
    except ImportError as e:
        return tool_error(
            f"hermes_scan not installed: {e}. "
            f"Expected at ~/.hermes/scripts/hermes_scan/"
        )

    try:
        result = _scan_source(target_path=target, profile=profile)
    except Exception as e:
        return tool_error(f"Source scan failed: {e}")

    output = _format_result(result)
    return output


# ── Binary Scan ───────────────────────────────────────────────────────────────

SCAN_BINARY_SCHEMA = {
    "name": "scan_binary",
    "description": (
        "Run a binary file security scan against ELF, PE, or Mach-O executables. "
        "Checks binary protections (NX, PIE, RELRO, Stack Canary), extracts "
        "sensitive strings (URLs, IPs, secrets), detects dangerous function "
        "imports (gets, system, strcpy), discovers embedded certificates/keys, "
        "analyzes debug symbols, and identifies DLL hijacking potential.\n\n"
        "Use this when you have access to compiled binaries during a penetration "
        "test or red team engagement."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Path to the executable binary file to scan.",
            },
            "profile": {
                "type": "string",
                "enum": ["stealth", "normal", "aggressive", "brutal"],
                "description": "Scan intensity profile. Default: normal.",
            },
        },
        "required": ["target"],
    },
}


def scan_binary_handler(args, **kwargs):
    """Handle scan_binary tool invocation."""
    target = args.get("target", "")
    if not target:
        return tool_error("target is required")

    profile = args.get("profile", "normal")

    try:
        from hermes_scan.platforms.binary import scan_binary as _scan_binary
    except ImportError as e:
        return tool_error(
            f"hermes_scan not installed: {e}. "
            f"Expected at ~/.hermes/scripts/hermes_scan/"
        )

    try:
        result = _scan_binary(target_path=target, profile=profile)
    except Exception as e:
        return tool_error(f"Binary scan failed: {e}")

    output = _format_result(result)
    return output


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="scan_web",
    toolset="security",
    schema=SCAN_WEB_SCHEMA,
    handler=lambda args, **kw: scan_web_handler(args, **kw),
    emoji="🔍",
    description="Run automated web vulnerability scan (13 modules: SQLi, XSS, SSRF, LFI, etc.)",
)

registry.register(
    name="scan_host",
    toolset="security",
    schema=SCAN_HOST_SCHEMA,
    handler=lambda args, **kw: scan_host_handler(args, **kw),
    emoji="🖥️",
    description="Run host/network vulnerability scan (ports, services, default creds, SSL/TLS)",
)

registry.register(
    name="scan_mobile",
    toolset="security",
    schema=SCAN_MOBILE_SCHEMA,
    handler=lambda args, **kw: scan_mobile_handler(args, **kw),
    emoji="📱",
    description="Run mobile app static security scan (APK/IPA analysis, secrets, permissions)",
)

registry.register(
    name="scan_source",
    toolset="security",
    schema=SCAN_SOURCE_SCHEMA,
    handler=lambda args, **kw: scan_source_handler(args, **kw),
    emoji="📄",
    description="Run source code security scan (secrets, SQLi, XSS, crypto, deps)",
)

registry.register(
    name="scan_binary",
    toolset="security",
    schema=SCAN_BINARY_SCHEMA,
    handler=lambda args, **kw: scan_binary_handler(args, **kw),
    emoji="⚙️",
    description="Run binary file security scan (protections, strings, dangerous imports, certs)",
)
