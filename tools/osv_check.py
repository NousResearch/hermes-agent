"""OSV malware check for MCP extension packages.

Before launching an MCP server via npx/uvx, queries the OSV (Open Source
Vulnerabilities) API to check if the package has any known malware advisories
(MAL-* IDs).  Regular CVEs are ignored — only confirmed malware is blocked.

The API is free, public, and maintained by Google.  Typical latency is ~300ms.
Fail-open: network errors allow the package to proceed.

Inspired by Block/goose's extension malware check.
"""

import json
import logging
import os
import re
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_OSV_ENDPOINT = os.getenv("OSV_ENDPOINT", "https://api.osv.dev/v1/query")
_TIMEOUT = 10  # seconds


def check_package_for_malware(
    command: str, args: list
) -> Optional[str]:
    """Check if an MCP server package has known malware advisories.

    Inspects the *command* (e.g. ``npx``, ``uvx``) and *args* to infer the
    package name and ecosystem.  Queries the OSV API for MAL-* advisories.

    Returns:
        An error message string if malware is found, or None if clean/unknown.
        Returns None (allow) on network errors or unrecognized commands.
    """
    ecosystem = _infer_ecosystem(command)
    if not ecosystem:
        return None  # not npx/uvx — skip

    package, version = _parse_package_from_args(args, ecosystem)
    if not package:
        return None

    try:
        malware = _query_osv(package, ecosystem, version)
    except Exception as exc:
        # Fail-open: network errors, timeouts, parse failures → allow
        logger.debug("OSV check failed for %s/%s (allowing): %s", ecosystem, package, exc)
        return None

    if malware:
        ids = ", ".join(m["id"] for m in malware[:3])
        summaries = "; ".join(
            m.get("summary", m["id"])[:100] for m in malware[:3]
        )
        return (
            f"BLOCKED: Package '{package}' ({ecosystem}) has known malware "
            f"advisories: {ids}. Details: {summaries}"
        )
    return None


def _infer_ecosystem(command: str) -> Optional[str]:
    """Infer package ecosystem from the command name."""
    base = os.path.basename(command).lower()
    if base in {"npx", "npx.cmd"}:
        return "npm"
    if base in {"uvx", "uvx.cmd", "pipx"}:
        return "PyPI"
    return None


def _parse_package_from_args(
    args: list, ecosystem: str
) -> Tuple[Optional[str], Optional[str]]:
    """Extract package name and optional version from command args.

    Returns (package_name, version) or (None, None) if not parseable.
    """
    if not args:
        return None, None

    # Skip flags to find the package token
    package_token = None
    for arg in args:
        if not isinstance(arg, str):
            continue
        if arg.startswith("-"):
            continue
        package_token = arg
        break

    if not package_token:
        return None, None

    if ecosystem == "npm":
        return _parse_npm_package(package_token)
    elif ecosystem == "PyPI":
        return _parse_pypi_package(package_token)
    return package_token, None


def _parse_npm_package(token: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse npm package: @scope/name@version or name@version."""
    if token.startswith("@"):
        # Scoped: @scope/name@version
        match = re.match(r"^(@[^/]+/[^@]+)(?:@(.+))?$", token)
        if match:
            return match.group(1), match.group(2)
        return token, None
    # Unscoped: name@version
    if "@" in token:
        parts = token.rsplit("@", 1)
        name = parts[0]
        version = parts[1] if len(parts) > 1 and parts[1] != "latest" else None
        return name, version
    return token, None


def _parse_pypi_package(token: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse PyPI package: name==version or name[extras]==version."""
    # Strip extras: name[extra1,extra2]==version
    match = re.match(r"^([a-zA-Z0-9._-]+)(?:\[[^\]]*\])?(?:==(.+))?$", token)
    if match:
        return match.group(1), match.group(2)
    return token, None


def _query_osv(
    package: str, ecosystem: str, version: Optional[str] = None
) -> list:
    """Query the OSV API for MAL-* advisories. Returns list of malware vulns."""
    payload = {"package": {"name": package, "ecosystem": ecosystem}}
    if version:
        payload["version"] = version

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _OSV_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "hermes-agent-osv-check/1.0",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        result = json.loads(resp.read())

    vulns = result.get("vulns", [])
    # Only malware advisories — ignore regular CVEs
    return [v for v in vulns if v.get("id", "").startswith("MAL-")]
