"""Safe, HTTP-only health checks for dedicated Chrome DevTools targets.

This module intentionally does *not* inspect browser sessions. It accepts only
supervisor-provided locator data, touches a small allowlist of DevTools HTTP
metadata endpoints, and emits a sanitized status object. It must never read
Chrome profile directories, attach to page targets, return target identifiers,
page URLs, raw websocket URLs, titles, headers, response bodies, cookies, DOM,
storage, screenshots, or request/response data.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

HEALTH_CHECK_VERSION = "chrome-target-health-v1"
_ALLOWED_HEALTH_ENDPOINTS = frozenset({"/json/version", "/json/list", "/json"})
_DEFAULT_TARGET_LIST_ENDPOINT = "/json/list"
_MAX_HEALTH_BODY_BYTES = 128 * 1024
_SAFE_VERSION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_.-]{0,63})/(\d{1,5})")
_SAFE_PROTOCOL_RE = re.compile(r"^[0-9][0-9.]{0,31}$")
_SAFE_TARGET_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")


@dataclass(frozen=True)
class ChromeTargetLocator:
    """Supervisor-provided Chrome DevTools host/port locator.

    The health check never discovers sessions by reading profile directories or
    process command lines. Callers must provide the dedicated target's host and
    port explicitly.
    """

    host: str
    port: int


@dataclass(frozen=True)
class ChromeTargetHealthOptions:
    """Runtime controls for the bounded health check."""

    include_targets: bool = False
    allow_remote: bool = False
    per_request_timeout_s: float = 3.0
    include_target_titles: bool = False
    include_raw_target_details: bool = False
    health_check_version: str = HEALTH_CHECK_VERSION


@dataclass(frozen=True)
class ChromeHealthHTTPResponse:
    """Minimal HTTP response surface consumed by the health checker.

    Headers are deliberately omitted: health checks must not read or emit
    request/response headers or page-originated bodies.
    """

    status: int
    body: bytes


HttpGet = Callable[[str, int, str, float], ChromeHealthHTTPResponse]


def is_allowed_health_endpoint(path: str) -> bool:
    """Return True only for side-effect-free DevTools health metadata paths."""

    return path in _ALLOWED_HEALTH_ENDPOINTS


def check_chrome_target_health(
    locator: ChromeTargetLocator,
    options: ChromeTargetHealthOptions | None = None,
    *,
    http_get: HttpGet | None = None,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Probe a dedicated Chrome target and return sanitized structured status.

    The routine is fail-closed: unsafe locator/options are rejected before any
    network I/O; malformed or unavailable DevTools metadata returns a bounded
    error class without trying deeper CDP/browser/page inspection.
    """

    opts = options or ChromeTargetHealthOptions()
    started = time.monotonic()
    operations: list[str] = []
    endpoint = _endpoint_label(locator.host, locator.port)

    boundary_problem = _boundary_problem(locator, opts)
    if boundary_problem is not None:
        return _finalize(
            {
                "ok": False,
                "endpoint": endpoint,
                "error_class": "boundary_denied",
                "error": "health check boundary denied",
                "operations": operations,
                "boundary_reason": boundary_problem,
            },
            started,
            opts,
            log,
        )

    getter = http_get or _default_http_get

    version_response = _safe_get(
        getter,
        locator,
        "/json/version",
        opts.per_request_timeout_s,
        operations,
    )
    if not version_response["ok"]:
        return _finalize(
            {
                "ok": False,
                "endpoint": endpoint,
                "error_class": version_response["error_class"],
                "error": version_response["error"],
                "operations": operations,
            },
            started,
            opts,
            log,
        )

    version_payload = _parse_json(version_response["body"])
    if version_payload["error_class"] is not None:
        return _finalize(
            {
                "ok": False,
                "endpoint": endpoint,
                "error_class": version_payload["error_class"],
                "error": version_payload["error"],
                "operations": operations,
            },
            started,
            opts,
            log,
        )
    if not isinstance(version_payload["value"], Mapping):
        return _finalize(
            {
                "ok": False,
                "endpoint": endpoint,
                "error_class": "unexpected_shape",
                "error": "unexpected health metadata shape",
                "operations": operations,
            },
            started,
            opts,
            log,
        )

    version_data = version_payload["value"]
    status: dict[str, Any] = {
        "ok": True,
        "endpoint": endpoint,
        "operations": operations,
    }
    browser_version = _sanitize_browser_version(version_data.get("Browser"))
    if browser_version:
        status["browser_version"] = browser_version
    protocol_version = _sanitize_protocol_version(version_data.get("Protocol-Version"))
    if protocol_version:
        status["protocol_version"] = protocol_version

    if opts.include_targets:
        target_response = _safe_get(
            getter,
            locator,
            _DEFAULT_TARGET_LIST_ENDPOINT,
            opts.per_request_timeout_s,
            operations,
        )
        if not target_response["ok"]:
            return _finalize(
                {
                    "ok": False,
                    "endpoint": endpoint,
                    "error_class": target_response["error_class"],
                    "error": target_response["error"],
                    "operations": operations,
                },
                started,
                opts,
                log,
            )
        target_payload = _parse_json(target_response["body"])
        if target_payload["error_class"] is not None:
            return _finalize(
                {
                    "ok": False,
                    "endpoint": endpoint,
                    "error_class": target_payload["error_class"],
                    "error": target_payload["error"],
                    "operations": operations,
                },
                started,
                opts,
                log,
            )
        if not isinstance(target_payload["value"], list):
            return _finalize(
                {
                    "ok": False,
                    "endpoint": endpoint,
                    "error_class": "unexpected_shape",
                    "error": "unexpected target metadata shape",
                    "operations": operations,
                },
                started,
                opts,
                log,
            )
        target_count, target_type_counts = _count_target_types(target_payload["value"])
        status["target_count"] = target_count
        status["target_type_counts"] = target_type_counts

    return _finalize(status, started, opts, log)


def _safe_get(
    http_get: HttpGet,
    locator: ChromeTargetLocator,
    path: str,
    timeout_s: float,
    operations: list[str],
) -> dict[str, Any]:
    if not is_allowed_health_endpoint(path):
        return {
            "ok": False,
            "error_class": "boundary_denied",
            "error": "health endpoint denied",
        }
    operations.append(f"GET {path}")
    try:
        response = http_get(locator.host, locator.port, path, timeout_s)
    except (TimeoutError, socket.timeout):
        return {"ok": False, "error_class": "timeout", "error": "health endpoint timeout"}
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
            return {"ok": False, "error_class": "timeout", "error": "health endpoint timeout"}
        return {"ok": False, "error_class": "unreachable", "error": "health endpoint unreachable"}
    except OSError:
        return {"ok": False, "error_class": "unreachable", "error": "health endpoint unreachable"}

    if response.status != 200:
        return {"ok": False, "error_class": "bad_status", "error": "health endpoint bad status"}
    if len(response.body) > _MAX_HEALTH_BODY_BYTES:
        return {"ok": False, "error_class": "oversize_body", "error": "health endpoint response too large"}
    return {"ok": True, "body": response.body}


def _parse_json(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"value": None, "error_class": "bad_json", "error": "health endpoint bad json"}
    return {"value": value, "error_class": None, "error": None}


def _default_http_get(host: str, port: int, path: str, timeout_s: float) -> ChromeHealthHTTPResponse:
    if not is_allowed_health_endpoint(path):
        raise ValueError("health endpoint denied")
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    url = f"http://{url_host}:{port}{path}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            status = int(getattr(response, "status", response.getcode()))
            body = response.read(_MAX_HEALTH_BODY_BYTES + 1)
            return ChromeHealthHTTPResponse(status=status, body=body)
    except urllib.error.HTTPError as exc:
        return ChromeHealthHTTPResponse(status=int(exc.code), body=b"")


def _boundary_problem(
    locator: ChromeTargetLocator,
    options: ChromeTargetHealthOptions,
) -> str | None:
    if options.include_target_titles or options.include_raw_target_details:
        return "unsafe_output_requested"
    if not isinstance(locator.host, str) or not locator.host.strip():
        return "invalid_host"
    host = locator.host.strip()
    if any(ch in host for ch in "/@?#") or host.startswith(("http://", "https://", "ws://", "wss://")):
        return "invalid_host"
    if not isinstance(locator.port, int) or not (1 <= locator.port <= 65535):
        return "invalid_port"
    if not isinstance(options.per_request_timeout_s, (int, float)) or options.per_request_timeout_s <= 0:
        return "invalid_timeout"
    if options.per_request_timeout_s > 10.0:
        return "invalid_timeout"
    if not options.allow_remote and not _is_loopback_host(host):
        return "remote_endpoint_denied"
    return None


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _endpoint_label(host: str, port: int) -> str:
    safe_host = str(host).strip() or "<invalid>"
    if any(ch in safe_host for ch in "/@?#"):
        safe_host = "<invalid>"
    if ":" in safe_host and not safe_host.startswith("["):
        safe_host = f"[{safe_host}]"
    return f"{safe_host}:{port}"


def _sanitize_browser_version(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().split()[0] if raw.strip() else ""
    match = _SAFE_VERSION_RE.match(candidate)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    if not candidate:
        return None
    product = re.sub(r"[^A-Za-z0-9_.-]", "", candidate.split("/", 1)[0])[:64]
    return product or None


def _sanitize_protocol_version(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if _SAFE_PROTOCOL_RE.match(value):
        return value
    return None


def _count_target_types(targets: list[Any]) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    for item in targets:
        target_type = "unknown"
        if isinstance(item, Mapping):
            raw_type = item.get("type")
            if isinstance(raw_type, str) and _SAFE_TARGET_TYPE_RE.match(raw_type):
                target_type = raw_type
        counts[target_type] = counts.get(target_type, 0) + 1
    return len(targets), dict(sorted(counts.items()))


def _finalize(
    status: dict[str, Any],
    started: float,
    options: ChromeTargetHealthOptions,
    log: logging.Logger | None,
) -> dict[str, Any]:
    duration_ms = max(0, int(round((time.monotonic() - started) * 1000)))
    status["duration_ms"] = duration_ms
    audit = {
        "timestamp_ms": int(time.time() * 1000),
        "health_check_version": options.health_check_version,
        "endpoint": status.get("endpoint"),
        "operations": list(status.get("operations", [])),
        "result": "ok" if status.get("ok") else status.get("error_class", "error"),
        "duration_ms": duration_ms,
    }
    status["audit"] = audit
    active_logger = log
    if active_logger is not None:
        active_logger.info("chrome target health check: %s", json.dumps(audit, sort_keys=True))
    return status


__all__ = [
    "ChromeHealthHTTPResponse",
    "ChromeTargetHealthOptions",
    "ChromeTargetLocator",
    "HEALTH_CHECK_VERSION",
    "check_chrome_target_health",
    "is_allowed_health_endpoint",
]
