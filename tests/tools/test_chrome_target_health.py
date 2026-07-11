"""Safety tests for dedicated Chrome target health checks.

The health check must stay inside the session boundary: supervisor-provided
loopback host/port, allowlisted DevTools HTTP endpoints only, and sanitized
status/log output. These tests intentionally seed sensitive-looking material in
DevTools payloads to prove it is not emitted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pytest

from tools.chrome_target_health import (
    ChromeHealthHTTPResponse,
    ChromeTargetHealthOptions,
    ChromeTargetLocator,
    check_chrome_target_health,
    is_allowed_health_endpoint,
)


@dataclass
class _Transport:
    responses: dict[str, ChromeHealthHTTPResponse]

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str, int, float]] = []

    def __call__(self, host: str, port: int, path: str, timeout_s: float):
        self.calls.append((host, path, port, timeout_s))
        return self.responses[path]


def _response(status: int, payload) -> ChromeHealthHTTPResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return ChromeHealthHTTPResponse(status=status, body=body)


def test_success_uses_only_allowlisted_http_metadata_and_sanitizes_payloads():
    sensitive_token = "raw-secret-token-123"
    transport = _Transport(
        {
            "/json/version": _response(
                200,
                {
                    "Browser": "Chrome/124.0.6367.91",
                    "Protocol-Version": "1.3",
                    "User-Agent": f"Mozilla/5.0 {sensitive_token}",
                    "webSocketDebuggerUrl": (
                        "ws://127.0.0.1:9223/devtools/browser/browser-id"
                        f"?token={sensitive_token}"
                    ),
                },
            ),
            "/json/list": _response(
                200,
                [
                    {
                        "id": "target-1",
                        "type": "page",
                        "title": f"alice@example.com {sensitive_token}",
                        "url": f"https://example.test/?token={sensitive_token}",
                        "webSocketDebuggerUrl": (
                            "ws://127.0.0.1:9223/devtools/page/target-1"
                        ),
                    },
                    {
                        "id": "target-2",
                        "type": "service_worker",
                        "url": "https://worker.example.test/secret.js",
                    },
                ],
            ),
        }
    )

    result = check_chrome_target_health(
        ChromeTargetLocator(host="127.0.0.1", port=9223),
        ChromeTargetHealthOptions(include_targets=True, per_request_timeout_s=2.5),
        http_get=transport,
    )

    assert result["ok"] is True
    assert result["endpoint"] == "127.0.0.1:9223"
    assert result["browser_version"] == "Chrome/124"
    assert result["protocol_version"] == "1.3"
    assert result["target_count"] == 2
    assert result["target_type_counts"] == {"page": 1, "service_worker": 1}
    assert result["operations"] == ["GET /json/version", "GET /json/list"]
    assert transport.calls == [
        ("127.0.0.1", "/json/version", 9223, 2.5),
        ("127.0.0.1", "/json/list", 9223, 2.5),
    ]

    emitted = json.dumps(result, sort_keys=True)
    forbidden_fragments = [
        sensitive_token,
        "alice@example.com",
        "example.test",
        "secret.js",
        "target-1",
        "target-2",
        "devtools/page",
        "webSocketDebuggerUrl",
        "User-Agent",
        "https://",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in emitted


def test_unsafe_output_options_fail_closed_before_network_access():
    def forbidden_transport(host: str, port: int, path: str, timeout_s: float):
        raise AssertionError("transport must not be touched after boundary denial")

    result = check_chrome_target_health(
        ChromeTargetLocator(host="127.0.0.1", port=9223),
        ChromeTargetHealthOptions(include_target_titles=True),
        http_get=forbidden_transport,
    )

    assert result["ok"] is False
    assert result["error_class"] == "boundary_denied"
    assert result["boundary_reason"] == "unsafe_output_requested"
    assert result["operations"] == []
    assert "title" not in json.dumps(result).lower()


def test_non_loopback_endpoint_is_denied_unless_explicitly_allowed():
    def forbidden_transport(host: str, port: int, path: str, timeout_s: float):
        raise AssertionError("remote endpoints must fail closed before network I/O")

    result = check_chrome_target_health(
        ChromeTargetLocator(host="192.0.2.10", port=9223),
        ChromeTargetHealthOptions(),
        http_get=forbidden_transport,
    )

    assert result["ok"] is False
    assert result["error_class"] == "boundary_denied"
    assert result["boundary_reason"] == "remote_endpoint_denied"
    assert result["endpoint"] == "192.0.2.10:9223"


def test_remote_endpoint_can_be_allowed_explicitly_with_sanitized_metadata_only():
    transport = _Transport(
        {
            "/json/version": _response(
                200,
                {
                    "Browser": "Chrome/126.0.1",
                    "Protocol-Version": "1.5",
                    "webSocketDebuggerUrl": "ws://192.0.2.10:9223/devtools/browser/raw-id",
                },
            )
        }
    )

    result = check_chrome_target_health(
        ChromeTargetLocator(host="192.0.2.10", port=9223),
        ChromeTargetHealthOptions(allow_remote=True),
        http_get=transport,
    )

    assert result["ok"] is True
    assert result["endpoint"] == "192.0.2.10:9223"
    assert result["browser_version"] == "Chrome/126"
    assert result["operations"] == ["GET /json/version"]
    emitted = json.dumps(result, sort_keys=True)
    assert "webSocketDebuggerUrl" not in emitted
    assert "raw-id" not in emitted


def test_malformed_version_response_fails_closed_without_target_fallback():
    raw_body = b"not-json raw-secret-token-123 https://example.test/body"
    transport = _Transport({"/json/version": _response(200, raw_body)})

    result = check_chrome_target_health(
        ChromeTargetLocator(host="127.0.0.1", port=9223),
        ChromeTargetHealthOptions(include_targets=True),
        http_get=transport,
    )

    assert result["ok"] is False
    assert result["error_class"] == "bad_json"
    assert result["operations"] == ["GET /json/version"]
    assert transport.calls == [("127.0.0.1", "/json/version", 9223, 3.0)]

    emitted = json.dumps(result, sort_keys=True)
    assert "raw-secret-token-123" not in emitted
    assert "example.test" not in emitted
    assert "not-json" not in emitted


def test_oversize_health_body_fails_closed_without_parsing_or_echoing_body():
    raw_body = (
        b'{"Browser":"Chrome/127","secret":"raw-secret-token-oversize"}'
        + b"x" * (128 * 1024)
    )
    transport = _Transport({"/json/version": _response(200, raw_body)})

    result = check_chrome_target_health(
        ChromeTargetLocator(host="127.0.0.1", port=9223),
        ChromeTargetHealthOptions(include_targets=True),
        http_get=transport,
    )

    assert result["ok"] is False
    assert result["error_class"] == "oversize_body"
    assert result["operations"] == ["GET /json/version"]
    emitted = json.dumps(result, sort_keys=True)
    assert "raw-secret-token-oversize" not in emitted


def test_sanitized_logging_uses_audit_only(caplog):
    sensitive_token = "raw-secret-token-456"
    transport = _Transport(
        {
            "/json/version": _response(
                200,
                {
                    "Browser": "Chrome/125.1.2.3",
                    "Protocol-Version": "1.4",
                    "User-Agent": f"Mozilla {sensitive_token}",
                    "webSocketDebuggerUrl": (
                        "ws://127.0.0.1:9223/devtools/browser/browser-id"
                        f"?token={sensitive_token}"
                    ),
                },
            ),
            "/json/list": _response(
                200,
                [
                    {
                        "id": "target-log-id",
                        "type": "page",
                        "title": f"log-title {sensitive_token}",
                        "url": f"https://log.example.test/?token={sensitive_token}",
                    }
                ],
            ),
        }
    )
    test_logger = logging.getLogger("test.chrome_target_health")

    with caplog.at_level(logging.INFO, logger=test_logger.name):
        result = check_chrome_target_health(
            ChromeTargetLocator(host="127.0.0.1", port=9223),
            ChromeTargetHealthOptions(include_targets=True),
            http_get=transport,
            log=test_logger,
        )

    assert result["ok"] is True
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "GET /json/version" in log_text
    forbidden_fragments = [
        sensitive_token,
        "log-title",
        "log.example.test",
        "target-log-id",
        "devtools/browser",
        "webSocketDebuggerUrl",
        "User-Agent",
        "https://",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in log_text


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/json/version", True),
        ("/json/list", True),
        ("/json", True),
        ("/json/new", False),
        ("/json/new?https://example.test", False),
        ("/json/activate/target-1", False),
        ("/json/close/target-1", False),
        ("/json/version?token=raw-secret-token-123", False),
        ("/devtools/page/target-1", False),
    ],
)
def test_endpoint_allowlist_rejects_side_effect_and_identifier_paths(path, expected):
    assert is_allowed_health_endpoint(path) is expected
