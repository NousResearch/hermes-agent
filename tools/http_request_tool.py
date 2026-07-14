"""HTTP Request Tool — make generic HTTP/HTTPS API calls safely.

Provides a safe, structured way for agents to call arbitrary REST APIs
without shelling out to curl (which risks injection and produces
unstructured output).

Security:
- SSRF protection via :func:`tools.url_safety.is_safe_url`, re-applied to
  every redirect target (see :class:`_SafeRedirectHandler`) so a public URL
  cannot 302 into a private/metadata address
- Cloud metadata endpoints always blocked
- file:// and other non-http(s) schemes rejected, on redirects too
- URL secret exfiltration guard (blocks URLs embedding API keys)
- Response size and timeout are clamped server-side; the model cannot
  request an unbounded read

Returns structured JSON with status_code, headers, body, and timing.
"""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from tools.registry import registry
from tools.url_safety import is_safe_url
from agent.redact import _PREFIX_RE
from urllib.parse import unquote

logger = logging.getLogger(__name__)

_MAX_RESPONSE_SIZE_DEFAULT = 100_000  # chars
_MAX_RESPONSE_SIZE_CAP = 200_000  # server-side ceiling, model input is clamped
_TIMEOUT_DEFAULT = 30
_TIMEOUT_CAP = 120


def _check_http_request_requirements() -> bool:
    """Always available — no external API key needed."""
    return True


def _unsafe_url_reason(url: str) -> Optional[str]:
    """Shared request/redirect gate. Returns a block reason or None if safe."""
    if not url.startswith(("http://", "https://")):
        return "URL must use http:// or https:// scheme"
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(unquote(url)):
        return (
            "URL contains what appears to be an API key or token. "
            "Secrets must not be sent in URLs."
        )
    if not is_safe_url(url):
        return "URL targets a private, internal, or cloud-metadata address."
    return None


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target before following it.

    Without this, an attacker can host a public URL that 302-redirects to
    http://169.254.169.254/ and bypass the pre-flight is_safe_url check —
    the redirect-revalidation requirement documented in tools/url_safety.py.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        reason = _unsafe_url_reason(newurl)
        if reason is not None:
            raise urllib.error.HTTPError(
                req.full_url, code, f"Blocked unsafe redirect target: {reason}", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(req: urllib.request.Request, timeout: int):
    """Open *req* through an opener whose redirect handler enforces the gate."""
    opener = urllib.request.build_opener(_SafeRedirectHandler)
    return opener.open(req, timeout=timeout)


def http_request_tool(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = _TIMEOUT_DEFAULT,
    max_response_size: int = _MAX_RESPONSE_SIZE_DEFAULT,
) -> str:
    """Make an HTTP/HTTPS request and return structured response data.

    Args:
        url: Target URL (must be http:// or https://).
        method: HTTP method — GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS.
        headers: Optional dict of custom request headers.
        body: Optional request body (string). For JSON APIs, pass a
            JSON-stringified string in ``body`` and set
            ``headers={"Content-Type": "application/json"}``.
        timeout: Request timeout in seconds (default 30, clamped to 120).
        max_response_size: Maximum response body length in characters
            (default 100_000, clamped to 200_000). Larger responses are
            truncated with a note.

    Returns:
        JSON string with keys:
        - ``success`` (bool)
        - ``status_code`` (int)
        - ``status_text`` (str, e.g. "OK")
        - ``headers`` (dict[str, str])
        - ``body`` (str, possibly truncated)
        - ``elapsed_ms`` (float)
        - ``error`` (str, only when success=false)
    """
    # Validate method
    method = (method or "GET").upper().strip()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        return json.dumps({"success": False, "error": f"Unsupported HTTP method: {method}"})

    # Validate URL
    if not url or not isinstance(url, str):
        return json.dumps({"success": False, "error": "URL is required"})

    url = url.strip()
    reason = _unsafe_url_reason(url)
    if reason is not None:
        return json.dumps({"success": False, "error": f"Blocked: {reason}"})

    # Clamp model-provided limits server-side — the schema documents bounds,
    # but nothing stops a model from sending a larger number.
    try:
        timeout = max(1, min(int(timeout), _TIMEOUT_CAP))
    except (TypeError, ValueError):
        timeout = _TIMEOUT_DEFAULT
    try:
        max_response_size = max(1, min(int(max_response_size), _MAX_RESPONSE_SIZE_CAP))
    except (TypeError, ValueError):
        max_response_size = _MAX_RESPONSE_SIZE_DEFAULT

    # Build request
    req_headers = {"User-Agent": "hermes-agent-http-request/1.0"}
    if headers and isinstance(headers, dict):
        for k, v in headers.items():
            if isinstance(k, str) and isinstance(v, str):
                req_headers[k] = v

    data = body.encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)

    start = time.monotonic()
    try:
        with _open(req, timeout=timeout) as resp:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            status_code = resp.getcode()
            status_text = resp.reason or ""

            # Read headers
            resp_headers = dict(resp.getheaders())

            # Read body with size cap — never read more than the clamped
            # limit (+1 to detect truncation) off the wire.
            raw_body = resp.read(max_response_size + 1)
            body_text = raw_body.decode("utf-8", errors="replace")
            truncated = False
            if len(body_text) > max_response_size:
                body_text = body_text[:max_response_size]
                truncated = True

            result = {
                "success": True,
                "status_code": status_code,
                "status_text": status_text,
                "headers": resp_headers,
                "body": body_text,
                "elapsed_ms": round(elapsed_ms, 2),
            }
            if truncated:
                result["truncated"] = True
                result["note"] = (
                    f"Response truncated at {max_response_size} chars. "
                    "Use browser_navigate for very large payloads."
                )
            return json.dumps(result, ensure_ascii=False)

    except urllib.error.HTTPError as e:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        # HTTPError can still have a response body
        body_text = ""
        try:
            raw = e.read(10_000)
            body_text = raw.decode("utf-8", errors="replace")
        except Exception:
            pass
        return json.dumps(
            {
                "success": False,
                "status_code": e.code,
                "status_text": e.reason or "",
                "headers": dict(e.headers) if e.headers else {},
                "body": body_text,
                "elapsed_ms": round(elapsed_ms, 2),
                "error": f"HTTP {e.code}: {e.reason}",
            },
            ensure_ascii=False,
        )

    except urllib.error.URLError as e:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return json.dumps(
            {
                "success": False,
                "error": f"Request failed: {e.reason}",
                "elapsed_ms": round(elapsed_ms, 2),
            },
            ensure_ascii=False,
        )

    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.debug("http_request unexpected error: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "elapsed_ms": round(elapsed_ms, 2),
        }, ensure_ascii=False)


HTTP_REQUEST_SCHEMA: Dict[str, Any] = {
    "name": "http_request",
    "description": (
        "Make an HTTP or HTTPS request to a public API or web endpoint. "
        "Supports GET, POST, PUT, PATCH, DELETE, HEAD, and OPTIONS. "
        "Returns status code, headers, and body as structured JSON. "
        "Use this for REST API calls, webhooks, or fetching JSON/XML data. "
        "For web page content extraction, prefer web_extract. "
        "For browser-based interaction (clicking, scrolling), use browser_navigate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Target URL. Must start with http:// or https://.",
            },
            "method": {
                "type": "string",
                "description": "HTTP method. Defaults to GET.",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
            },
            "headers": {
                "type": "object",
                "description": (
                    "Optional custom headers as a JSON object. "
                    'Example: {"Authorization": "Bearer token", "Content-Type": "application/json"}'
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "Optional request body as a string. "
                    "For JSON APIs, pass a JSON-stringified string and set "
                    "Content-Type header to application/json."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": f"Request timeout in seconds (default: {_TIMEOUT_DEFAULT}).",
                "minimum": 1,
                "maximum": _TIMEOUT_CAP,
            },
            "max_response_size": {
                "type": "integer",
                "description": (
                    f"Maximum response body length in characters (default: "
                    f"{_MAX_RESPONSE_SIZE_DEFAULT}). Larger responses are truncated."
                ),
                "minimum": 1,
                "maximum": _MAX_RESPONSE_SIZE_CAP,
            },
        },
        "required": ["url"],
    },
}

registry.register(
    name="http_request",
    toolset="web",
    schema=HTTP_REQUEST_SCHEMA,
    handler=lambda args, **kw: http_request_tool(
        url=args.get("url", ""),
        method=args.get("method", "GET"),
        headers=args.get("headers"),
        body=args.get("body"),
        timeout=args.get("timeout", _TIMEOUT_DEFAULT),
        max_response_size=args.get("max_response_size", _MAX_RESPONSE_SIZE_DEFAULT),
    ),
    check_fn=_check_http_request_requirements,
    requires_env=[],
    description="Make structured HTTP/HTTPS API calls safely.",
    emoji="🌐",
    max_result_size_chars=150_000,
)
