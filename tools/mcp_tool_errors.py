"""MCP connection/transport error classification: URL validation, TLS client certs, identity
headers, redirect header stripping, exception-group unwrapping, auth/session-expired/
method-not-found detection and connect-error formatting. Split from tools/mcp_tool.py."""

import asyncio
import contextlib
import errno
import importlib
import logging
import os
import re
from typing import Any, List, Optional
from urllib.parse import urlparse
from tools.mcp_tool_common import _sanitize_error, _core

logger = logging.getLogger("tools.mcp_tool")

# Stateless (2026-07-28) servers reject a legacy ``initialize`` with this or plain method-not-found.
_JSONRPC_UNSUPPORTED_PROTOCOL_VERSION = -32022


def _jsonrpc_matches(exc: BaseException, codes: tuple, markers: tuple, code=None) -> bool:
    """Structural ``MCPError.error.code`` (or *code*) in *codes*, else any *marker* in ``str(exc).lower()``. Never
    ``isinstance`` on SDK exception types: they arrive wrapped in ExceptionGroups and drift across generations."""
    code = getattr(getattr(exc, "error", None), "code", None) or code
    return code in codes or any(marker in str(exc).lower() for marker in markers)


def _handshake_rejected_as_modern(exc: BaseException) -> bool:
    """True when a failed ``initialize`` signals a stateless-only (2026-07-28) server."""
    return _jsonrpc_matches(
        exc, (_JSONRPC_UNSUPPORTED_PROTOCOL_VERSION, _core._JSONRPC_METHOD_NOT_FOUND),
        ("unsupported protocol version", str(_JSONRPC_UNSUPPORTED_PROTOCOL_VERSION)),
        code=getattr(exc, "code", None)) or _is_method_not_found_error(exc)


def _handshake_answered_with_unsupported_version(exc: BaseException) -> bool:
    """True when ``initialize`` SUCCEEDED on the wire (HTTP 200, a valid InitializeResult) but the SDK
    refused the ``protocolVersion`` the server named — its ``RuntimeError("Unsupported protocol version
    from the server: ...")``. Distinct from a JSON-RPC -32022 rejection, where the server refused us."""
    return "unsupported protocol version from the server" in str(_unwrap_exception_group(exc)).lower()


def _is_method_not_found_error(exc: BaseException) -> bool:
    """True if *exc* is a JSON-RPC ``method not found`` (-32601; ``ping`` is optional in MCP). The
    substring fallback includes "Unknown method: <name>" — without it the ping→list_tools keepalive
    fallback never latches and reconnect-loops.

    The substring fallback matters when a server reports method-not-found without a structural ``-32601``
    code (e.g. surfaced as a plain exception string). Besides the canonical "method not found", many
    JSON-RPC implementations phrase it as "Unknown method: <name>" — agentmemory's MCP server is one such
    case (#50028).
    """
    return _jsonrpc_matches(
        exc, (_core._JSONRPC_METHOD_NOT_FOUND,),
        (str(_core._JSONRPC_METHOD_NOT_FOUND), "method not found", "unknown method", "not found: ping"))


class InvalidMcpUrlError(ValueError):
    """A remote MCP server's ``url`` is not parseable http(s):// — validated once at startup to fail fast.

    Validated once at startup so we fail fast with a clear message instead of burning through the
    reconnect-backoff loop on every attempt. (Ported from anomalyco/opencode#25019.)
    """


class NonMcpEndpointError(ConnectionError):
    """An HTTP MCP URL served a non-MCP 2xx (e.g. ``text/html``). Non-retryable: every attempt gets
    the same page, so backoff is skipped and the server fails immediately. Subclasses ConnectionError
    so broad catches still see a connection problem."""


# Streamable-HTTP rejection statuses an SSE-only server (or its load balancer) produces for the
# chunked ``initialize`` POST: Bad Request, Method Not Allowed, Not Acceptable, Length Required.
_STREAMABLE_REJECT_STATUSES = (400, 405, 406, 411)


def _is_streamable_http_rejection(exc: BaseException) -> bool:
    """True when a Streamable-HTTP connect failure looks like a transport mismatch rather than a
    broken server: a 400-family rejection of the initialize POST, or the SDK's opaque INTERNAL_ERROR
    (-32603 ``Server returned an error response``) it maps such rejections to on mcp >= 2.0 (error
    class per PR #104363, @RohithPariki). Timeouts and auth errors never qualify — neither carries
    these markers — so a slow or 401ing server is not retried on the wrong transport.
    """
    root = _unwrap_exception_group(exc)
    if getattr(getattr(root, "response", None), "status_code", None) in _STREAMABLE_REJECT_STATUSES:
        return True
    code = getattr(getattr(root, "error", None), "code", None)
    return code == -32603 and "server returned an error response" in str(root).lower()


_HTTP_REJECTION_BODY_CHARS = 300


def _make_http_rejection_recorder(sink: dict):
    """httpx response hook for the owned Streamable HTTP client: remembers the last 4xx/5xx the server
    sent (status, method, URL, head of the body). mcp >= 2.0 folds a non-2xx whose body it cannot
    parse as a JSON-RPC error into the opaque ``-32603 Server returned an error response`` — the
    status and the server's own words (e.g. ``400 {"code":-32020,"message":"Unsupported
    MCP-Protocol-Version"}``) never reach the exception, so this is the only place they can be
    observed. SSE bodies are never read (a stream would block the hook)."""

    async def _record(response):
        if response.status_code < 400:
            return
        body = ""
        if response.headers.get("content-type", "").split(";")[0].strip().lower() != "text/event-stream":
            try:
                raw = await response.aread()  # buffered: the SDK's own aread() afterwards sees the same bytes
                body = " ".join(raw[:_HTTP_REJECTION_BODY_CHARS * 4].decode("utf-8", "replace").split())
            except Exception:  # the failure itself is still reported, just without the body
                body = ""
        sink.update(status=response.status_code, method=response.request.method,
                    url=str(response.request.url), body=body[:_HTTP_REJECTION_BODY_CHARS])

    return _record


def _describe_http_failure(exc: BaseException, rejection: dict) -> str:
    """``str(root cause)`` of a Streamable HTTP connect failure; when that root is the SDK's opaque
    ``-32603 Server returned an error response`` and the recorder saw the rejection, the HTTP status,
    request URL and body head are appended so the message names what the server actually said."""
    root = _unwrap_exception_group(exc)
    text = str(root)
    opaque = (getattr(getattr(root, "error", None), "code", None) == -32603
              and "server returned an error response" in text.lower())
    if not (opaque and rejection):
        return text
    detail = f"HTTP {rejection['status']} from {rejection['method']} {rejection['url']}"
    if rejection["body"]:
        detail += f": {rejection['body']}"
    return f"{text} ({detail})"


def _unwrap_exception_group(exc: BaseException) -> BaseException:
    """Root-cause leaf of anyio ``(Base)ExceptionGroup`` wrappers (group ``str()`` is opaque). A
    ``KeyboardInterrupt``/``SystemExit`` leaf anywhere is re-raised, never flattened into a loggable
    error; a non-cancellation leaf is preferred over the ``CancelledError`` noise anyio sprays on siblings."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        leaf: BaseException = exc.split((KeyboardInterrupt, SystemExit))[0]
        if leaf is not None:
            while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
                leaf = leaf.exceptions[0]
            raise leaf
        exc = next((sub for sub in exc.exceptions if not _contains_only_cancellation(sub)), exc.exceptions[0])
    return exc


def _contains_only_cancellation(exc: BaseException) -> bool:
    """True if ``exc`` is (or a group containing only) CancelledError."""
    if isinstance(exc, BaseExceptionGroup):
        return all(_contains_only_cancellation(sub) for sub in exc.exceptions)
    return isinstance(exc, asyncio.CancelledError)


def _classify_mcp_failure(exc: BaseException) -> str:
    """``'permanent'`` (``run()`` parks instead of burning the retry ladder: auth 401/403,
    NonMcpEndpointError, InvalidMcpUrlError, missing stdio command) or ``'transient'`` (backoff retry)."""
    root = _unwrap_exception_group(exc)
    permanent = (_is_auth_error(root)
                 or isinstance(root, (NonMcpEndpointError, InvalidMcpUrlError, FileNotFoundError))
                 or (isinstance(root, OSError) and getattr(root, "errno", None) == errno.ENOENT)
                 # 401/403 HTTPStatusError that _is_auth_error's type-gate missed (auth types not importable here)
                 or getattr(getattr(root, "response", None), "status_code", None) in (401, 403))
    return "permanent" if permanent else "transient"


def _validate_remote_mcp_url(server_name: str, url: Any) -> str:
    """The stripped URL if valid http(s); else InvalidMcpUrlError naming the server (non-string, other scheme —
    stdio servers use ``command`` — or empty host)."""
    def _bad(detail: str) -> InvalidMcpUrlError:
        return InvalidMcpUrlError(f"Invalid MCP URL for '{server_name}': {detail}")
    if not isinstance(url, str):
        raise _bad(f"expected a string, got {type(url).__name__}")
    stripped = url.strip()
    if not stripped:
        raise _bad("empty url")
    try:
        parsed = urlparse(stripped)
    except Exception as exc:  # urlparse is very permissive — belt and braces
        raise _bad(f"{stripped!r} ({exc})") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise _bad(f"scheme must be http or https, got {parsed.scheme!r} ({stripped!r})")
    if not parsed.netloc:
        raise _bad(f"missing host ({stripped!r})")
    if not parsed.hostname:  # ``urlparse`` accepts ``http://:8080`` (empty host, explicit port)
        raise _bad(f"missing hostname ({stripped!r})")
    return stripped


def _resolve_client_cert(server_name: str, config: dict):
    """``client_cert`` / ``client_key`` in httpx's ``cert=`` shape: None, a combined-PEM path,
    ``(cert, key)`` or ``(cert, key, password)``. ``~`` is expanded; missing files raise a
    server-scoped FileNotFoundError instead of an opaque TLS handshake error."""
    raw_cert = config.get("client_cert")
    raw_key = config.get("client_key")
    if raw_cert is None and raw_key is None:
        return None
    prefix = f"MCP server '{server_name}': "

    def _expand(path: Any, label: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{prefix}{label} must be a non-empty string path (got {type(path).__name__})")
        expanded = os.path.expanduser(path.strip())
        if not os.path.isfile(expanded):
            raise FileNotFoundError(f"{prefix}{label} not found at {expanded!r}")
        return expanded
    if not isinstance(raw_cert, (list, tuple)):
        cert_path = _expand(raw_cert, "client_cert")
        return (cert_path, _expand(raw_key, "client_key")) if raw_key is not None else cert_path  # combined PEM
    if raw_key is not None:
        raise ValueError(f"{prefix}specify either client_cert as a list [cert, key] OR client_cert + client_key, not both")
    if len(raw_cert) not in (2, 3):
        raise ValueError(f"{prefix}client_cert list form must have 2 or 3 elements (got {len(raw_cert)})")
    pair = (_expand(raw_cert[0], "client_cert[0]"), _expand(raw_cert[1], "client_cert[1]"))
    if len(raw_cert) == 2:
        return pair
    if not isinstance(raw_cert[2], str):
        raise ValueError(f"{prefix}client_cert[2] (key passphrase) must be a string")
    return (*pair, raw_cert[2])


def _resolve_identity_header(server_name: str, config: dict):
    """``identity_header`` ``{name, value_from: "static"|"profile", value}`` → ``(name, value)`` or
    None. Invalid configs warn and are ignored — an identity header must never break the connection.
    ``profile`` resolves once at connect time."""
    raw = config.get("identity_header")
    if raw is None:
        return None

    def _ignore(detail: str, *args):
        logger.warning("MCP server '%s': identity_header " + detail + " — ignoring", server_name, *args)
        return None
    if not isinstance(raw, dict):
        return _ignore("must be a mapping with 'name' and 'value'/'value_from' keys (got %s)", type(raw).__name__)
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return _ignore("requires a non-empty 'name'")
    value_from = (raw.get("value_from") or "static").strip().lower()
    if value_from == "profile":
        from hermes_cli.profiles import get_active_profile_name
        return (name.strip(), get_active_profile_name())
    if value_from != "static":
        return _ignore("value_from must be 'static' or 'profile' (got %r)", value_from)
    value = raw.get("value")
    if not isinstance(value, str) or not value.strip():
        return _ignore("with value_from: static requires a non-empty string 'value'")
    return (name.strip(), value)


def _apply_identity_header(server_name: str, config: dict, headers: dict) -> dict:
    """Merge the identity header into ``headers`` in place; an explicit entry of the same name (any
    casing) wins — never silently override user config."""
    name, value = _resolve_identity_header(server_name, config) or (None, None)
    if name is None:
        return headers
    if any(key.lower() == name.lower() for key in headers):
        logger.debug("MCP server '%s': identity_header '%s' already set via explicit "
                     "headers config — keeping the explicit value", server_name, name)
    else:
        headers[name] = value
    return headers


def _make_redirect_header_stripper(httpx_mod, original_url, *, strict: bool = False,
                                   configured_header_names: "set[str] | frozenset[str]" = frozenset()):
    """Client factory enforcing the redirect credential boundary: on a cross-origin redirect
    follow-up it strips ``Authorization``; with *strict* (Agent Plugins v1 ``strict_redirect_headers``)
    every configured header (lowercase names in *configured_header_names*) is stripped too — v1 forbids
    forwarding them cross-origin.

    The factory builds ``httpx_mod.AsyncClient(**kwargs)`` — resolved at call time, so the proxy
    ``mounts=`` / ``transport=`` the caller passes reach the SDK's real client class (and anything a
    caller swapped in for it) unchanged — and installs the boundary on that instance's
    ``_build_redirect_request``. This MUST live on ``_build_redirect_request``: ``response.next_request``
    is unset when response event hooks fire (httpx populates it later in the redirect loop), so a
    response hook can never mutate the follow-up; and a *request* hook would fire on non-redirect traffic
    too — the OAuth auth flow yields token/metadata/registration requests through the same client, often
    to a different-origin authorization server whose own credentials must NOT be stripped."""
    origin = (original_url.scheme, original_url.host, original_url.port)

    def _build_client(**kwargs):
        client = httpx_mod.AsyncClient(**kwargs)
        base_build = getattr(type(client), "_build_redirect_request", None)

        def _build_redirect_request(request, response):
            next_request = base_build(client, request, response)
            target = next_request.url
            if (target.scheme, target.host, target.port) != origin:
                headers = next_request.headers
                headers.pop("authorization", None)
                headers.pop("Authorization", None)
                for _name in configured_header_names if strict else ():
                    while _name in headers:
                        del headers[_name]
            return next_request

        client._build_redirect_request = _build_redirect_request
        return client

    return _build_client


# Wire-body cap, applied at the httpx transport before the SDK buffers/JSON-parses a response. A
# hostile or misbehaving remote MCP server can stream an unbounded catalog/tool-result body and none
# of the post-parse limits (resource cap, tool-result truncation) run before the parse blows up.
# Finite HTTP bodies are capped at this many bytes (a larger Content-Length is rejected up front);
# each SSE *event* is capped, with the counter reset at completed event boundaries so a long-lived
# stream and its keepalives have no cumulative limit. Violations raise the SDK httpx's ReadError and
# flow through the ordinary transport teardown/reconnect path (#66092).
_MCP_HTTP_MAX_BODY_BYTES = 10 * 1024 * 1024
# An SSE event ends at a blank line: two consecutive line terminators. The spec allows CR,
# LF, or CRLF terminators and permits mixing them, so the boundary is any of \n\n, \r\r,
# \n\r, \r\n\r\n, \r\n\n, \r\n\r, \n\r\n, \r\r\n. "\r\n" alone is ONE terminator, not two:
# the lookahead keeps a plain CRLF line ending from backtracking into a \r + \n boundary.
_SSE_BOUNDARY_RE = re.compile(rb"(?:\r\n|\r(?!\n)|\n){2}")
_SSE_BOUNDARY_CARRY = 3  # longest boundary ("\r\n\r\n") minus one byte


def _make_mcp_body_cap_transport(httpx_mod, inner_transport, limit: int = _MCP_HTTP_MAX_BODY_BYTES):
    """Wrap ``inner_transport`` so every response body is size-capped. ``httpx_mod`` must be the SDK's
    own httpx module (``sdk_httpx()``): the transport is handed to that SDK's ``AsyncClient``."""

    class _CappedStream(httpx_mod.AsyncByteStream):
        def __init__(self, inner, is_sse: bool, url: str):
            self._inner, self._is_sse, self._url = inner, is_sse, url

        def _reject(self, kind: str):
            return httpx_mod.ReadError(f"MCP {kind} exceeds {limit} bytes (from {self._url})")

        async def __aiter__(self):
            counted = 0
            tail = b""  # last _SSE_BOUNDARY_CARRY stream bytes; a boundary can straddle chunks
            async for chunk in self._inner:
                if self._is_sse:
                    # Charge each completed event once: the carried prefix plus bytes up to its
                    # boundary must fit the cap, then the next event starts after it. Scan the
                    # carried suffix plus this chunk so a boundary split across chunks is still
                    # seen; bytes before len(tail) were already counted into `counted`.
                    window = tail + chunk
                    pos = 0
                    for match in _SSE_BOUNDARY_RE.finditer(window):
                        end = match.end()
                        if end <= len(tail):
                            continue  # boundary completed inside the carried suffix: already counted
                        if counted + end - max(pos, len(tail)) > limit:
                            raise self._reject("SSE event")
                        counted, pos = 0, end
                    counted += len(window) - max(pos, len(tail))
                    tail = window[-_SSE_BOUNDARY_CARRY:]
                else:
                    counted += len(chunk)
                if counted > limit:
                    raise self._reject("SSE event" if self._is_sse else "HTTP response")
                yield chunk

        async def aclose(self):
            await self._inner.aclose()

    class _BodyCapTransport(httpx_mod.AsyncBaseTransport):
        def __init__(self, inner):
            self._inner = inner

        async def handle_async_request(self, request):
            response = await self._inner.handle_async_request(request)
            declared = response.headers.get("content-length")
            with contextlib.suppress(ValueError):  # malformed header: the streamed cap still applies
                if declared is not None and int(declared) > limit:
                    await response.aclose()
                    raise httpx_mod.ReadError(f"MCP HTTP response declares Content-Length {declared} > {limit} "
                                              f"bytes cap (from {request.url})")
            is_sse = "text/event-stream" in response.headers.get("content-type", "").lower()
            response.stream = _CappedStream(response.stream, is_sse, str(request.url))
            return response

        async def aclose(self):
            await self._inner.aclose()

    return _BodyCapTransport(inner_transport)


# Node budget for ``_iter_exception_nodes`` (the visited set breaks cycles; this bounds acyclic blow-ups).
# Well above ``sys.getrecursionlimit()`` so deep task-group nesting is fully scanned.
_EXC_TRAVERSAL_MAX_NODES = 10_000


def _exc_children(exc: BaseException) -> List[BaseException]:
    """A group's sub-exceptions (if any) followed by ``__cause__``/``__context__`` when they are exceptions — a
    group raised inside an ``except`` block carries the caught error as ``__context__``, so the chain is never
    skipped."""
    nested = getattr(exc, "exceptions", None) or ()
    return [*nested, *(c for c in (exc.__cause__, exc.__context__) if isinstance(c, BaseException))]


def _iter_exception_nodes(exc: BaseException) -> List[BaseException]:
    """Pre-order, left-to-right walk of an exception tree/chain, each node once. ``__cause__``/``__context__``
    can point back at an ancestor (a raised-and-caught pair does this routinely, e.g. the same OAuth error
    raised on the Streamable-HTTP attempt and again on the SSE fallback), so a naive recursive walk dies with
    RecursionError and hides the real connect error; the visited set breaks cycles, the budget bounds acyclic
    blow-ups."""
    stack = [exc]
    seen: set[int] = set()
    ordered: List[BaseException] = []
    while stack and len(ordered) < _EXC_TRAVERSAL_MAX_NODES:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        ordered.append(current)
        stack.extend(reversed(_exc_children(current)))
    return ordered


def _format_connect_error(exc: BaseException) -> str:
    """Render nested MCP connection errors into an actionable short message."""
    nodes = _iter_exception_nodes(exc)

    def _find_missing() -> Optional[str]:
        for current in nodes:
            if isinstance(current, FileNotFoundError):
                if getattr(current, "filename", None):
                    return str(current.filename)
                match = re.search(r"No such file or directory: '([^']+)'", str(current))
                if match:
                    return match.group(1)
        return None

    def _flatten_messages() -> List[str]:
        messages: List[str] = []
        for current in nodes:
            # A group's own str() is opaque — only its children speak; a message-less leaf still names its type.
            text = "" if getattr(current, "exceptions", None) else str(current).strip()
            if text:
                messages.append(text)
            elif not _exc_children(current):
                messages.append(current.__class__.__name__)
        return messages or [exc.__class__.__name__]

    missing = _find_missing()
    if not missing:
        return _sanitize_error("; ".join(list(dict.fromkeys(_flatten_messages()))[:3]))
    message = f"missing executable '{missing}'"
    if os.path.basename(missing) in {"npx", "npm", "node"}:
        message += (" (ensure Node.js is installed and PATH includes its bin directory, "
                    "or set mcp_servers.<name>.command to an absolute path and include "
                    "that directory in mcp_servers.<name>.env.PATH)")
    return _sanitize_error(message)


def _optional_types(module: str, *names: str) -> list:
    """``[module.name, ...]`` or ``[]`` when the module/attribute is unavailable."""
    try:
        mod = importlib.import_module(module)
        return [getattr(mod, name) for name in names]
    except (ImportError, AttributeError):
        return []


# Lazily-built ``(auth_types, http_status_types)`` so this module imports without the SDK OAuth module.
_AUTH_ERROR_TYPES: Optional[tuple] = None


def _get_auth_error_types() -> tuple:
    """Cached ``(auth_types, http_status_types)``: SDK ``OAuthFlowError``/``OAuthTokenError`` (+ legacy
    ``UnauthorizedError``), our ``OAuthNonInteractiveError``, and ``HTTPStatusError`` from both httpx
    flavours — a 401 may come from the SDK's own stack (``httpx2`` on mcp >= 2.0) or Hermes' pinned
    ``httpx``; the classes are unrelated and still need the 401 check in :func:`_is_auth_error`."""
    global _AUTH_ERROR_TYPES
    if not (_AUTH_ERROR_TYPES and _AUTH_ERROR_TYPES[0]):  # retry while empty (SDK may import later)
        sdk_mod = _core.sdk_httpx()
        http_types = tuple(dict.fromkeys(
            ([sdk_mod.HTTPStatusError] if sdk_mod is not None else []) + _optional_types("httpx", "HTTPStatusError")))
        auth_types = (*_optional_types("mcp.client.auth", "OAuthFlowError", "OAuthTokenError"),
                      *_optional_types("mcp.client.auth", "UnauthorizedError"),  # older SDKs
                      *_optional_types("tools.mcp_oauth", "OAuthNonInteractiveError"), *http_types)
        _AUTH_ERROR_TYPES = (auth_types, http_types)
    return _AUTH_ERROR_TYPES


def _is_auth_error(exc: BaseException) -> bool:
    """True if ``exc`` indicates an MCP OAuth failure; ``HTTPStatusError`` counts only with status 401."""
    auth_types, http_types = _get_auth_error_types()
    if not isinstance(exc, auth_types):
        return False
    return getattr(exc.response, "status_code", None) == 401 if isinstance(exc, http_types) else True


# Lower-cased substrings meaning the transport session expired / was GC'd (OAuth token still valid).
# Substrings (lower-cased match) that indicate the MCP server rejected the request because its server-side
# transport session expired / was garbage-collected. See #13383.
_SESSION_EXPIRED_MARKERS: tuple = (
    "invalid or expired session", "expired session", "session expired", "session not found",
    "unknown session", "session terminated", "closedresourceerror", "closed resource",
    "transport is closed", "connection closed", "broken pipe", "end of file")

# Stale-session wordings that interpolate an opaque session id between the stable words, so no contiguous
# _SESSION_EXPIRED_MARKERS entry can match them. Observed provider behaviors, not spec'd strings:
#   BrowserOS neo:     "BrowserOS neo session <uuid> is no longer live"
#   other MCP servers: "mcp session <uuid> is not registered"
# Requiring the id BETWEEN the stable words keeps unrelated errors that merely mention a session
# elsewhere from being misread as transport expiry.
_STALE_SESSION_WITH_ID_RE = re.compile(r"session \S+ (?:is no longer live|is not registered)")


def _is_session_expired_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a transport session expiry (Streamable-HTTP servers GC session state on idle TTL /
    restart / pod rotation while the OAuth token stays valid) — the fix is a transport reconnect, not an OAuth
    refresh. Every node ``_iter_exception_nodes`` reaches is inspected so an InterruptedError anywhere overrides
    transport markers; the chain walk matters because SDK wrappers raise a generic RuntimeError *from* a
    message-less ClosedResourceError."""
    # AnyIO stream exceptions are often message-less, so type checks complement marker matching.
    transport_error_types = tuple(_optional_types("anyio", "BrokenResourceError", "ClosedResourceError", "EndOfStream"))
    found = False
    for current in _iter_exception_nodes(exc):
        if isinstance(current, InterruptedError):
            return False
        # Messages vary across SDK versions/servers: a narrow allow-list of stable substrings avoids false positives.
        msg = str(current).lower()
        found = (found or isinstance(current, transport_error_types)
                 or any(m in msg for m in _SESSION_EXPIRED_MARKERS)
                 or bool(_STALE_SESSION_WITH_ID_RE.search(msg)))
    return found
