"""Deterministic fallback routing for Hermes web search and extraction.

Firecrawl intentionally does not appear in this module's automatic paths. It
remains an explicit crawl/map tool, never a surprise search/extract escalation.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from agent.redact import _PREFIX_RE
from hermes_constants import get_hermes_home
from tools.url_safety import async_is_safe_url, sensitive_query_param_name
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

_NEAR_EMPTY_BLOCK_CHARS = 200
_HARD_INTERSTITIAL_MIN_CHARS = 300
_CDP_CALL_TIMEOUT_S = 10.0
_CDP_LOAD_TIMEOUT_S = 15.0
_CDP_CLEANUP_TIMEOUT_S = 2.0
_URL_POLICY_TIMEOUT_S = 5.0
_BROWSER_USE_CLEANUP_TIMEOUT_S = 12.0
_BLOCK_MARKERS = (
    "captcha",
    "cloudflare",
    "datadome",
    "akamai",
    "perimeterx",
    "imperva",
    "incapsula",
    "ddos-guard",
    "access is temporarily restricted",
    "you've been blocked",
    "you have been blocked",
    "bot detection",
    "verification required",
    "checking your browser",
    "just a moment",
)


def _search_with_provider_fallback(
    primary: Any, ddgs_fallback: Optional[Any], query: str, limit: int
) -> Dict[str, Any]:
    """Run Brave, then DDGS only after a real Brave failure.

    A valid zero-result response is not a failure, so it does not get silently
    replaced by a different provider's results.
    """
    providers = [primary]
    if getattr(primary, "name", "") == "brave-free" and ddgs_fallback is not None:
        providers.append(ddgs_fallback)

    attempted: List[str] = []
    last_error = "Search failed"
    for provider in providers:
        name = str(getattr(provider, "name", "unknown"))
        attempted.append(name)
        try:
            response = provider.search(query, limit)
        except Exception as exc:  # provider failure is the fallback signal
            last_error = f"{name} unavailable: {exc}"
            continue
        if isinstance(response, dict) and response.get("success") is not False:
            response = dict(response)
            response["routing"] = {"attempted": attempted, "selected": name}
            return response
        last_error = (
            str(response.get("error") or f"{name} failed")
            if isinstance(response, dict)
            else f"{name} returned an invalid response"
        )

    return {
        "success": False,
        "error": last_error,
        "routing": {"attempted": attempted, "selected": None},
    }


def _result_text(result: Dict[str, Any]) -> str:
    # Providers commonly mirror the same text into ``content`` and
    # ``raw_content``. Count it once so the near-empty threshold reflects the
    # actual page instead of an implementation detail of the provider shape.
    parts: List[str] = []
    for key in ("error", "content", "raw_content"):
        value = str(result.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    return "\n".join(parts)


def _status_code(result: Dict[str, Any]) -> Optional[int]:
    metadata = result.get("metadata")
    value = metadata.get("status_code") if isinstance(metadata, dict) else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_block_signal(result: Dict[str, Any]) -> Optional[str]:
    """Classify actual bot blocks while avoiding false positives in articles."""
    status = _status_code(result)
    if status in {403, 429}:
        return f"HTTP {status}"

    text = _result_text(result)
    lowered = text.lower()
    # Matches both the classic wording ("returned error 403") and provider
    # wrappers such as the crawl4ai plugin ("Crawl4AI returned HTTP 403").
    wrapped = re.search(r"(?:target\s+url\s+)?returned\s+(?:error|http)\s+(403|429)\b", lowered)
    if wrapped:
        return f"wrapped HTTP {wrapped.group(1)}"

    if len(text) <= _NEAR_EMPTY_BLOCK_CHARS:
        for marker in _BLOCK_MARKERS:
            if marker in lowered:
                return marker
    return None


def _extract_is_technical_failure(result: Dict[str, Any]) -> bool:
    """Classify transport failures without turning a valid 404 into escalation."""
    status = _status_code(result)
    if status in {400, 404, 410} or _extract_block_signal(result):
        return False
    error = str(result.get("error") or "").lower()
    return bool(error and any(token in error for token in (
        "timeout", "connection", "temporarily", "network", "dns", "reset",
    )))


def _is_hard_interstitial(result: Dict[str, Any]) -> bool:
    """Detect a rendered challenge page that is too large for near-empty rules."""
    text = _result_text(result)
    if len(text) < _HARD_INTERSTITIAL_MIN_CHARS:
        return False
    lowered = text.lower()
    hits = sum(marker in lowered for marker in _BLOCK_MARKERS)
    return hits >= 2 or (
        "datadome" in lowered and "access is temporarily restricted" in lowered
    )


def _is_usable_extraction(result: Dict[str, Any]) -> bool:
    return bool(
        result
        and not result.get("error")
        and str(result.get("content") or result.get("raw_content") or "").strip()
        and not _extract_block_signal(result)
        and not _is_hard_interstitial(result)
    )


def _set_extraction_routing(
    result: Dict[str, Any], lanes: List[str], *, reason: Optional[str] = None
) -> Dict[str, Any]:
    routed = dict(result)
    routing: Dict[str, Any] = {"lanes": list(lanes), "selected": lanes[-1]}
    if reason:
        routing["escalation_reason"] = reason
    routed["routing"] = routing
    return routed


def _append_capped_lane_log(url: str, lanes: List[str], outcome: str) -> None:
    """Write a bounded audit record for the paid Browser Use escalation."""
    log_path = Path(get_hermes_home()) / "logs" / "lane_escalation.log"
    line = (
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"url={_redact_url_for_logs(url)} lanes={','.join(lanes)} outcome={outcome}\n"
    )
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.stat().st_size > 256_000:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-128_000:]
            log_path.write_text(tail, encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError as exc:
        logger.debug("Could not write web lane escalation log: %s", exc)


def _redact_url_for_logs(url: str) -> str:
    """Drop userinfo, query data, and fragments from a URL before logging."""
    try:
        parsed = urlsplit(str(url or "").strip())
        if not parsed.scheme or not parsed.netloc:
            return "<redacted-url>"
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<redacted-url>"


def _cloud_sensitive_url_reason(url: str) -> Optional[str]:
    """Reject credentials before a URL leaves the local/VPS trust boundary."""
    try:
        parsed = urlsplit(str(url or "").strip())
    except (TypeError, ValueError):
        return "URL could not be parsed safely"
    if parsed.username or parsed.password:
        return "URL contains user credentials"
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(unquote(url)):
        return "URL contains what appears to be an API key or token"
    sensitive_key = sensitive_query_param_name(url)
    if sensitive_key:
        return f"credential-like query parameter ({sensitive_key})"
    if parsed.fragment:
        fragment_url = f"https://fragment.invalid/?{parsed.fragment}"
        sensitive_key = sensitive_query_param_name(fragment_url)
        if sensitive_key:
            return f"credential-like fragment parameter ({sensitive_key})"
    return None


async def _navigation_block_reason(
    url: str,
    lane: str,
    *,
    check_cloud_credentials: bool = True,
) -> Optional[str]:
    """Return a fail-closed reason before a browser may dispatch ``url``."""
    if lane == "browser_use_cloud" and check_cloud_credentials:
        sensitive_reason = _cloud_sensitive_url_reason(url)
        if sensitive_reason:
            return sensitive_reason

    try:
        policy = check_website_access(_redact_url_for_logs(url))
    except Exception:
        return "website policy validation failed"
    if policy:
        return str(policy.get("message") or "blocked by website policy")

    try:
        is_safe = await asyncio.wait_for(
            async_is_safe_url(url), timeout=_URL_POLICY_TIMEOUT_S
        )
        if not is_safe:
            return "URL targets a private or internal network address"
    except Exception:
        return "URL safety validation failed"
    return None


class _CDPRequestBlocked(RuntimeError):
    def __init__(self, url: str, reason: str):
        super().__init__(reason)
        self.url = url
        self.reason = reason


async def _dispatch_cdp_event(
    on_event: Optional[Callable[[Dict[str, Any]], Any]],
    payload: Dict[str, Any],
    deadline: float,
    timeout_message: str,
) -> None:
    if on_event is None:
        return
    handled = on_event(payload)
    if not inspect.isawaitable(handled):
        return
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError(timeout_message)
    try:
        await asyncio.wait_for(handled, timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(timeout_message) from exc


async def _cdp_call(
    ws: Any,
    request_ids: Any,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    *,
    timeout_s: float = _CDP_CALL_TIMEOUT_S,
    on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    timeout_message = f"CDP {method} timed out"
    deadline = asyncio.get_running_loop().time() + timeout_s
    request_id = next(request_ids)
    message: Dict[str, Any] = {"id": request_id, "method": method}
    if params:
        message["params"] = params
    if session_id:
        message["sessionId"] = session_id
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError(timeout_message)
    try:
        await asyncio.wait_for(ws.send(json.dumps(message)), timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(timeout_message) from exc
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(timeout_message)
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(timeout_message) from exc
        payload = json.loads(raw)
        if payload.get("id") != request_id:
            await _dispatch_cdp_event(on_event, payload, deadline, timeout_message)
            continue
        if "error" in payload:
            raise RuntimeError(str(payload["error"].get("message") or payload["error"]))
        return payload.get("result") or {}


async def _cdp_wait_for_load(
    ws: Any,
    session_id: str,
    timeout_s: float = _CDP_LOAD_TIMEOUT_S,
    on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> None:
    """Wait for an initial load event, but do not reject JS-heavy sites on timeout."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        try:
            payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
        except asyncio.TimeoutError:
            return
        await _dispatch_cdp_event(
            on_event, payload, deadline, "CDP page load handling timed out"
        )
        if payload.get("sessionId") == session_id and payload.get("method") == "Page.loadEventFired":
            return


async def _send_cdp_command(
    ws: Any,
    request_ids: Any,
    method: str,
    params: Dict[str, Any],
    session_id: str,
    timeout_s: float = _CDP_CLEANUP_TIMEOUT_S,
) -> None:
    """Send a CDP command without recursively consuming the outer call's reply."""
    message = {
        "id": next(request_ids),
        "method": method,
        "params": params,
        "sessionId": session_id,
    }
    try:
        await asyncio.wait_for(ws.send(json.dumps(message)), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"CDP {method} send timed out") from exc


async def _extract_via_cdp(url: str, cdp_url: str, lane: str) -> Dict[str, Any]:
    """Read title/body text from a temporary target without reading browser secrets."""
    if not cdp_url.startswith(("ws://", "wss://")):
        return {"url": url, "error": f"{lane} CDP endpoint is unavailable"}
    initial_block = await _navigation_block_reason(url, lane)
    if initial_block:
        return {"url": url, "error": f"{lane} refused URL before browser dispatch: {initial_block}"}
    try:
        import websockets
    except ImportError:
        return {"url": url, "error": "websockets dependency is unavailable"}

    try:
        async with websockets.connect(
            cdp_url, open_timeout=10, close_timeout=5, max_size=4_000_000
        ) as ws:
            request_ids = itertools.count(1)
            target_id = ""
            try:
                created = await _cdp_call(
                    ws, request_ids, "Target.createTarget", {"url": "about:blank"}
                )
                target_id = str(created.get("targetId") or "")
                if not target_id:
                    raise RuntimeError("Target.createTarget returned no target id")
                attached = await _cdp_call(
                    ws, request_ids, "Target.attachToTarget", {"targetId": target_id, "flatten": True}
                )
                session_id = str(attached.get("sessionId") or "")
                if not session_id:
                    raise RuntimeError("Target.attachToTarget returned no session id")

                load_event_seen = False
                main_document_frame_id = ""
                main_document_seen = False

                async def handle_fetch_event(payload: Dict[str, Any]) -> None:
                    nonlocal load_event_seen, main_document_frame_id, main_document_seen
                    if (
                        payload.get("sessionId") == session_id
                        and payload.get("method") == "Page.loadEventFired"
                    ):
                        load_event_seen = True
                        return
                    if (
                        payload.get("sessionId") != session_id
                        or payload.get("method") != "Fetch.requestPaused"
                    ):
                        return
                    event_params = payload.get("params") or {}
                    request_id = str(event_params.get("requestId") or "")
                    request = event_params.get("request") or {}
                    request_url = str(request.get("url") or "")
                    if not request_id or not request_url:
                        raise RuntimeError("Fetch.requestPaused omitted request metadata")

                    resource_type = str(event_params.get("resourceType") or "")
                    frame_id = str(event_params.get("frameId") or "")
                    if resource_type == "Document" and not main_document_seen:
                        main_document_seen = True
                        main_document_frame_id = frame_id
                    is_main_document = (
                        resource_type == "Document"
                        and (
                            not frame_id
                            or frame_id == main_document_frame_id
                        )
                    )
                    reason = await _navigation_block_reason(
                        request_url,
                        lane,
                        check_cloud_credentials=is_main_document,
                    )
                    if reason:
                        await _send_cdp_command(
                            ws,
                            request_ids,
                            "Fetch.failRequest",
                            {"requestId": request_id, "errorReason": "BlockedByClient"},
                            session_id,
                        )
                        raise _CDPRequestBlocked(request_url, reason)
                    await _send_cdp_command(
                        ws,
                        request_ids,
                        "Fetch.continueRequest",
                        {"requestId": request_id},
                        session_id,
                    )

                await _cdp_call(ws, request_ids, "Page.enable", session_id=session_id)
                await _cdp_call(
                    ws,
                    request_ids,
                    "Fetch.enable",
                    {
                        "patterns": [
                            {"urlPattern": "http://*", "requestStage": "Request"},
                            {"urlPattern": "https://*", "requestStage": "Request"},
                        ]
                    },
                    session_id,
                )
                await _cdp_call(
                    ws,
                    request_ids,
                    "Page.navigate",
                    {"url": url},
                    session_id,
                    on_event=handle_fetch_event,
                )
                if not load_event_seen:
                    await _cdp_wait_for_load(
                        ws,
                        session_id,
                        on_event=handle_fetch_event,
                    )

                location = await _cdp_call(
                    ws, request_ids, "Runtime.evaluate",
                    {"expression": "location.href", "returnByValue": True}, session_id,
                    on_event=handle_fetch_event,
                )
                final_url = str(location.get("result", {}).get("value") or url)
                final_block = await _navigation_block_reason(final_url, lane)
                if final_block:
                    return {
                        "url": url,
                        "error": f"{lane} redirect was rejected: {final_block}",
                    }

                page = await _cdp_call(
                    ws, request_ids, "Runtime.evaluate",
                    {
                        "expression": "JSON.stringify({title:document.title||'',content:document.body?.innerText||''})",
                        "returnByValue": True,
                    }, session_id,
                    on_event=handle_fetch_event,
                )
                value = page.get("result", {}).get("value") or "{}"
                extracted = json.loads(value) if isinstance(value, str) else value
                if not isinstance(extracted, dict):
                    raise RuntimeError("CDP extraction returned invalid page data")
                content = str(extracted.get("content") or "")
                return {
                    "url": final_url,
                    "title": str(extracted.get("title") or ""),
                    "content": content,
                    "raw_content": content,
                    "metadata": {"lane": lane},
                }
            finally:
                if target_id:
                    try:
                        await _cdp_call(
                            ws,
                            request_ids,
                            "Target.closeTarget",
                            {"targetId": target_id},
                            timeout_s=_CDP_CLEANUP_TIMEOUT_S,
                        )
                    except Exception:  # best-effort tab cleanup only
                        pass
    except _CDPRequestBlocked as exc:
        logger.info(
            "%s blocked a browser request before dispatch: url=%s reason=%s",
            lane,
            _redact_url_for_logs(exc.url),
            exc.reason,
        )
        return {
            "url": url,
            "error": f"{lane} blocked a browser request before dispatch: {exc.reason}",
        }
    except Exception as exc:  # lane error is returned for deterministic fallback
        logger.info(
            "%s extract failed for %s: %s",
            lane,
            _redact_url_for_logs(url),
            type(exc).__name__,
        )
        return {"url": url, "error": f"{lane} extraction failed: {type(exc).__name__}"}


async def _extract_via_home_chrome_cdp(url: str) -> Dict[str, Any]:
    """Refuse automatic navigation of Darin's authenticated Home Chrome.

    The VPS can validate DNS and redirect URLs, but Windows Chrome performs
    the actual DNS lookup and connection. That split cannot safely defeat DNS
    rebinding without a browser-side egress proxy or equivalent connect-time
    control. Explicit browser tools remain available through the CDP bridge.
    """
    return {
        "url": url,
        "error": (
            "Automatic Home Chrome extraction is disabled: use an explicit "
            "browser workflow for authenticated or sensitive pages"
        ),
    }


async def _close_browser_use_session(provider: Any, session_id: str) -> bool:
    """Retry normal close, then attempt bounded emergency cleanup."""
    for _attempt in range(2):
        try:
            closed = await asyncio.wait_for(
                asyncio.to_thread(provider.close_session, session_id),
                timeout=_BROWSER_USE_CLEANUP_TIMEOUT_S,
            )
        except Exception:
            closed = False
        if closed is True:
            return True

    try:
        await asyncio.wait_for(
            asyncio.to_thread(provider.emergency_cleanup, session_id),
            timeout=_BROWSER_USE_CLEANUP_TIMEOUT_S,
        )
    except Exception:
        pass
    return False


async def _extract_via_browser_use_cloud(url: str, lanes: List[str]) -> Dict[str, Any]:
    """Create and always close a one-off Browser Use session for a public page."""
    session: Optional[Dict[str, Any]] = None
    provider: Any = None
    result: Dict[str, Any] = {"url": url, "error": "Browser Use is not configured"}
    cleanup_failed = False
    try:
        preflight_block = await _navigation_block_reason(url, "browser_use_cloud")
        if preflight_block:
            result = {
                "url": url,
                "error": f"Browser Use automatic escalation refused: {preflight_block}",
            }
        else:
            from tools.browser_tool_cloud import _get_cloud_provider

            provider = _get_cloud_provider()
            provider_name = str(getattr(provider, "name", "") or "")
            if provider is None:
                result = {
                    "url": url,
                    "error": "Browser Use cloud escalation is disabled or unavailable",
                }
            elif provider_name != "browser-use":
                result = {
                    "url": url,
                    "error": (
                        f"Browser Use cloud escalation is not allowed by the active "
                        f"provider policy ({provider_name or 'unknown'})"
                    ),
                }
            elif not provider.is_available():
                result = {"url": url, "error": "Browser Use is not configured"}
            else:
                create_task = asyncio.create_task(
                    asyncio.to_thread(
                        provider.create_session,
                        f"web-extract-{uuid.uuid4().hex}",
                    )
                )
                try:
                    created_session = await asyncio.shield(create_task)
                except asyncio.CancelledError:
                    try:
                        late_session = await asyncio.shield(create_task)
                        if isinstance(late_session, dict):
                            session = late_session
                    except Exception:
                        pass
                    raise
                if not isinstance(created_session, dict):
                    raise RuntimeError("Browser Use returned invalid session metadata")
                session = created_session
                if not session.get("bb_session_id"):
                    raise RuntimeError("Browser Use returned no session id")
                result = await _extract_via_cdp(
                    url,
                    str(session.get("cdp_url") or ""),
                    "browser_use_cloud",
                )
    except Exception as exc:
        result = {"url": url, "error": f"Browser Use extraction failed: {type(exc).__name__}"}
    finally:
        if isinstance(session, dict) and provider and session.get("bb_session_id"):
            cleanup_failed = not await _close_browser_use_session(
                provider, str(session["bb_session_id"])
            )
            if cleanup_failed:
                logger.warning(
                    "Browser Use session cleanup could not be confirmed for %s",
                    _redact_url_for_logs(url),
                )

    if cleanup_failed:
        prior_error = str(result.get("error") or "").strip()
        result = dict(result)
        result["error"] = "Browser Use session cleanup failed"
        if prior_error:
            result["error"] += f"; extraction also failed: {prior_error}"

    _append_capped_lane_log(url, lanes, "success" if _is_usable_extraction(result) else "failed")
    return result


async def _extract_with_jina_escalation(provider: Any, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
    """Primary backend → Home Chrome CDP → Browser Use for classified public
    anti-bot blocks and lane failures.

    The first lane is the ACTUAL primary backend (provider.name), not a
    hardcoded name: the escalation chain must serve whichever backend is
    configured (Jina, Crawl4AI, Firecrawl, ...) — a lane dying must never
    take extract down with no fallback.
    """
    if inspect.iscoroutinefunction(provider.extract):
        primary = provider.extract(urls, **kwargs)
    else:
        primary = await asyncio.to_thread(provider.extract, urls, **kwargs)
    primary_results = await primary if inspect.isawaitable(primary) else primary
    results: List[Dict[str, Any]] = []
    for original in primary_results:
        result = dict(original)
        lanes = [provider.name or "jina"]
        signal = _extract_block_signal(result)
        if not signal and _is_hard_interstitial(result):
            signal = "hard anti-bot interstitial"
        if not signal and _status_code(result) == 402:
            # 402 = lane unavailable (credits/access), not a page property —
            # the page is probably fine, so the next lanes deserve a try.
            signal = "HTTP 402 (lane unavailable)"
        if not signal and _extract_is_technical_failure(result):
            # transport failures are not anti-bot blocks: do not spend the
            # CDP or paid cloud lanes on them — record the routing and keep
            # the primary result.
            results.append(_set_extraction_routing(result, lanes, reason="technical failure"))
            continue
        if not signal:
            results.append(_set_extraction_routing(result, lanes))
            continue

        lanes.append("home_chrome_cdp")
        chrome = await _extract_via_home_chrome_cdp(str(result.get("url") or ""))
        if _is_usable_extraction(chrome):
            results.append(_set_extraction_routing(chrome, lanes, reason=signal))
            continue

        lanes.append("browser_use_cloud")
        cloud = await _extract_via_browser_use_cloud(str(result.get("url") or ""), lanes)
        if _is_usable_extraction(cloud):
            results.append(_set_extraction_routing(cloud, lanes, reason=signal))
            continue

        # Preserve the original reader result; it often contains the clearest
        # server-side error, while routing records every attempted lane.
        results.append(_set_extraction_routing(result, lanes, reason=signal))
    return results
