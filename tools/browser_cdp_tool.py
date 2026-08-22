#!/usr/bin/env python3
"""
Raw Chrome DevTools Protocol (CDP) passthrough tool.

Exposes a single tool, ``browser_cdp``, that sends arbitrary CDP commands to
the browser's DevTools WebSocket endpoint.  Works when a CDP URL is
configured — either via ``/browser connect`` (sets ``BROWSER_CDP_URL``) or
``browser.cdp_url`` in ``config.yaml`` — or when a CDP-backed cloud provider
session is active.

This is the escape hatch for browser operations not covered by the main
browser tool surface (``browser_navigate``, ``browser_click``,
``browser_console``, etc.) — handling native dialogs, iframe-scoped
evaluation, cookie/network control, low-level tab management, etc.

Method reference: https://chromedevtools.github.io/devtools-protocol/
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error
from tools.browser_extension_router import routed_browser_handler

logger = logging.getLogger(__name__)

CDP_DOCS_URL = "https://chromedevtools.github.io/devtools-protocol/"

_CDP_PRIVATE_PAGE_ALLOWED_METHODS = {
    # Browser/target inspection does not read the current page body, cookies,
    # DOM, storage, or screenshots. Keep these working so the model can list
    # tabs or navigate away from a blocked page.
    # Page.reload is intentionally excluded: reloading an already-private
    # page re-requests private/internal content. Use Page.navigate to leave.
    "Browser.getVersion",
    "Target.getTargets",
    "Target.attachToTarget",
    "Target.detachFromTarget",
    "Page.navigate",
    "Page.stopLoading",
}

_CDP_TARGET_NAVIGATION_METHODS = {"Page.navigate", "Page.reload"}


def _find_frame_in_tree(
    frame_tree: Dict[str, Any], frame_id: str
) -> Optional[Dict[str, Any]]:
    frame = frame_tree.get("frame", {})
    if str(frame.get("id") or "") == frame_id:
        return frame
    for child in frame_tree.get("childFrames", []) or []:
        found = _find_frame_in_tree(child, frame_id)
        if found is not None:
            return found
    return None


def _redact_cdp_output(value: Any) -> Any:
    """Redact browser-originated CDP result data before returning it."""
    from agent.redact import redact_sensitive_text

    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, list):
        return [_redact_cdp_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_cdp_output(item) for item in value)
    if isinstance(value, dict):
        redacted: Dict[Any, Any] = {}
        next_duplicate: Dict[Any, int] = {}
        for key, item in value.items():
            redacted_key = _redact_cdp_output(key) if isinstance(key, str) else key
            candidate = redacted_key
            if candidate in redacted:
                duplicate = next_duplicate.get(redacted_key, 2)
                candidate = f"{redacted_key} ({duplicate})"
                while candidate in redacted:
                    duplicate += 1
                    candidate = f"{redacted_key} ({duplicate})"
                next_duplicate[redacted_key] = duplicate + 1
            else:
                next_duplicate.setdefault(redacted_key, 2)
            redacted[candidate] = _redact_cdp_output(item)
        return redacted
    return value

# ``websockets`` is a direct hermes-agent dependency because the browser CDP
# supervisor and browser_dialog tool import it during tool discovery. Wrap the
# import so a clean error surfaces if an environment is stale or incomplete.
try:
    import websockets
    from websockets.exceptions import WebSocketException

    _WS_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    WebSocketException = Exception  # type: ignore[assignment,misc]
    _WS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Async-from-sync bridge (matches the pattern in homeassistant_tool.py)
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from a sync handler, safe inside or outside a loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def _resolve_cdp_endpoint() -> str:
    """Return the normalized CDP WebSocket URL, or empty string if unavailable.

    Delegates to ``tools.browser_tool._get_cdp_override`` so precedence stays
    consistent with the rest of the browser tool surface:

    1. ``BROWSER_CDP_URL`` env var (live override from ``/browser connect``)
    2. ``browser.cdp_url`` in ``config.yaml``
    """
    try:
        from tools.browser_tool import _get_cdp_override  # type: ignore[import-not-found]

        return (_get_cdp_override() or "").strip()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("browser_cdp: failed to resolve CDP endpoint: %s", exc)
        return ""


def _private_page_guard_error(blocked_url: str, method: str) -> str:
    return tool_error(
        "Blocked: page URL targets a private or internal address "
        f"({blocked_url}). Raw CDP method {method!r} could expose private "
        "page content or state.",
        method=method,
        cdp_docs=CDP_DOCS_URL,
    )


def _private_address_from_candidates(*candidates: Any) -> Optional[str]:
    """Return the first private/always-blocked URL-or-origin among candidates."""
    from tools import browser_tool as bt  # type: ignore[import-not-found]

    for raw in candidates:
        candidate = str(raw or "").strip()
        if not candidate:
            continue
        if bt._is_always_blocked_url(candidate) or not bt._is_safe_url(candidate):  # type: ignore[attr-defined]
            return candidate
    return None


class _BrowserCdpFrameGuardBlocked(Exception):
    """Carry a ready-to-return tool_error JSON across the supervisor loop."""

    def __init__(self, error_json: str):
        self.error_json = error_json
        super().__init__(error_json)


def _browser_cdp_selected_frame_private_guard(
    *,
    task_id: str,
    method: str,
    frame_info: Dict[str, Any],
) -> Optional[str]:
    """Block page-content CDP calls against a private selected OOPIF frame.

    Top-level ``_current_page_private_url`` is not enough for ``frame_id``
    routing: a public parent can embed a private OOPIF, and the supervisor
    dispatches into that child session independently. Navigation/inspection
    allowlisted methods still pass so the model can leave or inspect tabs.

    When the selected frame already has a child ``session_id`` but URL/origin
    metadata is still empty (common briefly after Target.attachedToTarget),
    fail closed for non-allowlisted methods instead of dispatching blind.

    Guard-activation and URL/origin probe failures also fail closed —
    returning ``None`` here would re-open the public-top / private-child
    boundary that this path exists to enforce. Unlike the top-level
    private-page guard (best-effort for local CDP), selected-frame routing
    crosses into a model-chosen OOPIF child session and must not dispatch
    page-content CDP when guard state is unknown.
    """
    if method in _CDP_PRIVATE_PAGE_ALLOWED_METHODS:
        return None

    try:
        from tools import browser_tool as bt  # type: ignore[import-not-found]

        if not bt._eval_ssrf_guard_active(task_id):  # type: ignore[attr-defined]
            return None
    except Exception as exc:  # noqa: BLE001
        # Cannot tell whether the cloud/private-network guard is active;
        # do not fail-open into a selected child session.
        logger.debug(
            "browser_cdp: selected-frame guard activation probe failed: %s",
            exc,
        )
        return tool_error(
            "Blocked: selected-frame SSRF guard activation probe failed; "
            f"raw CDP method {method!r} could expose private page content "
            "or state.",
            method=method,
            cdp_docs=CDP_DOCS_URL,
        )

    try:
        frame_url = str(frame_info.get("url") or "").strip()
        frame_origin = str(frame_info.get("origin") or "").strip()
        blocked = _private_address_from_candidates(frame_url, frame_origin)
        if blocked:
            return _private_page_guard_error(blocked, method)
        # OOPIF session without address metadata: cannot prove the frame is
        # public, so do not fail-open into page-content CDP.
        if frame_info.get("session_id") and not frame_url and not frame_origin:
            return tool_error(
                "Blocked: selected OOPIF frame has no URL/origin metadata "
                f"yet; raw CDP method {method!r} could expose private page "
                "content or state. Retry after browser_snapshot once the "
                "frame has navigated.",
                method=method,
                cdp_docs=CDP_DOCS_URL,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "browser_cdp: selected-frame private-page guard probe failed: %s",
            exc,
        )
        return tool_error(
            "Blocked: selected-frame private-page guard probe failed; "
            f"raw CDP method {method!r} could expose private page content "
            "or state.",
            method=method,
            cdp_docs=CDP_DOCS_URL,
        )
    return None


def _live_selected_frame_info(supervisor: Any, frame_id: str) -> Optional[Dict[str, Any]]:
    """Return the live frame dict from supervisor ``_frames`` under the lock.

    ``_on_frame_navigated`` updates URL/origin in place while preserving the
    child ``cdp_session_id``. Copied snapshot()/to_dict() views can therefore
    go stale between the pre-schedule check and ``_cdp``; the live map is the
    dispatch-time source of truth.
    """
    with supervisor._state_lock:  # type: ignore[attr-defined]
        raw = supervisor._frames.get(frame_id)  # type: ignore[attr-defined]
        if raw is None:
            return None
        return raw.to_dict()


def _selected_frame_from_tree(
    frame_tree: Dict[str, Any], frame_id: str
) -> Dict[str, Any]:
    selected = _find_frame_in_tree(frame_tree, frame_id)
    if selected is None:
        # OOPIF child sessions often expose only the selected frame as root.
        selected = frame_tree.get("frame", {})
    return selected if isinstance(selected, dict) else {}


async def _supervisor_selected_frame_tree_entry(
    *,
    supervisor: Any,
    frame_id: str,
    session_id: str,
    timeout: float,
) -> Dict[str, Any]:
    tree_msg = await supervisor._cdp(  # type: ignore[attr-defined]
        "Page.getFrameTree",
        {},
        session_id=session_id,
        timeout=timeout,
    )
    frame_tree = tree_msg.get("result", {}).get("frameTree", {})
    if not isinstance(frame_tree, dict):
        return {}
    return _selected_frame_from_tree(frame_tree, frame_id)


async def _prepare_supervisor_frame_navigation(
    *,
    supervisor: Any,
    frame_id: str,
    session_id: str,
    method: str,
    timeout: float,
) -> str:
    """Enable page events and capture the pre-dispatch loaderId for commit waits."""
    try:
        await supervisor._cdp(  # type: ignore[attr-defined]
            "Page.enable",
            {},
            session_id=session_id,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise _BrowserCdpFrameGuardBlocked(
            tool_error(
                f"Blocked: Page.enable failed before {method}: {exc}",
                method=method,
                cdp_docs=CDP_DOCS_URL,
            )
        ) from exc

    try:
        selected = await _supervisor_selected_frame_tree_entry(
            supervisor=supervisor,
            frame_id=frame_id,
            session_id=session_id,
            timeout=timeout,
        )
    except _BrowserCdpFrameGuardBlocked:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _BrowserCdpFrameGuardBlocked(
            tool_error(
                f"Blocked: Page.getFrameTree failed before {method}: {exc}",
                method=method,
                cdp_docs=CDP_DOCS_URL,
            )
        ) from exc

    if not selected:
        raise _BrowserCdpFrameGuardBlocked(
            tool_error(
                f"Blocked: selected frame is missing before {method}",
                method=method,
                cdp_docs=CDP_DOCS_URL,
            )
        )
    return str(selected.get("loaderId") or "").strip()


def _supervisor_frame_commit_matched(
    *,
    frame: Dict[str, Any],
    expected_frame_id: str,
    expected_loader_id: str,
    initial_loader_id: str,
) -> bool:
    frame_entry_id = str(frame.get("id") or frame.get("frame_id") or "").strip()
    if expected_frame_id and frame_entry_id and frame_entry_id != expected_frame_id:
        return False
    loader_id = str(frame.get("loaderId") or "").strip()
    if expected_loader_id:
        return loader_id == expected_loader_id
    return bool(loader_id and loader_id != initial_loader_id)


async def _wait_supervisor_frame_commit(
    *,
    supervisor: Any,
    frame_id: str,
    session_id: str,
    method: str,
    expected_frame_id: str,
    expected_loader_id: str,
    initial_loader_id: str,
    timeout: float,
) -> Dict[str, Any]:
    """Poll ``Page.getFrameTree`` until the navigate/reload loader commits.

    ``Page.navigate`` can return before a redirect lands. Matching the
    target-scoped path, wait for the new ``loaderId`` (or a changed loader on
    reload) before treating the landing URL as authoritative.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise _BrowserCdpFrameGuardBlocked(
                tool_error(
                    f"Blocked: timed out waiting for frame commit after {method}",
                    method=method,
                    cdp_docs=CDP_DOCS_URL,
                )
            )
        try:
            selected = await _supervisor_selected_frame_tree_entry(
                supervisor=supervisor,
                frame_id=frame_id,
                session_id=session_id,
                timeout=max(0.05, remaining),
            )
        except _BrowserCdpFrameGuardBlocked:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "browser_cdp: Page.getFrameTree while waiting for frame %s commit: %s",
                method,
                exc,
            )
            await asyncio.sleep(0.05)
            continue

        if _supervisor_frame_commit_matched(
            frame=selected,
            expected_frame_id=expected_frame_id,
            expected_loader_id=expected_loader_id,
            initial_loader_id=initial_loader_id,
        ):
            return selected
        await asyncio.sleep(0.05)


async def _reset_supervisor_frame_to_blank(
    *,
    supervisor: Any,
    frame_id: str,
    session_id: str,
    timeout: float,
) -> Optional[str]:
    """Navigate the child session to about:blank and wait for that commit."""
    try:
        blank_msg = await supervisor._cdp(  # type: ignore[attr-defined]
            "Page.navigate",
            {"url": "about:blank"},
            session_id=session_id,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return f"failed to reset frame to about:blank: {exc}"

    blank_result = blank_msg.get("result", {}) if isinstance(blank_msg, dict) else {}
    blank_frame_id = str(
        (blank_result or {}).get("frameId") or frame_id
    ).strip() or frame_id
    blank_loader_id = str((blank_result or {}).get("loaderId") or "").strip()
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return "timed out waiting for about:blank reset to commit"
        try:
            selected = await _supervisor_selected_frame_tree_entry(
                supervisor=supervisor,
                frame_id=blank_frame_id,
                session_id=session_id,
                timeout=max(0.05, remaining),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "browser_cdp: Page.getFrameTree while waiting for about:blank: %s",
                exc,
            )
            await asyncio.sleep(0.05)
            continue

        url = str(selected.get("url") or "").strip()
        loader_id = str(selected.get("loaderId") or "").strip()
        if url == "about:blank" and (
            not blank_loader_id or loader_id == blank_loader_id
        ):
            return None
        await asyncio.sleep(0.05)


async def _revalidate_supervisor_frame_navigation(
    *,
    supervisor: Any,
    task_id: str,
    frame_id: str,
    session_id: str,
    method: str,
    timeout: float,
    nav_result: Optional[Dict[str, Any]] = None,
    initial_loader_id: str = "",
) -> None:
    """Fail closed when frame_id navigate/reload lands on a private address.

    ``Page.navigate`` remains allowlisted so a private OOPIF can be left, but
    public-to-private redirects (and reload of a page that becomes private)
    must not keep the child session on an internal URL. Mirror the target-scoped
    post-check: wait for the navigation commit, inspect live frame metadata +
    ``Page.getFrameTree``, then reset to ``about:blank`` when blocked.
    """
    try:
        from tools import browser_tool as bt  # type: ignore[import-not-found]

        if not bt._eval_ssrf_guard_active(task_id):  # type: ignore[attr-defined]
            return
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "browser_cdp: frame navigation guard activation probe failed: %s",
            exc,
        )
        raise _BrowserCdpFrameGuardBlocked(
            tool_error(
                "Blocked: selected-frame SSRF guard activation probe failed after "
                f"{method}; raw CDP navigation could expose private page content "
                "or state.",
                method=method,
                cdp_docs=CDP_DOCS_URL,
            )
        ) from exc

    nav_payload = nav_result if isinstance(nav_result, dict) else {}
    expected_frame_id = str(nav_payload.get("frameId") or frame_id).strip() or frame_id
    expected_loader_id = str(nav_payload.get("loaderId") or "").strip()
    # Cross-document Page.navigate returns a loaderId. Page.reload has no
    # result payload, so compare against the pre-dispatch loader. Same-document
    # Page.navigate has no loaderId and is already committed on return.
    commit_required = method == "Page.reload" or bool(expected_loader_id)

    commit_url = ""
    if commit_required:
        committed = await _wait_supervisor_frame_commit(
            supervisor=supervisor,
            frame_id=frame_id,
            session_id=session_id,
            method=method,
            expected_frame_id=expected_frame_id,
            expected_loader_id=expected_loader_id,
            initial_loader_id=initial_loader_id,
            timeout=timeout,
        )
        commit_url = str(committed.get("url") or "").strip()
        if not commit_url:
            raise _BrowserCdpFrameGuardBlocked(
                tool_error(
                    f"Blocked: committed frame has no URL metadata after {method}",
                    method=method,
                    cdp_docs=CDP_DOCS_URL,
                )
            )
    else:
        try:
            selected = await _supervisor_selected_frame_tree_entry(
                supervisor=supervisor,
                frame_id=frame_id,
                session_id=session_id,
                timeout=timeout,
            )
            commit_url = str(selected.get("url") or "").strip()
        except _BrowserCdpFrameGuardBlocked:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "browser_cdp: Page.getFrameTree after frame %s failed: %s",
                method,
                exc,
            )

    live_info = _live_selected_frame_info(supervisor, frame_id)
    live_url = str((live_info or {}).get("url") or "").strip()
    live_origin = str((live_info or {}).get("origin") or "").strip()

    try:
        blocked = _private_address_from_candidates(live_url, live_origin, commit_url)
    except Exception as exc:  # noqa: BLE001
        raise _BrowserCdpFrameGuardBlocked(
            tool_error(
                "Blocked: selected-frame private-page guard probe failed after "
                f"{method}; raw CDP navigation could expose private page content "
                "or state.",
                method=method,
                cdp_docs=CDP_DOCS_URL,
            )
        ) from exc

    if not live_url and not live_origin and not commit_url:
        raise _BrowserCdpFrameGuardBlocked(
            tool_error(
                "Blocked: selected OOPIF frame has no URL/origin metadata after "
                f"{method}; raw CDP navigation could expose private page content "
                "or state.",
                method=method,
                cdp_docs=CDP_DOCS_URL,
            )
        )

    if not blocked:
        return

    blank_error = await _reset_supervisor_frame_to_blank(
        supervisor=supervisor,
        frame_id=expected_frame_id,
        session_id=session_id,
        timeout=timeout,
    )
    reset_status = (
        f"; {blank_error}" if blank_error else "; frame reset to about:blank"
    )
    raise _BrowserCdpFrameGuardBlocked(
        tool_error(
            "Blocked: frame navigation landed on a private or internal address "
            f"({blocked}){reset_status}",
            method=method,
            cdp_docs=CDP_DOCS_URL,
        )
    )


def _browser_cdp_private_guard(
    *,
    task_id: str,
    method: str,
    params: Dict[str, Any],
) -> Optional[str]:
    """Apply the browser SSRF/private-page guard to raw CDP calls.

    ``browser_cdp`` is intentionally an escape hatch, but it still shares the
    same cloud/private-network boundary as ``browser_snapshot``,
    ``browser_console`` and ``browser_eval``.  If a cloud browser has landed on
    a private/internal URL (for example via a prior eval navigation), raw CDP
    calls like ``Runtime.evaluate`` or ``DOM.getDocument`` must not become the
    sibling bypass for the guarded browser tools.
    """
    try:
        from tools import browser_tool as bt  # type: ignore[import-not-found]

        if not bt._eval_ssrf_guard_active(task_id):  # type: ignore[attr-defined]
            return None

        if method == "Page.navigate":
            target_url = str((params or {}).get("url") or "").strip()
            if target_url and (
                bt._is_always_blocked_url(target_url)  # type: ignore[attr-defined]
                or not bt._is_safe_url(target_url)  # type: ignore[attr-defined]
            ):
                return tool_error(
                    "Blocked: CDP Page.navigate target is a private or "
                    f"internal address ({target_url}).",
                    method=method,
                    cdp_docs=CDP_DOCS_URL,
                )

        if method == "Runtime.evaluate":
            expression = str((params or {}).get("expression") or "")
            blocked_literal = bt._expression_targets_private_url(expression)  # type: ignore[attr-defined]
            if blocked_literal:
                return tool_error(
                    "Blocked: CDP Runtime.evaluate expression targets a "
                    f"private or internal address ({blocked_literal}).",
                    method=method,
                    cdp_docs=CDP_DOCS_URL,
                )

        if method not in _CDP_PRIVATE_PAGE_ALLOWED_METHODS:
            blocked_url = bt._current_page_private_url(task_id)  # type: ignore[attr-defined]
            if blocked_url:
                return _private_page_guard_error(blocked_url, method)
    except Exception as exc:  # noqa: BLE001
        # Match the existing browser guards' posture: guard probes are
        # best-effort and should not break local/custom CDP workflows.
        logger.debug("browser_cdp: private-page guard probe failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Core CDP call
# ---------------------------------------------------------------------------


async def _cdp_call(
    ws_url: str,
    method: str,
    params: Dict[str, Any],
    target_id: Optional[str],
    timeout: float,
    guard_selected_target_url: bool = False,
    guard_navigation_result_url: bool = False,
) -> Dict[str, Any]:
    """Make a single CDP call, optionally attaching to a target first.

    When ``target_id`` is provided, we call ``Target.attachToTarget`` with
    ``flatten=True`` to multiplex a page-level session over the same
    browser-level WebSocket, then send ``method`` with that ``sessionId``.
    When ``target_id`` is None, ``method`` is sent at browser level — which
    works for ``Target.*``, ``Browser.*``, ``Storage.*`` and a few other
    globally-scoped domains.
    """
    assert websockets is not None  # guarded by _WS_AVAILABLE at call-site

    async with websockets.connect(
        ws_url,
        max_size=None,  # CDP responses (e.g. DOM.getDocument) can be large
        open_timeout=timeout,
        close_timeout=5,
        ping_interval=None,  # CDP server doesn't expect pings
    ) as ws:
        next_id = 1
        session_id: Optional[str] = None

        # --- Step 1: attach to target if requested ---
        if target_id:
            targets_id = next_id
            next_id += 1
            await ws.send(
                json.dumps({"id": targets_id, "method": "Target.getTargets", "params": {}})
            )
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out resolving target {target_id} before attach")
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                msg = json.loads(raw)
                if msg.get("id") != targets_id:
                    continue
                if "error" in msg:
                    raise RuntimeError(
                        f"Target.getTargets failed before attach: {msg['error']}"
                    )
                target_info = next(
                    (
                        info
                        for info in msg.get("result", {}).get("targetInfos", [])
                        if info.get("targetId") == target_id
                    ),
                    None,
                )
                if target_info is None:
                    raise RuntimeError(f"Blocked: selected target {target_id!r} was not found")
                target_type = str(target_info.get("type") or "").strip()
                target_url = str(target_info.get("url") or "").strip()
                if target_type != "page":
                    raise RuntimeError(
                        "Blocked: target_id is only valid for a top-level page target; "
                        f"selected target type was {target_type or 'unknown'!r}"
                    )
                if not target_url:
                    raise RuntimeError(
                        "Blocked: selected target has no URL metadata; refusing to attach"
                    )
                if guard_selected_target_url:
                    try:
                        blocked_target = _private_address_from_candidates(target_url)
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            "Blocked: selected-target private-page guard probe failed"
                        ) from exc
                    if blocked_target:
                        raise RuntimeError(
                            "Blocked: selected target is a private or internal address "
                            f"({blocked_target})"
                        )
                break

            attach_id = next_id
            next_id += 1
            await ws.send(
                json.dumps(
                    {
                        "id": attach_id,
                        "method": "Target.attachToTarget",
                        "params": {"targetId": target_id, "flatten": True},
                    }
                )
            )
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out attaching to target {target_id}"
                    )
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                msg = json.loads(raw)
                if msg.get("id") == attach_id:
                    if "error" in msg:
                        raise RuntimeError(
                            f"Target.attachToTarget failed: {msg['error']}"
                        )
                    session_id = msg.get("result", {}).get("sessionId")
                    if not session_id:
                        raise RuntimeError(
                            "Target.attachToTarget did not return a sessionId"
                        )
                    break
                # Ignore events (messages without "id") while waiting

            if guard_selected_target_url:
                frame_tree_id = next_id
                next_id += 1
                await ws.send(
                    json.dumps(
                        {
                            "id": frame_tree_id,
                            "method": "Page.getFrameTree",
                            "params": {},
                            "sessionId": session_id,
                        }
                    )
                )
                deadline = asyncio.get_running_loop().time() + timeout
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Timed out revalidating target {target_id} after attach"
                        )
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    msg = json.loads(raw)
                    if msg.get("id") != frame_tree_id:
                        continue
                    if "error" in msg:
                        raise RuntimeError(
                            "Blocked: Page.getFrameTree failed while revalidating "
                            f"selected target: {msg['error']}"
                        )
                    live_url = str(
                        msg.get("result", {})
                        .get("frameTree", {})
                        .get("frame", {})
                        .get("url")
                        or ""
                    ).strip()
                    if not live_url:
                        raise RuntimeError(
                            "Blocked: attached target has no live URL metadata"
                        )
                    try:
                        blocked_live_target = _private_address_from_candidates(live_url)
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            "Blocked: attached-target private-page guard probe failed"
                        ) from exc
                    if blocked_live_target:
                        raise RuntimeError(
                            "Blocked: attached target navigated to a private or internal "
                            f"address ({blocked_live_target})"
                        )
                    break

            navigation_frame_id = ""
            initial_loader_id = ""
            if guard_navigation_result_url:
                enable_id = next_id
                next_id += 1
                await ws.send(
                    json.dumps(
                        {
                            "id": enable_id,
                            "method": "Page.enable",
                            "params": {},
                            "sessionId": session_id,
                        }
                    )
                )
                deadline = asyncio.get_running_loop().time() + timeout
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Timed out enabling navigation events for target {target_id}"
                        )
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    msg = json.loads(raw)
                    if msg.get("id") != enable_id:
                        continue
                    if "error" in msg:
                        raise RuntimeError(
                            f"Blocked: Page.enable failed before {method}: {msg['error']}"
                        )
                    break

                initial_tree_id = next_id
                next_id += 1
                await ws.send(
                    json.dumps(
                        {
                            "id": initial_tree_id,
                            "method": "Page.getFrameTree",
                            "params": {},
                            "sessionId": session_id,
                        }
                    )
                )
                deadline = asyncio.get_running_loop().time() + timeout
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Timed out resolving target {target_id} before {method}"
                        )
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    msg = json.loads(raw)
                    if msg.get("id") != initial_tree_id:
                        continue
                    if "error" in msg:
                        raise RuntimeError(
                            f"Blocked: Page.getFrameTree failed before {method}: {msg['error']}"
                        )
                    initial_tree = msg.get("result", {}).get("frameTree", {})
                    top_frame_id = str(
                        initial_tree.get("frame", {}).get("id") or ""
                    ).strip()
                    navigation_frame_id = str(
                        (params or {}).get("frameId") or top_frame_id
                    ).strip()
                    initial_frame = _find_frame_in_tree(
                        initial_tree, navigation_frame_id
                    )
                    if initial_frame is None:
                        raise RuntimeError(
                            f"Blocked: selected frame is missing before {method}"
                        )
                    initial_loader_id = str(
                        initial_frame.get("loaderId") or ""
                    ).strip()
                    break

        # --- Step 2: dispatch the real method ---
        call_id = next_id
        next_id += 1
        req: Dict[str, Any] = {
            "id": call_id,
            "method": method,
            "params": params or {},
        }
        if session_id:
            req["sessionId"] = session_id
        await ws.send(json.dumps(req))

        deadline = asyncio.get_running_loop().time() + timeout
        navigation_events: list[Dict[str, Any]] = []
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for response to {method}"
                )
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if msg.get("method") == "Page.frameNavigated":
                navigation_events.append(msg)
            if msg.get("id") == call_id:
                if "error" in msg:
                    raise RuntimeError(f"CDP error: {msg['error']}")
                result = msg.get("result", {})
                break
            # Ignore events / out-of-order responses

        if guard_navigation_result_url and session_id:
            expected_frame_id = str(result.get("frameId") or navigation_frame_id).strip()
            expected_loader_id = str(result.get("loaderId") or "").strip()

            def _matching_commit(event: Dict[str, Any]) -> bool:
                frame = event.get("params", {}).get("frame", {})
                if str(frame.get("id") or "") != expected_frame_id:
                    return False
                event_loader_id = str(frame.get("loaderId") or "")
                if expected_loader_id:
                    return event_loader_id == expected_loader_id
                return bool(event_loader_id and event_loader_id != initial_loader_id)

            commit_event = next(
                (event for event in navigation_events if _matching_commit(event)),
                None,
            )
            # Cross-document Page.navigate returns a loaderId. Page.reload has
            # no result payload, so compare the top frame's new loader against
            # the one captured before dispatch. Same-document Page.navigate
            # has no loaderId and is already committed when the command returns.
            commit_required = method == "Page.reload" or bool(expected_loader_id)
            if commit_required and commit_event is None:
                deadline = asyncio.get_running_loop().time() + timeout
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Timed out waiting for target {target_id} to commit after {method}"
                        )
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    event = json.loads(raw)
                    if event.get("method") == "Page.frameNavigated" and _matching_commit(
                        event
                    ):
                        commit_event = event
                        break

            blocked_commit_url: Optional[str] = None
            if commit_event is not None:
                commit_url = str(
                    commit_event.get("params", {}).get("frame", {}).get("url") or ""
                ).strip()
                if not commit_url:
                    raise RuntimeError(
                        f"Blocked: committed frame has no URL metadata after {method}"
                    )
                try:
                    blocked_commit_url = _private_address_from_candidates(commit_url)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Blocked: committed-frame private-page guard probe failed after {method}"
                    ) from exc

            frame_tree_id = next_id
            next_id += 1
            await ws.send(
                json.dumps(
                    {
                        "id": frame_tree_id,
                        "method": "Page.getFrameTree",
                        "params": {},
                        "sessionId": session_id,
                    }
                )
            )
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out validating target {target_id} after {method}"
                    )
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                msg = json.loads(raw)
                if msg.get("id") != frame_tree_id:
                    continue
                if "error" in msg:
                    raise RuntimeError(
                        f"Blocked: Page.getFrameTree failed after {method}: {msg['error']}"
                    )
                final_tree = msg.get("result", {}).get("frameTree", {})
                final_frame = _find_frame_in_tree(final_tree, expected_frame_id)
                if final_frame is None:
                    raise RuntimeError(
                        f"Blocked: selected frame is missing after {method}"
                    )
                final_url = str(final_frame.get("url") or "").strip()
                if not final_url:
                    raise RuntimeError(
                        f"Blocked: target has no live URL metadata after {method}"
                    )
                try:
                    blocked_final_url = _private_address_from_candidates(final_url)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Blocked: target private-page guard probe failed after {method}"
                    ) from exc
                break

            blocked_navigation_url = blocked_commit_url or blocked_final_url
            if blocked_navigation_url:
                blank_id = next_id
                await ws.send(
                    json.dumps(
                        {
                            "id": blank_id,
                            "method": "Page.navigate",
                            "params": {"url": "about:blank"},
                            "sessionId": session_id,
                        }
                    )
                )
                deadline = asyncio.get_running_loop().time() + timeout
                blank_error: Optional[str] = None
                blank_result: Dict[str, Any] = {}
                blank_events: list[Dict[str, Any]] = []
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        blank_error = "timed out resetting the target to about:blank"
                        break
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    msg = json.loads(raw)
                    if msg.get("method") == "Page.frameNavigated":
                        blank_events.append(msg)
                    if msg.get("id") != blank_id:
                        continue
                    if "error" in msg:
                        blank_error = f"failed to reset target to about:blank: {msg['error']}"
                    else:
                        blank_result = msg.get("result", {})
                    break
                if blank_error is None:
                    blank_frame_id = str(
                        blank_result.get("frameId") or expected_frame_id
                    ).strip()
                    blank_loader_id = str(blank_result.get("loaderId") or "").strip()

                    def _blank_committed(event: Dict[str, Any]) -> bool:
                        frame = event.get("params", {}).get("frame", {})
                        if str(frame.get("id") or "") != blank_frame_id:
                            return False
                        if blank_loader_id and str(frame.get("loaderId") or "") != blank_loader_id:
                            return False
                        return str(frame.get("url") or "").strip() == "about:blank"

                    blank_commit = next(
                        (event for event in blank_events if _blank_committed(event)),
                        None,
                    )
                    if blank_commit is None:
                        deadline = asyncio.get_running_loop().time() + timeout
                        while True:
                            remaining = deadline - asyncio.get_running_loop().time()
                            if remaining <= 0:
                                blank_error = (
                                    "timed out waiting for about:blank reset to commit"
                                )
                                break
                            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                            event = json.loads(raw)
                            if event.get("method") == "Page.frameNavigated" and _blank_committed(
                                event
                            ):
                                blank_commit = event
                                break
                reset_status = (
                    f"; {blank_error}" if blank_error else "; target reset to about:blank"
                )
                raise RuntimeError(
                    "Blocked: target navigation landed on a private or internal address "
                    f"({blocked_navigation_url}){reset_status}"
                )

        return result


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


def _browser_cdp_via_supervisor(
    task_id: str,
    frame_id: str,
    method: str,
    params: Optional[Dict[str, Any]],
    timeout: float,
) -> str:
    """Route a CDP call through the live supervisor session for an OOPIF frame.

    Looks up the frame in the supervisor's snapshot, extracts its child
    ``cdp_session_id``, and dispatches ``method`` with that sessionId via
    the supervisor's already-connected WebSocket (using
    ``asyncio.run_coroutine_threadsafe`` onto the supervisor loop).
    """
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover — defensive
        return tool_error(
            f"CDP supervisor is not available: {exc}. frame_id routing requires "
            f"a running supervisor attached via /browser connect or an active "
            f"Browserbase session."
        )

    supervisor = SUPERVISOR_REGISTRY.get(task_id)
    if supervisor is None:
        return tool_error(
            f"No CDP supervisor is attached for task={task_id!r}. Call "
            f"browser_navigate or /browser connect first so the supervisor "
            f"can attach. Once attached, browser_snapshot will populate "
            f"frame_tree with frame_ids you can pass here."
        )

    snap = supervisor.snapshot()
    # Search both the top frame and the children for the requested id.
    top = snap.frame_tree.get("top")
    frame_info: Optional[Dict[str, Any]] = None
    if top and top.get("frame_id") == frame_id:
        frame_info = top
    else:
        for child in snap.frame_tree.get("children", []) or []:
            if child.get("frame_id") == frame_id:
                frame_info = child
                break
    if frame_info is None:
        # Check the raw frames dict too (frame_tree is capped at 30 entries)
        with supervisor._state_lock:  # type: ignore[attr-defined]
            raw = supervisor._frames.get(frame_id)  # type: ignore[attr-defined]
        if raw is not None:
            frame_info = raw.to_dict()

    if frame_info is None:
        return tool_error(
            f"frame_id {frame_id!r} not found in supervisor state. "
            f"Call browser_snapshot to see current frame_tree."
        )

    # Validate the selected frame (frame_tree hit or raw _frames fallback)
    # before any child-session dispatch. A public top-level page can embed a
    # private OOPIF; top-page probing alone would miss that boundary.
    blocked_frame = _browser_cdp_selected_frame_private_guard(
        task_id=task_id,
        method=method,
        frame_info=frame_info,
    )
    if blocked_frame:
        return blocked_frame

    child_sid = frame_info.get("session_id")
    if not child_sid:
        # Not an OOPIF — fall back to top-level session (evaluating at page
        # scope).  Same-origin iframes don't get their own sessionId; the
        # agent can still use contentWindow/contentDocument from the parent.
        return tool_error(
            f"frame_id {frame_id!r} is not an out-of-process iframe (no "
            f"dedicated CDP session). For same-origin iframes, use "
            f"`browser_cdp(method='Runtime.evaluate', params={{'expression': "
            f"\"document.querySelector('iframe').contentDocument.title\"}})` "
            f"at the top-level page instead."
        )

    # Dispatch onto the supervisor's loop.
    loop = supervisor._loop  # type: ignore[attr-defined]
    if loop is None or not loop.is_running():
        return tool_error(
            "CDP supervisor loop is not running. Try reconnecting with "
            "/browser connect."
        )

    async def _do_cdp():
        # Re-resolve on the supervisor loop immediately before _cdp().
        # Page.frameNavigated can replace URL/origin on the same child
        # session after the copied snapshot check above; dispatch must see
        # the live address (or fail closed if the frame is gone / metadata-
        # less / private).
        live_info = _live_selected_frame_info(supervisor, frame_id)
        if live_info is None:
            raise _BrowserCdpFrameGuardBlocked(
                tool_error(
                    "Blocked: selected frame is missing or transitioning; "
                    f"raw CDP method {method!r} could expose private page "
                    "content or state. Retry after browser_snapshot.",
                    method=method,
                    cdp_docs=CDP_DOCS_URL,
                )
            )
        live_blocked = _browser_cdp_selected_frame_private_guard(
            task_id=task_id,
            method=method,
            frame_info=live_info,
        )
        if live_blocked:
            raise _BrowserCdpFrameGuardBlocked(live_blocked)
        live_sid = live_info.get("session_id")
        if not live_sid:
            raise _BrowserCdpFrameGuardBlocked(
                tool_error(
                    f"frame_id {frame_id!r} is not an out-of-process iframe "
                    f"(no dedicated CDP session). For same-origin iframes, use "
                    f"`browser_cdp(method='Runtime.evaluate', params={{'expression': "
                    f"\"document.querySelector('iframe').contentDocument.title\"}})` "
                    f"at the top-level page instead."
                )
            )
        initial_loader_id = ""
        if method in _CDP_TARGET_NAVIGATION_METHODS:
            initial_loader_id = await _prepare_supervisor_frame_navigation(
                supervisor=supervisor,
                frame_id=frame_id,
                session_id=live_sid,
                method=method,
                timeout=timeout,
            )
        result_msg = await supervisor._cdp(  # type: ignore[attr-defined]
            method,
            params or {},
            session_id=live_sid,
            timeout=timeout,
        )
        if method in _CDP_TARGET_NAVIGATION_METHODS:
            await _revalidate_supervisor_frame_navigation(
                supervisor=supervisor,
                task_id=task_id,
                frame_id=frame_id,
                session_id=live_sid,
                method=method,
                timeout=timeout,
                nav_result=result_msg.get("result", {}),
                initial_loader_id=initial_loader_id,
            )
        return result_msg, live_sid

    try:
        from agent.async_utils import safe_schedule_threadsafe
        fut = safe_schedule_threadsafe(_do_cdp(), loop)
        if fut is None:
            return tool_error(
                "CDP call via supervisor failed: loop unavailable",
                cdp_docs=CDP_DOCS_URL,
            )
        result_msg, dispatched_sid = fut.result(timeout=timeout + 2)
    except _BrowserCdpFrameGuardBlocked as blocked:
        return blocked.error_json
    except Exception as exc:
        return tool_error(
            f"CDP call via supervisor failed: {type(exc).__name__}: {exc}",
            cdp_docs=CDP_DOCS_URL,
        )

    payload: Dict[str, Any] = {
        "success": True,
        "method": method,
        "frame_id": frame_id,
        "session_id": dispatched_sid,
        "result": _redact_cdp_output(result_msg.get("result", {})),
    }
    return json.dumps(payload, ensure_ascii=False)


def browser_cdp(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    target_id: Optional[str] = None,
    frame_id: Optional[str] = None,
    timeout: float = 30.0,
    task_id: Optional[str] = None,
) -> str:
    """Send a raw CDP command.  See ``CDP_DOCS_URL`` for method documentation.

    Args:
        method: CDP method name, e.g. ``"Target.getTargets"``.
        params: Method-specific parameters; defaults to ``{}``.
        target_id: Optional target/tab ID for page-level methods.  When set,
            we first attach to the target (``flatten=True``) and send
            ``method`` with the resulting ``sessionId``.  Uses a fresh
            stateless CDP connection.
        frame_id: Optional cross-origin (OOPIF) iframe ``frame_id`` from
            ``browser_snapshot.frame_tree.children[]``.  When set (and the
            frame is an OOPIF with a live session tracked by the CDP
            supervisor), routes the call through the supervisor's existing
            WebSocket — which is how you Runtime.evaluate *inside* an
            iframe on backends where per-call fresh CDP connections would
            hit signed-URL expiry (Browserbase) or expensive reattach.
        timeout: Seconds to wait for the call to complete.
        task_id: Task identifier for supervisor lookup.  When ``frame_id``
            is set, this identifies which task's supervisor to use; the
            handler will default to ``"default"`` otherwise.

    Returns:
        JSON string ``{"success": True, "method": ..., "result": {...}}`` on
        success, or ``{"error": "..."}`` on failure.
    """
    effective_task_id = task_id or "default"

    # Validate params before any path (including frame_id early-return).
    # A non-dict would AttributeError inside _browser_cdp_private_guard's
    # (params or {}).get(...); that probe's broad except fail-opens, so
    # rejecting here keeps the SSRF/private-page boundary fail-closed and
    # matches the clear input-validation error the stateless path already
    # returned.
    call_params: Dict[str, Any] = params or {}
    if not isinstance(call_params, dict):
        return tool_error(
            f"'params' must be an object/dict, got {type(call_params).__name__}"
        )

    # --- Route iframe-scoped calls through the supervisor ---------------
    if frame_id:
        # Same private-page/SSRF boundary as the stateless path below —
        # frame_id routing must not become the sibling bypass for it.
        blocked = _browser_cdp_private_guard(
            task_id=effective_task_id,
            method=method,
            params=call_params,
        )
        if blocked:
            return blocked
        return _browser_cdp_via_supervisor(
            task_id=effective_task_id,
            frame_id=frame_id,
            method=method,
            params=call_params,
            timeout=timeout,
        )

    if not method or not isinstance(method, str):
        return tool_error(
            "'method' is required (e.g. 'Target.getTargets')",
            cdp_docs=CDP_DOCS_URL,
        )

    if not _WS_AVAILABLE:
        return tool_error(
            "The 'websockets' Python package is required but not installed. "
            "Install it with: pip install websockets"
        )

    endpoint = _resolve_cdp_endpoint()
    if not endpoint:
        return tool_error(
            "No CDP endpoint is available. Run '/browser connect' to attach "
            "to a running Chrome, Brave, Chromium, or Edge browser, or set "
            "'browser.cdp_url' in config.yaml. The Camofox backend is REST-only "
            "and does not expose CDP.",
            cdp_docs=CDP_DOCS_URL,
        )

    if not endpoint.startswith(("ws://", "wss://")):
        return tool_error(
            f"CDP endpoint is not a WebSocket URL: {endpoint!r}. "
            "Expected ws://... or wss://... — the /browser connect "
            "resolver should have rewritten this. Check that a Chromium-family "
            "browser is actually listening on the debug port."
        )

    blocked = _browser_cdp_private_guard(
        task_id=effective_task_id,
        method=method,
        params=call_params,
    )
    if blocked:
        return blocked

    guard_selected_target_url = False
    guard_navigation_result_url = False
    selected_target_guard_needed = method not in _CDP_PRIVATE_PAGE_ALLOWED_METHODS
    navigation_result_guard_needed = method in _CDP_TARGET_NAVIGATION_METHODS
    if target_id and (selected_target_guard_needed or navigation_result_guard_needed):
        try:
            from tools import browser_tool as bt  # type: ignore[import-not-found]

            guard_active = bool(bt._eval_ssrf_guard_active(effective_task_id))  # type: ignore[attr-defined]
            guard_selected_target_url = guard_active and selected_target_guard_needed
            guard_navigation_result_url = guard_active and navigation_result_guard_needed
        except Exception as exc:  # noqa: BLE001
            logger.debug("browser_cdp: selected-target guard activation probe failed: %s", exc)
            return tool_error(
                "Blocked: selected-target SSRF guard activation probe failed; "
                f"raw CDP method {method!r} could expose private page content or state.",
                method=method,
                cdp_docs=CDP_DOCS_URL,
            )

    try:
        safe_timeout = float(timeout) if timeout else 30.0
    except (TypeError, ValueError):
        safe_timeout = 30.0
    safe_timeout = max(1.0, min(safe_timeout, 300.0))

    try:
        result = _run_async(
            _cdp_call(
                endpoint,
                method,
                call_params,
                target_id,
                safe_timeout,
                guard_selected_target_url=guard_selected_target_url,
                guard_navigation_result_url=guard_navigation_result_url,
            )
        )
    except asyncio.TimeoutError as exc:
        return tool_error(
            f"CDP call timed out after {safe_timeout}s: {exc}",
            method=method,
        )
    except TimeoutError as exc:
        return tool_error(str(exc), method=method)
    except RuntimeError as exc:
        return tool_error(str(exc), method=method)
    except WebSocketException as exc:
        return tool_error(
            f"WebSocket error talking to CDP at {endpoint}: {exc}. The "
            "browser may have disconnected — try '/browser connect' again.",
            method=method,
        )
    except Exception as exc:  # pragma: no cover — unexpected
        logger.exception("browser_cdp unexpected error")
        return tool_error(
            f"Unexpected error: {type(exc).__name__}: {exc}",
            method=method,
        )

    payload: Dict[str, Any] = {
        "success": True,
        "method": method,
        "result": _redact_cdp_output(result),
    }
    if target_id:
        payload["target_id"] = target_id
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BROWSER_CDP_SCHEMA: Dict[str, Any] = {
    "name": "browser_cdp",
    "description": (
        "Send a raw Chrome DevTools Protocol (CDP) command. Escape hatch for "
        "browser operations not covered by browser_navigate, browser_click, "
        "browser_console, etc.\n\n"
        "**Requires a reachable CDP endpoint.** Available when the user has "
        "run '/browser connect' to attach to a running Chrome, Brave, Chromium, "
        "or Edge browser, or when 'browser.cdp_url' is set in config.yaml. "
        "Not currently wired up for cloud backends (Browserbase, Browser Use, "
        "Firecrawl) — those expose CDP per session but live-session routing is "
        "a follow-up. Camofox is REST-only and will never support CDP. If the "
        "tool is in your toolset at all, a CDP endpoint is already reachable.\n\n"
        f"**CDP method reference:** {CDP_DOCS_URL} — use web_extract on a "
        "method's URL (e.g. '/tot/Page/#method-handleJavaScriptDialog') "
        "to look up parameters and return shape.\n\n"
        "**Common patterns:**\n"
        "- List tabs: method='Target.getTargets', params={}\n"
        "- Handle a native JS dialog: method='Page.handleJavaScriptDialog', "
        "params={'accept': true, 'promptText': ''}, target_id=<tabId>\n"
        "- Get all cookies: method='Network.getAllCookies', params={}\n"
        "- Eval in a specific tab: method='Runtime.evaluate', "
        "params={'expression': '...', 'returnByValue': true}, "
        "target_id=<tabId>\n"
        "- Set viewport for a tab: method='Emulation.setDeviceMetricsOverride', "
        "params={'width': 1280, 'height': 720, 'deviceScaleFactor': 1, "
        "'mobile': false}, target_id=<tabId>\n\n"
        "**Usage rules:**\n"
        "- Browser-level methods (Target.*, Browser.*, Storage.*): omit "
        "target_id and frame_id.\n"
        "- Page-level methods (Page.*, Runtime.*, DOM.*, Emulation.*, "
        "Network.* scoped to a tab): pass target_id from Target.getTargets.\n"
        "- **Cross-origin iframe scope** (Runtime.evaluate inside an OOPIF, "
        "Page.* targeting a frame target, etc.): pass frame_id from the "
        "browser_snapshot frame_tree output. This routes through the CDP "
        "supervisor's live connection — the only reliable way on "
        "Browserbase where stateless CDP calls hit signed-URL expiry.\n"
        "- Each stateless call (without frame_id) is independent — sessions "
        "and event subscriptions do not persist between calls. For stateful "
        "workflows, prefer the dedicated browser tools or use frame_id "
        "routing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": (
                    "CDP method name, e.g. 'Target.getTargets', "
                    "'Runtime.evaluate', 'Page.handleJavaScriptDialog'."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Method-specific parameters as a JSON object. Omit or "
                    "pass {} for methods that take no parameters."
                ),
                "properties": {},
                "additionalProperties": True,
            },
            "target_id": {
                "type": "string",
                "description": (
                    "Optional. Target/tab ID from Target.getTargets result "
                    "(each entry's 'targetId'). Use for page-level methods "
                    "at the top-level tab scope. Mutually exclusive with "
                    "frame_id."
                ),
            },
            "frame_id": {
                "type": "string",
                "description": (
                    "Optional. Out-of-process iframe (OOPIF) frame_id from "
                    "browser_snapshot.frame_tree.children[] where "
                    "is_oopif=true. When set, routes the call through the "
                    "CDP supervisor's live session for that iframe. "
                    "Essential for Runtime.evaluate inside cross-origin "
                    "iframes, especially on Browserbase where fresh "
                    "per-call CDP connections can't keep up with signed "
                    "URL rotation. For same-origin iframes, use parent "
                    "contentWindow/contentDocument from Runtime.evaluate "
                    "at the top-level page instead."
                ),
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Timeout in seconds (default 30, max 300)."
                ),
                "default": 30,
            },
        },
        "required": ["method"],
    },
}


def _browser_cdp_check() -> bool:
    """Availability check for browser_cdp.

    The tool is only offered when the Python side can actually reach a CDP
    endpoint right now — meaning a static URL is set via ``/browser connect``
    (``BROWSER_CDP_URL``) or ``browser.cdp_url`` in ``config.yaml``.

    Backends that do *not* currently expose CDP to us — Camofox (REST-only),
    the default local agent-browser mode (Playwright hides its internal CDP
    port), and cloud providers whose per-session ``cdp_url`` is not yet
    surfaced — are gated out so the model doesn't see a tool that would
    reliably fail.  Cloud-provider CDP routing is a follow-up.

    Kept in a thin wrapper so the registration statement stays at module top
    level (the tool-discovery AST scan only picks up top-level
    ``registry.register(...)`` calls).
    """
    try:
        from tools.browser_tool import (  # type: ignore[import-not-found]
            _get_cdp_override_raw,
            check_browser_requirements,
        )
    except ImportError as exc:  # pragma: no cover — defensive
        logger.debug("browser_cdp check: browser_tool import failed: %s", exc)
        return False
    if not check_browser_requirements():
        return False
    # Raw (no-I/O) gate: check_fns run during tool-schema assembly at every
    # startup; resolving the endpoint over HTTP here would block launch when
    # the configured endpoint is stale/unreachable.
    return bool(_get_cdp_override_raw())


registry.register(
    name="browser_cdp",
    toolset="browser-cdp",
    schema=BROWSER_CDP_SCHEMA,
    handler=lambda args, **kw: routed_browser_handler(
        "browser_cdp",
        args,
        fallback=lambda: browser_cdp(
            method=args.get("method", ""),
            params=args.get("params"),
            target_id=args.get("target_id"),
            frame_id=args.get("frame_id"),
            timeout=args.get("timeout", 30.0),
            task_id=kw.get("task_id"),
        ),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
    ),
    check_fn=_browser_cdp_check,
    emoji="🧪",
)
