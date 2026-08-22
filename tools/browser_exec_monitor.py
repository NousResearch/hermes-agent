"""Host-side CDP Network monitor for ``browser_exec`` SSRF enforcement.

Region A of the browser_exec network-boundary class closure (PR #84999).
``NetworkExecMonitor`` connects to the SAME CDP endpoint the browser-use CLI
drives (``BU_CDP_WS``/``BU_CDP_URL`` in the exec env — the single source of
truth), attaches to every page target, enables ``Network`` on each session,
and validates every request URL with a strict, ungated, fail-closed predicate
(:func:`exec_url_violation`). The monitor withholds nothing by itself — the
caller (``tools/browser_use_cli.browser_exec`` / ``tools/browser_use_guard``)
uses its three-state semantics (``attach_failed`` / ``armed`` /
``saw_activity``) and the write-once violation latch to fail closed before
any output is released.

Design notes (see the adjudicated consensus contract):

* The monitor is a Hermes-side observer — it lives in neither the model's
  namespace nor the CLI's, so the model cannot forge CDP frames or mute the
  per-session ``Network.enable``.
* Three-state activity semantics: ``armed`` alone (WS connected, >=1 page
  session with ``Network.enable`` acked) does NOT release output; the caller
  must ALSO observe ``saw_activity`` during the exec window (any Network
  event / new page target, or a successful trusted landing probe against the
  same endpoint) before it can trust the observation.
* Validation is per-event with NO cross-request verdict cache: DNS-rebinding
  TOCTOU is a documented residual, not amplified. ``exec_url_violation``
  fails closed on ``gaierror`` regardless of proxy environment and never
  honors the ``allow_private_urls`` toggle or trusted-host lists.
"""

import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from tools.browser_supervisor import _redact_cdp_error_text

logger = logging.getLogger(__name__)

# Event set the monitor reacts to (subset of Network.* + Target.*).
_EVENT_REQUEST_WILL_BE_SENT = "Network.requestWillBeSent"
_EVENT_RESPONSE_RECEIVED = "Network.responseReceived"
_EVENT_REQUEST_SERVED_FROM_CACHE = "Network.requestServedFromCache"
_EVENT_LOADING_FAILED = "Network.loadingFailed"
_EVENT_TARGET_CREATED = "Target.targetCreated"
_EVENT_TARGET_ATTACHED = "Target.attachedToTarget"

# Target types the monitor arms (per-session Network.enable). Every
# auto-attached target — page, OOPIF iframe, dedicated/shared/service
# worker, fenced frame, and the worklet family (auction / interest-group /
# shared storage) — can make network requests, so every one of them must be
# observed: a worker or fenced-frame fetch to 169.254.169.254 is unobserved
# (and therefore unblocked) if its session is never Network.enable'd.
_MONITOR_ARMED_TARGET_TYPES = frozenset({
    "page", "iframe", "worker", "service_worker", "shared_worker",
    "background_page", "webview", "fencedframe",
    "auction_worklet", "interest_group_worklet", "shared_storage_worklet",
})

# Policy tags produced by exec_url_violation.
POLICY_METADATA = "metadata"
POLICY_PRIVATE = "private"
POLICY_MALFORMED = "malformed"

# Bounded event ring for request_log().
_REQUEST_LOG_MAX = 400


def _hostname_of(url: str) -> str:
    try:
        parsed = urlsplit(url or "")
        return (parsed.hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return ""


def exec_url_violation(url: str) -> Optional[str]:
    """Strict, ungated, fail-closed per-request policy predicate.

    Returns ``None`` when the URL passes (public / non-egress), or a policy
    tag: ``"metadata"`` | ``"private"`` | ``"malformed"``.

    Deliberately does NOT call ``is_safe_url`` (proxy-environment DNS
    carve-out + global toggle) and does NOT call ``_url_is_private``
    (gaierror → False). Rules, in order:

    1. Scheme gate: ``http``/``https``/``ws``/``wss`` only (case-insensitive);
       any other scheme passes (not SSRF egress at this layer).
    2. Malformed-authority floor: percent-encoded separators (``%2f``,
       ``%5c``) or a literal backslash in the raw authority, or userinfo
       (``@``) in the netloc → ``"malformed"`` (parser-divergence class —
       Chromium decodes these as authority separators).
    3. Metadata floor first (DNS-independent): ``_is_always_blocked_url``.
    4. Literal-IP hosts → classify directly (private/loopback/link-local/
       reserved/CGNAT → ``"private"``; always-blocked set → ``"metadata"``).
    5. Hostnames: blocked-hostname floor, ``localhost``, ``.local``/
       ``.lan``/``.internal`` suffixes → ``"private"`` without DNS; else
       resolve; ANY private/blocked answer → block; ``gaierror`` →
       ``"private"`` unconditionally (no proxy delegation, no env).
    6. Ungated by construction: no ``allow_private_urls`` consultation.
    7. ``ws``/``wss`` validated by the same host rules.
    8. No cross-request verdict cache.
    """
    raw = (url or "").strip()
    if not raw:
        return None

    try:
        parsed = urlsplit(raw)
    except ValueError:
        return POLICY_MALFORMED
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https", "ws", "wss"}:
        return None  # data:/blob:/about:/chrome:/file:/javascript: → not SSRF egress

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return POLICY_MALFORMED

    # 2. Malformed-authority / parser-divergence floor (raw netloc).
    netloc = parsed.netloc or ""
    if "%2f" in netloc.lower() or "%5c" in netloc.lower() or "\\" in netloc:
        return POLICY_MALFORMED
    if "@" in netloc and (parsed.username or ""):
        return POLICY_MALFORMED

    # 3. Metadata floor first (DNS-independent).
    try:
        from tools.browser_tool import _is_always_blocked_url

        if _is_always_blocked_url(raw):
            return POLICY_METADATA
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("exec_url_violation floor check failed for %s: %s", url, exc)

    # 4/5. Host classification.
    from tools.url_safety import (
        _ALWAYS_BLOCKED_IPS,
        _ALWAYS_BLOCKED_NETWORKS,
        _BLOCKED_HOSTNAMES,
        _CGNAT_NETWORK,
        _classify_ip,
    )

    def _literal_policy(ip: ipaddress._BaseAddress) -> Optional[str]:
        blocked, reason = _classify_ip(ip)
        if not blocked:
            return None
        plain = reason[len("mapped:"):] if reason.startswith("mapped:") else reason
        if ip in _ALWAYS_BLOCKED_IPS or plain in ("metadata-ip", "link-local", "ipv4-compatible"):
            return POLICY_METADATA
        return POLICY_PRIVATE

    try:
        host_for_ip = hostname.split("%", 1)[0]
        ip = ipaddress.ip_address(host_for_ip)
        return _literal_policy(ip)
    except ValueError:
        pass

    # Hostname hosts.
    if hostname in _BLOCKED_HOSTNAMES or hostname in ("localhost", "localhost.localdomain"):
        return POLICY_PRIVATE
    if hostname.endswith(".localhost") or hostname.endswith(".local") or \
            hostname.endswith(".lan") or hostname.endswith(".internal"):
        return POLICY_PRIVATE

    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        # Fail closed unconditionally — the browser's own connection may be
        # proxied, but Hermes' inability to resolve must never become a pass.
        return POLICY_PRIVATE
    except Exception:
        return POLICY_PRIVATE

    for _family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        if "%" in ip_str:
            ip_str = ip_str.split("%", 1)[0]
        try:
            resolved = ipaddress.ip_address(ip_str)
        except ValueError:
            return POLICY_PRIVATE  # unparseable answer → fail closed
        policy = _literal_policy(resolved)
        if policy is not None:
            return policy
    return None


class NetworkExecMonitor:
    """Per-exec CDP ``Network`` observer for one ``browser_exec`` window.

    All CDP I/O runs on the monitor's own daemon-thread asyncio loop. The
    monitor creates no browser state (never ``Target.createTarget``) and
    mutates no pages (no ``Fetch.enable``, no injected scripts) — it only
    observes and records.
    """

    STATE_NOT_STARTED = "not_started"
    STATE_ATTACH_FAILED = "attach_failed"
    STATE_ARMED = "armed"

    def __init__(self, cdp_url: str, *, task_id: str) -> None:
        self.cdp_url = cdp_url
        self.task_id = task_id
        self._state = self.STATE_NOT_STARTED
        self._state_lock = threading.Lock()
        self._violation: Optional[dict] = None
        self._request_log: List[dict] = []
        self._session_network_armed: set[str] = set()
        self._new_target_times: List[float] = []
        self._event_times: List[float] = []
        self._probe_success = False
        self._request_urls: Dict[str, str] = {}
        self._page_urls: List[str] = []
        self._dropped = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None
        self._stop_requested = False
        self._ready_event = threading.Event()
        self._start_error: Optional[str] = None
        self._next_call_id = 1
        self._pending_calls: Dict[int, asyncio.Future] = {}
        self._attached_targets: set[str] = set()

    # ── Public sync API ────────────────────────────────────────────────────

    def start(self, timeout: float = 15.0) -> None:
        """Launch the background loop and attach to all page targets.

        Never raises: any failure (WS never connected, attach error, enable
        error) leaves the monitor in ``attach_failed`` state so the caller's
        decision rule withholds output. On success the monitor is ``armed``
        (>=1 page session with ``Network.enable`` acked).
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop_requested = False
        self._ready_event.clear()
        self._start_error = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"exec-net-monitor-{self.task_id[:24]}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            with self._state_lock:
                self._state = self.STATE_ATTACH_FAILED
            self.stop()
            return
        if self._start_error is not None:
            with self._state_lock:
                self._state = self.STATE_ATTACH_FAILED
            self.stop()
            return
        with self._state_lock:
            if not self._session_network_armed:
                self._state = self.STATE_ATTACH_FAILED

    def stop(self, timeout: float = 5.0) -> None:
        """Cancel the monitor loop and join the thread."""
        self._stop_requested = True
        loop = self._loop
        if loop is not None and loop.is_running():
            async def _close_ws():
                ws = self._ws
                self._ws = None
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass

            try:
                from agent.async_utils import safe_schedule_threadsafe

                fut = safe_schedule_threadsafe(_close_ws(), loop)
                if fut is not None:
                    try:
                        fut.result(timeout=2.0)
                    except Exception:
                        pass
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def attach_failed(self) -> bool:
        with self._state_lock:
            return self._state == self.STATE_ATTACH_FAILED

    def armed(self) -> bool:
        with self._state_lock:
            return self._state == self.STATE_ARMED and bool(self._session_network_armed)

    def saw_activity(self, exec_started: float) -> bool:
        """True when the monitor observed exec-window traffic.

        ``exec_started`` is the caller's ``time.monotonic()`` recorded at
        spawn. Any Network request/response event, any new page target, or a
        probe-success flag (set by the caller after the trusted landing probe
        succeeded against the same endpoint) counts.
        """
        if self._probe_success:
            return True
        with self._state_lock:
            if any(t >= exec_started for t in self._event_times):
                return True
            if any(t >= exec_started for t in self._new_target_times):
                return True
        return False

    def mark_probe_success(self) -> None:
        """Record that the trusted landing probe succeeded on this endpoint."""
        with self._state_lock:
            self._probe_success = True

    def violation(self) -> Optional[dict]:
        """Return the write-once violation latch, or None."""
        with self._state_lock:
            return dict(self._violation) if self._violation else None

    def reset(self) -> None:
        """Clear the latch + ring (window boundary)."""
        with self._state_lock:
            self._violation = None
            self._request_log = []
            self._event_times = []
            self._new_target_times = []
            self._probe_success = False

    def request_log(self, limit: int = 200) -> list[dict]:
        with self._state_lock:
            return list(self._request_log[-limit:])

    # ── Region E listener extensions (browser-level coverage) ─────────────

    def last_known_url(self) -> str:
        """Most recent page URL observed across all targets (Region E L)."""
        with self._state_lock:
            if self._page_urls:
                return self._page_urls[-1]
            return ""

    def event_count(self) -> int:
        with self._state_lock:
            return len(self._event_times)

    def dropped(self) -> bool:
        """True when the WS dropped after a successful arm (attestation gap)."""
        with self._state_lock:
            return self._dropped

    def target_urls(self) -> list[str]:
        with self._state_lock:
            return list(self._page_urls[-50:])

    # ── Internals ──────────────────────────────────────────────────────────

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run())
        except BaseException as e:  # noqa: BLE001 — propagate via _start_error
            if not self._ready_event.is_set():
                self._start_error = _redact_cdp_error_text(e)
                self._ready_event.set()
            else:
                logger.warning("browser_exec network monitor crashed: %s", e)
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    async def _run(self) -> None:
        import websockets

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(self.cdp_url, max_size=50 * 1024 * 1024),
                timeout=10.0,
            )
        except Exception as e:
            self._start_error = _redact_cdp_error_text(e)
            self._ready_event.set()
            return

        reader_task = asyncio.create_task(self._read_loop(), name="exec-net-reader")
        try:
            await self._attach_initial_pages()
            with self._state_lock:
                if self._session_network_armed:
                    self._state = self.STATE_ARMED
            self._ready_event.set()
            await reader_task
        except BaseException as e:
            if not self._ready_event.is_set():
                self._start_error = _redact_cdp_error_text(e)
                self._ready_event.set()
                raise
            logger.warning("browser_exec network monitor session dropped: %s", e)
        finally:
            if not reader_task.done():
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
            ws = self._ws
            self._ws = None
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass

    async def _cdp(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("monitor WebSocket is not connected")
        call_id = self._next_call_id
        self._next_call_id += 1
        payload: Dict[str, Any] = {"id": call_id, "method": method}
        if params:
            payload["params"] = params
        if session_id:
            payload["sessionId"] = session_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_calls[call_id] = fut
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending_calls.pop(call_id, None)

    async def _attach_initial_pages(self) -> None:
        """Attach flattened to every existing target and enable Network.

        Every target type is armed (page, OOPIF iframe, worker family) — a
        worker or iframe request to a metadata address must be observed too.
        """
        resp = await self._cdp("Target.getTargets")
        targets = resp.get("result", {}).get("targetInfos", [])
        for target in targets:
            if target.get("type") not in _MONITOR_ARMED_TARGET_TYPES:
                continue
            try:
                await self._attach_page_target(target.get("targetId"))
            except Exception as e:
                logger.debug(
                    "network monitor: initial attach failed for target %s: %s",
                    str(target.get("targetId"))[:16], _redact_cdp_error_text(e),
                )
        # Auto-attach for OOPIF/worker children and targets created later
        # (new_tab() etc.). Browser-domain command on the root session.
        try:
            await self._cdp(
                "Target.setAutoAttach",
                {"autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True},
            )
        except Exception as e:
            logger.debug("network monitor: setAutoAttach failed: %s", _redact_cdp_error_text(e))

    async def _attach_page_target(self, target_id: Optional[str]) -> None:
        if not target_id:
            return
        attach = await self._cdp(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
        )
        session_id = attach.get("result", {}).get("sessionId")
        if not session_id:
            return
        await self._cdp("Network.enable", session_id=session_id)
        try:
            await self._cdp("Page.enable", session_id=session_id)
        except Exception:
            pass
        with self._state_lock:
            self._session_network_armed.add(session_id)
            self._attached_targets.add(target_id)

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if self._stop_requested:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if "id" in msg:
                    fut = self._pending_calls.pop(msg["id"], None)
                    if fut is not None and not fut.done():
                        if "error" in msg:
                            fut.set_exception(
                                RuntimeError(f"CDP error on id={msg['id']}: {msg['error']}")
                            )
                        else:
                            fut.set_result(msg)
                elif "method" in msg:
                    await self._on_event(
                        msg["method"], msg.get("params", {}), msg.get("sessionId")
                    )
        except Exception as e:
            logger.debug("browser_exec network monitor read loop exited: %s", e)
        if not self._stop_requested:
            # Unexpected WS drop after a successful arm = attestation gap.
            with self._state_lock:
                if self._state == self.STATE_ARMED:
                    self._dropped = True

    async def _on_event(
        self, method: str, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        if method == _EVENT_REQUEST_WILL_BE_SENT:
            await self._on_request_will_be_sent(params, session_id)
        elif method == _EVENT_RESPONSE_RECEIVED:
            await self._on_response_received(params, session_id)
        elif method == _EVENT_REQUEST_SERVED_FROM_CACHE:
            await self._on_request_served_from_cache(params, session_id)
        elif method == _EVENT_LOADING_FAILED:
            self._record_event()
        elif method == _EVENT_TARGET_ATTACHED:
            await self._on_target_attached(params)
        elif method == _EVENT_TARGET_CREATED:
            await self._on_target_created(params)
        elif method == "Page.frameNavigated":
            self._record_event()
            frame = params.get("frame") or {}
            url = str(frame.get("url") or "")
            if url:
                with self._state_lock:
                    self._page_urls.append(url)

    def _record_event(self) -> None:
        with self._state_lock:
            self._event_times.append(time.monotonic())

    def _record_request(self, entry: dict) -> None:
        with self._state_lock:
            self._request_log.append(entry)
            if len(self._request_log) > _REQUEST_LOG_MAX:
                self._request_log = self._request_log[-_REQUEST_LOG_MAX:]

    def _maybe_latch(self, url: str, policy: str, event: str, params: Dict[str, Any],
                     session_id: Optional[str]) -> None:
        with self._state_lock:
            if self._violation is not None:
                return  # write-once
            self._violation = {
                "url": url,
                "policy": policy,
                "event": event,
                "request_id": str(params.get("requestId") or ""),
                "ts": time.time(),
                "frame_id": str(params.get("frameId") or ""),
                "session_id": str(session_id or ""),
            }

    async def _on_request_will_be_sent(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        self._record_event()
        url = str(params.get("url") or "")
        request_id = str(params.get("requestId") or "")
        with self._state_lock:
            self._request_urls[request_id] = url
            if params.get("type") == "Document" and url:
                self._page_urls.append(url)
        # A redirect hop is validated by its own URL (redirectResponse.url).
        redirect = params.get("redirectResponse") or {}
        hop_url = str(redirect.get("url") or "") if redirect else ""
        for candidate in (url, hop_url):
            if not candidate:
                continue
            await self._validate(candidate, _EVENT_REQUEST_WILL_BE_SENT, params, session_id)

    async def _on_response_received(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        self._record_event()
        response = params.get("response") or {}
        url = str(response.get("url") or "")
        if not url and params.get("requestId"):
            url = self._request_urls.get(str(params["requestId"]), "")
        if not url:
            return
        event = _EVENT_RESPONSE_RECEIVED
        # Cache-served responses carry no new wire traffic — the URL still
        # gets validated (fromDiskCache/fromServiceWorker).
        await self._validate(url, event, params, session_id)

    async def _on_request_served_from_cache(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        self._record_event()
        request_id = str(params.get("requestId") or "")
        url = self._request_urls.get(request_id, "")
        if not url:
            return
        await self._validate(url, _EVENT_REQUEST_SERVED_FROM_CACHE, params, session_id)

    async def _on_target_attached(self, params: Dict[str, Any]) -> None:
        target_info = params.get("targetInfo") or {}
        if target_info.get("type") not in _MONITOR_ARMED_TARGET_TYPES:
            return
        self._record_event()
        with self._state_lock:
            self._new_target_times.append(time.monotonic())
        # The new session id travels in the event; attach + Network.enable it
        # (any type: page, OOPIF iframe, worker family).
        session_id = params.get("sessionId")
        target_id = target_info.get("targetId")
        if session_id and target_id:
            try:
                await self._cdp("Network.enable", session_id=session_id)
                with self._state_lock:
                    self._session_network_armed.add(session_id)
                    self._attached_targets.add(target_id)
            except Exception as e:
                logger.debug(
                    "network monitor: new-target enable failed: %s",
                    _redact_cdp_error_text(e),
                )
        else:
            await self._attach_page_target(target_id)

    async def _on_target_created(self, params: Dict[str, Any]) -> None:
        target_info = params.get("targetInfo") or {}
        if target_info.get("type") not in _MONITOR_ARMED_TARGET_TYPES:
            return
        self._record_event()
        with self._state_lock:
            self._new_target_times.append(time.monotonic())

    async def _validate(self, url: str, event: str, params: Dict[str, Any],
                        session_id: Optional[str]) -> None:
        entry = {
            "url": url,
            "event": event,
            "request_id": str(params.get("requestId") or ""),
            "ts": time.time(),
            "frame_id": str(params.get("frameId") or ""),
            "session_id": str(session_id or ""),
        }
        self._record_request(entry)

        def _check() -> Optional[str]:
            return exec_url_violation(url)

        try:
            policy = await asyncio.to_thread(_check)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("exec_url_violation raised for %s: %s", url, exc)
            policy = POLICY_MALFORMED
        if policy is not None:
            logger.warning(
                "browser_exec network monitor: %s request %s (%s) — latch",
                policy, url[:160], event,
            )
            self._maybe_latch(url, policy, event, params, session_id)


def spawn_supervised_chrome(tag: str) -> str:
    """Spawn a Hermes-supervised headless Chrome and return its WS URL.

    Used as the Region A fallback when no CDP endpoint is configured: the CLI
    gets exactly one browser to drive — the monitored one. Uses
    ``--remote-debugging-port=0`` and parses the ``DevTools listening on``
    stderr line for collision-safe port selection. The Chrome process is
    registered for atexit teardown and cached per tag (task_id/session) so
    repeated execs reuse the same browser + profile.

    Raises RuntimeError on any failure (caller converts to a tool error).
    """
    import atexit

    cache: Dict[str, dict] = spawn_supervised_chrome._cache  # type: ignore[attr-defined]
    existing = cache.get(tag)
    if existing:
        return existing["ws_url"]

    chrome = _find_chrome_binary()
    if not chrome:
        raise RuntimeError(
            "No CDP endpoint configured and no Chrome binary found; set "
            "browser.cdp_url / BROWSER_CDP_URL or a cloud provider."
        )
    profile = tempfile.mkdtemp(prefix=f"hermes-exec-chrome-{tag[:24]}-")
    try:
        proc = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--remote-debugging-port=0",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-gpu",
                "--site-per-process",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as e:
        raise RuntimeError(f"Failed to launch headless Chrome: {e}") from e

    ws_url = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        line = proc.stderr.readline() if proc.stderr is not None else ""
        if line:
            if "DevTools listening on" in line:
                m = re.search(r"ws://\S+", line)
                if m:
                    ws_url = m.group(0)
                    break
        else:
            time.sleep(0.1)

    if ws_url is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
        raise RuntimeError("Headless Chrome did not expose a CDP endpoint within 15s")

    def _teardown() -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            pass

    atexit.register(_teardown)
    cache[tag] = {"ws_url": ws_url, "proc": proc, "profile": profile}
    return ws_url


def _find_chrome_binary() -> Optional[str]:
    """Locate a Chrome/Chromium binary (mirrors the supervisor test fixture)."""
    import shutil

    for candidate in ("chrome", "google-chrome", "chromium", "chromium-browser",
                      "msedge", "chrome.exe", "msedge.exe"):
        found = shutil.which(candidate)
        if found:
            return found
    # Windows fallback: common install locations.
    for base in (
        os.environ.get("PROGRAMFILES", ""),
        os.environ.get("PROGRAMFILES(X86)", ""),
        os.environ.get("LOCALAPPDATA", ""),
    ):
        for rel in (
            r"Google\Chrome\Application\chrome.exe",
            r"Microsoft\Edge\Application\msedge.exe",
        ):
            candidate = os.path.join(base, rel)
            if os.path.isfile(candidate):
                return candidate
    return None


spawn_supervised_chrome._cache = {}  # type: ignore[attr-defined]
