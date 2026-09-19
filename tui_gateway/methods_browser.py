"""Browser connect/disconnect helpers for the browser.* RPCs (CDP probing, no network I/O on
status)."""

from __future__ import annotations

from .contracts.tools_commands import BrowserManageParams, BrowserManageResult
from .contracts.events import BrowserProgressPayload
from .method_ctx import HandlerRegistry, bind_module
import contextlib
import os
import time

_registry = HandlerRegistry()

_CDP_SCHEMES = {"http", "https", "ws", "wss"}


def _resolve_browser_cdp_url() -> str:
    """Configured browser CDP override without network I/O (``/browser status`` must be fast;
    ``tools.browser_tool_cdp._get_cdp_override`` HTTP-probes discovery URLs). Same precedence (env,
    then ``browser.cdp_url``) minus WS resolution; ``browser_navigate`` normalizes on the next call."""
    if env_url := os.environ.get("BROWSER_CDP_URL", "").strip():
        return env_url
    with contextlib.suppress(Exception):
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    return ""


def _is_default_local_cdp(parsed) -> bool:
    """Match the discovery-style local default; never the concrete WS form — a
    ``ws://127.0.0.1:9222/devtools/browser/<id>`` is connectable as-is and collapsing it to bare
    ``http://...:9222`` would break the connect."""
    with contextlib.suppress(ValueError):
        return (parsed.scheme in {"http", "ws"} and parsed.hostname in {"127.0.0.1", "localhost"}
                and (parsed.port or 80) == 9222 and parsed.path in {"", "/", "/json", "/json/version"})
    return False


def _cdp_http_reachable(parsed, timeout: float = 2.0) -> bool:
    """True when ``/json/version`` or ``/json`` on the CDP host answers 2xx."""
    import urllib.request
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    root = f"{scheme}://{parsed.netloc}".rstrip("/")
    for url in (f"{root}/json/version", f"{root}/json"):
        with contextlib.suppress(Exception), urllib.request.urlopen(url, timeout=timeout) as resp:
            if 200 <= getattr(resp, "status", 200) < 300:
                return True
    return False


def _connect_local_default(port: int, system: str, announce) -> str | None:
    """Discover (or launch) the default local debug browser → CDP URL, or None after announcing."""
    from hermes_cli.browser_connect import (
        discover_local_cdp_url, find_free_debug_port, launch_chrome_debug, local_port_in_use,
        manual_chrome_debug_command)

    # Dual-stack discovery: when another app squats the IPv4 loopback on the debug port, a
    # browser bound there comes up on [::1] only; an IPv4-only probe misses it AND hangs
    # against squatters that accept TCP but never answer HTTP.
    discovered = discover_local_cdp_url(port, timeout=2.0)
    if discovered is not None:
        announce(f"Chromium-family browser is already listening at {discovered}")
        return discovered
    launch_port = port
    if local_port_in_use(port):
        launch_port = find_free_debug_port(port)
        announce(f"Port {port} is occupied by another application that isn't a CDP browser "
                 "(an IDE debugger or dev server may be using it) — launching a debug browser "
                 f"on port {launch_port} instead...")
    else:
        announce("Chromium-family browser isn't running with remote debugging — attempting to launch...")
    launch = launch_chrome_debug(launch_port, system)
    if launch.launched:
        # Bounded wait: the whole connect must finish inside the client RPC timeout.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if discovered := discover_local_cdp_url(launch_port, timeout=1.0):
                break
            time.sleep(0.5)
    if discovered:
        announce(f"Chromium-family browser launched and listening on port {launch_port}")
        return discovered
    if launch.hint:
        announce(launch.hint, level="error")
    command = manual_chrome_debug_command(launch_port, system)
    hints = (
        ["Start a Chromium-family browser with remote debugging, then retry /browser connect:", command]
        if command else [
            "No supported Chromium-family browser executable was found in this environment.",
            f"Install one or start a Chromium-family browser with --remote-debugging-port={launch_port}, then retry /browser connect."])
    hints.append("Browser not connected — start a Chromium-family browser with remote debugging and retry /browser connect")
    for line in hints:
        announce(line, level="error")
    return None


def _browser_connect(rid, params: BrowserManageParams) -> BrowserManageResult | dict:
    import platform
    from hermes_cli.browser_connect import DEFAULT_BROWSER_CDP_URL
    from tools.browser_tool_lifecycle import cleanup_all_browsers
    from urllib.parse import urlparse
    url = (params.url or "").strip() or DEFAULT_BROWSER_CDP_URL
    sid, system, messages = params.session_id or "", platform.system(), []

    def announce(message: str, *, level: str = "info") -> None:
        messages.append(message)
        # Without a session id the TUI prints `messages` from the response (an event would double-render).
        if sid:
            srv._emit("browser.progress", sid, BrowserProgressPayload(message=message, level=level))
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.scheme not in srv._CDP_SCHEMES:
        return srv._err(rid, 4015, f"unsupported browser url: {url}")
    if not parsed.hostname:
        return srv._err(rid, 4015, f"missing host in browser url: {url}")
    try:
        port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    except ValueError:
        return srv._err(rid, 4015, f"invalid port in browser url: {url}")
    # Normalize default-local to 127.0.0.1:9222 so comparisons + messaging match what we persist.
    if srv._is_default_local_cdp(parsed):
        url = DEFAULT_BROWSER_CDP_URL
        parsed = urlparse(url)
        port = parsed.port or 9222
    try:
        # Hosted ws[s]://.../devtools/browser/<id> endpoints don't serve the HTTP discovery path:
        # check TCP reachability only and let browser_navigate handshake.
        if parsed.scheme in {"ws", "wss"} and parsed.path.startswith("/devtools/browser/"):
            import socket
            try:
                with socket.create_connection((parsed.hostname, port), timeout=2.0):
                    pass
            except OSError as e:
                return srv._err(rid, 5031, f"could not reach browser CDP at {url}: {e}")
        elif srv._is_default_local_cdp(parsed):
            discovered = srv._connect_local_default(port, system, announce)
            if discovered is None:
                return BrowserManageResult(connected=False, url=url, messages=messages or None)
            # Adopt whatever loopback/port answered ([::1] and/or an alternate port when 9222 was squatted).
            url = discovered
            parsed = urlparse(url)
        elif not srv._cdp_http_reachable(parsed):
            return srv._err(rid, 5031, f"could not reach browser CDP at {url}")
        # Concrete ``/devtools/browser/<id>`` endpoints stay as-is; discovery-style inputs collapse
        # to ``scheme://host:port`` so ``_resolve_cdp_override`` can append ``/json/version``.
        normalized = (parsed.geturl() if parsed.path.startswith("/devtools/browser/")
                      else parsed._replace(path="", params="", query="", fragment="").geturl())
        # Reap BEFORE publishing the new env (an in-flight tool call sees the old supervisor closed)
        # and AFTER (the default task's cached supervisor drains against the new URL).
        cleanup_all_browsers()
        os.environ["BROWSER_CDP_URL"] = normalized
        cleanup_all_browsers()
    except Exception as e:
        return srv._err(rid, 5031, str(e))
    return BrowserManageResult(connected=True, url=normalized, messages=messages or None)


def _browser_disconnect(rid) -> BrowserManageResult:
    # Reap, drop the override, reap again — same swap window as ``_browser_connect``.
    def reap() -> None:
        with contextlib.suppress(Exception):
            from tools.browser_tool_lifecycle import cleanup_all_browsers
            cleanup_all_browsers()

    reap()
    os.environ.pop("BROWSER_CDP_URL", None)
    reap()
    return BrowserManageResult(connected=False)


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server`` and install its handlers."""
    bind_module(globals(), server)

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
