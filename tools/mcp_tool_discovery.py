"""Connecting and discovery for tools.mcp_tool: per-server connect cooldown, connect /
lazy-start / recycled-stdio wake-up, ``register_mcp_servers`` / ``discover_mcp_tools`` and
the status / probe public API. Origin state (``_servers``, ``_lock``, the loop, patchable
helpers) is read through ``_core`` so ``mock.patch("tools.mcp_tool.X")`` keeps working."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tools.mcp_tool_common import _exc_str, _core, _parse_boolish, _sanitize_error
from tools import mcp_tool_config as _config
from tools import mcp_tool_errors as _errors
from tools import mcp_tool_lifecycle as _lifecycle
from tools import mcp_tool_loop as _loop
from tools import mcp_tool_registration as _registration
from tools.mcp_tool_schema import MCP_TOOL_NAME_PREFIX
from tools.mcp_tool_scope import _key_name, _key_scope, _key_visible_in_scope, _resolve_server_key, _server_key

logger = logging.getLogger("tools.mcp_tool")

# Default max concurrent MCP server connections per discovery pass (one unbounded
# `asyncio.gather` spawned every server's subprocess tree simultaneously); config.yaml
# ``mcp.discovery_concurrency`` overrides it, 0 = unlimited (#117373).
_DISCOVERY_CONNECT_CONCURRENCY = 4


def _discovery_connect_concurrency() -> int:
    """``mcp.discovery_concurrency`` from config (0 = unlimited); a non-integer or negative value
    warns and falls back to the default rather than silently running unbounded."""
    try:
        from hermes_cli.config import load_config
        raw = (load_config().get("mcp") or {}).get("discovery_concurrency", _DISCOVERY_CONNECT_CONCURRENCY)
    except Exception:
        return _DISCOVERY_CONNECT_CONCURRENCY
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        logger.warning("mcp.discovery_concurrency=%r is not a non-negative integer; using %d",
                       raw, _DISCOVERY_CONNECT_CONCURRENCY)
        return _DISCOVERY_CONNECT_CONCURRENCY
    return raw


def _record_connect_failure(server_name: str) -> None:
    """Stamp a geometric, capped retry cooldown after a failed connect (under ``_lock``)."""
    key = _server_key(server_name)
    n = _core._server_connect_failures.get(key, 0) + 1
    _core._server_connect_failures[key] = n
    backoff = min(_core._CONNECT_RETRY_BASE_BACKOFF_SEC * (2 ** (n - 1)), _core._CONNECT_RETRY_MAX_BACKOFF_SEC)
    _core._server_connect_retry_after[key] = time.monotonic() + backoff


def _clear_connect_failure(server_name: str) -> None:
    """Clear the connect-cooldown state after a successful connection."""
    key = _server_key(server_name)
    _core._server_connect_failures.pop(key, None)
    _core._server_connect_retry_after.pop(key, None)


def _connect_cooldown_active(server_name: str) -> bool:
    """True if ``server_name`` is still within its retry cooldown (this scope's connection: one
    profile's failing ``x`` must not shadow another profile's healthy ``x``)."""
    deadline = _core._server_connect_retry_after.get(_server_key(server_name))
    return deadline is not None and time.monotonic() < deadline


def _enabled(cfg: dict) -> bool:
    return _parse_boolish(cfg.get("enabled", True), default=True)


def _owner_scope_home() -> Optional[Path]:
    """The profile home whose secret scope MCP credential reads must resolve under, or None when
    the caller is already scoped or this is a single-profile process (scope key ``None``).

    The owner is the profile the connection is keyed under (``_mcp_registry_scope()``), never the
    ambient one, so a served profile is never handed another profile's token (#111151)."""
    from agent.secret_scope import current_secret_scope
    if current_secret_scope() is not None:
        return None
    scope_key = _core._mcp_registry_scope()
    return None if scope_key is None else Path(scope_key)


async def _install_owner_secret_scope():
    """Bind the connection OWNER's profile secret scope when the caller has none; else None.

    ``MCPServerTask.start`` ensure_futures the run task, which copies THIS context, so one
    binding covers transport bring-up (``_build_safe_env`` stdio child env) and every later
    revival inside that task; unscoped, those ``get_secret`` reads fail closed under multiplexing
    and the server parks with zero tools (#113746). ``${VAR}`` refs are interpolated earlier, at
    config load, under :func:`_owner_secret_scope`.
    """
    from agent.secret_scope import build_profile_secret_scope, set_secret_scope
    home = _owner_scope_home()
    if home is None:
        return None
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    # Off-loop: an external source runs a helper subprocess (once per home, then cached).
    await asyncio.to_thread(hydrate_profile_secret_sources, home)
    return set_secret_scope(build_profile_secret_scope(home))


@contextmanager
def _owner_secret_scope():
    """Sync twin of :func:`_install_owner_secret_scope` for the config load in the caller's thread.

    ``_load_mcp_config`` interpolates ``${VAR}`` header/URL refs through ``get_secret`` and swallows
    the resulting ``UnscopedSecretError`` into ``{}``, so an unscoped discover / reconcile / status /
    probe for a routed profile saw ZERO servers — stdio siblings included — before the connect-site
    binding could ever run (#113746). Same owner rule; scope key ``None`` binds nothing."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    home = _owner_scope_home()
    if home is None:
        yield
        return
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    hydrate_profile_secret_sources(home)
    token = set_secret_scope(build_profile_secret_scope(home))
    try:
        yield
    finally:
        reset_secret_scope(token)


async def _connect_server(name: str, config: dict) -> _core.MCPServerTask:
    """Create an MCPServerTask, start it, return once ready (tear down with ``server.shutdown()``
    on the same loop). Raises on bad config, missing HTTP support or connect failure."""
    server = _core.MCPServerTask(name)
    claim = _core._connect_server_claim.get()
    if claim is not None:
        claim(server)
    # The run task copies this context: don't retain the discovery closure for its life.
    claim_token = _core._connect_server_claim.set(None) if claim is not None else None
    scope_token = None
    try:
        scope_token = await _install_owner_secret_scope()
        await server.start(config)
    except asyncio.CancelledError:
        raise  # start() already reaps server._task; shutdown() here could swallow the cancel
    except BaseException:
        # Discovery owns claimed tasks (recoverable park); standalone probes must reap locally.
        if claim is None:
            try:
                await server.shutdown()
            except Exception as shutdown_exc:  # noqa: BLE001 -- best-effort reap, don't mask the real error
                logger.debug("MCP server '%s' shutdown during orphan-reap failed: %s", name, shutdown_exc)
        raise
    finally:
        if scope_token is not None:
            from agent.secret_scope import reset_secret_scope
            reset_secret_scope(scope_token)
        if claim_token is not None:
            _core._connect_server_claim.reset(claim_token)
    return server


def _request_lazy_reconnect(server_name: str, server: _core.MCPServerTask) -> bool:
    """Wake a recycled stdio server and wait briefly for a fresh session."""
    loop = _loop._running_loop() if server._is_recycled_stdio() else None
    if loop is None:
        return False

    def _wake() -> None:
        server._ready.clear()
        server._reconnect_event.set()

    loop.call_soon_threadsafe(_wake)

    async def _await_ready() -> bool:
        deadline = time.monotonic() + _core._RECYCLED_RECONNECT_TIMEOUT
        while time.monotonic() < deadline:
            if server.session is not None and server._ready.is_set():
                return True
            await asyncio.sleep(0.05)
        return False

    try:
        return bool(_loop._run_on_mcp_loop(_await_ready, timeout=_core._RECYCLED_RECONNECT_TIMEOUT))
    except Exception as exc:
        logger.warning("MCP server '%s': lazy reconnect after stdio recycle failed: %s", server_name,
                       _sanitize_error(_exc_str(exc), getattr(server, "_redaction_values", ())))
        return False


def _resolve_server_lazy(name: str, config: dict) -> bool:
    """True when ``mcp_servers.<name>.lazy`` defers connect to first tool use (default off).

    Gated per-server by ``mcp_servers.<name>.lazy`` in config (default OFF), following the same per-server
    key pattern as ``idle_timeout_seconds``. Design from #56832 (Vansh5632).
    """
    return _parse_boolish(config.get("lazy", False), default=False)


def _note_connect_failure(name: str, exc: BaseException, redaction_values=()) -> str:
    """Record a failed connect (under ``_lock``): error text for status, cooldown stamp."""
    message = _errors._format_connect_error(exc, redaction_values)
    with _core._lock:
        key = _server_key(name)
        _core._server_connecting.discard(key)
        _core._server_connect_errors[key] = message
        _record_connect_failure(name)
    return message


def _note_connect_success(name: str) -> None:
    """Clear connecting/error/cooldown state after a successful connect (under ``_lock``)."""
    with _core._lock:
        key = _server_key(name)
        _core._server_connecting.discard(key)
        _core._server_connect_errors.pop(key, None)
        _clear_connect_failure(name)


def _adopt_server(name: str, server: _core.MCPServerTask) -> None:
    """Publish *server* into ``_servers`` under the connecting scope's key (under ``_lock``)."""
    with _core._lock:
        key = _server_key(name)
        _core._servers[key] = server
        _core._server_scope_keys[key] = _core._mcp_registry_scope()


def _ensure_lazy_server_connected(server_name: str) -> bool:
    """Connect a lazily-registered server on demand (sync; blocks). Honours the cooldown and the
    ``_server_connecting`` dedup set; routes through ``_discover_and_register_server`` so
    park/recycle/cooldown bookkeeping stays in one place. True when a live session exists.

    See #50394.
    """
    with _core._lock:
        key = _resolve_server_key(server_name)
        server = _core._servers.get(key)
        if server is not None and server.session is not None:
            return True
        config = _core._lazy_server_configs.get(key)
        if (not config or _connect_cooldown_active(server_name)
                or key in _core._server_connecting):
            return False
        _core._server_connecting.add(key)
        _core._server_connect_errors.pop(key, None)
    logger.info("MCP server '%s': lazy start on first use", server_name)
    _loop._ensure_mcp_loop()
    connect_timeout = config.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT)
    try:
        _loop._run_on_mcp_loop(lambda: _discover_and_register_server(server_name, config),
                               timeout=float(connect_timeout) + 30.0)
    except BaseException as exc:
        logger.warning("Lazy MCP connect failed for '%s': %s", server_name, _note_connect_failure(server_name, exc))
        return False
    _note_connect_success(server_name)
    with _core._lock:
        _core._lazy_server_configs.pop(key, None)
        stale_fingerprint = _core._lazy_server_fingerprints.pop(key, None)
        cached_names = _core._lazy_server_tool_names.pop(key, None) or []
        server = _core._servers.get(key)
        live_names = set(getattr(server, "_registered_tool_names", []) or [])
    # The cached manifest may advertise tools the live server no longer serves.
    phantom_names = [n for n in cached_names if n not in live_names]
    if phantom_names:
        for tool_name in phantom_names:
            _registration._deregister_mcp_tool_all_scopes(key, tool_name)
        logger.info("MCP server '%s': deregistered %d phantom cached tool(s) not served live (stale schema-cache "
                    "fingerprint %s): %s", server_name, len(phantom_names), stale_fingerprint, ", ".join(phantom_names))
    return server is not None and server.session is not None


def _get_connected_server_for_call(server_name: str) -> Optional[_core.MCPServerTask]:
    """Return a connected server; the single first-use connect point for lazy servers and
    the wake-up point for recycled stdio ones.

    Also the single first-use connect point for lazy (schema-cache registered) servers, so raw tool calls
    AND the resource/prompt utility handlers all trigger the deferred spawn (#56832).
    """
    with _core._lock:
        key = _resolve_server_key(server_name)
        server = _core._servers.get(key)
        is_lazy = key in _core._lazy_server_configs
    if is_lazy and (server is None or server.session is None):
        _ensure_lazy_server_connected(server_name)
    elif server is not None and server.session is None and server._is_recycled_stdio():
        _request_lazy_reconnect(server_name, server)
    else:
        return server
    with _core._lock:
        return _core._servers.get(key)


async def _discover_and_register_server(name: str, config: dict) -> List[str]:
    """Connect one server, register its tools; return the registered names."""
    # The claim fires inside _connect_server while this frame is suspended (list, not nonlocal).
    claimed: List[_core.MCPServerTask] = []
    claim_token = _core._connect_server_claim.set(claimed.append)
    try:
        server = await asyncio.wait_for(_connect_server(name, config),
                                        timeout=config.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT))
    except BaseException:
        server = claimed[0] if claimed else None
        task = server._task if server is not None else None
        task_cancelling = task.cancelling() if task is not None and hasattr(task, "cancelling") else 0
        if (server is not None and server._error is not None and task is not None
                and not task.done() and not task_cancelling):
            # Recoverable park: the run task self-probes, so adopt it for shutdown/revival.
            _adopt_server(name, server)
        elif server is not None:
            await server.shutdown()
        raise
    finally:
        _core._connect_server_claim.reset(claim_token)
    with _core._lock:
        key = _server_key(name)
        _core._server_connecting.discard(key)
        _core._server_connect_errors.pop(key, None)
    _adopt_server(name, server)
    registered_names = _registration._register_server_tools(name, server, config)
    server._registered_tool_names = list(registered_names)
    logger.info("MCP server '%s' (%s): registered %d tool(s): %s", name,
                "HTTP" if "url" in config else "stdio", len(registered_names), ", ".join(registered_names))
    return registered_names


def _select_new_servers(servers: Dict[str, dict]) -> Dict[str, dict]:
    """Pick connect candidates (enabled, not connected/connecting/lazy, not in backoff) and
    refresh per-server bookkeeping. Known servers without a live session are parked or
    mid-reconnect with tools deregistered, so nothing else can nudge them: signal a reconnect."""
    with _core._lock:
        current_scope = _core._mcp_registry_scope()
        # This scope's own connections OR shared ones it adopted (``register_connected_into_current_scope``
        # ran first): a same-named server owned by ANOTHER profile with other credentials is not
        # "connected" for us and must be a candidate, or this profile ends up silently tool-less.
        keys = {k: _resolve_server_key(k, current_scope, current=False) for k in servers}
        # Only attempt servers that aren't already connected (or currently connecting) and are enabled.
        # Checking ``_server_connecting`` prevents duplicate subprocess spawns when ``discover_mcp_tools()``
        # is called from multiple entry-points before the first batch finishes (#58862).
        new_servers = {
            k: v for k, v in servers.items()
            if keys[k] not in _core._servers and keys[k] not in _core._server_connecting
            and keys[k] not in _core._lazy_server_configs
            and _enabled(v) and not _connect_cooldown_active(k)}
        stale_cached = [_core._servers[keys[k]] for k, v in servers.items()
                        if keys[k] in _core._servers and _enabled(v)
                        and getattr(_core._servers[keys[k]], "session", None) is None]
        for srv_name in new_servers:
            _core._server_connecting.add(keys[srv_name])
            _core._server_scope_keys[keys[srv_name]] = current_scope
            _core._server_connect_errors.pop(keys[srv_name], None)
        # Track which servers opt-in to parallel tool calls (idempotent). Keyed by THIS profile's own
        # key: the opt-in is the calling profile's policy, so B's parallel-safe `x` never makes A's
        # same-named serial `x` (own connection or adopted) run two calls at once.
        for srv_name, srv_cfg in servers.items():
            own_key = _server_key(srv_name, current_scope, current=False)
            if _parse_boolish(srv_cfg.get("supports_parallel_tool_calls", False), default=False):
                _core._parallel_safe_servers.add(own_key)
            else:
                _core._parallel_safe_servers.discard(own_key)
    for srv in stale_cached:
        _loop._signal_reconnect(srv)
    return new_servers


def _register_lazy_from_cache(new_servers: Dict[str, dict]) -> Tuple[Dict[str, dict], int, int]:
    """Register ``lazy: true`` servers from a valid schema-cache entry without connecting
    (missing/stale entry or failed registration -> eager). Returns (eager servers, lazy tool
    count, lazy server count)."""
    # A missing or stale cache entry falls back to the normal eager connect below (which write-through
    # refreshes the cache for next time). See #56832.
    eager_servers: Dict[str, dict] = dict(new_servers)
    lazy_registered = 0
    lazy_server_count = 0
    try:
        from tools.mcp_schema_cache import config_fingerprint, get_cached_entry
    except Exception:  # pragma: no cover - cache module missing
        return eager_servers, 0, 0
    for name, cfg in new_servers.items():
        if not _resolve_server_lazy(name, cfg):
            continue
        entry = get_cached_entry(name, config_fingerprint(cfg))
        if not entry:
            continue
        with _core._lock:
            _core._server_connecting.discard(_server_key(name))
        try:
            names = _registration._register_from_cache_sync(name, cfg, entry)
        except Exception as exc:
            logger.warning("Failed lazy MCP registration for '%s': %s", name, exc)
            with _core._lock:
                _core._server_connecting.add(_server_key(name))
            continue
        eager_servers.pop(name, None)
        lazy_registered += len(names)
        lazy_server_count += 1
    return eager_servers, lazy_registered, lazy_server_count


async def _discover_all(new_servers: Dict[str, dict]) -> None:
    """Connect every candidate concurrently; record per-server outcome.

    Concurrency is bounded: every stdio server spawns child processes, so an
    unbounded gather turns a config with many servers into a simultaneous
    N-process spawn burst (RAM/CPU spike, EMFILE risk) on every backend boot.
    """
    # Flat cap for all transports; 0 (unlimited) keeps the original unbounded gather.
    cap = _discovery_connect_concurrency()
    semaphore = asyncio.Semaphore(cap) if cap > 0 else None

    async def _connect_bounded(name: str, cfg: dict):
        if semaphore is None:
            return await _discover_and_register_server(name, cfg)
        async with semaphore:
            return await _discover_and_register_server(name, cfg)

    results = await asyncio.gather(
        *(_connect_bounded(name, cfg) for name, cfg in new_servers.items()),
        return_exceptions=True)
    for name, result in zip(new_servers, results):
        if isinstance(result, BaseException):
            command = new_servers.get(name, {}).get("command")
            redaction_values = tuple(_config._load_mcp_server_env(new_servers.get(name, {})).values())
            message = _note_connect_failure(name, result, redaction_values)
            if command:
                command = _sanitize_error(str(command), redaction_values)
            logger.warning("Failed to connect to MCP server '%s'%s: %s",
                           name, f" (command={command})" if command else "", message)
        else:
            _note_connect_success(name)


def _run_discovery_pass(new_servers: Dict[str, dict]) -> None:
    """Run ``_discover_all`` on the MCP loop with the interrupt flag parked; clean up
    ``_server_connecting`` when the pass dies early."""
    # Executor threads are reused: a prior session's stale interrupt must not cancel this pass.
    from tools.interrupt import is_interrupted as _is_interrupted, set_interrupt as _set_interrupt
    _was_interrupted = _is_interrupted()
    if _was_interrupted:
        _set_interrupt(False)
    try:
        # Budget scales with the concurrency cap: a bounded gather finishes in
        # ceil(N/cap) waves, so N > cap multiplies the wall clock the base
        # (unbounded) gather never needed. 120s per wave keeps the original
        # per-wave ceiling; a slow fleet aborts later, not never. Capped by
        # _MCP_DISCOVERY_PASS_MAX_SEC so one stuck connect cannot pin the
        # calling thread (and the cross-process discovery lock) for tens of
        # minutes; the lock waiter's budget is derived from the same ceiling.
        cap = _discovery_connect_concurrency() or len(new_servers) or 1
        waves = max(1, -(-len(new_servers) // cap))
        timeout = min(120 * waves, _core._MCP_DISCOVERY_PASS_MAX_SEC)
        _loop._run_on_mcp_loop(lambda: _discover_all(new_servers), timeout=timeout)
    except (TimeoutError, InterruptedError) as _e:
        # Stranded _server_connecting entries would block future reconnects.
        how = "timed out" if isinstance(_e, TimeoutError) else "interrupted"
        with _core._lock:
            stale = [n for n in new_servers if _server_key(n) in _core._server_connecting]
            if stale:
                logger.warning("MCP discovery %s while %d server(s) were still connecting; clearing stale "
                               "connecting set: %s", how, len(stale), ", ".join(stale))
                for _sn in stale:
                    _core._server_connecting.discard(_server_key(_sn))
                    _core._server_connect_errors.setdefault(
                        _server_key(_sn), f"Connection attempt {how} during discovery")
                    # Its attempt is still running on the MCP loop; without a cooldown the next
                    # reconcile tick would spawn a second one beside it.
                    _record_connect_failure(_sn)
        raise
    finally:
        if _was_interrupted:
            _set_interrupt(True)


def _connected_summary(names, *, lazy_tools: int = 0,
                       lazy_servers: int = 0) -> Tuple[int, int, List[Tuple[str, str]]]:
    """(tool count, connected count, ``[(failed name, reason)]``) for candidate names, plus lazy
    servers. The reason is the recorded connect error; a candidate this pass never attempted (still
    inside its retry cooldown from an earlier failure) has none."""
    with _core._lock:
        keys = {n: _server_key(n) for n in names}
        connected = [n for n in names
                     if keys[n] in _core._servers and keys[n] not in _core._server_connect_errors]
        tool_count = sum(len(getattr(_core._servers[keys[n]], "_registered_tool_names", [])) for n in connected)
        failed = [(n, _core._server_connect_errors.get(keys[n]) or "not attempted (in retry cooldown)")
                  for n in names if n not in connected]
    return tool_count + lazy_tools, len(connected) + lazy_servers, failed


def _log_summary(prefix: str, names, **lazy) -> None:
    """Log ``<prefix> N tool(s) from M server(s) (K failed: name (reason), ...)`` when anything
    happened. The failures are named on the summary line itself (#114746): the count alone left
    the failing server identifiable only by elimination, and a candidate skipped for its retry
    cooldown never gets a per-server WARNING at all."""
    new_tool_count, connected_count, failed = _connected_summary(names, **lazy)
    if new_tool_count or failed or lazy.get("lazy_servers"):
        summary = f"{prefix} {new_tool_count} tool(s) from {connected_count} server(s)"
        if failed:
            summary += f" ({len(failed)} failed: " + "; ".join(
                f"{name} ({reason})" for name, reason in failed) + ")"
        if lazy.get("lazy_servers"):
            summary += f" ({lazy['lazy_servers']} lazy, not spawned yet)"
        logger.info(summary)


def register_mcp_servers(servers: Dict[str, dict]) -> List[str]:
    """Connect ``{name: config}`` servers and register their tools; idempotent for connected
    names, ``enabled: false`` skipped without disconnecting. Returns every MCP tool name."""
    if not _core._ensure_mcp_sdk():
        logger.debug("MCP SDK not available -- skipping explicit MCP registration")
        return []
    servers = _config._filter_suspicious_mcp_servers(servers)
    try:
        return _register_mcp_servers(servers)
    finally:
        # An owner's scoped reload orphaned adopters of its shared connections: now that this
        # pass (its rediscovery) is done, give them their tools back under their own scope.
        _lifecycle._reregister_orphaned_adopters()


def _register_mcp_servers(servers: Dict[str, dict]) -> List[str]:
    scoped_healed = _registration.register_connected_into_current_scope(servers)
    if not servers:
        logger.debug("No explicit MCP servers provided")
        return _registration._existing_tool_names() if _core._mcp_registry_scope() is not None else []
    new_servers = _select_new_servers(servers)
    if scoped_healed:
        logger.info("MCP: registered %d already-connected server(s) into this profile scope", scoped_healed)
    if not new_servers:
        return _registration._existing_tool_names()
    new_servers, lazy_registered, lazy_server_count = _register_lazy_from_cache(new_servers)
    if not new_servers:
        if lazy_registered:
            logger.info("MCP: registered %d lazy tool(s) from schema cache (no processes spawned)",
                        lazy_registered)
        return _registration._existing_tool_names()
    _loop._ensure_mcp_loop()
    _run_discovery_pass(new_servers)
    _log_summary("MCP: registered", new_servers, lazy_tools=lazy_registered, lazy_servers=lazy_server_count)
    return _registration._existing_tool_names()


def _acquire_discovery_lock_with_retry():
    """Cross-process guard: a lock loser waits for the holder then discovers itself; unavailable
    locking or an expired wait runs unguarded (fail-soft). None / _LOCK_UNAVAILABLE = unguarded."""
    cookie = _loop._try_acquire_mcp_discovery_lock()
    if cookie is not None:
        return cookie
    logger.debug("Another process holds MCP discovery lock -- retrying with backoff")
    for _ in range(_core._MCP_DISCOVERY_LOCK_MAX_RETRIES):
        time.sleep(_core._MCP_DISCOVERY_LOCK_RETRY_DELAY_S)
        cookie = _loop._try_acquire_mcp_discovery_lock()
        if cookie is not None:
            break
    # Cross-process discovery guard (#62771). A lock loser waits for the holder, then performs its own
    # process-local discovery. If locking is unavailable or the bounded wait expires, preserve the previous
    # fail-soft behavior by running discovery unguarded.
    if cookie is None:
        logger.warning("MCP discovery lock still held after %d retries -- running discovery unguarded",
                       _core._MCP_DISCOVERY_LOCK_MAX_RETRIES)
    elif cookie is not _core._LOCK_UNAVAILABLE:
        logger.debug("Retry succeeded -- acquired MCP discovery lock")
    return cookie


def discover_mcp_tools(allowed_mcp_names: Optional[List[str]] = None) -> List[str]:
    """Entry point: load config, connect servers, register tools. [] without the ``mcp``
    package; idempotent (only servers missing from a previous call are retried).

    ``allowed_mcp_names``: spawn only the MCP servers named in it (built-in toolset names in the
    list simply don't match); ``None`` spawns every configured server. Used by
    ``hermes -z -t <toolsets>`` to skip cold-starting servers the caller doesn't need (10-60s
    each); it only affects which servers start, not which names ``-t`` validation can see."""
    with _owner_secret_scope():
        servers = _config._load_mcp_config()
    if not servers:
        logger.debug("No MCP servers configured")
        return []
    if allowed_mcp_names is not None:
        allowed_set = {str(n) for n in allowed_mcp_names}
        filtered = {name: cfg for name, cfg in servers.items() if name in allowed_set}
        if len(filtered) != len(servers):
            logger.debug("MCP discovery filter: spawning %d/%d configured server(s) per --toolsets filter "
                         "(skipped: %s)", len(filtered), len(servers), ",".join(sorted(set(servers) - set(filtered))))
        servers = filtered
        if not servers:
            logger.debug("No MCP servers in --toolsets filter; skipping MCP load entirely")
            return []
    # SDK import deferred to here so a config without servers — or a -t filter that keeps
    # none — never pays it.
    if not _core._ensure_mcp_sdk():
        logger.debug("MCP SDK not available -- skipping MCP tool discovery")
        return []
    cookie = _acquire_discovery_lock_with_retry()
    try:
        with _core._lock:
            keys = {name: _resolve_server_key(name) for name in servers}
            new_server_names = [name for name, cfg in servers.items()
                                if keys[name] not in _core._servers and keys[name] not in _core._server_connecting
                                and _enabled(cfg)]
            prior_lazy = set(_core._lazy_server_configs)
        tool_names = register_mcp_servers(servers)
        if new_server_names:
            # A lazily registered server never connected, so it must not be counted as failed
            # (the old summary read "N failed" for a healthy all-lazy config, #111717). Reporting
            # it separately also keeps an already-lazy server from being re-announced on a
            # repeat discovery. Lazy state is keyed by resolved server key, not by name.
            with _core._lock:
                lazy_now = [n for n in new_server_names if keys[n] in _core._lazy_server_configs]
                newly_lazy = [n for n in lazy_now if keys[n] not in prior_lazy]
                lazy_tools = sum(len(_core._lazy_server_tool_names.get(keys[n], [])) for n in newly_lazy)
            _log_summary("  MCP:", [n for n in new_server_names if n not in lazy_now],
                         lazy_tools=lazy_tools, lazy_servers=len(newly_lazy))
        return tool_names
    finally:
        if cookie not in (None, _core._LOCK_UNAVAILABLE):
            cookie.release()


def reconcile_mcp_servers_with_config() -> Dict[str, List[str]]:
    """Bring the live server set in step with ``mcp_servers`` as it is on disk NOW: tear down
    servers that were removed from config or set ``enabled: false`` (a parked server keeps
    self-probing forever otherwise — for hours after the user deleted its entry), then connect
    anything enabled that is not live via :func:`discover_mcp_tools` — newly configured, or one
    whose earlier connect failed and whose cooldown has lapsed. Scoped to the current registry
    scope (one multiplexed profile's config prunes only its own connections). A lazily registered
    (schema-cache) server loses its cached tools; one still mid-connect cannot be torn down yet and
    is reported under ``"pending"`` so the caller retries. Returns
    ``{"removed": [...], "added": [...], "pending": [...]}``; a no-op when nothing changed."""
    with _owner_secret_scope():
        servers = _config._load_mcp_config()
    wanted = {name for name, cfg in servers.items() if _enabled(cfg)}
    scope = _core._mcp_registry_scope()
    with _core._lock:
        owned = [key for key, owner in _core._server_scope_keys.items() if owner == scope]
        live = {_key_name(key) for key in owned if key in _core._servers}
        connecting = {_key_name(key) for key in owned if key in _core._server_connecting}
        lazy = {key for key in _core._lazy_server_configs
                if _key_scope(key) == scope and _key_name(key) not in wanted}
    stale = sorted(live - wanted)
    if stale:
        logger.info("MCP server(s) %s no longer in config (or disabled); disconnecting", ", ".join(stale))
        _lifecycle.shutdown_mcp_servers(scope=scope, names=set(stale))
    for key in lazy:
        _forget_lazy_server(key)
    with _core._lock:
        # Same resolution ``_select_new_servers`` applies: this scope's own connection OR a shared
        # one it adopted from another profile counts as live. Owner==scope alone misses the adopted
        # case, so a multiplexed profile would re-enter discovery (cross-process lock) and log
        # "added" every tick forever for a server that is already serving it.
        known = {name for name in wanted
                 if (key := _resolve_server_key(name, scope, current=False)) in _core._servers
                 or key in _core._server_connecting or key in _core._lazy_server_configs}
    # A configured server that is not live is retried here — this is the only reviver for one whose
    # FIRST connect failed (#112445) — but only once its connect cooldown lapsed: ``discover_mcp_tools``
    # would skip it anyway, and entering it takes the cross-process discovery lock (up to 120 s of
    # waiting when another process holds it) and logs a failed pass, every tick, for nothing.
    added = sorted(name for name in wanted - known if not _connect_cooldown_active(name))
    if added:
        discover_mcp_tools()
    return {"removed": stale + sorted(_key_name(k) for k in lazy), "added": added,
            "pending": sorted(connecting - wanted)}


def _forget_lazy_server(key) -> None:
    """Drop a schema-cache (lazy) registration whose config entry is gone: its cached tools would
    otherwise stay callable and spawn the server on first use."""
    with _core._lock:
        _core._lazy_server_configs.pop(key, None)
        _core._lazy_server_fingerprints.pop(key, None)
        cached_names = _core._lazy_server_tool_names.pop(key, None) or []
    for tool_name in cached_names:
        _registration._deregister_mcp_tool_all_scopes(key, tool_name)


def is_mcp_tool_parallel_safe(tool_name: str) -> bool:
    """True when the tool's server opted into ``supports_parallel_tool_calls`` (provenance
    captured at registration, never the ambiguous ``mcp__{server}__{tool}`` shape)."""
    if not tool_name.startswith(MCP_TOOL_NAME_PREFIX):
        return False
    with _core._lock:
        server_name = _core._mcp_tool_server_names.get(tool_name)
        return bool(server_name and _server_key(server_name) in _core._parallel_safe_servers)


def get_mcp_status(configured: Optional[Dict[str, dict]] = None, *, include_runtime: bool = True) -> List[dict]:
    """Per-server status dicts for banner/TUI: name, transport, tools, connected, disabled,
    status (connected / disabled / connecting / failed / lazy / configured) and error for failed.
    Reads cached runtime state only; never connects.

    ``lazy`` is a registered-but-not-spawned server (``lazy: true``, tools from the schema
    cache): its tools are callable and the process starts on first use. Reporting it as
    ``configured`` (never registered) misreads a working setup (#111717)."""
    if configured is None:
        with _owner_secret_scope():
            configured = _config._load_mcp_config()
    else:
        configured = dict(configured)
    if not configured:
        return []
    current_scope = _core._mcp_registry_scope()
    with _core._lock:
        def visible(key) -> bool:
            # Runtime state belongs to the profile that adopted it; under a multiplexer only that
            # profile's view may show it, and ``include_runtime=False`` hides the launch profile's
            # servers from a status read scoped to a different profile.
            return include_runtime and _core._server_visible_in_scope(key, current_scope)

        active_servers = {_key_name(k): s for k, s in _core._servers.items() if visible(k)}
        connecting = {_key_name(k) for k in _core._server_connecting if visible(k)}
        connect_errors = {_key_name(k): e for k, e in _core._server_connect_errors.items() if visible(k)}
        # A lazy registration is not a live connection: ``_server_visible_in_scope`` reads the
        # adoption/teardown maps a lazy server never populates, so use the registration-level
        # predicate ``_resolve_server_key`` already relies on for this state.
        lazy_tools = {_key_name(k): len(v) for k, v in _core._lazy_server_tool_names.items()
                      if include_runtime and _key_visible_in_scope(k, current_scope)
                      and k in _core._lazy_server_configs}

    result: List[dict] = []
    for name, cfg in configured.items():
        enabled = _enabled(cfg)  # evaluated unconditionally: malformed values warn even when connected
        server = active_servers.get(name)
        live = server is not None and server.session is not None
        # An in-flight or failed first-use connect outranks "lazy": that server is no longer
        # merely waiting to be spawned, and the error is the actionable part.
        status = ("connected" if live else "disabled" if not enabled else "connecting" if name in connecting
                  else "failed" if name in connect_errors else "lazy" if name in lazy_tools else "configured")
        entry = {"name": name, "transport": cfg.get("transport", "http") if "url" in cfg else "stdio",
                 "tools": 0, "connected": False, "disabled": status == "disabled", "status": status}
        if live:
            entry["connected"] = True
            entry["tools"] = (len(server._registered_tool_names) if hasattr(server, "_registered_tool_names")
                              else len(server._tools))
            if server._sampling:
                entry["sampling"] = dict(server._sampling.metrics)
        elif status == "failed":
            entry["error"] = connect_errors[name]
        elif status == "lazy":
            entry["tools"] = lazy_tools[name]
        result.append(entry)
    return result


def mcp_server_reconnecting(name: str) -> bool:
    """True when this profile's connection to *name* connected once in this process and is now
    between sessions (degraded/parked) after a transient failure: the run task is alive and
    self-probing, so the outage is environmental and heals on its own. A server that never
    connected here (wrong URL, other profile's credentials) is not reconnecting, and neither is one
    parked on a PERMANENT error (revoked credentials, endpoint gone): its self-probe fails the same
    way every time, so callers must treat it as blocked rather than wait forever. Reads cached
    state; never connects."""
    with _core._lock:
        server = _core._servers.get(_resolve_server_key(name))
    if server is None or server.session is not None or not server._ever_connected:
        return False
    if server._park_reason and "permanent" in server._park_reason:
        return False
    return server._task is None or not server._task.done()


def probe_mcp_server_tools() -> Dict[str, List[tuple]]:
    """Connect each enabled server, list ``(tool_name, description)``, disconnect; nothing is
    registered and failed servers are omitted."""
    if not _core._ensure_mcp_sdk():
        return {}
    with _owner_secret_scope():
        enabled = {k: v for k, v in (_config._load_mcp_config() or {}).items() if _enabled(v)}
    if not enabled:
        return {}
    _loop._ensure_mcp_loop()
    result: Dict[str, List[tuple]] = {}
    probed_servers: List[_core.MCPServerTask] = []

    async def _probe_all():
        coros = [asyncio.wait_for(_connect_server(name, cfg),
                                  timeout=cfg.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT))
                 for name, cfg in enabled.items()]
        outcomes = await asyncio.gather(*coros, return_exceptions=True)
        for name, outcome in zip(enabled, outcomes):
            if isinstance(outcome, Exception):
                logger.debug("Probe: failed to connect to '%s': %s", name, outcome)
                continue
            probed_servers.append(outcome)
            result[name] = [(t.name, getattr(t, "description", "") or "") for t in outcome._tools]
        await asyncio.gather(*(s.shutdown() for s in probed_servers), return_exceptions=True)

    try:
        _loop._run_on_mcp_loop(_probe_all, timeout=120)
    except Exception as exc:
        logger.debug("MCP probe failed: %s", exc)
    finally:
        _lifecycle._stop_mcp_loop_if_idle()
    return result


def has_registered_mcp_tools() -> bool:
    """True if any MCP server has registered TOOLS (not merely connected), so the per-turn
    refresh hook stays idle for zero-tool servers."""
    with _core._lock:
        return bool(_core._mcp_tool_server_names)


def get_registered_mcp_server_names() -> set:
    """Server names that registered at least one tool (live, filtered — not config.yaml)."""
    with _core._lock:
        return set(_core._mcp_tool_server_names.values())
