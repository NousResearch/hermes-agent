"""Transport bring-up for MCPServerTask: stdio spawn (OSV preflight, cached-npx swap, child PID
ledger + death-supervisor registration), Streamable HTTP / SSE connect (preflight, identity header, client certs, OAuth),
protocol negotiation and initial tool discovery. Split from tools/mcp_tool.py."""

import logging
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set
from tools.mcp_tool_errors import NonMcpEndpointError, _apply_identity_header, _handshake_rejected_as_modern, _make_redirect_header_stripper, _resolve_client_cert
from tools.mcp_tool_lifecycle import _filter_mcp_children, _orphan_stdio_pid_servers, _orphan_stdio_pids, _stdio_pgids, _stdio_pids
from tools.mcp_tool_common import _core
from tools import mcp_tool_config as _config
from tools import mcp_tool_lifecycle as _lifecycle
from tools import mcp_tool_registration as _registration

logger = logging.getLogger("tools.mcp_tool")

_PROBE_INITIALIZE_BODY = (  # JSON-RPC ``initialize`` body for the content-type preflight POST
    '{"jsonrpc":"2.0","id":"_probe","method":"initialize","params":{"protocolVersion":"2025-03-26",'
    '"capabilities":{},"clientInfo":{"name":"hermes-probe","version":"0.1"}}}')


def _content_type_base(resp) -> str:
    """``content-type`` header of *resp* without parameters, lowercased."""
    return resp.headers.get("content-type", "").split(";")[0].strip().lower()


def _is_2xx(resp) -> bool:
    return 200 <= resp.status_code < 300


def _present(**kwargs) -> dict:
    """*kwargs* minus the ``None`` values (optional httpx client arguments)."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _pgroup_alive(pgid: Optional[int]) -> bool:
    """Signal 0 to the group succeeds iff any member is alive (POSIX only)."""
    try:
        os.killpg(pgid, 0)  # windows-footgun: ok — guarded by AttributeError below
        return True
    except (AttributeError, TypeError, OSError):  # non-POSIX / pgid None / gone
        return False


class MCPServerTransportMixin:
    """Methods of :class:`tools.mcp_tool.MCPServerTask` (mixed in; relies on its attributes)."""

    __slots__ = ()

    def _advertises_tools(self) -> bool:
        """False only when captured capabilities omit ``tools`` (prompt-/resource-only servers,
        where ``tools/list`` raises -32601); True without capability info (legacy fallback).

        Per the MCP spec, ``InitializeResult.capabilities.tools`` is non-None iff the server implements the
        ``tools/*`` request family. Prompt-only or resource-only servers omit it, and calling ``tools/list``
        against them raises ``MCPError(-32601 Method not found)`` — which previously killed the connection
        during discovery and made every keepalive fail. (Ported from anomalyco/opencode#31271.)
        """
        caps = getattr(self.initialize_result, "capabilities", None)
        return caps is None or getattr(caps, "tools", None) is not None

    def _session_kwargs(self) -> dict:
        """ClientSession kwargs: sampling, elicitation, notification + logging callbacks."""
        kwargs = {}
        for handler in (self._sampling, self._elicitation):
            if handler:
                kwargs.update(handler.session_kwargs())
        if _core._MCP_NOTIFICATION_TYPES and _core._MCP_MESSAGE_HANDLER_SUPPORTED:
            kwargs["message_handler"] = self._make_message_handler()
        if _core._MCP_LOGGING_CALLBACK_SUPPORTED:
            kwargs["logging_callback"] = self._make_logging_callback()
        return kwargs

    async def _negotiate_session(self, session, connect_timeout: float):
        """Negotiate the protocol era (``initialize`` vs ``server/discover``; both expose
        ``.capabilities``). ``auto`` tries the legacy handshake FIRST, falling back to discover only
        on a modern-only signal (-32022 / initialize -32601) — the reverse of the SDK's discover-first
        mode, so handshake-era servers pay zero extra round-trips. ``stateless`` probes discover first
        (one legacy retry on any error); ``legacy`` is handshake only. A TIMEOUT never falls back."""
        def call(method: str):
            return asyncio.wait_for(getattr(session, method)(), timeout=connect_timeout)

        async def attempt(primary, fallback, should_fallback, log_fmt, *log_extra):
            try:
                return await call(primary)
            except Exception as exc:
                if isinstance(exc, asyncio.TimeoutError) or not should_fallback(exc):
                    raise
                logger.info(log_fmt, self.name, exc, *log_extra)
                return await call(fallback)
        mode = str((self._config or {}).get("protocol", "auto")).lower().strip()
        if mode in ("stateless", "modern", "2026-07-28"):
            return await attempt("discover", "initialize", lambda exc: True,
                                 "MCP server '%s': server/discover rejected (%s) despite "
                                 "protocol=%s — falling back to the legacy handshake", mode)
        if mode in ("legacy", "handshake"):
            return await call("initialize")
        if mode != "auto":
            logger.warning("MCP server '%s': unknown protocol=%r — treating as 'auto' "
                           "(valid: auto, stateless, legacy)", self.name, mode)
        # mcp 1.x has no server/discover client — nothing to fall back to.
        return await attempt(
            "initialize", "discover", lambda exc: _handshake_rejected_as_modern(exc) and hasattr(session, "discover"),
            "MCP server '%s': legacy handshake rejected (%s) — retrying via server/discover (2026-07-28 stateless server)")

    async def _serve_session(self, session, connect_timeout: float,
                             label: str = "", mark_lifecycle: bool = False) -> str:
        """Handshake, discover, publish readiness, then serve until a lifecycle event. Clears stale
        breaker state but leaves the session UNPROVEN: flapping transports handshake fine and drop
        moments later, so only keepalive/tool-call success clears the reconnect budget."""
        self.initialize_result = await self._negotiate_session(session, connect_timeout)
        self.session = session
        if mark_lifecycle:
            self._mark_lifecycle_started()
        await self._discover_tools()
        self._ready.set()
        self._ever_connected = True
        _core._reset_server_error(self.name)
        # Session is live again: clear any breaker state from a prior outage so the first call after
        # recovery isn't gated on a stale consecutive-failure count (#16788).
        # A completed handshake alone is NOT proof of health: a flapping transport can handshake fine and
        # drop moments later, forever (#62212). The session must prove itself (keepalive success or a
        # successful tool call) before the reconnect budget is cleared — see _mark_session_proven.
        # Session is live again: clear any breaker state from a prior outage so the first call after
        # recovery isn't gated on a stale consecutive-failure count (#16788).
        # Unproven until keepalive/tool-call success (#62212).
        # Session is live again: clear any breaker state from a prior outage so the first call after
        # recovery isn't gated on a stale failure count (#16788).
        # Unproven until keepalive/tool-call success (#62212).
        # Session is live again: clear any breaker state from a prior outage so the first call after
        # recovery isn't gated on a stale consecutive-failure count (#16788).
        # Unproven until keepalive/tool-call success (#62212).
        self._session_proven = False
        reason = await self._wait_for_lifecycle_event()
        if label and reason == "reconnect":
            logger.info("MCP server '%s': reconnect requested — tearing down %s session", self.name, label)
        return reason

    async def _serve_transport(self, transport_cm, label: str, connect_timeout: float) -> str:
        """Open *transport_cm*, wrap its streams in a ClientSession and serve it. Streams are indexed,
        not unpacked (mcp 1.x yields a 3-tuple, 2.x a pair); a TaskGroup drop maps to ``"reconnect"``."""
        try:
            async with transport_cm as _streams:
                async with _core.ClientSession(_streams[0], _streams[1], **self._session_kwargs()) as session:
                    return await self._serve_session(session, connect_timeout, label)
        except BaseExceptionGroup as _eg:
            return self._reconnect_or_reraise_group(_eg)

    # ------------------------------------------------------------------ stdio

    def _track_spawned_children(self, new_pids: Set[int]) -> None:
        """Ledger the freshly spawned stdio children (pids, pgids, machine spawn ledger). pgids are
        captured while alive (getpgid fails after exit; the sweep needs them for reparented descendants)."""
        new_pgids: Dict[int, int] = {}
        for pid in new_pids:
            try:
                new_pgids[pid] = os.getpgid(pid)
            except ProcessLookupError:
                # Raced and already exited. The SDK spawns with start_new_session=True, so the
                # child was its own group leader (pgid == pid): keep that group covered — any
                # descendant it left behind still has to be reaped; the prune forgets the group
                # once nothing in it is alive.
                new_pgids[pid] = pid
            except (AttributeError, OSError):  # Windows (os.getpgid is POSIX-only)
                pass
        with _core._lock:
            _stdio_pids.update(dict.fromkeys(new_pids, self.name))
            _stdio_pgids.update(new_pgids)
        # Machine spawn ledger (startup sweeps reap orphans after an unclean exit); best-effort.
        for _pid in new_pids:
            try:
                from hermes_cli.process_identity import register_child
                register_child(_pid, "mcp-helper")
            except Exception:
                logger.debug("spawn-ledger register_child failed for MCP helper pid %s", _pid, exc_info=True)
        # Hand the pgroups to the shared parent-death supervisor so an ungraceful exit of this
        # process (kill -9, crash, force-quit) can't leave this server — or its descendants, e.g.
        # mcp-remote's spawned `node` — running forever. The graceful paths (shutdown,
        # _kill_orphaned_mcp_children) still reap as before; this only covers when they never run.
        _core._update_death_supervisor("register", new_pgids.values())

    def _release_spawned_children(self, new_pids: Set[int]) -> None:
        """Drop the ledger entries; a child (or its pgroup) still alive means SDK teardown failed
        (common on mid-way cancel on Linux: setsid() children escape) — mark it orphaned for the sweep."""
        from gateway.status import _pid_exists
        # Groups with nothing left alive; the supervisor forgets them after the lock is released.
        # Groups still alive stay registered on purpose, so the supervisor still reaps them if this
        # process dies before the orphan sweep runs.
        released_pgids: list = []
        with _core._lock:
            for pid in new_pids:
                _stdio_pids.pop(pid, None)
                # Windows-safe pid probe; the child may be gone while descendants remain in its pgroup.
                if _pid_exists(pid) or _pgroup_alive(_stdio_pgids.get(pid)):
                    _orphan_stdio_pids.add(pid)
                    _orphan_stdio_pid_servers[pid] = self.name
                else:  # nothing to reap — drop the pgid so PID reuse can't surface stale pgroup state
                    dropped = _stdio_pgids.pop(pid, None)
                    if dropped is not None:
                        released_pgids.append(dropped)
        _core._update_death_supervisor("unregister", released_pgids)

    async def _run_stdio(self, config: dict):
        """Run the server using stdio transport."""
        if config.get("identity_header") is not None:  # copy-pasted HTTP block: warn, don't mislead
            logger.warning("MCP server '%s': identity_header is only supported on "
                           "HTTP/SSE transports — ignored for stdio servers", self.name)
        if not _core._ensure_mcp_sdk():
            raise ImportError(f"MCP server '{self.name}' requires the 'mcp' Python SDK, but "
                              "it is not installed. Run `hermes setup` to install MCP support, then retry.")
        command = config.get("command")
        if not command:
            raise ValueError(f"MCP server '{self.name}' has no 'command' in config")
        command, safe_env = _config._resolve_stdio_command(command, _config._build_safe_env(config.get("env")))
        # OSV malware preflight, then the cached-npx swap (ordering enforced there).
        command, args = await _core._preflight_stdio_command(self.name, command, config.get("args", []))
        server_params = _core.StdioServerParameters(
            command=command, args=args, env=safe_env or None, cwd=config.get("cwd"),
            # Windows pipes can split non-UTF-8 bytes at chunk boundaries; substitute, don't raise.
            encoding_error_handler="replace")
        # Reap orphans of prior attempts first (else retries pile up zombie pairs); unscoped on purpose;
        # off-loop because the reaper blocks up to 2s.
        await asyncio.to_thread(_lifecycle._kill_orphaned_mcp_children)
        pids_before = _lifecycle._snapshot_child_pids()  # so the new child can be identified after spawn
        # Reap any orphaned subprocesses from prior failed connection attempts before spawning a new one.
        # Without this, each retry in the run() reconnect loop spawns a fresh process pair while the
        # previous failed pair lingers — leading to rapid zombie accumulation (see #57355, #57228). The
        # unscoped sweep also opportunistically reaps orphans left by *other* servers that never reconnect;
        # per-server filtering via ``server_name`` remains available for scoped call sites. Run in a worker
        # thread: the reaper blocks up to 2s (SIGTERM → wait → SIGKILL) when orphans exist, which would
        # otherwise stall the shared MCP event loop.
        new_pids: set = set()
        # Subprocess stderr goes to ~/.hermes/logs/mcp-stderr.log so banners can't corrupt the TUI.
        _config._write_stderr_log_header(self.name)
        try:
            errlog = _config._get_mcp_stderr_log()
            async with _core.stdio_client(server_params, errlog=errlog) as (read_stream, write_stream):
                # New PIDs for force-kill cleanup, minus non-MCP children (slash_worker, LSP) racing
                # into the window: they share the TUI's pgid — leaking them would killpg() the TUI.
                new_pids = _filter_mcp_children(_lifecycle._snapshot_child_pids() - pids_before)
                if new_pids:
                    self._track_spawned_children(new_pids)
                self._stdio_child_pids = set(new_pids)  # so in-flight calls fail fast when the child dies
                async with _core.ClientSession(read_stream, write_stream, **self._session_kwargs()) as session:
                    # Bound the handshake here (``connect_timeout`` only bounds the caller's ``.result()``):
                    # a server that never answers ``initialize`` would leak child + pipes per retry until EMFILE.
                    connect_timeout = float(config.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT))
                    return await self._serve_session(session, connect_timeout, mark_lifecycle=True)
        finally:  # clean exit, exceptions AND cancellation
            if new_pids:
                self._release_spawned_children(new_pids)

    # ------------------------------------------------------------------- HTTP

    async def _preflight_content_type(self, url: str, *, headers: Optional[dict] = None,
                                      ssl_verify: bool = True, client_cert=None, timeout: float = 5.0) -> None:
        """Probe *url* before the SDK connects: a plain web page would make the SDK sit out the full
        ``connect_timeout`` before an opaque ``CancelledError``; this raises NonMcpEndpointError within
        ``timeout``. Allow-list based: only a 2xx with a definite non-MCP content type is rejected, and
        only after a JSON-RPC ``initialize`` POST also fails to look like MCP (some servers serve a UI
        on GET but speak MCP via POST). Anything else passes — the handshake stays the source of truth.
        Own httpx client, OUTSIDE the SDK's anyio task group, so the error isn't group-wrapped."""
        try:
            import httpx as _httpx
        except ImportError:
            return  # No httpx → skip probe; SDK import would have failed first.

        def _non_mcp_2xx(resp) -> bool:
            # Only judge 2xx (4xx/5xx may be an auth challenge); no content type advertised → trust the SDK.
            ct = _content_type_base(resp)
            return _is_2xx(resp) and bool(ct) and ct not in self._MCP_CONTENT_TYPES
        probe_headers = dict(headers) if headers else {}
        try:
            async with _httpx.AsyncClient(verify=ssl_verify, follow_redirects=True, timeout=_httpx.Timeout(timeout),
                                          **_present(cert=client_cert)) as client:
                resp = await client.head(url, headers=probe_headers)  # cheapest; GET on 405/501
                if resp.status_code in (405, 501):
                    resp = await client.get(url, headers=probe_headers)
                # Non-MCP content type on HEAD/GET: try a JSON-RPC POST so POST-only servers pass.
                if _non_mcp_2xx(resp):
                    post_resp = await client.post(
                        url, content=_PROBE_INITIALIZE_BODY,
                        headers={**probe_headers, "Content-Type": "application/json",
                                 "Accept": "application/json, text/event-stream"})
                    if _is_2xx(post_resp) and _content_type_base(post_resp) in self._MCP_CONTENT_TYPES:
                        resp = post_resp
        except _httpx.HTTPError:
            return  # DNS/connect/timeout/transport error — let the SDK try.
        if not _non_mcp_2xx(resp):
            return
        ct_base = _content_type_base(resp)
        raise NonMcpEndpointError(f"MCP server '{self.name}' at {url} returned Content-Type '{ct_base}', not an MCP "
            f"response (expected one of: {', '.join(self._MCP_CONTENT_TYPES)}). The URL most likely "
            "points at a web page rather than an MCP endpoint — check it resolves to a Streamable "
            "HTTP / SSE endpoint (e.g. https://host/mcp, not https://host/).")

    def _reconnect_or_reraise_group(self, eg: BaseExceptionGroup) -> str:
        """Map an SDK transport TaskGroup failure to a clean ``"reconnect"``: HTTP/SSE stream pumps run in an anyio
        TaskGroup, so a transient drop escapes as a ``BaseExceptionGroup`` that would otherwise park the server for
        300s over a sub-second glitch. Re-raise when it is not one: shutdown in progress (``_shutdown_event`` is
        set before cancel), KeyboardInterrupt/SystemExit or a real CancelledError in the group, or no live session
        this attempt (``_ready`` unset — connect failures must back off, not hot-loop).

        Streamable-HTTP / SSE transports run their stream pump inside an anyio TaskGroup. A transient stream
        drop (idle timeout, brief backend blip, server-side TCP close) surfaces as a ``BaseExceptionGroup``
        escaping the transport context manager. Left unwrapped it reaches ``run()``'s error path, which
        applies exponential backoff and eventually *parks* the server for 300s and deregisters its tools — a
        multi-minute tool outage for what is usually a sub-second glitch while the POST path stays healthy
        (issue #66092).
        - the group carries a ``KeyboardInterrupt`` / ``SystemExit`` — fatal signals must propagate to the
        interpreter, never be converted into a reconnect; - the group carries a real ``CancelledError``
        (task cancellation must propagate to asyncio, mirroring the ``run()`` guard for #9930); - we never
        reached a live session this attempt (``_ready`` unset) — a connect/handshake failure SHOULD fall
        through to ``run()``'s backoff rather than hot-loop reconnects against a broken endpoint.
        """
        if (self._shutdown_event.is_set()
                or eg.split((KeyboardInterrupt, SystemExit))[0] is not None
                or eg.split(asyncio.CancelledError)[0] is not None
                or not self._ready.is_set()):
            raise eg
        logger.debug("MCP server '%s': transport TaskGroup exited after a live session "
                     "(%r) — reconnecting immediately instead of backing off", self.name, eg)
        return "reconnect"

    def _build_oauth_auth(self, url: str, config: dict):
        """OAuth 2.1 PKCE via the central MCPOAuthManager (one provider reused across reconnects and
        CLI paths). Setup failures re-raise (after a warning) so only this server is reported failed."""
        if self._auth_type != "oauth":
            return None
        try:
            from tools.mcp_oauth_manager import get_manager
            return get_manager().get_or_build_provider(self.name, url, config.get("oauth"))
        except Exception as exc:
            logger.warning("MCP OAuth setup failed for '%s': %s", self.name, exc)
            raise

    def _sse_transport(self, url: str, headers: dict, connect_timeout: float,
                       ssl_verify, client_cert, oauth_auth, strict_cfg_headers: bool):
        """``sse_client`` context manager for ``transport: sse`` entries."""
        if strict_cfg_headers:  # fail closed: SSE cannot enforce the redirect boundary
            raise ValueError(f"MCP server '{self.name}': strict_redirect_headers is "
                             "not supported on the SSE transport.")
        if _core.sse_client is None:
            raise ImportError(f"MCP server '{self.name}' requires SSE transport but "
                              "mcp.client.sse.sse_client is not available. "
                              "Upgrade the mcp package to get SSE support.")
        # sse_read_timeout bounds the gap between events: SSE servers idle for minutes, so 300s (the
        # Streamable HTTP read timeout), not tool_timeout. ``auth`` must be forwarded or OAuth SSE 401s silently.
        sse_kwargs: dict = {"url": url, "headers": headers or None, "timeout": float(connect_timeout),
                            "sse_read_timeout": 300.0, **_present(auth=oauth_auth)}
        if client_cert is not None or ssl_verify is not True:
            # sse_client has no verify/cert kwargs: an httpx_client_factory forwards the SDK's (headers,
            # auth, timeout) and layers TLS on top. Client MUST come from the SDK's httpx (httpx2 on mcp >= 2.0).
            _httpx_mod = _core.sdk_httpx()
            sse_kwargs["httpx_client_factory"] = lambda headers=None, timeout=None, auth=None: _httpx_mod.AsyncClient(
                follow_redirects=True, verify=ssl_verify,
                timeout=timeout if timeout is not None else _httpx_mod.Timeout(30.0, read=300.0),
                **_present(headers=headers, auth=auth, cert=client_cert))
        return _core.sse_client(**sse_kwargs)

    def _streamable_http_transport(self, url: str, headers: dict, connect_timeout: float,
                                   ssl_verify, client_cert, oauth_auth,
                                   strict_cfg_headers: bool, configured_header_names: set):
        """Streamable HTTP context manager: mcp >= 1.24.0 gets a caller-owned httpx client; on the
        deprecated API (mcp < 1.24.0) the SDK owns the client."""
        if not _core._MCP_NEW_HTTP:
            if strict_cfg_headers:  # fail closed: without an owned client redirects can't be hooked
                raise ImportError(f"MCP server '{self.name}' requires mcp >= 1.24.0 to "
                                  "enforce the portable redirect-header boundary "
                                  "(strict_redirect_headers). Upgrade the mcp package.")
            return _core.streamablehttp_client(url, headers=headers, timeout=float(connect_timeout), verify=ssl_verify,
                                               **_present(auth=oauth_auth))
        # Explicit AsyncClient matching the SDK's create_mcp_http_client defaults; MUST come from the
        # SDK's httpx (httpx2 on mcp >= 2.0) since the SDK sends its own Requests through it.
        httpx = _core.sdk_httpx()
        _strip_auth_on_cross_origin_redirect = _make_redirect_header_stripper(
            httpx.URL(url), strict=strict_cfg_headers, configured_header_names=configured_header_names)
        client_kwargs: dict = {"follow_redirects": True, "timeout": httpx.Timeout(float(connect_timeout), read=300.0),
                               "verify": ssl_verify, **({"headers": headers} if headers else {}),
                               "event_hooks": {"response": [_strip_auth_on_cross_origin_redirect]},
                               **_present(auth=oauth_auth, cert=client_cert)}

        @asynccontextmanager
        async def _owned_client_streams():  # the SDK skips cleanup when http_client is provided
            async with httpx.AsyncClient(**client_kwargs) as http_client:
                async with _core.streamable_http_client(url, http_client=http_client) as streams:
                    yield streams
        return _owned_client_streams()

    async def _run_http(self, config: dict):
        """Run the server using HTTP/StreamableHTTP (or SSE) transport."""
        _core._ensure_mcp_sdk()
        if not _core._MCP_HTTP_AVAILABLE:
            raise ImportError(f"MCP server '{self.name}' requires HTTP transport but "
                              "mcp.client.streamable_http is not available. "
                              "Upgrade the mcp package to get HTTP support.")
        url = config["url"]
        headers = dict(config.get("headers") or {})
        # Agent Plugins v1 strict_redirect_headers: configured headers MUST NOT follow a cross-origin
        # redirect — capture their names BEFORE client-generated headers are merged in.
        configured_header_names = {key.lower() for key in headers}
        headers = _apply_identity_header(self.name, config, headers)  # explicit same-name headers win
        # Seed MCP-Protocol-Version (user override wins) from the HANDSHAKE version, not the latest: a
        # 2026-07-28 header routes the handshake-era ``initialize()`` onto the envelope ladder, which rejects it.
        if not any(key.lower() == "mcp-protocol-version" for key in headers):
            headers["mcp-protocol-version"] = _core.LATEST_HANDSHAKE_VERSION
        connect_timeout = config.get("connect_timeout", _core._DEFAULT_CONNECT_TIMEOUT)
        common = (url, headers, connect_timeout, config.get("ssl_verify", True), _resolve_client_cert(self.name, config),
                  self._build_oauth_auth(url, config), bool(config.get("strict_redirect_headers")))
        if config.get("transport") == "sse":
            transport, label = self._sse_transport(*common), "SSE"
        else:
            transport = self._streamable_http_transport(*common, configured_header_names)
            label = "HTTP" if _core._MCP_NEW_HTTP else "legacy HTTP"
        return await self._serve_transport(transport, label, float(connect_timeout))

    # -------------------------------------------------------------- discovery

    # Legacy Streamable-HTTP transport TaskGroup dropped: reconnect immediately instead of backoff/park
    # (#66092).
    async def _discover_tools(self):
        """Discover tools from the connected session. Capability-gated: prompt-/resource-only
        servers raise ``MCPError(-32601)`` on ``tools/list``, which would abort the connection.

        Skip the call when the server doesn't advertise the ``tools`` capability. (Ported from
        anomalyco/opencode#31271.)
        """
        self._ping_unsupported = False  # fresh transport: re-probe ``ping`` across the reconnect
        if self.session is None:
            return
        if not self._advertises_tools():
            logger.info("MCP server '%s': does not advertise 'tools' capability — "
                        "skipping tools/list (prompts/resources remain available)", self.name)
            self._tools = []
        else:
            async with self._rpc_lock:
                self._list_cache_meta = {}
                self._tools = await _core._paginate_full_list(
                    self.session.list_tools, "tools", self.name, cache_meta_out=self._list_cache_meta)
        self._register_discovered_tools_if_needed()

    def _register_discovered_tools_if_needed(self) -> None:
        """Publish freshly discovered tools when none are registered (initial registration normally happens in
        ``_discover_and_register_server``). Outage handling may clear ``_ready`` and deregister stale tools;
        ownership via ``_servers`` authorizes publishing before readiness is restored so a revival (or a server
        retained after a recoverable initial failure) never comes back with zero tools."""
        if self._registered_tool_names:
            return
        with _core._lock:
            owned = _core._servers.get(self.name) is self
        if not owned and not self._ready.is_set():
            return
        self._registered_tool_names = _registration._register_server_tools(self.name, self, self._config)
        with _core._lock:  # a retained initial-failure server that just published tools has recovered
            if _core._servers.get(self.name) is self:
                _core._server_connect_errors.pop(self.name, None)
