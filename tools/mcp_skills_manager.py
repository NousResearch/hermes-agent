"""Skills-over-MCP instance manager: capability gating, discovery, and verified reads.

The manager deliberately does not own a connection. It observes the ``MCPServerTask``
that ``tools.mcp_tool`` already runs, sends its requests through the same dispatch
and lock the ordinary MCP tools use, and never opens a second client — a private
client would miss the profile's credentials, its scope binding and its reconnect
budget, and would hand the model content from a different trust domain than the
tools it then invokes.

Three states are tracked separately per instance and must never be collapsed:
transport reachability, skills-channel support, and catalog freshness. A skills
listing that times out is not a disconnect, and an instance whose skills channel is
unsupported still serves its ordinary MCP tools.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Optional

from agent.skills_contracts import (
    BACKEND_MCP,
    BackendState,
    CatalogSnapshot,
    CatalogState,
    OriginInfo,
    SkillEntry,
    SkillError,
    SkillErrorCode,
    SkillIdentity,
    SkillsSupport,
    VerifiedRead,
)
from tools import mcp_skills_protocol as proto

logger = logging.getLogger(__name__)

METADATA_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 30.0
INITIAL_PRELOAD_BUDGET_SECONDS = 5.0
MAX_PAGES = 50
DEFAULT_TTL_MS = 300_000.0
MIN_REFRESH_INTERVAL_SECONDS = 60.0


def mcp_server_skills_enabled(config: Any) -> bool:
    """Whether this server opted in to the skills channel (``mcp_servers.<name>.skills.enabled``).

    Opt-in is explicit and per server: advertising the extension is not consent to
    have its catalog injected into every prompt, and the flag is read through one
    function so the connect trigger and the projection can never disagree.
    """
    if not isinstance(config, Mapping):
        return False
    block = config.get("skills")
    if not isinstance(block, Mapping):
        return False
    return bool(block.get("enabled"))


@dataclass
class InstanceView:
    """Read-only projection of one instance's skills state for UI/diagnostics."""

    server_name: str
    scope_id: str
    support: SkillsSupport
    support_reason: str
    transport_state: BackendState
    directory_read: bool
    protocol_version: str
    generation: int
    catalog_state: CatalogState
    entry_count: int
    snapshot_revision: int
    discovered_at: Optional[float]
    fresh_until: Optional[float]
    last_error: str
    quarantined: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "server": self.server_name, "scope": self.scope_id, "support": self.support.value,
            "support_reason": self.support_reason, "transport": self.transport_state.value,
            "directory_read": self.directory_read, "protocol_version": self.protocol_version,
            "generation": self.generation, "catalog": self.catalog_state.value,
            "entries": self.entry_count, "revision": self.snapshot_revision,
            "last_error": self.last_error, "quarantined": list(self.quarantined),
        }


@dataclass
class _Instance:
    server_name: str
    scope_id: str
    support: SkillsSupport = SkillsSupport.UNKNOWN
    support_reason: str = ""
    directory_read: bool = False
    protocol_version: str = ""
    session_token: Any = None
    generation: int = 0
    snapshot: Optional[CatalogSnapshot] = None
    catalog_state: CatalogState = CatalogState.ABSENT
    last_error: str = ""
    discovered_at: Optional[float] = None
    quarantined: tuple[str, ...] = ()
    refreshing: bool = False
    last_refresh_attempt: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def view(self, transport_state: BackendState) -> InstanceView:
        snapshot = self.snapshot
        return InstanceView(
            server_name=self.server_name, scope_id=self.scope_id, support=self.support,
            support_reason=self.support_reason, transport_state=transport_state,
            directory_read=self.directory_read, protocol_version=self.protocol_version,
            generation=self.generation, catalog_state=self.catalog_state,
            entry_count=len(snapshot.entries) if snapshot else 0,
            snapshot_revision=snapshot.snapshot_revision if snapshot else 0,
            discovered_at=self.discovered_at,
            fresh_until=snapshot.fresh_until if snapshot else None,
            last_error=self.last_error, quarantined=self.quarantined,
        )


class _McpSender:
    """The production sender: rides the connection ``tools.mcp_tool`` already owns.

    Two entry points because the manager is reached from two contexts and they are
    not interchangeable. ``send`` is awaited from code already running on the MCP
    loop (the connect hook) — bridging there would block the loop on itself.
    ``send_sync`` is called from an ordinary thread (``skill_view`` on the agent's
    turn thread, which has no running loop) and must hand the coroutine to the MCP
    loop rather than opening one of its own.
    """

    async def send(self, server_name: str, op: str, request: Any, result_type: Any, timeout: float) -> Any:
        from tools.mcp_tool_discovery import _get_connected_server_for_call
        from tools.mcp_tool_handlers import _track_inflight_rpc

        server = _get_connected_server_for_call(server_name)
        if server is None or getattr(server, "session", None) is None:
            raise SkillError(SkillErrorCode.UNAVAILABLE, f"MCP server '{server_name}' is not connected")
        async with _track_inflight_rpc(server, server_name, op, retry_safe=True):
            async with server._rpc_lock:
                session = getattr(server, "session", None)
                if session is None:
                    raise SkillError(SkillErrorCode.UNAVAILABLE,
                                     f"MCP server '{server_name}' disconnected before {op}")
                return await session.send_request(request, result_type,
                                                  request_read_timeout_seconds=timeout)

    def send_sync(self, server_name: str, op: str, request: Any, result_type: Any, timeout: float) -> Any:
        from tools.mcp_tool_loop import _run_on_mcp_loop

        return _run_on_mcp_loop(
            lambda: self.send(server_name, op, request, result_type, timeout),
            timeout=timeout + _BRIDGE_HEADROOM_SECONDS)


# The loop bridge polls in 100ms slices and cancels on its own deadline; a hair of
# headroom lets the request's own read timeout fire first, so the failure is reported
# as the server's timeout rather than the bridge's.
_BRIDGE_HEADROOM_SECONDS = 2.0


def _default_sender() -> _McpSender:
    """Lazily built so importing this module never drags in the MCP SDK."""
    return _McpSender()


class SkillsOverMcpManager:
    """One process-wide manager; instances are keyed by ``(scope, server label)``.

    The scope half of the key is not decoration: two profiles may configure a
    same-named server with different credentials, and connection adoption lets one
    physical transport serve both. Sharing a transport is fine; sharing a catalog,
    an approval or a cached body across trust boundaries is not.
    """

    def __init__(self, sender: Optional[Any] = None) -> None:
        self._instances: dict[tuple[str, str], _Instance] = {}
        self._lock = threading.Lock()
        self._sender = sender

    # ── instance state ──────────────────────────────────────────────────────

    def _instance(self, server_name: str, scope_id: str) -> _Instance:
        key = (scope_id, server_name)
        with self._lock:
            inst = self._instances.get(key)
            if inst is None:
                inst = _Instance(server_name=server_name, scope_id=scope_id)
                self._instances[key] = inst
            return inst

    def observe_connection(self, server_name: str, scope_id: str, *, initialize_result: Any,
                           session: Any, transport_state: BackendState = BackendState.READY) -> InstanceView:
        """Record what a (re)connect negotiated; bump the generation when the session changed.

        Called on every observation rather than only on a connect event: a transport
        that dropped and came back without anyone announcing it must invalidate the
        old catalog all the same, and a stale in-flight result must not resurrect it.
        """
        inst = self._instance(server_name, scope_id)
        token = id(session) if session is not None else None
        if token != inst.session_token:
            # New session: everything learned from the previous one is void. The
            # snapshot is dropped rather than kept as "stale", because a stale
            # catalog would still be routable and the model would act on a version
            # this host can no longer read.
            inst.session_token = token
            inst.generation += 1
            inst.snapshot = None
            inst.catalog_state = CatalogState.ABSENT
            inst.quarantined = ()
        verdict = proto.parse_capabilities(
            proto.capabilities_mapping(getattr(initialize_result, "capabilities", None)),
            protocol_version=_negotiated_version(session))
        inst.support = SkillsSupport(verdict.support)
        inst.support_reason = verdict.reason
        inst.directory_read = verdict.directory_read
        inst.protocol_version = verdict.protocol_version
        if not verdict.supported:
            inst.catalog_state = CatalogState.ABSENT
            inst.snapshot = None
        return inst.view(transport_state)

    def instance_view(self, server_name: str, scope_id: str) -> Optional[InstanceView]:
        with self._lock:
            inst = self._instances.get((scope_id, server_name))
        return None if inst is None else inst.view(BackendState.READY)

    def instances(self, scope_id: Optional[str] = None) -> list[InstanceView]:
        with self._lock:
            found = [i for (scope, _n), i in self._instances.items() if scope_id is None or scope == scope_id]
        return [i.view(BackendState.READY) for i in found]

    def snapshot(self, server_name: str, scope_id: str) -> Optional[CatalogSnapshot]:
        """Published catalog for one instance, or None. Never issues RPC."""
        with self._lock:
            inst = self._instances.get((scope_id, server_name))
        if inst is None or inst.snapshot is None:
            return None
        return inst.snapshot if inst.snapshot.usable() else None

    def published(self, scope_id: str) -> list[CatalogSnapshot]:
        """Every usable snapshot this scope may route to — the prompt projection's input."""
        with self._lock:
            insts = [i for (scope, _n), i in self._instances.items() if scope == scope_id]
        return [i.snapshot for i in insts if i.snapshot is not None and i.snapshot.usable()]

    def entries(self, scope_id: str) -> list[SkillEntry]:
        """All routable remote entries, ordered by originating server label then name."""
        rows: list[SkillEntry] = []
        for snapshot in sorted(self.published(scope_id), key=lambda s: s.origin_id):
            rows.extend(sorted(snapshot.entries.values(), key=lambda e: e.name))
        return rows

    def request_refresh(self, server_name: str, scope_id: str, *, reason: str = "stale",
                        min_interval: float = MIN_REFRESH_INTERVAL_SECONDS) -> bool:
        """Schedule a background catalog refresh for one instance; False when one is not due.

        Debounced on the *attempt* clock rather than on success, so a server that
        publishes a zero TTL and then fails cannot turn every prompt build into a retry
        storm. Fire-and-forget by necessity: the caller is a prompt build, and a prompt
        build must never wait on the network for metadata it can do without.
        """
        inst = self._instance(server_name, scope_id)
        with self._lock:
            now = time.monotonic()
            if inst.refreshing or (now - inst.last_refresh_attempt) < min_interval:
                return False
            inst.refreshing = True
        loop = _mcp_loop()
        if loop is None:
            inst.refreshing = False
            return False
        try:
            asyncio.run_coroutine_threadsafe(self._refresh_in_background(inst, reason), loop)
        except (RuntimeError, ValueError) as exc:
            inst.refreshing = False
            logger.debug("Could not schedule a skills refresh for '%s': %s", server_name, exc)
            return False
        return True

    async def _refresh_in_background(self, inst: _Instance, reason: str) -> None:
        try:
            await self.discover(inst.server_name, inst.scope_id, reason=reason)
        except SkillError:
            pass  # discover already logged it; the debounce clock governs the next attempt
        finally:
            inst.refreshing = False

    def mark_disconnected(self, server_name: str, scope_id: str, reason: str = "") -> None:
        """Drop the catalog when the transport is confirmed gone.

        The catalog is removed outright: keeping it would leave the model routing
        to skills this host cannot read, and no on-disk copy may stand in for the
        connection the user's consent was tied to.
        """
        with self._lock:
            inst = self._instances.get((scope_id, server_name))
            if inst is None:
                return
            inst.snapshot = None
            inst.catalog_state = CatalogState.ABSENT
            inst.last_error = reason or "connection lost"

    # ── discovery ───────────────────────────────────────────────────────────

    async def discover(self, server_name: str, scope_id: str, *, reason: str = "initial") -> CatalogSnapshot:
        """List the whole catalog, validate it, and publish it atomically.

        Single-flight per instance: concurrent callers await the same refresh rather
        than doubling the server's page load. A failure publishes nothing — the
        previous batch stays as the last known-good diagnostic, its
        ``refresh_status`` marked failed so nothing routes on it as current.
        """
        inst = self._instance(server_name, scope_id)
        async with inst.lock:
            return await self._discover_locked(inst, reason=reason)

    async def _discover_locked(self, inst: _Instance, *, reason: str) -> CatalogSnapshot:
        if inst.support is not SkillsSupport.SUPPORTED:
            raise SkillError(SkillErrorCode.UNSUPPORTED_EXTENSION,
                             f"MCP server '{inst.server_name}' does not serve the skills extension "
                             f"({inst.support_reason or inst.support.value})")
        generation = inst.generation
        inst.catalog_state = CatalogState.LOADING
        inst.last_refresh_attempt = time.monotonic()
        started = inst.last_refresh_attempt
        try:
            entries, diagnostics, ttl_ms = await self._list_all(inst)
        except SkillError as exc:
            inst.catalog_state = (CatalogState.REFRESH_FAILED if inst.snapshot is not None else CatalogState.ABSENT)
            inst.last_error = exc.message
            if inst.snapshot is not None:
                inst.snapshot = replace(inst.snapshot, refresh_status=f"failed: {exc.code.value}", diagnostics=exc.message)
            logger.warning("MCP server '%s': skills refresh (%s) failed: %s", inst.server_name, reason, exc.message)
            raise
        if generation != inst.generation:
            # The transport was replaced while we were paginating. Publishing now
            # would attach one session's listing to another session's identity.
            raise SkillError(SkillErrorCode.VERSION_CHANGED,
                             f"MCP server '{inst.server_name}' reconnected during skills discovery; retry")
        previous_revision = inst.snapshot.snapshot_revision if inst.snapshot else 0
        snapshot = CatalogSnapshot(
            origin_id=inst.server_name, scope_id=inst.scope_id,
            entries={e.identity.as_key(): e for e in entries},
            snapshot_revision=previous_revision + 1,
            connection_generation=generation,
            discovered_at=time.monotonic(),
            fresh_until=_fresh_until(ttl_ms),
            availability=BackendState.READY, support=SkillsSupport.SUPPORTED,
            refresh_status=f"ok ({reason})", pagination_finished=True, diagnostics=diagnostics)
        inst.snapshot = snapshot
        inst.catalog_state = CatalogState.FRESH if snapshot.is_fresh() else CatalogState.STALE
        inst.discovered_at = snapshot.discovered_at
        inst.last_error = ""
        logger.info("MCP server '%s': published skills catalog revision %d with %d entries in %.2fs",
                    inst.server_name, snapshot.snapshot_revision, len(entries), time.monotonic() - started)
        return snapshot

    async def _list_all(self, inst: _Instance) -> tuple[list[SkillEntry], str, Optional[float]]:
        """Follow ``nextCursor`` to the end, quarantining entries that fail validation."""
        seen_cursors: set[str] = set()
        cursor: Optional[str] = None
        entries: list[SkillEntry] = []
        quarantined: list[str] = []
        ttl_ms: Optional[float] = None
        for page in range(MAX_PAGES):
            result = await self._rpc_async(
                inst, proto.METHOD_LIST,
                proto.SkillsListRequest(params=proto.SkillsListParams(cursor=cursor)),
                proto.SkillsListResult)
            if page == 0:
                ttl_ms = _read_ttl_ms(result)
            for raw in result.skills:
                try:
                    entries.append(proto.parse_skill_entry(raw, origin_id=inst.server_name, label=inst.server_name))
                except SkillError as exc:
                    uri = raw.get("uri") if isinstance(raw, Mapping) else None
                    quarantined.append(f"{uri}: {exc.message}")
            nxt = result.nextCursor
            if not isinstance(nxt, str) or not nxt:
                inst.quarantined = tuple(quarantined)
                return entries, _diagnostics(entries, quarantined), ttl_ms
            if nxt in seen_cursors:
                raise SkillError(SkillErrorCode.INCOMPLETE_CATALOG,
                                 f"MCP server '{inst.server_name}' repeated pagination cursor {nxt!r}; "
                                 "refusing to publish a partial catalog")
            seen_cursors.add(nxt)
            cursor = nxt
        raise SkillError(SkillErrorCode.INCOMPLETE_CATALOG,
                         f"MCP server '{inst.server_name}' exceeded the {MAX_PAGES}-page discovery budget")

    # ── entry resolution and reads ──────────────────────────────────────────

    def find(self, scope_id: str, *, server: Optional[str] = None, uri: Optional[str] = None,
             name: Optional[str] = None) -> list[SkillEntry]:
        """Candidates for a locator, without any RPC.

        ``uri`` matches an entry's own SKILL.md URI; ``name`` matches the display
        name across the requested servers. Returns every candidate so the caller
        can refuse an ambiguous lookup instead of guessing an origin.
        """
        candidates: list[SkillEntry] = []
        for snapshot in self.published(scope_id):
            if server is not None and snapshot.origin_id != server:
                continue
            candidates.extend(snapshot.entries.values())
        if uri is not None:
            return [e for e in candidates if e.identity.locator == uri]
        if name is not None:
            return [e for e in candidates if e.name == name]
        return candidates

    def resolve_entry(self, server_name: str, scope_id: str, uri: str, *, refresh: bool = False) -> SkillEntry:
        """Held entry for *uri*, or a fresh single-entry ``skills/get``.

        ``refresh=False`` prefers the published snapshot — re-fetching an entry we
        already hold would add a round trip to every load without changing the
        answer. An explicit URI is honoured even when the catalog never listed it:
        a listing is not a claim of completeness, so absence from it is not proof
        that the skill does not exist.
        """
        if not refresh:
            held = self._held_entry(server_name, scope_id, uri)
            if held is not None:
                return held
        inst = self._instance(server_name, scope_id)
        result = self._rpc_sync(inst, proto.METHOD_GET,
                                proto.SkillsGetRequest(params=proto.SkillsGetParams(uri=uri)),
                                proto.SkillsGetResult)
        raw = result.skill
        if not isinstance(raw, Mapping):
            raise SkillError(SkillErrorCode.NOT_PROVIDED,
                             f"MCP server '{server_name}' returned no skill for {uri!r}")
        return proto.parse_skill_entry(raw, origin_id=server_name, label=server_name)

    def _held_entry(self, server_name: str, scope_id: str, uri: str) -> Optional[SkillEntry]:
        """The published entry for *uri*, or None when it is absent **or no longer current**.

        A stale snapshot still routes — the prompt keeps showing those skills — but it
        cannot authorize a load, so a miss here sends the caller to a single-entry
        ``skills/get``. That is deliberately not a full catalog refresh: one load should
        cost one round trip, and the client must not turn a zero TTL into a list storm.
        """
        snapshot = self.snapshot(server_name, scope_id)
        if snapshot is None or not snapshot.is_fresh():
            return None
        return snapshot.entries.get(SkillIdentity(BACKEND_MCP, server_name, uri).as_key())

    def read_resource(self, server_name: str, scope_id: str, entry: SkillEntry, resource_uri: str) -> VerifiedRead:
        """Read one resource and verify it against the held manifest before returning it.

        Verification runs on the raw bytes exactly as the server sent them; the
        text handed to the model is derived after the check, never before.
        """
        manifest = entry.manifest
        if manifest is None:
            raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill {entry.identity.locator!r} has no manifest")
        if manifest.dynamic:
            raise SkillError(SkillErrorCode.DYNAMIC_DISABLED,
                             f"skill {entry.identity.locator!r} publishes a dynamic manifest, which carries no "
                             "content binding; loading it is disabled")
        expected = manifest.find(resource_uri)
        if expected is None:
            raise SkillError(SkillErrorCode.NOT_PROVIDED,
                             f"{resource_uri!r} is not part of skill {entry.identity.locator!r}",
                             available_files=sorted(f.relative_path for f in manifest.files))
        inst = self._instance(server_name, scope_id)
        raw, mime = self._read_raw(inst, resource_uri)
        is_main = expected.relative_path == proto.MAIN_FILE_NAME
        verification = proto.verify_bytes(raw, expected, expect_text=is_main)
        if not verification.ok:
            self._invalidate(server_name, scope_id, entry, verification)
            raise SkillError(SkillErrorCode.VERIFICATION_FAILED,
                             f"skill {entry.identity.locator!r} failed content verification: "
                             + "; ".join(verification.errors),
                             resource=resource_uri)
        return VerifiedRead(
            identity=entry.identity, resource_locator=resource_uri, raw_bytes=raw,
            text=proto.decode_text(raw), mime_type=mime, content_revision=entry.content_revision,
            provenance=entry.provenance, verification=verification,
            skill_root_uri=proto.skill_root_uri(entry.identity.locator))

    def _read_raw(self, inst: _Instance, resource_uri: str) -> tuple[bytes, Optional[str]]:
        result = self._rpc_sync(inst, "resources/read",
                                proto.ReadResourceRequest(params=proto.ReadResourceParams(uri=resource_uri)),
                                proto.ReadResourceResult, timeout=READ_TIMEOUT_SECONDS)
        block = next((b for b in result.contents if b.uri == resource_uri), None)
        if block is None:
            raise SkillError(SkillErrorCode.PROTOCOL_INVALID,
                             f"MCP server '{inst.server_name}' did not return the requested resource {resource_uri!r}")
        if block.text is not None:
            return block.text.encode("utf-8"), block.mimeType
        if block.blob is None:
            raise SkillError(SkillErrorCode.PROTOCOL_INVALID,
                             f"resource {resource_uri!r} carried neither text nor blob")
        return base64.b64decode(block.blob), block.mimeType

    def _invalidate(self, server_name: str, scope_id: str, entry: SkillEntry,
                    verification: Any) -> None:
        """Drop a snapshot whose bytes no longer match the manifest it published.

        The bytes are refused (the caller raises) and the catalog is retired, so no
        later load can reuse the entry whose binding just failed.
        """
        with self._lock:
            inst = self._instances.get((scope_id, server_name))
            if inst is None:
                return
            inst.snapshot = None
            inst.catalog_state = CatalogState.REFRESH_FAILED
            inst.last_error = f"verification failed for {entry.identity.locator}: {'; '.join(verification.errors)}"
        logger.warning("MCP server '%s': retired skills catalog after a verification failure on %s",
                       server_name, entry.identity.locator)

    # ── RPC plumbing ────────────────────────────────────────────────────────

    def _sender_for(self, inst: _Instance) -> Any:
        if self._sender is None:
            self._sender = _default_sender()
        return self._sender

    async def _rpc_async(self, inst: _Instance, op: str, request: Any, result_type: Any,
                         timeout: float = METADATA_TIMEOUT_SECONDS) -> Any:
        """One request awaited on the MCP loop. Read-only ops only — never records a breaker outcome."""
        try:
            return await self._sender_for(inst).send(inst.server_name, op, request, result_type, timeout)
        except SkillError:
            raise
        except Exception as exc:
            raise self._translate(inst, op, exc) from exc

    def _rpc_sync(self, inst: _Instance, op: str, request: Any, result_type: Any,
                  timeout: float = METADATA_TIMEOUT_SECONDS) -> Any:
        try:
            return self._sender_for(inst).send_sync(inst.server_name, op, request, result_type, timeout)
        except SkillError:
            raise
        except Exception as exc:
            raise self._translate(inst, op, exc) from exc

    def _translate(self, inst: _Instance, op: str, exc: BaseException) -> SkillError:
        """Map a transport/SDK failure onto the typed contract.

        A timeout is not a disconnect, and a JSON-RPC error is the server's own
        vocabulary — neither may be flattened into "the skill does not exist".
        """
        if isinstance(exc, (TimeoutError, InterruptedError)):
            return SkillError(SkillErrorCode.TIMEOUT,
                              f"MCP server '{inst.server_name}' timed out on {op}")
        # Duck-typed on the JSON-RPC error shape rather than the SDK's class name: the
        # SDK is an optional import here and its exception classes move between majors.
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            message = getattr(exc, "message", None) or str(exc)
            if code == -32601:
                return self._mismatch(inst, op, message)
            if code == -32602 and op in (proto.METHOD_LIST, proto.METHOD_GET):
                return SkillError(SkillErrorCode.NOT_PROVIDED,
                                  f"MCP server '{inst.server_name}' does not provide that skill: {message}")
            return SkillError(SkillErrorCode.PROTOCOL_INVALID,
                              f"MCP server '{inst.server_name}' answered {op} with error {code}: {message}")
        if isinstance(exc, RuntimeError) and "event loop" in str(exc).lower():
            return SkillError(SkillErrorCode.UNAVAILABLE,
                              f"MCP server '{inst.server_name}' is not connected")
        return SkillError(SkillErrorCode.UNAVAILABLE,
                          f"MCP server '{inst.server_name}' failed {op}: {type(exc).__name__}: {exc}")

    def _mismatch(self, inst: _Instance, op: str, message: str) -> SkillError:
        """Method-not-found on an advertised channel disables skills for this instance only."""
        with self._lock:
            inst.support = SkillsSupport.PROTOCOL_MISMATCH
            inst.support_reason = f"{op} not implemented: {message}"
            inst.snapshot = None
            inst.catalog_state = CatalogState.ABSENT
        logger.warning("MCP server '%s' advertises the skills extension but does not implement %s; "
                       "disabling its skills channel (its other MCP capabilities are unaffected)",
                       inst.server_name, op)
        return SkillError(SkillErrorCode.PROTOCOL_MISMATCH,
                          f"MCP server '{inst.server_name}' advertises the skills extension but {op} is not implemented")


def _negotiated_version(session: Any) -> str:
    """The version the session actually settled on.

    Read from the session, never from the version we offered: ``LATEST_HANDSHAKE_VERSION``
    is this client's wish, and a server may answer with anything it supports.
    """
    return str(getattr(session, "_negotiated_version", "") or "")


def _read_ttl_ms(result: Any) -> Optional[float]:
    """The server's freshness hint, or ``None`` when it gave a usable one.

    ``ttlMs == 0`` is a real answer, not a missing one: the server is declining to
    promise that the listing is still current, so entries expire on arrival and a
    load re-confirms its own entry. Treating that as "never expires" would be the
    host overruling the server's own statement about its data.
    """
    ttl = getattr(result, "ttlMs", None)
    if isinstance(ttl, (int, float)) and not isinstance(ttl, bool) and ttl >= 0:
        return float(ttl)
    return None


def _fresh_until(ttl_ms: Optional[float]) -> float:
    """Wall-clock deadline for this snapshot's metadata.

    An absent hint falls back to the host default rather than to infinity: a server
    that says nothing has not promised anything, and an unbounded cache would make
    the host's routing depend on a listing nothing can invalidate.
    """
    return time.monotonic() + (DEFAULT_TTL_MS if ttl_ms is None else ttl_ms) / 1000.0


def _diagnostics(entries: Iterable[SkillEntry], quarantined: Iterable[str]) -> str:
    quarantined = list(quarantined)
    if not quarantined:
        return ""
    return f"{len(quarantined)} entr(ies) quarantined by validation: " + "; ".join(quarantined[:5])


_MANAGER: Optional[SkillsOverMcpManager] = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> SkillsOverMcpManager:
    """Process-wide manager (one per process: one MCP connection ledger, one catalog ledger)."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = SkillsOverMcpManager()
        return _MANAGER


def set_manager(manager: Optional[SkillsOverMcpManager]) -> None:
    """Install a manager (tests); ``None`` restores the lazy default."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = manager


def _mcp_loop() -> Any:
    """The MCP loop when one is running, else None (read under the connection ledger's lock)."""
    try:
        from tools import mcp_tool as _mcp
        with _mcp._core._lock:
            loop = _mcp._core._mcp_loop
        return loop if loop is not None and loop.is_running() else None
    except (ImportError, AttributeError):
        logger.debug("MCP loop unavailable for a background skills refresh", exc_info=True)
        return None


def current_scope_id() -> str:
    """The scope the skills ledger keys on, matching the MCP connection ledger's.

    Empty string (not None) for single-profile processes, so every caller can use
    it as a dict key without a None special case. Read through the connection
    layer's resolver rather than re-deriving it, or a multiplexed process would
    file its catalogs under a scope its connections do not belong to.
    """
    try:
        from tools.mcp_tool import _mcp_registry_scope
        return _mcp_registry_scope() or ""
    except Exception:
        logger.debug("MCP registry scope unavailable; defaulting to the single-profile scope", exc_info=True)
        return ""


async def publish_catalog_after_connect(server_name: str, config: Any, server: Any) -> None:
    """Build the skills catalog for one opted-in server as part of its connect pass.

    Awaited for at most ``INITIAL_PRELOAD_BUDGET_SECONDS`` so the first session
    prompt usually carries the catalog. Past that the discovery keeps running in
    the background rather than being cancelled: the connection is healthy and the
    listing is merely slow, which is exactly the case the per-turn catalog note
    exists for. Cancelling instead would waste the pages already fetched and
    re-issue them on the next attempt.
    """
    if not mcp_server_skills_enabled(config):
        return
    manager = get_manager()
    scope_id = current_scope_id()
    view = manager.observe_connection(server_name, scope_id,
                                      initialize_result=getattr(server, "initialize_result", None),
                                      session=getattr(server, "session", None))
    if view.support is not SkillsSupport.SUPPORTED:
        logger.info("MCP server '%s': skills channel not enabled (%s)", server_name, view.support_reason)
        return
    task = asyncio.ensure_future(manager.discover(server_name, scope_id, reason="connect"))
    _log_when_done(task, server_name)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=INITIAL_PRELOAD_BUDGET_SECONDS)
    except TimeoutError:
        logger.info("MCP server '%s': skills catalog still loading after %.0fs; it will be announced on a "
                    "later turn instead of holding up this session's prompt",
                    server_name, INITIAL_PRELOAD_BUDGET_SECONDS)
    except SkillError:
        pass  # already logged by _log_when_done


def _log_when_done(task: asyncio.Task[Any], server_name: str) -> None:
    def _done(finished: asyncio.Task[Any]) -> None:
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is not None:
            logger.warning("MCP server '%s': skills discovery failed: %s", server_name, exc)
    task.add_done_callback(_done)
