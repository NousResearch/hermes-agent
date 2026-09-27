# PostgreSQL State Store v16 contextual-search projection audit

## Direct-history reconciliation

- `b68ccbe76f4e85b539a4c05d69ec59ea8e06f636` — `Change-Id: I83b2756b073f43529edf63e0a9edee40`; migration v16 bounded browse projection, 42 tests. This did not route the complete contextual tool or authorize a backend cutover.

## Decision: complete contextual projection is portable, but runtime activation is bounded

The complete backend-neutral contextual `session_search` projection group is exposed
through `StateStore` and admitted for a configured PostgreSQL tenant only after its
generated-search health gate passes. This document records that verified scope; it is
not a general PostgreSQL runtime cutover.

## SQLite contract that a cutover must preserve

`SessionDB.search_messages()` returns canonical candidates in the field order `id`, `session_id`, `role`, `snippet`, `timestamp`, `tool_name`, `source`, `model`, `session_started`, and `context`. Unless `fields` omits `context`, each candidate is enriched in one batched query with its same-session chronological neighbours (anchor plus one physical row on either side, ordered by timestamp/id), decoded and shortened to 200 characters. Context never crosses sessions; tied timestamps retain stable identity order; duplicate candidates retain identical context.

The tool's discovery flow uses those raw candidates only as an input. It defaults roles to `user`/`assistant`, excludes `kanban`, `subagent`, and `tool` sources, applies stable cron demotion and lineage de-duplication, then hydrates a selected anchor with `get_anchored_view()`. The anchored view has a physical ±5 window, retains a tool anchor even when filtering its neighbouring rows to user/assistant, and supplies first/last three non-empty role-filtered bookends strictly outside that window. Scroll uses `get_messages_around()` with a clamped ±1..20 physical window and deliberate boundary-anchor repetition across pages. Browse/read use distinct bounded projections.

Candidate visibility excludes hidden display rows and, unless requested, requires active or compaction-archived messages. SQLite's FTS failures detach derived FTS structures, mark search stale, serve canonical `LIKE` fallback, and during deferred backfill supplement the unindexed high-water gap. Rebuild admission, quarantine, retries, and tool-visible incompleteness are all part of the observable behavior.

## Implemented PostgreSQL contextual health/rebuild capability

PostgreSQL now exposes the complete contextual contract and is admitted to public routing only after a tenant-scoped generated-search health check passes. The status is intentionally **not** SQLite FTS progress: it records the generated `tsvector` validity, GIN catalog validity, query-path availability, whether a per-tenant maintenance operation is active, last successful repair/error, and explicit `false` capabilities for SQLite-only corruption-detach, canonical-LIKE fallback, deferred backfill/high-water, retry, and quarantine semantics. A missing or malformed GIN entry causes new contextual admission and live discovery to fail closed even though PostgreSQL could otherwise choose a sequential scan.

`rebuild_search_index()` is a trusted-schema-only operational repair, not an automatic side effect of status. It takes a cross-process advisory lock on a dedicated autocommit connection and uses `REINDEX INDEX CONCURRENTLY`, creates a missing GIN index concurrently, or replaces an invalid same-named catalog object. It cannot select a tenant or interpolate caller input. PG18 live coverage exercises healthy status, missing-index drift, invalid-index drift, concurrent repair, read/search behavior across the missing/repaired transition, pool `search_path` reset, and tenant isolation. SQLite `fts_rebuild_status()` remains unchanged.

PostgreSQL direct `search_messages()` returns the canonical ordered candidate fields `id`, `session_id`, `role`, `snippet`, `timestamp`, `tool_name`, `source`, `model`, `session_started`, and conditional `context`. It batches up to the returned candidate set into tenant-schema-qualified, parameterized same-session chronological neighbour seeks using `(created_at, id)` ties, decodes JSON-encoded multimodal content, flattens it exactly as SQLite does, and caps each context item at 200 characters. Projections that omit `context` do not issue the enrichment query; repeated input identities share the same enriched context. PG18/SQLite coverage covers ties, session boundaries, structured context, hidden candidates, source/role filters, tool-body queries, model/session metadata, deterministic oldest pagination, and conditional fields.

## Implemented intermediate contract

`state_store.ContextualSessionSearchStore` now publishes the atomic contextual-read boundary independently from the incremental write-oriented `StateStore`. `SqliteContextualSessionSearchStore` delegates every required operation to the existing `SessionDB` implementation: canonical candidate search, title/session lookup, anchored detail and scroll views, message visibility state, bounded browse, and public search-index rebuild status.

`tools.session_search_tool` coerces both lazily acquired and injected contextual stores through this capability before dispatching every public shape. The former tool-private `SessionDB._lock`/`_conn` visibility read now lives solely inside the SQLite adapter. The selected PostgreSQL CLI facade forwards the already-complete contextual read primitives to the public inline consumer; it is not SQLite emulation. Named-profile resolution likewise opens a read-only contextual store, preserving profile isolation and closing ownership.

## Profile-aware routing prerequisite

Contextual resolution now derives every root/default/named profile's backend from that profile's own `config.yaml`, not from the caller's profile metadata or a hard-coded `<profile>/state.db` path. `root`, `global`, and `default` explicitly canonicalize to the installation-root profile; a named target is registry-validated. An explicit target rejects injected database/factory seams, so an application cross-profile read cannot accidentally reuse the caller's SQLite store. The resolver reads the canonical profile directory directly, uses a context-local `HERMES_HOME` override only while acquiring PostgreSQL, and builds secrets from the same profile's `.env`; it never mutates process-global `HERMES_HOME` or borrows ambient secrets. SQLite retains its default and registry-backed behavior, with named-profile opens read-only. Cross-profile reads are intentional application routing, not a PostgreSQL permission grant or RLS boundary.

For a configured PostgreSQL profile, the resolver enters the existing canonical-home tenant acquisition path and admits the store only after the full contextual-method and generated-search-health gate passes. It does not open that profile's SQLite database, does not select a schema from user input, and does not fall back across backend boundaries. Invalid backend configuration and injection-shaped profile names also fail before any SQLite fallback.

PostgreSQL is admitted by `contextual_session_search_store(backend="postgresql")` only when every contextual method is present and its generated-search status is valid. No lexical PostgreSQL rows are routed if catalog drift makes the contextual search path unavailable. No runtime backend cutover is implied.

## Public-consumer evidence and bounded route

The real `AIAgent` inline consumer injects its selected `PostgreSQLCLISessionStore` into `session_search`; the resolver recognizes its backend-owned health status and admits only the full contextual method set. It reuses that already tenant-bound handle rather than opening `state.db`, resolving a second DSN, or silently wrapping it as SQLite. PG18 coverage exercises serialized CJK discovery, grammar failure, anchored/scroll windows, read, browse, health reporting, and explicit rebuild. The public direct tool path additionally resolves explicit root/default or named profiles read-only from their canonical configuration and secret scope; it never accepts an injected cross-profile store.

No gateway, cron, TUI, ACP, hosted-room, async-delegation, or generic legacy `SessionDB` consumer is enabled by this route.

## Exact remaining gaps / cutover boundary

Not implemented or claimed: FTS5 BM25/tokenizer/CJK/trigram equivalence; SQLite corruption-detach/canonical-LIKE fallback/deferred-rebuild parity; migration/import/export; RLS authorization; production deployment; runtime/config cutover; or production readiness. The public PostgreSQL contextual route is limited to a configured tenant with a valid generated-document/GIN health gate; repair is local explicit maintenance, not a deployment recovery plan.

## Shared-room root/global routing foundation

Hosted-room coordination resolves only through the canonical installation root/global namespace (`shared-state.db`), independent of any selected execution profile. Profile identifiers remain application-level room-member routing data and never become SQL selectors. The SQLite coordination protocol remains the only available adapter; a PostgreSQL hosted-room adapter, backend selection, and runtime cutover are still pending a complete protocol implementation and verification.

## Delivery-obligation ledger cutover prerequisite

`gateway.delivery_ledger` remains a separately durable SQLite recovery store; PostgreSQL ledger storage is not implemented or claimed. This refactor extracts its backend-neutral `DeliveryReceipt` fence contract and routes gateway production and recovery transitions through receipt-guarded claim, success, failure, and release calls. SQLite migration adds a zero-default `delivery_fence`, durable installation/host/process-generation ownership fields, expiry metadata, and a claim index; existing rows remain recoverable.

The next PostgreSQL StateStore cutover must still make an explicit ledger decision: retain this SQLite ledger as a separately durable co-resident store, or port it atomically with schema migration, recovery/readback contract, rollback, and live PG18 parity. A table-only port is expressly insufficient because stale owners must not mutate a newer receipt fence.
