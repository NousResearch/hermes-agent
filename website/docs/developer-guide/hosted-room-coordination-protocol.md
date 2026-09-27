# Hosted Room Coordination Protocol (extracted SQLite contract)

**Status:** protocol extracted and wired through the production SQLite hosted-room core; not a PostgreSQL implementation, migration plan, runtime backend selection, or cutover authorization.

This protocol names the durable coordination contract currently implemented by the hosted-room SQLite graph. A future backend may conform only by preserving the transactions and failure semantics below as one protocol. The objects are interdependent: no subset of tables is safe to route independently to PostgreSQL.

## Scope and non-goals

The target uses one PostgreSQL instance. Hosted-room coordination is root/global shared state and may intentionally span profile namespaces; this is not a mandatory profile ACL model. It does not share browser/YOLO/CUA approval authority: one-time approvals remain process-local and fail closed.

The protocol owns durable coordination state for a hosted room and its peer, driver, replica, and policy-projection workflows. It does **not** own:

- process-local work such as asyncio locks, active model calls, in-memory retry timers, or an adapter connection;
- a claim that an external prompt, notification, or message was delivered; and
- a replacement for the append-only room log with a mutable policy projection.

A committed database update proves only its own durable fact. In particular, a queued task, a remote-run receipt, or a policy publication record must never be interpreted as proof that outgoing delivery occurred. Irreversible delivery requires a separately designed provider acknowledgement/receipt protocol and reconciliation after crash.

## Stable identities and fence values

Every future backend operation MUST receive a trusted namespace-routing context and stable identity rather than derive ownership from a PID or client-provided profile string. This distinguishes trusted selector construction from product authorization: Hermes profiles are logical namespaces, and an explicit user-directed cross-profile read is allowed.

| Type | Required fields | Purpose |
|---|---|---|
| `NamespaceId` | root/profile namespace resolved by trusted runtime context, or root/global shared-coordination namespace | Routing and accidental-mixing boundary; not a profile ACL or entitlement. |
| `RoomId` | root/global shared-coordination opaque ID | Durable room identity, permanently reserved after retention expiry. |
| `AuthorityFence` | `room_id`, `gateway_id`, monotonic `authority_epoch` | Fences event admission, driver actions, replica promotion, and peer reservations. |
| `OwnerId` | installation ID, host identity, process-generation UUID | Identifies a live worker incarnation; PID alone is invalid across hosts/restarts. |
| `LeaseFence` | `AuthorityFence`, `OwnerId`, monotonic `lease_generation`, expiry | Authorizes driver transitions only while current and unexpired. |
| `TaskIdentity` | `room_id`, `task_id`, `thread_id`, `turn_id` | Immutable admitted turn; `(room, thread, turn)` is unique. |
| `AttemptFence` | `TaskIdentity`, execution generation, cancellation generation, `LeaseFence` | Authorizes exactly one running/settlement transition. |
| `RemoteRunIdentity` | room, home installation, authority gateway/epoch, member, target installation/profile, task, execution generation | Idempotency key for a remote run handle. |
| `GrantScope` | room, member, target profile, authority gateway/epoch, issued-at | Exact scope revoked by a monotonic revocation fence. |

The backend obtains expiry from its transaction-time clock or a documented clock abstraction. It must not accept an untrusted caller wall clock as the authority for lease validity.

## Durable state objects

| Object group | Current SQLite tables | Contract |
|---|---|---|
| Room source of truth | `hosted_rooms`, `hosted_room_events`, `hosted_room_retired_ids` | Room membership and authority plus immutable, ordered event log, tombstone and permanent ID reservation. |
| Private peer routing | `hosted_room_links`, `hosted_room_remote_runs` | Link/grant route state and idempotent remote-run handles; neither proves external execution/delivery. |
| Revocation/reservation | `hosted_room_revoked_grants`, `hosted_room_peer_reservations` | Target-side admission fence and exact-scope revocation watermark. |
| Driver | `hosted_room_driver_leases`, `hosted_room_driver_tasks` | Fenced single-driver lease and durable task/attempt/cancellation/recovery state machine. |
| Replica | `hosted_room_replicas`, `hosted_room_replica_events` | Contiguous replica history and authority lineage for promotion/demotion. |
| Policy projection | `hosted_room_policy_cursors`, `hosted_room_policy_threads`, `hosted_room_policy_events`, `hosted_room_policy_watermarks`, `hosted_room_policy_publications`, `hosted_room_policy_transcript`, `hosted_room_policy_transcript_state` | Bounded, rebuildable projection of the source log; it never replaces the source log. |

Session/profile content remains in its resolved profile namespace by default; resolvers may intentionally open a target profile namespace for an explicit user-directed cross-profile read. Hosted-room coordination is root/global shared state because a room may intentionally span profiles. A PostgreSQL implementation must carry its derived namespace key through every query, lock key, migration record, and administrative path to prevent accidental mixing or SQL-selector injection; that routing discipline is not mandatory RLS, database-role, or per-schema ACL enforcement.

## Required operations and atomicity

Each numbered operation below is one serializable backend transaction (or a proved-equivalent atomic compare-and-set) with the listed read/write set. Retrying an accepted request returns the original durable outcome only when every immutable request field matches; otherwise it fails closed.

1. **Create room** — reserve a new `RoomId`, store canonical membership and authority epoch 1. Exact retry is idempotent; changed identity/membership/authority conflicts. A retired ID never becomes reusable.
2. **Append event** — validate current `AuthorityFence`, insert exactly one immutable event at `next_seq`, and compare-and-set `next_seq`, revision, and byte budget together. Exact `event_id` replay returns the committed event only if kind, actor, payload, and epoch match. A stale fence or changed payload fails.
3. **Claim/disband authority** — atomically append the control event and update authority/tombstone state under the expected prior authority fence. A disband must fence future mutations and retain/reserve the room ID even after payload pruning.
4. **Store link / reserve peer / revoke grant** — atomically enforce capacity and scope identity; reservation replacement may only advance authority lineage. Revocation monotonically raises its `revoked_before` fence and invalidates matching live reservations in the same transaction. Expired reservations/revocations are not current.
5. **Record remote run receipt** — bind exactly one `RemoteRunIdentity` to immutable `run_id` and `session_id`; exact retry is idempotent, changed handles conflict. This is an intent/reconciliation handle, not an external-delivery acknowledgement.
6. **Acquire, renew, release driver lease** — atomically read current room authority and lease row; claim only absent/released/expired ownership and monotonically increase `lease_generation` on reclaim. Renew/release requires the exact active `LeaseFence`; release is forbidden while a task is running.
7. **Admit/start/settle/cancel/recover task** — task admission stores immutable payload and identity. Start requires the active lease and the earliest queued task ordered by `(source_event_seq, created_at, task_id)`. Settlement and cancellation compare status plus attempt/lease fences. A crash/takeover turns foreign running work into `indeterminate`, never silently requeues it. Only explicit operator-approved retry creates a new execution generation.
8. **Ingest/promote/demote replica** — accept only contiguous, authority-stamped, non-regressing replica pages. Promotion atomically copies eligible replica history, advances authority epoch, and appends its claim event. Demotion records loss only for a strictly newer observed epoch.
9. **Apply policy projection page** — read a bounded, contiguous source-log page, apply every projection mutation, and conditionally advance the cursor in one transaction. The cursor update must compare the previously observed cursor (or otherwise serialize per-room projection ownership); a bare update by `room_id` is insufficient for a multi-process backend. Watermarks and stopped cursors only advance monotonically. Publication receipts are exact durable log/publication facts, not outbound delivery proof.

## State-machine invariants

- **Ordering:** events are strictly sequenced per room; driver starts only the queue head; replicas are contiguous; projection is derived in source-log order.
- **Idempotency:** an ID is a replay key for one immutable request, not permission to overwrite a durable fact.
- **Fencing:** every mutating action that can race with ownership changes compares the full current authority, lease, and/or attempt fence at write time.
- **Expiry:** expired lease, grant, and reservation state has no authority. Reclamation makes older tokens unusable.
- **Cancellation:** cancellation increments/carries its generation and fences late success. A cancellation receipt may replay only when its cancellation ID and terminal state match.
- **Crash recovery:** no inferred delivery; uncertain running work is `indeterminate` until verified receipt, cancellation, deferment, or explicit retry resolves it.
- **Retention:** pruning source data must preserve required tombstone/identity fences and cannot permit projection re-creation for a retired room.

## Current SQLite conformance evidence

`tests/gateway/test_hosted_room_coordination_protocol.py` drives `SqliteHostedRoomCoordination`, the production adapter surface used by the hosted-room service/runtime, peer receipt path, and target grant reservation/revocation handlers. It proves: room creation and ordered event replay; lease expiry/reclaim with stale-fence rejection; remote receipt exact replay/conflict; peer reservation/revocation/expiry; and policy cursor plus watermark monotonicity. The adapter delegates each operation to its existing whole SQLite transaction; it does not split a transaction or soften a fence/replay rule. This is executable SQLite conformance evidence, not PostgreSQL differential proof.

Existing source-focused suites additionally cover concurrent append/claim, rollback sequence preservation, driver recovery/cancellation, replica continuity/promotion, and gateway/TUI lifecycle. Run all relevant suites before relying on this protocol for a port.

## PostgreSQL implementation preconditions (all unresolved)

A future implementation must not begin a partial table migration or runtime route until all of the following have a concrete design and executable evidence:

1. Trusted root/profile namespace resolver, propagated through every consumer; tests for default local routing, explicit cross-profile target reads, root/global shared-room routing, selector-injection rejection, and accidental-mixing prevention. RLS/database-role ACL is optional operator hardening, not a prerequisite.
2. Dedicated per-namespace migration ledger distinct from business tables, with resumable object-level verification and immutable source snapshot identity.
3. Declared transaction isolation and retry policy, including serialization/deadlock classification and no retry of irreversible side effects.
4. Row/advisory-lock design keyed by namespace + room, plus transactional fence-token guards on every authority/lease/attempt transition.
5. PostgreSQL server-time expiry semantics, stable installation+host+process-generation owner identity, and skew/crash takeover tests.
6. Complete SQLite import, single controlled cutover, and reverse rollback design. No dual-write, partial table cutover, or automatic runtime fallback is assumed safe.
7. PG18/SQLite differential contract suite covering all protocol operations, contention, crash/failure injection, ordering, retention, revocation, namespace-routing/mix-up protection, and externally observed reconciliation boundaries.
8. A verified consumer migration proving no hosted-room/driver/policy runtime path opens SQLite before any claim of PostgreSQL readiness.

Until then, PostgreSQL readiness is explicitly **not established**. Selected
PostgreSQL therefore fails closed at the hosted coordination factory and public
service/driver/peer/RPC boundaries with
`HostedRoomCoordinationUnavailableError`, before a separate SQLite coordinator,
room task, peer operation, replica write, or transport side effect can occur.

## Evidence reconciliation

The protocol-extraction evidence previously cited malformed commit `0719773...`. Its identical tree was safely amended and the verified source is `6bc2e58898ecb5e17d2c83c8a7a57a7f33847950` (`Change-Id: I1202528741df518d25e7fb082c7f7a2e`). This correction is evidence-only; it does not assert PostgreSQL readiness.
