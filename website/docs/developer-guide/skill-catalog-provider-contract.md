---
sidebar_label: "Skill catalog contract"
title: "Skill catalog provider contract"
description: "Experimental wire, HTTP, ownership and recovery contract for plugin-served skill catalogs"
---

# Skill catalog provider contract

:::caution Experimental
Companion to [Plugin-served skill catalogs](./skill-catalog-provider.md). Wire fields, signatures and handle shapes may change. This reference describes the core API, not evidence of a deployed consumer or complete product acceptance.
:::

## A. Provider wire contract

### A.1 Identity, validation and common envelope

Provider-aware routing applies to request bodies up to 256 KiB. Larger bodies are handled exactly as a host without providers handles them.

All methods accept one request object containing `context` and return `Result`. Required consumed fields are listed below. Unknown request fields refuse. Unknown response fields are ignored only after every consumed field validates.

```text
Result = {ok: true, data: <payload>}
       | {ok: false, error: {code: <named code>, message: <safe text>}}
Context = {profile_key, plugin_key, generation, request_id, deadline_ms}
```

Core stamps the whole Context. The selected connection/profile resolves and authorizes a home before any provider lookup or secret hydration. Requests cannot submit filesystem roots, credentials, plugin keys or generations. `profile_key` is an opaque core-resolved profile identity, not a user-selected path. Provider service credentials come from its scoped context.

| Field | Accepted shape and bounds |
|---|---|
| Catalog name | ASCII `[a-z][a-z0-9-]{0,47}`, exact case, no trimming or normalization |
| Owning plugin key | Exact canonical manifest registry key. Opaque UTF-8, 1..1024 bytes, no NUL/control characters. Core binds it from `manifest_key`, not provider data. Keys outside this bound refuse this registrar only. |
| Owner digest | Full lowercase 64-hex SHA-256 of the exact plugin-key UTF-8 bytes, without normalization. Core computes it. |
| Source/reference | `plugin.<owner_digest>.<catalog>` and `<source>:<item_id>`. Source is at most 120 ASCII bytes and reference at most 313. No aliases on effect dispatch. |
| Item/finding/setup id | ASCII `[A-Za-z0-9][A-Za-z0-9._~-]{0,191}`, exact case. Slash, colon, percent and `@` refuse. Native richer ids require a stable provider mapping. |
| Version | ASCII `[A-Za-z0-9][A-Za-z0-9._+-]{0,63}`, opaque. No lexical ordering. |
| Token/operation/reservation/cursor | ASCII `[A-Za-z0-9_-]{16,256}`. Tokens have at least 128 bits of provider randomness and contain no credentials. Ids are opaque, cursors nullable. Core reservation ids are nonsecret. |
| Core opaque keys | Profile key, request id, generation and registration id use the same 16..256 ASCII grammar. Process-machine and process-birth identities are core-generated opaque UTF-8 1..256 bytes. |
| Integers | Strict JSON integers, never bool, float, string or coercion. Revisions 0..9007199254740991. Unix millisecond timestamps 1..9007199254740991. Limit 1..100, default 20. File bytes 0..9007199254740991. File line 1..2147483647 or null. |
| Text | Valid UTF-8. Name 1..256 characters, label 1..80, description 0..4096, author 1..256 or null, message 0..2048, query 0..256. Policy revision, implementation revision, installation revision and content digest are opaque strings 1..256 characters unless explicitly null. No control characters except newline/tab in descriptive text. HTML renders as text. |
| Paths | Skills-root-relative POSIX path, 1..1024 UTF-8 bytes, each segment 1..255 bytes. No absolute prefix, drive, backslash, empty segment, `.` or `..`, NUL/control characters or trailing slash. Mutation targets also reject redirects, symlinks/junction traversal, reserved core roots and ancestor/descendant conflicts. Display file paths have no mutation authority. |
| Lists | Tags 0..32 strings of 1..64 characters. Check id batch 0..100. File inventory 0..2000. Findings 0..1000. Setup guidance 0..100. Action lists unique. Acknowledgements unique finding ids, at most 1000. |
| Payload | Request at most 256 KiB UTF-8 JSON, response at most 8 MiB. SKILL.md preview at most 256 KiB UTF-8. JSON values only with finite numbers. Strict booleans. Oversized result refuses or declares an incomplete preview, never silent truncation of a confirmable plan. |
| Omission/null | Omitted query means empty string, omitted limit means 20, omitted version/cursor means null. Explicit null is accepted only for fields marked nullable. Other required fields cannot be omitted. Empty id/version/token never means latest. |

Source/reference maxima are derived from the grammar, not an extra truncation rule. A source digest is an identity partition, not a signature or hostile-plugin boundary. Core checks its exact plugin-key match at every dispatch. Changing a plugin registry key changes source identity. No migration command or retained namespace ownership is introduced.

### A.2 Registration descriptor and method table

`RegistrationDescriptor = {schema_version: 1, name, label, implementation_revision}`. All fields are required. `schema_version` is strict integer 1. The provider supplies literal string class/instance data, inspected without property evaluation. A custom metaclass or overridden `__getattribute__`, `__getattr__`, `__class__` or descriptor-backed identity refuses before value access. Parent code constructs its stored owner fields from the manifest. It never accepts a descriptor-supplied owner.

The host child validates the original provider before describing its callable reference. Registration transports `{descriptor, object_ref}` as a catalog-specific tagged argument. The parent validates both shapes, attaches a frozen internal descriptor to its registry entry, and calls the shared registrar with an explicit key override. It does not inspect identity on the generated proxy. The row uses no name normalizer. Other registrar rows keep their current behavior.

| Method | Request beyond context | Successful payload |
|---|---|---|
| `availability` | none | `{state, message}`. State `ready\|hidden\|sign_in_required\|unavailable`. Default ready. |
| `search` | `{query, limit, cursor}` | `{items: [Entry], next_cursor}`. Empty query browses. |
| `inspect` | `{id, version}` | `{entry, skill_md, files, preview_complete}`. Metadata/text only, no install bundle. |
| `scan` | `{id, version}` | `{entry, scan}`. Optional, default unsupported. |
| `installed` | `{limit, cursor}` | `{items: [Installation], next_cursor}`. Local provider records, no remote library listing. |
| `check` | `{ids, cursor}` | `{items: [UpdateStatus], next_cursor}`. Empty ids means all. Optional, default unsupported. |
| `review` | `{action, id, version}` | `Review`. No install registration, file mutation or setup execution. |
| `reviewed` | `{review_token}` | `{kind, review, operation}`. Kind `unstarted\|pending\|completed`. Exact stored review and null operation for unstarted, saved operation otherwise. Read-only, including expired started tokens. |
| `confirm` | `{action, id, review_token, acknowledgements, admission}` | `Operation`. Admission nullable only on completed replay. Pending/fresh carry the scoped nonsecret binding in C. |
| `operation` | `{operation_id}` | `Operation`, read-only. No implicit retry or mutation. |

Every method except scan/check is required, with the availability default above. Future methods add defaults. Inspect/scan/review version null resolves current provider choice. Review resolves an exact version before it can be confirmable. Uninstall accepts no explicit requested version and snapshots its current installed version. Update/uninstall installation prerequisites apply only to unstarted authority, never completed replay.

### A.3 Consumed payload schemas

Provider registry keys and references are exact identifiers, not display names. Mutation helpers below are cooperative API boundaries, not same-UID security isolation.

All fields shown here are required except where omission is explicitly allowed. `ManagedState` is `not_installed|installed|removed|unmanaged_local|pending|unavailable`. `Action` is `install|update|uninstall`. `actions` is a unique subset of Action.

| Payload | Fields and rules |
|---|---|
| `Entry` | `{id, name, description, source, identifier, trust_level, repo, path, tags, provider}`. Repo/path are null for a provider entry. Trust is core-stamped `community`. Provider is `{id, plugin_id, label, version, author, managed_state, scan_verdict, actions}`. Here provider.id is catalog name, plugin_id is the exact bound registry key. Version/author nullable. Scan verdict `safe\|caution\|dangerous\|unknown`. Core overwrites source/identifier/owner fields from dispatch. |
| `Source` | `{id, label, searchable, provider, state, message}`. id is full source_id, searchable strict bool, provider `{id, plugin_id}`, state uses availability. Core stamps identity. Hidden sources are omitted. |
| `FileInfo` | `{path, size, change}`, change `add\|replace\|remove\|unchanged` |
| `Finding` | `{id, severity, category, file, line, description, acknowledgement_required}`. Severity `info\|low\|medium\|high\|critical`, category 1..64 characters, file nullable display path, line nullable. Finding ids unique in one review. |
| `Scan` | `{verdict, policy, summary, findings, complete}`. Policy `allow\|acknowledge\|block\|unavailable`, summary 0..4096 characters, complete strict bool. No unconditional dangerous-verdict block is invented. |
| `SetupGuidance` | `{id, summary, detail}`. Summary 1..256 characters, detail 0..4096. Display only, no executable/env fields or implicit setup consent. |
| `Installation` | `{identifier, name, target_relative_path, installed_version, latest_version, managed_state, installation_revision, actions}`. Installed/latest version nullable when unknown/absent. Installation revision is an opaque provider string or null for no prior record. Any mutation requires a concrete current revision for an existing installation. Core validates target before joining or granting protection. |
| `UpdateStatus` | `{identifier, installed_version, latest_version, state, installation_revision}`. Versions/revision nullable if unknown. State `current\|update_available\|removed\|pending\|unavailable\|blocked`. |
| `Review` | `{review_id, action, identifier, version, installation_revision, target_relative_path, summary, files, preview_complete, scan, acknowledgements_required, policy_revision, content_digest, setup_guidance, confirmable, review_token, expires_at_ms}`. Summary 0..4096 characters. Version/revision/digest/policy may be null only in a refused nonconfirmable plan, except a new install always has null installation_revision. |
| `Operation` | `{operation_id, identifier, action, state, outcome, managed_state, message, installation_revision}`. State `pending\|completed\|failed\|unknown`. Outcome null for pending/unknown, otherwise `installed\|updated\|uninstalled\|unchanged\|refused`. Completed cannot have refused, failed must have refused. Installation revision is resulting opaque revision or null. No core reservation is needed to read a completed operation. |

Provider wire requests use the token grammar for nonsecret `review_id` as well. Each stored review has one immutable provider-issued review_id, distinct from its secret token. Core uses `(registration_id, review_id)` to reserve once without persisting the token. Refused nonconfirmable plans may use null review_id. `admission` is null for completed replay, otherwise `{registration_id, generation, review_id, reservation_id, expected_path_revision, expected_installation_revision}` with the A.1 bounds.

**Review cross-fields** require all of the following:

- **Exact authority:** profile/enrolled owner, owning plugin key, stable implementation revision, action/id/version, content, file plan, findings, policy and installation revision are bound in the provider's durable token record.
- **Complete consent:** `confirmable` requires `preview_complete`, `scan.complete`, exact nonnull version/digest/policy revision, token and expiry. Every finding remains inspectable. A nonconfirmable review has null token/expiry. Token lifetime is at most 15 minutes before first admission.
- **Acknowledgement set:** required ids equal exactly findings with acknowledgement_required true. Allow has an empty set, acknowledge has a nonempty set, block/unavailable have no token. Confirm requires the exact set and rejects duplicates/unknown ids. No accept-all boolean exists.
- **Freshness:** the provider rechecks actual installed/package facts before effect. Drift returns review_changed/review_expired and clears prior selections. Core does not recompute opaque digests or carry acknowledgement into a new plan.

### A.4 Failure and observability contract

| Code | HTTP status | Required behavior |
|---|---|---|
| `invalid_request`, `invalid_identifier` | 400 | Validation before provider invocation |
| `not_found` | 404 | Exact item/source absent, no hub fallback |
| `provider_unavailable`, `ownership_state_unavailable` | 503 | No fresh mutation, preserve files |
| `consent_required`, `permission_denied` | 403 | No provider admission |
| `sign_in_required` | 401 | Existing plugin setup path |
| `policy_denied` | 403 | No force bypass |
| `unsupported` | 422 | No alternate implicit installer |
| `review_required`, `acknowledgement_required` | 409 | Render review or missing ids |
| `review_expired`, `review_changed`, `ownership_conflict` | 409 | Re-review, never silent replacement |
| `busy`, `operation_in_progress` | 429 / 409 | Bounded retry/status read |
| `timeout` | 504 | Read/review deadline, no inferred success |
| `outcome_unknown` | 202 | May have committed, retain exact token/id and reconcile |
| `provider_error`, `invalid_response` | 502 | Core-authored safe message, no raw exception |

Unexpected exceptions become provider_error. Masked diagnostics record opaque profile, plugin, source, action, generation, latency and outcome code. They omit query text, skill bodies, finding text, credentials and tokens. Startup print bytes go to bounded/redacted diagnostic logs, not HTTP serialization. Read failures must not look like an empty Team library on an active provider branch.

## B. HTTP routes and search scheduling

### B.1 Route table

Existing routes use their untouched stock serialization and legacy body-profile rules when no provider exists. With no provider registered, POST review/confirm preserve stock 405 with Allow: GET and the Method Not Allowed body. GET operation preserves stock 404 with its endpoint-specific body. Extension validation does not replace those responses. That check precedes extension payload validation after ordinary authentication. Provider requests with conflicting body/query profiles refuse with 400. Existing authorization and multi-profile destructive-target rules apply.

| Route | Registered-provider branch |
|---|---|
| `GET /api/skills/hub/official` | Stock skills plus optional `provider_sources: [Source]`. No provider means field absent. |
| `GET /api/skills/hub/sources` | Append Source rows and canonical provider installed references. Existing rows unchanged. |
| `GET /api/skills/hub/search` | Optional source/cursor. Empty query with explicit provider browses. Provider failure appears in optional provider_errors/timed_out, not swallowed as empty. |
| `GET /api/skills/hub/preview` | Explicit provider reference -> inspect, optional separate version. Never fetch bundle. |
| `GET /api/skills/hub/scan` | Explicit provider -> scan, no install authority. |
| `GET /api/skills/hub/updates` | Stock hub rows plus provider check, retaining canonical identity and partial failures. |
| `POST /api/skills/hub/install` | Provider reference without review token returns 409 review_required. With token uses confirm. Never spawns a CLI provider operation. |
| `POST /api/skills/hub/uninstall` | Identifier is additive alternative to legacy name. Bare managed name returns review_required with reference, never generic uninstall. |
| `POST /api/skills/hub/update` | Explicit provider reference follows confirm. No-argument Update all retains hub action and returns provider review candidates, never applies those candidates. |
| `POST /api/skills/hub/review` | `{identifier, action, version?}` -> Review |
| `POST /api/skills/hub/confirm` | `{identifier, action, review_token, acknowledgements}` -> Operation |
| `GET /api/skills/hub/operation` | `identifier` plus operation_id -> Operation, read-only, plus core-stamped `settlement` from C.4 |
| `GET /api/skills` | Valid live installed claims add provider provenance/management and the core-stamped `settlement` from C.4. No live claim follows exactly current provenance. |
| Generic local edit/delete/create/rename/restore, learning, archive and hub replacement | Shared final-path mutation gate from C. Provider ownership refuses generic effect while registered. No local-name inference may transfer ownership. |

Every extension accepts `?profile=`. When a provider is registered, its owner-qualified identifiers are resolved before any hub operation. An absent item under a currently registered source refuses without hub fallback. With zero providers, existing routes take the untouched stock path, including stock handling of plugin-looking or malformed identifiers. Core does not add a new error branch there. There is no new provider CLI dispatch path.

### B.2 Search/fallback independence

The HTTP orchestrator keeps built-in `create_source_router` and `parallel_search_sources` on their stock inputs. It runs provider search on separate capacity concurrently, and combines results after the public pipeline completes or reaches its existing overall deadline. It does not add provider futures to `_fan_out` or use a public worker to wait for a provider.

- **Public timing:** index-miss fallback starts immediately after stock public fan-out, using `min(public_absolute_deadline, now + existing_fallback_budget)`. Provider latency does not consume this budget.
- **Provider timing:** five seconds from provider search admission within the caller's overall deadline. Cold bootstrap shares that cap, not an extra five seconds followed by search. On official bootstrap, the caller budget is five seconds independently. Response serialization adds no replacement timeout after deadline.
- **Merge:** stock rows retain their relative order and trust logic. Provider rows dedupe by canonical reference. A same-name public and provider skill remain distinct. No provider enters the public index, public cache, public suppressed-source set or another profile's cache.
- **Cursor:** provider cursor binds query/source/item scope/profile/connection generation. A mixed cursor is an opaque core envelope, at most 4096 ASCII base64url bytes, with at most 16 source cursors. It is a separate type from the single-provider 256-byte cursor. No fixed slice hides later Team rows.

A non-streaming combined response can wait for the provider cap despite a fast public hit. This is not a claim of zero added provider-enabled latency. Public-only and providerless requests do not launch or wait for provider search.

## C. Live ownership and cross-process mutation admission

### C.1 Scope and durable authority

Core stores no persistent catalog owner, installation journal or recovery authority. The provider owns those records. Core's live projection is a disposable coordination view outside the skill tree, at `<profile-home>/cache/skill-catalog/live-v1/`. Removing a registration removes only its coordination rows, never installed files or provider-private receipts.

Owner-qualified references prevent cross-plugin transfer when a catalog name is reused. Core also fences each live registration by exact plugin key, opaque registration id and generation. Same-owner replacement cannot use an earlier generation's handle. A delayed completed replay can be read, but cannot republish current ownership. The digest is never accepted as a substitute for checking the stored exact plugin key.

A profile may have multiple serving processes. Each registration publishes its descriptor and claims under a core-created liveness lease held for that registration's lifetime. The lease is an operating-system-released advisory lock on a separate registration file. Core records process birth and machine identities, not PID alone. Generic mutation code can verify that a claim has a live owner without discovering/importing plugins. A PID timeout or wall-clock TTL is not proof of owner death.

- **Live owner:** claims are authoritative only while its lease is provably held and its consent/enable state remains valid. Cross-process conflicts on source or overlapping target refuse. A second same-source owner does not replace the first implicitly.
- **Proven dead owner:** stale coordination rows are ignored/pruned under the mutation gate. Provider files become local. Restart invokes a new registration and installed read rather than restoring claims.
- **Unreadable liveness:** affected writes refuse ownership_state_unavailable. Never treat uncertain live authority as empty. With no live owner and only corrupt stale projection, stock behavior remains authoritative after verifying no registration lease is held.

A fixed profile presence-lock file sits outside the JSON projection. Every live registration holds a shared OS lease on it. Under the mutation gate, a nonblocking exclusive acquisition proves there are zero live registrations without parsing any projection or liveness index. That proof selects the stock branch even if stale JSON is missing/corrupt. If the exclusive acquisition conflicts, registration leases/rows must validate before affected writes proceed. An unreadable presence-lock file is an I/O failure, not evidence of absence. No-cache fast path runs no provider code. All new generic actors use the mutation gate even on that fast path, so concurrent registration cannot appear between absence proof and filesystem action. Files are user-owned, same-UID cooperative arbitration, not protection against malicious plugins, direct shell edits or deleting the lock directory. OS lock release is crash recovery, not time-based mutation permission.

The fixed gate, the presence lock and the writer-hold directory are core-managed cache files, not entries discovered as skills. Readers first resolve a live registry/lease, not stale target metadata. Internal provider helpers never receive HTTP-client supplied lock paths. Lock acquisition is bounded by the caller's deadline and returns busy without silently degrading to unlocked I/O. A held gate cannot be expired on a timer.

### C.2 Projection schema and lifecycle

Logical `claims.json` is `{schema_version: 1, revision, registrations, targets, reservations}`. Revisions are strict safe integers. All entries use the A.1 bounds.

| Map | Consumed record |
|---|---|
| registrations, keyed registration_id | `{source_id, plugin_key, generation, implementation_revision, process: {machine_id, pid, birth_id}, state}`. pid strict 1..2147483647, state `reconciling\|ready\|fenced`. Lease filename is the registration_id, never provider-supplied. |
| targets, keyed target_relative_path | `{registration_id, identifier, installation_revision, path_revision, managed_state}`. Only claims validated from the provider's local installed records or current settlement. Missing/removed records can be shown by provider metadata but protect no absent tree automatically. |
| reservations, keyed reservation_id | `{registration_id, generation, review_id, identifier, target_relative_path, operation_id, expected_path_revision, expected_installation_revision, state}`. operation_id nullable only before provider atomic binding, state `reserved\|bound\|unknown`. Every later bind compares exact current record. |

`path_revision` is monotonic for a path within the current live registration epoch. Reuse advances it even when a previous target row was removed. A small epoch-local path-revision map is retained while any registration is live and included as `path_revisions` in the stored document. Its keys/values use the same path and safe-integer bounds. It disappears with the last registration. It is not durable ownership after disable. Registration ids never repeat, so an epoch reset cannot validate an old handle.

Every write validates the entire current document under the profile mutation gate, compares expected revisions, writes/flushes a temporary file, atomically replaces and syncs the containing directory where supported. Failure preserves the prior bytes and prevents a new effect. Missing projection with a live registration is unavailable, not empty. Unknown schema or corrupt bytes with a live owner refuse affected mutations. Schema v1 has no legacy format: future monotonic migration is explicit, under the gate, and failure leaves original bytes. Startup never repairs a corrupt live projection silently. Current hub corrupt-as-empty behavior is not reused.

**Registration starts reconciling.** It supplies bounded local installed pages, validates paths, then publishes a complete target set. While its local scan is incomplete, generic writes under that profile's skills root return busy rather than racing unknown ownership. Scan has a three-second caller budget and pagination limit of 100 pages. It runs outside the mutation gate. Failure makes this registered source unavailable and holds writes fail-closed until explicit unregister/retry. Unregister always releases the projection and does not require uninstalling records first.

**Disable/unload sequence** fences generation and permit admission, acquires the admission gate, then takes the writer-hold root exclusive. That root step waits, bounded, for every in-flight writer to exit, including a writer whose parent has already died, because the writer's own descriptors hold the root shared until its critical section ends. It then removes its targets/reservations/registration and releases the liveness lease. It requires no retained-install cleanup. A disable call may return transient busy while an actual local commit is still active, but never refuses merely because installed or recoverable records exist. After disable completes, no plugin-originated writer may acquire or use a stale permit. In a multi-process profile all live registrations for that plugin must acknowledge the fence before disable completion. Process death releases its OS lease. A shared configuration disable is rechecked at each commit admission, not only at method start.

**Re-enable** asks installed again. Deleted, edited and replaced paths are classified by provider rules, then validated by core. `installed()` is a read and cannot modify or auto-delete a target. Core neither retains nor resurrects previous protection on its own. Unavailable/replaced claims have no effect authority. A subsequent provider effect needs fresh review against current bytes or a safely resumable original pending operation with matching current local state.

### C.3 Shared gate and provider permit

One profile-wide cross-process mutation gate serializes cooperating filesystem commits. This is intentionally coarser than per-path locks. Equal paths and ancestor/descendant paths therefore cannot race while their gate owners live. The per-target writer-hold chain below is the second, writer-owned exclusion, and it is the one that survives a gate owner's death. Every core actor that writes, removes, renames, replaces or restores a skill participates in both, even if its initial discovery found no provider. Existing `skill_file_lock` is a reusable exclusive lock primitive, not evidence those paths already coordinate.

**Generic actor:** acquire gate, validate live leases and current projection, resolve both source/destination paths without redirect traversal, reject overlapping live provider claims/reservations, acquire the writer-hold chain for every affected path with the target step exclusive, then perform the short local filesystem action before releasing the chain and the gate. Expensive planning/network/scanning runs first, outside it. Final path identity and ownership are rechecked inside. Archive/rename checks both source and destination. No durable generic reservation or crash journal is added: process death releases its OS locks and the next actor sees actual files.

**Provider actor:** network work and package staging occur outside the gate. Before dispatch of fresh confirm, core records a nonsecret reservation bound to token's stored target and expected path/installation revision. This excludes later generic writes. Provider atomically claims token and journals its operation before effects, then binds the operation to that reservation by compare-and-set. All local writes occur only under a registration-scoped mutation permit. The provider still owns package validation, atomic tree replacement and its journal.

The public helper is `ctx.skill_catalog_mutations`, a narrow plain-data remote facade beside the registrar. It supports:

- **`bind`** with `{registration_id, review_id, reservation_id, operation_id, expected_path_revision, expected_installation_revision}`. After provider atomic token claim, core CAS-binds the operation to the existing exact reservation. Repeating that binding is idempotent. Different operation, owner or revision refuses. The provider cannot invent an unrelated reservation. The service has already reserved against the stored review.
- **`acquire`** with `{registration_id, reservation_id, operation_id, expected_path_revision, expected_installation_revision}`. Core derives plugin/profile from ctx, verifies bound record/current generation/consent, acquires the profile mutation gate and returns `{permit_id, target_relative_path, expected_path_revision, expected_installation_revision}`, all derived from the bound reservation. No caller-supplied target is accepted. It never trusts an owner in the payload.
- **`validate`** with `{permit_id}`. This read checks parent-held permit state/current generation/consent without attempting to reacquire the already-held admission gate. The hosted support library calls it after acquiring the writer-hold chain and before its first local write.
- **`settle`** with `{permit_id, reservation_id, operation_id, expected_path_revision, result: Operation}`. It validates current owner/registration/operation/revision and installs/removes the matching target projection. Exact duplicate accepted settlement returns the same revision. Changed terminal data conflicts. It does not replace provider receipts.
- **`release`** with `{permit_id}`. Core releases the gate only after the provider finishes its local critical section. The helper is optional on old cores, but a catalog provider must refuse catalog effects without it rather than downgrade safety.

A permit is nonsecret authority scoped to the authenticated owning context, with the opaque-id grammar. Host mode carries these calls over the existing context-facade RPC seam, with parent-owned gate state. The parent retains gate ownership across the child's short commit until release or verified host/process death. A dedicated permit owner must respect OS lock/thread requirements. A response timeout alone never releases a permit or slot while a child can still write. Registration/registry locks and provider worker-dispatch locks are never held during provider code or filesystem effects. The mutation gate is a distinct, explicit short commit lock, not a lock around arbitrary provider network code.

**Writer-side hold.** The process that performs the filesystem effect holds exclusive per-target OS locks for its whole critical section. The parent never holds them on the child's behalf. Lock files live under the fixed core-managed directory `<profile-home>/cache/skill-catalog/live-v1/writers/`, one file per validated skills-root-relative path, named by the lowercase SHA-256 of the exact path bytes. The existing fixed `writer-barrier.lock` is the root of that tree. A writer for target `a/b` acquires, in this order: the root file shared, each proper ancestor file shared (`a`), then the target file (`a/b`) exclusive. Overlap is therefore a lock conflict rather than a projection lookup. A writer on an ancestor needs that file exclusive and meets the descendant writer's shared hold. A writer on a descendant needs the ancestor shared and meets the exclusive hold. Equal paths meet exclusive against exclusive. Rename and replace actors acquire both endpoints' chains, endpoint order by exact path bytes, before any effect. Disable and whole-root actions take the root file exclusive. The A.1 path bounds keep every chain short.

Each step is a nonblocking try with bounded retry against the caller's remaining deadline, always in root-first order. A busy step releases the whole chain and returns busy. A fixed global order with nonblocking steps means two writers never wait on each other. Shared and exclusive modes are the ones the checkout lock already uses on one file, including the Windows LockFileEx shared mode because msvcrt has no shared lock. Per-key lock files held for exactly the critical window, kernel release on descriptor close including process death, bounded LOCK_NB polling and a structured busy refusal follow the per-profile turn lock and the cron jobs lock. Only the contention errnos retry. Any other lock error refuses the effect at once. No actor unlinks a writer-hold file, because a later opener of a replacement inode would not conflict with the holder of the old one. The directory is bounded by the set of paths ever written under this profile and goes away only with the whole coordination cache, after the presence lock proves zero live registrations. The hold is not a timeout, not host termination and not a settlement check: it is held or it is not.

**Order with the gate.** Admission gate first, writer-hold chain second, final validation third, filesystem effect last. The parent takes the gate inside `acquire` and returns a permit. The hosted child opens its own descriptors, acquires the chain, then performs the final `validate` RPC before its first write. `validate` reads parent-held permit state, takes no OS lock and waits for no child, so the parent's RPC handler cannot block on a lock the child holds, and the child never waits on the gate it does not own. If the parent died before validation, no write starts and the chain is released. If it dies afterward, the chain stays held by the child's descriptors. No generic actor, disable, recovery writer, re-registered instance of the same provider or other provider's writer can complete an overlapping chain until the child releases in its finally block or the operating system releases on its death. The parent-owned gate disappearing changes nothing about that hold, because the hold never lived on the gate owner. The child releases the chain before calling `release` on the parent-held admission gate. Core's host support library derives every lock path from the fixed profile directory, never from provider data. No two processes attempt to take the same exclusive lock on behalf of one permit. A child that cannot establish or retain the chain, or cannot validate its permit, refuses the effect. Core uses process birth plus lease state to clear claims, never PID or expiry alone.

**Pre-state recheck inside the hold.** After the chain is held and before the first write, the writer rechecks the actual target against the review's bound pre-state. An install requires the target absent, with no entry and no symlink. An update or uninstall requires the tree of the recorded installation revision. Under this contract the check is repeated inside the hold, because the projection cannot see a write whose settlement never landed, and `path_revision` is not evidence of absence. A mismatch refuses with review_changed and writes nothing. That is what stops a later writer from replacing a tree an orphaned writer committed, without a current review of the tree that is now present.

**In-process equivalent.** When the provider runs inside the serving process, the commit thread opens its own writer-hold descriptors and holds the same chain, while a dedicated gate owner thread holds the admission gate. flock and LockFileEx conflict per open file description, so a second commit thread in the same process meets the same busy result as a second process. The process-local `_SKILLS_PROFILE_LOCK` serializes scoped requests inside one process only. It is not held across the commit and is never accepted as exclusion against another serving process or a hosted child.

The local Python surface is `with ctx.skill_catalog_mutations.commit(admission, operation_id) as commit:` followed by `commit.settle(result)` inside the block. The wrapper derives and acquires the writer-hold chain for the admission's target path and checks its generic path/presence pre-state. It yields `target_relative_path` and `expected_installation_revision`. The provider repeats its exact opaque content/tree check inside the block before the first write. The wrapper guarantees chain release in finally. This local context manager never crosses RPC. Only the listed plain-data calls do. Unsettled exceptional exit leaves the reservation unknown, preserving provider journal reconciliation rather than inventing a terminal result. C.4 defines how core reconciles that unknown reservation while the registration stays alive.

Parent-owned admission/settlement metadata uses this facade, but the public provider-facing context-manager wrapper acquires/releases the writer-hold chain automatically. Providers do not hand-code this handshake. A dedicated gate owner performs acquire/settle/release through a bounded command queue so an RPC handler never unlocks a different thread's lock. Exceptions and duplicate release have explicit finally/idempotent paths, with timeout retaining the permit until the writer exits.

If host/process death occurs mid-commit, provider journal recovery inspects actual files before it reports/retries. Core leaves a live registration's reservation unknown until that reconciliation, but drops all coordination if the registration itself dies or is disabled. That is the approved local-on-disable consequence. Dropping those rows is a statement about ownership, not permission to write: an orphaned writer's chain outlives its registration's rows, so a later provider, or the same provider re-registered, reserves against the absent target, admits under its own gate, and then meets either the busy hold or, once the orphan has exited, the pre-state recheck against the tree it left. Provider recovery after re-enable cannot assume an unchanged path.

**Legacy plugin effects** that target this catalog while it is registered enter `ctx.skill_catalog_mutations.admit_native(native_receipt, approved_choices) -> {admission, operation_id}` through the same profile-scoped reservation service. The plain-data native receipt contains the generic Review fields plus `native_receipt_id`. Approved choices contain exact finding acknowledgements and the native exceptional boolean choices. `None` means definitive absence of a live catalog for the owning context only. Uncertain ownership or an active catalog in another process refuses rather than selecting an unlocked legacy writer. Repeated pending admission reuses its original binding even after expiry. Expiry forbids only a first admission. The adapter's internal execution kernel consumes the resulting admission and mutation wrapper. Existing native CLI/chat/Desktop screens may retain their presentation, but cannot call a raw legacy writer on a live claim. Every native filesystem effect on a live claim must use that admission. Generic CLI/model mutators use the shared gate, but no core CLI parser or provider workflow is added. This is a cooperative provider API, not an attempt to stop a malicious same-UID plugin bypassing its contract.

### C.4 Recovery ordering and fencing

**Bound-operation recovery.** `with ctx.skill_catalog_mutations.recover(operation_id) as recovery:` derives its target and admission from that owning catalog's pending-bound operation. It takes the same parent admission gate and writer-owned lock chain, validates current owner/generation/path-revision fences, then exposes `target_relative_path`, `expected_installation_revision`, `pre_revision_matches` and `observed_revision`. The pre-match boolean reports core's presence precondition only because installation revisions are provider-opaque. Providers still validate their exact durable rename topology and original file/directory identities inside the hold.

`observed_revision` is null for absence, otherwise the lowercase SHA-256 of canonical JSON relative-file hash facts: `json.dumps(facts, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()`. Each regular file maps to `sha256:<hex>`, prefixed `exec:` if any executable bit is set. Redirects, special files, more than 2,000 files, changed file identities and an observation exceeding three seconds refuse. Observation is explicit recovery work, not ownership inference or per-request scanning.

Exactly one `recovery.decide(answer)` is required: `{kind: reconciled, journal_digest, result: pending Operation}` compares the provider journal's committed-tree digest with the core observation and records the local pending marker without a filesystem effect. `{kind: retry_effect}` permits the original operation's remaining local work followed by `recovery.settle(pending Operation)`. `{kind: abandon, reason}` settles core failed and preserves all files. A mismatching reconciliation digest settles failed and raises review_changed. Missing/invalid decisions or foreign/nonpending operations raise review_changed. Hosted recovery keeps this handle and descriptors in the writer child. Only `recovery_acquire`, validate, settle and release plain data cross RPC. New-registration pending re-admission is marked recovery-only and cannot enter fresh `commit`. HTTP verifies pending `reviewed`, native admission verifies the exact pending operation through the provider's read-only journal. No old permit is restored.

**Local commit then remote reconciliation.** A provider may settle `state: pending`, `outcome: null`, with the locally committed `installation_revision` inside the writer hold. Core publishes that local installed/removed state with `settlement: reconciling` and retains the same reservation/operation binding. After releasing the hold, the provider performs remote receipt reconciliation and terminal-settles that binding without `permit_id`. Recovery never repeats the file phase. A terminal remote refusal preserves the already committed local state. Exact terminal repeats are idempotent. Changed results conflict.

**Claim limits.** Author-source enrollment is management-only: it writes no skill files and creates no managed claim in v1. An edited update requiring preservation to a second target refuses with no filesystem effect and guidance. Automatic edited-copy preservation is deferred until a multi-target reservation/lock contract exists. No generic ownership bypass is implied by that guidance: generic archive/rename still refuses a live claim, so provider reclassification or completed disable is required first.

Confirm first authorizes exact profile/source owner and calls reviewed read-only. Before it applies install prerequisites:

1. Completed authority returns its saved Operation. No reservation, target lookup, installation prerequisite, permit or projection settlement runs. Completed uninstall replay after target reuse and old install replay after later uninstall are harmless reads.
2. Pending authority returns the existing provider operation. If a matching current reservation exists, recovery reuses it. Repeating exact confirmation never creates a second operation. A lost response between token claim and operation-id return rebinds the same provider operation to the original reservation by CAS.
3. Unstarted authority validates the full review and expected install state, then reserves once under `(registration_id, review_id)`. Concurrent exact confirmations reuse that nonsecret reservation and the provider atomically binds one operation. A different review for an overlapping target conflicts. A losing caller cannot clear or replace another review's reservation. Provider token claim remains the final one-effect arbiter.

A new registration after restart/disable has no inherited core reservation. Pending recovery may create a replacement coordination reservation only for the same provider operation after provider reconciliation proves current target bytes/installation revision still match its resumable phase. If it cannot prove that, it reports ownership_conflict/review_changed without mutation. Completed replay remains read-only and does not run this reconciliation. Core does not retain a reservation after registration ends.

Every settlement compares registration id, generation, owner, operation, reservation and expected path revision. A late worker after a newer operation or disable cannot republish ownership. Exact settlement duplicate is recognized within the live epoch using its saved terminal fingerprint. After the epoch ends, only provider completed replay remains valid, without a core write. Remote effects already committed cannot be rolled back by unload: the provider reports them honestly, but no local write is admitted afterward.

**Terminal journal without core settlement.** The commit order is fixed: provider journals the pending operation, performs the filesystem effect under the writer-hold chain, durably saves its terminal outcome, calls `settle`, releases the chain, then calls `release`. When `settle` fails after the terminal save, through an I/O error, a CAS conflict, a response timeout or parent death, the provider's journal is terminal while core's reservation is `unknown`. Core reconciles that window itself, without a plugin disable and without a second effect:

- **Who:** the live registration's own reconciler, run by core in the serving process that owns the registration. It calls only provider read methods. There is no timer and no sweep.
- **When:** immediately when the wrapper's unsettled exit reaches the gate owner while the registration is alive, on any `operation` or `reviewed` read whose provider state is terminal while core holds an `unknown` reservation for that operation, and before any review or confirm admission that meets an `unknown` reservation on an overlapping path. Completed-token replay stays read-only and triggers nothing by itself.
- **How:** under the admission gate, take the target's writer-hold chain with a nonblocking exclusive target step. Success proves no writer is still inside. Busy leaves the reservation `unknown` and returns operation_in_progress. Then read `operation(operation_id)` within the three-second read budget. Completed installs or removes the matching target projection and clears the reservation through the ordinary `settle` CAS path, comparing registration, generation, operation, reservation and the expected path revision the reservation recorded, which nothing else can have advanced while the hold was held. Failed clears the reservation and leaves the projection unchanged. Pending leaves the reservation `unknown` and defers to the provider pending-recovery rules above. An unreadable provider leaves it `unknown` and affected overlapping admission returns ownership_state_unavailable, never a silent reset.
- **What the user sees:** the HTTP operation response and `/api/skills` rows carry a core-stamped `settlement: settled|reconciling`. Desktop shows the provider's terminal outcome as the row state with a nonblocking finalizing status while reconciling, and no second confirm action. A review or confirm on that path during reconciling runs the reconciliation first within its own budget, then proceeds, or returns operation_in_progress while a writer still holds the path, or ownership_state_unavailable only when the provider cannot be read.

If the registration dies or is disabled before reconciliation, its rows drop as above and the next registration's `installed()` read, which already reflects the provider's terminal record, rebuilds the projection. The inverse install case, a stale installation revision after a terminal provider install, resolves the same way: the completed read replaces the stale target row by CAS. No reconciliation path performs a filesystem write.

## D. Activation, worker budgets and output containment

### D.1 Activation

HTTP service activation preserves ordinary launch discovery and the secondary-profile host boundary:

1. Authorize and resolve the selected profile through the existing readers and scopes.
2. Read the live registry snapshot without invoking provider attributes. A proven live owner in another serving process returns source-unavailable rather than selecting stock.
3. A retained, bounded bootstrap worker checks metadata eligibility and ordinary isolation. A cold host-isolated manager uses ordinary discovery, version, consent and publication rules.
4. A cold in-process manager returns stock without importing plugins from the request. Ordinary startup/reload may register it. Already-registered in-process catalogs remain supported.

For an active bootstrap with no admitted catalog, existing routes preserve stock bytes without provider_sources/loading/provider_errors keys. Explicit extension routes retain actual stock routing until admission: POST review/confirm return 405 with `Allow: GET` and the stock Method Not Allowed body, while GET operation returns the stock endpoint-specific 404. Diagnostic loading exists in internal logs. Once registered, unavailable/loading belongs to that source and does not replace official/local rows. A late result can enter a later request only if profile, manager generation, enable state and consent remain current. It never alters an already-completed response.

### D.2 Output boundary

Host output already redirects fd 1 before plugin code, parent captures pipes and logs stderr. Core routes catalog-triggered diagnostics to the server's bounded/redacted diagnostic sink, not an inherited console handler or response stream. This includes normal-loader warnings and no-op/non-catalog plugins in the sweep. No global redirect_stdout/redirect_stderr occurs in a multiplexed process.

Cold in-process discovery is not performed by the new HTTP path because arbitrary Python writes cannot be contained request-locally. Existing launch startup stdout behavior stays exactly as main, including plugin-authored prints. The stock guarantee is no *new* server stream delta versus the same pure-main startup, and exact route bytes. This does not retroactively promise silence from third-party code already running in-process. The stock HTTP fixture captures raw status, content type and bodies from pure main for never-used, previously-disabled, flag-off and signed-out profiles with identical local files and offline I/O boundaries.

### D.3 Bounds

| Work | Caller budget/cap |
|---|---|
| Bootstrap | Five seconds including manager-lock wait. One worker per profile/manager generation, not another thread on every timeout. |
| Availability/installed/reviewed/operation | Three seconds each, caller's remaining overall budget is authoritative |
| Search | Five seconds total including cold bootstrap, no additive wait |
| Inspect/check | Fifteen seconds |
| Scan | Thirty seconds |
| Review/confirm | Sixty seconds |
| Worker capacity | Four catalog method calls per profile, at most one read and one effect per provider. Bootstrap has one separate bounded slot so hung discovery cannot exhaust method/public pools. |
| Shutdown | Fence admission first, bounded five-second drain, then report still-active effects uncertain. No fake cancellation. |

Loader timeouts may be disabled, but the HTTP wait cannot. Timed-out calls retain capacity until they actually exit. Busy refuses rather than creating an unbounded queue. Operation reads retain capacity independent of effect calls. Hydration, manifest work, provider validation and cleanup happen off gateway infrastructure loops. Profile scope is entered inside that worker because hydration can block. A hung thread cannot be safely killed, so later requests return stock or provider_unavailable at their budget without spawning replacements. Worker caps are experimental constants, not new environment settings.

